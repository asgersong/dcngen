"""Acceptance on the real Hanoi dump: a two-row plan slice
driven process-parallel to completed, gate-stamped record folders with
the aggregated report. Together with the plan-runner slow pair (fouling +
bypass) this closes real-network coverage of all four classes: here the
normal and leak bulk rows run through the DRIVER path (spawned workers,
staging rename, card aggregation). Slow-marked (~20 min per 144-step row;
run with `pytest -m slow`); skipped when the local dump is not present.
"""

import json

import pytest

from dcngen.config import load_config
from dcngen.orchestrate.driver import generate_dataset, scenario_folder_name
from dcngen.orchestrate.sampler import build_plan
from dcngen.records.pyg_dataset import load_scenario
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.sampler.network

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
    ),
]


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    plan = build_plan(cfg, load_ditec_network(HANOI, scenario_id=0))
    uids = [
        next(
            r.uid for r in plan.rows
            if r.tier == "bulk" and r.scenario_class == cls
        )
        for cls in ("normal", "leak")
    ]
    out = tmp_path_factory.mktemp("hanoi_dataset")
    report = generate_dataset(cfg, plan, out, uids=uids, workers=2)
    return plan, uids, out, report


def test_slice_completes_with_gate_stamped_folders(driven):
    plan, uids, out, report = driven
    assert report.completed == 2 and not report.failed
    for uid in uids:
        card = json.loads(
            (out / scenario_folder_name(uid) / "card.json").read_text()
        )
        assert card["scenario_id"] == uid
        assert "validation" in card
    assert set(report.pass_rates) == {"normal", "leak"}
    assert report.core_seconds > 0


def assert_physics_rules_pass(card):
    """Full gate pass: the storage-closed balance verifies on every horizon, so the
    draws that used to be unverifiable (197, 203, 778) now simply pass."""
    failed = [
        f"{r['rule']}: {r['detail']}"
        for r in card["validation"]["rules"]
        if not r["passed"]
    ]
    assert card["validation"]["passed"], failed


def test_normal_row_passes_physics_and_loads(driven):
    plan, uids, out, report = driven
    normal_uid = uids[0]
    card = json.loads(
        (out / scenario_folder_name(normal_uid) / "card.json").read_text()
    )
    assert_physics_rules_pass(card)
    graph = load_scenario(out / scenario_folder_name(normal_uid))
    assert graph.scenario_class == "normal"
    assert not graph.fault_active.any()


def test_leak_row_keeps_its_signature_through_the_driver(driven):
    plan, uids, out, _ = driven
    leak_uid = uids[1]
    graph = load_scenario(out / scenario_folder_name(leak_uid))
    assert graph.scenario_class == "leak"
    realized = graph.realized_leak_flow.numpy()
    mask = graph.fault_active.numpy()
    assert (realized[mask] > 0.0).all()
    assert (realized[~mask] == 0.0).all()
    card = json.loads(
        (out / scenario_folder_name(leak_uid) / "card.json").read_text()
    )
    assert_physics_rules_pass(card)
