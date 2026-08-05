"""PoC runner: one call produces the pair — clean-normal
+ abrupt-leak — both gated, written, and reloaded as typed graphs; the
whole run is deterministic from recorded seeds (criteria 2-4).

Runs on the synthetic mini net via a config override (the real command
targets Hanoi with the same code path); 12 steps keep the leak onset
window meaningful (the energy balance is storage-closed and needs no
minimum horizon).
"""

import json
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.poc import run_poc

N_STEPS = 12


def mini_cfg(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    nets = tmp_path_factory.mktemp("nets")
    write_ditec_folder(nets / "mini_net")
    cfg = load_config()
    return replace(
        cfg,
        paths=replace(cfg.paths, ditec_data=nets),
        poc=replace(cfg.poc, network="mini_net", horizon=N_STEPS * cfg.solver.dt),
    )


@pytest.fixture(scope="module")
def poc_pair(tmp_path_factory):
    cfg = mini_cfg(tmp_path_factory)
    run = run_poc(cfg, tmp_path_factory.mktemp("poc") / "out")
    return cfg, run


def test_both_scenarios_pass_the_gate(poc_pair):
    cfg, run = poc_pair
    assert [s.name for s in run.scenarios] == ["normal", "leak"]
    for scenario in run.scenarios:
        assert scenario.report.passed, scenario.lines
    assert run.passed
    assert run.lines[-1].endswith("PASS ==")


def test_records_carry_the_gate_report(poc_pair):
    _, run = poc_pair
    for scenario in run.scenarios:
        card = json.loads((scenario.folder / "card.json").read_text())
        validation = card["validation"]
        assert validation["passed"] is True
        rules = {r["rule"]: r for r in validation["rules"]}
        # the 2.5 m/s exceedance fraction is stored as metadata
        assert "reference_exceedance_frac" in rules["velocity_envelope"]["metrics"]
        assert card["seed"] == scenario.seed


def test_reloaded_graphs_carry_correct_labels(poc_pair):
    _, run = poc_pair
    normal, leak = run.scenarios
    assert normal.graph_ok and leak.graph_ok

    # explicit no-fault label on the clean scenario (criterion 3)
    assert normal.graph.fault["kind"] == "none"
    assert not bool(normal.graph.fault_active.any())
    for store in ("supply", "return"):
        assert not bool(normal.graph[store].fault_location_mask.any())

    fault = leak.graph.fault
    assert fault["kind"] == "leak"
    assert fault["side"] in ("supply", "return")
    active = leak.graph.fault_active.numpy()
    assert active.any() and not active.all()  # onset strictly mid-horizon
    located = leak.graph[fault["side"]].fault_location_mask.numpy()
    assert located.sum() == 1
    junctions = leak.graph[fault["side"]].junction
    assert junctions[int(np.flatnonzero(located)[0])] == fault["location"]
    # the leak actually materialized — a null leak must not pass silently
    realized = leak.graph.realized_leak_flow.numpy()
    assert realized[active].min() > 0.0
    assert (realized[~active] == 0.0).all()


def test_the_poc_is_deterministic_from_recorded_seeds(poc_pair, tmp_path_factory):
    """Sub-seeds identical, generated inputs (loads,
    labels) bitwise identical, solved fields within solver tolerance."""
    cfg, first = poc_pair
    again = run_poc(cfg, tmp_path_factory.mktemp("poc2") / "out")
    assert again.passed

    for a, b in zip(first.scenarios, again.scenarios):
        assert a.seed == b.seed

        # generated inputs are bitwise reproducible; realized_* label
        # series are SOLVED fields, so they carry the solver tolerance instead
        labels_a, labels_b = (
            pq.read_table(f / "labels.parquet").to_pandas()
            for f in (a.folder, b.folder)
        )
        assert list(labels_a.columns) == list(labels_b.columns)
        for column in labels_a.columns:
            if column.startswith("realized_"):
                np.testing.assert_allclose(
                    labels_a[column], labels_b[column], rtol=1e-6, atol=1e-9
                )
            else:
                np.testing.assert_array_equal(labels_a[column], labels_b[column])
        load_a, load_b = (
            pq.read_table(f / "node_dynamics.parquet")
            .to_pandas()["cooling_load"]
            .to_numpy()
            for f in (a.folder, b.folder)
        )
        np.testing.assert_array_equal(load_a, load_b)

        # solved fields agree within solver tolerance, not bitwise
        for store_key, attr in (("supply", "x"), (("supply", "pipe", "supply"), "edge_attr")):
            ta = getattr(a.graph[store_key], attr).numpy()
            tb = getattr(b.graph[store_key], attr).numpy()
            np.testing.assert_allclose(ta, tb, rtol=1e-6, atol=1e-9, equal_nan=True)

        card_a = json.loads((a.folder / "card.json").read_text())
        card_b = json.loads((b.folder / "card.json").read_text())
        assert card_a["seed"] == card_b["seed"]
        assert card_a["fault"] == card_b["fault"]
        assert card_a["flow_directions"] == card_b["flow_directions"]
        # the heat-gain knob draw regenerates bitwise from the sub-seed
        assert card_a["heat_gain"] == card_b["heat_gain"]


def test_cards_record_the_heat_gain_draw(poc_pair):
    # derived mode is the default and its knob values are
    # recorded per scenario, inside the cited config bounds
    cfg, run = poc_pair
    hg = cfg.heat_gain
    for scenario in run.scenarios:
        block = json.loads((scenario.folder / "card.json").read_text())["heat_gain"]
        assert block["mode"] == "derived"
        assert hg.lambda_pur_min <= block["lambda_pur"] <= hg.lambda_pur_max
        assert hg.lambda_soil_min <= block["lambda_soil"] <= hg.lambda_soil_max
        assert hg.burial_cover_min <= block["burial_cover"] <= hg.burial_cover_max


def test_scalar_fallback_mode_runs_the_poc_green(tmp_path_factory):
    # both modes run the PoC green; the dev/ablation scalar
    # mode records its flat value instead of knob draws
    cfg = mini_cfg(tmp_path_factory)
    cfg = replace(cfg, heat_gain=replace(cfg.heat_gain, mode="scalar"))
    run = run_poc(cfg, tmp_path_factory.mktemp("poc-scalar") / "out")

    assert run.passed
    for scenario in run.scenarios:
        card = json.loads((scenario.folder / "card.json").read_text())
        assert card["heat_gain"] == {
            "mode": "scalar",
            "scalar_U": cfg.heat_gain.scalar_U,
        }
