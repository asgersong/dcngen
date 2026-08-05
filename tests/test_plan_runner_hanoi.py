"""Acceptance on the real Hanoi dump: manifest rows from the
full default plan run end-to-end and pass the physics gate. Two rows cover
the ETS fault classes newly exercised on the real network (fouling +
bypass); the leak path on Hanoi is already end-to-end-tested by the PoC.
Skipped when the local dump is not present.
"""

import pytest

from dcngen.config import load_config
from dcngen.faults.taxonomy import FaultKind
from dcngen.orchestrate.plan_runner import run_plan_row
from dcngen.orchestrate.sampler import build_plan
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.sampler.network

# slow: each full 144-step Hanoi plan row runs ~20 min (measured 2026-07-23,
# consistent with the #14 compute budget's ~6-7 s/step at scale) — this is
# release assurance, not a default-suite check
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
    ),
]


@pytest.fixture(scope="module")
def plan():
    return build_plan(cfg, load_ditec_network(HANOI, scenario_id=0))


def run_first(plan, cls):
    row = next(
        r for r in plan.rows if r.tier == "bulk" and r.scenario_class == cls
    )
    return run_plan_row(cfg, row)  # loads the row's own static draw


@pytest.fixture(scope="module")
def fouling_scenario(plan):
    return run_first(plan, "fouling")


@pytest.fixture(scope="module")
def bypass_scenario(plan):
    return run_first(plan, "bypass")


def assert_physics_rules_pass(s):
    """Full gate pass: with the steady initial condition and the
    storage-closed energy balance every rule is verifiable on every
    horizon, so nothing is excused."""
    failed = [f"{r.rule}: {r.detail}" for r in s.report.rules if not r.passed]
    assert s.report.passed, f"uid {s.row.uid}: {failed}"


def test_fouling_row_passes_the_gate_with_a_collapsed_split(fouling_scenario):
    s = fouling_scenario
    assert_physics_rules_pass(s)
    assert s.label.kind is FaultKind.FOULING and s.label.mask.all()
    # the fouled consumer ends below the design split while the network's
    # healthy consumers sit near it (weak, twin-free signature check)
    j = s.row.fault.junction
    split = s.result.ets_return.iloc[-1][j] - s.result.supply_temp.iloc[-1][j]
    assert split < s.config.thermal.dT_design - 0.2


def test_bypass_row_passes_the_gate_with_a_mixed_return(bypass_scenario):
    s = bypass_scenario
    assert_physics_rules_pass(s)
    assert s.label.kind is FaultKind.BYPASS
    # on fault-active steps the measured split sits well below design
    # (dT_mix = (1-f) dT_hx). Twin-free slack 1.2: the healthy-UA margin
    # and part-load co-move closure put the healthy split above dT_design
    # (measured 5.52 K at f = 0.303 on draw 197 = 1.13x the naive
    # (1-f) dT_design); the rigorous twin-relative identity at rel 0.02
    # lives in test_plan_runner.py.
    j, f = s.row.fault.junction, s.row.fault.severity
    split = (
        s.result.ets_return[j].to_numpy() - s.result.supply_temp[j].to_numpy()
    )
    mask = s.label.mask
    assert split[mask][-1] < (1.0 - f) * s.config.thermal.dT_design * 1.2
    if not mask.all():
        assert split[~mask][-1] > 0.9 * s.config.thermal.dT_design
