"""Validation gate: the full rule set, one rule at a time — the shared
leaked mini-net record passes whole, and every rule rejects a synthetic
violation built by perturbing a copy of the passing result.

Rules: mass balance + energy closure (with the leak advection term),
temperature envelope (T_supply <= T <= T_ground), velocity
self-consistency envelope (2.5 m/s demoted to exceedance metadata),
pressure rails (high-point floor, plant suction, 0.9xPN16 ceiling), ETS
reverse-flow guard, unmet load only under an active fault (flag, don't
reject — but a clean baseline must be feasible), convergence budget
(budgeted unconverged steps), PN25 ceiling escalation and the clean-unmet
budget.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.taxonomy import FaultLabel
from dcngen.orchestrate.scenario import run_fixed_dt_scenario
from dcngen.orchestrate.validate import validate_scenario
from tests.conftest import build_mini_leak_record

cfg = load_config()

RULES = (
    "mass_balance", "energy_closure", "temperature_envelope",
    "velocity_envelope", "pressure_rails", "ets_reverse_flow",
    "unmet_load", "convergence_budget",
)


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    dcn, record = build_mini_leak_record(tmp_path_factory.mktemp("nets"), cfg)
    report = validate_scenario(dcn, cfg, record.result, record.label)
    return dcn, record, report


def failed_rules(report):
    return [r.rule for r in report.rules if not r.passed]


def rule(report, name):
    return next(r for r in report.rules if r.rule == name)


# ------------------------------------------------------------- passing gate


def test_the_leaked_mini_scenario_passes_every_rule(gated):
    *_, report = gated
    assert failed_rules(report) == []
    assert report.passed
    assert tuple(r.rule for r in report.rules) == RULES


def test_a_steady_fixed_dt_scenario_passes(gated):
    dcn, record, _ = gated
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    result = run_fixed_dt_scenario(dcn, cfg, design, n_steps=4)
    label = FaultLabel.no_fault(np.arange(4) * cfg.solver.dt)
    report = validate_scenario(dcn, cfg, result, label)
    assert failed_rules(report) == []
    # steady design state: closure is essentially exact
    assert rule(report, "energy_closure").metrics["residual_rel"] < 1e-6


def test_report_carries_the_velocity_reference_metadata(gated):
    *_, report = gated
    metrics = rule(report, "velocity_envelope").metrics
    # the 2.5 m/s exceedance fraction is metadata, not a cap
    assert 0.0 <= metrics["reference_exceedance_frac"] <= 1.0
    assert metrics["max_utilization"] > 0.0


def test_report_serializes_to_plain_json_types(gated):
    import json

    *_, report = gated
    dumped = json.dumps(report.as_dict())
    assert '"passed"' in dumped


# ------------------------------------------- one violation rejects per rule


def violated(gated, **result_overrides):
    dcn, record, _ = gated
    broken = replace(record.result, **result_overrides)
    return validate_scenario(dcn, cfg, broken, record.label)


def test_mass_balance_rejects_unexplained_make_up(gated):
    dcn, record, _ = gated
    report = violated(gated, make_up_flow=record.result.leak_flow + 0.01)
    assert failed_rules(report) == ["mass_balance"]


def test_energy_closure_rejects_a_biased_plant_power(gated):
    dcn, record, _ = gated
    report = violated(gated, plant_power=record.result.plant_power * 1.5)
    assert "energy_closure" in failed_rules(report)


def test_temperature_envelope_rejects_water_above_ground(gated):
    dcn, record, _ = gated
    hot = record.result.supply_temp.copy()
    hot.iloc[-1, 0] = cfg.thermal.T_ground + 5.0
    report = violated(gated, supply_temp=hot)
    assert failed_rules(report) == ["temperature_envelope"]


def test_temperature_envelope_rejects_water_below_setpoint(gated):
    dcn, record, _ = gated
    cold = record.result.return_temp.copy()
    cold.iloc[0, 1] = cfg.thermal.T_supply - 1.0
    report = violated(gated, return_temp=cold)
    assert failed_rules(report) == ["temperature_envelope"]


def test_velocity_envelope_rejects_a_runaway_solve(gated):
    dcn, record, _ = gated
    report = violated(gated, pipe_velocity=record.result.pipe_velocity * 10.0)
    assert "velocity_envelope" in failed_rules(report)


def test_pressure_rails_reject_a_node_below_the_floor(gated):
    dcn, record, _ = gated
    low = record.result.pressure_r.copy()
    low.iloc[2, 2] = 0.5  # below (z_max - z) + 2 m for every mini junction
    report = violated(gated, pressure_r=low)
    assert failed_rules(report) == ["pressure_rails"]


def test_pressure_rails_reject_a_node_above_the_pn25_ceiling(gated):
    # PN16..PN25 escalates (see the escalation test); "above the
    # ceiling" now means above the PN25 continuous ceiling
    dcn, record, _ = gated
    high = record.result.pressure_s.copy()
    high.iloc[0, 0] = cfg.validation.pressure_ceiling_pn25 + 20.0
    report = violated(gated, pressure_s=high)
    assert failed_rules(report) == ["pressure_rails"]


def test_pressure_rails_reject_a_starved_plant_suction(gated):
    dcn, record, _ = gated
    starved = copy.deepcopy(dcn)
    header = starved.wn.get_node(starved.plant.supply_header).elevation
    starved.wn.get_node(starved.plant.reservoir).base_head = (
        header + 0.5 * cfg.validation.plant_suction_min
    )
    report = validate_scenario(starved, cfg, record.result, record.label)
    assert "pressure_rails" in failed_rules(report)


def test_reverse_flow_guard_rejects_a_backward_ets(gated):
    dcn, record, _ = gated
    backward = record.result.consumer_flow.copy()
    backward.iloc[-1, 0] = -backward.iloc[-1, 0]
    report = violated(gated, consumer_flow=backward)
    assert "ets_reverse_flow" in failed_rules(report)


def test_unmet_load_outside_the_fault_window_rejects(gated):
    dcn, record, _ = gated
    label = record.label
    clean_step = int(np.flatnonzero(~label.mask)[0])
    unmet = record.result.unmet.copy()
    unmet.iloc[clean_step, 0] = True
    report = violated(gated, unmet=unmet)
    assert failed_rules(report) == ["unmet_load"]


def test_unmet_load_inside_the_fault_window_is_recorded_not_rejected(gated):
    dcn, record, _ = gated
    label = record.label
    fault_step = int(np.flatnonzero(label.mask)[0])
    unmet = record.result.unmet.copy()
    unmet.iloc[fault_step, 0] = True
    report = violated(gated, unmet=unmet)
    assert rule(report, "unmet_load").passed
    assert rule(report, "unmet_load").metrics["unmet_frac_under_fault"] > 0.0


def test_convergence_budget_rejects_a_strained_scenario(gated):
    dcn, record, _ = gated
    strained = record.result.converged.copy()
    strained[:] = False
    report = violated(gated, converged=strained)
    assert failed_rules(report) == ["convergence_budget"]


def test_energy_closure_holds_on_horizons_the_old_rule_called_unverifiable(gated):
    """With steady init + the measured storage term there is no turnover
    precondition — a 2-step scenario (pumping far less than one loop
    volume, the old rule's hard FAIL) closes essentially exactly."""
    dcn, _, _ = gated
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    result = run_fixed_dt_scenario(dcn, cfg, design, n_steps=2)
    label = FaultLabel.no_fault(np.arange(2) * cfg.solver.dt)
    report = validate_scenario(dcn, cfg, result, label)
    energy = rule(report, "energy_closure")
    assert energy.passed
    assert energy.metrics["residual_rel"] < 1e-6
    assert "storage" in energy.detail


def test_energy_closure_reports_the_storage_rate_as_diagnostic(gated):
    """The recorded gains are advective (storage-inclusive), so the
    balance never subtracts storage — but the rate is reported for every
    scenario (it quantifies the init discharge). The balance's
    falsifiability lives in the biased-plant-power test above."""
    dcn, record, _ = gated
    energy = rule(gated[2], "energy_closure")
    horizon = len(record.result.time) * cfg.solver.dt
    expected = (
        record.result.stored_energy_end - record.result.stored_energy_start
    ) / horizon
    assert energy.metrics["storage_rate_W"] == pytest.approx(expected, rel=1e-9)


def test_pressure_ceiling_escalates_to_pn25_with_a_class_stamp(gated):
    # between the PN16 and PN25 continuous
    # ceilings the scenario passes with pn25_escalated recorded; only
    # exceeding PN25 fails
    dcn, record, _ = gated
    base = rule(gated[2], "pressure_rails")
    assert base.metrics["pn25_escalated"] == 0.0  # mini net: PN16 regime

    high = record.result.pressure_s.copy()
    high.iloc[0, 0] = (
        cfg.validation.pressure_ceiling + cfg.validation.pressure_ceiling_pn25
    ) / 2.0
    report = validate_scenario(dcn, cfg, replace(record.result, pressure_s=high), record.label)
    escalated = rule(report, "pressure_rails")
    assert escalated.passed
    assert escalated.metrics["pn25_escalated"] == 1.0
    assert "PN25" in escalated.detail

    over = record.result.pressure_s.copy()
    over.iloc[0, 0] = cfg.validation.pressure_ceiling_pn25 + 5.0
    report = validate_scenario(dcn, cfg, replace(record.result, pressure_s=over), record.label)
    assert not rule(report, "pressure_rails").passed


def test_clean_unmet_below_the_budget_is_recorded_not_rejected(gated):
    # one pinched consumer-step on a clean stretch (~7% of the CLEAN
    # consumer-steps, the rule's own denominator) fails the default 0.5%
    # budget but passes a widened one — the budget is the knob
    import dataclasses as dc

    dcn, record, _ = gated
    label = record.label
    clean_step = int(np.flatnonzero(~label.mask)[0])
    unmet = record.result.unmet.copy()
    unmet.iloc[clean_step, 0] = True
    broken = replace(record.result, unmet=unmet)

    strict = validate_scenario(dcn, cfg, broken, label)
    assert not rule(strict, "unmet_load").passed  # ~7% > 0.5% default

    # the fraction is measured over CLEAN steps only (1 cell of ~15 here,
    # ~7 %), so the widened budget must sit above that
    wide = dc.replace(
        cfg, validation=dc.replace(cfg.validation, max_clean_unmet_frac=0.10)
    )
    lenient = validate_scenario(dcn, wide, broken, label)
    assert rule(lenient, "unmet_load").passed
    assert rule(lenient, "unmet_load").metrics["unmet_frac_outside_fault"] > 0.0


def test_suction_rail_comparison_survives_float_roundoff(gated):
    # a rail_fallback anchor sits exactly ON the suction rail, and
    # (z + suction_min) - z loses ~1 ulp for many elevations — the gate
    # must not reject a draw the fallback anchoring admits. Search a
    # genuinely lossy elevation as the premise.
    dcn, record, _ = gated
    s_min = cfg.validation.plant_suction_min
    z = next(
        x / 10.0 for x in range(60, 300)
        if (x / 10.0 + s_min) - x / 10.0 < s_min
    )
    nasty = copy.deepcopy(dcn)
    nasty.wn.get_node(nasty.plant.supply_header).elevation = z
    nasty.wn.get_node(nasty.plant.reservoir).base_head = z + s_min
    report = validate_scenario(nasty, cfg, record.result, record.label)
    assert rule(report, "pressure_rails").passed  # slack absorbs the ulp


def test_rail_fallback_network_passes_the_gate_end_to_end(tmp_path):
    # the fallback-anchored mini network (huge min_dp forces it) must clear
    # the whole gate, suction rail included — the workflow found the
    # dcnifier-side test never exercised the gate
    from dcngen.topology.dcnifier import build_dcn
    from dcngen.topology.ditec_loader import load_ditec_network
    from tests.conftest import write_ditec_folder, write_mutated_default

    def raise_min_dp(d):
        d["ets"]["min_dp"] = 60.0

    cfg2 = load_config(write_mutated_default(tmp_path, raise_min_dp))
    folder = write_ditec_folder(tmp_path / "mini_net")
    dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg2)
    assert dcn.anchoring == "rail_fallback"

    design = {j: c.design_load for j, c in dcn.consumers.items()}
    result = run_fixed_dt_scenario(dcn, cfg2, design, n_steps=4)
    label = FaultLabel.no_fault(np.arange(4) * cfg2.solver.dt)
    report = validate_scenario(dcn, cfg2, result, label)
    assert rule(report, "pressure_rails").passed
    assert report.passed
