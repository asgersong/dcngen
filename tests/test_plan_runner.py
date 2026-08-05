"""Plan-runner seam: one manifest row runs end-to-end — static
draw -> DCN -> row-driven loads -> row fault -> eps-NTU -> gate. Rows come
straight from build_plan on the shrunk mini release (the same tier table as
test_sampler), so these are true plan-driven scenarios. Signature tests
compare each fault row against its same-seed healthy twin: the identical
uid keys the LOAD_PATH channel, so the twin sees bitwise-identical loads.
"""

import dataclasses

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.taxonomy import FaultKind
from dcngen.orchestrate.plan_runner import (
    plan_row_loads,
    row_config,
    run_plan_row,
)
from dcngen.orchestrate.sampler import FaultSpec, build_plan
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network
from tests.conftest import shrink_sampler_tiers as shrink_tiers
from tests.conftest import write_mutated_default


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    path = write_mutated_default(tmp_path_factory.mktemp("cfg"), shrink_tiers)
    return load_config(path)


@pytest.fixture(scope="module")
def mini_net(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return load_ditec_network(folder, scenario_id=0)


@pytest.fixture(scope="module")
def nets(mini_net, tmp_path_factory):
    """Static draws 0 and 1 of the mini network, keyed by draw id."""
    from tests.conftest import write_ditec_folder

    other = write_ditec_folder(tmp_path_factory.mktemp("nets2") / "mini_net")
    return {0: mini_net, 1: load_ditec_network(other, scenario_id=1)}


@pytest.fixture(scope="module")
def plan(cfg, mini_net):
    return build_plan(cfg, mini_net)


def pick(plan, tier, cls, key=None):
    rows = [r for r in plan.rows if r.tier == tier and r.scenario_class == cls]
    return min(rows, key=key) if key else rows[0]


def healthy_twin(row):
    return dataclasses.replace(
        row,
        scenario_class="normal",
        fault=FaultSpec("none", None, None, None, None),
    )


def run(cfg, nets, row):
    return run_plan_row(cfg, row, net=nets[row.static_draw_id])


@pytest.fixture(scope="module")
def normal_run(cfg, nets, plan):
    return run(cfg, nets, pick(plan, "bulk", "normal"))


@pytest.fixture(scope="module")
def fouling_pair(cfg, nets, plan):
    row = pick(plan, "bulk", "fouling", key=lambda r: r.fault.severity)
    return run(cfg, nets, row), run(cfg, nets, healthy_twin(row))


@pytest.fixture(scope="module")
def bypass_pair(cfg, nets, plan):
    row = pick(plan, "bulk", "bypass", key=lambda r: r.fault.severity)
    return run(cfg, nets, row), run(cfg, nets, healthy_twin(row))


@pytest.fixture(scope="module")
def leak_pair(cfg, nets, plan):
    row = pick(plan, "bulk", "leak")
    return run(cfg, nets, row), run(cfg, nets, healthy_twin(row))


def test_row_config_co_moves_the_temperature_program(cfg, plan):
    row = plan.rows[0]
    rcfg = row_config(cfg, row)
    assert rcfg.thermal.T_supply == row.T_supply
    assert rcfg.thermal.T_ground == row.T_ground
    assert rcfg.thermal.T_return_design == pytest.approx(
        row.T_supply + cfg.thermal.dT_design
    )
    assert rcfg.thermal.dT_design == cfg.thermal.dT_design
    # everything outside the temperature program is untouched
    assert rcfg.water == cfg.water and rcfg.solver == cfg.solver
    assert rcfg.seed == cfg.seed


def test_run_rejects_a_mismatched_static_draw(cfg, nets, plan):
    row = pick(plan, "bulk", "normal")
    wrong = nets[1 - row.static_draw_id]
    with pytest.raises(ValueError, match="draw"):
        run_plan_row(cfg, row, net=wrong)


def test_every_class_passes_the_physics_gate(
    normal_run, leak_pair, fouling_pair, bypass_pair
):
    # AC 1 + 4: plan-driven scenarios of all four classes clear the gate
    for scenario in (normal_run, leak_pair[0], fouling_pair[0], bypass_pair[0]):
        failed = [r.rule for r in scenario.report.rules if not r.passed]
        assert scenario.report.passed, (
            f"{scenario.row.scenario_class} row uid {scenario.row.uid}: {failed}"
        )


def test_normal_row_carries_the_explicit_no_fault_label(normal_run):
    assert normal_run.label.kind is FaultKind.NONE
    assert not normal_run.label.mask.any()
    assert not normal_run.result.unmet.to_numpy().any()


def test_loads_regenerate_bitwise_from_the_row(cfg, nets, plan):
    # the LOAD_PATH channel is keyed by
    # (master seed, uid), so the same row yields identical loads
    row = pick(plan, "bulk", "normal")
    dcn = build_dcn(nets[row.static_draw_id], row_config(cfg, row))
    a = plan_row_loads(row_config(cfg, row), row, dcn)
    b = plan_row_loads(row_config(cfg, row), row, dcn)
    for j in a:
        np.testing.assert_array_equal(a[j], b[j])


def test_fouling_row_collapses_dt_and_raises_flow(fouling_pair):
    # AC 2: same-seed healthy twin comparison at the fouled consumer
    fouled, twin = fouling_pair
    j = fouled.row.fault.junction
    for a, b in zip(sorted(fouled.loads), sorted(twin.loads)):
        np.testing.assert_array_equal(fouled.loads[a], twin.loads[b])

    f_split = fouled.result.ets_return.iloc[-1][j] - fouled.result.supply_temp.iloc[-1][j]
    t_split = twin.result.ets_return.iloc[-1][j] - twin.result.supply_temp.iloc[-1][j]
    f_flow = fouled.result.consumer_flow.iloc[-1][j]
    t_flow = twin.result.consumer_flow.iloc[-1][j]
    assert f_split < t_split - 0.1  # dT collapse
    assert f_flow > t_flow * 1.02  # primary overflow
    assert fouled.label.kind is FaultKind.FOULING
    assert fouled.label.mask.all()  # chronic


def test_bypass_row_satisfies_the_mixing_identity(bypass_pair):
    # AC 2: at the final (fault-active) step, total flow ~ 1/(1-f) x twin
    # and the measured split ~ (1-f) x twin (dT_mix = (1-f) dT_hx)
    bypassed, twin = bypass_pair
    spec = bypassed.row.fault
    j, f = spec.junction, spec.severity
    b_flow = bypassed.result.consumer_flow.iloc[-1][j]
    t_flow = twin.result.consumer_flow.iloc[-1][j]
    b_split = bypassed.result.ets_return.iloc[-1][j] - bypassed.result.supply_temp.iloc[-1][j]
    t_split = twin.result.ets_return.iloc[-1][j] - twin.result.supply_temp.iloc[-1][j]
    assert b_flow / t_flow == pytest.approx(1.0 / (1.0 - f), rel=0.02)
    assert b_split / t_split == pytest.approx(1.0 - f, rel=0.02)
    assert bypassed.label.kind is FaultKind.BYPASS
    if spec.onset_step is None:
        assert bypassed.label.mask.all()
    else:
        assert not bypassed.label.mask[: spec.onset_step].any()


def test_leak_row_keeps_the_leak_signature(leak_pair, cfg):
    # AC 2: realized leak flow only on masked steps, make-up supplies it,
    # and the pre-onset steps match the healthy twin
    leaked, twin = leak_pair
    spec = leaked.row.fault
    k = spec.onset_step
    leak_flow = leaked.result.leak_flow
    assert (leak_flow[:k] == 0.0).all()
    assert (leak_flow[k:] > 0.0).all()
    np.testing.assert_allclose(
        leaked.result.make_up_flow[k:], leak_flow[k:], rtol=1e-5
    )
    np.testing.assert_allclose(
        leaked.result.consumer_flow.iloc[:k].to_numpy(),
        twin.result.consumer_flow.iloc[:k].to_numpy(),
        rtol=1e-6,
    )
    # the label carries the realized series (ground truth for records)
    np.testing.assert_array_equal(leaked.label.realized["leak_flow"], leak_flow)


def test_row_load_knobs_take_effect(cfg, nets, plan):
    # AC 3: weekday, per-consumer sigma, and knot jitter all matter
    row = pick(plan, "bulk", "normal")
    rcfg = row_config(cfg, row)
    dcn = build_dcn(nets[row.static_draw_id], rcfg)
    base = plan_row_loads(rcfg, row, dcn)

    # force a day-kind flip: a 24 h bulk horizon lives inside one day, so
    # weekday->weekday shifts are invisible by construction
    weekend = dataclasses.replace(
        row, start_weekday=5 if row.start_weekday < 5 else 0
    )
    assert any(
        not np.allclose(base[j], plan_row_loads(rcfg, weekend, dcn)[j]) for j in base
    )

    quiet = dataclasses.replace(
        row, noise_sigma={j: 0.0 for j in row.noise_sigma}
    )
    quiet_loads = plan_row_loads(rcfg, quiet, dcn)
    assert all(not np.array_equal(base[j], quiet_loads[j]) for j in base)

    flat = dataclasses.replace(
        row,
        knot_jitter={
            a: {k: tuple(1.0 for _ in f) for k, f in kinds.items()}
            for a, kinds in row.knot_jitter.items()
        },
    )
    flat_loads = plan_row_loads(rcfg, flat, dcn)
    assert any(not np.allclose(base[j], flat_loads[j]) for j in base)


def test_year_row_applies_the_seasonal_draw(cfg, nets, plan):
    # AC 3: amplitude/phase from the row modulate the year-tier loads
    row = pick(plan, "month", "normal")
    rcfg = row_config(cfg, row)
    dcn = build_dcn(nets[row.static_draw_id], rcfg)
    seasonal = plan_row_loads(rcfg, row, dcn)
    flat = plan_row_loads(
        rcfg, dataclasses.replace(row, seasonal_amplitude=0.0), dcn
    )
    assert all(not np.array_equal(seasonal[j], flat[j]) for j in seasonal)
    # the scoop only ever reduces load relative to the flat twin
    for j in seasonal:
        assert np.all(seasonal[j] <= flat[j] + 1e-9)
