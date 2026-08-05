"""Scenario-runner seam: eps-NTU ETS with the warm-started
hydraulic-thermal fixed-point iteration on the mini network.

With the co-move secondary schedule (secondary = primary + ``ets.approach``
at both ends, so dT_building == dT_design) the design point is the
*balanced* C_r = 1 case, and these runs
exercise the balanced band in vivo. Design UA follows by inversion
(ets.design_ua).
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.injector import inject_bypass
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory, cfg):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def healthy_ua(dcn, cfg):
    return {j: design_ua(c.design_load, cfg) for j, c in dcn.consumers.items()}


def design_loads(dcn):
    return {j: c.design_load for j, c in dcn.consumers.items()}


def test_converges_within_three_iterations_near_design(mini_dcn, cfg):
    result = run_eps_ntu_scenario(
        mini_dcn, cfg, design_loads(mini_dcn), n_steps=10, ua=healthy_ua(mini_dcn, cfg)
    )

    assert result.converged.all()
    assert (result.iterations <= 3).all()
    assert not result.unmet.to_numpy().any()


def test_eps_ntu_agrees_with_fixed_dt_at_design(mini_dcn, cfg):
    from dcngen.orchestrate.scenario import run_fixed_dt_scenario

    loads = design_loads(mini_dcn)
    ntu = run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=15,
                               ua=healthy_ua(mini_dcn, cfg))
    fdt = run_fixed_dt_scenario(mini_dcn, cfg, loads, n_steps=15)

    # at the design point the two paths describe the same physics: flows
    # agree to the pipe-heat-gain scale (the eps-NTU valve correctly pulls
    # ~0.1 % more where arrival water is warmed en route — physics the
    # fixed-dT path ignores, hence the 0.5 % band, not exact equality)
    np.testing.assert_allclose(
        ntu.consumer_flow.iloc[-1].to_numpy(),
        fdt.consumer_flow.iloc[-1].to_numpy(),
        rtol=5e-3,
    )
    split = ntu.ets_return.iloc[-1] - ntu.supply_temp.iloc[-1][ntu.ets_return.columns]
    assert np.allclose(split.to_numpy(), cfg.thermal.dT_design, atol=0.05)


def test_plug_state_advances_once_regardless_of_iteration_count(mini_dcn, cfg):
    # snapshot/restore: if plugs advanced once per ITERATION
    # instead of once per timestep, water would travel a multiple of the
    # true distance wherever iterations exceed 1 — corrupting transport
    # delay. Force multi-iteration steps with a setpoint drop (arrival-temp
    # change makes the valves re-balance) at tenth loads (transit of pipe 1
    # = 6.28 steps) and require the front to arrive at consumer 2 at the
    # SAME step whether the cap is 1 or 5. Plateau values may differ only
    # at iteration-tolerance scale.
    loads = {j: q / 10.0 for j, q in design_loads(mini_dcn).items()}
    ua = {j: u / 10.0 for j, u in healthy_ua(mini_dcn, cfg).items()}
    setpoint = np.full(28, cfg.thermal.T_supply)
    setpoint[12:] = cfg.thermal.T_supply - 1.0

    runs = [
        run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=28, ua=ua,
                             supply_setpoint=setpoint, max_iterations=cap)
        for cap in (1, 5)
    ]
    assert (runs[1].iterations > 1).any(), "test premise: some steps iterate"

    arrivals = []
    for run in runs:
        t2 = run.supply_temp["2"].to_numpy()
        moved = np.nonzero(t2[12:] < t2[11] - 0.5)[0]
        assert moved.size, "front never arrived"
        arrivals.append(12 + moved[0])
    assert arrivals[0] == arrivals[1]
    # plateaus agree to iteration-accuracy; only the front-crossing steps
    # deviate visibly (the within-step mixing fraction of a 1 K front moves
    # with the flow iterate) — far below the 1 K front a double-advance
    # would displace
    np.testing.assert_allclose(
        runs[0].supply_temp.to_numpy(),
        runs[1].supply_temp.to_numpy(),
        rtol=0,
        atol=0.05,
    )


def _fixed_dt_model(cfg):
    from dcngen.thermal.thermal_solver import EtsStepOutcome

    def model(_j, arrive, m_now):
        return EtsStepOutcome(
            return_temp=arrive + cfg.thermal.dT_design,
            desired_mass_flow=m_now,
            delivered_load=m_now * cfg.water.cp * cfg.thermal.dT_design,
            unmet=False,
        )

    return model


def test_network_state_clone_is_independent(mini_dcn, cfg):
    # the snapshot the iteration restores from must be untouched by
    # propagating a clone
    from dcngen.hydraulics.wntr_model import solve_steady
    from dcngen.thermal.thermal_solver import init_network_state, propagate_step

    state = init_network_state(mini_dcn, cfg)
    snapshot = state.clone()
    before = {n: snapshot.pipes[n].volume.copy() for n in snapshot.pipes}

    hyd = solve_steady(mini_dcn)
    propagate_step(
        mini_dcn, state, hyd.flow, cfg.thermal.T_supply, _fixed_dt_model(cfg), cfg
    )
    for name, vol in before.items():
        np.testing.assert_array_equal(snapshot.pipes[name].volume, vol)
        assert snapshot.pipes[name].residence[0] == 0.0


def test_final_plug_state_equals_one_advance_at_recorded_flows(mini_dcn, cfg):
    # "plug state after a timestep is identical whether it
    # converged in 1 or 5 iterations". The strict reading: however many
    # iterations ran, the accepted state equals exactly ONE propagation of
    # the INITIAL snapshot at the recorded (last solved) flows — where the
    # initial snapshot is the steady operating point at the warm-start
    # flows, reconstructed here independently.
    # Fouling forces multiple iterations on the first step.
    from dcngen.hydraulics.wntr_model import solve_steady
    from dcngen.thermal.ets import (
        building_hot_inlet_temp,
        fixed_dt_primary_flow,
        transferred_load,
    )
    from dcngen.thermal.thermal_solver import (
        EtsStepOutcome,
        propagate_step,
        steady_initial_state,
    )

    loads = design_loads(mini_dcn)
    ua = healthy_ua(mini_dcn, cfg)
    ua["2"] = 0.5 * ua["2"]
    run = run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=1, ua=ua)
    assert run.iterations[0] > 1, "test premise: the step iterated"

    t_h_in = building_hot_inlet_temp(cfg)

    def init_model(j, arrive, m_now):
        m_sec = loads[j] / (cfg.water.cp * cfg.thermal.dT_design)
        q_now = transferred_load(m_now, ua[j], m_sec, cfg.water.cp, t_h_in - arrive)
        return EtsStepOutcome(
            arrive + (q_now / (m_now * cfg.water.cp) if m_now > 0 else 0.0),
            m_now, q_now, False,
        )

    warm = {j: fixed_dt_primary_flow(loads[j], cfg) for j in loads}
    hyd0 = solve_steady(mini_dcn, consumer_flows=warm)
    expected = steady_initial_state(
        mini_dcn, cfg, hyd0.flow, cfg.thermal.T_supply, init_model
    )
    hyd = solve_steady(
        mini_dcn, consumer_flows=run.consumer_flow.iloc[0].to_dict()
    )
    # the reference ETS model only needs to reproduce the return temps the
    # accepted iterate saw; those are recorded per consumer
    recorded_returns = run.ets_return.iloc[0].to_dict()

    propagate_step(
        mini_dcn,
        expected,
        hyd.flow,
        cfg.thermal.T_supply,
        lambda j, _arrive, m_now: EtsStepOutcome(recorded_returns[j], m_now, 0.0, False),
        cfg,
    )
    for name, queue in run.thermal_state.pipes.items():
        np.testing.assert_allclose(queue.volume, expected.pipes[name].volume, rtol=1e-12)
        np.testing.assert_allclose(
            queue.residence, expected.pipes[name].residence, rtol=0, atol=1e-9
        )


def test_fouling_collapses_dt_and_raises_flow(mini_dcn, cfg):
    # UA halved at consumer "2" vs healthy baseline
    loads = design_loads(mini_dcn)
    ua = healthy_ua(mini_dcn, cfg)
    fouled_ua = {**ua, "2": 0.5 * ua["2"]}

    healthy = run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=15, ua=ua)
    fouled = run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=15, ua=fouled_ua)

    h_split = healthy.ets_return.iloc[-1]["2"] - healthy.supply_temp.iloc[-1]["2"]
    f_split = fouled.ets_return.iloc[-1]["2"] - fouled.supply_temp.iloc[-1]["2"]
    h_flow = healthy.consumer_flow.iloc[-1]["2"]
    f_flow = fouled.consumer_flow.iloc[-1]["2"]

    assert f_split < h_split - 1.0  # dT visibly collapses (> 1 K)
    assert f_flow > h_flow * 1.15  # flow visibly rises (> 15 %)
    # the untouched consumer "4" is essentially unaffected
    assert fouled.consumer_flow.iloc[-1]["4"] == pytest.approx(
        healthy.consumer_flow.iloc[-1]["4"], rel=0.01
    )


def test_whole_horizon_bypass_scales_flow_and_collapses_split(mini_dcn, cfg):
    # against the healthy twin, a bypassed consumer must pull
    # ~1/(1-f) times the flow while its measured split collapses to ~(1-f)
    # of the healthy split (dT_mix = (1-f) * dT_hx) — the load still met.
    # Tolerances absorb the slightly different arrival temperatures the
    # changed network flow produces; the closed-form identity itself is
    # pinned at the kernel seam (test_ets).
    f = 0.3
    loads = design_loads(mini_dcn)
    times = np.arange(15) * cfg.solver.dt
    bypass = inject_bypass(
        mini_dcn, cfg, times, np.random.default_rng(0),
        consumer="2", fraction=f, whole_horizon=True,
    )

    healthy = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=15, ua=healthy_ua(mini_dcn, cfg)
    )
    bypassed = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=15, ua=healthy_ua(mini_dcn, cfg), bypass=bypass
    )

    assert not bypassed.unmet.to_numpy().any()
    assert bypassed.delivered_load.iloc[-1]["2"] == pytest.approx(
        loads["2"], rel=1e-3
    )
    flow_ratio = (
        bypassed.consumer_flow.iloc[-1]["2"] / healthy.consumer_flow.iloc[-1]["2"]
    )
    assert flow_ratio == pytest.approx(1.0 / (1.0 - f), rel=0.02)
    h_split = healthy.ets_return.iloc[-1]["2"] - healthy.supply_temp.iloc[-1]["2"]
    b_split = bypassed.ets_return.iloc[-1]["2"] - bypassed.supply_temp.iloc[-1]["2"]
    assert b_split == pytest.approx((1.0 - f) * h_split, rel=0.02)
    # the untouched consumer "4" is essentially unaffected
    assert bypassed.consumer_flow.iloc[-1]["4"] == pytest.approx(
        healthy.consumer_flow.iloc[-1]["4"], rel=0.01
    )


def test_abrupt_bypass_diverges_only_after_onset(mini_dcn, cfg):
    n_steps = 15
    loads = design_loads(mini_dcn)
    times = np.arange(n_steps) * cfg.solver.dt
    bypass = inject_bypass(
        mini_dcn, cfg, times, np.random.default_rng(7),
        consumer="2", fraction=0.3, whole_horizon=False,
    )
    k_onset = int(np.searchsorted(times, bypass.label.onset))
    assert 0 < k_onset < n_steps, "test premise: onset strictly inside"

    healthy = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=n_steps, ua=healthy_ua(mini_dcn, cfg)
    )
    faulty = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=n_steps, ua=healthy_ua(mini_dcn, cfg),
        bypass=bypass,
    )

    # before onset the fault does not exist: the twins solve the same steps
    np.testing.assert_allclose(
        faulty.consumer_flow.iloc[:k_onset].to_numpy(),
        healthy.consumer_flow.iloc[:k_onset].to_numpy(),
        rtol=1e-6,
    )
    # from onset the bypassed consumer visibly overdraws (f=0.3 -> ~1.43x)
    assert (
        faulty.consumer_flow.iloc[-1]["2"] > 1.2 * healthy.consumer_flow.iloc[-1]["2"]
    )


def test_bypass_mask_must_match_the_horizon(mini_dcn, cfg):
    times = np.arange(10) * cfg.solver.dt
    bypass = inject_bypass(
        mini_dcn, cfg, times, np.random.default_rng(0), consumer="2",
    )
    with pytest.raises(ValueError, match="mask"):
        run_eps_ntu_scenario(
            mini_dcn, cfg, design_loads(mini_dcn), n_steps=15,
            ua=healthy_ua(mini_dcn, cfg), bypass=bypass,
        )


def test_bypass_consumer_must_be_loaded(mini_dcn, cfg):
    loads = design_loads(mini_dcn)
    times = np.arange(10) * cfg.solver.dt
    bypass = inject_bypass(
        mini_dcn, cfg, times, np.random.default_rng(0), consumer="2",
    )
    loads.pop("2")
    ua = {j: u for j, u in healthy_ua(mini_dcn, cfg).items() if j != "2"}
    with pytest.raises(ValueError, match="consumer"):
        run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=10, ua=ua, bypass=bypass)


def test_severe_fouling_saturates_valve_and_flags_unmet(mini_dcn, cfg):
    # UA/10: even the fully open valve (max_flow_factor x design) cannot
    # transfer the demanded load -> the scenario-level unmet flag must fire
    # and the delivered load must fall short
    loads = design_loads(mini_dcn)
    ua = healthy_ua(mini_dcn, cfg)
    run = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=10, ua={**ua, "2": 0.1 * ua["2"]}
    )

    assert run.unmet["2"].iloc[-1]
    assert run.delivered_load["2"].iloc[-1] < 0.9 * loads["2"]
    assert not run.unmet["4"].any()


def test_whole_horizon_bypass_has_no_phantom_onset(mini_dcn, cfg):
    # an "open since commissioning" fault must START
    # in its faulted steady state. Before steady init, the design-temp fill
    # relaxed into the bypassed state over the first flush — a phantom
    # onset contradicting the whole-horizon label. Now the very first
    # recorded step already shows the collapsed split, and it matches the
    # settled value instead of drifting toward it.
    f = 0.3
    loads = design_loads(mini_dcn)
    times = np.arange(12) * cfg.solver.dt
    bypass = inject_bypass(
        mini_dcn, cfg, times, np.random.default_rng(0),
        consumer="2", fraction=f, whole_horizon=True,
    )
    run = run_eps_ntu_scenario(
        mini_dcn, cfg, loads, n_steps=12, ua=healthy_ua(mini_dcn, cfg),
        bypass=bypass,
    )

    split = run.ets_return["2"] - run.supply_temp["2"]
    # bypassed from the first step: far below the design split already
    assert split.iloc[0] < (1.0 - f) * cfg.thermal.dT_design * 1.2
    # ... and already settled: first and last steps agree closely
    assert split.iloc[0] == pytest.approx(split.iloc[-1], rel=0.02)


def test_leak_active_at_t0_is_rejected(mini_dcn, cfg):
    # the steady pre-history is leak-free; an onset-0 leak
    # would reintroduce a phantom onset at t=0
    from dcngen.faults.injector import inject_abrupt_leak

    times = np.arange(12) * cfg.solver.dt
    leak = inject_abrupt_leak(
        mini_dcn, cfg, times, None, junction="3", side="supply",
        severity=0.05, onset=0.0,
    )
    with pytest.raises(ValueError, match="t=0"):
        run_eps_ntu_scenario(
            mini_dcn, cfg, design_loads(mini_dcn), n_steps=12,
            ua=healthy_ua(mini_dcn, cfg), leak=leak,
        )
