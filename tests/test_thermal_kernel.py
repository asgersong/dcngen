"""Pure physics kernel seam: the plug-tracking pipe
step. Tight-tolerance numerics against hand-derived/analytic
expectations.

Kernel conventions under test: a pipe holds a FIFO of plugs (volume, entry
temperature, accumulated residence); temperature is *derived* from residence
(T = T_g + (T_entry - T_g) * exp(-residence / tau)), so decay can never be
applied twice. ``step_pipe`` advances one timestep with a given displaced
volume and inlet temperature and returns the volume-weighted outlet
temperature of the water that left.
"""

import math

import numpy as np
import pytest

from dcngen.thermal.thermal_solver import (
    PlugState,
    mix_temperature,
    step_pipe,
)

T_GROUND = 20.0


def full_pipe(volume: float, temp: float) -> PlugState:
    """A pipe initially filled with one plug of the given temperature."""
    return PlugState(
        volume=np.array([volume]),
        entry_temp=np.array([temp]),
        residence=np.array([0.0]),
    )


def test_step_change_arrives_after_exact_transport_delay_unsmeared():
    # Pipe volume 350 m3, displaced volume 100 m3/step -> transit = 3.5 steps.
    # A step change at the inlet must exit pure-old for 3 steps, mixed in the
    # step containing t = 3.5, and pure-new from step 4 on. tau_thermal=inf
    # isolates advection (no heat gain).
    state = full_pipe(350.0, 6.0)
    outlet = []
    for _ in range(6):
        state, t_out = step_pipe(
            state,
            displaced_volume=100.0,
            inlet_temp=9.0,
            dt=600.0,
            tau_thermal=np.inf,
            T_ground=T_GROUND,
        )
        outlet.append(t_out)

    assert outlet[0] == pytest.approx(6.0)
    assert outlet[1] == pytest.approx(6.0)
    assert outlet[2] == pytest.approx(6.0)
    # step 4 drains the last 50 m3 of old water and 50 m3 of new: even mix
    assert outlet[3] == pytest.approx(7.5)
    assert outlet[4] == pytest.approx(9.0)
    assert outlet[5] == pytest.approx(9.0)


def test_steady_outlet_matches_analytic_exponential_decay():
    # V=350 m3, 100 m3/step -> transit 3.5*600 s; tau=3000 s. After the
    # initial fill is flushed, the outlet must sit at the analytic
    # steady-state solution to near machine precision (every exiting parcel
    # then has residence exactly equal to the transit time).
    dt, tau, t_in = 600.0, 3000.0, 6.0
    transit = 3.5 * dt
    analytic = T_GROUND + (t_in - T_GROUND) * math.exp(-transit / tau)

    state = full_pipe(350.0, t_in)
    outlet = []
    for _ in range(15):
        state, t_out = step_pipe(
            state,
            displaced_volume=100.0,
            inlet_temp=t_in,
            dt=dt,
            tau_thermal=tau,
            T_ground=T_GROUND,
        )
        outlet.append(t_out)

    for t_out in outlet[5:]:
        assert t_out == pytest.approx(analytic, rel=1e-9)


def test_passthrough_regime_matches_analytic_decay():
    # Displaced volume (350) exceeds pipe volume (100): most water traverses
    # within one step. At steady state every exiting parcel still has
    # residence exactly V/Q = 100/(350/600) s, so the outlet must sit at the
    # analytic decay — this is the regime of short pipes (e.g. plant stub).
    dt, tau, t_in = 600.0, 1000.0, 6.0
    transit = 100.0 / (350.0 / dt)
    analytic = T_GROUND + (t_in - T_GROUND) * math.exp(-transit / tau)

    state = full_pipe(100.0, t_in)
    for _ in range(4):
        state, t_out = step_pipe(
            state, displaced_volume=350.0, inlet_temp=t_in, dt=dt,
            tau_thermal=tau, T_ground=T_GROUND,
        )
    assert t_out == pytest.approx(analytic, rel=1e-9)


def test_idle_plug_decays_by_total_residence_no_double_decay():
    # Three zero-flow steps then a full flush: the exiting water must have
    # decayed by its TOTAL residence (3*dt idle + dt/2 mean exit time),
    # regardless of how many steps touched it. tau >> dt keeps the exp
    # curvature across the flush negligible.
    dt, tau, t0 = 600.0, 1.0e6, 6.0
    state = full_pipe(350.0, t0)
    for _ in range(3):
        state, t_out = step_pipe(
            state, displaced_volume=0.0, inlet_temp=9.0, dt=dt,
            tau_thermal=tau, T_ground=T_GROUND,
        )
        assert math.isnan(t_out)  # nothing exits a zero-flow pipe

    state, t_out = step_pipe(
        state, displaced_volume=350.0, inlet_temp=9.0, dt=dt,
        tau_thermal=tau, T_ground=T_GROUND,
    )
    expected = T_GROUND + (t0 - T_GROUND) * math.exp(-3.5 * dt / tau)
    assert t_out == pytest.approx(expected, rel=1e-6)


def test_step_pipe_is_pure_snapshot_restore_safe():
    # the hydraulic-thermal iteration re-solves a timestep from a snapshot; the
    # kernel must not mutate its input, and re-stepping the same state must
    # reproduce identical results.
    state = full_pipe(350.0, 6.0)
    before = (state.volume.copy(), state.entry_temp.copy(),
              state.residence.copy(), state.res_span.copy())

    s1, out1 = step_pipe(state, 100.0, 9.0, 600.0, 3000.0, T_GROUND)
    s2, out2 = step_pipe(state, 100.0, 9.0, 600.0, 3000.0, T_GROUND)

    assert out1 == out2
    np.testing.assert_array_equal(state.volume, before[0])
    np.testing.assert_array_equal(state.entry_temp, before[1])
    np.testing.assert_array_equal(state.residence, before[2])
    np.testing.assert_array_equal(state.res_span, before[3])
    np.testing.assert_array_equal(s1.volume, s2.volume)
    np.testing.assert_array_equal(s1.residence, s2.residence)


def test_node_mixing_is_flow_weighted():
    # 2 volumes at 6 degC + 1 volume at 9 degC -> 7 degC (hand literal)
    assert mix_temperature(np.array([2.0, 1.0]), np.array([6.0, 9.0])) == 7.0


# ---------------------------------------------- steady initial state (#31)


@pytest.fixture(scope="module")
def steady_setup(tmp_path_factory):
    from dcngen.config import load_config
    from dcngen.hydraulics.wntr_model import solve_steady
    from dcngen.topology.dcnifier import build_dcn
    from dcngen.topology.ditec_loader import load_ditec_network
    from tests.conftest import write_ditec_folder

    cfg = load_config()
    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg)
    design = {j: c.design_flow for j, c in dcn.consumers.items()}
    hyd = solve_steady(dcn, consumer_flows=design)
    return cfg, dcn, hyd


def _fixed_dt_model(cfg):
    from dcngen.thermal.thermal_solver import EtsStepOutcome

    dT, cp = cfg.thermal.dT_design, cfg.water.cp

    def model(_j, arrive, m_now):
        return EtsStepOutcome(arrive + dT, m_now, m_now * cp * dT, False)

    return model


def test_stored_energy_is_rho_cp_volume_temperature_sum():
    from dcngen.config import load_config
    from dcngen.thermal.thermal_solver import (
        NetworkThermalState,
        PlugState,
        stored_energy,
    )

    cfg = load_config()
    # one pipe, two plugs at residence 0 (temps = entry temps), 10 + 5 m3
    state = NetworkThermalState(
        pipes={"p": PlugState(
            volume=np.array([10.0, 5.0]),
            entry_temp=np.array([6.0, 13.0]),
            residence=np.array([0.0, 0.0]),
        )},
        orientation={"p": 1},
        pipe_volume={"p": 15.0},
        tau_thermal={"p": 1.0e9},  # effectively no decay at residence 0
    )
    expected = cfg.water.rho * cfg.water.cp * (10.0 * 6.0 + 5.0 * 13.0)
    assert stored_energy(state, cfg) == pytest.approx(expected, rel=1e-12)


def test_steady_init_reproduces_the_analytic_pipe_profile(steady_setup):
    from dcngen.thermal.thermal_solver import plug_temperatures, steady_initial_state

    cfg, dcn, hyd = steady_setup
    state = steady_initial_state(
        dcn, cfg, hyd.flow, cfg.thermal.T_supply, _fixed_dt_model(cfg)
    )

    # first supply pipe out of the plant: inlet water is at the setpoint,
    # the outlet-end front has aged one transit time — closed form
    first = next(
        p for p in dcn.wn.pipe_name_list
        if dcn.wn.get_link(p).start_node_name == dcn.plant.supply_header
        or dcn.wn.get_link(p).end_node_name == dcn.plant.supply_header
    )
    queue = state.pipes[first]
    tau_th = state.tau_thermal[first]
    transit = state.pipe_volume[first] / abs(hyd.flow[first])
    # the plant-adjacent pipe's steady inlet IS the setpoint, and one plug
    # spans the pipe with residence [0, transit]: mean transit/2, span
    # transit — the exact exponential profile in residence-linear form
    assert queue.entry_temp[0] == pytest.approx(cfg.thermal.T_supply, rel=1e-12)
    assert queue.residence[0] == pytest.approx(transit / 2.0, rel=1e-12)
    assert queue.residence[0] + queue.res_span[0] / 2.0 == pytest.approx(
        transit, rel=1e-12
    )
    # the volume-mean plug temperature follows the mean residence
    temps = plug_temperatures(queue, tau_th, cfg.thermal.T_ground)
    assert temps[0] == pytest.approx(
        cfg.thermal.T_ground
        + (cfg.thermal.T_supply - cfg.thermal.T_ground)
        * np.exp(-(transit / 2.0) / tau_th),
        rel=1e-12,
    )


def test_marching_from_steady_init_is_an_exact_fixed_point_under_fixed_dt(steady_setup):
    # THE defining property: in steady state the plug
    # model's slice-curvature error term is exactly zero, so propagating
    # from a correct steady fill at constant conditions must reproduce the
    # same node temperatures step after step to float precision.
    from dcngen.thermal.thermal_solver import propagate_step, steady_initial_state

    cfg, dcn, hyd = steady_setup
    model = _fixed_dt_model(cfg)
    state = steady_initial_state(dcn, cfg, hyd.flow, cfg.thermal.T_supply, model)

    first = propagate_step(
        dcn, state.clone(), hyd.flow, cfg.thermal.T_supply, model, cfg
    )
    marching = state.clone()
    for _ in range(5):
        latest = propagate_step(
            dcn, marching, hyd.flow, cfg.thermal.T_supply, model, cfg
        )
    for node, temp in first.node_temp.items():
        if not np.isnan(temp):
            assert latest.node_temp[node] == pytest.approx(temp, abs=1e-9), node


def test_stagnant_branch_holds_day_old_design_water(steady_setup):
    # a consumer at zero draw stagnates its terminal branch CONSISTENTLY
    # (segment pipes + crossover together). Standing water is
    # yesterday's design-side water aged one diurnal cap — slightly warmed
    # toward the ground temperature, NEVER the unbounded steady limit
    # (ground-temperature heat slugs broke the #31 slow tests)
    from dcngen.thermal.thermal_solver import plug_temperatures, steady_initial_state

    cfg, dcn, hyd = steady_setup
    flows = hyd.flow.copy()
    seg_s, seg_r = dcn.pipe_pairing["3"]  # the 3->4 segment's twin pipes
    for link in (seg_s, seg_r, dcn.consumers["4"].ets_link):
        flows[link] = 0.0
    state = steady_initial_state(
        dcn, cfg, flows, cfg.thermal.T_supply, _fixed_dt_model(cfg)
    )
    for pipe, design_temp in (
        (seg_s, cfg.thermal.T_supply), (seg_r, cfg.thermal.T_return_design)
    ):
        temps = plug_temperatures(
            state.pipes[pipe], state.tau_thermal[pipe], cfg.thermal.T_ground
        )
        aged = cfg.thermal.T_ground + (
            design_temp - cfg.thermal.T_ground
        ) * np.exp(-cfg.thermal.init_residence_cap / state.tau_thermal[pipe])
        np.testing.assert_allclose(temps, aged, rtol=1e-12)
        # day-old water is still cold water: nowhere near the ground temp
        assert temps[0] < cfg.thermal.T_ground - 1.0
