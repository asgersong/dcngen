"""ETS models: the fixed-dT path, the pure eps-NTU kernel, and the
config-derived closure layer.

Kernel tests use explicit, thermodynamically feasible synthetic values
(hot inlet 14 degC against cold inlet 6) independent of the config — hand
literals throughout. The closure-layer tests below pin the co-move rule:
the secondary program derives as primary + ``ets.approach`` at both ends.
"""


import pytest

from dcngen.config import load_config
from dcngen.thermal.ets import (
    building_hot_inlet_temp,
    building_supply_temp,
    counterflow_effectiveness,
    design_ua,
    fixed_dt_primary_flow,
    fixed_dt_return_temp,
    installed_ua,
    solve_ets,
)
from tests.conftest import write_mutated_default


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_hinge_equation_load_to_flow(cfg):
    # 1 MW at dT=7 K: m = 1e6/(4186*7) = 34.1297... kg/s -> /999.5 m3/s
    volumetric = fixed_dt_primary_flow(1.0e6, cfg)
    assert volumetric == pytest.approx(0.034147, rel=1e-4)


def test_return_temp_is_supply_plus_design_split(cfg):
    assert fixed_dt_return_temp(6.4, cfg) == pytest.approx(13.4)


CP = 4186.0


def test_counterflow_effectiveness_hand_value():
    # NTU=2, C_r=0.5: e=exp(-1); eps=(1-e)/(1-0.5e) = 0.6321206/0.8160603
    assert counterflow_effectiveness(2.0, 0.5) == pytest.approx(0.774600, abs=1e-5)


def test_counterflow_effectiveness_balanced_limit_is_continuous():
    # C_r -> 1 must approach NTU/(1+NTU) without blowing up (0/0 in the
    # general formula); the co-move closure makes this a mandatory case.
    exact = 2.0 / 3.0
    assert counterflow_effectiveness(2.0, 1.0) == pytest.approx(exact, abs=1e-12)
    assert counterflow_effectiveness(2.0, 1.0 - 1e-9) == pytest.approx(exact, abs=1e-6)
    assert counterflow_effectiveness(2.0, 1.0 - 1e-4) == pytest.approx(exact, abs=1e-4)


def kernel_case(load=100_000.0, t_c_in=6.0, t_h_in=14.0, dt_building=7.0):
    """A feasible synthetic ETS case; secondary flow from the closure."""
    m_sec = load / (CP * dt_building)
    return dict(load=load, T_c_in=t_c_in, T_h_in=t_h_in, m_sec=m_sec, cp=CP)


def test_solve_ets_delivers_demand_when_feasible():
    # note the closure (m_sec ~ load) demands eps >= dT_b/dT_inlet = 7/8
    # whatever the load — UA must be generous (hand-checked: at m_p=10,
    # ua=60k delivers ~109.5 kW >= 100 kW)
    case = kernel_case()
    op = solve_ets(ua=60_000.0, m_p_max=10.0, **case)

    assert not op.unmet
    assert op.delivered_load == pytest.approx(case["load"], rel=1e-9)
    # energy consistency: Q = m_p * cp * (T_out - T_in)
    q = op.primary_mass_flow * CP * (op.return_temp - case["T_c_in"])
    assert q == pytest.approx(case["load"], rel=1e-9)
    # second law: primary cannot leave hotter than the hot inlet
    assert op.return_temp < case["T_h_in"]


def test_solve_ets_flags_unmet_load_at_valve_cap():
    case = kernel_case()
    # tiny UA: even the max flow cannot transfer the demanded load
    op = solve_ets(ua=2_000.0, m_p_max=5.0, **case)

    assert op.unmet
    assert op.primary_mass_flow == pytest.approx(5.0)
    assert op.delivered_load < case["load"]


def test_solve_ets_delivered_load_saturates_at_secondary_capacity():
    # with huge UA and huge primary flow, Q_max = C_sec * (T_h_in - T_c_in):
    # the C_min side switches to the secondary and caps the transfer
    case = kernel_case()
    c_sec = case["m_sec"] * CP
    ceiling = c_sec * (case["T_h_in"] - case["T_c_in"])

    op = solve_ets(ua=1e9, m_p_max=1e4, **{**case, "load": ceiling * 1.5})
    assert op.unmet
    assert op.delivered_load == pytest.approx(ceiling, rel=1e-6)


def test_solve_ets_fouling_monotonic_flow_rise_and_dt_collapse():
    # decreasing UA at fixed demanded load: primary flow must rise and the
    # primary split (return - supply) must collapse — low-dT syndrome
    case = kernel_case()
    ops = [solve_ets(ua=ua, m_p_max=50.0, **case) for ua in (60_000.0, 45_000.0, 35_000.0)]

    flows = [op.primary_mass_flow for op in ops]
    splits = [op.return_temp - case["T_c_in"] for op in ops]
    assert flows[0] < flows[1] < flows[2]
    assert splits[0] > splits[1] > splits[2]
    # all still meet the load; dT * flow stays proportional to it
    for op, split in zip(ops, splits):
        assert not op.unmet
        assert op.primary_mass_flow * CP * split == pytest.approx(case["load"], rel=1e-9)


def test_bypass_scales_total_flow_and_collapses_the_mixed_split():
    # bypass closed forms:
    # unsaturated, the exchanger must transfer the same load from the same
    # inlet states, so its flow is the no-bypass root and the TOTAL primary
    # flow scales as 1/(1-f); the mixed split obeys dT_mix = (1-f)*dT_hx.
    case = kernel_case()
    f = 0.3
    nobypass = solve_ets(ua=60_000.0, m_p_max=10.0, **case)
    op = solve_ets(ua=60_000.0, m_p_max=10.0, bypass_fraction=f, **case)

    assert not op.unmet
    assert op.delivered_load == pytest.approx(case["load"], rel=1e-9)
    # the exchanger sees (1-f) of the total primary flow
    assert op.primary_mass_flow * (1.0 - f) == pytest.approx(
        nobypass.primary_mass_flow, rel=1e-9
    )
    # mixing identity: T_ret = f*T_arrive + (1-f)*T_hx,out
    assert op.return_temp == pytest.approx(
        f * case["T_c_in"] + (1.0 - f) * nobypass.return_temp, rel=1e-9
    )
    # equivalent split form of the same identity
    assert op.return_temp - case["T_c_in"] == pytest.approx(
        (1.0 - f) * (nobypass.return_temp - case["T_c_in"]), rel=1e-9
    )
    # energy consistency at the TOTAL flow
    q = op.primary_mass_flow * CP * (op.return_temp - case["T_c_in"])
    assert q == pytest.approx(case["load"], rel=1e-9)


def test_bypass_valve_cap_and_unmet_semantics_unchanged():
    # the valve limit stays on the TOTAL primary flow: with f = 0.5 only
    # half of m_p_max reaches the exchanger, the load becomes infeasible,
    # and the ETS runs fully open with the shortfall flagged
    case = kernel_case()
    op = solve_ets(ua=60_000.0, m_p_max=5.0, bypass_fraction=0.5, **case)

    assert op.unmet
    assert op.primary_mass_flow == pytest.approx(5.0)
    assert op.delivered_load < case["load"]
    # the mixed return still balances energy at the total flow
    q = op.primary_mass_flow * CP * (op.return_temp - case["T_c_in"])
    assert q == pytest.approx(op.delivered_load, rel=1e-9)
    # ... and the same case without bypass was feasible (m* ~ 4.4 < 5)
    assert not solve_ets(ua=60_000.0, m_p_max=5.0, **case).unmet


def test_bypass_fraction_outside_the_unit_interval_rejected():
    case = kernel_case()
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            solve_ets(ua=60_000.0, m_p_max=10.0, bypass_fraction=bad, **case)


def test_installed_ua_composes_margin_and_fouling(cfg):
    # UA at construction = design sizing x healthy margin
    # (U(1.00, 1.10)) x fouling multiplier m; independent arithmetic
    # via the identity design UA = Q / approach
    load = 1.0e6
    assert installed_ua(load, cfg) == pytest.approx(
        load / cfg.ets.approach, rel=1e-12
    )
    assert installed_ua(load, cfg, healthy_factor=1.07, fouling=0.5) == pytest.approx(
        load * 1.07 * 0.5 / cfg.ets.approach, rel=1e-12
    )


def test_installed_ua_rejects_undersized_margin_and_unphysical_fouling(cfg):
    # installed healthy UA sits AT or ABOVE the design requirement;
    # fouling only ever removes conductance
    with pytest.raises(ValueError):
        installed_ua(1.0e6, cfg, healthy_factor=0.95)
    for bad in (0.0, -0.5, 1.2):
        with pytest.raises(ValueError):
            installed_ua(1.0e6, cfg, fouling=bad)


def test_composed_margin_fouling_bypass_keep_the_closed_form_identities(cfg):
    # healthy-UA factor x fouling m x bypass f at one
    # ETS, at the config design point. The margin alone pulls slightly LESS
    # than design flow (the exchanger is better than required); fouling
    # composed on top pushes flow above design; bypass then scales the total
    # flow by 1/(1-f) and the split by (1-f) against the fouled baseline.
    load = 1.0e6
    m_design = load / (cfg.water.cp * cfg.thermal.dT_design)
    common = dict(
        load=load,
        T_c_in=cfg.thermal.T_supply,
        m_p_max=cfg.ets.max_flow_factor * m_design,
        T_h_in=building_hot_inlet_temp(cfg),
        m_sec=load / (cfg.water.cp * cfg.thermal.dT_design),
        cp=cfg.water.cp,
    )
    h, m, f = 1.08, 0.6, 0.25
    margin_only = solve_ets(ua=installed_ua(load, cfg, healthy_factor=h), **common)
    fouled = solve_ets(ua=installed_ua(load, cfg, healthy_factor=h, fouling=m), **common)
    composed = solve_ets(
        ua=installed_ua(load, cfg, healthy_factor=h, fouling=m),
        bypass_fraction=f,
        **common,
    )

    assert margin_only.primary_mass_flow < m_design < fouled.primary_mass_flow
    for op in (margin_only, fouled, composed):
        assert not op.unmet
        assert op.delivered_load == pytest.approx(load, rel=1e-9)
    assert composed.primary_mass_flow == pytest.approx(
        fouled.primary_mass_flow / (1.0 - f), rel=1e-9
    )
    assert composed.return_temp - cfg.thermal.T_supply == pytest.approx(
        (1.0 - f) * (fouled.return_temp - cfg.thermal.T_supply), rel=1e-9
    )


def test_secondary_program_derives_from_primary_plus_approach(cfg):
    # identity: the derived program at the default 6/13 primary
    # equals the previously *stated* 7/14 degC secondary schedule exactly
    assert building_supply_temp(cfg) == 7.0
    assert building_hot_inlet_temp(cfg) == 14.0


def test_secondary_program_co_moves_with_plant_setpoint(tmp_path):
    # the point of the reshape: shift the plant program and the secondary
    # follows coherently, while effectiveness and UA stay invariant
    def shift_program(d):
        d["thermal"]["T_supply"] = 5.0
        d["thermal"]["T_return_design"] = 12.0

    shifted = load_config(write_mutated_default(tmp_path, shift_program))

    assert building_supply_temp(shifted) == 6.0
    assert building_hot_inlet_temp(shifted) == 13.0
    assert design_ua(1.0e6, shifted) == pytest.approx(
        1.0e6 / shifted.ets.approach, rel=1e-12
    )


def test_design_ua_is_load_per_approach(cfg):
    # balanced design point: eps = dT/(dT+approach),
    # NTU = dT/approach, hence UA = Q_design / approach
    assert design_ua(1.0e6, cfg) == pytest.approx(1.0e6 / cfg.ets.approach, rel=1e-12)
    assert design_ua(250_000.0, cfg) == pytest.approx(
        250_000.0 / cfg.ets.approach, rel=1e-12
    )


def test_design_ua_round_trip_reproduces_design_point_exactly(cfg):
    # "fixed-dT and eps-NTU agree at the design point with
    # reference UA" — the tight-tolerance kernel form: at arrival exactly
    # T_supply, design UA must reproduce the design flow and design return.
    load = 1.0e6  # W
    m_design = load / (cfg.water.cp * cfg.thermal.dT_design)
    op = solve_ets(
        load=load,
        T_c_in=cfg.thermal.T_supply,
        ua=design_ua(load, cfg),
        m_p_max=cfg.ets.max_flow_factor * m_design,
        T_h_in=building_hot_inlet_temp(cfg),
        m_sec=load / (cfg.water.cp * cfg.thermal.dT_design),
        cp=cfg.water.cp,
    )

    assert not op.unmet
    assert op.primary_mass_flow == pytest.approx(m_design, rel=1e-9)
    assert op.return_temp == pytest.approx(cfg.thermal.T_return_design, abs=1e-9)
