"""Scenario-runner seam, fixed-dT scope: constant loads,
fixed-dT ETS, multi-timestep run on the mini network.

Analytic expectations: at steady state the supply temperature decays from the
plant along each path as T = T_g + (T_in - T_g)*exp(-U'L/(m_dot c_p)) per
pipe (independent closed form, not the solver's bookkeeping), with the mini
draw's geometry from conftest and constants from the default config.
"""

import math

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.scenario import run_fixed_dt_scenario
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def design_loads(dcn) -> dict[str, float]:
    return {j: c.design_load for j, c in dcn.consumers.items()}


def pipe_decay(t_in: float, flow: float, length: float) -> float:
    """Analytic steady pipe outlet temperature [degC] at volumetric flow.

    The runner is called without ``pipe_U``, i.e. in the flat dev-fallback
    mode, so the expected decay uses ``heat_gain.scalar_U``.
    """
    m_dot = flow * cfg.water.rho
    exponent = cfg.heat_gain.scalar_U * length / (m_dot * cfg.water.cp)
    return cfg.thermal.T_ground + (t_in - cfg.thermal.T_ground) * math.exp(-exponent)


def test_steady_supply_temps_match_analytic_chain(mini_dcn):
    result = run_fixed_dt_scenario(mini_dcn, cfg, design_loads(mini_dcn), n_steps=20)

    # plant -> 2_s via pipe 1 (L=1000, total flow 0.75); onward 0.40 via
    # pipe 2 (L=800); onward 0.15 via pipe 3 (L=600) — conftest draw sc0
    t2 = pipe_decay(cfg.thermal.T_supply, 0.75, 1000.0)
    t3 = pipe_decay(t2, 0.40, 800.0)
    t4 = pipe_decay(t3, 0.15, 600.0)

    final = result.supply_temp.iloc[-1]
    assert final["2"] == pytest.approx(t2, rel=1e-9)
    assert final["3"] == pytest.approx(t3, rel=1e-9)
    assert final["4"] == pytest.approx(t4, rel=1e-9)


def test_fixed_dt_ets_split_exact_node_dt_approximate(mini_dcn):
    result = run_fixed_dt_scenario(mini_dcn, cfg, design_loads(mini_dcn), n_steps=20)

    # the fixed-dT invariant is exact at each ETS outlet ...
    ets_split = result.ets_return.iloc[-1] - result.supply_temp.iloc[-1][
        result.ets_return.columns
    ]
    assert np.allclose(ets_split.to_numpy(), cfg.thermal.dT_design, atol=1e-9)

    # ... while the *node-level* dT wobbles slightly: return mains mix
    # streams from consumers with different arrival temps and gain heat en
    # route (physical, small at design flows).
    node_dt = result.return_temp.iloc[-1] - result.supply_temp.iloc[-1]
    assert np.allclose(node_dt.to_numpy(), cfg.thermal.dT_design, atol=0.05)


def test_plant_energy_balance_closes_at_steady_state(mini_dcn):
    loads = design_loads(mini_dcn)
    result = run_fixed_dt_scenario(mini_dcn, cfg, loads, n_steps=20)

    total_load = sum(loads.values())
    gains = result.pipe_heat_gain.iloc[-1].sum()
    plant = result.plant_power[-1]
    residual = abs(plant - (total_load + gains)) / plant
    assert residual <= 1e-6  # steady-state bookkeeping is exact
    assert residual <= cfg.validation.energy_rel_tol


def test_temperature_envelope_holds_everywhere(mini_dcn):
    result = run_fixed_dt_scenario(mini_dcn, cfg, design_loads(mini_dcn), n_steps=20)

    for frame in (result.supply_temp, result.return_temp):
        values = frame.to_numpy()
        assert (values >= cfg.thermal.T_supply - 1e-9).all()
        assert (values <= cfg.thermal.T_ground + 1e-9).all()


def test_setpoint_step_arrives_after_transport_delay_within_one_step(mini_dcn):
    # Tenth loads -> pipe 1 carries 0.075 m3/s; V = pi/4*0.6^2*1000 = 282.74
    # m3 -> transit 3770 s = 6.28 steps. Setpoint drops at step 25: node 2
    # must be flat through step 30 (front spans steps 31/32) and carry the
    # FULL 1 K drop by step 33 — no smearing beyond plug resolution.
    loads = {j: q / 10.0 for j, q in design_loads(mini_dcn).items()}
    setpoint = np.full(55, cfg.thermal.T_supply)
    setpoint[25:] = cfg.thermal.T_supply - 1.0

    result = run_fixed_dt_scenario(
        mini_dcn, cfg, loads, n_steps=55, supply_setpoint=setpoint
    )

    t2 = result.supply_temp["2"].to_numpy()
    steady = t2[24]
    assert np.allclose(t2[26:31], steady, atol=1e-9), "front arrived too early"
    assert t2[33] < steady - 0.95, "front arrived too late or too smeared"

    # multi-pipe path: consumer 4 sits behind three pipes (transits 6.28 +
    # 6.55 + 8.38 = 21.2 steps); node-method quantization moves the front's
    # leading edge up to ONE step per node hop in either direction (a
    # partial front joins that step's mix), giving a 44.2..48.2
    # arrival window: flat through step 44, full drop by step 49.
    t4 = result.supply_temp["4"].to_numpy()
    steady4 = t4[24]
    assert np.allclose(t4[26:45], steady4, atol=1e-9), "front arrived too early"
    assert t4[49] < steady4 - 0.95, "front arrived too late or too smeared"
