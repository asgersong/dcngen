"""Acceptance on real Hanoi: 36 h constant-design-load run in
fixed-dT mode — plant energy balance closes within the configured tolerance
and the temperature envelope holds. Skipped without the dump.
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.scenario import run_fixed_dt_scenario
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.poc.network

pytestmark = pytest.mark.skipif(
    not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
)


@pytest.fixture(scope="module")
def result():
    net = load_ditec_network(HANOI, scenario_id=cfg.poc.static_scenario_id)
    dcn = build_dcn(net, cfg)
    loads = {j: c.design_load for j, c in dcn.consumers.items()}
    # 36 h: the slowest supply spur runs ~0.4 m/s over multi-km pipes, so the
    # loop needs well over 24 h to flush its initial fill
    return dcn, run_fixed_dt_scenario(dcn, cfg, loads, n_steps=216)


def test_hanoi_energy_balance_closes_at_horizon_end(result):
    dcn, res = result
    total_load = sum(c.design_load for c in dcn.consumers.values())

    final_gain = res.pipe_heat_gain.iloc[-1].sum()
    residual = abs(res.plant_power[-1] - (total_load + final_gain)) / res.plant_power[-1]
    assert residual <= cfg.validation.energy_rel_tol


def test_hanoi_temperature_envelope_holds(result):
    _, res = result
    for frame in (res.supply_temp, res.return_temp):
        values = frame.to_numpy()
        assert (values >= cfg.thermal.T_supply - 1e-9).all()
        assert (values <= cfg.thermal.T_ground + 1e-9).all()


def test_hanoi_delivered_loads_match_commanded(result):
    dcn, res = result
    commanded = np.array([dcn.consumers[j].design_load for j in res.delivered_load])
    np.testing.assert_allclose(res.delivered_load.iloc[-1].to_numpy(), commanded, rtol=1e-9)
