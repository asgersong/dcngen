"""Acceptance on real Hanoi: the mirrored network solves as a
closed loop with flow-equivalence design flows. Skipped without the dump.
"""

import pytest

from dcngen.config import load_config
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.topology.dcnifier import PROBE_PUMP_HEAD, build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.poc.network

pytestmark = pytest.mark.skipif(
    not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
)


@pytest.fixture(scope="module")
def hanoi_dcn():
    net = load_ditec_network(HANOI, scenario_id=cfg.poc.static_scenario_id)
    return build_dcn(net, cfg)


@pytest.fixture(scope="module")
def state(hanoi_dcn):
    return solve_steady(hanoi_dcn)


def test_hanoi_mirror_is_complete(hanoi_dcn):
    wn = hanoi_dcn.wn
    assert len(hanoi_dcn.consumers) == 31  # every Hanoi junction draws water
    assert len(wn.junction_name_list) == 2 * 31 + 2  # mirrored + headers
    assert len(wn.pipe_name_list) == 2 * 34 + 1  # mirrored + plant stub
    assert len(wn.valve_name_list) == 31
    assert len(wn.pump_name_list) == 1


def test_hanoi_every_fcv_delivers_its_set_flow(hanoi_dcn, state):
    tol = cfg.validation.mass_balance_rel_tol
    for c in hanoi_dcn.consumers.values():
        assert state.flow[c.ets_link] == pytest.approx(c.design_flow, rel=tol)


def test_hanoi_closed_loop_mass_balance(hanoi_dcn, state):
    total = sum(c.design_flow for c in hanoi_dcn.consumers.values())
    pump = state.flow[hanoi_dcn.plant.pump]
    assert abs(pump - total) / total <= cfg.validation.mass_balance_rel_tol
    makeup = pump - state.flow[hanoi_dcn.plant.stub]
    assert abs(makeup) / total <= cfg.validation.mass_balance_rel_tol


def test_hanoi_no_negative_pressure(hanoi_dcn, state):
    worst = min(state.pressure[n] for n in hanoi_dcn.wn.junction_name_list)
    assert worst >= 0.0


def test_hanoi_pump_sized_below_probe_and_anchored(hanoi_dcn, state):
    q0, h0 = hanoi_dcn.pump_design
    assert 0.0 < h0 < PROBE_PUMP_HEAD
    assert q0 == pytest.approx(
        sum(c.design_flow for c in hanoi_dcn.consumers.values())
    )
    assert state.head[hanoi_dcn.plant.supply_header] == pytest.approx(
        hanoi_dcn.ditec_reservoir_head, abs=1e-3
    )
