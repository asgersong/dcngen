"""Injector seam: abrupt junction leak on either network
side — severity as target leak-flow fraction of network design flow,
orifice area back-computed from nominal local pressure, onset sampled from
the configured middle window of the horizon, snapped to a step start.
"""

import math

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.injector import inject_abrupt_leak
from dcngen.faults.taxonomy import FaultKind
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()

N_STEPS = 36
TIMES = np.arange(N_STEPS) * cfg.solver.dt
HORIZON = N_STEPS * cfg.solver.dt


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def test_sampled_leak_is_deterministic_and_inside_the_config_windows(mini_dcn):
    def draw(seed):
        return inject_abrupt_leak(
            mini_dcn, cfg, TIMES, np.random.default_rng(seed),
            junction="3", side="return",
        )

    a, b = draw(5), draw(5)
    assert a.label.severity == b.label.severity  # rng-drawn: bitwise
    assert a.label.onset == b.label.onset
    # area derives from a solved nominal pressure -> solver tolerance
    assert a.area == pytest.approx(b.area, rel=1e-9)

    assert cfg.faults.leak_severity_min <= a.label.severity <= cfg.faults.leak_severity_max
    assert cfg.faults.onset_window_start * HORIZON <= a.label.onset
    assert a.label.onset <= cfg.faults.onset_window_end * HORIZON
    assert a.label.onset in TIMES  # snapped to a step start
    assert a.label.offset == HORIZON  # abrupt leak persists to horizon end
    assert draw(6).label.severity != a.label.severity


def test_label_records_kind_location_side_and_aligned_mask(mini_dcn):
    inj = inject_abrupt_leak(
        mini_dcn, cfg, TIMES, np.random.default_rng(5), junction="3", side="return"
    )
    label = inj.label
    assert label.kind is FaultKind.LEAK
    assert (label.location, label.element, label.side) == ("3", "node", "return")
    np.testing.assert_array_equal(label.mask, TIMES >= label.onset)
    assert inj.node == mini_dcn.pairing["3"][1]  # return-side WNTR node


def test_orifice_area_matches_commanded_flow_at_nominal_pressure(mini_dcn):
    """Contract pin: at the healthy design operating point's local pressure,
    the orifice equation Q = Cd * A * sqrt(2 g h) returns the commanded
    target flow (severity x network design flow). g = 9.81 per WNTR's leak
    model; independent healthy solve supplies h."""
    severity = 0.05
    inj = inject_abrupt_leak(
        mini_dcn, cfg, TIMES, np.random.default_rng(0),
        junction="2", side="supply", severity=severity,
    )
    target = severity * mini_dcn.pump_design[0]
    assert inj.target_flow == pytest.approx(target)

    h_nominal = solve_steady(mini_dcn).pressure[mini_dcn.pairing["2"][0]]
    q_orifice = inj.discharge_coeff * inj.area * math.sqrt(2.0 * 9.81 * h_nominal)
    assert q_orifice == pytest.approx(target, rel=1e-9)


def test_solve_steady_with_leak_reports_flow_and_makeup_and_restores(mini_dcn):
    """Leak areas enter the steady solve: realized leak flow is reported,
    the plant make-up (pump minus return stub) supplies exactly the leaked
    water, and the model is left leak-free for the next solve."""
    inj = inject_abrupt_leak(
        mini_dcn, cfg, TIMES, np.random.default_rng(0),
        junction="3", side="return", severity=0.05,
    )
    design = {j: c.design_flow for j, c in mini_dcn.consumers.items()}

    leaked = solve_steady(
        mini_dcn, consumer_flows=design, leak_areas={inj.node: inj.area}
    )
    q_leak = leaked.leak_flow[inj.node]
    assert q_leak > 0.0
    # realized floats with actual pressure; at 5% severity it stays close
    # to the commanded target computed at nominal pressure
    assert q_leak == pytest.approx(inj.target_flow, rel=0.05)
    makeup = leaked.flow[mini_dcn.plant.pump] - leaked.flow[mini_dcn.plant.stub]
    assert makeup == pytest.approx(q_leak, rel=1e-6)

    healthy = solve_steady(mini_dcn, consumer_flows=design)
    assert healthy.leak_flow.empty  # no silent leak left behind
    makeup_healthy = healthy.flow[mini_dcn.plant.pump] - healthy.flow[mini_dcn.plant.stub]
    assert abs(makeup_healthy) <= 1e-9  # closed loop again


def test_unknown_side_or_junction_rejected(mini_dcn):
    with pytest.raises(ValueError):
        inject_abrupt_leak(
            mini_dcn, cfg, TIMES, np.random.default_rng(0), junction="2", side="up"
        )
    with pytest.raises(KeyError):
        inject_abrupt_leak(
            mini_dcn, cfg, TIMES, np.random.default_rng(0), junction="99", side="supply"
        )


def test_row_specified_leak_needs_no_rng(mini_dcn):
    # severity and onset from the plan row; rng=None
    onset = float(TIMES[20])
    inj = inject_abrupt_leak(
        mini_dcn, cfg, TIMES, None, junction="3", side="supply",
        severity=0.05, onset=onset,
    )
    assert inj.label.severity == 0.05
    assert inj.label.onset == onset
    np.testing.assert_array_equal(inj.label.mask, TIMES >= onset)

    with pytest.raises(ValueError, match="rng"):
        inject_abrupt_leak(
            mini_dcn, cfg, TIMES, None, junction="3", side="supply", severity=0.05,
        )
