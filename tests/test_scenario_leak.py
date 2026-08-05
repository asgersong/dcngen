"""Seam 3: the abrupt leak through the eps-NTU runner —
paired normal vs leaked scenario on the mini net with generated loads.

Acceptance: pressure drops near the leak, plant make-up flow
matches the leak flow, realized leak flow tracks the commanded severity,
and the per-timestep mask aligns exactly with onset.
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.injector import inject_abrupt_leak
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()

N_STEPS = 36
TIMES = np.arange(N_STEPS) * cfg.solver.dt
LEAK_AT = "3"


@pytest.fixture(scope="module")
def paired_runs(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg)

    design = {j: c.design_load for j, c in dcn.consumers.items()}
    loads = generate_loads(
        {"2": "office", "3": "residential", "4": "hotel"},
        design, TIMES, cfg.loads, np.random.default_rng(11),
    )
    ua = {j: design_ua(design[j], cfg) for j in design}

    leak = inject_abrupt_leak(
        dcn, cfg, TIMES, np.random.default_rng(7),
        junction=LEAK_AT, side="supply", severity=0.05,
    )
    healthy = run_eps_ntu_scenario(dcn, cfg, loads, n_steps=N_STEPS, ua=ua)
    leaked = run_eps_ntu_scenario(
        dcn, cfg, loads, n_steps=N_STEPS, ua=ua, leak=leak
    )
    return dcn, leak, healthy, leaked


def test_leak_flow_aligns_exactly_with_the_mask(paired_runs):
    _, leak, _, leaked = paired_runs
    mask = leak.label.mask
    assert (leaked.leak_flow[mask] > 0.0).all()
    assert (leaked.leak_flow[~mask] == 0.0).all()


def test_makeup_flow_supplies_exactly_the_leak(paired_runs):
    _, leak, healthy, leaked = paired_runs
    np.testing.assert_allclose(
        leaked.make_up_flow, leaked.leak_flow, rtol=1e-6, atol=1e-12
    )
    assert (np.abs(healthy.make_up_flow) <= 1e-9).all()  # closed when clean


def test_realized_leak_tracks_commanded_severity(paired_runs):
    """Commanded-at-nominal is pinned at the solve level (test_leak_injector,
    design flows). Under the generated part-load day, local pressure sits
    ABOVE the design-point nominal (less friction loss + the one-point pump
    curve riding up its head), so realized > commanded is physical — assert
    a band plus orifice-law consistency at the actually solved pressure."""
    _, leak, _, leaked = paired_runs
    mask = leak.label.mask
    realized = leaked.leak_flow[mask].mean()
    assert 0.5 * leak.target_flow < realized < 2.0 * leak.target_flow

    p_last = leaked.hydraulics.pressure[leak.node]  # final-step operating point
    q_orifice = leak.discharge_coeff * leak.area * np.sqrt(2.0 * 9.81 * p_last)
    assert leaked.leak_flow[-1] == pytest.approx(q_orifice, rel=1e-6)


def test_pressure_drops_near_the_leak_vs_paired_normal(paired_runs):
    dcn, leak, healthy, leaked = paired_runs
    # both runs' stored hydraulics are the final step's operating point,
    # where the leak is active (onset is mid-horizon)
    assert leak.label.mask[-1]
    p_healthy = healthy.hydraulics.pressure[leak.node]
    p_leaked = leaked.hydraulics.pressure[leak.node]
    assert p_leaked < p_healthy - 0.01  # visible local drop [m]
    # "near": the extra trunk flow depresses the neighbourhood too
    for neighbour in ("2_s", "4_s"):
        assert (
            leaked.hydraulics.pressure[neighbour]
            < healthy.hydraulics.pressure[neighbour]
        )


def test_completed_label_carries_the_realized_series(paired_runs):
    _, leak, _, leaked = paired_runs
    final = leak.label.with_realized(leak_flow=leaked.leak_flow)
    np.testing.assert_array_equal(final.realized["leak_flow"], leaked.leak_flow)
    assert final.severity == 0.05
    assert (final.location, final.side) == (LEAK_AT, "supply")
