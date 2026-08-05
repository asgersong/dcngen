"""Seam B: generated time-varying loads drive the eps-NTU
scenario runner end-to-end — 24 h at 10-minute resolution on the mini net.

Loads are sampled at start-of-step times ``k * dt``; per-consumer entries in
``loads`` may be scalars (unchanged behaviour) or ``(n_steps,)``
arrays. The invariant assertions: the
validation-gate convergence budget, integrated energy closure, temperature
envelope, closed-loop mass balance at the plant, and delivered-load
tracking; plus same-seed end-to-end reproducibility.
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario, run_fixed_dt_scenario
from dcngen.thermal.ets import design_ua
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()

ARCHETYPES = {"2": "office", "3": "residential", "4": "hotel"}
N_STEPS = 144  # 24 h at dt = 600 s


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def generated_day(dcn, seed: int, n_steps: int = N_STEPS):
    times = np.arange(n_steps) * cfg.solver.dt
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    loads = generate_loads(
        ARCHETYPES, design, times, cfg.loads, np.random.default_rng(seed)
    )
    ua = {j: design_ua(design[j], cfg) for j in design}
    return loads, ua


@pytest.fixture(scope="module")
def day_run(mini_dcn):
    loads, ua = generated_day(mini_dcn, seed=11)
    return loads, run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=N_STEPS, ua=ua)


def test_day_stays_within_convergence_budget(day_run):
    _, result = day_run
    assert (~result.converged).mean() <= cfg.validation.max_unconverged_frac
    assert not result.unmet.to_numpy().any()  # normal operation: no unmet load


def test_day_energy_closure_integrated(day_run):
    """Integrated over the day, plant power matches delivered loads plus
    pipe gains within the gate tolerance (transport delay and plug storage
    shift energy between steps but must not create or destroy it)."""
    _, result = day_run
    plant = result.plant_power.sum()
    delivered = result.delivered_load.to_numpy().sum()
    gains = result.pipe_heat_gain.to_numpy().sum()
    residual = abs(plant - (delivered + gains)) / plant
    assert residual <= cfg.validation.energy_rel_tol


def test_day_temperature_envelope(day_run):
    _, result = day_run
    for frame in (result.supply_temp, result.return_temp):
        values = frame.to_numpy()
        assert (values >= cfg.thermal.T_supply - 1e-9).all()
        assert (values <= cfg.thermal.T_ground + 1e-9).all()


def test_day_mass_balance_and_load_tracking(day_run):
    loads, result = day_run
    # closed loop, no fault: everything the pump circulates crosses the
    # ETSs — at every step, from the recorded plant_flow channel
    pump = result.plant_flow
    consumers = result.consumer_flow.to_numpy().sum(axis=1)
    assert (np.abs(pump - consumers) / pump <= cfg.validation.mass_balance_rel_tol).all()

    # on converged, met steps the valve delivered the demanded load
    demanded = np.column_stack([loads[j] for j in result.delivered_load.columns])
    delivered = result.delivered_load.to_numpy()
    ok = result.converged[:, None] & ~result.unmet.to_numpy()
    np.testing.assert_allclose(delivered[ok], demanded[ok], rtol=0.01)


def test_same_seed_reproduces_the_scenario_end_to_end(tmp_path):
    """Solver-tolerance reproducibility: loads are bitwise-identical
    from the seed (seam A tests); solved fields reproduce within solver
    tolerance. Bitwise equality of solved fields is unattainable — WNTR's
    aml layer iterates object sets in memory-address order, so identical
    fresh models drift at the Newton stopping tolerance (~1e-10 m pressure,
    probe 2026-07-20), in and across processes, PYTHONHASHSEED regardless.
    Asserted at 1e-6 (three decades above observed drift, far below
    physical significance); one fresh DCN per run, as the driver must."""
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path / "mini_net")

    def run():
        dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg)
        loads, ua = generated_day(dcn, seed=3, n_steps=36)
        return run_eps_ntu_scenario(dcn, cfg, loads, n_steps=36, ua=ua)

    a, b = run(), run()
    np.testing.assert_allclose(
        a.supply_temp.to_numpy(), b.supply_temp.to_numpy(), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        a.consumer_flow.to_numpy(), b.consumer_flow.to_numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(a.plant_power, b.plant_power, rtol=1e-6)


def test_fixed_dt_flows_track_a_load_step_change(mini_dcn):
    """The fixed-dT fast path re-solves hydraulics when loads move: flows
    follow the hinge equation per step (halved load -> halved flow) and the
    ETS split stays exactly at design."""
    design = {j: c.design_load for j, c in mini_dcn.consumers.items()}
    n = 6
    loads = {j: np.full(n, q) for j, q in design.items()}
    for j in loads:
        loads[j][3:] = 0.5 * design[j]

    result = run_fixed_dt_scenario(mini_dcn, cfg, loads, n_steps=n)

    flows = result.consumer_flow.to_numpy()
    np.testing.assert_allclose(flows[3:], 0.5 * flows[:3], rtol=1e-12)
    demanded = np.column_stack([loads[j] for j in result.delivered_load.columns])
    np.testing.assert_allclose(result.delivered_load.to_numpy(), demanded, rtol=1e-12)
    ets_split = result.ets_return.to_numpy() - result.supply_temp[
        result.ets_return.columns
    ].to_numpy()
    np.testing.assert_allclose(ets_split, cfg.thermal.dT_design, atol=1e-9)


def test_load_arrays_must_match_n_steps(mini_dcn):
    loads, ua = generated_day(mini_dcn, seed=0, n_steps=10)
    with pytest.raises(ValueError, match="n_steps"):
        run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=20, ua=ua)
