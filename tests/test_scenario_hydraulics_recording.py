"""Runners record per-step hydraulic channels —
node pressures on both sides, per-pipe signed flow, velocity, headloss —
so the records writer can emit the dynamic channels without re-solving.

Conventions: pressures [m] on logical junctions (pressure_s / pressure_r);
pipe channels keyed by WNTR pipe name, flow signed against the pipe's
nominal direction, headloss = head[start] − head[end] (same orientation,
so friction makes sign(headloss) == sign(flow) on every pipe).
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.scenario import run_eps_ntu_scenario, run_fixed_dt_scenario
from dcngen.thermal.ets import design_ua
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def design_loads(dcn):
    return {j: c.design_load for j, c in dcn.consumers.items()}


@pytest.fixture(scope="module")
def runs(mini_dcn):
    loads = design_loads(mini_dcn)
    ua = {j: design_ua(q, cfg) for j, q in loads.items()}
    fixed = run_fixed_dt_scenario(mini_dcn, cfg, loads, n_steps=5)
    eps = run_eps_ntu_scenario(mini_dcn, cfg, loads, n_steps=5, ua=ua)
    return mini_dcn, fixed, eps


def test_node_pressures_recorded_per_step_on_both_sides(runs):
    dcn, fixed, eps = runs
    junctions = sorted(dcn.pairing)
    for result in (fixed, eps):
        for frame in (result.pressure_s, result.pressure_r):
            assert list(frame.columns) == junctions
            assert frame.shape == (5, len(junctions))
        # last recorded row is the stored final operating point
        for j in junctions:
            s_node, r_node = dcn.pairing[j]
            assert result.pressure_s.iloc[-1][j] == pytest.approx(
                result.hydraulics.pressure[s_node]
            )
            assert result.pressure_r.iloc[-1][j] == pytest.approx(
                result.hydraulics.pressure[r_node]
            )


def test_pipe_channels_recorded_per_step(runs):
    dcn, fixed, eps = runs
    pipes = sorted(dcn.wn.pipe_name_list)
    for result in (fixed, eps):
        for frame in (result.pipe_flow, result.pipe_velocity, result.pipe_headloss):
            assert sorted(frame.columns) == pipes
            assert frame.shape[0] == 5
        assert (result.pipe_velocity.to_numpy() >= 0.0).all()
        np.testing.assert_allclose(
            result.pipe_flow.iloc[-1][pipes], result.hydraulics.flow[pipes]
        )


def test_friction_headloss_sign_matches_flow_sign(runs):
    _, fixed, eps = runs
    for result in (fixed, eps):
        flow = result.pipe_flow.to_numpy()
        headloss = result.pipe_headloss.to_numpy()
        moving = np.abs(flow) > 1e-12
        assert (np.sign(headloss[moving]) == np.sign(flow[moving])).all()
