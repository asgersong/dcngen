"""Acceptance on real Hanoi: the fouling case against a
healthy baseline under eps-NTU, per-timestep convergence recorded. Short
horizon (spin-up suffices for a differential comparison at the fouled
consumer, which sits early on the trunk). Skipped without the dump.
"""

import pytest

from dcngen.config import load_config
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.poc.network

pytestmark = pytest.mark.skipif(
    not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
)

FOULED = "3"  # early trunk consumer: short transport delay, fast spin-up


@pytest.fixture(scope="module")
def runs():
    net = load_ditec_network(HANOI, scenario_id=cfg.poc.static_scenario_id)
    dcn = build_dcn(net, cfg)
    loads = {j: c.design_load for j, c in dcn.consumers.items()}
    ua = {j: design_ua(loads[j], cfg) for j in loads}
    healthy = run_eps_ntu_scenario(dcn, cfg, loads, n_steps=12, ua=ua)
    fouled = run_eps_ntu_scenario(
        dcn, cfg, loads, n_steps=12, ua={**ua, FOULED: 0.5 * ua[FOULED]}
    )
    return dcn, healthy, fouled


def test_hanoi_converges_with_flags_recorded(runs):
    _, healthy, fouled = runs
    assert healthy.converged.all()
    # steady init: step 0 converges from the cold
    # fixed-dT warm start to the transit-warmed operating point (measured
    # 6 iterations on draw 0), after which the warm start is exact and
    # every later step needs a single pass — less total work than the old
    # cold-start profile (<= 3 every step)
    assert healthy.iterations[0] <= 8
    assert (healthy.iterations[1:] <= 2).all()
    # accept-but-record: the fouling onset step may exhaust the iteration
    # cap (flagged, never rejected); the warm start then tracks the fouled
    # fixed point. (The <= 1 % scenario budget is a gate rule enforced at
    # M6b over full horizons — here the flags themselves are under test.)
    assert (~fouled.converged).sum() <= 1
    assert fouled.converged[2:].all()
    assert len(fouled.iterations) == 12


def test_hanoi_fouling_dt_collapse_and_flow_rise(runs):
    _, healthy, fouled = runs
    h_split = healthy.ets_return.iloc[-1][FOULED] - healthy.supply_temp.iloc[-1][FOULED]
    f_split = fouled.ets_return.iloc[-1][FOULED] - fouled.supply_temp.iloc[-1][FOULED]
    h_flow = healthy.consumer_flow.iloc[-1][FOULED]
    f_flow = fouled.consumer_flow.iloc[-1][FOULED]

    assert f_split < h_split - 1.0  # low-dT syndrome, visibly
    assert f_flow > h_flow * 1.15
    assert not fouled.unmet[FOULED].iloc[-1]  # UA/2 is severe, not saturating


def test_hanoi_healthy_eps_ntu_stays_at_design_split(runs):
    _, healthy, _ = runs
    split = healthy.ets_return.iloc[-1] - healthy.supply_temp.iloc[-1][
        healthy.ets_return.columns
    ]
    # steady arrivals carry the full transit warming from step 0:
    # the local split sits AT or BELOW design — never above — with the
    # deficit growing toward the network extremities (measured up to
    # ~0.21 K at consumer 15 on draw 0); the tight fixed-dT agreement
    # check lives in the mini twin at rel 5e-3
    assert (split <= cfg.thermal.dT_design + 1e-6).all()
    assert (split > cfg.thermal.dT_design - 0.35).all()
