"""DCN-ifier seam: topology + statics in -> DCN
network object out; closed loop solves under WNTRSimulator PDD.

Mirror conventions under test: junction j -> j_s / j_r; pipe p ->
p_s (same direction) and p_r (reversed direction, so that in normal operation
the same water gives the SAME signed flow on both sides); consumer j -> FCV
``ets_j`` from j_s to j_r; the DiTEC reservoir node mirrors into the supply /
return headers, with Plant triple reservoir -> pump -> supply header and
return header -> stub -> reservoir.
"""

import pytest

from dcngen.config import load_config
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def mini_dcn(mini_ditec_folder, cfg):
    net = load_ditec_network(mini_ditec_folder, scenario_id=0)
    return build_dcn(net, cfg)


def test_mirror_doubles_every_junction_and_pipe(mini_dcn):
    wn = mini_dcn.wn

    for j in ("2", "3", "4"):
        assert f"{j}_s" in wn.junction_name_list
        assert f"{j}_r" in wn.junction_name_list
    # supply pipe keeps the original direction, return pipe reverses it
    p2s = wn.get_link("2_s")  # original pipe 2: 2 -> 3
    assert (p2s.start_node_name, p2s.end_node_name) == ("2_s", "3_s")
    p2r = wn.get_link("2_r")
    assert (p2r.start_node_name, p2r.end_node_name) == ("3_r", "2_r")
    # both sides inherit the draw's statics (pipe 2: D=0.5, L=800, C=120)
    for name in ("2_s", "2_r"):
        pipe = wn.get_link(name)
        assert (pipe.diameter, pipe.length, pipe.roughness) == (0.5, 800.0, 120.0)


def test_pairing_metadata_complete_and_invertible(mini_dcn):
    assert mini_dcn.pairing == {
        "2": ("2_s", "2_r"),
        "3": ("3_s", "3_r"),
        "4": ("4_s", "4_r"),
    }
    assert mini_dcn.pipe_pairing == {
        "1": ("1_s", "1_r"),
        "2": ("2_s", "2_r"),
        "3": ("3_s", "3_r"),
    }
    # invertible: no mirrored element belongs to two originals
    mirrored = [n for pair in mini_dcn.pairing.values() for n in pair]
    assert len(mirrored) == len(set(mirrored))
    mirrored_pipes = [p for pair in mini_dcn.pipe_pairing.values() for p in pair]
    assert len(mirrored_pipes) == len(set(mirrored_pipes))


def test_plant_triple_wired_through_the_headers(mini_dcn):
    wn, plant = mini_dcn.wn, mini_dcn.plant

    assert plant.supply_header == "1_s"
    assert plant.return_header == "1_r"
    pump = wn.get_link(plant.pump)
    assert (pump.start_node_name, pump.end_node_name) == (plant.reservoir, "1_s")
    stub = wn.get_link(plant.stub)
    assert (stub.start_node_name, stub.end_node_name) == ("1_r", plant.reservoir)
    assert plant.reservoir in wn.reservoir_name_list


def test_every_positive_demand_junction_becomes_an_ets_crossover(mini_dcn):
    assert set(mini_dcn.consumers) == {"2", "3", "4"}
    ets = mini_dcn.wn.get_link(mini_dcn.consumers["2"].ets_link)
    assert (ets.start_node_name, ets.end_node_name) == ("2_s", "2_r")
    assert ets.valve_type == "FCV"


def test_design_flows_follow_flow_equivalence(mini_dcn):
    # design volumetric flow = DiTEC base demand of the pinned draw
    # (mini draw time-means: 0.35 / 0.25 / 0.15 m3/s), and design load =
    # rho * D * cp * dT_design — hand-computed literals, not recomputed.
    flows = {j: c.design_flow for j, c in mini_dcn.consumers.items()}
    assert flows == pytest.approx({"2": 0.35, "3": 0.25, "4": 0.15})
    assert mini_dcn.consumers["2"].design_load == pytest.approx(10_250_572.15)
    assert mini_dcn.consumers["3"].design_load == pytest.approx(7_321_837.25)


def test_closed_loop_solves_with_design_flows(mini_dcn, cfg):
    from dcngen.hydraulics.wntr_model import solve_steady

    state = solve_steady(mini_dcn)
    tol = cfg.validation.mass_balance_rel_tol

    # every consumer FCV delivers its set flow
    for c in mini_dcn.consumers.values():
        assert state.flow[c.ets_link] == pytest.approx(c.design_flow, rel=tol)

    # closed-loop mass balance: pump flow == sum of consumer flows
    total = sum(c.design_flow for c in mini_dcn.consumers.values())
    assert abs(state.flow[mini_dcn.plant.pump] - total) / total <= tol

    # healthy loop: make-up flow (pump minus return stub) is zero
    makeup = state.flow[mini_dcn.plant.pump] - state.flow[mini_dcn.plant.stub]
    assert abs(makeup) / total <= tol

    # no negative pressure at any junction
    for node in mini_dcn.wn.junction_name_list:
        assert state.pressure[node] >= 0.0, node

    # mirror sign convention: the same water gives the same signed flow on
    # the reversed return pipe as on its supply twin
    for p_s, p_r in mini_dcn.pipe_pairing.values():
        assert state.flow[p_r] == pytest.approx(state.flow[p_s], rel=1e-6)


def test_plant_sized_exactly_from_probe_solve(mini_dcn, cfg):
    from dcngen.hydraulics.wntr_model import solve_steady
    from dcngen.topology.dcnifier import PROBE_PUMP_HEAD

    q0, h0 = mini_dcn.pump_design
    assert q0 == pytest.approx(0.75)  # sum of design flows
    assert 0.0 < h0 < PROBE_PUMP_HEAD  # sized, not the big-M probe

    state = solve_steady(mini_dcn)
    # exact sizing: the tightest ETS keeps precisely its minimum dP
    min_drop = min(
        state.head[f"{j}_s"] - state.head[f"{j}_r"] for j in mini_dcn.consumers
    )
    assert min_drop == pytest.approx(cfg.ets.min_dp, abs=1e-3)
    # anchoring: at design flow the supply header sits at the DiTEC source head
    assert state.head[mini_dcn.plant.supply_header] == pytest.approx(
        mini_dcn.ditec_reservoir_head, abs=1e-3
    )


def test_edge_classification_is_total_and_disjoint(mini_dcn):
    kinds = mini_dcn.edge_kind
    assert set(kinds) == set(mini_dcn.wn.link_name_list)
    by_kind = {}
    for name, kind in kinds.items():
        by_kind.setdefault(kind, []).append(name)
    assert sorted(by_kind["supply_pipe"]) == ["1_s", "2_s", "3_s"]
    assert sorted(by_kind["return_pipe"]) == ["1_r", "2_r", "3_r"]
    assert sorted(by_kind["ets"]) == ["ets_2", "ets_3", "ets_4"]
    assert by_kind["plant_pump"] == [mini_dcn.plant.pump]
    assert by_kind["plant_stub"] == [mini_dcn.plant.stub]


def test_default_anchoring_is_the_ditec_head(mini_dcn):
    # with headroom available, the supply header sits at the draw's
    # validated source head
    assert mini_dcn.anchoring == "ditec"


def test_rail_fallback_anchor_when_ditec_headroom_is_insufficient(tmp_path):
    # draws whose pump head exceeds
    # the DiTEC source head's headroom anchor at the minimal
    # rail-satisfying head instead of failing the build (~3 % of Hanoi
    # draws hit this). A huge ets.min_dp forces the same condition on the
    # mini net.
    from dcngen.hydraulics.wntr_model import solve_steady
    from tests.conftest import write_ditec_folder, write_mutated_default

    def raise_min_dp(d):
        d["ets"]["min_dp"] = 60.0

    cfg2 = load_config(write_mutated_default(tmp_path, raise_min_dp))
    folder = write_ditec_folder(tmp_path / "mini_net")
    dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg2)

    assert dcn.anchoring == "rail_fallback"
    floor = max(
        dcn.wn.get_node(n).elevation for n in dcn.wn.junction_name_list
    ) + cfg2.solver.required_pressure
    suction_rail = (
        dcn.wn.get_node(dcn.plant.supply_header).elevation
        + cfg2.validation.plant_suction_min
    )
    head = dcn.wn.get_node(dcn.plant.reservoir).base_head
    assert head == pytest.approx(max(floor, suction_rail))
    # the fallback-anchored loop still solves at design flows with every
    # JUNCTION pressurised (the reservoir reads 0 by definition)
    design = {j: c.design_flow for j, c in dcn.consumers.items()}
    state = solve_steady(dcn, consumer_flows=design)
    assert float(state.pressure[dcn.wn.junction_name_list].min()) > 0.0
