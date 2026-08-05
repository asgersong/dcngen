"""Records + graph view of plan-driven scenarios: everything
the sampler and the low-dT faults introduced survives the round trip —
healthy-UA statics, temperature program, plan-row echo + environment in
the card, bypass localisation on the ETS store, and the #16 deviation
(ground truth only, no materialised sensor masks).
"""

import dataclasses
import json

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.plan_runner import run_plan_row
from dcngen.orchestrate.sampler import FaultSpec, build_plan
from dcngen.records.pyg_dataset import load_scenario
from dcngen.records.writer import plan_scenario_record, write_scenario
from dcngen.topology.ditec_loader import load_ditec_network
from tests.conftest import shrink_sampler_tiers as shrink_tiers
from tests.conftest import write_ditec_folder, write_mutated_default


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    path = write_mutated_default(tmp_path_factory.mktemp("cfg"), shrink_tiers)
    return load_config(path)


@pytest.fixture(scope="module")
def mini_net(tmp_path_factory):
    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return load_ditec_network(folder, scenario_id=0)


@pytest.fixture(scope="module")
def nets(mini_net, tmp_path_factory):
    """Static draws 0 and 1 of the mini network, keyed by draw id."""
    other = write_ditec_folder(tmp_path_factory.mktemp("nets2") / "mini_net")
    return {0: mini_net, 1: load_ditec_network(other, scenario_id=1)}


@pytest.fixture(scope="module")
def plan(cfg, mini_net):
    return build_plan(cfg, mini_net)


def _row(plan, cls, tier="bulk"):
    return next(
        r for r in plan.rows if r.tier == tier and r.scenario_class == cls
    )


@pytest.fixture(scope="module")
def bypass_written(cfg, nets, plan, tmp_path_factory):
    """A whole-horizon bypass plan row, run, recorded, and written.

    Picked from the manifest as-is (any tier): the record bridge rejects
    mutated rows, and test_sampler pins that both temporal variants occur, so
    a whole-horizon row always exists.
    """
    row = next(
        r for r in plan.rows
        if r.scenario_class == "bypass" and r.fault.onset_step is None
    )
    scenario = run_plan_row(cfg, row, net=nets[row.static_draw_id])
    record = plan_scenario_record(scenario, plan)
    out = write_scenario(
        tmp_path_factory.mktemp("records") / f"uid{row.uid}",
        scenario.dcn, scenario.config, record,
    )
    return scenario, record, out


@pytest.fixture(scope="module")
def normal_written(cfg, nets, plan, tmp_path_factory):
    row = _row(plan, "normal")
    scenario = run_plan_row(cfg, row, net=nets[row.static_draw_id])
    record = plan_scenario_record(scenario, plan)
    out = write_scenario(
        tmp_path_factory.mktemp("records") / f"uid{row.uid}",
        scenario.dcn, scenario.config, record,
    )
    return scenario, record, out


def test_record_identity_comes_from_the_row(bypass_written):
    scenario, record, _ = bypass_written
    assert record.scenario_id == scenario.row.uid
    assert record.static_draw_id == scenario.row.static_draw_id
    assert record.archetypes == scenario.row.archetypes
    assert record.sensor_mask is None  # ground truth only (#16 deviation)


def test_node_static_carries_the_healthy_ua_factor(bypass_written):
    import pyarrow.parquet as pq

    scenario, _, out = bypass_written
    statics = pq.read_table(out / "node_static.parquet").to_pandas()
    factors = statics.set_index("junction")["healthy_ua_factor"]
    for j, expected in scenario.row.healthy_ua_factor.items():
        assert factors[j] == expected


def test_card_pins_plan_environment_and_temperature_program(bypass_written):
    scenario, _, out = bypass_written
    card = json.loads((out / "card.json").read_text())
    row = scenario.row

    plan_block = card["plan"]
    assert plan_block["master_seed"] == scenario.config.seed
    assert plan_block["bit_generator"] == "PCG64"
    assert plan_block["numpy_version"] == np.__version__
    assert "networkx_version" in plan_block
    assert plan_block["allocations"]  # stratified-allocation records
    # the allocation strata are interpretable from the card alone: the
    # complete meta rides along (communities, consumers, reference draw)
    assert plan_block["communities"] and plan_block["consumers"]
    assert plan_block["reference_draw_id"] == 0
    echo = plan_block["row"]
    assert (echo["uid"], echo["tier"], echo["scenario_class"]) == (
        row.uid, row.tier, "bypass",
    )
    assert echo["T_supply"] == row.T_supply

    # full config echo of the config the scenario actually ran under
    assert card["config"]["thermal"]["T_supply"] == row.T_supply
    assert card["config"]["thermal"]["T_ground"] == row.T_ground
    assert card["config"]["solver"]["dt"] == scenario.config.solver.dt

    fault = card["fault"]
    assert fault["kind"] == "bypass"
    assert fault["severity"] == row.fault.severity
    assert fault["whole_horizon"] is True
    assert fault["onset_step"] == 0

    assert card["sensors"] == {"materialised": False}
    assert not (out / "sensor_mask.parquet").exists()

    # the storage term that closes the energy balance rides along
    storage = card["storage"]
    assert storage["stored_energy_start_J"] == scenario.result.stored_energy_start
    assert storage["stored_energy_end_J"] == scenario.result.stored_energy_end
    assert storage["stored_energy_start_J"] > 0.0


def test_bypass_label_locates_on_the_ets_store(bypass_written):
    scenario, _, out = bypass_written
    graph = load_scenario(out)
    j = scenario.row.fault.junction

    located = graph["supply", "ets", "return"].fault_location_mask
    consumers = graph["supply", "ets", "return"].junction
    assert bool(located[consumers.index(j)])
    assert int(located.sum()) == 1
    # a node-store fault this is not: no junction is marked there
    assert not graph["supply"].fault_location_mask.any()
    assert bool(graph.fault_active.all())  # whole-horizon


def test_loaded_graph_exposes_the_new_statics(bypass_written):
    scenario, _, out = bypass_written
    graph = load_scenario(out)
    row = scenario.row

    assert graph.T_supply == pytest.approx(row.T_supply)
    assert graph.T_ground == pytest.approx(row.T_ground)
    assert graph.tier == row.tier
    assert graph.scenario_class == "bypass"

    channels = graph["supply"].static_channels
    assert "healthy_ua_factor" in channels
    col = channels.index("healthy_ua_factor")
    junctions = graph["supply"].junction
    for j, expected in row.healthy_ua_factor.items():
        got = float(graph["supply"].static[junctions.index(j), col])
        assert got == pytest.approx(expected, rel=1e-6)


def test_mask_free_folder_loads_with_full_observability(bypass_written):
    _, _, out = bypass_written
    graph = load_scenario(out)
    assert bool(graph["supply"].heat_metered.all())
    assert bool(graph["return"].pressure_sensed.all())


def test_normal_plan_record_carries_the_explicit_no_fault_label(normal_written):
    scenario, _, out = normal_written
    card = json.loads((out / "card.json").read_text())
    assert card["fault"]["kind"] == "none"
    assert card["fault"]["severity"] is None
    assert card["fault"]["whole_horizon"] is False

    graph = load_scenario(out)
    assert not graph.fault_active.any()
    assert graph.scenario_class == "normal"
    for store in (graph["supply"], graph["supply", "ets", "return"]):
        assert not store.fault_location_mask.any()


def test_gate_report_rides_the_card(normal_written):
    scenario, _, out = normal_written
    card = json.loads((out / "card.json").read_text())
    assert card["validation"]["passed"] == scenario.report.passed
    rules = {r["rule"] for r in card["validation"]["rules"]}
    assert "energy_closure" in rules and "pressure_rails" in rules


def test_plan_scenario_record_rejects_a_foreign_plan_row(cfg, nets, plan):
    # the record bridge must refuse a row that is not the scenario's own
    row = _row(plan, "normal")
    scenario = run_plan_row(cfg, row, net=nets[row.static_draw_id])
    other = dataclasses.replace(
        scenario, row=dataclasses.replace(scenario.row, uid=9999)
    )
    with pytest.raises(ValueError, match="uid"):
        plan_scenario_record(other, plan)


def test_fouling_and_leak_labels_round_trip(cfg, nets, plan, tmp_path_factory):
    # AC: labels for every class survive the round trip with location masks
    for cls, store_key in (("fouling", ("supply", "ets", "return")), ("leak", None)):
        row = _row(plan, cls)
        scenario = run_plan_row(cfg, row, net=nets[row.static_draw_id])
        out = write_scenario(
            tmp_path_factory.mktemp("records") / f"uid{row.uid}",
            scenario.dcn, scenario.config, plan_scenario_record(scenario, plan),
        )
        graph = load_scenario(out)
        assert graph.scenario_class == cls
        assert graph.fault["severity"] == pytest.approx(row.fault.severity)
        if cls == "fouling":
            located = graph[store_key].fault_location_mask
            ids = graph[store_key].junction
            assert bool(located[ids.index(row.fault.junction)])
            assert graph.fault["whole_horizon"] is True
        else:
            located = graph[row.fault.side].fault_location_mask
            ids = graph[row.fault.side].junction
            assert bool(located[ids.index(row.fault.junction)])
            assert graph.fault["whole_horizon"] is False
            assert graph.fault["onset_step"] == row.fault.onset_step


def test_faultspec_vocabulary_matches_the_store_routing(plan):
    # regression pin for the silent-miss found in prep: every fault kind
    # the sampler can emit must be routable to a store's location mask
    from dcngen.records.pyg_dataset import _STORE_KINDS

    routable = {k for kinds in _STORE_KINDS.values() for k in kinds}
    emitted = {r.fault.kind for r in plan.rows} - {"none"}
    assert emitted <= routable
    assert isinstance(plan.rows[0].fault, FaultSpec)
