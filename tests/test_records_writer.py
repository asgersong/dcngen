"""Writer seam: a completed scenario as a folder of
Parquet tables + card.json — logical-paired long-format tables per the
seam decision (2026-07-20), every table carrying scenario_id.

The fixture runs one small leaked eps-NTU scenario end-to-end (generated
loads, injected leak, completed label, drawn sensor mask) and writes it
once; tests read tables back with pyarrow and check the schema contract.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from dcngen.config import load_config
from dcngen.records.writer import write_scenario
from tests.conftest import (
    MINI_RECORD_ARCHETYPES as ARCHETYPES,
)
from tests.conftest import (
    MINI_RECORD_N_STEPS as N_STEPS,
)
from tests.conftest import (
    MINI_RECORD_SCENARIO_ID as SCENARIO_ID,
)
from tests.conftest import (
    MINI_RECORD_SEED as SEED,
)
from tests.conftest import (
    build_mini_leak_record,
)

cfg = load_config()


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    dcn, record = build_mini_leak_record(tmp_path_factory.mktemp("nets"), cfg)
    out = write_scenario(tmp_path_factory.mktemp("records") / "s7", dcn, cfg, record)
    return dcn, record, out


def table(out, name):
    return pq.read_table(out / f"{name}.parquet").to_pandas()


def test_node_dynamics_has_the_schema_contract_channels(written):
    dcn, record, out = written
    nodes = table(out, "node_dynamics")

    junctions = sorted(dcn.pairing)
    assert len(nodes) == N_STEPS * len(junctions)
    for channel in (
        "scenario_id", "time_id", "junction", "supply_temp", "return_temp",
        "pressure_s", "pressure_r", "dT", "cooling_load", "consumer_flow",
        "ets_return", "delivered_load", "unmet",
    ):
        assert channel in nodes.columns, channel
    assert (nodes["scenario_id"] == SCENARIO_ID).all()

    # spot-check one consumer row against the result frames
    row = nodes[(nodes["time_id"] == N_STEPS - 1) & (nodes["junction"] == "3")].iloc[0]
    r = record.result
    assert row["supply_temp"] == r.supply_temp.iloc[-1]["3"]
    assert row["dT"] == pytest.approx(
        r.return_temp.iloc[-1]["3"] - r.supply_temp.iloc[-1]["3"]
    )
    assert row["cooling_load"] == record.loads["3"][-1]
    assert row["consumer_flow"] == r.consumer_flow.iloc[-1]["3"]
    assert row["pressure_s"] == r.pressure_s.iloc[-1]["3"]

    # non-consumer junctions exist with NaN consumer channels (mini net has
    # none, so every junction is a consumer here — assert the invariant
    # from the data instead: consumer rows are fully populated)
    assert nodes["cooling_load"].notna().all()


def test_edge_dynamics_pairs_supply_and_return_channels(written):
    dcn, record, out = written
    edges = table(out, "edge_dynamics")

    pipes = sorted(dcn.pipe_pairing)
    assert len(edges) == N_STEPS * len(pipes)
    for channel in (
        "scenario_id", "time_id", "pipe", "flow_s", "flow_r", "velocity_s",
        "velocity_r", "headloss_s", "headloss_r", "heat_gain_s", "heat_gain_r",
    ):
        assert channel in edges.columns, channel
    assert (edges["scenario_id"] == SCENARIO_ID).all()

    r = record.result
    pipe = pipes[0]
    p_s, p_r = dcn.pipe_pairing[pipe]
    row = edges[(edges["time_id"] == N_STEPS - 1) & (edges["pipe"] == pipe)].iloc[0]
    assert row["flow_s"] == r.pipe_flow.iloc[-1][p_s]
    assert row["flow_r"] == r.pipe_flow.iloc[-1][p_r]
    assert row["velocity_s"] == r.pipe_velocity.iloc[-1][p_s]
    assert row["headloss_r"] == r.pipe_headloss.iloc[-1][p_r]
    assert row["heat_gain_s"] == r.pipe_heat_gain.iloc[-1][p_s]

    # signed-flow contract: a healthy loop shows equal signs on both sides
    assert np.sign(row["flow_s"]) == np.sign(row["flow_r"])


def test_static_tables_carry_geometry_and_ets_design(written):
    dcn, record, out = written

    nodes = table(out, "node_static").set_index("junction")
    assert (nodes["scenario_id"] == SCENARIO_ID).all()
    for j, consumer in dcn.consumers.items():
        assert bool(nodes.loc[j, "is_consumer"])
        assert nodes.loc[j, "archetype"] == ARCHETYPES[j]
        assert nodes.loc[j, "design_load"] == consumer.design_load
        assert nodes.loc[j, "design_flow"] == consumer.design_flow
        assert nodes.loc[j, "ua"] == record.ua[j]
        assert nodes.loc[j, "dT_design"] == cfg.thermal.dT_design
        s_node = dcn.pairing[j][0]
        assert nodes.loc[j, "elevation"] == dcn.wn.get_node(s_node).elevation

    edges = table(out, "edge_static").set_index("pipe")
    assert (edges["scenario_id"] == SCENARIO_ID).all()
    for pipe, (p_s, _) in dcn.pipe_pairing.items():
        link = dcn.wn.get_link(p_s)
        assert edges.loc[pipe, "length"] == link.length
        assert edges.loc[pipe, "diameter"] == link.diameter
        assert edges.loc[pipe, "roughness"] == link.roughness
        # the per-pipe derived U' and its product-range flag,
        # supply-twin read like the elevation
        assert edges.loc[pipe, "U"] == record.heat_gain.U[p_s]
        assert (
            edges.loc[pipe, "insulation_extrapolated"]
            == record.heat_gain.extrapolated[p_s]
        )


def test_plant_labels_and_sensor_tables(written):
    dcn, record, out = written
    r = record.result

    plant = table(out, "plant_dynamics")
    assert len(plant) == N_STEPS
    assert (plant["scenario_id"] == SCENARIO_ID).all()
    np.testing.assert_array_equal(plant["plant_power"], r.plant_power)
    np.testing.assert_array_equal(plant["plant_flow"], r.plant_flow)
    np.testing.assert_array_equal(plant["make_up_flow"], r.make_up_flow)
    np.testing.assert_array_equal(plant["leak_flow"], r.leak_flow)
    np.testing.assert_array_equal(plant["converged"], r.converged)
    np.testing.assert_array_equal(plant["iterations"], r.iterations)

    labels = table(out, "labels")
    assert len(labels) == N_STEPS
    assert (labels["scenario_id"] == SCENARIO_ID).all()
    np.testing.assert_array_equal(labels["fault_active"], record.label.mask)
    np.testing.assert_array_equal(
        labels["realized_leak_flow"], record.label.realized["leak_flow"]
    )

    sensors = table(out, "sensor_mask").set_index("junction")
    assert (sensors["scenario_id"] == SCENARIO_ID).all()
    assert set(sensors.index) == set(dcn.pairing)
    for j in dcn.pairing:
        assert sensors.loc[j, "heat_metered"] == (
            j in record.sensor_mask.heat_metered
        )
        assert sensors.loc[j, "pressure_sensed"] == (
            j in record.sensor_mask.pressure_sensed
        )
    # ground truth fully stored regardless of the mask: dynamics cover every
    # junction, not just sensed ones
    nodes = table(out, "node_dynamics")
    assert set(nodes["junction"]) == set(dcn.pairing)


def test_metadata_card_is_complete(written):
    dcn, record, out = written
    card = json.loads((out / "card.json").read_text())

    assert card["network"] == dcn.name
    assert card["scenario_id"] == SCENARIO_ID
    assert card["seed"] == SEED
    assert card["static_draw_id"] == 0
    assert card["dt"] == cfg.solver.dt
    assert card["n_steps"] == N_STEPS

    # topology: original DiTEC adj_list reconstructed from the supply mirror
    assert sorted(card["adj_list"]) == sorted(
        [["2", "1", "1"], ["2", "3", "2"], ["3", "4", "3"]]
    )
    assert card["pairing"]["3"] == ["3_s", "3_r"]
    assert card["pipe_pairing"]["2"] == ["2_s", "2_r"]

    # per-scenario flow directions: majority sign per logical pipe per side
    directions = card["flow_directions"]
    for pipe in dcn.pipe_pairing:
        assert set(directions[pipe]) == {"s", "r"}
        assert directions[pipe]["s"] in (-1, 0, 1)

    conv = card["convergence"]
    r = record.result
    assert conv["unconverged_steps"] == int((~r.converged).sum())
    assert conv["max_iterations_used"] == int(r.iterations.max())
    assert conv["budget_ok"] == bool(
        (~r.converged).mean() <= cfg.validation.max_unconverged_frac
    )

    fault = card["fault"]
    assert fault["kind"] == "leak"
    assert fault["location"] == "3"
    assert fault["side"] == "return"
    assert fault["severity"] == record.label.severity
    assert fault["onset"] == record.label.onset

    assert card["sensors"]["plant_sensed"] is True

    # plant params: the record is self-describing
    plant = card["plant"]
    assert plant["site"] == "1"
    assert plant["pump_design_flow"] == dcn.pump_design[0]
    assert plant["pump_design_head"] == dcn.pump_design[1]
    assert plant["ditec_reservoir_head"] == dcn.ditec_reservoir_head
    assert plant["T_supply_design"] == cfg.thermal.T_supply
    assert plant["elements"]["pump"] == dcn.plant.pump
    # units/dtypes documented for every column of every parquet table
    for name in (
        "node_dynamics", "edge_dynamics", "node_static", "edge_static",
        "plant_dynamics", "labels", "sensor_mask",
    ):
        frame = table(out, name)
        for column in frame.columns:
            entry = card["channels"][name][column]
            assert "unit" in entry and "dtype" in entry, (name, column)


def test_undocumented_column_fails_before_any_file_is_written(
    written, tmp_path, monkeypatch
):
    """The card (and its units validation) is built before any file
    lands on disk — a schema/documentation mismatch leaves no partial
    folder behind."""
    import dcngen.records.writer as writer_module

    dcn, record, _ = written
    broken = dict(writer_module._UNITS)
    del broken["dT"]
    monkeypatch.setattr(writer_module, "_UNITS", broken)

    target = tmp_path / "partial"
    with pytest.raises(KeyError):
        write_scenario(target, dcn, cfg, record)
    assert not any(target.iterdir())


def test_writing_the_same_record_twice_is_content_equal(written, tmp_path):
    """Same completed result -> byte-identical
    records (no timestamps or other nondeterminism in tables or card)."""
    dcn, record, out = written
    again = write_scenario(tmp_path / "again", dcn, cfg, record)

    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(p.name for p in again.iterdir())
    for name in files:
        assert (out / name).read_bytes() == (again / name).read_bytes(), name
