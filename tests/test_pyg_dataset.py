"""Loader seam: a written scenario folder loads back as
a typed heterogeneous graph — supply/return node types; supply-pipe,
return-pipe, ETS, and plant edge types, the plant edge closing the loop so
make-up flow is on-graph; features stacked over time; labels, masks, and
the sensor mask exposed as tensors aligned with the time axis.

The fixture runs the SAME mini-net leaked eps-NTU scenario as the writer
tests, writes it once, and loads it back through ``load_scenario`` — only
the folder path crosses the seam, so the in-memory record doubles as the
independent ground truth for every round-trip assertion.
"""

import shutil

import numpy as np
import pytest
import torch

from dcngen.config import load_config
from dcngen.records.pyg_dataset import load_scenario
from dcngen.records.writer import write_scenario
from tests.conftest import (
    MINI_RECORD_ARCHETYPES as ARCHETYPES,
)
from tests.conftest import (
    MINI_RECORD_N_STEPS as N_STEPS,
)
from tests.conftest import (
    build_mini_leak_record,
)

cfg = load_config()

TIMES = np.arange(N_STEPS) * cfg.solver.dt


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    dcn, record = build_mini_leak_record(tmp_path_factory.mktemp("nets"), cfg)
    out = write_scenario(tmp_path_factory.mktemp("records") / "s7", dcn, cfg, record)
    return dcn, record, out, load_scenario(out)


def junction_order(dcn):
    """Node index space the loader must reproduce: sorted junctions, then
    the plant header (the mirrored DiTEC source) last."""
    return sorted(dcn.pairing) + ["1"]


# ---------------------------------------------------------------- structure


def test_node_types_pair_supply_and_return_by_index(loaded):
    dcn, record, out, data = loaded
    ids = junction_order(dcn)

    for side in ("supply", "return"):
        store = data[side]
        assert store.junction == ids
        assert store.num_nodes == len(ids)
        assert store.x.shape == (len(ids), N_STEPS, 2)
        assert store.x.dtype == torch.float32
        # the plant header is a node (so plant/pipe edges close the loop)
        # but its junction dynamics are unrecorded -> NaN, flagged is_plant
        assert store.is_plant.tolist() == [False] * (len(ids) - 1) + [True]
        assert torch.isnan(store.x[-1]).all()
        assert not torch.isnan(store.x[:-1]).any()

    assert data["supply"].channels == ["supply_temp", "pressure_s"]
    assert data["return"].channels == ["return_temp", "pressure_r"]
    # twin pairing by index: dT is a positional subtraction, no joins
    assert data["supply"].junction == data["return"].junction


def test_pipe_edges_are_directed_and_mirror_reversed(loaded):
    dcn, record, out, data = loaded
    ids = junction_order(dcn)
    index = {j: i for i, j in enumerate(ids)}
    pipes = sorted(dcn.pipe_pairing)

    supply = data["supply", "pipe", "supply"]
    ret = data["return", "pipe", "return"]
    assert supply.pipe == pipes
    assert ret.pipe == pipes
    assert supply.edge_index.shape == (2, len(pipes))

    # supply follows the DiTEC adj_list orientation; return is reversed
    adj = {"1": ("2", "1"), "2": ("2", "3"), "3": ("3", "4")}
    for e, p in enumerate(pipes):
        a, b = adj[p]
        assert supply.edge_index[:, e].tolist() == [index[a], index[b]]
        assert ret.edge_index[:, e].tolist() == [index[b], index[a]]


def test_ets_edges_cross_from_consumer_supply_to_return(loaded):
    dcn, record, out, data = loaded
    ids = junction_order(dcn)
    index = {j: i for i, j in enumerate(ids)}
    consumers = sorted(dcn.consumers)

    ets = data["supply", "ets", "return"]
    assert ets.junction == consumers
    for e, j in enumerate(consumers):
        assert ets.edge_index[:, e].tolist() == [index[j], index[j]]


def test_plant_edge_closes_the_loop_with_make_up_flow_on_graph(loaded):
    dcn, record, out, data = loaded
    header = len(junction_order(dcn)) - 1

    plant = data["return", "plant", "supply"]
    assert plant.edge_index.tolist() == [[header], [header]]
    assert plant.channels == [
        "plant_power", "plant_flow", "make_up_flow", "leak_flow",
    ]
    r = record.result
    attrs = plant.edge_attr.double().numpy()[0]
    np.testing.assert_allclose(attrs[:, 1], r.plant_flow, rtol=1e-6)
    np.testing.assert_allclose(attrs[:, 2], r.make_up_flow, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(attrs[:, 3], r.leak_flow, rtol=1e-6, atol=1e-12)
    # leak scenario: the make-up channel actually carries the signature
    assert attrs[:, 2].max() > 0.0


def test_hetero_data_validates(loaded):
    *_, data = loaded
    assert data.validate(raise_on_error=True)


# -------------------------------------------------------------- round-trip


def test_signed_pipe_flows_round_trip(loaded):
    dcn, record, out, data = loaded
    r = record.result
    pipes = sorted(dcn.pipe_pairing)

    for store, side in ((data["supply", "pipe", "supply"], 0),
                        (data["return", "pipe", "return"], 1)):
        assert store.channels == ["flow", "velocity", "headloss", "heat_gain"]
        assert store.edge_attr.shape == (len(pipes), N_STEPS, 4)
        for e, p in enumerate(pipes):
            wntr_name = dcn.pipe_pairing[p][side]
            written = r.pipe_flow[wntr_name].to_numpy()
            flow = store.edge_attr[e, :, 0].double().numpy()
            np.testing.assert_allclose(flow, written, rtol=1e-6)
            # float32 keeps every sign exactly
            np.testing.assert_array_equal(np.sign(flow), np.sign(written))


def test_node_and_ets_features_round_trip(loaded):
    dcn, record, out, data = loaded
    r = record.result
    ids = junction_order(dcn)

    for col, j in enumerate(ids[:-1]):
        np.testing.assert_allclose(
            data["supply"].x[col, :, 0].double().numpy(),
            r.supply_temp[j].to_numpy(), rtol=1e-6,
        )
        np.testing.assert_allclose(
            data["return"].x[col, :, 1].double().numpy(),
            r.pressure_r[j].to_numpy(), rtol=1e-6,
        )

    ets = data["supply", "ets", "return"]
    assert ets.channels == [
        "cooling_load", "consumer_flow", "delivered_load", "ets_return",
        "dT", "unmet",
    ]
    for e, j in enumerate(ets.junction):
        attrs = ets.edge_attr[e].double().numpy()
        np.testing.assert_allclose(attrs[:, 0], record.loads[j], rtol=1e-6)
        np.testing.assert_allclose(
            attrs[:, 1], r.consumer_flow[j].to_numpy(), rtol=1e-6
        )
        np.testing.assert_allclose(
            attrs[:, 4],
            (r.return_temp[j] - r.supply_temp[j]).to_numpy(),
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            attrs[:, 5], r.unmet[j].to_numpy().astype(float)
        )


def test_static_features_round_trip(loaded):
    dcn, record, out, data = loaded
    ids = junction_order(dcn)

    for side in ("supply", "return"):
        store = data[side]
        assert store.static_channels == [
            "elevation", "design_load", "design_flow", "ua", "dT_design",
            "healthy_ua_factor",  # NaN on this legacy (pre-plan) record
        ]
        static = store.static.double().numpy()
        for row, j in enumerate(ids[:-1]):
            assert static[row, 0] == pytest.approx(
                dcn.wn.get_node(dcn.pairing[j][0]).elevation
            )
        assert store.is_consumer.tolist() == [
            j in dcn.consumers for j in ids
        ]
        assert store.archetype == [ARCHETYPES.get(j) for j in ids]
    for j in sorted(dcn.consumers):
        assert data["supply"].static.double().numpy()[
            ids.index(j), 3
        ] == pytest.approx(record.ua[j], rel=1e-6)

    pipes = sorted(dcn.pipe_pairing)
    for store in (data["supply", "pipe", "supply"], data["return", "pipe", "return"]):
        assert store.static_channels == [
            "length", "diameter", "roughness", "U", "insulation_extrapolated",
        ]
        geometry = store.static.double().numpy()
        for e, p in enumerate(pipes):
            p_s = dcn.pipe_pairing[p][0]
            link = dcn.wn.get_link(p_s)
            assert geometry[e, 0] == pytest.approx(link.length)
            assert geometry[e, 1] == pytest.approx(link.diameter, rel=1e-6)
            # per-pipe derived U' + product-range flag as 0/1
            assert geometry[e, 3] == pytest.approx(
                record.heat_gain.U[p_s], rel=1e-6
            )
            assert geometry[e, 4] == float(record.heat_gain.extrapolated[p_s])

    plant = data["return", "plant", "supply"]
    assert plant.static_channels == [
        "pump_design_flow", "pump_design_head", "ditec_reservoir_head",
        "T_supply_design",
    ]
    assert plant.static.double().numpy()[0, 0] == pytest.approx(
        dcn.pump_design[0], rel=1e-6
    )


# ------------------------------------------------- labels, masks, alignment


def test_labels_and_masks_align_with_the_time_axis(loaded):
    dcn, record, out, data = loaded

    assert data.n_steps == N_STEPS
    assert data.dt == cfg.solver.dt
    assert data.time.shape == (N_STEPS,)
    np.testing.assert_allclose(data.time.numpy(), TIMES)

    # every dynamic tensor shares dim 1 == n_steps with the labels/masks
    stores = [data["supply"].x, data["return"].x]
    stores += [
        data[t].edge_attr
        for t in [
            ("supply", "pipe", "supply"), ("return", "pipe", "return"),
            ("supply", "ets", "return"), ("return", "plant", "supply"),
        ]
    ]
    for tensor in stores:
        assert tensor.shape[1] == N_STEPS

    assert data.fault_active.dtype == torch.bool
    np.testing.assert_array_equal(data.fault_active.numpy(), record.label.mask)
    np.testing.assert_allclose(
        data.realized_leak_flow.double().numpy(),
        record.label.realized["leak_flow"], rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_array_equal(
        data.converged.numpy(), record.result.converged
    )
    np.testing.assert_array_equal(
        data.iterations.numpy(), record.result.iterations
    )

    fault = data.fault
    assert fault["kind"] == "leak"
    assert fault["location"] == "3"
    assert fault["side"] == "return"
    assert fault["onset"] == record.label.onset


def test_fault_location_mask_marks_the_leaked_node_only(loaded):
    """Localisation target: the return-side leak at
    junction "3" marks exactly that node on the return store — and nothing
    else, even though a pipe "3" and a consumer "3" also exist (junction
    and pipe ids share a namespace; routing is by fault kind)."""
    dcn, record, out, data = loaded
    ids = junction_order(dcn)

    ret = data["return"].fault_location_mask
    assert ret.dtype == torch.bool
    assert ret.tolist() == [j == "3" for j in ids]
    assert not data["supply"].fault_location_mask.any()
    assert not data["supply", "pipe", "supply"].fault_location_mask.any()
    assert not data["return", "pipe", "return"].fault_location_mask.any()
    assert not data["supply", "ets", "return"].fault_location_mask.any()
    assert not data["return", "plant", "supply"].fault_location_mask.any()


def test_sensor_mask_is_exposed_per_node(loaded):
    dcn, record, out, data = loaded
    ids = junction_order(dcn)
    mask = record.sensor_mask

    for side in ("supply", "return"):
        store = data[side]
        assert store.heat_metered.dtype == torch.bool
        # plant fully sensed -> header row True on both channels
        assert store.heat_metered.tolist() == [
            j in mask.heat_metered for j in ids[:-1]
        ] + [True]
        assert store.pressure_sensed.tolist() == [
            j in mask.pressure_sensed for j in ids[:-1]
        ] + [True]


# ------------------------------------------------------------- independence


def test_loading_depends_only_on_the_records_folder(loaded, tmp_path):
    """The folder is the whole interface — a copy
    of the files loads identically with no writer state in memory."""
    *_, out, data = loaded
    copy = tmp_path / "elsewhere"
    shutil.copytree(out, copy)

    again = load_scenario(copy)
    torch.testing.assert_close(again["supply"].x, data["supply"].x, equal_nan=True)
    torch.testing.assert_close(
        again["supply", "pipe", "supply"].edge_attr,
        data["supply", "pipe", "supply"].edge_attr,
    )
    assert torch.equal(
        again["return", "plant", "supply"].edge_index,
        data["return", "plant", "supply"].edge_index,
    )
    assert torch.equal(again.fault_active, data.fault_active)
    assert again["supply"].junction == data["supply"].junction
    assert again.network == data.network
