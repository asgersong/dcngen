"""Loader seam (input side of the DCN-ifier seam): DiTEC network
folder in -> topology + pinned static draw out, SI units.

Expected values are hand-derived literals from the synthetic mini-network in
``conftest.py``; real-dump integration tests live in
``test_ditec_loader_hanoi.py``.
"""

import pytest

from dcngen.topology.ditec_loader import DitecFormatError, load_ditec_network
from tests.conftest import MINI_ADJ_LIST, MINI_STATICS, write_ditec_folder


def test_static_draw_pinned_by_scenario_id(mini_ditec_folder):
    net = load_ditec_network(mini_ditec_folder, scenario_id=1)

    draw = net.draw
    assert draw.scenario_id == 1
    assert draw.pipe_diameter == {"1": 0.7, "2": 0.55, "3": 0.45}
    assert draw.pipe_length == {"1": 1100.0, "2": 900.0, "3": 650.0}
    assert draw.pipe_roughness == {"1": 140.0, "2": 125.0, "3": 115.0}
    assert draw.junction_elevation == {"2": 2.5, "3": 1.5, "4": 0.5}
    assert draw.reservoir_head == {"1": 65.0}


def test_base_demand_is_time_mean_over_all_shards(mini_ditec_folder):
    net = load_ditec_network(mini_ditec_folder, scenario_id=0)

    # Hand-derived: mean of [0.30,0.35,0.40] + [0.40,0.35,0.30] = 0.35, etc.
    assert net.draw.base_demand == pytest.approx({"2": 0.35, "3": 0.25, "4": 0.15})
    assert net.draw.demand_reduction == "time_mean"


def test_draw_provenance_captured_for_metadata(mini_ditec_folder):
    net = load_ditec_network(mini_ditec_folder, scenario_id=0)

    prov = net.draw.provenance
    assert prov["pipe_diameter"] == "pipe_diameter-0-static_input.parquet"
    assert prov["junction_base_demand"] == (
        "junction_base_demand-0-dynamic_input.parquet"
        ",junction_base_demand-1-dynamic_input.parquet"
    )


def test_topology_parsed_from_adj_list_footer(mini_ditec_folder):
    net = load_ditec_network(mini_ditec_folder, scenario_id=0)

    assert net.name == "mini_net"
    assert net.junctions == ("2", "3", "4")
    assert net.reservoirs == ("1",)
    assert [(p.id, p.start, p.end) for p in net.pipes] == [
        ("1", "2", "1"),
        ("2", "2", "3"),
        ("3", "3", "4"),
    ]


def test_summary_printable_for_any_folder(mini_ditec_folder):
    from dcngen.topology.ditec_loader import summarize_network

    text = summarize_network(mini_ditec_folder, scenario_id=0)

    assert "mini_net" in text
    assert "3 junctions" in text
    assert "1 reservoir" in text
    assert "3 pipes" in text
    # static ranges of the pinned draw, so a human can eyeball units
    assert "0.400" in text and "0.600" in text  # diameter range [m]


def test_unit_suspect_magnitudes_rejected(tmp_path):
    # Diameters that look like millimetres must not pass silently as metres.
    statics = {**MINI_STATICS, "pipe_diameter": {"1": [600.0], "2": [500.0], "3": [400.0]}}
    statics = {k: ({e: [v[0]] for e, v in cols.items()}) for k, cols in statics.items()}
    folder = write_ditec_folder(tmp_path / "mm_net", statics=statics, demand_scenario_ids=(0,))

    with pytest.raises(DitecFormatError, match="pipe_diameter"):
        load_ditec_network(folder, scenario_id=0)


def test_adj_list_node_missing_from_shards_rejected(tmp_path):
    bad_adj = MINI_ADJ_LIST + [["4", "99", "4"]]  # node "99" exists nowhere
    folder = write_ditec_folder(tmp_path / "bad_net", adj_list=bad_adj)

    with pytest.raises(DitecFormatError, match="99"):
        load_ditec_network(folder, scenario_id=0)


def test_missing_attrs_footer_rejected(tmp_path):
    import pyarrow.parquet as pq

    folder = write_ditec_folder(tmp_path / "no_attrs")
    path = folder / "junction_elevation-0-static_input.parquet"
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata(None), path)

    with pytest.raises(DitecFormatError, match="attrs"):
        load_ditec_network(folder, scenario_id=0)
