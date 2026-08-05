"""Acceptance: the real Hanoi folder of the DiTEC dump.

Expected counts are canonical Hanoi (31 junctions, 1 reservoir, 34 pipes);
expected magnitude ranges combine canonical Hanoi with the generation ranges
recorded in the dump's own ``attrs`` footer (roughness/length/head are HSPO
*sampled* for this network, so canonical point values do not apply).
Skipped when the local dump is not present.
"""

import pytest

from dcngen.config import load_config
from dcngen.topology.ditec_loader import load_ditec_network, summarize_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.poc.network

pytestmark = pytest.mark.skipif(
    not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
)


@pytest.fixture(scope="module")
def hanoi():
    return load_ditec_network(HANOI, scenario_id=cfg.poc.static_scenario_id)


def test_hanoi_counts_match_canonical_topology(hanoi):
    assert len(hanoi.junctions) == 31
    assert hanoi.reservoirs == ("1",)
    assert len(hanoi.pipes) == 34
    # every pipe endpoint resolves against the shard-declared nodes
    nodes = set(hanoi.junctions) | set(hanoi.reservoirs)
    assert all(p.start in nodes and p.end in nodes for p in hanoi.pipes)


def test_hanoi_pinned_draw_magnitudes_in_expected_ranges(hanoi):
    d = hanoi.draw
    assert all(0.30 <= v <= 1.10 for v in d.pipe_diameter.values())  # m
    assert all(1_000.0 <= v <= 10_000.0 for v in d.pipe_length.values())  # m
    assert all(1_500.0 <= v <= 2_500.0 for v in d.pipe_roughness.values())  # HW C
    assert all(50.0 <= v <= 160.0 for v in d.reservoir_head.values())  # m
    assert all(-10.0 <= v <= 20.0 for v in d.junction_elevation.values())  # m
    assert all(0.0 < v <= 0.8 for v in d.base_demand.values())  # m3/s
    assert set(d.base_demand) == set(hanoi.junctions)


def test_hanoi_draw_identity_recorded(hanoi):
    d = hanoi.draw
    assert d.scenario_id == cfg.poc.static_scenario_id
    assert d.demand_reduction == "time_mean"
    assert d.provenance["pipe_diameter"] == "pipe_diameter-0-static_input.parquet"
    assert "junction_base_demand-0" in d.provenance["junction_base_demand"]


def test_hanoi_summary_prints():
    text = summarize_network(HANOI, scenario_id=cfg.poc.static_scenario_id)
    assert "31 junctions, 1 reservoir, 34 pipes" in text
