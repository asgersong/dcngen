"""Acceptance at release scale: the full default plan on the
real Hanoi dump — 3,016 scenarios over three tiers, exact stratified
counts with seeded-shuffle trims (leaks 13 x 62 -> 800, fouling
18 x 31 -> 550, bypass 15 x 31 -> 450), a non-degenerate Louvain partition
driving the residential-minority assignment, and byte-identical
rebuild from the master seed. Skipped when the local dump is not present.
"""

from collections import Counter

import pytest

from dcngen.config import load_config
from dcngen.orchestrate.sampler import build_plan, plan_to_json
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.sampler.network

pytestmark = pytest.mark.skipif(
    not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}"
)


@pytest.fixture(scope="module")
def hanoi():
    return load_ditec_network(HANOI, scenario_id=0)


@pytest.fixture(scope="module")
def plan(hanoi):
    return build_plan(cfg, hanoi)


def class_rows(plan, tier, scenario_class):
    return [r for r in plan.rows if r.tier == tier and r.scenario_class == scenario_class]


def test_release_shape_3016_scenarios(plan):
    assert len(plan.rows) == 3016
    assert [r.uid for r in plan.rows] == list(range(3016))
    for tier, steps in (("bulk", 144), ("week", 1008), ("month", 4320)):
        assert all(r.n_steps == steps for r in plan.rows if r.tier == tier)
    for tier, counts in (
        ("bulk", (1000, 800, 550, 450)),
        ("week", (80, 50, 40, 30)),
        ("month", (6, 5, 3, 2)),
    ):
        normal, leak, fouling, bypass = counts
        assert len(class_rows(plan, tier, "normal")) == normal
        assert len(class_rows(plan, tier, "leak")) == leak
        assert len(class_rows(plan, tier, "fouling")) == fouling
        assert len(class_rows(plan, tier, "bypass")) == bypass


def test_bulk_strata_match_the_spec_trims(plan):
    # leaks: 13 x 62 (junction, side) trim 806 -> 800: six strata at 12
    leak = Counter(
        (r.fault.junction, r.fault.side) for r in class_rows(plan, "bulk", "leak")
    )
    assert len(leak) == 62
    assert sorted(Counter(leak.values()).items()) == [(12, 6), (13, 56)]
    # fouling: 18 x 31 trim 558 -> 550; bypass: 15 x 31 trim 465 -> 450
    fouling = Counter(r.fault.junction for r in class_rows(plan, "bulk", "fouling"))
    assert len(fouling) == 31
    assert sorted(Counter(fouling.values()).items()) == [(17, 8), (18, 23)]
    bypass = Counter(r.fault.junction for r in class_rows(plan, "bulk", "bypass"))
    assert len(bypass) == 31
    assert sorted(Counter(bypass.values()).items()) == [(14, 15), (15, 16)]


def test_thin_tiers_spread_across_strata(plan):
    week_leak = Counter(
        (r.fault.junction, r.fault.side) for r in class_rows(plan, "week", "leak")
    )
    assert len(week_leak) == 50 and set(week_leak.values()) == {1}
    week_fouling = Counter(r.fault.junction for r in class_rows(plan, "week", "fouling"))
    assert sorted(Counter(week_fouling.values()).items()) == [(1, 22), (2, 9)]
    month_bypass = Counter(r.fault.junction for r in class_rows(plan, "month", "bypass"))
    assert len(month_bypass) == 2 and set(month_bypass.values()) == {1}


def test_louvain_partition_is_non_degenerate_and_covers_hanoi(plan, hanoi):
    communities = plan.meta.communities
    assert len(communities) >= 2  # the mechanism can express a minority
    nodes = [j for community in communities for j in community]
    assert sorted(nodes) == sorted(hanoi.junctions)  # disjoint cover
    sizes = [len(c) for c in communities]
    assert sizes == sorted(sizes)  # smallest first


def test_residential_is_a_genuine_minority_with_commercial_majority(plan):
    # the release-gate counterpart of the mini net's degenerate case: every
    # scenario keeps residential a strict, present minority and draws the
    # commercial majority from the commercial trio; assignments vary across rows
    assignments = set()
    for r in plan.rows:
        share = sum(a == "residential" for a in r.archetypes.values()) / 31
        assert 0.0 < share < 0.5
        assert {a for a in r.archetypes.values() if a != "residential"} <= {
            "office", "retail", "hotel",
        }
        assignments.add(tuple(sorted(r.archetypes.items())))
    assert len(assignments) > 100  # sampled per scenario, not a constant


def test_year_tier_activates_the_seasonal_knob_only_there(plan):
    for r in plan.rows:
        if r.tier == "month":
            assert 0.1 <= r.seasonal_amplitude <= 0.5
            assert 0.0 <= r.seasonal_phase <= 31536000.0
        else:
            assert (r.seasonal_amplitude, r.seasonal_phase) == (0.0, 0.0)


def test_release_manifest_rebuilds_byte_identically(plan, hanoi):
    assert plan_to_json(build_plan(cfg, hanoi)) == plan_to_json(plan)
