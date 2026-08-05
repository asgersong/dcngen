"""Sampler seam: build_plan materialises every draw for the
release into the scenario-plan manifest — counts exact, every value inside
its locked band, byte-deterministic from the master seed, channels
independent. Mini-network tests use a shrunk tier table; the full-shape
run lives in test_sampler_hanoi.py.
"""

import networkx as nx
import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.loads.archetypes import ARCHETYPE_NAMES, profile_knots
from dcngen.orchestrate.sampler import (
    PLAN_KEY,
    Channel,
    build_plan,
    channel_rng,
    load_plan,
    plan_to_json,
    write_plan,
)
from dcngen.topology.ditec_loader import load_ditec_network
from tests.conftest import shrink_sampler_tiers as shrink_tiers
from tests.conftest import write_mutated_default


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    path = write_mutated_default(tmp_path_factory.mktemp("cfg"), shrink_tiers)
    return load_config(path)


@pytest.fixture(scope="module")
def mini_net(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return load_ditec_network(folder, scenario_id=0)


@pytest.fixture(scope="module")
def plan(cfg, mini_net):
    return build_plan(cfg, mini_net)


def test_channel_streams_are_independent():
    # AC: adding a draw to one channel leaves the other channels' streams
    # unchanged — each (uid, channel) pair owns a spawned SeedSequence
    before = channel_rng(42, 7, Channel.TEMPERATURE).uniform(size=5)

    # consume extra draws from every OTHER channel, then re-derive
    for ch in Channel:
        if ch is not Channel.TEMPERATURE:
            channel_rng(42, 7, ch).uniform(size=100)
    after = channel_rng(42, 7, Channel.TEMPERATURE).uniform(size=5)

    np.testing.assert_array_equal(before, after)


def test_channel_streams_differ_across_uids_channels_and_seeds():
    base = channel_rng(42, 7, Channel.TEMPERATURE).uniform(size=5)
    for other in (
        channel_rng(42, 8, Channel.TEMPERATURE),
        channel_rng(42, 7, Channel.HEAT_GAIN),
        channel_rng(43, 7, Channel.TEMPERATURE),
    ):
        assert not np.array_equal(base, other.uniform(size=5))


def rows_of(plan, tier, scenario_class):
    return [r for r in plan.rows if r.tier == tier and r.scenario_class == scenario_class]


def test_plan_covers_all_tiers_and_classes_with_exact_counts(plan, cfg):
    sp = cfg.sampler
    for tier in ("bulk", "week", "month"):
        tp = getattr(sp, tier)
        assert len(rows_of(plan, tier, "normal")) == tp.normal
        assert len(rows_of(plan, tier, "leak")) == tp.leak
        assert len(rows_of(plan, tier, "fouling")) == tp.fouling
        assert len(rows_of(plan, tier, "bypass")) == tp.bypass
    n_total = sum(
        getattr(getattr(sp, t), c)
        for t in ("bulk", "week", "month")
        for c in ("normal", "leak", "fouling", "bypass")
    )
    assert [r.uid for r in plan.rows] == list(range(n_total))
    for tier, steps in (("bulk", 24), ("week", 48), ("month", 72)):
        assert all(r.n_steps == steps for r in plan.rows if r.tier == tier)


def test_stratified_allocation_is_balanced_and_recorded(plan, cfg):
    # bulk leaks: 7 over 6 (junction, side) strata -> one stratum gets 2,
    # the rest 1 (largest remainder); the meta record matches the rows
    leak_locs = [
        (r.fault.junction, r.fault.side) for r in rows_of(plan, "bulk", "leak")
    ]
    strata = {(j, s) for j in ("2", "3", "4") for s in ("supply", "return")}
    assert set(leak_locs) == strata  # full coverage
    counts = {loc: leak_locs.count(loc) for loc in strata}
    assert sorted(counts.values()) == [1, 1, 1, 1, 1, 2]
    recorded = plan.meta.allocations["bulk/leak"]
    assert {tuple(k.rsplit(":", 1)): v for k, v in recorded.items()} == counts

    # bulk ETS faults stratify over the 3 consumers
    fouling_locs = [r.fault.junction for r in rows_of(plan, "bulk", "fouling")]
    assert sorted(fouling_locs.count(j) for j in ("2", "3", "4")) == [1, 1, 2]
    bypass_locs = [r.fault.junction for r in rows_of(plan, "bulk", "bypass")]
    assert sorted(bypass_locs.count(j) for j in ("2", "3", "4")) == [1, 2, 2]


def test_every_sampled_value_lies_inside_its_locked_band(plan, cfg):
    sp, hg, ets = cfg.sampler, cfg.heat_gain, cfg.ets
    for r in plan.rows:
        assert 0 <= r.static_draw_id < sp.static_draw_count
        assert sp.T_supply_min <= r.T_supply <= sp.T_supply_max
        assert sp.T_ground_min <= r.T_ground <= sp.T_ground_max
        assert hg.lambda_pur_min <= r.lambda_pur <= hg.lambda_pur_max
        assert hg.lambda_soil_min <= r.lambda_soil <= hg.lambda_soil_max
        assert hg.burial_cover_min <= r.burial_cover <= hg.burial_cover_max
        assert 0 <= r.start_weekday <= 6
        assert set(r.noise_sigma) == {"2", "3", "4"}
        for sigma in r.noise_sigma.values():
            assert sp.noise_sigma_min <= sigma <= sp.noise_sigma_max
        for factor in r.healthy_ua_factor.values():
            assert ets.healthy_ua_factor_min <= factor <= ets.healthy_ua_factor_max
        for kinds in r.knot_jitter.values():
            for factors in kinds.values():
                assert all(
                    1.0 - sp.knot_jitter <= x <= 1.0 + sp.knot_jitter for x in factors
                )
        if r.tier == "month":
            assert sp.seasonal_amplitude_min <= r.seasonal_amplitude <= sp.seasonal_amplitude_max
            assert 0.0 <= r.seasonal_phase <= 365.0 * 24 * 3600  # phase spans the YEAR
        else:
            assert (r.seasonal_amplitude, r.seasonal_phase) == (0.0, 0.0)


def test_fault_severities_and_onsets_follow_their_classes(plan, cfg):
    f = cfg.faults
    for r in plan.rows:
        spec = r.fault
        k_lo = int(np.ceil(f.onset_window_start * r.n_steps))
        k_hi = int(np.floor(f.onset_window_end * r.n_steps))
        if spec.kind == "none":
            assert spec == type(spec)("none", None, None, None, None)
        elif spec.kind == "leak":
            assert f.leak_severity_min <= spec.severity <= f.leak_severity_max
            assert spec.side in ("supply", "return")
            assert k_lo <= spec.onset_step <= k_hi  # always abrupt
        elif spec.kind == "fouling":
            assert f.fouling_severity_min <= spec.severity <= f.fouling_severity_max
            assert spec.onset_step is None and spec.side is None  # chronic
        else:
            assert f.bypass_fraction_min <= spec.severity <= f.bypass_fraction_max
            assert spec.side is None
            if spec.onset_step is not None:
                assert k_lo <= spec.onset_step <= k_hi
    # both bypass temporal variants are reachable (coin-flip)
    onsets = [r.fault.onset_step for r in plan.rows if r.fault.kind == "bypass"]
    assert any(o is None for o in onsets) and any(o is not None for o in onsets)


def test_archetype_assignment_is_community_coherent(plan):
    # residential nodes are exactly a smallest-first prefix of whole
    # communities. Louvain merges the 3-node mini path into ONE community
    # for any seed, so the mechanism faithfully degenerates to
    # all-residential here — the real minority/commercial behaviour (and
    # non-degeneracy of the release partition) is asserted at Hanoi scale
    # in test_sampler_hanoi.py.
    communities = plan.meta.communities
    assert len(communities) == 1  # test premise: the degenerate partition
    for r in plan.rows:
        assert set(r.archetypes) == {"2", "3", "4"}
        residential = {j for j, a in r.archetypes.items() if a == "residential"}
        covered: set[str] = set()
        for community in communities:  # smallest first
            if covered >= residential:
                break
            covered |= set(community)
        assert covered == residential  # a prefix of whole communities
        assert residential == {"2", "3", "4"}  # the degenerate outcome


def test_knot_jitter_matches_knot_counts_and_ties_the_day_join(plan):
    r = plan.rows[0]
    assert set(r.knot_jitter) == set(ARCHETYPE_NAMES)
    for archetype in ARCHETYPE_NAMES:
        knots = profile_knots(archetype)
        factors = r.knot_jitter[archetype]
        for kind in ("weekday", "weekend"):
            assert len(factors[kind]) == len(knots[kind])
        # the day-join endpoints (hour 0/24, both kinds) share one factor
        wd, we = factors["weekday"], factors["weekend"]
        assert wd[0] == wd[-1] == we[0] == we[-1]


def test_same_master_seed_reproduces_draws_and_different_seed_moves_them(
    tmp_path, cfg, mini_net, plan
):
    again = build_plan(cfg, mini_net)
    assert again == plan

    def reseed(d):
        shrink_tiers(d)
        d["seed"] = 43

    other_cfg = load_config(write_mutated_default(tmp_path, reseed))
    other = build_plan(other_cfg, mini_net)
    assert other != plan


def test_changing_one_channels_band_leaves_other_channels_untouched(
    tmp_path, cfg, mini_net, plan
):
    # AC channel independence, end to end: widening the noise band must not
    # move any other channel's materialised values
    def widen_noise(d):
        shrink_tiers(d)
        d["sampler"]["noise_sigma_max"] = 0.5

    widened = build_plan(
        load_config(write_mutated_default(tmp_path, widen_noise)), mini_net
    )
    for a, b in zip(plan.rows, widened.rows):
        assert a.noise_sigma != b.noise_sigma  # the widened channel moved
        assert a.static_draw_id == b.static_draw_id
        assert (a.T_supply, a.T_ground) == (b.T_supply, b.T_ground)
        assert (a.lambda_pur, a.lambda_soil, a.burial_cover) == (
            b.lambda_pur, b.lambda_soil, b.burial_cover,
        )
        assert a.archetypes == b.archetypes
        assert a.knot_jitter == b.knot_jitter
        assert a.healthy_ua_factor == b.healthy_ua_factor
        assert a.start_weekday == b.start_weekday
        assert a.fault == b.fault


def test_same_master_seed_reproduces_the_manifest_byte_identically(
    cfg, mini_net, plan
):
    assert plan_to_json(build_plan(cfg, mini_net)).encode() == plan_to_json(plan).encode()


def test_manifest_round_trips_through_the_file(tmp_path, plan):
    path = write_plan(plan, tmp_path / "plan.json")
    assert load_plan(path) == plan


def test_meta_records_environment_and_scheme(plan, cfg, mini_net):
    meta = plan.meta
    assert meta.master_seed == cfg.seed
    assert meta.bit_generator == "PCG64"  # numpy default; part of the card
    assert meta.numpy_version == np.__version__
    assert meta.networkx_version == nx.__version__
    assert (meta.network, meta.reference_draw_id) == ("mini_net", 0)
    assert meta.dt == cfg.solver.dt
    assert meta.plan_key == PLAN_KEY
    assert meta.channels == {c.name: int(c) for c in Channel}
    assert meta.consumers == ("2", "3", "4")
    assert set(meta.allocations) == {
        f"{t}/{c}" for t in ("bulk", "week", "month") for c in ("leak", "fouling", "bypass")
    }


def test_channel_values_are_stable_and_plan_key_reserved():
    # the enum values are part of the release contract: reordering or
    # renumbering them silently re-seeds every published scenario
    assert [(c.name, c.value) for c in Channel] == [
        ("STATIC_DRAW", 0),
        ("TEMPERATURE", 1),
        ("HEAT_GAIN", 2),
        ("ARCHETYPES", 3),
        ("LOAD_NOISE", 4),
        ("START_WEEKDAY", 5),
        ("KNOT_JITTER", 6),
        ("HEALTHY_UA", 7),
        ("SEASONAL", 8),
        ("FAULT", 9),
        ("LOAD_PATH", 10),  # runtime AR(1) path (plan runner)
    ]
    assert PLAN_KEY == 2**32 - 1  # outside the uid space 0..3015
