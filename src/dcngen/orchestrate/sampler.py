"""Scenario-plan sampler: config + master seed in, complete scenario-plan
manifest out — one row per scenario with every draw materialised, so the
whole release is reviewable before any simulation runs. The manifest is
the resume/chunk unit the driver consumes.

HSPO classes and every band are uniform within anchored bounds; ranges
live in ``cfg.sampler`` / ``cfg.heat_gain`` / ``cfg.faults`` /
``cfg.ets`` — none in code.

Seeding: one master seed per release; each scenario draws from
independent per-channel streams ``SeedSequence(master, spawn_key=(uid,
channel))`` with the named :class:`Channel` enum (values are release
contract — never renumber), so adding draws to one channel never shifts
another. Plan-level draws (Louvain seed, allocations) use the reserved
head :data:`PLAN_KEY` outside the uid space. Any single row regenerates
from (master seed, uid) plus the recomputable allocation.

Archetype assignment is DiTEC's Louvain minority mechanism with the
direction inverted for DC: communities of the junction graph (edge
weight = pipe length, computed once per network from the reference draw),
smallest first, whole communities assigned *residential* until the drawn
minority fraction of consumers is reached; remaining consumers draw
uniformly from the commercial archetypes.

Knot-jitter draws are per (scenario, archetype, day-kind) knot; the
day-join endpoints (hour 0/24 of both kinds) share the weekday factor so
consecutive days still join continuously after application (application =
knots x factors, then weekly-peak renormalisation — the runner's job).
"""

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from math import ceil, floor
from pathlib import Path

import networkx as nx
import numpy as np

from dcngen.config import Config, TierPlan
from dcngen.faults.taxonomy import FaultKind
from dcngen.loads.archetypes import ARCHETYPE_NAMES, profile_knots
from dcngen.loads.load_generator import YEAR_SECONDS
from dcngen.thermal.heat_gain import draw_knobs
from dcngen.topology.ditec_loader import DitecNetwork

TIERS = ("bulk", "week", "month")
# Scenario classes = the TierPlan count keys. "normal" is the dataset class
# whose rows carry the explicit no-fault label, hence fault kind "none"
# (taxonomy.FaultKind.NONE) — the one place the two vocabularies meet.
CLASSES = ("normal", "leak", "fouling", "bypass")
SIDES = ("supply", "return")
_WEEKDAYS = 7  # calendar structure, not a tunable
_DAY_KINDS = ("weekday", "weekend")

# Commercial archetypes in the fixed draw order (uniform among them;
# "residential" is the Louvain minority class).
COMMERCIAL_ARCHETYPES = ("office", "retail", "hotel")
assert set(COMMERCIAL_ARCHETYPES) | {"residential"} == set(ARCHETYPE_NAMES), (
    "the minority split must cover exactly the archetype vocabulary"
)

# Reserved spawn-key head for plan-level draws (Louvain seed, allocations):
# max uint32, far outside the uid space (0..3015 for the v1 release).
PLAN_KEY = 2**32 - 1


class Channel(IntEnum):
    """Named per-scenario draw streams.

    Values are part of the release contract: they enter every spawn key,
    so renumbering silently re-seeds every published scenario. Append new
    channels with fresh values; never reuse or reorder.
    """

    STATIC_DRAW = 0
    TEMPERATURE = 1
    HEAT_GAIN = 2
    ARCHETYPES = 3
    LOAD_NOISE = 4
    START_WEEKDAY = 5
    KNOT_JITTER = 6
    HEALTHY_UA = 7
    SEASONAL = 8
    FAULT = 9
    # runtime streams (consumed by the plan runner, not the sampler): the
    # AR(1) load-noise path — hyperparameters live in the row, the path
    # itself is simulation-time randomness keyed the same way so a scenario
    # still regenerates from (master seed, uid) alone
    LOAD_PATH = 10


class PlanChannel(IntEnum):
    """Plan-level draw streams under the reserved :data:`PLAN_KEY` head."""

    LOUVAIN = 0
    ALLOCATION = 1


def channel_rng(master_seed: int, uid: int, channel: Channel) -> np.random.Generator:
    """The (uid, channel) stream: independent of every other channel's use."""
    return np.random.default_rng(
        np.random.SeedSequence(master_seed, spawn_key=(int(uid), int(channel)))
    )


def plan_rng(master_seed: int, *key: int) -> np.random.Generator:
    """A plan-level stream under the reserved key head."""
    return np.random.default_rng(
        np.random.SeedSequence(master_seed, spawn_key=(PLAN_KEY, *(int(k) for k in key)))
    )


@dataclass(frozen=True)
class FaultSpec:
    """The materialised fault draw of one scenario row.

    ``kind`` is the dataset class (``none``/``leak``/``fouling``/``bypass``).
    ``junction`` is the leak junction or the ETS consumer; ``side`` is set
    for leaks only. ``severity`` is the class's own unit (leak: fraction of
    network design flow; fouling: UA multiplier m; bypass: fraction f).
    ``onset_step`` indexes the step grid; ``None`` means whole-horizon
    (always for fouling, a coin-flip for bypass, never for leaks).
    """

    kind: str
    junction: str | None
    side: str | None
    severity: float | None
    onset_step: int | None


@dataclass(frozen=True)
class ScenarioRow:
    """One scenario of the release, every draw materialised.

    Rows carry no seed material: every stream derives from the meta's
    master seed as ``SeedSequence(master, spawn_key=(uid, channel))``,
    so (master seed, uid) alone regenerates the row.
    """

    uid: int
    tier: str
    scenario_class: str  # one of CLASSES; "normal" rows have fault kind "none"
    n_steps: int
    static_draw_id: int
    T_supply: float  # [degC]; T_return co-moves as T_supply + dT_design
    T_ground: float  # [degC]
    lambda_pur: float  # heat-gain knobs [W/(m K)]
    lambda_soil: float  # [W/(m K)]
    burial_cover: float  # [m]
    start_weekday: int  # 0 = Monday .. 6 = Sunday
    archetypes: dict[str, str]  # consumer -> archetype
    noise_sigma: dict[str, float]  # consumer -> AR(1) sigma
    knot_jitter: dict[str, dict[str, tuple[float, ...]]]  # archetype -> kind -> factors
    healthy_ua_factor: dict[str, float]  # consumer -> UA margin
    seasonal_amplitude: float  # 0 outside the month tier (Keep)
    seasonal_phase: float  # [s]; 0 outside the month tier; spans the full year
    fault: FaultSpec


@dataclass(frozen=True)
class PlanMeta:
    """Release-level record: seeding scheme, environment, allocations."""

    master_seed: int
    bit_generator: str  # numpy bit generator behind every stream
    numpy_version: str
    networkx_version: str  # Louvain partition depends on it
    network: str
    reference_draw_id: int  # draw whose lengths weighted the Louvain graph
    dt: float  # [s]
    plan_key: int
    channels: dict[str, int]  # Channel name -> spawn-key value
    consumers: tuple[str, ...]
    communities: tuple[tuple[str, ...], ...]  # smallest-first, nodes sorted
    allocations: dict[str, dict[str, int]]  # "tier/class" -> stratum -> count


@dataclass(frozen=True)
class ScenarioPlan:
    """The scenario-plan manifest: meta + one row per scenario, uid-ordered."""

    meta: PlanMeta
    rows: tuple[ScenarioRow, ...]


def _junction_graph(net: DitecNetwork) -> nx.Graph:
    """Junction-only graph, edge weight = pipe length (DiTEC's Louvain
    weighting); reservoir connections are not community ties."""
    g = nx.Graph()
    g.add_nodes_from(net.junctions)
    junctions = set(net.junctions)
    for pipe in net.pipes:
        if pipe.start in junctions and pipe.end in junctions:
            g.add_edge(pipe.start, pipe.end, weight=net.draw.pipe_length[pipe.id])
    return g


def _louvain_communities(
    net: DitecNetwork, master_seed: int
) -> tuple[tuple[str, ...], ...]:
    """Seeded Louvain partition, canonicalised smallest-first (ties by the
    sorted node ids) — computed once per network per release."""
    seed = int(plan_rng(master_seed, PlanChannel.LOUVAIN).integers(2**32))
    parts = nx.community.louvain_communities(_junction_graph(net), seed=seed)
    ordered = sorted(
        (tuple(sorted(c)) for c in parts), key=lambda c: (len(c), c)
    )
    return tuple(ordered)


def _stratified_counts(
    n: int, strata: list[str], rng: np.random.Generator
) -> dict[str, int]:
    """Largest-remainder allocation with a seeded-shuffle remainder: every
    stratum gets ``n // len(strata)``; a shuffled prefix gets one extra.
    Equivalent to the spec's ceil-copies-then-trim arithmetic (e.g. leaks
    13 x 62 -> trim 806 -> 800)."""
    base, extra = divmod(n, len(strata))
    counts = {s: base for s in strata}
    for i in rng.permutation(len(strata))[:extra]:
        counts[strata[i]] += 1
    return counts


def _expand_allocation(counts: dict[str, int], strata: list[str]) -> list[str]:
    """Stratum id per scenario slot, in canonical stratum order."""
    return [s for s in strata for _ in range(counts[s])]


def _assign_archetypes(
    communities: tuple[tuple[str, ...], ...],
    consumers: tuple[str, ...],
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Whole smallest communities become residential until the drawn
    minority fraction of consumers is reached; the rest draw commercial."""
    fraction = float(
        rng.uniform(
            cfg.sampler.residential_minority_min, cfg.sampler.residential_minority_max
        )
    )
    target = fraction * len(consumers)
    consumer_set = set(consumers)
    residential: set[str] = set()
    for community in communities:  # smallest first
        if len(residential) >= target:
            break
        residential |= set(community) & consumer_set
    return {
        j: (
            "residential"
            if j in residential
            else COMMERCIAL_ARCHETYPES[int(rng.integers(len(COMMERCIAL_ARCHETYPES)))]
        )
        for j in consumers
    }


def _knot_jitter(
    cfg: Config, rng: np.random.Generator
) -> dict[str, dict[str, tuple[float, ...]]]:
    """Perturb factors per (archetype, day-kind, knot), endpoints tied:
    the hour-0/24 knots of both kinds share the weekday factor so days keep
    joining continuously after application."""
    j = cfg.sampler.knot_jitter
    out: dict[str, dict[str, tuple[float, ...]]] = {}
    for archetype in ARCHETYPE_NAMES:
        knots = profile_knots(archetype)
        factors = {
            kind: [float(f) for f in rng.uniform(1.0 - j, 1.0 + j, len(knots[kind]))]
            for kind in _DAY_KINDS
        }
        shared = factors["weekday"][0]
        factors["weekday"][-1] = shared
        factors["weekend"][0] = shared
        factors["weekend"][-1] = shared
        out[archetype] = {kind: tuple(factors[kind]) for kind in _DAY_KINDS}
    return out


def _onset_step(n_steps: int, cfg: Config, rng: np.random.Generator) -> int:
    """Abrupt onset as a step index in the configured window (the same
    quantisation as ``faults.injector._abrupt_onset_time``)."""
    k_lo = ceil(cfg.faults.onset_window_start * n_steps)
    k_hi = floor(cfg.faults.onset_window_end * n_steps)
    if k_lo > k_hi:
        raise ValueError(
            f"onset window [{cfg.faults.onset_window_start}, "
            f"{cfg.faults.onset_window_end}] holds no step start on a "
            f"{n_steps}-step horizon"
        )
    return int(rng.integers(k_lo, k_hi + 1))


def _fault_spec(
    cls: str, location: str | None, n_steps: int, cfg: Config, rng: np.random.Generator
) -> FaultSpec:
    """Materialise one row's fault draw for its scenario class: severity
    from the class band, onset per the class's temporal convention (leak
    abrupt, fouling chronic, bypass the coin-flip). Kind strings come
    from the taxonomy enum — the single vocabulary the labels use."""
    f = cfg.faults
    if cls == "normal":
        return FaultSpec(FaultKind.NONE.value, None, None, None, None)
    if cls == "leak":
        junction, side = location.rsplit(":", 1)
        severity = float(rng.uniform(f.leak_severity_min, f.leak_severity_max))
        return FaultSpec(
            FaultKind.LEAK.value, junction, side, severity, _onset_step(n_steps, cfg, rng)
        )
    if cls == "fouling":
        severity = float(rng.uniform(f.fouling_severity_min, f.fouling_severity_max))
        return FaultSpec(FaultKind.FOULING.value, location, None, severity, None)
    if cls == "bypass":
        fraction = float(rng.uniform(f.bypass_fraction_min, f.bypass_fraction_max))
        whole = bool(rng.random() < f.bypass_whole_horizon_prob)
        onset = None if whole else _onset_step(n_steps, cfg, rng)
        return FaultSpec(FaultKind.BYPASS.value, location, None, fraction, onset)
    raise ValueError(f"unknown scenario class {cls!r}")


def _draw_row(
    uid: int,
    tier: str,
    cls: str,
    location: str | None,
    n_steps: int,
    consumers: tuple[str, ...],
    communities: tuple[tuple[str, ...], ...],
    cfg: Config,
) -> ScenarioRow:
    """Materialise one scenario row: every per-scenario draw, each from its
    own (uid, channel) stream, in the fixed order the fields document."""
    m, sp = cfg.seed, cfg.sampler
    r_temp = channel_rng(m, uid, Channel.TEMPERATURE)
    knobs = draw_knobs(cfg.heat_gain, channel_rng(m, uid, Channel.HEAT_GAIN))
    r_noise = channel_rng(m, uid, Channel.LOAD_NOISE)
    r_healthy = channel_rng(m, uid, Channel.HEALTHY_UA)
    if tier == "month":
        r_seasonal = channel_rng(m, uid, Channel.SEASONAL)
        amplitude = float(
            r_seasonal.uniform(sp.seasonal_amplitude_min, sp.seasonal_amplitude_max)
        )
        # the phase spans the FULL annual cycle regardless of horizon: a
        # month-long record starts anywhere in the year
        phase = float(r_seasonal.uniform(0.0, YEAR_SECONDS))
    else:
        amplitude, phase = 0.0, 0.0  # Keep: flat outside the month tier
    return ScenarioRow(
        uid=uid,
        tier=tier,
        scenario_class=cls,
        n_steps=n_steps,
        static_draw_id=int(
            channel_rng(m, uid, Channel.STATIC_DRAW).integers(sp.static_draw_count)
        ),
        T_supply=float(r_temp.uniform(sp.T_supply_min, sp.T_supply_max)),
        T_ground=float(r_temp.uniform(sp.T_ground_min, sp.T_ground_max)),
        lambda_pur=knobs.lambda_pur,
        lambda_soil=knobs.lambda_soil,
        burial_cover=knobs.burial_cover,
        start_weekday=int(
            channel_rng(m, uid, Channel.START_WEEKDAY).integers(_WEEKDAYS)
        ),
        archetypes=_assign_archetypes(
            communities, consumers, cfg, channel_rng(m, uid, Channel.ARCHETYPES)
        ),
        noise_sigma={
            j: float(r_noise.uniform(sp.noise_sigma_min, sp.noise_sigma_max))
            for j in consumers
        },
        knot_jitter=_knot_jitter(cfg, channel_rng(m, uid, Channel.KNOT_JITTER)),
        healthy_ua_factor={
            j: float(
                r_healthy.uniform(
                    cfg.ets.healthy_ua_factor_min, cfg.ets.healthy_ua_factor_max
                )
            )
            for j in consumers
        },
        seasonal_amplitude=amplitude,
        seasonal_phase=phase,
        fault=_fault_spec(cls, location, n_steps, cfg, channel_rng(m, uid, Channel.FAULT)),
    )


def _tier_steps(tier_plan: TierPlan, cfg: Config) -> int:
    """Steps per scenario of a tier (config validates the grid divides)."""
    return int(round(tier_plan.horizon / cfg.solver.dt))


def build_plan(cfg: Config, net: DitecNetwork) -> ScenarioPlan:
    """Materialise the release: every scenario row, every draw, up front.

    Args:
        cfg: configuration; ``cfg.seed`` is the master seed, ``cfg.sampler``
            the tier table and bands.
        net: the release network topology (its draw supplies the Louvain
            edge weights and the consumer set — census #15: every junction
            is a consumer in all 36 networks). Must be the network the
            config names, so the manifest cannot contradict its config.
    """
    if net.name != cfg.sampler.network:
        raise ValueError(
            f"plan network mismatch: config names {cfg.sampler.network!r} "
            f"but got topology {net.name!r}"
        )
    consumers = tuple(
        sorted(j for j in net.junctions if net.draw.base_demand.get(j, 0.0) > 0.0)
    )
    if not consumers:
        raise ValueError(f"network {net.name} has no consumers")
    communities = _louvain_communities(net, cfg.seed)

    leak_strata = [f"{j}:{side}" for j in sorted(net.junctions) for side in SIDES]
    ets_strata = list(consumers)

    allocations: dict[str, dict[str, int]] = {}
    rows: list[ScenarioRow] = []
    uid = 0
    for tier_idx, tier in enumerate(TIERS):
        tier_plan: TierPlan = getattr(cfg.sampler, tier)
        n_steps = _tier_steps(tier_plan, cfg)
        for cls_idx, cls in enumerate(CLASSES):
            n = getattr(tier_plan, cls)
            if n == 0:
                continue
            if cls == "normal":
                locations: list[str | None] = [None] * n
            else:
                strata = leak_strata if cls == "leak" else ets_strata
                counts = _stratified_counts(
                    n, strata, plan_rng(cfg.seed, PlanChannel.ALLOCATION, tier_idx, cls_idx)
                )
                allocations[f"{tier}/{cls}"] = counts
                locations = _expand_allocation(counts, strata)
            for location in locations:
                rows.append(
                    _draw_row(uid, tier, cls, location, n_steps, consumers, communities, cfg)
                )
                uid += 1

    meta = PlanMeta(
        master_seed=cfg.seed,
        # derived from an actual stream so the record can never drift from
        # what channel_rng constructs
        bit_generator=type(
            channel_rng(cfg.seed, 0, Channel.STATIC_DRAW).bit_generator
        ).__name__,
        numpy_version=np.__version__,
        networkx_version=nx.__version__,
        network=net.name,
        reference_draw_id=net.draw.scenario_id,
        dt=cfg.solver.dt,
        plan_key=PLAN_KEY,
        channels={c.name: int(c) for c in Channel},
        consumers=consumers,
        communities=communities,
        allocations=allocations,
    )
    return ScenarioPlan(meta=meta, rows=tuple(rows))


def plan_to_json(plan: ScenarioPlan) -> str:
    """Canonical serialization: sorted keys, compact separators, no NaN —
    byte-deterministic (same master seed + environment -> identical
    bytes)."""
    return json.dumps(
        asdict(plan), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def write_plan(plan: ScenarioPlan, path: str | Path) -> Path:
    """Write the manifest to ``path`` (canonical JSON, utf-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_to_json(plan) + "\n", encoding="utf-8")
    return path


def load_plan(path: str | Path) -> ScenarioPlan:
    """Load a manifest back into the typed plan (inverse of write_plan)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = PlanMeta(
        **{
            **raw["meta"],
            "consumers": tuple(raw["meta"]["consumers"]),
            "communities": tuple(tuple(c) for c in raw["meta"]["communities"]),
        }
    )
    rows = tuple(
        ScenarioRow(
            **{
                **r,
                "knot_jitter": {
                    a: {k: tuple(f) for k, f in kinds.items()}
                    for a, kinds in r["knot_jitter"].items()
                },
                "fault": FaultSpec(**r["fault"]),
            }
        )
        for r in raw["rows"]
    )
    return ScenarioPlan(meta=meta, rows=rows)
