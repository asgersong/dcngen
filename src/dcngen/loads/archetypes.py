"""Normalized cooling-load shapes for the four building archetypes.

Archetypes shape the *normalized time profile only* — never
magnitudes (those come from flow equivalence). Shapes are stylized daily
knot curves: office weekday-business peaked, residential
evening-peaked, hotel with morning/evening occupancy bumps, retail running
business hours including weekends. Knot values are plausibility-crafted
(DOE-commercial-reference-inspired), not citations; the qualitative
structure is pinned by tests. Datacentre (nearly flat) is out of the
dataset scope.

Contract: piecewise-linear knots per (archetype, weekday/weekend) over the
24 h clock; each archetype's maximum knot value is exactly 1.0, so the
weekly profile peaks at 1.0 independent of sampling; first and last knots
match so days join continuously. Values are strictly positive (buildings
never drop to zero cooling).
"""

import numpy as np

ARCHETYPE_NAMES = ("office", "residential", "hotel", "retail")

# (hour, value) knots; hour 0 and 24 values must match within an archetype
# so consecutive days join continuously.
_KNOTS: dict[str, dict[str, list[tuple[float, float]]]] = {
    "office": {
        "weekday": [(0, 0.25), (5, 0.25), (7, 0.50), (9, 0.95), (12, 1.00),
                    (16, 0.95), (18, 0.60), (21, 0.35), (24, 0.25)],
        "weekend": [(0, 0.25), (8, 0.30), (12, 0.40), (18, 0.30), (24, 0.25)],
    },
    "residential": {
        "weekday": [(0, 0.55), (4, 0.45), (7, 0.75), (9, 0.60), (14, 0.55),
                    (17, 0.80), (20, 1.00), (22, 0.85), (24, 0.55)],
        "weekend": [(0, 0.55), (4, 0.45), (9, 0.80), (11, 0.85), (14, 0.80),
                    (17, 0.85), (20, 1.00), (22, 0.85), (24, 0.55)],
    },
    "hotel": {
        "weekday": [(0, 0.80), (3, 0.70), (6, 0.85), (7.5, 0.95), (9, 0.80),
                    (12, 0.60), (15, 0.65), (18, 0.85), (21, 1.00),
                    (23, 0.90), (24, 0.80)],
        "weekend": [(0, 0.80), (3, 0.70), (6, 0.85), (7.5, 0.95), (9, 0.80),
                    (12, 0.60), (15, 0.65), (18, 0.85), (21, 1.00),
                    (23, 0.90), (24, 0.80)],
    },
    "retail": {
        "weekday": [(0, 0.15), (8, 0.20), (10, 0.85), (14, 0.95), (18, 0.90),
                    (20, 0.50), (22, 0.20), (24, 0.15)],
        "weekend": [(0, 0.15), (8, 0.20), (10, 0.90), (14, 1.00), (18, 0.90),
                    (20, 0.50), (22, 0.20), (24, 0.15)],
    },
}

# shape contract: peak knot exactly 1.0, endpoints match, values positive
for _name, _shapes in _KNOTS.items():
    _peak = max(v for knots in _shapes.values() for _, v in knots)
    assert _peak == 1.0, f"{_name}: peak knot must be exactly 1.0, got {_peak}"
    for _kind, _knots in _shapes.items():
        assert _knots[0][1] == _knots[-1][1], f"{_name}/{_kind}: day endpoints differ"
        assert all(v > 0.0 for _, v in _knots), f"{_name}/{_kind}: non-positive knot"


def profile_knots(archetype: str) -> dict[str, list[tuple[float, float]]]:
    """The (hour, value) knots of one archetype per day kind — the public
    read-only view the sampler's knot-jitter draw sizes against."""
    if archetype not in _KNOTS:
        raise KeyError(f"unknown archetype {archetype!r}; choose from {ARCHETYPE_NAMES}")
    return {kind: list(knots) for kind, knots in _KNOTS[archetype].items()}


def apply_knot_factors(
    archetype: str, factors: dict[str, tuple[float, ...]]
) -> dict[str, list[tuple[float, float]]]:
    """Perturbed knots for one archetype: value per knot
    times its factor, then every value divided by the new weekly peak so
    the profile peaks at exactly 1.0 again (flow equivalence keeps design
    load the weekly maximum).

    Args:
        archetype: one of :data:`ARCHETYPE_NAMES`.
        factors: day-kind -> one factor per knot (the sampler's
            ``knot_jitter`` row field). The day-join endpoints (hour 0/24
            of both kinds) must carry one shared factor.

    Raises:
        ValueError: on a factor-count mismatch or when the factored knots
            break the day-join contract (endpoint values differ).
    """
    base = profile_knots(archetype)
    scaled: dict[str, list[tuple[float, float]]] = {}
    for kind, knots in base.items():
        f = factors[kind]
        if len(f) != len(knots):
            raise ValueError(
                f"{archetype}/{kind}: {len(f)} factors for {len(knots)} knots"
            )
        scaled[kind] = [(h, v * x) for (h, v), x in zip(knots, f)]
        if scaled[kind][0][1] != scaled[kind][-1][1]:
            raise ValueError(
                f"{archetype}/{kind}: factored endpoints differ — the hour-0/24 "
                f"factors must be tied so days keep joining"
            )
    if scaled["weekday"][0][1] != scaled["weekend"][0][1]:
        raise ValueError(
            f"{archetype}: weekday and weekend endpoint values differ after "
            f"factoring — the day-join tie spans both kinds, else "
            f"the Friday->Saturday join jumps"
        )
    peak = max(v for knots in scaled.values() for _, v in knots)
    return {
        kind: [(h, v / peak) for h, v in knots] for kind, knots in scaled.items()
    }


def normalized_profile(
    archetype: str,
    times: np.ndarray,
    start_day: int = 0,
    knots: dict[str, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Normalized load shape in (0, 1] at the given times.

    Args:
        archetype: one of :data:`ARCHETYPE_NAMES`.
        times: seconds from scenario start (any sampling).
        start_day: weekday of time zero, 0 = Monday ... 6 = Sunday.
        knots: optional replacement knot table (e.g. from
            :func:`apply_knot_factors`); ``None`` uses the canonical shape.
    """
    if archetype not in _KNOTS:
        raise KeyError(f"unknown archetype {archetype!r}; choose from {ARCHETYPE_NAMES}")
    table = _KNOTS[archetype] if knots is None else knots
    hours_of_week = (np.asarray(times, dtype=float) / 3600.0 + start_day * 24.0) % 168.0
    day = (hours_of_week // 24.0).astype(int)
    hour = hours_of_week % 24.0

    def interp(kind: str) -> np.ndarray:
        xp = np.array([h for h, _ in table[kind]])
        fp = np.array([v for _, v in table[kind]])
        return np.interp(hour, xp, fp)

    return np.where(day >= 5, interp("weekend"), interp("weekday"))
