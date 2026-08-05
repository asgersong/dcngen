"""Archetype seam: five normalized cooling-load shapes.

Assertions are structural literals about building operation (when each
archetype peaks/idles and how weekends differ) — the shapes themselves are
stylized, so tests pin the *qualitative* physics the dataset
depends on, not exact knot values.

Time convention: ``times`` are seconds from scenario start; ``start_day``
0 = Monday. Values in (0, 1]; each archetype's weekly maximum is 1.
"""

import numpy as np
import pytest

from dcngen.loads.archetypes import ARCHETYPE_NAMES, normalized_profile

DT = 600.0
WEEK = 7 * 24 * 3600.0


def week_times() -> np.ndarray:
    return np.arange(0.0, WEEK, DT)


def at(profile: np.ndarray, day: int, hour: float) -> float:
    """Profile value on day (0=Mon) at clock hour, for DT sampling."""
    idx = int((day * 24.0 + hour) * 3600.0 / DT)
    return float(profile[idx])


@pytest.fixture(scope="module")
def profiles():
    t = week_times()
    return {name: normalized_profile(name, t, start_day=0) for name in ARCHETYPE_NAMES}


def test_four_archetypes_exist():
    # datacentre removed from the dataset scope 2026-07-21
    assert set(ARCHETYPE_NAMES) == {"office", "residential", "hotel", "retail"}


def test_profiles_are_normalized_and_positive(profiles):
    for name, p in profiles.items():
        assert p.max() == pytest.approx(1.0), name
        assert p.min() > 0.0, name  # buildings never drop to zero cooling


def test_office_peaks_weekday_business_hours(profiles):
    p = profiles["office"]
    assert at(p, 1, 14.0) >= 0.9  # Tuesday afternoon: near peak
    assert at(p, 1, 3.0) <= 0.35  # Tuesday night: baseload
    assert at(p, 5, 14.0) <= 0.5 * at(p, 1, 14.0)  # Saturday: mostly empty


def test_residential_peaks_in_the_evening(profiles):
    p = profiles["residential"]
    assert at(p, 1, 20.0) > at(p, 1, 14.0)  # evening above afternoon
    assert at(p, 1, 20.0) >= 0.85
    assert at(p, 5, 11.0) > at(p, 1, 11.0)  # weekend late morning at home


def test_hotel_has_morning_and_evening_occupancy_bumps(profiles):
    p = profiles["hotel"]
    assert at(p, 1, 7.5) > at(p, 1, 12.0)  # morning bump above midday dip
    assert at(p, 1, 21.0) > at(p, 1, 12.0)  # evening bump above midday dip
    assert at(p, 1, 21.0) >= 0.85
    # hotels run seven days: deliberately NO weekday/weekend contrast
    assert at(p, 5, 21.0) == pytest.approx(at(p, 1, 21.0), abs=1e-12)
    assert at(p, 6, 7.5) == pytest.approx(at(p, 1, 7.5), abs=1e-12)


def test_retail_runs_business_hours_including_weekends(profiles):
    p = profiles["retail"]
    assert at(p, 5, 14.0) >= 0.9  # Saturday afternoon: peak trade
    assert at(p, 1, 3.0) <= 0.3  # closed overnight
    assert at(p, 5, 14.0) >= at(p, 1, 14.0)  # weekend at least as busy


def test_unknown_archetype_is_rejected():
    # also pins the datacentre removal: it must not silently reappear
    with pytest.raises(KeyError):
        normalized_profile("datacentre", week_times())


def test_archetypes_are_mutually_distinct(profiles):
    names = sorted(profiles)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            mean_gap = float(np.mean(np.abs(profiles[a] - profiles[b])))
            assert mean_gap >= 0.05, (a, b)


def test_profiles_are_smooth_at_fine_dt(profiles):
    for name, p in profiles.items():
        assert float(np.max(np.abs(np.diff(p)))) <= 0.1, name


def test_apply_knot_factors_scales_and_renormalises_to_weekly_peak_one():
    # knots x factors, then weekly-peak renormalisation.
    # Damping the office weekday midday peak by 0.9 makes the weekend's
    # unchanged knots the relative winners; the new weekly peak is 1 again.
    from dcngen.loads.archetypes import apply_knot_factors, profile_knots

    base = profile_knots("office")
    factors = {
        "weekday": tuple(
            0.9 if 0 < i < len(base["weekday"]) - 1 else 1.0
            for i in range(len(base["weekday"]))
        ),
        "weekend": tuple(1.0 for _ in base["weekend"]),
    }
    jittered = apply_knot_factors("office", factors)

    # weekly peak is 1.0 and hours are untouched
    peak = max(v for knots in jittered.values() for _, v in knots)
    assert peak == 1.0
    for kind in ("weekday", "weekend"):
        assert [h for h, _ in jittered[kind]] == [h for h, _ in base[kind]]
    # closed form: the damped office weekday peak (1.00 x 0.9 at 12 h) is
    # still the weekly winner, so every value renormalises by 0.9 — the
    # untouched weekend midday knot 0.40 lands at 0.40 / 0.9
    weekend_midday = dict(jittered["weekend"])[12]
    assert weekend_midday == pytest.approx(0.40 / 0.9)


def test_apply_knot_factors_rejects_shape_or_continuity_breaks():
    from dcngen.loads.archetypes import apply_knot_factors, profile_knots

    base = profile_knots("office")
    with pytest.raises(ValueError):  # wrong factor count
        apply_knot_factors("office", {
            "weekday": (1.0,), "weekend": tuple(1.0 for _ in base["weekend"]),
        })
    # untied endpoints would break the day join
    bad = {
        "weekday": tuple(
            1.1 if i == 0 else 1.0 for i in range(len(base["weekday"]))
        ),
        "weekend": tuple(1.0 for _ in base["weekend"]),
    }
    with pytest.raises(ValueError):
        apply_knot_factors("office", bad)
    # tied within each kind but not across kinds: the Friday->Saturday
    # join would jump
    cross = {
        "weekday": tuple(
            1.1 if i in (0, len(base["weekday"]) - 1) else 1.0
            for i in range(len(base["weekday"]))
        ),
        "weekend": tuple(1.0 for _ in base["weekend"]),
    }
    with pytest.raises(ValueError, match="both kinds"):
        apply_knot_factors("office", cross)


def test_normalized_profile_accepts_custom_knots():
    # the override is what lets a scenario's jittered shape drive the loads
    from dcngen.loads.archetypes import (
        apply_knot_factors,
        normalized_profile,
        profile_knots,
    )

    base = profile_knots("residential")
    factors = {
        kind: tuple(
            1.05 if 0 < i < len(base[kind]) - 1 else 1.0
            for i in range(len(base[kind]))
        )
        for kind in ("weekday", "weekend")
    }
    jittered = apply_knot_factors("residential", factors)
    times = np.arange(0, 86400.0, 600.0)

    plain = normalized_profile("residential", times)
    custom = normalized_profile("residential", times, knots=jittered)
    assert not np.allclose(plain, custom)
    # closed form at an interior knot: hour 4 weekday value 0.45 scaled
    # 1.05 then renormalised by the new peak (1.00 * 1.05 at 20 h)
    k = int(4 * 3600 / 600)
    assert custom[k] == pytest.approx(0.45 * 1.05 / 1.05)
