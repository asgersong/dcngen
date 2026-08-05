"""Load-generator seam: archetype shape × design load ×
seasonal knob × lognormal AR(1) noise × weather hook.

Contract: ``generate_loads(archetypes, design_loads, times,
cfg, rng, start_day=0, weather=None) -> {consumer_id: Q(t) [W]}``.
Archetype assignment is caller-supplied (the sampler owns it).
Magnitudes come from flow equivalence via ``design_loads``;
archetypes shape only the normalized profile.
"""

import dataclasses

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.loads.load_generator import generate_loads

DT = 600.0
DAY = 24 * 3600.0
WEEK = 7 * DAY

ARCHETYPES = {"c1": "office", "c2": "residential", "c3": "hotel"}
DESIGN_LOADS = {"c1": 500e3, "c2": 120e3, "c3": 2e6}  # [W]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def loads_cfg(cfg, **overrides):
    return dataclasses.replace(cfg.loads, **overrides)


def week_times() -> np.ndarray:
    return np.arange(0.0, WEEK, DT)


def noise_multipliers(cfg, times, seed=42, **overrides) -> dict[str, np.ndarray]:
    """Recover the noise multiplier through the seam: noisy / noiseless run."""
    noisy = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times, loads_cfg(cfg, **overrides),
        np.random.default_rng(seed),
    )
    quiet = {k: v for k, v in overrides.items() if k != "noise_sigma"}
    clean = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times,
        loads_cfg(cfg, noise_sigma=0.0, **quiet),
        np.random.default_rng(seed),
    )
    return {j: noisy[j] / clean[j] for j in noisy}


def lag_autocorr(x: np.ndarray, lag: int) -> float:
    d = np.log(x) - np.mean(np.log(x))
    return float(np.dot(d[:-lag], d[lag:]) / np.dot(d, d))


def test_noiseless_loads_scale_profile_to_design_load(cfg):
    """With noise off, each consumer's weekly peak is exactly its design
    load (normalized profiles peak at 1.0) and loads never reach zero."""
    t = week_times()
    loads = generate_loads(
        ARCHETYPES,
        DESIGN_LOADS,
        t,
        loads_cfg(cfg, noise_sigma=0.0),
        np.random.default_rng(0),
    )
    assert set(loads) == set(ARCHETYPES)
    for j, q in loads.items():
        assert q.shape == t.shape, j
        assert q.max() == pytest.approx(DESIGN_LOADS[j]), j
        assert q.min() > 0.0, j


def test_noise_is_ar1_around_the_configured_correlation_time(cfg):
    """The multiplicative noise is mean-one lognormal AR(1): relative std
    ~= noise_sigma, lag-1 autocorrelation ~= exp(-dt/tau) (far from white
    noise's ~0), decayed to ~= 1/e one correlation time later.

    Default config: tau = 3600 s, dt = 600 s -> phi = exp(-1/6) ~= 0.846.
    Four weeks of samples (4032 points/consumer) keeps estimator noise well
    inside the asserted bands for the fixed seed.
    """
    t = np.arange(0.0, 4 * WEEK, DT)
    phi = np.exp(-DT / cfg.loads.ar1_correlation_time)
    steps_per_tau = round(cfg.loads.ar1_correlation_time / DT)
    for j, m in noise_multipliers(cfg, t).items():
        assert m.min() > 0.0, j
        assert np.mean(m) == pytest.approx(1.0, abs=0.02), j
        assert np.std(m) == pytest.approx(cfg.loads.noise_sigma, abs=0.015), j
        assert lag_autocorr(m, 1) == pytest.approx(phi, abs=0.08), j
        assert lag_autocorr(m, 1) >= 0.5, j  # white noise would sit near 0
        assert lag_autocorr(m, steps_per_tau) == pytest.approx(np.exp(-1.0), abs=0.15), j


def test_same_seed_reproduces_identical_loads(cfg):
    t = week_times()

    def run(seed, archetypes=ARCHETYPES):
        return generate_loads(
            archetypes, DESIGN_LOADS, t, cfg.loads, np.random.default_rng(seed)
        )

    a, b = run(7), run(7)
    for j in ARCHETYPES:
        np.testing.assert_array_equal(a[j], b[j])

    reordered = run(7, dict(reversed(list(ARCHETYPES.items()))))
    for j in ARCHETYPES:
        np.testing.assert_array_equal(a[j], reordered[j])

    other = run(8)
    assert any(not np.array_equal(a[j], other[j]) for j in ARCHETYPES)


def test_seasonal_knob_defaults_flat_and_modulates_below_design_load(cfg):
    """Default seasonal_amplitude=0: the noiseless year is week-periodic.
    With amplitude a > 0 the annual envelope dips to (1-a) of design but the
    annual peak stays the design load — flow equivalence anchors
    peak flow to the DiTEC regime, so the knob only scoops downward."""
    year = np.arange(0.0, 52 * WEEK, DT)
    n_week = len(week_times())

    flat = generate_loads(
        ARCHETYPES, DESIGN_LOADS, year, loads_cfg(cfg, noise_sigma=0.0),
        np.random.default_rng(0),
    )["c1"]
    # week-periodic to float precision (hours-of-week wrap accumulates ulps)
    np.testing.assert_allclose(flat[:n_week], flat[30 * n_week : 31 * n_week], rtol=1e-9)

    a = 0.3
    seasonal = generate_loads(
        ARCHETYPES, DESIGN_LOADS, year,
        loads_cfg(cfg, noise_sigma=0.0, seasonal_amplitude=a),
        np.random.default_rng(0),
    )["c1"]
    # seasonal peak (t=0) and daily profile peak (noon) don't sample-align,
    # so the achieved annual peak sits a few ppm under design — never above
    assert seasonal.max() == pytest.approx(DESIGN_LOADS["c1"], rel=1e-4)
    weekly_peaks = seasonal[: 52 * n_week].reshape(52, n_week).max(axis=1)
    assert weekly_peaks.min() == pytest.approx((1 - a) * DESIGN_LOADS["c1"], rel=0.02)
    assert np.all(seasonal <= flat + 1e-9)  # scoop only, never boost


def test_weather_hook_applies_a_shared_multiplier(cfg):
    """Locked decision 5: weather coupling deferred, but the injection point
    exists now — a callable times -> multiplier applied to every consumer."""
    t = week_times()

    def run(weather=None):
        return generate_loads(
            ARCHETYPES, DESIGN_LOADS, t, loads_cfg(cfg, noise_sigma=0.0),
            np.random.default_rng(0), weather=weather,
        )

    base = run()
    halved = run(weather=lambda times: np.full(times.shape, 0.5))
    for j in ARCHETYPES:
        np.testing.assert_allclose(halved[j], 0.5 * base[j])


def test_consumers_get_independent_noise_paths(cfg):
    t = week_times()
    mult = noise_multipliers(cfg, t)
    pairs = [("c1", "c2"), ("c1", "c3"), ("c2", "c3")]
    for a, b in pairs:
        r = np.corrcoef(np.log(mult[a]), np.log(mult[b]))[0, 1]
        assert abs(r) < 0.2, (a, b)


def test_per_consumer_sigma_overrides_the_scalar(cfg):
    # the plan row carries sigma per consumer; a zero-sigma
    # consumer rides the exact clean profile while its neighbours vary
    sigma = {"c1": 0.0, "c2": 0.1, "c3": 0.3}
    times = week_times()
    loads = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times, loads_cfg(cfg),
        np.random.default_rng(3), noise_sigma=sigma,
    )
    clean = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times, loads_cfg(cfg, noise_sigma=0.0),
        np.random.default_rng(3),
    )
    np.testing.assert_array_equal(loads["c1"], clean["c1"])
    for j in ("c2", "c3"):
        assert not np.allclose(loads[j], clean[j])
    # larger sigma -> visibly wider relative spread
    spread = {j: np.std(loads[j] / clean[j]) for j in ("c2", "c3")}
    assert spread["c3"] > 2.0 * spread["c2"]


def test_per_consumer_sigma_must_cover_exactly_the_consumers(cfg):
    with pytest.raises(ValueError, match="noise_sigma"):
        generate_loads(
            ARCHETYPES, DESIGN_LOADS, week_times(), loads_cfg(cfg),
            np.random.default_rng(0), noise_sigma={"c1": 0.1},
        )


def test_seasonal_phase_shifts_the_scoop(cfg):
    # phase = half a year at full amplitude: the scenario starts at the
    # trough (multiplier 1 - a) instead of the peak (multiplier 1)
    times = np.arange(0.0, DAY, DT)
    base = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times,
        loads_cfg(cfg, noise_sigma=0.0, seasonal_amplitude=0.4),
        np.random.default_rng(0),
    )
    shifted = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times,
        loads_cfg(cfg, noise_sigma=0.0, seasonal_amplitude=0.4),
        np.random.default_rng(0), seasonal_phase=365.0 * DAY / 2.0,
    )
    # at t=0: base multiplier is exactly 1, shifted is exactly 1 - 0.4
    assert shifted["c1"][0] == pytest.approx(0.6 * base["c1"][0])
    # loads never exceed the design-anchored clean profile either way
    assert np.all(shifted["c1"] <= base["c1"] + 1e-9)


def test_knot_factors_reshape_the_profile_through_the_seam(cfg):
    from dcngen.loads.archetypes import profile_knots

    # damp the office weekday peak only; the other archetypes are untouched
    base = profile_knots("office")
    factors = {
        "office": {
            "weekday": tuple(
                0.8 if 0 < i < len(base["weekday"]) - 1 else 1.0
                for i in range(len(base["weekday"]))
            ),
            "weekend": tuple(1.0 for _ in base["weekend"]),
        }
    }
    times = week_times()
    plain = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times, loads_cfg(cfg, noise_sigma=0.0),
        np.random.default_rng(0),
    )
    shaped = generate_loads(
        ARCHETYPES, DESIGN_LOADS, times, loads_cfg(cfg, noise_sigma=0.0),
        np.random.default_rng(0), knot_factors=factors,
    )
    assert not np.allclose(shaped["c1"], plain["c1"])
    np.testing.assert_array_equal(shaped["c2"], plain["c2"])
    np.testing.assert_array_equal(shaped["c3"], plain["c3"])
    # renormalisation keeps the weekly peak pinned to the design load
    assert shaped["c1"].max() == pytest.approx(DESIGN_LOADS["c1"])
