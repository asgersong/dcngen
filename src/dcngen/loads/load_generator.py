"""Per-consumer cooling-load time series.

Composes ``Q_design × normalized archetype profile × lognormal AR(1)
noise`` per consumer. Magnitudes come from flow equivalence via
``design_loads``; archetypes shape only the normalized profile.

Noise: multiplier ``exp(x_t - s²/2)`` with
``x_t`` a stationary Gaussian AR(1), ``phi = exp(-Δt/τ)`` per step and
``s² = ln(1 + σ²)`` — mean exactly 1, relative std exactly ``σ``, strictly
positive by construction, and the log-multiplier autocorrelation decays as
``exp(-lag/τ)``. Consumers get independent paths, drawn in sorted-id order
so the same seed gives the same loads regardless of mapping order.
"""

from collections.abc import Callable, Mapping

import numpy as np

from dcngen.config import LoadModel
from dcngen.loads.archetypes import apply_knot_factors, normalized_profile

YEAR_SECONDS = 365.0 * 24.0 * 3600.0  # [s] 365-day year, no leap handling


def _seasonal(times: np.ndarray, amplitude: float, phase: float = 0.0) -> np.ndarray:
    """Annual multiplier in [1 - amplitude, 1].

    Args:
        times: seconds from scenario start.
        amplitude: dimensionless seasonal depth in [0, 1].
        phase: annual-cycle offset [s] of the scenario start (0 = start at
            the peak; the long-horizon tier samples it — no synthetic
            calendar exists).

    Scoops downward only: flow equivalence anchors the profile
    peak to the DiTEC hydraulic regime, so design load stays the annual
    maximum for any amplitude or phase. A real weather phase comes with
    the weather hook later.
    """
    if amplitude == 0.0:
        return np.ones_like(times)
    return 1.0 - amplitude * (1.0 - np.cos(2.0 * np.pi * (times + phase) / YEAR_SECONDS)) / 2.0


def _lognormal_ar1(
    times: np.ndarray,
    sigma: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(len(sigma), len(times)) mean-one multipliers; uneven sampling ok.

    Args:
        times: seconds from scenario start.
        sigma: dimensionless relative std of the multiplier, one per path
            (zeros give exactly-one multipliers on their path).
        tau: autocorrelation time [s].
    """
    n, n_paths = len(times), len(sigma)
    if n == 0 or not np.any(sigma):
        return np.ones((n_paths, n))
    s2 = np.log1p(sigma * sigma)[:, None]  # per path
    s = np.sqrt(s2)
    phi = np.exp(-np.diff(times) / tau)
    eps = rng.standard_normal((n_paths, n))
    x = np.empty((n_paths, n))
    # sequential in time (AR recursion), vectorised over consumers; the
    # cumprod closed form underflows for horizons >> tau, so keep the loop
    x[:, 0] = s[:, 0] * eps[:, 0]
    for k in range(1, n):
        x[:, k] = phi[k - 1] * x[:, k - 1] + s[:, 0] * np.sqrt(1.0 - phi[k - 1] ** 2) * eps[:, k]
    return np.exp(x - s2 / 2.0)


def generate_loads(
    archetypes: Mapping[str, str],
    design_loads: Mapping[str, float],
    times: np.ndarray,
    cfg: LoadModel,
    rng: np.random.Generator,
    start_day: int = 0,
    weather: Callable[[np.ndarray], np.ndarray] | None = None,
    noise_sigma: Mapping[str, float] | None = None,
    knot_factors: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
    seasonal_amplitude: float | None = None,
    seasonal_phase: float = 0.0,
) -> dict[str, np.ndarray]:
    """Cooling load [W] per consumer at the given times.

    The optional row-driven overrides let a scenario-plan row
    replace the config's scalar knobs: per-consumer noise sigma, the
    perturbed archetype knots, and the long-horizon seasonal draw.

    Args:
        archetypes: consumer junction id -> archetype name.
        design_loads: consumer junction id -> design cooling load [W]
            (flow-equivalent).
        times: seconds from scenario start (any sampling).
        cfg: load-model parameters (noise, seasonal knob).
        rng: seeded generator; the sole source of randomness.
        start_day: weekday of time zero, 0 = Monday ... 6 = Sunday.
        weather: optional times -> multiplier shared across consumers.
            Injection point for future weather coupling (locked decision 5);
            no implementation ships with it.
        noise_sigma: per-consumer sigma overriding ``cfg.noise_sigma``;
            must key exactly the consumers.
        knot_factors: archetype -> day-kind -> per-knot factors (the plan
            row's ``knot_jitter``), applied via
            :func:`archetypes.apply_knot_factors`; archetypes absent from
            the mapping keep their canonical shape.
        seasonal_amplitude: overrides ``cfg.seasonal_amplitude``.
        seasonal_phase: annual-cycle offset [s] of the scenario start.
    """
    if sorted(archetypes) != sorted(design_loads):
        raise ValueError("archetypes and design_loads must key the same consumers")
    times = np.asarray(times, dtype=float)
    consumers = sorted(archetypes)
    if noise_sigma is not None and sorted(noise_sigma) != consumers:
        raise ValueError("noise_sigma must key exactly the consumers")
    amplitude = cfg.seasonal_amplitude if seasonal_amplitude is None else seasonal_amplitude
    seasonal = _seasonal(times, amplitude, seasonal_phase)
    if weather is not None:
        seasonal = seasonal * np.asarray(weather(times), dtype=float)
    sigmas = np.array(
        [cfg.noise_sigma if noise_sigma is None else noise_sigma[j] for j in consumers]
    )
    noise = _lognormal_ar1(times, sigmas, cfg.ar1_correlation_time, rng)
    knots = {
        a: apply_knot_factors(a, factors)
        for a, factors in (knot_factors or {}).items()
    }
    return {
        j: design_loads[j]
        * normalized_profile(archetypes[j], times, start_day, knots.get(archetypes[j]))
        * seasonal
        * noise[i]
        for i, j in enumerate(consumers)
    }
