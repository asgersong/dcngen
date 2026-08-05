"""Fault injection: the abrupt junction leak and the two ETS low-dT
faults — chronic fouling and primary bypass.

Leak severity is the target leak flow as a fraction of the network design
flow (the plant pump's design flow); the orifice area is back-computed from
the nominal local pressure at the leak node — a healthy solve at design
consumer flows — via ``Q = C_d · A · sqrt(2 g h)`` (WNTR's leak model).
The realized leak flow then floats with the actual pressure; the paired
acceptance check bounds it against the commanded target.

Fouling severity is the UA multiplier m (applied by the caller on the
installed UA, ``thermal.ets.installed_ua``), always whole-scenario; bypass
severity is the short-circuit fraction f consumed by the scenario runner's
ETS closure, in either temporal variant (abrupt mid-horizon onset or
whole-horizon). Severity bands and the variant split live in
``cfg.faults``.

Abrupt onset is drawn as a step index inside the configured horizon window,
so the label's onset is exactly a step start and both window bounds hold;
whole-horizon faults carry onset 0 with an all-true mask.
"""

import math
from dataclasses import dataclass

import numpy as np

from dcngen.config import Config
from dcngen.faults.taxonomy import FaultKind, FaultLabel, OnsetProfile
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.topology.dcnifier import DCNetwork

_G = 9.81  # [m/s2] gravitational acceleration; matches WNTR's leak model


@dataclass(frozen=True)
class LeakInjection:
    """One abrupt leak, ready for the scenario runner's ``leak=`` hook."""

    label: FaultLabel
    node: str  # WNTR node name (supply- or return-side twin)
    area: float  # [m2] orifice area
    discharge_coeff: float
    target_flow: float  # [m3/s] commanded leak flow at nominal pressure


@dataclass(frozen=True)
class FoulingInjection:
    """Chronic HX fouling at one consumer: scale its installed UA by
    ``multiplier`` for the whole scenario (the caller composes it via
    ``thermal.ets.installed_ua``); the label records the ground truth."""

    label: FaultLabel
    consumer: str  # original DiTEC junction id hosting the ETS
    multiplier: float  # UA multiplier m in (0, 1); == label.severity


@dataclass(frozen=True)
class BypassInjection:
    """One ETS bypass, ready for the scenario runner's ``bypass=`` hook."""

    label: FaultLabel
    consumer: str  # original DiTEC junction id hosting the ETS
    fraction: float  # short-circuit fraction f; == label.severity
    whole_horizon: bool  # temporal variant (False = abrupt onset)


def _horizon(times: np.ndarray) -> float:
    """Scenario horizon [s]: the end of the last step of a uniform grid."""
    return float(times[-1]) + float(times[1] - times[0])


def _abrupt_onset_time(
    times: np.ndarray,
    cfg: Config,
    rng: np.random.Generator | None,
    onset: float | None,
) -> float:
    """Onset [s]: the explicit row value (must sit on the step grid), or a
    draw from the configured fraction-of-horizon window snapped to a start
    on the uniform step grid ``times`` [s]."""
    if onset is not None:
        if not np.any(times == onset):
            raise ValueError(f"onset ({onset}) must be a step start on the grid")
        return float(onset)
    if rng is None:
        raise ValueError("rng required to sample an onset (or pass onset)")
    dt = float(times[1] - times[0])
    horizon = _horizon(times)
    k_lo = math.ceil(cfg.faults.onset_window_start * horizon / dt)
    k_hi = math.floor(cfg.faults.onset_window_end * horizon / dt)
    return float(times[int(rng.integers(k_lo, k_hi + 1))])


def inject_abrupt_leak(
    dcn: DCNetwork,
    cfg: Config,
    times: np.ndarray,
    rng: np.random.Generator | None,
    junction: str,
    side: str,
    severity: float | None = None,
    onset: float | None = None,
) -> LeakInjection:
    """Build an abrupt junction leak with its ground-truth label.

    Args:
        dcn: built DCN.
        cfg: configuration (severity range, onset window, orifice C_d).
        times: scenario step-start times [s], uniform sampling.
        rng: seeded generator; may be ``None`` when both ``severity`` and
            ``onset`` are given (plan-row injection).
        junction: original DiTEC junction id hosting the leak.
        side: ``"supply"`` or ``"return"`` — which twin node leaks.
        severity: leak flow as a fraction of network design flow; ``None``
            samples uniformly from the configured range.
        onset: explicit onset [s] on the step grid; ``None`` samples from
            the configured window.
    """
    if side not in ("supply", "return"):
        raise ValueError(f"side must be 'supply' or 'return', got {side!r}")
    node = dcn.pairing[junction][0 if side == "supply" else 1]

    f = cfg.faults
    if severity is None:
        if rng is None:
            raise ValueError("rng required to sample a severity (or pass severity)")
        severity = float(rng.uniform(f.leak_severity_min, f.leak_severity_max))

    times = np.asarray(times, dtype=float)
    onset = _abrupt_onset_time(times, cfg, rng, onset)
    profile = OnsetProfile.abrupt(onset=onset, offset=_horizon(times))

    target_flow = severity * dcn.pump_design[0]
    design_flows = {j: c.design_flow for j, c in dcn.consumers.items()}
    h_nominal = float(solve_steady(dcn, consumer_flows=design_flows).pressure[node])
    area = target_flow / (f.leak_discharge_coeff * math.sqrt(2.0 * _G * h_nominal))

    label = FaultLabel.from_profile(
        kind=FaultKind.LEAK,
        location=junction,
        element="node",
        side=side,
        severity=severity,
        profile=profile,
        times=times,
    )
    return LeakInjection(
        label=label,
        node=node,
        area=area,
        discharge_coeff=f.leak_discharge_coeff,
        target_flow=target_flow,
    )


def inject_fouling(
    dcn: DCNetwork,
    cfg: Config,
    times: np.ndarray,
    rng: np.random.Generator | None,
    consumer: str,
    severity: float | None = None,
) -> FoulingInjection:
    """Build a chronic-fouling fault with its ground-truth label.

    Fouling is always whole-scenario: the label is active
    from t = 0 to the horizon end with onset 0. The caller applies the
    multiplier at ETS construction (``installed_ua(..., fouling=m)``).

    Args:
        dcn: built DCN; ``consumer`` must host an ETS.
        cfg: configuration (severity band).
        times: scenario step-start times [s], uniform sampling.
        rng: seeded generator; may be ``None`` when ``severity`` is given
            (plan-row injection).
        consumer: original DiTEC junction id of the fouled ETS.
        severity: UA multiplier m in (0, 1); ``None`` samples uniformly
            from the configured band.
    """
    if consumer not in dcn.consumers:
        raise KeyError(f"{consumer!r} is not a consumer junction of {dcn.name}")
    if severity is None:
        if rng is None:
            raise ValueError("rng required to sample a severity (or pass severity)")
        severity = float(
            rng.uniform(cfg.faults.fouling_severity_min, cfg.faults.fouling_severity_max)
        )
    elif not 0.0 < severity < 1.0:
        raise ValueError(
            f"fouling severity ({severity}) must be in (0, 1): m = 1 is the "
            f"healthy exchanger"
        )
    times = np.asarray(times, dtype=float)
    label = FaultLabel.from_profile(
        kind=FaultKind.FOULING,
        location=consumer,
        element="node",
        side=None,
        severity=severity,
        profile=OnsetProfile.abrupt(onset=0.0, offset=_horizon(times)),
        times=times,
    )
    return FoulingInjection(label=label, consumer=consumer, multiplier=severity)


def inject_bypass(
    dcn: DCNetwork,
    cfg: Config,
    times: np.ndarray,
    rng: np.random.Generator | None,
    consumer: str,
    fraction: float | None = None,
    whole_horizon: bool | None = None,
    onset: float | None = None,
) -> BypassInjection:
    """Build an ETS-bypass fault with its ground-truth label.

    Both temporal variants: ``whole_horizon=True`` (bypass open since
    commissioning; onset 0, all-true mask) or ``False`` (valve failure
    mid-operation; abrupt onset from ``onset`` or drawn from the configured
    window). ``None`` draws the variant at
    ``cfg.faults.bypass_whole_horizon_prob``.

    Args:
        dcn: built DCN; ``consumer`` must host an ETS.
        cfg: configuration (fraction band, variant split, onset window).
        times: scenario step-start times [s], uniform sampling.
        rng: seeded generator; may be ``None`` when the fraction, variant,
            and (for the abrupt variant) onset are all given (plan-row
            injection).
        consumer: original DiTEC junction id of the bypassed ETS.
        fraction: short-circuit fraction f in (0, 1); ``None`` samples
            uniformly from the configured band.
        whole_horizon: temporal variant; ``None`` draws it.
        onset: explicit abrupt onset [s] on the step grid; ``None`` samples
            from the configured window. Ignored when whole-horizon.
    """
    if consumer not in dcn.consumers:
        raise KeyError(f"{consumer!r} is not a consumer junction of {dcn.name}")
    f = cfg.faults
    if fraction is None:
        if rng is None:
            raise ValueError("rng required to sample a fraction (or pass fraction)")
        fraction = float(rng.uniform(f.bypass_fraction_min, f.bypass_fraction_max))
    elif not 0.0 < fraction < 1.0:
        raise ValueError(
            f"bypass fraction ({fraction}) must be in (0, 1): f = 0 is the "
            f"healthy crossover, f = 1 starves the exchanger"
        )
    if whole_horizon is None:
        if rng is None:
            raise ValueError("rng required to draw the variant (or pass whole_horizon)")
        whole_horizon = bool(rng.random() < f.bypass_whole_horizon_prob)
    times = np.asarray(times, dtype=float)
    onset = 0.0 if whole_horizon else _abrupt_onset_time(times, cfg, rng, onset)
    label = FaultLabel.from_profile(
        kind=FaultKind.BYPASS,
        location=consumer,
        element="node",
        side=None,
        severity=fraction,
        profile=OnsetProfile.abrupt(onset=onset, offset=_horizon(times)),
        times=times,
    )
    return BypassInjection(
        label=label, consumer=consumer, fraction=fraction, whole_horizon=whole_horizon
    )
