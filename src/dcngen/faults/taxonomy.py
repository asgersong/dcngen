"""Fault taxonomy: kinds, onset envelopes, label schema.

One label schema covers every planned fault kind so later
fault types (fouling routes as a UA mapping, pump faults as curve scaling)
need no schema changes. Envelopes are severity fractions in [0, 1] over
scenario time [s from start]; the per-timestep mask is envelope > 0.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal

import numpy as np


class FaultKind(Enum):
    """Planned fault kinds + NONE for clean-normal.

    BYPASS is the supply->return short-circuit member of the ETS/valve
    family, split out as its own kind because it is a first-class dataset
    class (cause discrimination against FOULING inside low-dT syndrome);
    ETS_VALVE remains for the stuck/offset-FCV faults.
    """

    NONE = "none"
    LEAK = "leak"
    PUMP = "pump"
    SENSOR = "sensor"
    ETS_VALVE = "ets_valve"
    BYPASS = "bypass"
    FOULING = "fouling"
    BLOCKAGE = "blockage"
    PLANT_SETPOINT = "plant_setpoint"


@dataclass(frozen=True)
class OnsetProfile:
    """Severity envelope over scenario time.

    ``kind`` selects the shape; ``onset``/``offset`` [s] bound the active
    window (half-open, [onset, offset)); ``ramp`` [s] is the incipient
    rise time; ``period`` [s] and ``duty`` (fraction on) drive the
    intermittent cycle, phase-anchored at onset.
    """

    kind: Literal["abrupt", "incipient", "intermittent"]
    onset: float
    offset: float
    ramp: float = 0.0
    period: float = 0.0
    duty: float = 0.0

    def __post_init__(self) -> None:
        if not self.offset > self.onset:
            raise ValueError(f"offset ({self.offset}) must exceed onset ({self.onset})")
        if self.kind == "incipient" and not self.ramp > 0.0:
            raise ValueError("incipient onset needs ramp > 0")
        if self.kind == "intermittent":
            if not self.period > 0.0:
                raise ValueError("intermittent onset needs period > 0")
            if not 0.0 < self.duty < 1.0:
                raise ValueError(f"duty ({self.duty}) must be in (0, 1)")

    @classmethod
    def abrupt(cls, onset: float, offset: float) -> "OnsetProfile":
        """Step to full severity at onset, off at offset."""
        return cls(kind="abrupt", onset=onset, offset=offset)

    @classmethod
    def incipient(cls, onset: float, offset: float, ramp: float) -> "OnsetProfile":
        """Linear rise over ``ramp`` [s] from onset, then hold to offset."""
        return cls(kind="incipient", onset=onset, offset=offset, ramp=ramp)

    @classmethod
    def intermittent(
        cls, onset: float, offset: float, period: float, duty: float
    ) -> "OnsetProfile":
        """On/off cycling at ``period`` [s], on for the ``duty`` fraction."""
        return cls(
            kind="intermittent", onset=onset, offset=offset, period=period, duty=duty
        )

    def envelope(self, times: np.ndarray) -> np.ndarray:
        """Severity fraction in [0, 1] at each time [s from start]."""
        times = np.asarray(times, dtype=float)
        window = (times >= self.onset) & (times < self.offset)
        if self.kind == "abrupt":
            return window.astype(float)
        if self.kind == "incipient":
            rise = np.clip((times - self.onset) / self.ramp, 0.0, 1.0)
            return np.where(window, rise, 0.0)
        phase = (times - self.onset) % self.period
        return np.where(window & (phase < self.duty * self.period), 1.0, 0.0)

    def mask(self, times: np.ndarray) -> np.ndarray:
        """Per-timestep fault-active mask (envelope > 0)."""
        return self.envelope(times) > 0.0


@dataclass(frozen=True)
class FaultLabel:
    """Ground-truth label for one scenario; one schema for every kind.

    ``severity`` is the *commanded* magnitude in the kind's own unit (leak:
    fraction of network design flow; fouling: UA scale; ...); realized
    quantities land in ``realized`` as named per-timestep series (e.g.
    ``leak_flow`` [m3/s]). ``mask`` is the per-timestep fault-active flag;
    clean-normal scenarios carry the explicit :meth:`no_fault` label.
    """

    kind: FaultKind
    location: str | None  # original DiTEC element id
    element: Literal["node", "edge"] | None
    side: Literal["supply", "return"] | None
    severity: float | None
    onset: float | None  # [s from scenario start]
    offset: float | None
    mask: np.ndarray  # bool, per timestep
    realized: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_profile(
        cls,
        kind: FaultKind,
        location: str,
        element: Literal["node", "edge"],
        side: Literal["supply", "return"] | None,
        severity: float,
        profile: OnsetProfile,
        times: np.ndarray,
    ) -> "FaultLabel":
        """Label with onset/offset and mask taken from the profile."""
        return cls(
            kind=kind,
            location=location,
            element=element,
            side=side,
            severity=severity,
            onset=profile.onset,
            offset=profile.offset,
            mask=profile.mask(times),
        )

    @classmethod
    def no_fault(cls, times: np.ndarray) -> "FaultLabel":
        """The explicit clean-normal label (all-false mask)."""
        return cls(
            kind=FaultKind.NONE,
            location=None,
            element=None,
            side=None,
            severity=None,
            onset=None,
            offset=None,
            mask=np.zeros(np.asarray(times).shape, dtype=bool),
        )

    def with_realized(self, **series: np.ndarray) -> "FaultLabel":
        """A copy with named realized series added (original untouched)."""
        return replace(self, realized={**self.realized, **series})
