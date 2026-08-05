"""Taxonomy seam: fault kinds, onset envelopes, labels.

Envelopes are severity fractions in [0, 1] over scenario time (seconds from
start, same convention as the load generator); the mask is envelope > 0.
Expected values are worked examples from the envelope definitions (abrupt =
indicator, incipient = linear ramp then hold, intermittent = duty cycling),
not re-derived through the implementation.
"""

import numpy as np
import pytest

from dcngen.faults.taxonomy import FaultKind, FaultLabel, OnsetProfile

DT = 600.0
HORIZON = 24 * 3600.0
TIMES = np.arange(0.0, HORIZON, DT)


def test_kinds_cover_the_planned_taxonomy():
    # every planned fault kind + an explicit none for clean-normal
    # scenarios; bypass split out of the ETS/valve family as its own
    # dataset class
    names = {k.name.lower() for k in FaultKind}
    assert names == {
        "none", "leak", "pump", "sensor", "ets_valve", "bypass",
        "fouling", "blockage", "plant_setpoint",
    }


def test_abrupt_envelope_is_an_indicator_between_onset_and_offset():
    p = OnsetProfile.abrupt(onset=6 * 3600.0, offset=18 * 3600.0)
    e = p.envelope(TIMES)
    assert e[TIMES < 6 * 3600.0].max() == 0.0
    assert e[(TIMES >= 6 * 3600.0) & (TIMES < 18 * 3600.0)].min() == 1.0
    assert e[TIMES >= 18 * 3600.0].max() == 0.0
    # mask aligns exactly: first true step starts at onset, last before offset
    mask = p.mask(TIMES)
    assert TIMES[mask][0] == 6 * 3600.0
    assert TIMES[mask][-1] == 18 * 3600.0 - DT


def test_incipient_envelope_ramps_then_holds():
    p = OnsetProfile.incipient(onset=6 * 3600.0, offset=18 * 3600.0, ramp=4 * 3600.0)
    e = p.envelope(TIMES)

    def at(h):
        return float(e[int(h * 3600.0 / DT)])

    assert at(5.0) == 0.0
    assert at(8.0) == pytest.approx(0.5)  # halfway up the 4 h ramp
    assert at(10.0) == pytest.approx(1.0)  # ramp done
    assert at(17.5) == pytest.approx(1.0)  # holds to offset
    assert at(18.0) == 0.0


def test_intermittent_envelope_cycles_at_the_duty_fraction():
    p = OnsetProfile.intermittent(
        onset=6 * 3600.0, offset=18 * 3600.0, period=2 * 3600.0, duty=0.5
    )
    e = p.envelope(TIMES)
    window = (TIMES >= 6 * 3600.0) & (TIMES < 18 * 3600.0)
    assert set(np.unique(e[window])) == {0.0, 1.0}  # on/off, no partial
    assert np.mean(e[window]) == pytest.approx(0.5, abs=0.05)
    assert e[~window].max() == 0.0
    # first hour of each period is on (phase starts at onset)
    def at(h):
        return float(e[int(h * 3600.0 / DT)])

    assert at(6.5) == 1.0 and at(7.5) == 0.0


def test_leak_label_carries_location_side_severity_mask_and_realized():
    profile = OnsetProfile.abrupt(onset=6 * 3600.0, offset=HORIZON)
    label = FaultLabel.from_profile(
        kind=FaultKind.LEAK, location="3", element="node", side="return",
        severity=0.05, profile=profile, times=TIMES,
    )
    assert label.kind is FaultKind.LEAK
    assert (label.location, label.element, label.side) == ("3", "node", "return")
    assert label.severity == 0.05
    assert label.onset == 6 * 3600.0 and label.offset == HORIZON
    np.testing.assert_array_equal(label.mask, TIMES >= 6 * 3600.0)

    realized = np.where(label.mask, 1.7e-3, 0.0)
    final = label.with_realized(leak_flow=realized)
    np.testing.assert_array_equal(final.realized["leak_flow"], realized)
    assert label.realized == {}  # frozen: original untouched


def test_clean_scenarios_get_an_explicit_no_fault_label():
    label = FaultLabel.no_fault(TIMES)
    assert label.kind is FaultKind.NONE
    assert label.location is None and label.side is None
    assert label.severity is None
    assert label.mask.shape == TIMES.shape and not label.mask.any()


def test_schema_routes_fouling_and_pump_faults_without_changes():
    """The other planned kinds fit the same label
    schema — fouling as a UA scale on a consumer node (incipient), a pump
    fault as a curve scale on the plant pump edge (intermittent) — with no
    schema extensions beyond kind-specific realized-series names."""
    fouling = FaultLabel.from_profile(
        kind=FaultKind.FOULING, location="7", element="node", side=None,
        severity=0.5,  # UA scale
        profile=OnsetProfile.incipient(onset=0.0, offset=HORIZON, ramp=12 * 3600.0),
        times=TIMES,
    ).with_realized(ua=np.full(TIMES.shape, 1.0e4))
    assert fouling.kind is FaultKind.FOULING
    assert fouling.realized["ua"].shape == TIMES.shape

    pump = FaultLabel.from_profile(
        kind=FaultKind.PUMP, location="plant_pump", element="edge", side=None,
        severity=0.8,  # curve scale
        profile=OnsetProfile.intermittent(
            onset=6 * 3600.0, offset=HORIZON, period=3600.0, duty=0.5
        ),
        times=TIMES,
    )
    assert pump.element == "edge" and pump.side is None


def test_profiles_validate_their_parameters():
    with pytest.raises(ValueError):
        OnsetProfile.abrupt(onset=10.0, offset=10.0)  # empty window
    with pytest.raises(ValueError):
        OnsetProfile.incipient(onset=0.0, offset=100.0, ramp=0.0)
    with pytest.raises(ValueError):
        OnsetProfile.intermittent(onset=0.0, offset=100.0, period=10.0, duty=1.5)
