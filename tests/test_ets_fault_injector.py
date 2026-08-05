"""Injector seam: the two ETS low-dT faults. Chronic fouling is
a whole-horizon UA multiplier m; bypass is a primary short-circuit fraction
f in both temporal variants (abrupt mid-horizon onset / whole-horizon).
Labels ride the one shared schema: class, consumer id, severity, onset (0 =
whole-horizon), per-step mask.
"""

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.faults.injector import inject_bypass, inject_fouling
from dcngen.faults.taxonomy import FaultKind
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()

N_STEPS = 36
TIMES = np.arange(N_STEPS) * cfg.solver.dt
HORIZON = N_STEPS * cfg.solver.dt


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def test_fouling_is_whole_horizon_with_sampled_multiplier(mini_dcn):
    def draw(seed):
        return inject_fouling(
            mini_dcn, cfg, TIMES, np.random.default_rng(seed), consumer="2"
        )

    a, b = draw(5), draw(5)
    assert a.label.severity == b.label.severity  # rng-drawn: bitwise
    assert cfg.faults.fouling_severity_min <= a.label.severity
    assert a.label.severity <= cfg.faults.fouling_severity_max
    assert a.multiplier == a.label.severity
    assert draw(6).label.severity != a.label.severity

    label = a.label
    assert label.kind is FaultKind.FOULING
    assert (label.location, label.element, label.side) == ("2", "node", None)
    # chronic: active over the whole horizon, onset at t = 0
    assert (label.onset, label.offset) == (0.0, HORIZON)
    assert label.mask.shape == (N_STEPS,) and label.mask.all()


def test_fouling_accepts_explicit_severity(mini_dcn):
    inj = inject_fouling(
        mini_dcn, cfg, TIMES, np.random.default_rng(0), consumer="4", severity=0.42
    )
    assert inj.label.severity == 0.42
    assert inj.consumer == "4"


def test_bypass_whole_horizon_variant(mini_dcn):
    inj = inject_bypass(
        mini_dcn, cfg, TIMES, np.random.default_rng(3),
        consumer="2", whole_horizon=True,
    )
    label = inj.label
    assert label.kind is FaultKind.BYPASS
    assert (label.location, label.element, label.side) == ("2", "node", None)
    assert cfg.faults.bypass_fraction_min <= label.severity
    assert label.severity <= cfg.faults.bypass_fraction_max
    assert inj.fraction == label.severity
    assert inj.whole_horizon
    assert (label.onset, label.offset) == (0.0, HORIZON)
    assert label.mask.all()


def test_bypass_abrupt_variant_onsets_inside_the_config_window(mini_dcn):
    inj = inject_bypass(
        mini_dcn, cfg, TIMES, np.random.default_rng(3),
        consumer="2", whole_horizon=False,
    )
    label = inj.label
    assert not inj.whole_horizon
    assert cfg.faults.onset_window_start * HORIZON <= label.onset
    assert label.onset <= cfg.faults.onset_window_end * HORIZON
    assert label.onset in TIMES  # snapped to a step start
    assert label.offset == HORIZON  # persists to horizon end
    np.testing.assert_array_equal(label.mask, TIMES >= label.onset)


def test_bypass_variant_draw_is_deterministic_and_covers_both(mini_dcn):
    def draw(seed):
        return inject_bypass(
            mini_dcn, cfg, TIMES, np.random.default_rng(seed), consumer="4"
        )

    a, b = draw(11), draw(11)
    assert a.whole_horizon == b.whole_horizon
    assert a.label.severity == b.label.severity
    assert a.label.onset == b.label.onset

    variants = {draw(seed).whole_horizon for seed in range(20)}
    assert variants == {True, False}  # the 50/50 variant split is reachable


def test_explicit_fraction_with_drawn_variant(mini_dcn):
    inj = inject_bypass(
        mini_dcn, cfg, TIMES, np.random.default_rng(0), consumer="2", fraction=0.25
    )
    assert inj.label.severity == 0.25
    assert inj.fraction == 0.25


def test_unknown_consumer_rejected(mini_dcn):
    with pytest.raises(KeyError):
        inject_fouling(mini_dcn, cfg, TIMES, np.random.default_rng(0), consumer="99")
    with pytest.raises(KeyError):
        inject_bypass(mini_dcn, cfg, TIMES, np.random.default_rng(0), consumer="99")


def test_unphysical_explicit_severities_rejected(mini_dcn):
    # ground-truth labels must never claim impossible severities: m = 1 is
    # the healthy exchanger, f outside (0, 1) breaks the mixing closure
    rng = np.random.default_rng(0)
    for m in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            inject_fouling(mini_dcn, cfg, TIMES, rng, consumer="2", severity=m)
    for f in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            inject_bypass(mini_dcn, cfg, TIMES, rng, consumer="2", fraction=f)


def test_row_specified_injections_need_no_rng(mini_dcn):
    # the plan row carries every draw, so fully-specified
    # injections run with rng=None
    fouling = inject_fouling(mini_dcn, cfg, TIMES, None, consumer="2", severity=0.5)
    assert fouling.label.severity == 0.5

    onset = float(TIMES[12])
    bypass = inject_bypass(
        mini_dcn, cfg, TIMES, None, consumer="4",
        fraction=0.25, whole_horizon=False, onset=onset,
    )
    assert bypass.label.onset == onset
    assert not bypass.whole_horizon
    np.testing.assert_array_equal(bypass.label.mask, TIMES >= onset)

    whole = inject_bypass(
        mini_dcn, cfg, TIMES, None, consumer="4", fraction=0.25, whole_horizon=True,
    )
    assert whole.label.onset == 0.0 and whole.label.mask.all()


def test_missing_draws_without_rng_are_rejected(mini_dcn):
    with pytest.raises(ValueError, match="rng"):
        inject_fouling(mini_dcn, cfg, TIMES, None, consumer="2")
    with pytest.raises(ValueError, match="rng"):
        inject_bypass(mini_dcn, cfg, TIMES, None, consumer="2", fraction=0.2)
    with pytest.raises(ValueError, match="rng"):
        inject_bypass(
            mini_dcn, cfg, TIMES, None, consumer="2",
            fraction=0.2, whole_horizon=False,
        )


def test_explicit_onset_must_sit_on_the_step_grid(mini_dcn):
    with pytest.raises(ValueError, match="onset"):
        inject_bypass(
            mini_dcn, cfg, TIMES, None, consumer="2",
            fraction=0.2, whole_horizon=False, onset=float(TIMES[12]) + 1.0,
        )
