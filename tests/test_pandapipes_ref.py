"""The pandapipes cross-check.

Two layers. The friction-law bridge is pure arithmetic and is tested
directly against pandapipes' own correlation, in the default suite. The
end-to-end cross-check needs the real dump and a full Hanoi run, so it is
slow-marked (run with ``pytest -m slow``) and skipped without the dump --
the same convention as the other ``*_hanoi`` acceptance tests.

The end-to-end test is
deliberately a *criterion*, not a smoke test: it asserts the
tolerances (0.5 K, 3 %, 1 %) from config rather than re-hardcoding them,
so tightening the config tightens the test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from dcngen.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verify"))

pandapipes = pytest.importorskip("pandapipes", reason="verification-only dependency")
import pandapipes_ref as ref  # noqa: E402  (after the importorskip guard)

cfg = load_config()
HANOI = cfg.paths.ditec_data / cfg.poc.network


def _pandapipes_lambda(k: np.ndarray, d: np.ndarray, re: np.ndarray) -> np.ndarray:
    """pandapipes' own friction law, transcribed from its source.

    ``pf/derivative_calculation.py`` forms ``lambda = lambda_laminar +
    lambda_nikuradse`` with ``lambda_laminar = 64/Re`` added at every
    Reynolds number. Restating it here (rather than importing an internal)
    is what makes the round-trip test meaningful: it pins the inversion to
    a written-down formula that a pandapipes upgrade would have to change
    visibly.
    """
    return 64.0 / re + 1.0 / (-2.0 * np.log10(k / (3.71 * d))) ** 2


def test_equivalent_roughness_round_trips_through_the_friction_law():
    """k is chosen so pandapipes reproduces exactly the given headloss."""
    d = np.array([0.4, 0.6, 1.0])
    length = np.array([2000.0, 5000.0, 800.0])
    flow = np.array([0.3, 1.2, 2.0])
    # headlosses chosen well ABOVE the 64/Re floor so every pipe is reachable
    headloss = np.array([4.0, 6.0, 2.0])
    rho, mu = 999.5, 1.3e-3

    k, unreachable = ref.equivalent_roughness(d, length, flow, headloss, rho, mu)
    assert not unreachable.any(), "test fixture should sit above the laminar floor"

    v = flow / (np.pi / 4.0 * d**2)
    re = rho * v * d / mu
    lam = _pandapipes_lambda(k, d, re)
    reproduced = lam * length / d * v**2 / (2.0 * ref.G)
    np.testing.assert_allclose(reproduced, headloss, rtol=1e-9)


def test_unreachable_pipes_are_flagged_not_silently_wrong():
    """Below the 64/Re floor no roughness works; say so rather than clamp.

    This is not a corner case on the DiTEC substrate -- its sampled
    Hazen-Williams C is far outside the physical range -- so the flag
    has to survive into the report.
    """
    d = np.array([0.5])
    length = np.array([1000.0])
    flow = np.array([1.0])
    v = flow / (np.pi / 4.0 * d**2)
    re = 999.5 * v * d / 1.3e-3
    # ask for HALF the headloss the unavoidable laminar term alone produces
    floor_headloss = (64.0 / re) * length / d * v**2 / (2.0 * ref.G)

    k, unreachable = ref.equivalent_roughness(
        d, length, flow, floor_headloss / 2.0, 999.5, 1.3e-3
    )
    assert unreachable.all()
    assert (k > 0.0).all(), "a zero k would make pandapipes' Jacobian singular"


def test_stagnant_pipes_are_flagged_and_carry_a_usable_roughness():
    """No flow means no operating point to calibrate against."""
    k, unreachable = ref.equivalent_roughness(
        np.array([0.5]), np.array([100.0]), np.array([0.0]), np.array([0.0]),
        999.5, 1.3e-3,
    )
    assert unreachable.all()
    assert np.isfinite(k).all() and (k > 0.0).all()


@pytest.mark.slow
@pytest.mark.skipif(not HANOI.is_dir(), reason=f"DiTEC dump not present at {HANOI}")
@pytest.mark.parametrize("draw", [0, 17, 203])
def test_hanoi_steady_fields_agree_with_pandapipes(draw):
    """The end-to-end acceptance criterion, at the config tolerances.

    Three draws, because one cannot distinguish "the physics agrees" from
    "this network happened to line up". Draw 203 earns its place: it is
    the pathological draw (near-stagnant pipes, the slowest
    transit of the three at ~165 steps) and it has 32 of 69 pipes below
    the friction floor against draw 0's 22 -- so it exercises the harness
    where the friction bridge is weakest, and its flow metric lands at
    0.73 % against the 1 % tolerance rather than passing comfortably.
    """
    report = ref.verify_network(cfg, static_draw_id=draw)
    detail = report.text()

    assert report.passed, detail
    scored = {c.name: c for c in report.comparisons}
    assert set(scored) == {
        "node temperature",
        "pipe flow",
        "node pressure",
        "plant energy balance",
    }, detail
    # the tolerances are the config's, not this test's
    assert scored["node temperature"].tolerance == cfg.verification.temperature_tol
    assert scored["plant energy balance"].tolerance == cfg.verification.energy_rel_tol
    assert scored["pipe flow"].tolerance == cfg.verification.flow_rel_tol

    # Temperatures are the field this harness exists to check: hold them to
    # a tenth of the gate, so a real regression cannot hide inside the band.
    assert scored["node temperature"].deviation < 0.1 * cfg.verification.temperature_tol, detail
    # The substrate's unphysical smoothness must stay visible.
    assert report.unreachable_pipes > 0, detail
    assert report.total_pipes == 69
