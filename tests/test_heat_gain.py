"""Per-pipe linear heat-gain coefficient U'.

Anchors: the DC-specific
DN200 series-1 value (U6: Wärme Hamburg network, d_o 0.2191 m, casing
0.315 m, H 0.9575 m, lambda_PUR 0.027, lambda_soil 1.2 -> U' ~ 0.41),
the strong monotonic diameter dependence (U5/U11), and the DN1200 end of
the bonded-pipe product range (U12/U13).
"""

from dataclasses import replace

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.thermal.heat_gain import (
    DN1200_STEEL_OD,
    HeatGainKnobs,
    derive_linear_U,
    draw_knobs,
    insulation_extrapolated,
    resolve_pipe_heat_gain,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory, cfg):
    from dcngen.topology.dcnifier import build_dcn
    from dcngen.topology.ditec_loader import load_ditec_network
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


U6_KNOBS = HeatGainKnobs(lambda_pur=0.027, lambda_soil=1.2, burial_cover=0.8)


def test_dn200_series1_anchor_reproduces_u6():
    # DN200 steel OD 0.2191 m sits on a table row -> casing 0.315 m exactly;
    # cover 0.8 puts the axis at H = 0.8 + 0.315/2 = 0.9575 m as in U6
    u = derive_linear_U(np.array([0.2191]), U6_KNOBS)
    assert u[0] == pytest.approx(0.41, abs=0.01)


def test_u_prime_monotonic_in_diameter():
    # U5: "insulation thickness does not increase proportionally with the
    # nominal diameter" -> U' rises with bore; the Hanoi draw span 0.13-2.26
    diameters = np.linspace(0.13, 2.26, 200)
    u = derive_linear_U(diameters, U6_KNOBS)
    assert (np.diff(u) > 0).all()


def test_u_prime_band_matches_research_envelope():
    # research note: effective U' ~ 0.19-0.6 (DN125-600), 0.6-1.8 (DN800-2200);
    # series-1 geometry keeps the 0.1 tail unreachable at any knob corner
    corners = [
        HeatGainKnobs(0.023, 1.0, 0.6),
        HeatGainKnobs(0.023, 1.6, 1.2),
        HeatGainKnobs(0.029, 1.0, 0.6),
        HeatGainKnobs(0.029, 1.6, 1.2),
    ]
    diameters = np.linspace(0.13, 2.26, 50)
    for knobs in corners:
        u = derive_linear_U(diameters, knobs)
        assert (u > 0.2).all() and (u < 2.0).all()


def test_extrapolation_flag_above_dn1200():
    flags = insulation_extrapolated(np.array([0.13, DN1200_STEEL_OD, 1.3, 2.26]))
    assert flags.tolist() == [False, False, True, True]


def test_derive_is_vectorised_and_matches_elementwise():
    diameters = np.array([0.2, 0.6, 1.5])
    u = derive_linear_U(diameters, U6_KNOBS)
    assert u.shape == (3,)
    for i, d in enumerate(diameters):
        assert u[i] == derive_linear_U(np.array([d]), U6_KNOBS)[0]


def test_draw_knobs_within_ranges_and_deterministic(cfg):
    hg = cfg.heat_gain
    knobs = draw_knobs(hg, np.random.default_rng(7))
    again = draw_knobs(hg, np.random.default_rng(7))
    assert knobs == again
    assert hg.lambda_pur_min <= knobs.lambda_pur <= hg.lambda_pur_max
    assert hg.lambda_soil_min <= knobs.lambda_soil <= hg.lambda_soil_max
    assert hg.burial_cover_min <= knobs.burial_cover <= hg.burial_cover_max


def test_resolve_derived_covers_every_wntr_pipe_with_equal_twins(mini_dcn, cfg):
    hg = resolve_pipe_heat_gain(mini_dcn, cfg, np.random.default_rng(3))

    assert hg.mode == "derived"
    assert hg.knobs is not None
    assert sorted(hg.U) == sorted(mini_dcn.wn.pipe_name_list)
    assert sorted(hg.extrapolated) == sorted(hg.U)
    # mirrored twins share the draw geometry, hence the derived U'
    for p, (p_s, p_r) in mini_dcn.pipe_pairing.items():
        assert hg.U[p_s] == hg.U[p_r]
        assert hg.extrapolated[p_s] == hg.extrapolated[p_r]
    assert all(u > 0 for u in hg.U.values())


def test_resolve_scalar_mode_is_flat_and_knobless(mini_dcn, cfg):
    scalar_cfg = replace(cfg, heat_gain=replace(cfg.heat_gain, mode="scalar"))
    hg = resolve_pipe_heat_gain(mini_dcn, scalar_cfg, np.random.default_rng(3))

    assert hg.mode == "scalar"
    assert hg.knobs is None
    assert set(hg.U.values()) == {scalar_cfg.heat_gain.scalar_U}
    assert not any(hg.extrapolated.values())


def test_resolve_accepts_explicit_row_knobs_without_rng(mini_dcn, cfg):
    # the plan row carries the knob draws; resolution must
    # reproduce them exactly and need no generator
    knobs = HeatGainKnobs(lambda_pur=0.025, lambda_soil=1.3, burial_cover=0.9)
    hg = resolve_pipe_heat_gain(mini_dcn, cfg, rng=None, knobs=knobs)

    assert hg.knobs == knobs
    name = sorted(hg.U)[0]
    d = mini_dcn.wn.get_link(name).diameter
    expected = derive_linear_U(np.array([d]), knobs)[0]
    assert hg.U[name] == pytest.approx(expected, rel=1e-12)

    with pytest.raises(ValueError, match="rng"):
        resolve_pipe_heat_gain(mini_dcn, cfg, rng=None)
