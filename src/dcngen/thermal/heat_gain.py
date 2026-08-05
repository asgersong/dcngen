"""Per-pipe linear heat-gain coefficient U' [W/(m K)].

The water-ground coefficient per metre of buried bonded pipe follows the
standard two-resistance form — insulation ring plus ground —

    U' = 1 / [ ln(D_c/D_o)/(2 pi l_PUR) + arccosh(2H/D_c)/(2 pi l_soil) ]

(Wallenten 1991 -> EN 13941-1 -> ASHRAE ch. 11 provenance chain). The
casing diameter D_c comes from the EN 253 series-1 tables via the casing
ratio D_c/D_o; the burial depth to the pipe axis is H = cover + D_c/2,
which keeps arbitrarily large bores below grade automatically.

Bores beyond the bonded-pipe product range (> DN1200) extrapolate at the
last table row's constant casing ratio (~1.15) and carry an
``insulation_extrapolated`` flag so the out-of-range assumption stays
visible downstream.

The three quality knobs (PUR conductivity, soil conductivity, burial
cover) are sampled uniformly within the cited config bounds, once
per scenario; the fallback ``scalar`` mode applies one flat U' instead
(dev/ablation only).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dcngen.config import Config, HeatGainModel

if TYPE_CHECKING:
    from dcngen.topology.dcnifier import DCNetwork

# EN 253 series-1 anchor rows verified in the research note (steel OD ->
# casing OD, both [m]): DN125/DN400/DN600/DN1200 from the LOGSTOR design
# manual + catalogue [U11, U12], DN200 from the Waerme Hamburg DC study
# [U6]. The casing ratio is interpolated linearly in steel OD between the
# rows and held constant beyond both ends — below DN125 that is mildly
# conservative; above DN1200 it IS the constant-ratio (~1.15)
# extrapolation rule.
_SERIES1_STEEL_OD = np.array([0.1397, 0.2191, 0.4064, 0.610, 1.219])
_SERIES1_CASING_OD = np.array([0.225, 0.315, 0.560, 0.800, 1.400])
_SERIES1_RATIO = _SERIES1_CASING_OD / _SERIES1_STEEL_OD

# Bonded single pipes end at DN1200 — the table's last row (LOGSTOR [U12]);
# larger bores are outside any product range and get flagged.
DN1200_STEEL_OD = float(_SERIES1_STEEL_OD[-1])


@dataclass(frozen=True)
class HeatGainKnobs:
    """One scenario's draw of the three insulation-physics knobs."""

    lambda_pur: float  # PUR foam conductivity [W/(m K)]
    lambda_soil: float  # soil conductivity [W/(m K)]
    burial_cover: float  # ground cover above the casing [m]


@dataclass(frozen=True)
class PipeHeatGain:
    """Per-pipe U' as used by one scenario, plus how it was obtained."""

    mode: str  # "derived" | "scalar"
    U: dict[str, float]  # WNTR pipe name -> linear U' [W/(m K)]
    extrapolated: dict[str, bool]  # pipe name -> beyond-product-range flag
    knobs: HeatGainKnobs | None  # the draw (None in scalar mode)


def draw_knobs(hg: HeatGainModel, rng: np.random.Generator) -> HeatGainKnobs:
    """Sample the three knobs uniformly within their config bounds."""
    return HeatGainKnobs(
        lambda_pur=float(rng.uniform(hg.lambda_pur_min, hg.lambda_pur_max)),
        lambda_soil=float(rng.uniform(hg.lambda_soil_min, hg.lambda_soil_max)),
        burial_cover=float(rng.uniform(hg.burial_cover_min, hg.burial_cover_max)),
    )


def insulation_extrapolated(diameters: np.ndarray) -> np.ndarray:
    """Boolean flags: service OD beyond the bonded-pipe product range."""
    return np.asarray(diameters) > DN1200_STEEL_OD


def derive_linear_U(diameters: np.ndarray, knobs: HeatGainKnobs) -> np.ndarray:
    """Per-pipe U' [W/(m K)] from service-pipe ODs [m] and one knob draw.

    Vectorised two-resistance formula over the EN 253 series-1 casing
    geometry; the draw's hydraulic diameter stands in for the service OD
    (wall thickness is second-order here).
    """
    d_o = np.asarray(diameters, dtype=float)
    ratio = np.interp(d_o, _SERIES1_STEEL_OD, _SERIES1_RATIO)
    d_c = ratio * d_o
    # H = cover + D_c/2 -> 2H/D_c = 1 + 2*cover/D_c >= 1: arccosh-safe
    axis_ratio = 1.0 + 2.0 * knobs.burial_cover / d_c
    r_insulation = np.log(ratio) / (2.0 * np.pi * knobs.lambda_pur)
    r_ground = np.arccosh(axis_ratio) / (2.0 * np.pi * knobs.lambda_soil)
    return 1.0 / (r_insulation + r_ground)


def resolve_pipe_heat_gain(
    dcn: "DCNetwork",
    cfg: Config,
    rng: np.random.Generator | None,
    knobs: HeatGainKnobs | None = None,
) -> PipeHeatGain:
    """The per-scenario U' map for every WNTR pipe, per the configured mode.

    Derived mode uses the explicit ``knobs`` when given (plan-row
    values), else draws one :class:`HeatGainKnobs` from ``rng``, and
    derives U' from each pipe's diameter (mirrored twins share the draw
    geometry, hence the value; the 1 m plant stub is model plumbing and
    simply gets the same treatment). Scalar mode applies
    ``heat_gain.scalar_U`` flat.
    """
    names = list(dcn.wn.pipe_name_list)
    if cfg.heat_gain.mode == "scalar":
        return PipeHeatGain(
            mode="scalar",
            U={name: cfg.heat_gain.scalar_U for name in names},
            extrapolated={name: False for name in names},
            knobs=None,
        )
    if cfg.heat_gain.mode != "derived":  # config validation makes this unreachable
        raise ValueError(f"unknown heat_gain.mode {cfg.heat_gain.mode!r}")
    if knobs is None:
        if rng is None:
            raise ValueError("rng required to draw heat-gain knobs (or pass knobs)")
        knobs = draw_knobs(cfg.heat_gain, rng)
    diameters = np.array([dcn.wn.get_link(name).diameter for name in names])
    u_prime = derive_linear_U(diameters, knobs)
    flags = insulation_extrapolated(diameters)
    return PipeHeatGain(
        mode="derived",
        U={name: float(u) for name, u in zip(names, u_prime)},
        extrapolated={name: bool(f) for name, f in zip(names, flags)},
        knobs=knobs,
    )
