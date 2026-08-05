"""ETS models: fixed-dT fast path and the counterflow eps-NTU kernel.

Fixed-dT asserts the design split (dT is an *input*): the primary valve
draws exactly m = Q / (c_p * dT_design) and returns water at
T_arrive + dT_design. No heat-exchanger physics, no fouling.

The eps-NTU kernel takes explicit physical arguments (no config) so it is a
pure function of the heat-exchanger state; the config-aware closure layer
(secondary temperatures, design-UA sizing, fouling multiplier) sits above
it. The secondary program *derives* from the primary: shifted by
``ets.approach`` at both ends, so the building split equals ``dT_design``
and the program co-moves with the plant setpoint. Note a structural
property of this closure: with ``m_sec = Q / (c_p * dT_design)`` the
required effectiveness is ``dT_design / (T_h_in - T_c_in)`` *independent
of load*, so design UA values are necessarily large.
"""

from dataclasses import dataclass
from math import exp

from scipy.optimize import brentq

from dcngen.config import Config

# Relative width of the C_r ~ 1 band where the balanced-flow limit
# eps = NTU/(1+NTU) replaces the 0/0-singular general formula.
_BALANCED_BAND = 1e-6

# Numerical (not physical) root-finding settings for the control solve: the
# bracket nudge keeps brentq's lower end strictly below the root at the
# eps = 1 flow; the tolerance is far inside the physical tolerances built
# on top (flow_tol_rel and the validation gate).
_BRACKET_NUDGE = 1e-12
_BRENTQ_RTOL = 1e-14

# Float slack for the addendum-2 identity in design_ua: the temperature-derived
# effectiveness and the invariant form dT/(dT+approach) agree to the config's
# dT-consistency tolerance plus rounding — never a physical margin.
_EPS_INVARIANT_TOL = 1e-9


def fixed_dt_primary_flow(load: float, cfg: Config) -> float:
    """Primary volumetric flow [m3/s] for a cooling load [W] at design dT.

    The hinge equation: m_dot = Q / (c_p * dT_design), converted to
    volumetric flow with the configured density.
    """
    mass_flow = load / (cfg.water.cp * cfg.thermal.dT_design)
    return mass_flow / cfg.water.rho


def fixed_dt_return_temp(supply_temp: float, cfg: Config) -> float:
    """Primary return temperature [degC]: arrival temp plus the design split."""
    return supply_temp + cfg.thermal.dT_design


def counterflow_effectiveness(ntu: float, c_r: float) -> float:
    """Counterflow heat-exchanger effectiveness.

    Args:
        ntu: number of transfer units UA / C_min [-], >= 0.
        c_r: capacity-rate ratio C_min / C_max in [0, 1].
    """
    if not 0.0 <= c_r <= 1.0:
        raise ValueError(f"c_r must be in [0, 1], got {c_r}")
    if 1.0 - c_r < _BALANCED_BAND:
        return ntu / (1.0 + ntu)
    e = exp(-ntu * (1.0 - c_r))
    return (1.0 - e) / (1.0 - c_r * e)


@dataclass(frozen=True)
class EtsOperatingPoint:
    """Solved primary-side state of one ETS at one timestep."""

    primary_mass_flow: float  # [kg/s]
    return_temp: float  # primary outlet [degC]
    delivered_load: float  # [W]; < demanded when ``unmet``
    unmet: bool  # valve saturated at m_p_max before meeting demand


def transferred_load(
    m_p: float, ua: float, m_sec: float, cp: float, dt_inlet: float
) -> float:
    """Heat transferred [W] at a given primary flow (handles the C_min side
    switch); 0 when there is no flow or no driving temperature difference."""
    if m_p <= 0.0 or m_sec <= 0.0 or dt_inlet <= 0.0:
        return 0.0
    c_p, c_s = m_p * cp, m_sec * cp
    c_min, c_max = min(c_p, c_s), max(c_p, c_s)
    eps = counterflow_effectiveness(ua / c_min, c_min / c_max)
    return eps * c_min * dt_inlet


def building_supply_temp(cfg: Config) -> float:
    """Secondary (building-loop) supply setpoint [degC], derived per the
    co-move rule: primary supply + ``ets.approach``."""
    return cfg.thermal.T_supply + cfg.ets.approach


def building_hot_inlet_temp(cfg: Config) -> float:
    """Secondary hot-inlet (building return) temperature [degC], the other
    end of the co-move rule: primary design return + ``ets.approach``."""
    return cfg.thermal.T_return_design + cfg.ets.approach


def design_ua(load: float, cfg: Config) -> float:
    """Heat-exchanger conductance [W/K] that delivers ``load`` exactly at the
    design point (primary at T_supply/design flow) — the healthy UA that
    makes eps-NTU and fixed-dT agree at design.

    The co-move rule makes the design point exactly balanced: both streams
    run at C = load / dT_design, the design effectiveness is the invariant
    dT_design / (dT_design + approach), and UA = C * NTU with
    NTU = eps / (1 - eps) — i.e. UA = load / approach.

    Raises:
        ValueError: when the derived closure is thermodynamically infeasible
            (required effectiveness >= 1) — structurally impossible for a
            positive approach; retained as an invariant check — or when the
            derived effectiveness deviates from the co-move invariant (an
            inconsistent primary program slipped past config).
    """
    dt_inlet = building_hot_inlet_temp(cfg) - cfg.thermal.T_supply
    eps_design = cfg.thermal.dT_design / dt_inlet
    if eps_design >= 1.0:
        raise ValueError(
            f"design point infeasible: required effectiveness {eps_design:.3f} >= 1 "
            f"(hot inlet {building_hot_inlet_temp(cfg)} degC vs primary return "
            f"{cfg.thermal.T_return_design} degC)"
        )
    invariant = cfg.thermal.dT_design / (cfg.thermal.dT_design + cfg.ets.approach)
    if abs(eps_design - invariant) > _EPS_INVARIANT_TOL:
        raise ValueError(
            f"design effectiveness {eps_design!r} deviates from the co-move "
            f"invariant dT/(dT+approach) = {invariant!r}"
        )
    ntu = eps_design / (1.0 - eps_design)
    c = load / cfg.thermal.dT_design  # both streams, balanced by construction [W/K]
    return ntu * c


def installed_ua(
    load: float, cfg: Config, healthy_factor: float = 1.0, fouling: float = 1.0
) -> float:
    """Heat-exchanger conductance [W/K] as installed: the design sizing
    (:func:`design_ua`) times the healthy sizing margin (sampled
    U(1.00, 1.10)) times the whole-scenario fouling multiplier m. Fouling
    composes on the margin-carrying healthy UA, never on the bare design
    requirement.

    Raises:
        ValueError: when ``healthy_factor`` < 1 (installed exchangers carry
            the design margin) or ``fouling`` is outside (0, 1]
            (fouling only removes conductance).
    """
    if healthy_factor < 1.0:
        raise ValueError(
            f"healthy_factor ({healthy_factor}) must be >= 1: installed UA "
            f"sits at or above the design requirement"
        )
    if not 0.0 < fouling <= 1.0:
        raise ValueError(f"fouling multiplier ({fouling}) must be in (0, 1]")
    return design_ua(load, cfg) * healthy_factor * fouling


def solve_ets(
    *,
    load: float,
    T_c_in: float,
    ua: float,
    m_p_max: float,
    T_h_in: float,
    m_sec: float,
    cp: float,
    bypass_fraction: float = 0.0,
) -> EtsOperatingPoint:
    """Primary-flow control solve: the valve modulates the primary flow to
    deliver the demanded load; if even the fully open valve cannot, the load
    is flagged unmet and the ETS runs at the cap.

    Under a bypass fault a fraction ``f`` of the primary flow
    short-circuits past the exchanger: the exchanger sees ``(1-f)`` of the
    total flow and the primary return is the mix
    ``T_ret = f*T_c_in + (1-f)*T_hx,out``. The valve limit stays on the
    total flow and the unmet semantics are unchanged, so the solve reduces
    to the no-bypass problem at exchanger cap ``(1-f)*m_p_max`` with the
    total flow and mixed return recovered afterwards.

    Args:
        load: demanded cooling load [W], > 0.
        T_c_in: primary (network) water arrival temperature [degC].
        ua: heat-exchanger conductance [W/K] (fouling scales this down).
        m_p_max: valve limit on primary mass flow [kg/s].
        T_h_in: secondary (building) hot-inlet temperature [degC].
        m_sec: secondary mass flow [kg/s] (closure: proportional to load).
        cp: water specific heat [J/(kg K)].
        bypass_fraction: primary short-circuit fraction f in [0, 1); 0 is
            the healthy crossover.
    """
    if not 0.0 <= bypass_fraction < 1.0:
        raise ValueError(f"bypass_fraction ({bypass_fraction}) must be in [0, 1)")
    if bypass_fraction > 0.0:
        through = 1.0 - bypass_fraction
        hx = solve_ets(
            load=load,
            T_c_in=T_c_in,
            ua=ua,
            m_p_max=through * m_p_max,
            T_h_in=T_h_in,
            m_sec=m_sec,
            cp=cp,
        )
        return EtsOperatingPoint(
            primary_mass_flow=hx.primary_mass_flow / through,
            return_temp=bypass_fraction * T_c_in + through * hx.return_temp,
            delivered_load=hx.delivered_load,
            unmet=hx.unmet,
        )

    dt_inlet = T_h_in - T_c_in
    if dt_inlet <= 0.0 or load <= 0.0:
        # no driving temperature difference (or nothing demanded): no
        # transfer is possible; a saturated valve at zero delivery
        return EtsOperatingPoint(m_p_max, T_c_in, 0.0, unmet=load > 0.0)

    at_cap = transferred_load(m_p_max, ua, m_sec, cp, dt_inlet)
    if at_cap < load:
        return EtsOperatingPoint(
            primary_mass_flow=m_p_max,
            return_temp=T_c_in + at_cap / (m_p_max * cp),
            delivered_load=at_cap,
            unmet=True,
        )

    # transferred(m_p) rises monotonically from 0; the physical lower
    # bracket is the eps = 1 flow, where transferred <= load with equality
    # only at infinite UA
    m_lo = load / (cp * dt_inlet)
    m_star = brentq(
        lambda m: transferred_load(m, ua, m_sec, cp, dt_inlet) - load,
        m_lo * (1.0 - _BRACKET_NUDGE),
        m_p_max,
        rtol=_BRENTQ_RTOL,
    )
    delivered = transferred_load(m_star, ua, m_sec, cp, dt_inlet)
    return EtsOperatingPoint(
        primary_mass_flow=m_star,
        return_temp=T_c_in + delivered / (m_star * cp),
        delivered_load=delivered,
        unmet=False,
    )
