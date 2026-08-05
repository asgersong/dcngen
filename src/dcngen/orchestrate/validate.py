"""Validation gate: the rule set a scenario must pass to enter the dataset.

Rules, in report order:

1. ``mass_balance`` — the loop neither creates nor loses water beyond the
   leak: ``|make_up - leak| <= tol * design flow`` per step.
2. ``energy_closure`` — first-law balance: mean plant power balances mean
   delivered loads + recorded pipe gains + the leak advection term over
   the WHOLE horizon. The recorded gains are ADVECTIVE differences
   (storage-inclusive), so the identity is exact for any horizon and any
   initial state — no spin-up window, no turnover precondition; the
   storage rate from the plug states rides along as a reported diagnostic.
   Leak advection: leaked water leaves at its node
   temperature and is implicitly replaced at the plant arrival temperature
   (the thermal model's make-up convention), costing
   ``rho*cp*q_leak*(T_arrival - T_leak_node)``.
3. ``temperature_envelope`` — ``T_supply - tol <= T <= T_ground + tol``
   for supply, return, and ETS-outlet temperatures.
4. ``velocity_envelope`` — self-consistency via the circulation bound:
   ``v(edge, t) <= (max_flow_factor + leak severity) * Q_design_network
   / A_edge`` — in a single-plant closed loop no pipe can carry more
   than the pump can deliver. Per-edge redistribution and the textbook
   2.5 m/s are NOT caps: both are reported as metadata.
5. ``pressure_rails`` — every junction >= high-point floor ``(z_max - z)
   + pressure_floor_overpressure``; plant suction (reservoir head minus
   plant-room elevation, the anchoring's static property) >=
   ``plant_suction_min``; ceilings by pressure class: <=
   ``pressure_ceiling`` (0.9 x PN16) stays PN16, above it the scenario
   escalates to PN25 (stamped via the ``pn25_escalated`` metric) and
   only exceeding ``pressure_ceiling_pn25`` (0.9 x PN25) fails.
6. ``ets_reverse_flow`` — realized crossover flows never run return ->
   supply.
7. ``unmet_load`` — clean stretches carry a small unmet budget
   (outside-fault fraction <= ``max_clean_unmet_frac``, recorded like the
   convergence budget — transient peak valve pinches are real behaviour);
   under faults unmet remains pure signal, recorded never rejected.
8. ``convergence_budget`` — unconverged step fraction <=
   ``max_unconverged_frac`` (accept-but-record budget).

Side effect: the velocity rule runs one steady solve at design flows, so
the DCN's FCV settings are left at their design values afterwards.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from dcngen.config import Config
from dcngen.faults.taxonomy import FaultKind, FaultLabel
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.orchestrate.scenario import (
    EpsNtuScenarioResult,
    FixedDtScenarioResult,
    eps_channels,
)
from dcngen.topology.dcnifier import DCNetwork

_Result = EpsNtuScenarioResult | FixedDtScenarioResult

# divide-by-zero guard for ratios (not a physical constant or rule threshold)
_TINY = 1e-12

# numerical slack [m] for the suction-rail comparison ONLY (not physics):
# a rail_fallback anchor sits exactly ON the rail, and the gate's
# re-derivation (z + suction_min) - z loses ~1 ulp for a measurable
# fraction of elevations (workflow finding 2026-07-25) — same rationale
# as validation.temperature_tol absorbing float roundoff on the envelope
_SUCTION_ROUNDOFF = 1e-9


@dataclass(frozen=True)
class RuleResult:
    """One gate rule's verdict."""

    rule: str
    passed: bool
    detail: str  # one human-readable sentence for the run_poc printout
    metrics: Mapping[str, float]  # numbers worth keeping as card metadata


@dataclass(frozen=True)
class GateReport:
    """The whole gate: passed iff every rule passed."""

    passed: bool
    rules: tuple[RuleResult, ...]

    def as_dict(self) -> dict:
        """Plain-JSON form for the metadata card (exceedance metrics etc.)."""
        return {
            "passed": bool(self.passed),
            "rules": [
                {
                    "rule": r.rule,
                    "passed": bool(r.passed),
                    "detail": r.detail,
                    "metrics": {k: float(v) for k, v in r.metrics.items()},
                }
                for r in self.rules
            ],
        }


def _pipe_geometry(
    dcn: DCNetwork, pipe_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(area [m2], volume [m3]) arrays aligned with ``pipe_names``."""
    links = [dcn.wn.get_link(p) for p in pipe_names]
    area = np.array([np.pi / 4.0 * link.diameter**2 for link in links])
    return area, area * np.array([link.length for link in links])


def _mass_balance(extra: dict, dcn: DCNetwork, cfg: Config) -> RuleResult:
    scale = dcn.pump_design[0]
    worst = float(np.max(np.abs(extra["make_up_flow"] - extra["leak_flow"])))
    tol = cfg.validation.mass_balance_rel_tol * scale
    return RuleResult(
        rule="mass_balance",
        passed=worst <= tol,
        detail=(
            f"max |make-up - leak| = {worst:.3e} m3/s "
            f"(tol {tol:.3e}, design flow {scale:.3f})"
        ),
        metrics={"max_unexplained_make_up": worst, "tolerance": tol},
    )


def _energy_closure(
    result: _Result, extra: dict, label: FaultLabel, dcn: DCNetwork, cfg: Config
) -> RuleResult:
    """First-law balance over the whole horizon: mean plant power balances
    mean delivered loads + recorded pipe gains + the leak advection term.
    The recorded per-pipe gain is the ADVECTIVE enthalpy difference, which
    already includes that pipe's storage release, so the two-term identity
    is storage-closed by construction — exact for any horizon and any
    initial state. The storage-change rate is reported as a diagnostic
    metric, not balanced (subtracting it double-counts)."""
    water, thermal = cfg.water, cfg.thermal
    n_steps = len(result.time)

    rhs = (
        result.delivered_load.sum(axis=1).to_numpy()
        + result.pipe_heat_gain.sum(axis=1).to_numpy()
    )
    leak_flow = np.asarray(extra["leak_flow"])
    if label.kind is FaultKind.LEAK and np.any(leak_flow > 0.0):
        rho_cp = water.rho * water.cp
        arrival = thermal.T_supply + result.plant_power / (
            rho_cp * np.maximum(result.plant_flow, _TINY)
        )
        side_frame = (
            result.supply_temp if label.side == "supply" else result.return_temp
        )
        at_leak = side_frame[label.location].to_numpy()
        rhs = rhs + rho_cp * leak_flow * (arrival - at_leak)

    storage_rate = (
        result.stored_energy_end - result.stored_energy_start
    ) / (n_steps * cfg.solver.dt)  # diagnostic only: gains are advective
    lhs_mean = float(result.plant_power.mean())
    rhs_mean = float(rhs.mean())
    residual = abs(lhs_mean - rhs_mean) / max(abs(lhs_mean), _TINY)
    return RuleResult(
        rule="energy_closure",
        passed=residual <= cfg.validation.energy_rel_tol,
        detail=(
            f"first-law mean residual {residual:.2%} "
            f"(tol {cfg.validation.energy_rel_tol:.0%}, "
            f"storage rate {storage_rate:+.0f} W reported over {n_steps} steps)"
        ),
        metrics={"residual_rel": residual, "storage_rate_W": storage_rate},
    )


def _temperature_envelope(result: _Result, cfg: Config) -> RuleResult:
    lo = cfg.thermal.T_supply - cfg.validation.temperature_tol
    hi = cfg.thermal.T_ground + cfg.validation.temperature_tol
    coldest, hottest = np.inf, -np.inf
    for frame in (result.supply_temp, result.return_temp, result.ets_return):
        values = frame.to_numpy(dtype=float)
        if values.size:
            coldest = min(coldest, float(np.nanmin(values)))
            hottest = max(hottest, float(np.nanmax(values)))
    return RuleResult(
        rule="temperature_envelope",
        passed=lo <= coldest and hottest <= hi,
        detail=(
            f"temperatures span [{coldest:.2f}, {hottest:.2f}] degC "
            f"(envelope [{cfg.thermal.T_supply}, {cfg.thermal.T_ground}])"
        ),
        metrics={"coldest": coldest, "hottest": hottest},
    )


def _velocity_envelope(
    result: _Result, label: FaultLabel, dcn: DCNetwork, cfg: Config
) -> RuleResult:
    """Self-consistency via the circulation bound: in a single-plant
    closed loop no pipe can carry more than the pump can deliver, and the
    pump is bounded by every valve at its limit plus the commanded leak —
    ``v <= (max_flow_factor + severity) * Q_design_network / A_pipe``.
    The per-edge sketch ``v <= mff * v_design(edge)`` false-rejects
    healthy loop redistribution (Hanoi normal operation trips it below
    the 2.5 m/s reference), so per-edge redistribution is reported as
    metadata instead of gated."""
    pipes = list(result.pipe_velocity.columns)
    area, _ = _pipe_geometry(dcn, pipes)
    severity = (
        label.severity
        if (label.kind is FaultKind.LEAK and label.severity)
        else 0.0
    )
    cap = (cfg.ets.max_flow_factor + severity) * dcn.pump_design[0] / area

    design = solve_steady(
        dcn,
        consumer_flows={j: c.design_flow for j, c in dcn.consumers.items()},
    )
    v_design = np.abs(design.velocity[pipes].to_numpy(dtype=float))

    v = np.abs(result.pipe_velocity.to_numpy(dtype=float))
    v_max = v.max(axis=0)
    utilization = float(np.max(v_max / np.maximum(cap, _TINY)))
    redistribution = float(
        np.max(v_max / np.maximum(cfg.ets.max_flow_factor * v_design, _TINY))
    )
    exceedance = float((v > cfg.validation.max_velocity).mean())
    return RuleResult(
        rule="velocity_envelope",
        passed=utilization <= 1.0,
        detail=(
            f"max v/circulation-cap = {utilization:.2f}; "
            f"loop redistribution up to {redistribution:.2f}x the per-edge "
            f"envelope and {exceedance:.1%} of pipe-steps above the "
            f"{cfg.validation.max_velocity} m/s reference (both metadata)"
        ),
        metrics={
            "max_utilization": utilization,
            "max_redistribution": redistribution,
            "reference_exceedance_frac": exceedance,
        },
    )


def _pressure_rails(result: _Result, dcn: DCNetwork, cfg: Config) -> RuleResult:
    gate = cfg.validation
    z = {
        j: dcn.wn.get_node(dcn.pairing[j][0]).elevation
        for j in result.pressure_s.columns
    }
    z_max = max(dcn.wn.get_node(n).elevation for n in dcn.wn.junction_name_list)
    floor = np.array([(z_max - z[j]) + gate.pressure_floor_overpressure
                      for j in result.pressure_s.columns])

    pressures = np.stack(
        [result.pressure_s.to_numpy(dtype=float),
         result.pressure_r.to_numpy(dtype=float)]
    )
    floor_margin = float((pressures - floor[None, None, :]).min())
    # Pressures above the PN16 continuous
    # ceiling escalate to the PN25 class instead of failing — recorded per
    # scenario, so high-head draws stay in the dataset with an honest
    # pressure-class stamp; only exceeding PN25 fails
    p_max = float(pressures.max())
    pn25 = bool(p_max > gate.pressure_ceiling)
    applicable_ceiling = gate.pressure_ceiling_pn25 if pn25 else gate.pressure_ceiling
    ceiling_margin = float(applicable_ceiling - p_max)
    # the suction rail is a static property of the pressure anchoring: the
    # reservoir head IS the suction-side reference, read against the
    # plant-room (header) elevation
    suction = float(
        dcn.wn.get_node(dcn.plant.reservoir).base_head
        - dcn.wn.get_node(dcn.plant.supply_header).elevation
    )
    passed = (
        floor_margin >= 0.0
        and ceiling_margin >= 0.0
        and suction >= gate.plant_suction_min - _SUCTION_ROUNDOFF
    )
    return RuleResult(
        rule="pressure_rails",
        passed=passed,
        detail=(
            f"floor margin {floor_margin:.1f} m, "
            f"{'PN25' if pn25 else 'PN16'} ceiling margin "
            f"{ceiling_margin:.1f} m, plant suction {suction:.1f} m "
            f"(min {gate.plant_suction_min})"
        ),
        metrics={
            "floor_margin": floor_margin,
            "ceiling_margin": ceiling_margin,
            "plant_suction": suction,
            "pn25_escalated": float(pn25),  # pressure class stamp
        },
    )


def _ets_reverse_flow(result: _Result, dcn: DCNetwork, cfg: Config) -> RuleResult:
    flows = result.consumer_flow
    design = np.array([dcn.consumers[j].design_flow for j in flows.columns])
    slack = cfg.validation.mass_balance_rel_tol * design
    worst = float((flows.to_numpy(dtype=float) + slack[None, :]).min())
    return RuleResult(
        rule="ets_reverse_flow",
        passed=worst >= 0.0,
        detail=f"min crossover flow {float(flows.to_numpy().min()):.3e} m3/s",
        metrics={"min_crossover_flow": float(flows.to_numpy().min())},
    )


def _unmet_load(extra: dict, label: FaultLabel, cfg: Config) -> RuleResult:
    """Clean scenarios carry a small unmet budget — transient valve
    pinches at peak under hot-ground draws are real-network behaviour
    (pilot max 0.34 % of consumer-steps), recorded like the convergence
    budget rather than rejected at strict zero. Under an active fault,
    unmet stays pure signal."""
    unmet = extra["unmet"].to_numpy(dtype=bool)
    mask = np.asarray(label.mask, dtype=bool)
    if unmet.size == 0:
        outside, under = 0.0, 0.0
    else:
        outside = float(unmet[~mask].mean()) if (~mask).any() else 0.0
        under = float(unmet[mask].mean()) if mask.any() else 0.0
    return RuleResult(
        rule="unmet_load",
        passed=outside <= cfg.validation.max_clean_unmet_frac,
        detail=(
            f"unmet outside fault window: {outside:.1%}; under fault "
            f"(recorded, never rejected): {under:.1%}"
        ),
        metrics={
            "unmet_frac_outside_fault": outside,
            "unmet_frac_under_fault": under,
        },
    )


def _convergence_budget(extra: dict, cfg: Config) -> RuleResult:
    frac = float((~np.asarray(extra["converged"], dtype=bool)).mean())
    return RuleResult(
        rule="convergence_budget",
        passed=frac <= cfg.validation.max_unconverged_frac,
        detail=(
            f"{frac:.1%} unconverged steps "
            f"(budget {cfg.validation.max_unconverged_frac:.0%})"
        ),
        metrics={"unconverged_frac": frac},
    )


def validate_scenario(
    dcn: DCNetwork,
    cfg: Config,
    result: _Result,
    label: FaultLabel,
) -> GateReport:
    """Run every gate rule against one completed scenario.

    Args:
        dcn: the network the scenario was solved on (geometry, design
            flows, plant anchoring). Its FCV settings are left at design
            values by the velocity rule's reference solve.
        cfg: configuration (all tolerances and rails).
        result: the completed scenario time series.
        label: the scenario's fault label — clean scenarios pass the
            explicit no-fault label; the mask scopes the unmet-load rule
            and the severity sizes the velocity leak allowance.
    """
    extra = eps_channels(result, len(result.time))
    rules = (
        _mass_balance(extra, dcn, cfg),
        _energy_closure(result, extra, label, dcn, cfg),
        _temperature_envelope(result, cfg),
        _velocity_envelope(result, label, dcn, cfg),
        _pressure_rails(result, dcn, cfg),
        _ets_reverse_flow(result, dcn, cfg),
        _unmet_load(extra, label, cfg),
        _convergence_budget(extra, cfg),
    )
    return GateReport(passed=all(r.passed for r in rules), rules=rules)
