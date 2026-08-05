"""Scenario runners: fixed-dT and eps-NTU with the warm-started
hydraulic-thermal fixed-point iteration.

Fixed-dT short-circuits the iteration: consumer flows follow
directly from the hinge equation, and with constant loads the flow field is
constant, so hydraulics is solved once and the thermal state marches through
it. The eps-NTU path iterates per timestep: solve hydraulics at the current
consumer flows, propagate the thermal state from the per-timestep snapshot,
read each valve's desired flow from the ETS outcomes, under-relax, repeat —
accept-but-record on non-convergence.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcngen.config import Config
from dcngen.faults.injector import BypassInjection, LeakInjection
from dcngen.hydraulics.wntr_model import HydraulicState, solve_steady
from dcngen.thermal.ets import (
    building_hot_inlet_temp,
    fixed_dt_primary_flow,
    solve_ets,
    transferred_load,
)
from dcngen.thermal.thermal_solver import (
    EtsStepOutcome,
    NetworkThermalState,
    ThermalStepResult,
    propagate_step,
    steady_initial_state,
    stored_energy,
)
from dcngen.topology.dcnifier import DCNetwork


@dataclass(frozen=True)
class FixedDtScenarioResult:
    """Time series of one fixed-dT scenario.

    Frames are indexed by timestep; columns are original DiTEC ids
    (junctions for node channels, link names for pipe channels). SI units;
    temperatures degC, powers W, flows m3/s.
    """

    time: np.ndarray  # [s] end of each step
    supply_temp: pd.DataFrame  # node temp at j_s, per junction
    return_temp: pd.DataFrame  # node temp at j_r, per junction (mixed)
    ets_return: pd.DataFrame  # ETS outlet temp, per consumer
    consumer_flow: pd.DataFrame  # volumetric primary flow, per consumer
    delivered_load: pd.DataFrame  # per consumer [W]
    pressure_s: pd.DataFrame  # [m] supply-side node pressure, per junction
    pressure_r: pd.DataFrame  # [m] return-side node pressure, per junction
    pipe_flow: pd.DataFrame  # [m3/s] signed vs nominal direction, per WNTR pipe
    pipe_velocity: pd.DataFrame  # [m/s] >= 0, per WNTR pipe
    pipe_headloss: pd.DataFrame  # [m] head[start]-head[end], per WNTR pipe
    plant_power: np.ndarray  # [W]
    plant_flow: np.ndarray  # [m3/s] pump flow per step (mass-balance channel)
    pipe_heat_gain: pd.DataFrame  # per pipe link (incl. plant stub) [W]
    stored_energy_start: float  # loop water thermal energy at t=0 [J]
    stored_energy_end: float  # ... and at the horizon end [J]
    hydraulics: HydraulicState  # last solved operating point (constant loads: the only one)
    thermal_state: NetworkThermalState  # plug state at the horizon end


@dataclass(frozen=True)
class EpsNtuScenarioResult:
    """Time series of one eps-NTU scenario with per-timestep convergence.

    Same channel conventions as :class:`FixedDtScenarioResult`, plus the
    iteration record (accept-but-record policy: non-convergence never
    rejects a timestep, it flags it). ``consumer_flow`` holds the flows the
    recorded state was actually solved at — never a post-hoc update.
    """

    time: np.ndarray
    supply_temp: pd.DataFrame
    return_temp: pd.DataFrame
    ets_return: pd.DataFrame
    consumer_flow: pd.DataFrame  # volumetric, per step (time-varying)
    delivered_load: pd.DataFrame  # [W] at the realized flow
    unmet: pd.DataFrame  # bool per consumer per step
    pressure_s: pd.DataFrame  # [m] supply-side node pressure, per junction
    pressure_r: pd.DataFrame  # [m] return-side node pressure, per junction
    pipe_flow: pd.DataFrame  # [m3/s] signed vs nominal direction, per WNTR pipe
    pipe_velocity: pd.DataFrame  # [m/s] >= 0, per WNTR pipe
    pipe_headloss: pd.DataFrame  # [m] head[start]-head[end], per WNTR pipe
    plant_power: np.ndarray
    plant_flow: np.ndarray  # [m3/s] pump flow per step (mass-balance channel)
    make_up_flow: np.ndarray  # [m3/s] pump minus return stub; = leakage under leaks
    leak_flow: np.ndarray  # [m3/s] realized leak outflow per step; zeros when clean
    pipe_heat_gain: pd.DataFrame
    stored_energy_start: float  # loop water thermal energy at t=0 [J]
    stored_energy_end: float  # ... and at the horizon end [J]
    converged: np.ndarray  # bool per step
    iterations: np.ndarray  # int per step
    hydraulics: HydraulicState  # last solved operating point (converged or flagged)
    thermal_state: NetworkThermalState  # plug state at the horizon end


def eps_channels(
    result: "EpsNtuScenarioResult | FixedDtScenarioResult", n_steps: int
) -> dict:
    """Eps-NTU-only channels, with fixed-dT's exact semantics as defaults:
    a clean closed loop (no leak possible, zero make-up), load always met,
    and the direct solve trivially converged in one pass. The ONE place
    that discriminates the two result types (records writer + gate).
    """
    if isinstance(result, EpsNtuScenarioResult):
        return {
            "unmet": result.unmet,
            "make_up_flow": result.make_up_flow,
            "leak_flow": result.leak_flow,
            "converged": result.converged,
            "iterations": result.iterations,
        }
    return {
        "unmet": pd.DataFrame(index=range(n_steps)),
        "make_up_flow": np.zeros(n_steps),
        "leak_flow": np.zeros(n_steps),
        "converged": np.ones(n_steps, dtype=bool),
        "iterations": np.ones(n_steps, dtype=int),
    }


def _check_scenario_args(
    dcn: DCNetwork, loads: Mapping[str, float | np.ndarray], n_steps: int
) -> None:
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    unknown = sorted(set(loads) - set(dcn.consumers))
    if unknown:
        raise ValueError(f"loads for non-consumers: {unknown}")


def _broadcast_series(
    value: float | np.ndarray, n_steps: int, name: str
) -> np.ndarray:
    """Scalar or ``(n_steps,)`` array -> ``(n_steps,)`` array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.broadcast_to(arr, (n_steps,))
    if arr.shape != (n_steps,):
        raise ValueError(
            f"{name} has shape {arr.shape}, expected scalar or (n_steps,) = ({n_steps},)"
        )
    return arr


def _broadcast_loads(
    loads: Mapping[str, float | np.ndarray], n_steps: int
) -> dict[str, np.ndarray]:
    """Per-consumer load series, (n_steps,) each; scalars broadcast."""
    return {j: _broadcast_series(q, n_steps, f"loads[{j!r}]") for j, q in loads.items()}


def _broadcast_setpoints(
    supply_setpoint: float | np.ndarray | None, n_steps: int, cfg: Config
) -> np.ndarray:
    setpoint = cfg.thermal.T_supply if supply_setpoint is None else supply_setpoint
    return _broadcast_series(setpoint, n_steps, "supply_setpoint")


class _Recorder:
    """Per-step channel arrays shared by both runners."""

    def __init__(self, dcn: DCNetwork, pipe_names: list[str], n_steps: int):
        self.junctions = sorted(dcn.pairing)
        self.consumers = sorted(dcn.consumers)
        self.pipe_names = pipe_names
        self.supply = np.empty((n_steps, len(self.junctions)))
        self.ret = np.empty((n_steps, len(self.junctions)))
        self.ets_return = np.full((n_steps, len(self.consumers)), np.nan)
        self.delivered = np.zeros((n_steps, len(self.consumers)))
        self.unmet = np.zeros((n_steps, len(self.consumers)), dtype=bool)
        self.gains = np.zeros((n_steps, len(pipe_names)))
        self.plant = np.empty(n_steps)
        # per-step hydraulic channels (the records writer emits the
        # dynamic channels without re-solving)
        self.s_nodes = [dcn.pairing[j][0] for j in self.junctions]
        self.r_nodes = [dcn.pairing[j][1] for j in self.junctions]
        self.wn_pipes = sorted(dcn.wn.pipe_name_list)
        links = [dcn.wn.get_link(p) for p in self.wn_pipes]
        self.pipe_starts = [link.start_node_name for link in links]
        self.pipe_ends = [link.end_node_name for link in links]
        self.p_s = np.empty((n_steps, len(self.junctions)))
        self.p_r = np.empty((n_steps, len(self.junctions)))
        self.pipe_flow = np.empty((n_steps, len(self.wn_pipes)))
        self.pipe_velocity = np.empty((n_steps, len(self.wn_pipes)))
        self.pipe_headloss = np.empty((n_steps, len(self.wn_pipes)))

    def hydraulic_frames(self) -> dict[str, pd.DataFrame]:
        """The per-step hydraulic channels as result-field DataFrames."""
        return {
            "pressure_s": pd.DataFrame(self.p_s, columns=self.junctions),
            "pressure_r": pd.DataFrame(self.p_r, columns=self.junctions),
            "pipe_flow": pd.DataFrame(self.pipe_flow, columns=self.wn_pipes),
            "pipe_velocity": pd.DataFrame(self.pipe_velocity, columns=self.wn_pipes),
            "pipe_headloss": pd.DataFrame(self.pipe_headloss, columns=self.wn_pipes),
        }

    def record_hydraulics(self, k: int, hyd: HydraulicState) -> None:
        """Capture step ``k``'s solved hydraulic fields."""
        self.p_s[k] = hyd.pressure[self.s_nodes].to_numpy()
        self.p_r[k] = hyd.pressure[self.r_nodes].to_numpy()
        self.pipe_flow[k] = hyd.flow[self.wn_pipes].to_numpy()
        self.pipe_velocity[k] = hyd.velocity[self.wn_pipes].to_numpy()
        self.pipe_headloss[k] = (
            hyd.head[self.pipe_starts].to_numpy() - hyd.head[self.pipe_ends].to_numpy()
        )

    def record(self, k: int, step: ThermalStepResult, dcn: DCNetwork) -> None:
        for col, j in enumerate(self.junctions):
            s_node, r_node = dcn.pairing[j]
            self.supply[k, col] = step.node_temp[s_node]
            self.ret[k, col] = step.node_temp[r_node]
        for col, j in enumerate(self.consumers):
            outcome = step.ets.get(j)
            if outcome is not None:
                self.ets_return[k, col] = outcome.return_temp
                self.delivered[k, col] = outcome.delivered_load
                self.unmet[k, col] = outcome.unmet
        for col, name in enumerate(self.pipe_names):
            self.gains[k, col] = step.pipe_heat_gain.get(name, 0.0)
        self.plant[k] = step.plant_power


def run_fixed_dt_scenario(
    dcn: DCNetwork,
    cfg: Config,
    loads: Mapping[str, float | np.ndarray],
    n_steps: int,
    supply_setpoint: float | np.ndarray | None = None,
    pipe_U: Mapping[str, float] | None = None,
) -> FixedDtScenarioResult:
    """Run ``n_steps`` in fixed-dT mode; loads may vary in time.

    Args:
        dcn: built DCN.
        cfg: configuration.
        loads: cooling load [W] per consumer junction id; scalar (constant)
            or ``(n_steps,)`` array sampled at start-of-step times.
        n_steps: number of timesteps of ``cfg.solver.dt``.
        supply_setpoint: chiller outlet temperature [degC]; scalar or array
            of length ``n_steps``; defaults to ``cfg.thermal.T_supply``.
        pipe_U: per-pipe linear U' [W/(m K)], resolved by the
            caller via ``thermal.heat_gain.resolve_pipe_heat_gain``;
            ``None`` = flat ``cfg.heat_gain.scalar_U`` (dev fallback).
    """
    _check_scenario_args(dcn, loads, n_steps)
    loads_t = _broadcast_loads(loads, n_steps)
    setpoints = _broadcast_setpoints(supply_setpoint, n_steps, cfg)

    cp, dt_design = cfg.water.cp, cfg.thermal.dT_design

    def ets_model(_j: str, arrive: float, m_now: float) -> EtsStepOutcome:
        return EtsStepOutcome(
            return_temp=arrive + dt_design,
            desired_mass_flow=m_now,
            delivered_load=m_now * cp * dt_design,
            unmet=False,
        )

    flow_series = {
        j: np.array([fixed_dt_primary_flow(float(q), cfg) for q in arr])
        for j, arr in loads_t.items()
    }
    # fixed-dT needs no iteration either way; constant loads keep the
    # fast path of a single hydraulic solve for the whole horizon
    constant = all(np.ptp(a) == 0.0 for a in flow_series.values())
    hyd = solve_steady(dcn, consumer_flows={j: a[0] for j, a in flow_series.items()})
    # the loop starts at its quasi-steady operating point for the first
    # step's conditions — no spin-up
    tstate = steady_initial_state(
        dcn, cfg, hyd.flow, float(setpoints[0]), ets_model, pipe_U
    )
    e_start = stored_energy(tstate, cfg)
    rec = _Recorder(dcn, sorted(tstate.pipes), n_steps)
    plant_flow = np.empty(n_steps)

    for k in range(n_steps):
        if not constant and k > 0:
            hyd = solve_steady(
                dcn, consumer_flows={j: a[k] for j, a in flow_series.items()}
            )
        step = propagate_step(dcn, tstate, hyd.flow, float(setpoints[k]), ets_model, cfg)
        rec.record(k, step, dcn)
        rec.record_hydraulics(k, hyd)
        plant_flow[k] = hyd.flow[dcn.plant.pump]

    flow_rec = np.column_stack(
        [flow_series.get(j, np.zeros(n_steps)) for j in rec.consumers]
    )
    return FixedDtScenarioResult(
        time=(np.arange(n_steps) + 1) * cfg.solver.dt,
        supply_temp=pd.DataFrame(rec.supply, columns=rec.junctions),
        return_temp=pd.DataFrame(rec.ret, columns=rec.junctions),
        ets_return=pd.DataFrame(rec.ets_return, columns=rec.consumers),
        consumer_flow=pd.DataFrame(flow_rec, columns=rec.consumers),
        delivered_load=pd.DataFrame(rec.delivered, columns=rec.consumers),
        **rec.hydraulic_frames(),
        plant_power=rec.plant,
        plant_flow=plant_flow,
        pipe_heat_gain=pd.DataFrame(rec.gains, columns=rec.pipe_names),
        stored_energy_start=e_start,
        stored_energy_end=stored_energy(tstate, cfg),
        hydraulics=hyd,
        thermal_state=tstate,
    )


def run_eps_ntu_scenario(
    dcn: DCNetwork,
    cfg: Config,
    loads: Mapping[str, float | np.ndarray],
    n_steps: int,
    ua: Mapping[str, float],
    supply_setpoint: float | np.ndarray | None = None,
    max_iterations: int | None = None,
    leak: LeakInjection | None = None,
    bypass: BypassInjection | None = None,
    pipe_U: Mapping[str, float] | None = None,
) -> EpsNtuScenarioResult:
    """Run an eps-NTU scenario; loads may vary in time.

    Per timestep: warm-started fixed-point iteration between the hydraulic
    solve and the thermal/ETS propagation on consumer flows (under-relaxed,
    relative tolerance ``cfg.solver.flow_tol_rel``), with the plug state
    snapshot/restored so re-solving a timestep never double-advances the
    water. Flows are only updated *between* iterations, so every
    recorded channel is consistent with the flows it was solved at — also
    on unconverged (flagged) steps.

    Args:
        loads: cooling load [W] per consumer junction id; scalar (constant)
            or ``(n_steps,)`` array sampled at start-of-step times.
        ua: heat-exchanger conductance [W/K] per consumer (fouling = scale
            a consumer's healthy/design UA down before passing it in).
        max_iterations: override of ``cfg.solver.max_iterations`` (tests).
        leak: an injected abrupt leak; active on its label's masked steps.
            The realized outflow lands in ``leak_flow`` (attach it to the
            label via ``label.with_realized`` when recording the scenario).
        bypass: an injected ETS bypass; on its label's masked
            steps the consumer's exchanger sees ``(1-f)`` of the primary
            flow and returns the ``f``-weighted mix — fouling, by contrast,
            needs no hook here (scale the consumer's entry in ``ua`` via
            ``thermal.ets.installed_ua``).
        pipe_U: per-pipe linear U' [W/(m K)], resolved by the
            caller via ``thermal.heat_gain.resolve_pipe_heat_gain``;
            ``None`` = flat ``cfg.heat_gain.scalar_U`` (dev fallback).
    """
    _check_scenario_args(dcn, loads, n_steps)
    if leak is not None and leak.label.mask.shape != (n_steps,):
        raise ValueError(
            f"leak mask has shape {leak.label.mask.shape}, expected ({n_steps},)"
        )
    if leak is not None and bool(leak.label.mask[0]):
        # the steady pre-history is leak-free: fouling and bypass are
        # init-aware, but an always-on leak has no place in the v1
        # taxonomy (abrupt onsets sample the middle window) and would
        # reintroduce a phantom onset at t=0
        raise ValueError("leak active at t=0 unsupported: pre-history is leak-free")
    if bypass is not None:
        if bypass.label.mask.shape != (n_steps,):
            raise ValueError(
                f"bypass mask has shape {bypass.label.mask.shape}, "
                f"expected ({n_steps},)"
            )
        if bypass.consumer not in loads:
            raise ValueError(
                f"bypass consumer {bypass.consumer!r} is not among the loaded "
                f"consumers"
            )
    if sorted(ua) != sorted(loads):
        raise ValueError("ua must cover exactly the loaded consumers")
    loads_t = _broadcast_loads(loads, n_steps)
    setpoints = _broadcast_setpoints(supply_setpoint, n_steps, cfg)
    max_iters = cfg.solver.max_iterations if max_iterations is None else max_iterations

    cp, rho = cfg.water.cp, cfg.water.rho
    t_h_in = building_hot_inlet_temp(cfg)  # co-move closure
    m_p_max = {
        j: cfg.ets.max_flow_factor * dcn.consumers[j].design_flow * rho for j in loads
    }

    # per-step demand and bypass state the closure reads; rebound via
    # step_fault_state at init (k=0) and at the top of each timestep
    q_step: dict[str, float] = {}
    f_step: dict[str, float] = {}

    def step_fault_state(k: int) -> tuple[dict[str, float], dict[str, float]]:
        return (
            {j: float(loads_t[j][k]) for j in loads_t},
            {bypass.consumer: bypass.fraction}
            if bypass is not None and bool(bypass.label.mask[k])
            else {},
        )

    def ets_model(j: str, arrive: float, m_now: float) -> EtsStepOutcome:
        m_sec = q_step[j] / (cp * cfg.thermal.dT_design)  # dT_building == dT_design
        f_now = f_step.get(j, 0.0)
        # the exchanger sees (1-f) of the crossover flow; the energy-based
        # return temp at the TOTAL flow is then exactly the bypass mix
        # f*arrive + (1-f)*T_hx,out
        q_now = transferred_load((1.0 - f_now) * m_now, ua[j], m_sec, cp, t_h_in - arrive)
        control = solve_ets(
            load=q_step[j],
            T_c_in=arrive,
            ua=ua[j],
            m_p_max=m_p_max[j],
            T_h_in=t_h_in,
            m_sec=m_sec,
            cp=cp,
            bypass_fraction=f_now,
        )
        return EtsStepOutcome(
            return_temp=arrive + (q_now / (m_now * cp) if m_now > 0 else 0.0),
            desired_mass_flow=control.primary_mass_flow,
            delivered_load=q_now,
            unmet=control.unmet,
        )

    # warm start at the first step's fixed-dT design equivalent — also the
    # flow field of the steady initial condition (single analytic pass:
    # the residual mismatch to the converged eps-NTU point is far below
    # one step of load noise)
    flows = {j: fixed_dt_primary_flow(float(loads_t[j][0]), cfg) for j in loads}
    # whole-horizon faults start IN their faulted steady state
    q_step, f_step = step_fault_state(0)
    hyd0 = solve_steady(dcn, consumer_flows=flows)
    tstate = steady_initial_state(
        dcn, cfg, hyd0.flow, float(setpoints[0]), ets_model, pipe_U
    )
    e_start = stored_energy(tstate, cfg)
    rec = _Recorder(dcn, sorted(tstate.pipes), n_steps)
    flow_rec = np.zeros((n_steps, len(rec.consumers)))
    plant_flow = np.empty(n_steps)
    make_up_flow = np.empty(n_steps)
    leak_flow = np.zeros(n_steps)
    converged = np.zeros(n_steps, dtype=bool)
    iterations = np.zeros(n_steps, dtype=int)
    for k in range(n_steps):
        q_step, f_step = step_fault_state(k)
        leak_areas = (
            {leak.node: leak.area}
            if leak is not None and bool(leak.label.mask[k])
            else None
        )
        snapshot = tstate.clone()
        for it in range(1, max_iters + 1):
            hyd = solve_steady(
                dcn,
                consumer_flows=flows,
                leak_areas=leak_areas,
                leak_discharge_coeff=(
                    leak.discharge_coeff
                    if leak is not None
                    else cfg.faults.leak_discharge_coeff
                ),
            )
            trial = snapshot.clone()
            step = propagate_step(
                dcn, trial, hyd.flow, float(setpoints[k]), ets_model, cfg
            )
            desired = {j: step.ets[j].desired_mass_flow / rho for j in loads}
            rel = max(
                abs(desired[j] - flows[j]) / max(flows[j], 1e-12) for j in loads
            )
            if rel <= cfg.solver.flow_tol_rel:
                converged[k] = True
                break
            if it < max_iters:  # keep records consistent with the last solve
                flows = {
                    j: flows[j] + cfg.solver.relaxation * (desired[j] - flows[j])
                    for j in loads
                }
        iterations[k] = it
        tstate = trial  # accept the last iterate (accept-but-record policy)

        rec.record(k, step, dcn)
        rec.record_hydraulics(k, hyd)
        plant_flow[k] = hyd.flow[dcn.plant.pump]
        make_up_flow[k] = hyd.flow[dcn.plant.pump] - hyd.flow[dcn.plant.stub]
        if leak_areas is not None:
            leak_flow[k] = hyd.leak_flow[leak.node]
        for col, j in enumerate(rec.consumers):
            flow_rec[k, col] = flows[j]
        if not converged[k]:
            # valve controllers keep acting between timesteps: warm-start
            # the next step one relaxation move closer to the last desired
            # flows (records above stay consistent with what was solved)
            flows = {
                j: flows[j] + cfg.solver.relaxation * (desired[j] - flows[j])
                for j in loads
            }
        if k + 1 < n_steps:
            # valve feed-forward: desired flow is proportional to load at
            # the realized dT (hinge equation), so pre-scale the warm start
            # by the load ratio; exactly 1.0 when loads are constant
            flows = {
                j: (
                    flows[j] * (float(loads_t[j][k + 1]) / q_step[j])
                    if q_step[j] > 0.0
                    else fixed_dt_primary_flow(float(loads_t[j][k + 1]), cfg)
                )
                for j in loads
            }

    return EpsNtuScenarioResult(
        time=(np.arange(n_steps) + 1) * cfg.solver.dt,
        supply_temp=pd.DataFrame(rec.supply, columns=rec.junctions),
        return_temp=pd.DataFrame(rec.ret, columns=rec.junctions),
        ets_return=pd.DataFrame(rec.ets_return, columns=rec.consumers),
        consumer_flow=pd.DataFrame(flow_rec, columns=rec.consumers),
        delivered_load=pd.DataFrame(rec.delivered, columns=rec.consumers),
        unmet=pd.DataFrame(rec.unmet, columns=rec.consumers),
        **rec.hydraulic_frames(),
        plant_power=rec.plant,
        plant_flow=plant_flow,
        make_up_flow=make_up_flow,
        leak_flow=leak_flow,
        pipe_heat_gain=pd.DataFrame(rec.gains, columns=rec.pipe_names),
        stored_energy_start=e_start,
        stored_energy_end=stored_energy(tstate, cfg),
        converged=converged,
        iterations=iterations,
        hydraulics=hyd,
        thermal_state=tstate,
    )
