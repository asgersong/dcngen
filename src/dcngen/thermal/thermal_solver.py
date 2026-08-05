"""Lagrangian plug-tracking thermal transport (the "node method").

Each pipe holds a FIFO of water plugs. A plug records its *entry temperature*
and its residence-time statistics; its current temperature is **derived**:

    T(plug) = T_ground + (entry_temp - T_ground) * exp(-residence / tau)

so heat gain can never be applied twice (no mutation), and snapshot/restore
for the hydraulic-thermal iteration is a plain copy. ``tau`` is the pipe's
thermal time constant ``rho * A * c_p / U'`` [s] with ``U'`` the linear
heat-loss coefficient [W/(m K)]; a parcel resident for the transit time
``V_pipe / Q`` then reproduces the steady solution
``T_out = T_g + (T_in - T_g) * exp(-U' L / (m_dot c_p))`` exactly.

Residence bookkeeping is exact under piston flow: a plug's parcels entered
uniformly over a time interval, so residence varies *linearly* across the
plug's volume. Each plug therefore carries the volume-mean ``residence`` and
the front-to-back ``res_span``; splits and partial exits shift the mean by
linear interpolation, which keeps every exiting slice's mean residence exact
(the only approximation left is the second-order curvature of exp() across a
slice's span — zero in steady state, O((dt/tau)^2) otherwise).

Units: SI throughout (m3, s, degC).
"""

import bisect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid runtime cycle: topology imports hydraulics only
    from dcngen.config import Config
    from dcngen.topology.dcnifier import DCNetwork

_FLOW_EPS = 1e-12  # below this a link transports nothing this step [m3/s]


@dataclass(frozen=True)
class PlugState:
    """FIFO plug queue of one pipe; index 0 is next to exit (outlet end).

    ``volume`` [m3]; ``entry_temp`` [degC] at pipe entry; ``residence`` [s]
    volume-mean time in the pipe as of the step start; ``res_span`` [s]
    front-minus-back residence spread (front parcels entered earlier, so the
    front of a plug has ``residence + res_span/2``).
    """

    volume: np.ndarray
    entry_temp: np.ndarray
    residence: np.ndarray
    res_span: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.res_span is None:
            object.__setattr__(self, "res_span", np.zeros_like(self.volume))

    def clone(self) -> "PlugState":
        """Snapshot for the hydraulic-thermal iteration."""
        return PlugState(
            volume=self.volume.copy(),
            entry_temp=self.entry_temp.copy(),
            residence=self.residence.copy(),
            res_span=self.res_span.copy(),
        )


def plug_temperatures(state: PlugState, tau_thermal: float, T_ground: float) -> np.ndarray:
    """Current plug temperatures, derived from residence (never stored)."""
    return T_ground + (state.entry_temp - T_ground) * np.exp(
        -state.residence / tau_thermal
    )


def uniform_state(pipe_volume: float, temp: float) -> PlugState:
    """A pipe filled with fresh water at one temperature (initial condition)."""
    return PlugState(
        volume=np.array([pipe_volume]),
        entry_temp=np.array([float(temp)]),
        residence=np.array([0.0]),
    )


def mix_temperature(volumes: np.ndarray, temps: np.ndarray) -> float:
    """Volume(=mass)-weighted mixing temperature of merging streams."""
    total = float(np.sum(volumes))
    if total <= 0.0:
        raise ValueError("mixing requires positive total volume")
    return float(np.dot(volumes, temps) / total)


def step_pipe(
    state: PlugState,
    displaced_volume: float,
    inlet_temp: float,
    dt: float,
    tau_thermal: float,
    T_ground: float,
) -> tuple[PlugState, float]:
    """Advance one pipe by one timestep of piston flow.

    Args:
        state: plug queue at step start (pipe is always full; its volume is
            ``state.volume.sum()``).
        displaced_volume: ``|Q| * dt`` [m3] pushed through this step (>= 0;
            flow reversal is handled by the network layer reversing the
            queue before calling).
        inlet_temp: temperature of the water entering this step [degC]
            (constant within the step — node-method resolution).
        dt: timestep [s].
        tau_thermal: pipe thermal time constant [s]; ``np.inf`` = adiabatic.
        T_ground: ground/ambient temperature [degC].

    Returns:
        ``(new_state, outlet_temp)`` where ``outlet_temp`` is the
        volume-weighted mean temperature of the water that exited, or NaN if
        nothing exited.
    """
    pipe_volume = float(np.sum(state.volume))
    if displaced_volume <= 0.0:
        aged = PlugState(
            volume=state.volume.copy(),
            entry_temp=state.entry_temp.copy(),
            residence=state.residence + dt,
            res_span=state.res_span.copy(),
        )
        return aged, float("nan")

    rate = displaced_volume / dt  # volumetric flow [m3/s]

    # --- exiting slices ------------------------------------------------
    out_vol: list[float] = []
    out_temp: list[float] = []
    exited = 0.0  # cumulative volume ahead of the current slice

    def emit(volume: float, entry_temp: float, res_at_start: float) -> None:
        """Record an exiting slice; its exit happens over a known interval."""
        nonlocal exited
        mean_exit_time = (exited + exited + volume) / (2.0 * rate)
        residence = res_at_start + mean_exit_time
        temp = T_ground + (entry_temp - T_ground) * np.exp(-residence / tau_thermal)
        out_vol.append(volume)
        out_temp.append(float(temp))
        exited += volume

    remaining = min(displaced_volume, pipe_volume)
    keep_from = 0  # first old plug (partially) retained
    partial_kept: tuple[float, float, float, float] | None = None
    for i in range(len(state.volume)):
        v = float(state.volume[i])
        r, s, te = float(state.residence[i]), float(state.res_span[i]), float(
            state.entry_temp[i]
        )
        if remaining <= 0.0:
            break
        if v <= remaining:  # whole plug exits
            emit(v, te, r)
            remaining -= v
            keep_from = i + 1
        else:  # split: front fraction f exits, back is retained
            f = remaining / v
            # front piece mean residence at step start: r + span*(1-f)/2
            emit(remaining, te, r + s * (1.0 - f) / 2.0)
            partial_kept = (v - remaining, te, r - s * f / 2.0, s * (1.0 - f))
            remaining = 0.0
            keep_from = i + 1

    passthrough = displaced_volume - pipe_volume
    if passthrough > 0.0:
        # Inflow parcels that traverse the whole pipe within the step exit
        # with residence exactly V_pipe/rate each (zero spread: exact).
        # emit() adds the slice's mean exit time, so cancel it upfront:
        # residence = res_at_start + mean_exit = V/rate.
        mean_exit = (2.0 * exited + passthrough) / (2.0 * rate)
        emit(passthrough, inlet_temp, pipe_volume / rate - mean_exit)

    outlet_temp = mix_temperature(np.array(out_vol), np.array(out_temp))

    # --- retained contents at step end ----------------------------------
    volumes: list[float] = []
    entry_temps: list[float] = []
    residences: list[float] = []
    spans: list[float] = []

    if displaced_volume < pipe_volume:
        if partial_kept is not None:
            v, te, r, s = partial_kept
            volumes.append(v)
            entry_temps.append(te)
            residences.append(r + dt)
            spans.append(s)
        for i in range(keep_from, len(state.volume)):
            volumes.append(float(state.volume[i]))
            entry_temps.append(float(state.entry_temp[i]))
            residences.append(float(state.residence[i]) + dt)
            spans.append(float(state.res_span[i]))
        # retained inflow entered uniformly over [0, dt]
        volumes.append(displaced_volume)
        entry_temps.append(inlet_temp)
        residences.append(dt / 2.0)
        spans.append(dt)
    else:
        # everything old flushed; pipe holds the newest V_pipe of the inflow,
        # entered uniformly over the last V_pipe/rate of the step
        transit = pipe_volume / rate
        volumes.append(pipe_volume)
        entry_temps.append(inlet_temp)
        residences.append(transit / 2.0)
        spans.append(transit)

    new_state = PlugState(
        volume=np.array(volumes),
        entry_temp=np.array(entry_temps),
        residence=np.array(residences),
        res_span=np.array(spans),
    )
    return new_state, outlet_temp


# --------------------------------------------------------------------------
# Network layer: plug queues for every pipe, advanced one timestep at a time
# in flow-topological order (the flow digraph of a solved hydraulic state is
# acyclic — head strictly decreases along flow — with the loop broken at the
# Plant: the pump is a boundary *injection* at the supply header at the
# chiller setpoint, the reservoir a pure sink recording the return arrival).
# --------------------------------------------------------------------------


@dataclass
class NetworkThermalState:
    """Plug queues plus per-pipe constants; mutated step by step."""

    pipes: dict[str, PlugState]
    orientation: dict[str, int]  # +1: queue aligned with the nominal link direction
    pipe_volume: dict[str, float]  # [m3]
    tau_thermal: dict[str, float]  # [s]

    def clone(self) -> "NetworkThermalState":
        """Snapshot of the whole network's plug state."""
        return NetworkThermalState(
            pipes={name: s.clone() for name, s in self.pipes.items()},
            orientation=dict(self.orientation),
            pipe_volume=self.pipe_volume,
            tau_thermal=self.tau_thermal,
        )


@dataclass(frozen=True)
class EtsStepOutcome:
    """What one ETS did this timestep — the thermal layer's ETS contract.

    ``desired_mass_flow`` is what the control valve *wants* next (drives the
    hydraulic-thermal iteration); ``return_temp``/``delivered_load`` describe
    the physical state at the flow that actually ran.
    """

    return_temp: float  # primary outlet [degC]
    desired_mass_flow: float  # [kg/s]
    delivered_load: float  # [W] at the realized flow
    unmet: bool


@dataclass(frozen=True)
class ThermalStepResult:
    """One propagated timestep of the whole network."""

    node_temp: dict[str, float]  # mixed temperature at every node [degC]
    ets: dict[str, EtsStepOutcome]  # per consumer
    plant_power: float  # thermal power removed by the chiller [W]
    pipe_heat_gain: dict[str, float]  # per pipe link, ground -> water [W]


def init_network_state(
    dcn: "DCNetwork",
    cfg: "Config",
    pipe_U: Mapping[str, float] | None = None,
) -> NetworkThermalState:
    """Pipes start full of design-temperature water (supply side at the
    supply setpoint, return side and plant stub at the design return).

    Args:
        pipe_U: linear heat-gain coefficient U' [W/(m K)] per WNTR pipe
            name (derived per pipe, resolved by the caller via
            ``thermal.heat_gain``). ``None`` falls back to the flat
            ``cfg.heat_gain.scalar_U`` on every pipe — the dev/ablation
            scalar mode.
    """
    pipes: dict[str, PlugState] = {}
    volumes: dict[str, float] = {}
    taus: dict[str, float] = {}
    for name in dcn.wn.pipe_name_list:
        link = dcn.wn.get_link(name)
        area = np.pi / 4.0 * link.diameter**2
        volumes[name] = area * link.length
        u_prime = cfg.heat_gain.scalar_U if pipe_U is None else pipe_U[name]
        taus[name] = cfg.water.rho * area * cfg.water.cp / u_prime
        temp = (
            cfg.thermal.T_supply
            if dcn.edge_kind[name] == "supply_pipe"
            else cfg.thermal.T_return_design
        )
        pipes[name] = uniform_state(volumes[name], temp)
    return NetworkThermalState(
        pipes=pipes,
        orientation={name: 1 for name in pipes},
        pipe_volume=volumes,
        tau_thermal=taus,
    )


def stored_energy(state: NetworkThermalState, cfg: "Config") -> float:
    """Thermal energy of the loop water [J]: rho*c_p*Sum(V_plug*T_plug)
    over every pipe (temperatures derived, never stored), against a 0 degC
    datum — the absolute value is datum-dependent; only DIFFERENCES are
    physical, and it is the difference that closes the gate's energy
    balance, exact for any horizon and any initial state."""
    rho_cp = cfg.water.rho * cfg.water.cp
    total = 0.0
    for name, queue in state.pipes.items():
        temps = plug_temperatures(queue, state.tau_thermal[name], cfg.thermal.T_ground)
        total += float(np.dot(queue.volume, temps))
    return rho_cp * total


def _flow_dag(
    dcn: "DCNetwork", flows: Mapping[str, float]
) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]], dict[str, int]]:
    """Directed active links + traversal structures under a flow field
    (shared by :func:`propagate_step` and :func:`steady_initial_state`).
    The pump is a boundary, not a graph edge; near-zero flows drop out."""
    wn = dcn.wn
    directed: dict[str, tuple[str, str]] = {}
    for name in wn.link_name_list:
        if name == dcn.plant.pump or abs(flows[name]) <= _FLOW_EPS:
            continue
        link = wn.get_link(name)
        a, b = link.start_node_name, link.end_node_name
        directed[name] = (a, b) if flows[name] > 0 else (b, a)

    out_links: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {n: 0 for n in wn.node_name_list}
    for name, (src, dst) in sorted(directed.items()):
        out_links.setdefault(src, []).append(name)
        in_degree[dst] += 1
    return directed, out_links, in_degree


def steady_initial_state(
    dcn: "DCNetwork",
    cfg: "Config",
    flows: Mapping[str, float],
    supply_setpoint: float,
    ets_model: Callable[[str, float, float], EtsStepOutcome],
    pipe_U: Mapping[str, float] | None = None,
) -> NetworkThermalState:
    """The quasi-steady operating point as the initial condition.

    One analytic pass in flow order: supply temperatures propagate from
    the plant with the steady per-pipe decay ``T_out = T_g + (T_in - T_g)
    * exp(-transit/tau)``, consumers answer through ``ets_model`` (the
    caller's closure — fault-aware at t = 0, so whole-horizon faults start
    in their faulted steady state instead of relaxing into it), and the
    return side propagates back to the plant. Each pipe is then filled
    with one plug whose entry temperature is its steady inlet and whose
    residence spans [0, transit] linearly — by the plug model's
    construction this reproduces the along-pipe exponential profile
    exactly, so marching at constant conditions is a fixed point (the
    slice-curvature error term is zero in steady state).

    The pre-history is bounded: residence is capped at
    ``thermal.init_residence_cap`` (one diurnal cycle), because the t=0
    flow field did NOT hold forever — a pipe near-stagnant at the sampled
    scenario start carried yesterday's flows, so its water is at most a
    day old and only slightly warmed (tau_thermal ~ weeks), never the
    ground-temperature water of the unbounded steady limit. Uncapped, a
    near-stagnant-at-midnight pipe initialises as a ground-temperature
    slug that discharges when morning flows return — unmet loads,
    convergence thrash, and a multi-MW storage transient (observed on
    Hanoi draws 203/778). Fully stagnant pipes hold their side's
    design-temperature water aged by the same cap.

    Args:
        dcn: the built DCN.
        cfg: configuration (water properties, ground temperature).
        flows: signed link flows [m3/s] of the t = 0 operating point.
        supply_setpoint: chiller outlet temperature at t = 0 [degC].
        ets_model: the same callable contract as :func:`propagate_step`.
        pipe_U: per-pipe linear U' [W/(m K)]; ``None`` = flat
            ``cfg.heat_gain.scalar_U``.
    """
    state = init_network_state(dcn, cfg, pipe_U)  # geometry + taus; fills replaced
    t_ground = cfg.thermal.T_ground
    ets_consumer = {c.ets_link: j for j, c in dcn.consumers.items()}
    directed, out_links, in_degree = _flow_dag(dcn, flows)

    # flow-RATE mixing weights: proportional to propagate_step's volumes
    # under the common dt, so the mixes agree
    inflow_w: dict[str, list[float]] = {n: [] for n in dcn.wn.node_name_list}
    inflow_t: dict[str, list[float]] = {n: [] for n in dcn.wn.node_name_list}
    inflow_w[dcn.plant.supply_header].append(abs(flows[dcn.plant.pump]))
    inflow_t[dcn.plant.supply_header].append(supply_setpoint)

    inlet_temp: dict[str, float] = {}  # steady inlet per flowing pipe
    ready = sorted(n for n, d in in_degree.items() if d == 0)
    processed = 0
    while ready:
        node = ready.pop(0)
        processed += 1
        t_node = (
            mix_temperature(np.array(inflow_w[node]), np.array(inflow_t[node]))
            if inflow_w[node]
            else float("nan")
        )
        for name in out_links.get(node, ()):
            _, dst = directed[name]
            if np.isnan(t_node):
                raise RuntimeError(
                    f"link {name} carries flow out of node {node} with no inflow"
                )
            q = abs(flows[name])
            if name in state.pipes:  # pipe or plant stub: steady decay
                inlet_temp[name] = t_node
                transit = state.pipe_volume[name] / q
                t_out = t_ground + (t_node - t_ground) * np.exp(
                    -transit / state.tau_thermal[name]
                )
            else:  # ETS crossover: instantaneous closure
                outcome = ets_model(ets_consumer[name], t_node, q * cfg.water.rho)
                t_out = outcome.return_temp
            inflow_w[dst].append(q)
            inflow_t[dst].append(float(t_out))
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                bisect.insort(ready, dst)  # deterministic order, as in the walk twin
    if processed < len(in_degree):  # parity with propagate_step's cycle guard
        raise RuntimeError(
            "flow digraph has a cycle: steady initial state undefined"
        )

    cap = cfg.thermal.init_residence_cap  # one diurnal pre-history
    for name, volume in state.pipe_volume.items():
        if name in inlet_temp:
            aged = min(volume / abs(flows[name]), cap)
            state.pipes[name] = PlugState(
                volume=np.array([volume]),
                entry_temp=np.array([inlet_temp[name]]),
                residence=np.array([aged / 2.0]),
                res_span=np.array([aged]),
            )
            state.orientation[name] = 1 if flows[name] > 0 else -1
        elif abs(flows[name]) <= _FLOW_EPS:
            # fully stagnant: yesterday's design-side water, aged one cap
            temp = (
                cfg.thermal.T_supply
                if dcn.edge_kind[name] == "supply_pipe"
                else cfg.thermal.T_return_design
            )
            state.pipes[name] = PlugState(
                volume=np.array([volume]),
                entry_temp=np.array([float(temp)]),
                residence=np.array([cap]),
                res_span=np.array([0.0]),
            )
            state.orientation[name] = 1
        else:  # flowing but never reached: the guard above should have fired
            raise RuntimeError(f"flowing pipe {name} unreached by the steady pass")
    return state


def _oriented(state: PlugState, stored: int, needed: int) -> tuple[PlugState, int]:
    """Reverse a queue when the pipe's flow direction flipped."""
    if stored == needed:
        return state, stored
    return (
        PlugState(
            volume=state.volume[::-1].copy(),
            entry_temp=state.entry_temp[::-1].copy(),
            residence=state.residence[::-1].copy(),
            res_span=-state.res_span[::-1].copy(),
        ),
        needed,
    )


def propagate_step(
    dcn: "DCNetwork",
    state: NetworkThermalState,
    flows: Mapping[str, float],
    supply_setpoint: float,
    ets_model: Callable[[str, float, float], EtsStepOutcome],
    cfg: "Config",
) -> ThermalStepResult:
    """Advance every pipe by one timestep and mix at nodes.

    Args:
        dcn: the built DCN (topology metadata; geometry via its wn).
        state: network plug state, mutated in place.
        flows: signed link flows [m3/s] from the hydraulic solve.
        supply_setpoint: chiller outlet temperature this step [degC].
        ets_model: ``(consumer id, arrival temp [degC], realized primary
            mass flow [kg/s]) -> EtsStepOutcome`` (fixed-dT or eps-NTU).
            ETS crossovers and the pump are thermally instantaneous.
        cfg: configuration (water properties, ground temperature, dt).
    """
    wn, dt = dcn.wn, cfg.solver.dt
    rho_cp = cfg.water.rho * cfg.water.cp
    ets_consumer = {c.ets_link: j for j, c in dcn.consumers.items()}
    directed, out_links, in_degree = _flow_dag(dcn, flows)

    inflow_vol: dict[str, list[float]] = {n: [] for n in wn.node_name_list}
    inflow_temp: dict[str, list[float]] = {n: [] for n in wn.node_name_list}
    pump_volume = abs(flows[dcn.plant.pump]) * dt
    inflow_vol[dcn.plant.supply_header].append(pump_volume)
    inflow_temp[dcn.plant.supply_header].append(supply_setpoint)

    node_temp: dict[str, float] = {}
    ets_out: dict[str, EtsStepOutcome] = {}
    heat_gain: dict[str, float] = {}

    ready = sorted(n for n, d in in_degree.items() if d == 0)
    processed = 0
    while ready:
        node = ready.pop(0)
        processed += 1
        if inflow_vol[node]:
            t_node = mix_temperature(
                np.array(inflow_vol[node]), np.array(inflow_temp[node])
            )
        else:
            t_node = float("nan")
        node_temp[node] = t_node

        for name in out_links.get(node, ()):
            src, dst = directed[name]
            if np.isnan(t_node):
                raise RuntimeError(
                    f"link {name} carries flow out of node {node} with no inflow"
                )
            displaced = abs(flows[name]) * dt
            if name in state.pipes:  # pipe or plant stub: real transport
                needed = 1 if flows[name] > 0 else -1
                queue, orient = _oriented(
                    state.pipes[name], state.orientation[name], needed
                )
                queue, t_out = step_pipe(
                    queue,
                    displaced_volume=displaced,
                    inlet_temp=t_node,
                    dt=dt,
                    tau_thermal=state.tau_thermal[name],
                    T_ground=cfg.thermal.T_ground,
                )
                state.pipes[name], state.orientation[name] = queue, orient
                heat_gain[name] = rho_cp * displaced * (t_out - t_node) / dt
            else:  # ETS crossover: instantaneous, adds the building's heat
                consumer = ets_consumer[name]
                outcome = ets_model(
                    consumer, t_node, abs(flows[name]) * cfg.water.rho
                )
                ets_out[consumer] = outcome
                t_out = outcome.return_temp
            inflow_vol[dst].append(displaced)
            inflow_temp[dst].append(t_out)

            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                # deterministic order: keep the ready list sorted
                bisect.insort(ready, dst)

    if processed < len(in_degree):
        unprocessed = sorted(n for n, d in in_degree.items() if d > 0)
        raise RuntimeError(f"flow digraph has a cycle through {unprocessed[:5]}")

    # idle pipes still age (and exchange heat with the ground)
    for name, queue in state.pipes.items():
        if name not in heat_gain and name not in directed:
            state.pipes[name], _ = step_pipe(
                queue, 0.0, float("nan"), dt, state.tau_thermal[name],
                cfg.thermal.T_ground,
            )

    arrival = node_temp[dcn.plant.reservoir]
    plant_power = rho_cp * (pump_volume / dt) * (arrival - supply_setpoint)
    return ThermalStepResult(
        node_temp=node_temp,
        ets=ets_out,
        plant_power=plant_power,
        pipe_heat_gain=heat_gain,
    )
