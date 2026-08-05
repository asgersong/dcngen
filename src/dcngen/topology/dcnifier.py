"""Supply/return mirror: DiTEC topology + static draw -> closed-loop DCN.

Every junction ``j`` mirrors into ``j_s`` (supply) and
``j_r`` (return) at the same elevation; every pipe ``p: a -> b`` mirrors into
``p_s: a_s -> b_s`` and ``p_r: b_r -> a_r`` — the return pipe is *reversed*
so that in normal operation the same water yields the same signed flow on
both sides. The DiTEC reservoir node mirrors into the supply/return headers,
joined by the Plant triple::

    return header --(plant_stub)--> plant_reservoir --(plant_pump)--> supply header

Each consumer (junction with positive base demand) gets one FCV crossover
``ets_j: j_s -> j_r`` whose setting carries the primary flow. Design flows
follow flow equivalence: design volumetric flow = DiTEC base demand,
hence design load = rho * D_j * c_p * dT_design.
"""

import math
from dataclasses import dataclass

import wntr

from dcngen.config import Config
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.topology.ditec_loader import DitecNetwork


class DCNBuildError(ValueError):
    """The topology cannot be turned into a well-posed DCN."""


@dataclass(frozen=True)
class Consumer:
    """One ETS crossover: the consumer at original junction ``junction``."""

    junction: str
    ets_link: str
    design_flow: float  # primary volumetric flow at design [m3/s]
    design_load: float  # cooling load at design [W]


@dataclass(frozen=True)
class PlantLayout:
    """WNTR element names making up the Plant (triple + headers)."""

    reservoir: str
    pump: str
    stub: str
    supply_header: str
    return_header: str


@dataclass
class DCNetwork:
    """A closed-loop DCN: the WNTR model plus the mirror's metadata."""

    wn: wntr.network.WaterNetworkModel
    name: str
    pairing: dict[str, tuple[str, str]]  # junction j -> (j_s, j_r)
    pipe_pairing: dict[str, tuple[str, str]]  # pipe p -> (p_s, p_r)
    consumers: dict[str, Consumer]  # junction id -> Consumer
    plant: PlantLayout
    edge_kind: dict[str, str]  # link -> supply_pipe|return_pipe|ets|plant_pump|plant_stub
    ditec_reservoir_head: float  # source head of the pinned draw [m]
    pump_design: tuple[float, float]  # (design flow [m3/s], design head [m])
    anchoring: str = "ditec"  # pressure-anchor policy used by _size_plant:
    # "ditec" (supply header at the draw's source head) or
    # "rail_fallback" (minimal rail-satisfying head)


# Nominal FCV bore sized for ~2 m/s at design flow. No hydraulic effect
# (minor loss is 0; an active FCV enforces its flow setting regardless);
# exists because WNTR requires valves to have a diameter.
_FCV_NOMINAL_VELOCITY = 2.0  # m/s

# Big-M probe head for plant sizing (see _size_plant below): any value above
# the true loop loss yields the identical flow field, so the probe is exact,
# not approximate. Numerical device, not a physical constant.
PROBE_PUMP_HEAD = 1.0e4  # m

_PUMP_CURVE = "plant_pump_curve"


def build_dcn(net: DitecNetwork, cfg: Config) -> DCNetwork:
    """Mirror a loaded DiTEC network into a closed-loop DCN.

    The returned DCN is ready to solve: plant sizing (one probe solve via
    ``hydraulics.solve_steady``) has already run, so callers never see an
    unsized network. That makes this module depend on ``hydraulics`` — a
    deliberate trade against module independence: geometry construction
    stays independently testable, and only this seam couples the two.
    """
    if len(net.reservoirs) != 1:
        raise DCNBuildError(
            f"{net.name}: expected exactly 1 reservoir (the Plant site), "
            f"got {list(net.reservoirs)} — multi-source Plants are future work"
        )
    source = net.reservoirs[0]
    draw = net.draw

    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.required_pressure = cfg.solver.required_pressure
    wn.options.hydraulic.minimum_pressure = cfg.solver.minimum_pressure
    wn.options.time.duration = 0  # repeated steady solves

    pairing: dict[str, tuple[str, str]] = {}
    for j in net.junctions:
        elevation = draw.junction_elevation[j]
        wn.add_junction(f"{j}_s", base_demand=0.0, elevation=elevation)
        wn.add_junction(f"{j}_r", base_demand=0.0, elevation=elevation)
        pairing[j] = (f"{j}_s", f"{j}_r")

    # Headers: the mirrored source node. Their elevation only shifts the
    # pressure *reading* at the Plant, never the flow field; use the mean of
    # the junctions the source feeds so the reading is representative.
    neighbours = [
        p.start if p.end == source else p.end
        for p in net.pipes
        if source in (p.start, p.end)
    ]
    if not neighbours:
        raise DCNBuildError(f"{net.name}: source node {source} has no incident pipe")
    header_elev = sum(draw.junction_elevation[n] for n in neighbours) / len(neighbours)
    supply_header, return_header = f"{source}_s", f"{source}_r"
    wn.add_junction(supply_header, base_demand=0.0, elevation=header_elev)
    wn.add_junction(return_header, base_demand=0.0, elevation=header_elev)

    edge_kind: dict[str, str] = {}
    pipe_pairing: dict[str, tuple[str, str]] = {}
    for p in net.pipes:
        for name, start, end, kind in (
            (f"{p.id}_s", f"{p.start}_s", f"{p.end}_s", "supply_pipe"),
            (f"{p.id}_r", f"{p.end}_r", f"{p.start}_r", "return_pipe"),
        ):
            wn.add_pipe(
                name,
                start,
                end,
                length=draw.pipe_length[p.id],
                diameter=draw.pipe_diameter[p.id],
                roughness=draw.pipe_roughness[p.id],
            )
            edge_kind[name] = kind
        pipe_pairing[p.id] = (f"{p.id}_s", f"{p.id}_r")

    water = cfg.water
    consumers: dict[str, Consumer] = {}
    for j in net.junctions:
        design_flow = draw.base_demand.get(j, 0.0)
        if design_flow <= 0.0:
            continue
        ets = f"ets_{j}"
        bore = (4.0 * design_flow / (math.pi * _FCV_NOMINAL_VELOCITY)) ** 0.5
        wn.add_valve(
            ets,
            f"{j}_s",
            f"{j}_r",
            diameter=bore,
            valve_type="FCV",
            initial_setting=design_flow,
        )
        edge_kind[ets] = "ets"
        consumers[j] = Consumer(
            junction=j,
            ets_link=ets,
            design_flow=design_flow,
            design_load=water.rho * design_flow * water.cp * cfg.thermal.dT_design,
        )
    if not consumers:
        raise DCNBuildError(f"{net.name}: no junction has positive base demand")

    # Plant triple. The reservoir starts at the draw's source head; sizing
    # (_size_plant) re-anchors it to H_ditec - H0 so the supply header sits
    # at the DiTEC source head at design conditions.
    ditec_head = draw.reservoir_head[source]
    wn.add_reservoir("plant_reservoir", base_head=ditec_head)
    total_design_flow = sum(c.design_flow for c in consumers.values())
    wn.add_curve(_PUMP_CURVE, "HEAD", [(total_design_flow, PROBE_PUMP_HEAD)])
    wn.add_pump(
        "plant_pump",
        "plant_reservoir",
        supply_header,
        pump_type="HEAD",
        pump_parameter=_PUMP_CURVE,
    )
    # Negligible-loss plumbing by construction: 1 m long, twice the fattest
    # network pipe, smoothest sampled roughness.
    wn.add_pipe(
        "plant_stub",
        return_header,
        "plant_reservoir",
        length=1.0,
        diameter=2.0 * max(draw.pipe_diameter.values()),
        roughness=max(draw.pipe_roughness.values()),
    )
    edge_kind["plant_pump"] = "plant_pump"
    edge_kind["plant_stub"] = "plant_stub"

    dcn = DCNetwork(
        wn=wn,
        name=net.name,
        pairing=pairing,
        pipe_pairing=pipe_pairing,
        consumers=consumers,
        plant=PlantLayout(
            reservoir="plant_reservoir",
            pump="plant_pump",
            stub="plant_stub",
            supply_header=supply_header,
            return_header=return_header,
        ),
        edge_kind=edge_kind,
        ditec_reservoir_head=ditec_head,
        pump_design=(total_design_flow, PROBE_PUMP_HEAD),
    )
    _size_plant(dcn, cfg)
    return dcn


def _size_plant(dcn: DCNetwork, cfg: Config) -> None:
    """Size the pump from one big-M probe solve; re-anchor the reservoir.

    With every ETS an FCV, any sufficient pump head yields the *identical*
    flow field — surplus head is throttled away at the valves and head
    levels shift uniformly on the supply side. So one probe solve at
    ``PROBE_PUMP_HEAD`` measures the tightest consumer's surplus exactly, and

        H0 = PROBE_PUMP_HEAD - min_FCV_drop + ets.min_dp

    is the smallest design head that leaves every ETS its minimum
    differential pressure (DC practice is 100-150 kPa per ETS).
    The single-point HEAD curve passes exactly through (Q_design, H0), so at
    design flow the pump delivers H0 and anchoring the reservoir at
    ``H_ditec - H0`` puts the supply header exactly at the DiTEC source head
    — the pressure regime the draw was validated in.
    Draws whose H0 exceeds that head's headroom fall back to the minimal
    rail-satisfying anchor instead (recorded on ``DCNetwork.anchoring``).
    """
    q_design, _ = dcn.pump_design
    state = solve_steady(dcn)
    tol = cfg.validation.mass_balance_rel_tol
    undelivered = {
        c.junction: state.flow[c.ets_link]
        for c in dcn.consumers.values()
        if abs(state.flow[c.ets_link] - c.design_flow) > tol * c.design_flow
    }
    if undelivered:
        raise DCNBuildError(
            f"{dcn.name}: FCVs cannot deliver design flows even at the "
            f"{PROBE_PUMP_HEAD} m probe head: {undelivered}"
        )

    drops = []
    for j in dcn.consumers:
        supply_node, return_node = dcn.pairing[j]
        drops.append(float(state.head[supply_node] - state.head[return_node]))
    min_drop = min(drops)
    # plain floats: WNTR's AML rejects numpy scalars in curve points / heads
    h0 = float(PROBE_PUMP_HEAD - min_drop + cfg.ets.min_dp)
    if h0 <= 0.0:
        raise DCNBuildError(
            f"{dcn.name}: probe solve implies a non-positive pump head ({h0:.2f} m)"
        )

    reservoir_head = float(dcn.ditec_reservoir_head - h0)
    # rails the anchor must clear regardless of policy: PDD full delivery
    # at the loop's high point, and the pump-suction minimum
    floor = (
        max(dcn.wn.get_node(n).elevation for n in dcn.wn.junction_name_list)
        + cfg.solver.required_pressure
    )
    suction_rail = (
        dcn.wn.get_node(dcn.plant.supply_header).elevation
        + cfg.validation.plant_suction_min
    )
    if reservoir_head >= max(floor, suction_rail):
        anchoring = "ditec"  # the draw's validated pressure regime
    else:
        # on draws whose pump head exceeds the DiTEC source head's
        # headroom (~3 % of Hanoi draws), anchor at the minimal
        # rail-satisfying head instead — the loop keeps its flow field
        # (one free pressure DOF) at the lowest pressure level the
        # rails admit
        reservoir_head = float(max(floor, suction_rail))
        anchoring = "rail_fallback"
    dcn.anchoring = anchoring

    dcn.wn.get_curve(_PUMP_CURVE).points = [(q_design, h0)]
    dcn.wn.get_node(dcn.plant.reservoir).base_head = reservoir_head
    dcn.pump_design = (q_design, h0)
