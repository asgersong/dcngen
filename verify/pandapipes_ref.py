"""Verification harness: an independent pandapipes rebuild of the loop.

This module rebuilds the **steady fixed-dT** DCN operating point in
pandapipes and cross-checks the transport fields no internal test can
vouch for -- supply/return temperature fields, pipe flows, node
pressures, and the plant energy balance.

Why a second engine exists at all. WNTR + our thermal layer stay the
*single generator*; pandapipes is an
oracle and nothing else. The division of validation labour is:

    transport dynamics (delay, plug arithmetic) -> analytic kernel tests
    eps-NTU ETS algebra                         -> closed-form unit tests
    leak model                                  -> WNTR's own (EPANET lineage)
    steady transport fields                     -> THIS harness

so the rebuild deliberately stops at the steady fixed-dT point: no
eps-NTU, no control loop, no time marching on the pandapipes side. Doing
more would be a second generator.

Model mapping (dcngen -> pandapipes)::

    junction j_s / j_r     ->  junction, height_m = elevation
    supply / return pipe   ->  pipe, u_w_per_m2k = U'/(pi D), text_k = T_ground
    plant stub             ->  pipe (no special case: it is 1 m of plumbing)
    ETS crossover (FCV)    ->  flow_control (m_dot) + heat_exchanger (qext = -Q)
    reservoir + pump       ->  circ_pump_const_pressure at the supply setpoint

Two model differences are deliberate:

* **Friction law.** EPANET/WNTR solve Hazen-Williams; pandapipes solves
  Darcy-Weisbach. :func:`equivalent_roughness` calibrates a per-pipe
  absolute roughness that reproduces each pipe's HW headloss under
  pandapipes' own ``64/Re + lambda_nikuradse`` correlation at the dcngen
  operating point ("calibrated at the design point").
  Two consequences, both stated plainly in
  :func:`compare_steady` and :func:`equivalent_roughness`: after this
  calibration the *pressure* comparison tests the loop assembly and the
  headloss bookkeeping rather than the friction correlation itself --
  a failure then indicates an implementation
  bug, not friction-law physics -- and on the DiTEC substrate a third of
  the pipes need less friction than the ``64/Re`` term alone, so no
  Darcy-Weisbach roughness represents them at all. That is not a harness
  defect but a measured property of the inherited network; it is
  counted in the report rather than clamped away.
* **Heat discretisation.** Our plug model integrates the pipe ODE exactly
  (``T_out = T_g + (T_in - T_g) exp(-U'L/(m cp))``); pandapipes uses an
  implicit finite-section scheme whose per-section error is second order
  in the pipe NTU ``U'L/(m cp)``. On pre-insulated DC pipe that NTU is
  ~1e-2, so the difference should be invisible -- and measurably is:
  sweeping ``sections`` from 1 to 40 on Hanoi draw 0 moves the compared
  temperature field by 6e-14 K, i.e. not at all. ``pipe_sections`` is
  therefore margin against a future network with a much larger NTU, not a
  quantity this dataset's comparison is sensitive to.

Run it::

    python verify/pandapipes_ref.py --draw 0 --json verify_report.json
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandapipes as pp
import pandas as pd

from dcngen.config import Config, load_config
from dcngen.hydraulics.wntr_model import solve_steady
from dcngen.orchestrate.scenario import FixedDtScenarioResult, run_fixed_dt_scenario
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import DCNetwork, build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

# EPANET's gravity: WNTR reports head in metres of water under g = 9.81,
# so the head <-> pressure conversion here must use the same value or the
# comparison would carry a systematic 0.02 % offset.
G = 9.81  # m/s2
KELVIN = 273.15

# pandapipes' friction law (pf/derivative_calculation.py:209) is the SUM
#     lambda = 64/Re + lambda_nikuradse,   lambda_nikuradse = 1/(2 log10(3.71 d/k))^2
# -- the laminar term is added unconditionally, at every Reynolds number.
# That sum is what equivalent_roughness inverts, and its 64/Re term is a
# hard floor: no roughness can produce less friction than that. The floor
# is the reason 20 of Hanoi's 69 pipes are unrepresentable.
_NIKURADSE_C = 3.71

# Roughness is inverted from the Nikuradse term alone, so that term must
# stay strictly positive: a pipe calibrated exactly onto the laminar floor
# would need k = 0, which sends log10(k/(3.71 d)) to -inf and makes
# pandapipes' Jacobian singular. Pipes at or below the floor are pinned to
# this residual instead, which puts their total friction as close to the
# floor as float64 allows -- k goes as 10^(-1/(2 sqrt(lambda))), so 1e-5
# gives k ~ 1e-158 m, while 1e-6 would give 1e-500 and silently underflow
# back to the zero this floor exists to avoid.
_MIN_NIKURADSE_LAMBDA = 1e-5

# Below this volumetric flow a pipe carries nothing worth calibrating
# against (its headloss is numerical noise): mirrors the thermal solver's
# _FLOW_EPS so both layers agree on what "not flowing" means.
_FLOW_EPS = 1e-12  # m3/s


class VerificationError(RuntimeError):
    """The rebuild could not be posed or solved -- not a physics failure."""


# ---------------------------------------------------------------------------
# Friction-law bridge
# ---------------------------------------------------------------------------


def equivalent_roughness(
    diameter: np.ndarray,
    length: np.ndarray,
    flow: np.ndarray,
    headloss: np.ndarray,
    rho: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pipe absolute roughness ``k`` [m] matching the given headloss.

    Given each pipe's Hazen-Williams headloss at the dcngen operating
    point, return the absolute roughness at which pandapipes reproduces
    that same headloss at that same velocity. The Darcy friction factor
    the headloss implies is

        lambda_req = 2 g D h / (L v^2)

    and pandapipes' law is the sum ``64/Re + lambda_nikuradse``, so the
    roughness must carry only the difference, which inverts in closed
    form (no iteration -- only ``k`` is unknown):

        k = 3.71 D * 10^( -1 / (2 sqrt(lambda_req - 64/Re)) )

    Args:
        diameter: inner diameter [m], per pipe.
        length: pipe length [m].
        flow: signed volumetric flow [m3/s] at the calibration point.
        headloss: head drop [m] along the flow direction at that point.
        rho: density [kg/m3]; mu: dynamic viscosity [Pa s].

    Returns:
        ``(k, unreachable)``. A pipe is ``unreachable`` when its required
        friction lies at or below ``64/Re`` -- pandapipes adds that
        laminar term at every Reynolds number, so no roughness, however
        small, can produce less loss. On the DiTEC substrate that is not
        an edge case: the sampled Hazen-Williams C is far outside the
        physical range (Hanoi C ~ 1576-2477 against ~130 for real pipe),
        which puts roughly a third of Hanoi's pipes below the floor.
        Those pipes are pinned to the smallest usable Nikuradse term and
        will show MORE headloss than the reference; the count is reported
        rather than hidden.
        Pipes with no flow are also flagged: they have no operating point
        to calibrate against and, carrying no flow, no headloss to get
        wrong.
    """
    d = np.asarray(diameter, dtype=float)
    L = np.asarray(length, dtype=float)
    q = np.abs(np.asarray(flow, dtype=float))
    h = np.abs(np.asarray(headloss, dtype=float))

    area = np.pi / 4.0 * d**2
    v = np.divide(q, area, out=np.zeros_like(q), where=area > 0.0)
    live = (q > _FLOW_EPS) & (v > 0.0) & (h > 0.0)

    lam_req = np.zeros_like(d)
    np.divide(2.0 * G * d * h, L * v**2, out=lam_req, where=live)
    re = np.full_like(d, np.inf)
    np.divide(rho * v * d, mu, out=re, where=live)

    lam_nikuradse = lam_req - 64.0 / re
    unreachable = ~live | (lam_nikuradse < _MIN_NIKURADSE_LAMBDA)
    lam_nikuradse = np.maximum(lam_nikuradse, _MIN_NIKURADSE_LAMBDA)
    k = _NIKURADSE_C * d * 10.0 ** (-1.0 / (2.0 * np.sqrt(lam_nikuradse)))
    return k, unreachable


# ---------------------------------------------------------------------------
# The rebuild
# ---------------------------------------------------------------------------


def _chilled_water(cfg: Config):
    """A constant-property fluid pinned to the config's water block.

    pandapipes' library water carries temperature-dependent density,
    viscosity and specific heat. Using it would make the comparison a test
    of *property models* rather than of transport, so the oracle is handed
    exactly the constants the generator uses.
    """
    return pp.create_constant_fluid(
        name="dcn_chilled_water",
        fluid_type="liquid",
        density=cfg.water.rho,
        viscosity=cfg.water.mu,
        heat_capacity=cfg.water.cp,
        molar_mass=18.015,
        compressibility=0.0,
        der_compressibility=0.0,
    )


@dataclass(frozen=True)
class ReferenceModel:
    """The pandapipes rebuild plus the index maps needed to read it back."""

    net: object  # pandapipesNet
    junction_index: dict[str, int]  # WNTR node name -> pandapipes junction
    pipe_index: dict[str, int]  # WNTR pipe name -> pandapipes pipe
    ets_index: dict[str, int]  # consumer junction id -> heat_exchanger index
    roughness: pd.Series  # calibrated k [m] per WNTR pipe
    roughness_unreachable: pd.Series  # bool per pipe (see equivalent_roughness)


def build_reference(
    dcn: DCNetwork,
    cfg: Config,
    result: FixedDtScenarioResult,
    pipe_U: dict[str, float],
    loads: dict[str, float],
    supply_setpoint: float,
    sections: int,
) -> ReferenceModel:
    """Assemble the pandapipes twin of ``dcn`` at ``result``'s steady point.

    The consumer mass flows and heat loads are *imposed* (flow controls +
    heat exchangers), exactly as the FCV crossovers impose them on the
    WNTR side; what pandapipes then solves independently is how those
    flows distribute through the looped network, what the resulting
    pressures are, and how the temperature field advects and mixes.

    Args:
        dcn: the built DCN whose geometry is mirrored.
        cfg: configuration (water properties, ground temperature).
        result: the steady fixed-dT scenario the rebuild is calibrated to
            and compared against; supplies the operating-point flows and
            headlosses for :func:`equivalent_roughness`.
        pipe_U: per-pipe linear U' [W/(m K)] -- the SAME map the generator
            ran with, so heat gain is compared, not re-derived.
        loads: cooling load [W] per consumer junction id.
        supply_setpoint: chiller outlet temperature [degC].
        sections: internal sub-elements per pipe (heat discretisation).
    """
    wn = dcn.wn
    net = pp.create_empty_network(fluid=_chilled_water(cfg))
    t_ground_k = cfg.thermal.T_ground + KELVIN

    junction_index: dict[str, int] = {}
    for name in wn.junction_name_list:
        junction_index[name] = pp.create_junction(
            net,
            pn_bar=1.0,
            tfluid_k=supply_setpoint + KELVIN,
            height_m=float(wn.get_node(name).elevation),
            name=name,
        )
    # The reservoir is a real junction on this side: our plant is a triple
    # (stub -> reservoir -> pump), and keeping the reservoir node makes the
    # stub's heat gain visible to the comparison instead of collapsing it
    # into the pump.
    junction_index[dcn.plant.reservoir] = pp.create_junction(
        net,
        pn_bar=1.0,
        tfluid_k=cfg.thermal.T_return_design + KELVIN,
        height_m=float(wn.get_node(dcn.plant.supply_header).elevation),
        name=dcn.plant.reservoir,
    )

    pipe_names = sorted(wn.pipe_name_list)
    steady = result.pipe_flow.iloc[-1]
    headloss = result.pipe_headloss.iloc[-1]
    diameters = np.array([wn.get_link(p).diameter for p in pipe_names])
    lengths = np.array([wn.get_link(p).length for p in pipe_names])
    k, unreachable = equivalent_roughness(
        diameters,
        lengths,
        steady[pipe_names].to_numpy(),
        headloss[pipe_names].to_numpy(),
        cfg.water.rho,
        cfg.water.mu,
    )

    pipe_index: dict[str, int] = {}
    for i, name in enumerate(pipe_names):
        link = wn.get_link(name)
        # u_w_per_m2k is applied by pandapipes over pi * D_outer * L, and
        # D_outer defaults to the inner diameter, so dividing our linear
        # U' by pi*D here reproduces U'*L exactly (pf/derivative_toolbox).
        pipe_index[name] = pp.create_pipe_from_parameters(
            net,
            from_junction=junction_index[link.start_node_name],
            to_junction=junction_index[link.end_node_name],
            length_km=link.length / 1000.0,
            inner_diameter_mm=link.diameter * 1000.0,
            k_mm=float(k[i]) * 1000.0,
            sections=sections,
            u_w_per_m2k=pipe_U[name] / (math.pi * link.diameter),
            text_k=t_ground_k,
            name=name,
        )

    ets_index: dict[str, int] = {}
    cp, dT = cfg.water.cp, cfg.thermal.dT_design
    for junction, consumer in sorted(dcn.consumers.items()):
        load = float(loads[junction])
        supply_node, return_node = dcn.pairing[junction]
        mid = pp.create_junction(
            net,
            pn_bar=1.0,
            tfluid_k=supply_setpoint + KELVIN,
            height_m=float(wn.get_node(supply_node).elevation),
            name=f"{consumer.ets_link}_mid",
        )
        pp.create_flow_control(
            net,
            from_junction=junction_index[supply_node],
            to_junction=mid,
            controlled_mdot_kg_per_s=load / (cp * dT),
            name=f"{consumer.ets_link}_fc",
        )
        # qext_w > 0 withdraws heat from the network; a building's cooling
        # load ADDS heat to the chilled water, hence the negative sign.
        ets_index[junction] = pp.create_heat_exchanger(
            net,
            from_junction=mid,
            to_junction=junction_index[return_node],
            qext_w=-load,
            inner_diameter_mm=wn.get_link(consumer.ets_link).diameter * 1000.0,
            name=consumer.ets_link,
        )

    # The plant as a *pressure* boundary, not a mass one. Fixing the pump's
    # mass flow as well would restate what the consumer flow controls
    # already impose, leaving the mass-slack variable undetermined and the
    # Jacobian exactly singular. Constant-pressure is also the truer mirror
    # of our own plant: WNTR's pump sits on a fixed head curve while the
    # FCVs absorb the surplus head, which is exactly how a flow_control
    # branch behaves here (its pressure drop is a free variable).
    #
    # One free pressure DOF, pinned where the generator pins it -- at the
    # supply header -- so the comparison tests the pressure *distribution*.
    # The absolute level is a boundary condition in both models.
    header_pressure = float(result.hydraulics.pressure[dcn.plant.supply_header])
    to_bar = cfg.water.rho * G / 1e5
    pp.create_circ_pump_const_pressure(
        net,
        return_junction=junction_index[dcn.plant.reservoir],
        flow_junction=junction_index[dcn.plant.supply_header],
        p_flow_bar=header_pressure * to_bar,
        plift_bar=dcn.pump_design[1] * to_bar,
        t_flow_k=supply_setpoint + KELVIN,
        name="plant",
    )
    return ReferenceModel(
        net=net,
        junction_index=junction_index,
        pipe_index=pipe_index,
        ets_index=ets_index,
        roughness=pd.Series(k, index=pipe_names),
        roughness_unreachable=pd.Series(unreachable, index=pipe_names),
    )


def solve_reference(model: ReferenceModel, friction_model: str = "nikuradse") -> None:
    """Run the pandapipes steady hydraulic + thermal solve, in place.

    ``mode="sequential"`` solves hydraulics first and the temperature
    field on the converged flow -- the same decoupling the generator
    relies on, and exact here because the fluid has constant properties.

    ``friction_model="nikuradse"`` matches what :func:`equivalent_roughness`
    inverts. Colebrook would add a Reynolds-dependent term the closed-form
    inversion does not account for, so the two must stay in step.
    """
    try:
        pp.pipeflow(
            model.net,
            mode="sequential",
            friction_model=friction_model,
            iter=300,
            tol_p=1e-8,
            tol_m=1e-8,
        )
    except Exception as exc:  # pandapipes raises bare PipeflowNotConverged
        raise VerificationError(f"pandapipes did not converge: {exc}") from exc


# ---------------------------------------------------------------------------
# Reading the oracle back, in the generator's own conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceFields:
    """The solved pandapipes state, expressed the way dcngen expresses it."""

    node_temp: pd.Series  # WNTR node name -> [degC]
    pressure: pd.Series  # WNTR node name -> [m]
    pipe_flow: pd.Series  # WNTR pipe name -> [m3/s], signed vs nominal direction
    plant_power: float  # [W] removed by the chiller
    heat_gain: float  # [W] total ground -> water over all pipes


def reference_fields(
    dcn: DCNetwork, cfg: Config, model: ReferenceModel, loads: dict[str, float]
) -> ReferenceFields:
    """Read the solved net back into dcngen's units, signs and node names.

    Total heat gain is taken as ``plant_power - sum(loads)`` rather than by
    summing per-pipe enthalpy rises. That is not a shortcut: it is the only
    convention-free way to ask pandapipes the question. Its per-pipe
    ``t_from_k`` / ``t_outlet_k`` are reported against each pipe's *nominal*
    orientation and, on a sectioned pipe, against its internal sub-elements,
    so a hand-rolled sum silently mis-signs every reversed pipe and
    mis-scales every sectioned one. The difference of two quantities both
    models agree on the meaning of -- what the chiller removes, and what the
    buildings put in -- is exact, and it is also the form the first law
    takes in our own gate.

    ``loads`` therefore enters only as that subtraction; every other number
    here is read from the solved net.
    """
    net = model.net
    rj, rp = net.res_junction, net.res_pipe
    rho, cp = cfg.water.rho, cfg.water.cp

    node_temp = pd.Series(
        {n: float(rj.t_k[i]) - KELVIN for n, i in model.junction_index.items()}
    )
    pressure = pd.Series(
        {
            n: float(rj.p_bar[i]) * 1e5 / (rho * G)
            for n, i in model.junction_index.items()
        }
    )
    # pandapipes' from/to were built as WNTR's start/end, so mdot_from is
    # already signed against the nominal direction dcngen records.
    pipe_flow = pd.Series(
        {n: float(rp.mdot_from_kg_per_s[i]) / rho for n, i in model.pipe_index.items()}
    )

    # Plant duty from pandapipes' OWN solved circulation and its own inlet
    # and outlet temperatures, rather than from a mass flow re-derived from
    # the loads. That keeps the energy comparison independent on both
    # factors, and folds in a free mass-balance check: the pump's solved
    # mdot has to come back equal to the sum of the imposed consumer flows,
    # since nothing else moves water in a closed loop.
    pump = model.net.res_circ_pump_pressure.iloc[0]
    circulated = abs(float(pump.mdot_from_kg_per_s))
    plant_power = circulated * cp * (float(pump.t_from_k) - float(pump.t_to_k))
    return ReferenceFields(
        node_temp=node_temp,
        pressure=pressure,
        pipe_flow=pipe_flow,
        plant_power=plant_power,
        heat_gain=plant_power - sum(loads.values()),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldComparison:
    """One compared field family and how far the two engines disagree."""

    name: str
    unit: str
    metric: str  # "abs" | "rel"
    tolerance: float
    deviation: float  # the gated number, in `metric`'s units
    worst: str  # element carrying it
    passed: bool
    detail: str = ""  # anything the number alone would hide

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        shown = (
            f"{self.deviation:.4g} {self.unit}"
            if self.metric == "abs"
            else f"{self.deviation:.4%}"
        )
        tol = (
            f"{self.tolerance:.4g} {self.unit}"
            if self.metric == "abs"
            else f"{self.tolerance:.4%}"
        )
        out = f"  [{mark}] {self.name:22s} {shown:>14s}  (tol {tol}"
        out += f", worst {self.worst})" if self.worst else ")"
        return out + (f"\n         {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class VerificationReport:
    """The verification verdict for one network and one static draw."""

    network: str
    static_draw_id: int
    comparisons: tuple[FieldComparison, ...]
    unreachable_pipes: int
    total_pipes: int
    passed: bool

    def text(self) -> str:
        head = (
            f"pandapipes cross-check -- {self.network}, static draw "
            f"{self.static_draw_id}\n"
        )
        body = "\n".join(c.line() for c in self.comparisons)
        tail = (
            f"\n  friction: {self.unreachable_pipes}/{self.total_pipes} pipes below "
            f"pandapipes' 64/Re floor"
        )
        verdict = "\n\nverification " + ("PASSED" if self.passed else "FAILED")
        return head + body + tail + verdict

    def to_dict(self) -> dict:
        return {
            "network": self.network,
            "static_draw_id": self.static_draw_id,
            "passed": self.passed,
            "unreachable_pipes": self.unreachable_pipes,
            "total_pipes": self.total_pipes,
            "comparisons": [
                {
                    "name": c.name,
                    "unit": c.unit,
                    "metric": c.metric,
                    "tolerance": c.tolerance,
                    "deviation": c.deviation,
                    "worst": c.worst,
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in self.comparisons
            ],
        }


def _abs_comparison(
    name: str, unit: str, dcngen: pd.Series, other: pd.Series, tol: float, detail: str = ""
) -> FieldComparison:
    dev = (other[dcngen.index] - dcngen).abs()
    worst = str(dev.idxmax())
    return FieldComparison(
        name=name,
        unit=unit,
        metric="abs",
        tolerance=tol,
        deviation=float(dev.max()),
        worst=worst,
        passed=bool(dev.max() <= tol),
        detail=detail,
    )


def compare_steady(
    dcn: DCNetwork,
    cfg: Config,
    result: FixedDtScenarioResult,
    fields: ReferenceFields,
    model: ReferenceModel,
    loads: dict[str, float],
) -> VerificationReport:
    """Score the oracle against the generator on the four #17 field families.

    Tolerances come from ``cfg.verification`` and were fixed by the #17
    resolution as *thermal-strict, hydraulic-documented*: temperatures and
    the plant energy balance are contract-anchored and tight; pressures are
    compared against a deliberately looser band because the two engines do
    not share a friction law.

    What the pressure and flow numbers do and do not prove. The equivalent
    roughness of :func:`equivalent_roughness` matches each pipe's headloss
    at this operating point, so the residual disagreement is what the
    *exponent* difference does as flow redistributes -- Hazen-Williams
    loses head as ``Q^1.852``, Darcy-Weisbach as ``Q^2``. These two
    comparisons therefore test the loop assembly, the mirror's direction
    convention and the headloss bookkeeping; they do not independently test
    the friction correlation, and are not meant to (#17: "a failure then
    indicates an implementation bug, not friction-law physics").
    """
    v = cfg.verification
    pipes = sorted(dcn.wn.pipe_name_list)
    comparisons: list[FieldComparison] = []

    # --- temperatures: the field this harness exists for ---------------
    dc_temp = {}
    for j, (supply_node, return_node) in dcn.pairing.items():
        dc_temp[supply_node] = float(result.supply_temp.iloc[-1][j])
        dc_temp[return_node] = float(result.return_temp.iloc[-1][j])
    comparisons.append(
        _abs_comparison(
            "node temperature",
            "K",
            pd.Series(dc_temp),
            fields.node_temp,
            v.temperature_tol,
        )
    )

    # --- pipe flows -----------------------------------------------------
    dc_flow = result.pipe_flow.iloc[-1][pipes]
    pp_flow = fields.pipe_flow[pipes]
    dev = (pp_flow - dc_flow).abs()
    # Normalised by what the plant circulates, not by each pipe's own flow:
    # this asks "how much of the network's throughput got routed
    # differently", which is the question a distribution comparison is
    # actually about, and it does not blow up on a pipe whose flow is
    # near zero because it sits at a loop's stagnation point.
    circulated = abs(float(result.plant_flow[-1]))
    # The per-pipe view is reported alongside rather than gated on: it is
    # where the friction-law difference actually shows, and burying it
    # under a normalised number would be the wrong kind of tidy.
    per_pipe = (dev / dc_flow.abs()).replace([np.inf, -np.inf], np.nan)
    detail = f"of plant circulation {circulated:.4g} m3/s"
    if per_pipe.notna().any():
        worst_pipe = str(per_pipe.idxmax())
        detail += (
            f"; worst single-pipe relative shift {per_pipe.max():.2%} on "
            f"{worst_pipe} (carrying {abs(dc_flow[worst_pipe]) / circulated:.2%} "
            "of it) -- the loop split moving under the Q^1.852 vs Q^2 exponent"
        )
    comparisons.append(
        FieldComparison(
            name="pipe flow",
            unit="",
            metric="rel",
            tolerance=v.flow_rel_tol,
            deviation=float(dev.max() / circulated),
            worst=str(dev.idxmax()),
            passed=bool(dev.max() / circulated <= v.flow_rel_tol),
            detail=detail,
        )
    )

    # --- pressures ------------------------------------------------------
    dc_pressure = {}
    for j, (supply_node, return_node) in dcn.pairing.items():
        dc_pressure[supply_node] = float(result.pressure_s.iloc[-1][j])
        dc_pressure[return_node] = float(result.pressure_r.iloc[-1][j])
    comparisons.append(
        _abs_comparison(
            "node pressure",
            "m",
            pd.Series(dc_pressure),
            fields.pressure,
            v.pressure_tol,
            detail=(
                "anchored at the supply header; band is documented, not "
                "contract-anchored (differing friction laws)"
            ),
        )
    )

    # --- plant energy balance + the heat gain it implies -----------------
    dc_plant = float(result.plant_power[-1])
    residual = abs(fields.plant_power - dc_plant) / abs(dc_plant)
    dc_gain = float(result.pipe_heat_gain.iloc[-1].sum())
    gain_dev = abs(fields.heat_gain - dc_gain) / abs(dc_gain) if dc_gain else 0.0
    comparisons.append(
        FieldComparison(
            name="plant energy balance",
            unit="",
            metric="rel",
            tolerance=v.energy_rel_tol,
            deviation=residual,
            worst="",
            passed=bool(residual <= v.energy_rel_tol),
            detail=(
                f"dcngen {dc_plant:.6g} W vs pandapipes {fields.plant_power:.6g} W; "
                f"the pipe heat gain this implies agrees to {gain_dev:.3%} "
                f"({dc_gain:.4g} W, {dc_gain / dc_plant:.2%} of plant load) -- "
                "the sharpest available test of the per-pipe U'"
            ),
        )
    )

    return VerificationReport(
        network=dcn.name,
        static_draw_id=-1,  # filled by verify_network, which knows the draw
        comparisons=tuple(comparisons),
        unreachable_pipes=int(model.roughness_unreachable.sum()),
        total_pipes=len(pipes),
        passed=all(c.passed for c in comparisons),
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def steady_march_steps(dcn: DCNetwork, cfg: Config, flows: pd.Series) -> int:
    """How many steps to march before the fields are worth comparing.

    pandapipes solves the steady state directly; dcngen marches to it. The
    march is settled once every plug placed by the initial condition has
    been flushed, which takes one transit of the *slowest* pipe --
    ``V_pipe / Q`` -- and that spans 40 to 165 steps across Hanoi draws
    alone. Deriving the count from the network beats any fixed constant,
    which would silently under-run the slow draws.
    """
    transits = [
        (np.pi / 4.0 * link.diameter**2 * link.length) / abs(flows[name])
        for name in dcn.wn.pipe_name_list
        for link in [dcn.wn.get_link(name)]
        if abs(flows[name]) > _FLOW_EPS
    ]
    if not transits:
        raise VerificationError(f"{dcn.name}: no pipe carries flow at the design point")
    steps = math.ceil(cfg.verification.steady_turnovers * max(transits) / cfg.solver.dt)
    return max(2, min(steps, cfg.verification.steady_max_steps))


def _assert_steady(result: FixedDtScenarioResult, tol: float) -> float:
    """Confirm the march actually settled; the comparison assumes it.

    If it has not, any disagreement downstream is the generator's own
    transient rather than a physics difference, so this checks rather than
    assumes. The residual never reaches zero: plugs cross node boundaries
    at step edges, which keeps a jitter floor of order 1e-3 K that more
    steps do not remove (hence ``steady_drift_tol`` sits at 2 % of the
    compared tolerance, not at machine precision).
    """
    drift = max(
        float(np.abs(f.iloc[-1] - f.iloc[-2]).max())
        for f in (result.supply_temp, result.return_temp)
    )
    if drift > tol:
        raise VerificationError(
            f"dcngen has not reached steady state: last-step temperature drift "
            f"{drift:.3g} K exceeds {tol:.3g} K -- run more steps before comparing"
        )
    return drift


def verify_network(
    cfg: Config,
    network: str | None = None,
    static_draw_id: int | None = None,
    seed: int = 0,
) -> VerificationReport:
    """Build, run and cross-check one network at one static draw.

    The generator side runs its own fixed-dT path at constant design
    loads -- the same ``run_fixed_dt_scenario`` the dataset was made with,
    not a reimplementation -- and the oracle is handed that run's flow
    field for calibration and its per-pipe U' map, so heat gain is
    *compared* rather than independently re-derived.
    """
    v = cfg.verification
    network = network or cfg.poc.network
    draw = cfg.poc.static_scenario_id if static_draw_id is None else static_draw_id

    ditec = load_ditec_network(cfg.paths.ditec_data / network, draw)
    dcn = build_dcn(ditec, cfg)
    heat_gain = resolve_pipe_heat_gain(dcn, cfg, np.random.default_rng(seed))
    loads = {j: c.design_load for j, c in dcn.consumers.items()}

    n_steps = steady_march_steps(dcn, cfg, solve_steady(dcn).flow)
    result = run_fixed_dt_scenario(dcn, cfg, loads, n_steps, pipe_U=heat_gain.U)
    _assert_steady(result, v.steady_drift_tol)

    model = build_reference(
        dcn, cfg, result, heat_gain.U, loads, cfg.thermal.T_supply, v.pipe_sections
    )
    solve_reference(model)
    fields = reference_fields(dcn, cfg, model, loads)
    report = compare_steady(dcn, cfg, result, fields, model, loads)
    return VerificationReport(
        network=network,
        static_draw_id=draw,
        comparisons=report.comparisons,
        unreachable_pipes=report.unreachable_pipes,
        total_pipes=report.total_pipes,
        passed=report.passed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=None, help="DiTEC network folder name")
    ap.add_argument("--draw", type=int, default=None, help="static scenario id")
    ap.add_argument("--seed", type=int, default=0, help="heat-gain knob draw")
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    args = ap.parse_args()

    report = verify_network(load_config(), args.network, args.draw, args.seed)
    print(report.text())
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
