"""Steady hydraulic solves of a DCNetwork under WNTRSimulator PDD.

The scenario loop runs one steady solve per timestep,
mutating FCV settings between solves; this module owns that seam. It only
duck-types :class:`~dcngen.topology.dcnifier.DCNetwork` (never imports it at
runtime) so the dependency stays one-directional: topology -> hydraulics.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import wntr

if TYPE_CHECKING:  # runtime import would be circular (dcnifier imports us)
    from dcngen.topology.dcnifier import DCNetwork


@dataclass(frozen=True)
class HydraulicState:
    """One steady solve: node and link fields keyed by WNTR element name.

    ``flow`` is signed against each link's nominal direction (supply pipes:
    the DiTEC direction; return pipes: the reversed mirror direction, so a
    healthy loop shows equal signs on both sides). ``velocity`` [m/s] covers
    pipes only.
    """

    pressure: pd.Series  # node -> [m]
    head: pd.Series  # node -> [m]
    flow: pd.Series  # link -> [m3/s], signed
    velocity: pd.Series  # pipe -> [m/s], >= 0
    leak_flow: pd.Series  # leaked node -> [m3/s] out; empty when no leaks


def solve_steady(
    dcn: "DCNetwork",
    consumer_flows: Mapping[str, float] | None = None,
    leak_areas: Mapping[str, float] | None = None,
    leak_discharge_coeff: float = 0.75,
) -> HydraulicState:
    """Run one steady WNTRSimulator PDD solve of the closed loop.

    Args:
        dcn: the built DCN (its WaterNetworkModel is configured for PDD and
            duration 0 by the DCN-ifier).
        consumer_flows: primary flow per consumer junction id [m3/s]; the
            corresponding FCV settings are mutated before solving. ``None``
            solves at the current settings (design flows, right after build).
        leak_areas: active leak orifice area [m2] per WNTR node name; the
            leaks exist only for this solve (the model is restored after).
        leak_discharge_coeff: orifice C_d shared by this solve's leaks
            (WNTR leak model: Q = C_d * A * sqrt(2 g h)). The default is
            WNTR's own ``add_leak`` default; callers with a configured
            value (``cfg.faults.leak_discharge_coeff``) should pass it.
    """
    wn = dcn.wn
    if consumer_flows is not None:
        for junction, flow in consumer_flows.items():
            wn.get_link(dcn.consumers[junction].ets_link).initial_setting = flow

    leaked = []
    if leak_areas:
        for node, area in leak_areas.items():
            j = wn.get_node(node)
            j.add_leak(wn, area=area, discharge_coeff=leak_discharge_coeff)
            leaked.append(j)
    try:
        wn.reset_initial_values()  # also forces every _leak_status False
        for j in leaked:
            # activate for this steady solve; the public setter is a timed
            # control, which a duration-0 solve never fires
            j._leak_status = True
        results = wntr.sim.WNTRSimulator(wn).run_sim()
    finally:
        for j in leaked:
            j.remove_leak(wn)
            j._leak_status = False

    if leak_areas:
        leak_demand = results.node["leak_demand"].iloc[-1]
        leak_flow = pd.Series({n: float(leak_demand[n]) for n in leak_areas})
    else:
        leak_flow = pd.Series(dtype=float)

    flow = results.link["flowrate"].iloc[-1]
    velocity = pd.Series(
        {
            name: abs(flow[name])
            / (math.pi / 4.0 * wn.get_link(name).diameter ** 2)
            for name in wn.pipe_name_list
        }
    )
    return HydraulicState(
        pressure=results.node["pressure"].iloc[-1],
        head=results.node["head"].iloc[-1],
        flow=flow,
        velocity=velocity,
        leak_flow=leak_flow,
    )
