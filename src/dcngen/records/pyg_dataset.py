"""Typed heterogeneous graph view of one scenario folder.

``load_scenario`` reads ONLY what :mod:`dcngen.records.writer` put on disk
(``card.json`` + the parquet tables) — no writer, network, or result object
is needed in memory, so any copy of a records folder loads identically.

Graph schema:

- node types ``supply`` / ``return``: index ``i`` on both sides is the same
  logical node, so ``return.x - supply.x`` is a positional
  subtraction — dT without joins. Order: sorted original junction ids, then
  the plant header last (``is_plant`` flags it; its junction dynamics are
  unrecorded -> NaN).
- edge types ``(supply, pipe, supply)`` in DiTEC ``adj_list`` orientation,
  ``(return, pipe, return)`` reversed (the mirror), ``(supply, ets,
  return)`` one crossover per consumer, and ``(return, plant, supply)``
  closing the loop at the plant header so make-up flow is on-graph.
- dynamic tensors are float32 shaped ``[n_elements, n_steps, n_channels]``
  (time is dim 1; per-store ``channels`` names dim 2); labels and masks are
  1-D over the same time axis, plus a per-store ``fault_location_mask``
  marking the faulted element (the localisation target). Units are SI,
  identical to the parquet columns, documented per column in the card's
  ``channels`` block. Parquet stays the lossless ground truth — float32 is
  the graph-view dtype, and flow signs survive the cast at any physically
  meaningful magnitude (only sub-denormal ~1e-38 flows would flush to 0).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

_NODE_CHANNELS = {"supply": ["supply_temp", "pressure_s"],
                  "return": ["return_temp", "pressure_r"]}
_NODE_STATIC = [
    "elevation", "design_load", "design_flow", "ua", "dT_design",
    # UA sizing margin; NaN when the writer had no plan row (the PoC path).
    # Folders written before the column existed are not supported —
    # unlike the sensor-mask fallback, which is disk-backward-compatible.
    "healthy_ua_factor",
]
_PIPE_CHANNELS = ["flow", "velocity", "headloss", "heat_gain"]
_PIPE_STATIC = ["length", "diameter", "roughness", "U", "insulation_extrapolated"]
_ETS_CHANNELS = ["cooling_load", "consumer_flow", "delivered_load",
                 "ets_return", "dT", "unmet"]
_PLANT_CHANNELS = ["plant_power", "plant_flow", "make_up_flow", "leak_flow"]
_PLANT_STATIC = ["pump_design_flow", "pump_design_head",
                 "ditec_reservoir_head", "T_supply_design"]

# torch.tensor (never from_numpy) throughout: it always copies, so pandas'
# read-only views are safe to hand it


def _stack(frame: pd.DataFrame, key: str, ids: list[str],
           channels: list[str]) -> torch.Tensor:
    """Long-format dynamics -> float32 ``[len(ids), n_steps, len(channels)]``.

    Pivots on (time_id, ``key``) so the result is row-order independent;
    ids absent from the table (the plant header) come out NaN.
    """
    per_channel = [
        frame.pivot(index="time_id", columns=key, values=channel)
        .reindex(columns=ids)
        .to_numpy(dtype=float)
        .T
        for channel in channels
    ]
    return torch.tensor(np.stack(per_channel, axis=-1), dtype=torch.float32)


def _static(frame: pd.DataFrame, key: str, ids: list[str],
            channels: list[str]) -> torch.Tensor:
    """One-row-per-element statics -> float32 ``[len(ids), len(channels)]``."""
    rows = frame.set_index(key).reindex(ids)
    return torch.tensor(rows[channels].to_numpy(dtype=float), dtype=torch.float32)


def _column(frame: pd.DataFrame, column: str, dtype) -> torch.Tensor:
    """One table column -> 1-D tensor (frame must already be time-sorted)."""
    return torch.tensor(frame[column].to_numpy(), dtype=dtype)


# which store hosts which fault kinds (the fault taxonomy). Routing by
# kind, not by the label's node/edge element: DiTEC junction and pipe ids
# share a namespace (junction "3" and pipe "3" can both exist), so element
# alone cannot say whether location "3" is a node or a pipe. SENSOR faults
# corrupt the measurement layer only — no physical location on any store.
_STORE_KINDS = {
    "node": ("leak",),
    "pipe": ("blockage",),
    "ets": ("ets_valve", "fouling", "bypass"),
    "plant": ("pump", "plant_setpoint"),
}


def _location_mask(ids: list[str], fault: dict, store: str,
                   side: str | None = None) -> torch.Tensor:
    """Bool ``[len(ids)]`` localisation target: True at the faulted element.

    ``store`` keys :data:`_STORE_KINDS`; ``side`` is the store's
    supply/return half where that distinction exists (a side-less fault
    label marks both halves). All False when the fault lives on another
    store — and everywhere, for clean-normal scenarios.
    """
    applies = fault["kind"] in _STORE_KINDS[store] and (
        side is None or fault["side"] is None or fault["side"] == side
    )
    hit = fault["location"] if applies else None
    return torch.tensor([i == hit for i in ids])


def _graph_attrs(data: HeteroData, card: dict, labels: pd.DataFrame,
                 plant_dyn: pd.DataFrame) -> None:
    """Scenario identity, fault label, and per-timestep flag tensors."""
    data.network = card["network"]
    data.scenario_id = card["scenario_id"]
    data.seed = card["seed"]
    data.static_draw_id = card["static_draw_id"]
    data.dt = card["dt"]
    data.n_steps = card["n_steps"]
    data.time = torch.arange(card["n_steps"], dtype=torch.float64) * card["dt"]
    data.fault = card["fault"]
    # per-scenario conditionable statics;
    # None where a legacy (pre-plan) card has no such field
    data.T_supply = card["plant"]["T_supply_design"]
    data.T_ground = card["plant"].get("T_ground")
    plan_row = card.get("plan", {}).get("row", {})
    data.tier = plan_row.get("tier")
    data.scenario_class = plan_row.get("scenario_class")

    data.fault_active = _column(labels, "fault_active", torch.bool)
    for column in labels.columns:
        if column.startswith("realized_"):
            setattr(data, column, _column(labels, column, torch.float32))
    data.converged = _column(plant_dyn, "converged", torch.bool)
    data.iterations = _column(plant_dyn, "iterations", torch.int64)


def _node_stores(data: HeteroData, card: dict, tables: dict,
                 ids: list[str], consumers: list[str]) -> None:
    """The supply/return twin stores, index-paired over ``ids``."""
    site = card["plant"]["site"]
    node_static = tables["node_static"].set_index("junction").reindex(ids[:-1])
    statics = _static(tables["node_static"], "junction", ids, _NODE_STATIC)
    archetypes = [
        None if pd.isna(a) else a for a in node_static["archetype"].tolist()
    ] + [None]

    if "sensor_mask" in tables:
        sensors = tables["sensor_mask"].set_index("junction").reindex(ids[:-1])
        plant_sensed = bool(card["sensors"]["plant_sensed"])
        masks = {
            channel: torch.tensor(np.append(
                sensors[channel].to_numpy(dtype=bool), plant_sensed
            ))
            for channel in ("heat_metered", "pressure_sensed")
        }
    else:
        # ground truth only: no mask was
        # materialised, so everything is observable; ML-phase views mask
        full = torch.ones(len(ids), dtype=torch.bool)
        masks = {"heat_metered": full, "pressure_sensed": full}

    for side, channels in _NODE_CHANNELS.items():
        store = data[side]
        store.x = _stack(tables["node_dynamics"], "junction", ids, channels)
        store.channels = channels
        store.static = statics
        store.static_channels = _NODE_STATIC
        store.junction = ids
        store.is_plant = torch.tensor([j == site for j in ids])
        store.is_consumer = torch.tensor([j in consumers for j in ids])
        store.archetype = archetypes
        store.heat_metered = masks["heat_metered"]
        store.pressure_sensed = masks["pressure_sensed"]
        store.fault_location_mask = _location_mask(
            ids, card["fault"], "node", side=side
        )


def _pipe_stores(data: HeteroData, card: dict, tables: dict,
                 index: dict[str, int]) -> None:
    """Directed supply/return pipe stores in mirror orientation."""
    pipes = sorted(card["pipe_pairing"])
    endpoints = {pipe: (a, b) for a, b, pipe in card["adj_list"]}
    pipe_static = _static(tables["edge_static"], "pipe", pipes, _PIPE_STATIC)

    for suffix, kind in (("s", "supply"), ("r", "return")):
        store = data[kind, "pipe", kind]
        heads = [endpoints[p][0] for p in pipes]
        tails = [endpoints[p][1] for p in pipes]
        if kind == "return":  # the return mirror is reversed
            heads, tails = tails, heads
        store.edge_index = torch.tensor(
            [[index[h] for h in heads], [index[t] for t in tails]]
        )
        store.edge_attr = _stack(
            tables["edge_dynamics"], "pipe", pipes,
            [f"{channel}_{suffix}" for channel in _PIPE_CHANNELS],
        )
        store.channels = _PIPE_CHANNELS
        store.static = pipe_static
        store.static_channels = _PIPE_STATIC
        store.pipe = pipes
        store.flow_sign = torch.tensor(
            [card["flow_directions"][p][suffix] for p in pipes],
            dtype=torch.int8,
        )
        store.fault_location_mask = _location_mask(
            pipes, card["fault"], "pipe", side=kind
        )


def _crossover_stores(data: HeteroData, card: dict, tables: dict,
                      index: dict[str, int], consumers: list[str],
                      plant_dyn: pd.DataFrame) -> None:
    """ETS crossovers (one per consumer) + the loop-closing plant edge."""
    ets = data["supply", "ets", "return"]
    rows = [index[j] for j in consumers]
    ets.edge_index = torch.tensor([rows, rows])
    ets.edge_attr = _stack(
        tables["node_dynamics"], "junction", consumers, _ETS_CHANNELS
    )
    ets.channels = _ETS_CHANNELS
    ets.junction = consumers
    # ETS faults (fouling, valve) are labelled by consumer junction id
    ets.fault_location_mask = _location_mask(consumers, card["fault"], "ets")

    site = card["plant"]["site"]
    plant = data["return", "plant", "supply"]
    plant.edge_index = torch.tensor([[index[site]], [index[site]]])
    plant.edge_attr = torch.tensor(
        plant_dyn[_PLANT_CHANNELS].to_numpy(dtype=float), dtype=torch.float32
    )[None, :, :]
    plant.channels = _PLANT_CHANNELS
    plant.static = torch.tensor(
        [[float(card["plant"][channel]) for channel in _PLANT_STATIC]]
    )
    plant.static_channels = _PLANT_STATIC
    plant.fault_location_mask = _location_mask([site], card["fault"], "plant")


def load_scenario(folder: Path | str) -> HeteroData:
    """Load one written scenario folder as a :class:`HeteroData` graph."""
    folder = Path(folder)
    card = json.loads((folder / "card.json").read_text())
    names = ["node_dynamics", "edge_dynamics", "node_static",
             "edge_static", "plant_dynamics", "labels"]
    if (folder / "sensor_mask.parquet").exists():  # absent on release
        names.append("sensor_mask")  # PoC-path records only
    tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in names}

    junctions = sorted(card["pairing"])
    ids = junctions + [card["plant"]["site"]]  # plant header last, both sides
    index = {j: i for i, j in enumerate(ids)}
    node_static = tables["node_static"].set_index("junction")
    consumers = [j for j in junctions if node_static.loc[j, "is_consumer"]]
    labels = tables["labels"].sort_values("time_id")
    plant_dyn = tables["plant_dynamics"].sort_values("time_id")

    data = HeteroData()
    _graph_attrs(data, card, labels, plant_dyn)
    _node_stores(data, card, tables, ids, consumers)
    _pipe_stores(data, card, tables, index)
    _crossover_stores(data, card, tables, index, consumers, plant_dyn)
    return data
