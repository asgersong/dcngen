"""Scenario records writer.

One completed scenario -> a folder of long-format Parquet tables on
logical elements: rows are scenario_id x
time_id x element, paired supply/return channels as columns. Every table
carries ``scenario_id`` so DiTEC-style sharding is a repack, not a schema
change. Ground truth is always written in full. A sensor mask, when one
is drawn (the PoC path), is stored alongside it and never filters the
record; release records materialise NO mask (masking is an ML-phase
view) and their card says so.

Units are SI throughout (temperatures degC, pressures/head m, flows m3/s,
powers W); the card's ``channels`` block documents unit and dtype per
column.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dcngen.config import Config
from dcngen.faults.taxonomy import FaultLabel
from dcngen.orchestrate.plan_runner import PlanScenario
from dcngen.orchestrate.sampler import PlanMeta, ScenarioPlan, ScenarioRow
from dcngen.orchestrate.scenario import (
    EpsNtuScenarioResult,
    FixedDtScenarioResult,
    eps_channels,
)
from dcngen.sensors.mask import SensorMask
from dcngen.thermal.heat_gain import PipeHeatGain
from dcngen.topology.dcnifier import DCNetwork


@dataclass(frozen=True)
class ScenarioRecord:
    """Everything one scenario contributes to the dataset.

    ``sensor_mask`` is ``None`` for release records: ground
    truth only, masks are ML-phase views.
    ``plan_row``/``plan_meta`` echo the scenario's manifest row and the
    release environment into the card.
    """

    scenario_id: int
    seed: int  # what this scenario regenerates from: the PoC path records a
    # per-scenario sub-seed; plan records carry the release master seed (the
    # row regenerates from (master seed, uid) — the card's plan
    # block disambiguates)
    static_draw_id: int
    archetypes: Mapping[str, str]  # consumer junction id -> archetype
    loads: Mapping[str, np.ndarray]  # demanded cooling load [W] per consumer
    ua: Mapping[str, float]  # ETS conductance [W/K] per consumer
    result: EpsNtuScenarioResult | FixedDtScenarioResult
    label: FaultLabel  # completed (realized series attached)
    sensor_mask: SensorMask | None = None
    gate: Mapping | None = None  # validation-gate report (GateReport.as_dict())
    heat_gain: PipeHeatGain | None = None  # per-pipe U' + knob draw;
    # None = legacy flat-scalar semantics (cfg.heat_gain.scalar_U everywhere)
    plan_row: ScenarioRow | None = None  # manifest row echo
    plan_meta: PlanMeta | None = None  # release environment for the card


def plan_scenario_record_from_meta(
    scenario: PlanScenario, meta: PlanMeta
) -> ScenarioRecord:
    """The record of one solved plan row, from the release meta alone —
    the driver's per-worker path, which vouches for the row's
    provenance itself instead of shipping the whole plan to every process.
    Prefer :func:`plan_scenario_record` wherever the plan is at hand.
    """
    row = scenario.row
    return ScenarioRecord(
        scenario_id=row.uid,
        seed=meta.master_seed,
        static_draw_id=row.static_draw_id,
        archetypes=row.archetypes,
        loads=scenario.loads,
        ua=scenario.ua,
        result=scenario.result,
        label=scenario.label,
        sensor_mask=None,
        gate=scenario.report.as_dict(),
        heat_gain=scenario.heat_gain,
        plan_row=row,
        plan_meta=meta,
    )


def plan_scenario_record(scenario: PlanScenario, plan: ScenarioPlan) -> ScenarioRecord:
    """The record of one solved plan row (the guarded bridge).

    The scenario id IS the row uid; the plan supplies the master seed and
    the environment/allocation records for the card. Ground truth only —
    no sensor mask is materialised (#16 deviation).

    Raises:
        ValueError: when the scenario's row is not a row of ``plan``.
    """
    row = scenario.row
    if not (0 <= row.uid < len(plan.rows)) or plan.rows[row.uid] != row:
        raise ValueError(
            f"scenario row uid {row.uid} is not a row of the given plan"
        )
    return plan_scenario_record_from_meta(scenario, plan.meta)


def _write(folder: Path, name: str, frame: pd.DataFrame) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), folder / f"{name}.parquet")


def _node_dynamics(dcn: DCNetwork, record: ScenarioRecord) -> pd.DataFrame:
    r = record.result
    junctions = sorted(dcn.pairing)
    n_steps = len(r.time)
    rows = {
        "scenario_id": np.repeat(np.int64(record.scenario_id), n_steps * len(junctions)),
        "time_id": np.repeat(np.arange(n_steps, dtype=np.int32), len(junctions)),
        "junction": np.tile(np.array(junctions, dtype=object), n_steps),
    }

    def tiled(frame: pd.DataFrame) -> np.ndarray:
        return frame[junctions].to_numpy().reshape(-1)

    rows["supply_temp"] = tiled(r.supply_temp)
    rows["return_temp"] = tiled(r.return_temp)
    rows["pressure_s"] = tiled(r.pressure_s)
    rows["pressure_r"] = tiled(r.pressure_r)
    rows["dT"] = rows["return_temp"] - rows["supply_temp"]

    def consumer_channel(
        values: Mapping[str, np.ndarray] | pd.DataFrame, fill, dtype=float
    ) -> np.ndarray:
        """Consumer-keyed series spread over all junctions, ``fill`` elsewhere."""
        out = np.full((n_steps, len(junctions)), fill, dtype=dtype)
        keys = values.columns if isinstance(values, pd.DataFrame) else values
        for col, j in enumerate(junctions):
            if j in keys:
                out[:, col] = np.asarray(values[j])
        return out.reshape(-1)

    rows["cooling_load"] = consumer_channel(record.loads, np.nan)
    rows["consumer_flow"] = consumer_channel(r.consumer_flow, np.nan)
    rows["ets_return"] = consumer_channel(r.ets_return, np.nan)
    rows["delivered_load"] = consumer_channel(r.delivered_load, np.nan)
    # result-type discrimination lives in scenario.eps_channels (shared
    # with the validation gate)
    rows["unmet"] = consumer_channel(
        eps_channels(r, n_steps)["unmet"], False, dtype=bool
    )
    return pd.DataFrame(rows)


def _edge_dynamics(dcn: DCNetwork, record: ScenarioRecord) -> pd.DataFrame:
    r = record.result
    pipes = sorted(dcn.pipe_pairing)
    n_steps = len(r.time)
    rows = {
        "scenario_id": np.repeat(np.int64(record.scenario_id), n_steps * len(pipes)),
        "time_id": np.repeat(np.arange(n_steps, dtype=np.int32), len(pipes)),
        "pipe": np.tile(np.array(pipes, dtype=object), n_steps),
    }
    sides = {"s": 0, "r": 1}
    channels = {
        "flow": r.pipe_flow,
        "velocity": r.pipe_velocity,
        "headloss": r.pipe_headloss,
        "heat_gain": r.pipe_heat_gain,
    }
    for channel, frame in channels.items():
        for suffix, idx in sides.items():
            cols = [dcn.pipe_pairing[p][idx] for p in pipes]
            rows[f"{channel}_{suffix}"] = frame[cols].to_numpy().reshape(-1)
    return pd.DataFrame(rows)


def _node_static(dcn: DCNetwork, cfg: Config, record: ScenarioRecord) -> pd.DataFrame:
    junctions = sorted(dcn.pairing)
    consumers = dcn.consumers
    return pd.DataFrame(
        {
            "scenario_id": np.repeat(np.int64(record.scenario_id), len(junctions)),
            "junction": np.array(junctions, dtype=object),
            # mirrored twins share the DiTEC elevation; supply side read
            "elevation": [dcn.wn.get_node(dcn.pairing[j][0]).elevation for j in junctions],
            "is_consumer": [j in consumers for j in junctions],
            "archetype": [record.archetypes.get(j) for j in junctions],
            "design_load": [
                consumers[j].design_load if j in consumers else np.nan for j in junctions
            ],
            "design_flow": [
                consumers[j].design_flow if j in consumers else np.nan for j in junctions
            ],
            "ua": [record.ua.get(j, np.nan) for j in junctions],
            # per-ETS dT_design is the global design split until per-ETS
            # draws exist (the schema keeps it per consumer)
            "dT_design": [
                cfg.thermal.dT_design if j in consumers else np.nan for j in junctions
            ],
            # UA sizing margin from the plan row; NaN on
            # legacy (pre-plan) records
            "healthy_ua_factor": [
                record.plan_row.healthy_ua_factor.get(j, np.nan)
                if record.plan_row is not None
                else np.nan
                for j in junctions
            ],
        }
    )


def _edge_static(dcn: DCNetwork, cfg: Config, record: ScenarioRecord) -> pd.DataFrame:
    pipes = sorted(dcn.pipe_pairing)
    links = [dcn.wn.get_link(dcn.pipe_pairing[p][0]) for p in pipes]
    # mirrored twins share the draw geometry, hence the derived U';
    # supply side read, like the elevation in the node table
    if record.heat_gain is not None:
        u_prime = [record.heat_gain.U[dcn.pipe_pairing[p][0]] for p in pipes]
        extrapolated = [
            record.heat_gain.extrapolated[dcn.pipe_pairing[p][0]] for p in pipes
        ]
    else:  # legacy flat-scalar semantics
        u_prime = [float(cfg.heat_gain.scalar_U)] * len(pipes)
        extrapolated = [False] * len(pipes)
    return pd.DataFrame(
        {
            "scenario_id": np.repeat(np.int64(record.scenario_id), len(pipes)),
            "pipe": np.array(pipes, dtype=object),
            "length": [link.length for link in links],
            "diameter": [link.diameter for link in links],
            "roughness": [link.roughness for link in links],
            "U": u_prime,
            "insulation_extrapolated": extrapolated,
        }
    )


def _plant_dynamics(record: ScenarioRecord) -> pd.DataFrame:
    r = record.result
    n_steps = len(r.time)
    extras = eps_channels(r, n_steps)
    return pd.DataFrame(
        {
            "scenario_id": np.repeat(np.int64(record.scenario_id), n_steps),
            "time_id": np.arange(n_steps, dtype=np.int32),
            "plant_power": r.plant_power,
            "plant_flow": r.plant_flow,
            "make_up_flow": extras["make_up_flow"],
            "leak_flow": extras["leak_flow"],
            "converged": extras["converged"],
            "iterations": extras["iterations"],
        }
    )


def _labels(record: ScenarioRecord) -> pd.DataFrame:
    label = record.label
    n_steps = len(label.mask)
    rows = {
        "scenario_id": np.repeat(np.int64(record.scenario_id), n_steps),
        "time_id": np.arange(n_steps, dtype=np.int32),
        "fault_active": np.asarray(label.mask, dtype=bool),
    }
    for name in sorted(label.realized):
        rows[f"realized_{name}"] = np.asarray(label.realized[name], dtype=float)
    return pd.DataFrame(rows)


def _sensor_mask(dcn: DCNetwork, record: ScenarioRecord) -> pd.DataFrame:
    junctions = sorted(dcn.pairing)
    mask = record.sensor_mask
    return pd.DataFrame(
        {
            "scenario_id": np.repeat(np.int64(record.scenario_id), len(junctions)),
            "junction": np.array(junctions, dtype=object),
            "heat_metered": [j in mask.heat_metered for j in junctions],
            "pressure_sensed": [j in mask.pressure_sensed for j in junctions],
        }
    )


# unit per channel; a column missing here fails the write loudly so the
# card's documentation can never silently lag the schema
_UNITS = {
    "scenario_id": "-", "time_id": "step", "junction": "-", "pipe": "-",
    "supply_temp": "degC", "return_temp": "degC", "ets_return": "degC",
    "pressure_s": "m", "pressure_r": "m", "dT": "K",
    "cooling_load": "W", "delivered_load": "W", "design_load": "W",
    "plant_power": "W", "heat_gain_s": "W", "heat_gain_r": "W",
    "consumer_flow": "m3/s", "design_flow": "m3/s", "plant_flow": "m3/s",
    "make_up_flow": "m3/s", "leak_flow": "m3/s", "realized_leak_flow": "m3/s",
    "flow_s": "m3/s", "flow_r": "m3/s",
    "velocity_s": "m/s", "velocity_r": "m/s",
    "headloss_s": "m", "headloss_r": "m",
    "elevation": "m", "length": "m", "diameter": "m",
    "roughness": "-", "U": "W/(m.K)", "ua": "W/K", "dT_design": "K",
    "healthy_ua_factor": "-", "insulation_extrapolated": "-",
    "is_consumer": "-", "archetype": "-", "unmet": "-",
    "converged": "-", "iterations": "-", "fault_active": "-",
    "heat_metered": "-", "pressure_sensed": "-",
}


def _heat_gain_block(cfg: Config, record: ScenarioRecord) -> dict:
    """Card provenance for the U' column: the mode and, in
    derived mode, the knob draw the column regenerates from. A missing
    ``record.heat_gain`` means the legacy flat-scalar semantics."""
    hg = record.heat_gain
    if hg is not None and hg.knobs is not None:
        return {
            "mode": hg.mode,
            "lambda_pur": hg.knobs.lambda_pur,
            "lambda_soil": hg.knobs.lambda_soil,
            "burial_cover": hg.knobs.burial_cover,
        }
    return {"mode": "scalar", "scalar_U": cfg.heat_gain.scalar_U}


def _config_dict(node: object) -> object:
    """The full config as plain JSON (dataclass tree; Paths as strings)."""
    if is_dataclass(node):
        return {f.name: _config_dict(getattr(node, f.name)) for f in fields(node)}
    if isinstance(node, Path):
        return str(node)
    return node


def _channels_doc(tables: Mapping[str, pd.DataFrame]) -> dict:
    return {
        name: {
            column: {"unit": _UNITS[column], "dtype": str(frame[column].dtype)}
            for column in frame.columns
        }
        for name, frame in tables.items()
    }


def _original_ids(dcn: DCNetwork) -> dict[str, str]:
    """WNTR twin/header name -> original DiTEC id, from the pairing (no
    suffix parsing, so ids that themselves end in _s/_r stay safe)."""
    ids = {}
    for j, (s_node, r_node) in dcn.pairing.items():
        ids[s_node] = j
        ids[r_node] = j
    # headers are built as f"{source}_s"/f"{source}_r" by the DCN-ifier
    ids[dcn.plant.supply_header] = dcn.plant.supply_header[:-2]
    ids[dcn.plant.return_header] = dcn.plant.return_header[:-2]
    return ids


def _card(
    dcn: DCNetwork,
    cfg: Config,
    record: ScenarioRecord,
    tables: Mapping[str, pd.DataFrame],
) -> dict:
    r = record.result
    # original DiTEC adj_list, reconstructed from the supply mirror
    original = _original_ids(dcn)
    adj_list = []
    for pipe in sorted(dcn.pipe_pairing):
        link = dcn.wn.get_link(dcn.pipe_pairing[pipe][0])
        adj_list.append(
            [original[link.start_node_name], original[link.end_node_name], pipe]
        )

    flow_directions = {}
    for pipe in sorted(dcn.pipe_pairing):
        p_s, p_r = dcn.pipe_pairing[pipe]
        flow_directions[pipe] = {
            side: int(np.sign(np.sign(r.pipe_flow[name].to_numpy()).sum()))
            for side, name in (("s", p_s), ("r", p_r))
        }

    n_steps = len(r.time)
    extras = eps_channels(r, n_steps)
    converged, iterations = extras["converged"], extras["iterations"]
    label = record.label
    # gate report if the scenario was gated (exceedance metadata etc.)
    validation = {} if record.gate is None else {"validation": dict(record.gate)}
    return {
        **validation,
        "network": dcn.name,
        "scenario_id": record.scenario_id,
        "seed": record.seed,
        "static_draw_id": record.static_draw_id,
        "dt": cfg.solver.dt,
        "n_steps": n_steps,
        "adj_list": adj_list,
        "pairing": {j: list(dcn.pairing[j]) for j in sorted(dcn.pairing)},
        "pipe_pairing": {
            p: list(dcn.pipe_pairing[p]) for p in sorted(dcn.pipe_pairing)
        },
        "flow_directions": flow_directions,
        "convergence": {
            "unconverged_steps": int((~converged).sum()),
            "max_iterations_used": int(np.max(iterations)),
            "budget_ok": bool(
                (~converged).mean() <= cfg.validation.max_unconverged_frac
            ),
        },
        "fault": {
            "kind": label.kind.value,
            "location": label.location,
            "element": label.element,
            "side": label.side,
            "severity": label.severity,
            "onset": label.onset,
            "offset": label.offset,
            # temporal variant: derived from the label so
            # plan-driven and legacy records agree — whole-horizon means
            # active from t = 0 through every step (chronic/whole)
            "onset_step": (
                None if label.onset is None else int(round(label.onset / cfg.solver.dt))
            ),
            "whole_horizon": bool(
                label.onset == 0.0 and np.asarray(label.mask, dtype=bool).all()
            ),
        },
        "sensors": (
            {
                "materialised": True,
                "plant_sensed": record.sensor_mask.plant_sensed,
                "consumer_coverage": cfg.sensors.consumer_coverage,
                "pressure_coverage": cfg.sensors.pressure_coverage,
            }
            if record.sensor_mask is not None
            # ground truth only: masks are ML-phase views (#16 deviation)
            else {"materialised": False}
        ),
        **(
            {
                # the COMPLETE release meta (asdict, never field-copied:
                # community membership is what makes the allocation records
                # interpretable, and new PlanMeta fields must not silently
                # miss the card) + this scenario's manifest row. The row's
                # fault.onset_step keeps the manifest convention (None =
                # whole-horizon); the fault block above is label-derived.
                "plan": {
                    **asdict(record.plan_meta),
                    "row": asdict(record.plan_row),
                }
            }
            if record.plan_meta is not None and record.plan_row is not None
            else {}
        ),
        "config": _config_dict(cfg),
        # loop water stored energy at the horizon ends: the
        # storage term that closes the gate's first-law balance. Absolute
        # values sit on a 0 degC datum — only the difference is physical.
        "storage": {
            "stored_energy_start_J": float(r.stored_energy_start),
            "stored_energy_end_J": float(r.stored_energy_end),
        },
        "heat_gain": _heat_gain_block(cfg, record),
        # plant params: the record is self-describing
        "plant": {
            # original DiTEC id of the junction hosting the Plant, so
            # readers never re-derive it from header naming conventions
            "site": original[dcn.plant.supply_header],
            "anchoring": dcn.anchoring,  # "ditec" | "rail_fallback"
            "pump_design_flow": dcn.pump_design[0],
            "pump_design_head": dcn.pump_design[1],
            "ditec_reservoir_head": dcn.ditec_reservoir_head,
            "T_supply_design": cfg.thermal.T_supply,
            "T_ground": cfg.thermal.T_ground,
            "elements": {
                "reservoir": dcn.plant.reservoir,
                "pump": dcn.plant.pump,
                "stub": dcn.plant.stub,
                "supply_header": dcn.plant.supply_header,
                "return_header": dcn.plant.return_header,
            },
        },
        "channels": _channels_doc(tables),
    }


def write_scenario(
    folder: Path, dcn: DCNetwork, cfg: Config, record: ScenarioRecord
) -> Path:
    """Write one scenario's records; returns the folder path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = {
        "node_dynamics": _node_dynamics(dcn, record),
        "edge_dynamics": _edge_dynamics(dcn, record),
        "node_static": _node_static(dcn, cfg, record),
        "edge_static": _edge_static(dcn, cfg, record),
        "plant_dynamics": _plant_dynamics(record),
        "labels": _labels(record),
    }
    if record.sensor_mask is not None:  # #16 deviation: ground truth only
        tables["sensor_mask"] = _sensor_mask(dcn, record)
    # build (and thereby validate) the card BEFORE any file lands on disk:
    # a schema/documentation mismatch must not leave a partial folder
    card = _card(dcn, cfg, record, tables)
    for name, frame in tables.items():
        _write(folder, name, frame)
    (folder / "card.json").write_text(json.dumps(card, indent=2, sort_keys=True))
    return folder
