"""Shared fixtures: a synthetic DiTEC-WDN parquet folder.

``write_ditec_folder`` reproduces the on-disk format of the DiTEC-WDN dump
(verified against ``hanoi_8GB_1Y``, 2026-07-14): one parquet file per input
kind, one column per network element plus a ``scenario_id`` column (float in
the real dump), the topology as an ``adj_list`` JSON list under the ``attrs``
key of the parquet footer metadata, and the demand *time series* row-sharded
over several ``junction_base_demand-*-dynamic_input.parquet`` files with a
``time_id`` column.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def write_mutated_default(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    """Copy ``configs/default.yaml`` through ``mutate(raw_dict)`` to a temp
    file — the shared way to probe strict loading and non-default programs."""
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    mutate(raw)
    out = tmp_path / "mutated.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out

# Mini-network: reservoir "1", junctions "2","3","4", pipes 1: 2->1, 2: 2->3, 3: 3->4.
MINI_ADJ_LIST = [["2", "1", "1"], ["2", "3", "2"], ["3", "4", "3"]]

# Static draws for scenario_ids 0 and 1 (SI units; scenario 1 differs so tests
# can prove the loader pins by id, not by row order).
MINI_STATICS = {
    "pipe_diameter": {"1": [0.6, 0.7], "2": [0.5, 0.55], "3": [0.4, 0.45]},
    "pipe_length": {"1": [1000.0, 1100.0], "2": [800.0, 900.0], "3": [600.0, 650.0]},
    "pipe_roughness": {"1": [130.0, 140.0], "2": [120.0, 125.0], "3": [110.0, 115.0]},
    "junction_elevation": {"2": [2.0, 2.5], "3": [1.0, 1.5], "4": [0.0, 0.5]},
    "reservoir_base_head": {"1": [60.0, 65.0]},
}

# Demand series for scenario 0, split over two shards (time 0-2 and 3-5) the
# way the real dump row-shards its year.  Time-means: 2 -> 0.35, 3 -> 0.25,
# 4 -> 0.15 m3/s.
MINI_DEMAND_SHARDS = [
    {
        "time_id": [0.0, 1.0, 2.0],
        "2": [0.30, 0.35, 0.40],
        "3": [0.20, 0.25, 0.30],
        "4": [0.10, 0.15, 0.20],
    },
    {
        "time_id": [3.0, 4.0, 5.0],
        "2": [0.40, 0.35, 0.30],
        "3": [0.30, 0.25, 0.20],
        "4": [0.20, 0.15, 0.10],
    },
]
MINI_BASE_DEMAND = {"2": 0.35, "3": 0.25, "4": 0.15}


def _write_with_attrs(path: Path, data: dict, adj_list: list) -> None:
    table = pa.Table.from_pydict(data)
    meta = dict(table.schema.metadata or {})
    meta[b"attrs"] = json.dumps({"adj_list": adj_list}).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def write_ditec_folder(
    folder: Path,
    adj_list: list = MINI_ADJ_LIST,
    statics: dict = MINI_STATICS,
    demand_shards: list = MINI_DEMAND_SHARDS,
    demand_scenario_ids: tuple = (0, 1),
) -> Path:
    """Write a synthetic network folder in the DiTEC dump format."""
    folder.mkdir(parents=True, exist_ok=True)
    for kind, cols in statics.items():
        n_scen = len(next(iter(cols.values())))
        data = {"scenario_id": [float(s) for s in range(n_scen)]}
        data.update({el: list(vals) for el, vals in cols.items()})
        _write_with_attrs(folder / f"{kind}-0-static_input.parquet", data, adj_list)
    for i, shard in enumerate(demand_shards):
        n_time = len(shard["time_id"])
        data = {"scenario_id": [], "time_id": []}
        data.update({el: [] for el in shard if el != "time_id"})
        for sid in demand_scenario_ids:
            data["scenario_id"] += [float(sid)] * n_time
            data["time_id"] += list(shard["time_id"])
            for el, vals in shard.items():
                if el != "time_id":
                    data[el] += list(vals)
        _write_with_attrs(
            folder / f"junction_base_demand-{i}-dynamic_input.parquet", data, adj_list
        )
    return folder


@pytest.fixture
def mini_ditec_folder(tmp_path) -> Path:
    """A synthetic DiTEC network folder named ``mini_net``."""
    return write_ditec_folder(tmp_path / "mini_net")


# The records-seam scenario shared by the writer and loader test modules
# (tickets #9/#10): one small leaked eps-NTU run, fully labelled and masked.
MINI_RECORD_N_STEPS = 12
MINI_RECORD_SEED = 21
MINI_RECORD_SCENARIO_ID = 7
MINI_RECORD_ARCHETYPES = {"2": "office", "3": "residential", "4": "hotel"}
MINI_RECORD_LEAK = {"junction": "3", "side": "return"}


def build_mini_leak_record(nets_dir: Path, cfg):
    """Run the shared scenario end-to-end -> (dcn, completed ScenarioRecord).

    Imports live inside so collecting unrelated test modules stays cheap.
    """
    import numpy as np

    from dcngen.faults.injector import inject_abrupt_leak
    from dcngen.loads.load_generator import generate_loads
    from dcngen.orchestrate.scenario import run_eps_ntu_scenario
    from dcngen.records.writer import ScenarioRecord
    from dcngen.sensors.mask import draw_sensor_mask
    from dcngen.thermal.ets import design_ua
    from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
    from dcngen.topology.dcnifier import build_dcn
    from dcngen.topology.ditec_loader import load_ditec_network

    folder = write_ditec_folder(nets_dir / "mini_net")
    dcn = build_dcn(load_ditec_network(folder, scenario_id=0), cfg)

    rng = np.random.default_rng(MINI_RECORD_SEED)
    times = np.arange(MINI_RECORD_N_STEPS) * cfg.solver.dt
    heat_gain = resolve_pipe_heat_gain(dcn, cfg, rng)  # per-scenario knob draw
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    loads = generate_loads(
        MINI_RECORD_ARCHETYPES, design, times, cfg.loads, rng
    )
    ua = {j: design_ua(design[j], cfg) for j in design}
    leak = inject_abrupt_leak(dcn, cfg, times, rng, **MINI_RECORD_LEAK)
    result = run_eps_ntu_scenario(
        dcn, cfg, loads, n_steps=MINI_RECORD_N_STEPS, ua=ua, leak=leak,
        pipe_U=heat_gain.U,
    )

    record = ScenarioRecord(
        scenario_id=MINI_RECORD_SCENARIO_ID,
        seed=MINI_RECORD_SEED,
        static_draw_id=0,
        archetypes=MINI_RECORD_ARCHETYPES,
        loads=loads,
        ua=ua,
        result=result,
        label=leak.label.with_realized(leak_flow=result.leak_flow),
        sensor_mask=draw_sensor_mask(dcn, cfg, rng),
        heat_gain=heat_gain,
    )
    return dcn, record


def shrink_sampler_tiers(d: dict) -> None:
    """Mini-release sampler shape shared by the sampler and plan-runner
    tests: 3-consumer network, tiny counts, short horizons."""
    d["sampler"]["network"] = "mini_net"
    d["sampler"]["static_draw_count"] = 2
    d["sampler"]["bulk"] = {
        "horizon": 14400.0, "normal": 4, "leak": 7, "fouling": 4, "bypass": 5,
    }
    d["sampler"]["week"] = {
        "horizon": 28800.0, "normal": 2, "leak": 2, "fouling": 1, "bypass": 1,
    }
    d["sampler"]["month"] = {
        "horizon": 43200.0, "normal": 1, "leak": 1, "fouling": 1, "bypass": 1,
    }
