"""The PoC pair.

``run_poc`` produces, on the configured network (Hanoi for the real PoC):

- a **clean-normal** scenario carrying the explicit no-fault label, and
- an **abrupt-leak** scenario with full ground-truth labels,

each generated with a fresh DCN (no state leaks between scenarios),
gated by :mod:`dcngen.orchestrate.validate`, written to records, and
reloaded as a typed heterogeneous graph whose structure and labels are
re-checked from the loaded form. The run passes
iff both scenarios pass the gate and both reloads check out.

Determinism: three integer sub-seeds (archetype assignment, normal, leak)
are derived from ``cfg.seed`` and recorded in each scenario's card; loads
and labels regenerate bitwise from them, solved fields within solver
tolerance.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch_geometric.data import HeteroData

from dcngen.config import Config
from dcngen.faults.injector import inject_abrupt_leak
from dcngen.faults.taxonomy import FaultLabel
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.orchestrate.validate import GateReport, validate_scenario
from dcngen.records.pyg_dataset import load_scenario
from dcngen.records.writer import ScenarioRecord, write_scenario
from dcngen.sensors.mask import draw_sensor_mask
from dcngen.thermal.ets import design_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import DCNetwork, build_dcn
from dcngen.topology.ditec_loader import DitecNetwork, load_ditec_network


@dataclass(frozen=True)
class PocScenario:
    """One half of the PoC pair, fully processed."""

    name: str  # "normal" | "leak"
    seed: int  # recorded sub-seed this scenario regenerates from
    folder: Path  # written records
    report: GateReport
    graph: HeteroData  # the reloaded typed graph
    graph_ok: bool  # reload checks (counts, labels) passed
    lines: tuple[str, ...]  # printable check report


@dataclass(frozen=True)
class PocRun:
    """The whole PoC: passed == the branch's definition of done."""

    scenarios: tuple[PocScenario, ...]
    passed: bool
    lines: tuple[str, ...]


def _graph_checks(
    graph: HeteroData, dcn: DCNetwork, label: FaultLabel, n_steps: int
) -> list[tuple[str, bool]]:
    """Reload checks (ticket criterion 3), evaluated on the LOADED graph."""
    n_nodes = len(dcn.pairing) + 1  # junctions + plant header
    mask = np.asarray(label.mask, dtype=bool)
    checks = [
        (
            "node counts + twin pairing",
            graph["supply"].num_nodes == n_nodes
            and graph["return"].num_nodes == n_nodes
            and graph["supply"].junction == graph["return"].junction,
        ),
        (
            "edge types complete",
            graph["supply", "pipe", "supply"].edge_index.shape[1]
            == len(dcn.pipe_pairing)
            and graph["supply", "ets", "return"].edge_index.shape[1]
            == len(dcn.consumers)
            and graph["return", "plant", "supply"].edge_index.shape[1] == 1,
        ),
        (
            "label kind matches",
            graph.fault["kind"] == label.kind.value,
        ),
        (
            "fault mask aligns with the time axis",
            graph.fault_active.shape == (n_steps,)
            and bool((graph.fault_active.numpy() == mask).all()),
        ),
    ]
    if label.location is not None:
        located = graph[label.side].fault_location_mask.numpy()
        junctions = graph[label.side].junction
        checks.append(
            (
                "leak located on the graph",
                bool(located[junctions.index(label.location)])
                and int(located.sum()) == 1,
            )
        )
        # a leak that never materialized would otherwise pass every rule
        # vacuously (zero make-up == zero leak) and still print PASS
        realized = graph.realized_leak_flow.numpy()
        checks.append(
            (
                "leak realized while active",
                bool(realized[mask].min() > 0.0)
                and bool((realized[~mask] == 0.0).all()),
            )
        )
    else:
        checks.append(
            (
                "explicit no-fault label",
                not bool(graph.fault_active.any())
                and graph.fault["location"] is None,
            )
        )
    return checks


def _run_one(
    name: str,
    seed: int,
    net: DitecNetwork,
    archetypes: dict[str, str],
    cfg: Config,
    out_dir: Path,
    scenario_id: int,
    n_steps: int,
    with_leak: bool,
) -> PocScenario:
    dcn = build_dcn(net, cfg)  # fresh model per scenario
    rng = np.random.default_rng(seed)
    times = np.arange(n_steps) * cfg.solver.dt

    heat_gain = resolve_pipe_heat_gain(dcn, cfg, rng)  # per-scenario knobs
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    loads = generate_loads(archetypes, design, times, cfg.loads, rng)
    ua = {j: design_ua(design[j], cfg) for j in design}

    leak = None
    if with_leak:
        leak = inject_abrupt_leak(
            dcn, cfg, times, rng,
            junction=str(rng.choice(sorted(dcn.pairing))),
            side=str(rng.choice(("supply", "return"))),
        )
    result = run_eps_ntu_scenario(
        dcn, cfg, loads, n_steps=n_steps, ua=ua, leak=leak, pipe_U=heat_gain.U
    )
    label = (
        leak.label.with_realized(leak_flow=result.leak_flow)
        if leak is not None
        else FaultLabel.no_fault(times)
    )

    report = validate_scenario(dcn, cfg, result, label)
    record = ScenarioRecord(
        scenario_id=scenario_id,
        seed=seed,
        static_draw_id=cfg.poc.static_scenario_id,
        archetypes=archetypes,
        loads=loads,
        ua=ua,
        result=result,
        label=label,
        sensor_mask=draw_sensor_mask(dcn, cfg, rng),
        gate=report.as_dict(),
        heat_gain=heat_gain,
    )
    folder = write_scenario(out_dir / name, dcn, cfg, record)
    graph = load_scenario(folder)
    checks = _graph_checks(graph, dcn, label, n_steps)

    lines = [f"[{name}] seed {seed} -> {folder}"]
    if with_leak:
        lines.append(
            f"[{name}] fault: leak @ junction {label.location} "
            f"({label.side}), severity {label.severity:.3f}, "
            f"onset {label.onset:.0f} s"
        )
    lines += [
        f"[{name}]   gate {r.rule:22s} {'PASS' if r.passed else 'FAIL'}  "
        f"{r.detail}"
        for r in report.rules
    ]
    lines += [
        f"[{name}]   graph {desc:38s} {'PASS' if ok else 'FAIL'}"
        for desc, ok in checks
    ]
    return PocScenario(
        name=name,
        seed=seed,
        folder=folder,
        report=report,
        graph=graph,
        graph_ok=all(ok for _, ok in checks),
        lines=tuple(lines),
    )


def run_poc(cfg: Config, out_dir: Path, n_steps: int | None = None) -> PocRun:
    """Produce, gate, write, and reload the PoC pair.

    Args:
        cfg: configuration; ``cfg.poc`` names the network, static draw,
            and horizon, ``cfg.seed`` is the master seed.
        out_dir: records land in ``out_dir/normal`` and ``out_dir/leak``.
        n_steps: horizon override in steps (default
            ``cfg.poc.horizon / cfg.solver.dt``); the leak onset window
            scales with it.
    """
    out_dir = Path(out_dir)
    steps = int(round(cfg.poc.horizon / cfg.solver.dt)) if n_steps is None else n_steps

    # recorded integer sub-seeds: archetype assignment + one per scenario
    static_seed, normal_seed, leak_seed = (
        int(s) for s in np.random.default_rng(cfg.seed).integers(2**63, size=3)
    )
    net = load_ditec_network(
        Path(cfg.paths.ditec_data) / cfg.poc.network, cfg.poc.static_scenario_id
    )
    consumers = sorted(
        j for j in net.junctions if net.draw.base_demand.get(j, 0.0) > 0.0
    )
    archetype_rng = np.random.default_rng(static_seed)
    archetypes = {
        j: str(a)
        for j, a in zip(
            consumers, archetype_rng.choice(ARCHETYPE_NAMES, size=len(consumers))
        )
    }

    scenarios = (
        _run_one("normal", normal_seed, net, archetypes, cfg, out_dir,
                 scenario_id=0, n_steps=steps, with_leak=False),
        _run_one("leak", leak_seed, net, archetypes, cfg, out_dir,
                 scenario_id=1, n_steps=steps, with_leak=True),
    )
    passed = all(s.report.passed and s.graph_ok for s in scenarios)
    lines = (
        f"== dcngen PoC: {cfg.poc.network}, {steps} steps of "
        f"{cfg.solver.dt:.0f} s, master seed {cfg.seed} ==",
        *(line for s in scenarios for line in s.lines),
        f"== definition of done: {'PASS' if passed else 'FAIL'} ==",
    )
    return PocRun(scenarios=scenarios, passed=passed, lines=lines)
