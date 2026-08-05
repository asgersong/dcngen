"""Parallel plan driver: ``generate_dataset``
consumes a plan slice and produces per-scenario record folders plus an
aggregated run report. The stratified pilot batch is this same seam on a
:func:`pilot_rows` slice — its report is the green-light instrument for
the full run.

Execution: process-parallel over plan rows (each worker loads the row's
static draw itself and runs ``run_plan_row``), month-tier rows submitted
first (the longest scenarios bound the wall clock) while bulk/week rows
backfill idle workers.

Resume: a scenario folder is complete iff its ``card.json`` exists (the
writer lands it last, gate-stamped). On start the driver sweeps the
output — complete folders are skipped and NEVER touched (hence
byte-identical across resumes); stamp-less folders and ``.tmp`` staging
leftovers (an interrupt's debris) are discarded, and the ones in the
requested slice are re-run. Workers write into a ``.tmp`` staging folder
renamed into place on success, so an interrupt can never leave a
stamped-but-partial folder. The manifest is written beside the scenarios
and re-invocations must present the same plan — resuming under a
different release is refused.

Report: invocation accounting (completed/skipped/failed, core-seconds,
wall clock, submission order) plus dataset-level aggregation read back
from every completed card in the output folder — per-class gate pass
rates and per-rule metric distributions — so the report always describes
the folder's full contents. ``run_report.json`` additionally keeps the
per-invocation history and the cumulative core-seconds, so the compute
budget survives interrupt/resume cycles.
"""

import json
import multiprocessing
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dcngen.config import Config
from dcngen.orchestrate.plan_runner import run_plan_row
from dcngen.orchestrate.sampler import (
    PlanMeta,
    ScenarioPlan,
    ScenarioRow,
    load_plan,
    write_plan,
)
from dcngen.records.writer import plan_scenario_record_from_meta, write_scenario

_STAGING_SUFFIX = ".tmp"


_FOLDER_RE = re.compile(r"uid(\d+)$")


def scenario_folder_name(uid: int) -> str:
    """Zero-padded per-scenario folder name (lexically sortable)."""
    return f"uid{uid:05d}"


@dataclass(frozen=True)
class RunReport:
    """One ``generate_dataset`` invocation + the folder's aggregate state."""

    n_rows: int  # rows in this invocation's slice
    completed: int  # run to completion by THIS invocation
    skipped: int  # already complete on disk (resume)
    failed: dict[int, str]  # uid -> error of rows that raised
    scheduled_order: tuple[int, ...]  # submission order (month tier first)
    pass_rates: dict[str, dict[str, int]]  # class -> {passed, total}; ALL cards
    rule_metrics: dict[str, dict[str, dict[str, float]]]  # rule -> metric ->
    # {min, mean, max} over ALL completed cards in the folder
    core_seconds: float  # per-row durations incl. failed rows (this invocation)
    core_seconds_total: float  # cumulative over every invocation on out_dir
    wall_seconds: float  # driver elapsed (this invocation)
    workers: int
    out_dir: str

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": {str(k): v for k, v in self.failed.items()},
            "scheduled_order": list(self.scheduled_order),
            "pass_rates": self.pass_rates,
            "rule_metrics": self.rule_metrics,
            "core_seconds": self.core_seconds,
            "core_seconds_total": self.core_seconds_total,
            "wall_seconds": self.wall_seconds,
            "workers": self.workers,
            "out_dir": self.out_dir,
        }


@dataclass(frozen=True)
class _WorkerState:
    """Per-process state the pool initializer pins (tasks pickle only rows)."""

    cfg: Config
    meta: PlanMeta
    out_dir: Path


_WORKER: _WorkerState | None = None


def _init_worker(cfg: Config, meta: PlanMeta, out_dir: Path) -> None:
    global _WORKER
    _WORKER = _WorkerState(cfg=cfg, meta=meta, out_dir=out_dir)


def _run_row(row: ScenarioRow) -> tuple[int, float, str | None]:
    """Run one row and stage->publish its folder.

    Returns (uid, seconds, error) — a row failure is DATA, not an
    exception, so its compute time still lands in the accounting and a
    raising future can only mean pool infrastructure trouble. Runs inside
    a worker process (or inline when ``workers=0``); state comes from
    :func:`_init_worker`. The ``.tmp`` staging rename is what makes an
    interrupt unable to leave a stamped partial folder.
    """
    assert _WORKER is not None, "worker used before _init_worker"
    t0 = time.perf_counter()
    try:
        scenario = run_plan_row(_WORKER.cfg, row)
        record = plan_scenario_record_from_meta(scenario, _WORKER.meta)
        staging = _WORKER.out_dir / (scenario_folder_name(row.uid) + _STAGING_SUFFIX)
        if staging.exists():
            shutil.rmtree(staging)
        write_scenario(staging, scenario.dcn, scenario.config, record)
        staging.rename(_WORKER.out_dir / scenario_folder_name(row.uid))
    except Exception as exc:  # noqa: BLE001 — a row must not kill the run
        return row.uid, time.perf_counter() - t0, repr(exc)
    return row.uid, time.perf_counter() - t0, None


def _sweep(out_dir: Path, uids: set[int]) -> set[int]:
    """Resume pass: return the uids already complete; discard the rest.

    Complete = the folder carries ``card.json``. Staging leftovers and
    stamp-less folders (an interrupt's debris) are deleted so the run
    regenerates them from the plan.
    """
    done: set[int] = set()
    for path in out_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("uid"):
            continue
        if path.name.endswith(_STAGING_SUFFIX):
            shutil.rmtree(path)
            continue
        match = _FOLDER_RE.fullmatch(path.name)
        if match is None:  # foreign folder: not ours to touch
            continue
        uid = int(match.group(1))
        if (path / "card.json").exists():
            if uid in uids:
                done.add(uid)
        else:
            shutil.rmtree(path)
    return done


def _guard_manifest(out_dir: Path, plan: ScenarioPlan) -> None:
    """The output folder belongs to exactly one release: the first
    invocation writes the manifest (via the sampler's own writer, so the
    on-disk format has one owner); later ones must present the
    same plan."""
    path = out_dir / "plan.json"
    if path.exists():
        if load_plan(path) != plan:
            raise ValueError(
                f"{path} holds a different release manifest: refusing to mix "
                f"plans in one output folder"
            )
    else:
        write_plan(plan, path)


def _schedule(rows: list[ScenarioRow]) -> list[ScenarioRow]:
    """Longest tier (month) first — wall-clock-bounding; bulk/week backfill; uid order
    within each group keeps the schedule deterministic."""
    return sorted(rows, key=lambda r: (r.tier != "month", r.uid))


def _aggregate_cards(out_dir: Path) -> tuple[dict, dict]:
    """Dataset-level (pass_rates, rule_metrics) over every completed card."""
    passes: dict[str, dict[str, int]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    for path in sorted(out_dir.iterdir()):
        card_path = path / "card.json"
        if not path.is_dir() or not card_path.exists():
            continue
        card = json.loads(card_path.read_text())
        cls = card.get("plan", {}).get("row", {}).get("scenario_class")
        gate = card.get("validation")
        if cls is None or gate is None:
            continue
        cell = passes.setdefault(cls, {"passed": 0, "total": 0})
        cell["total"] += 1
        cell["passed"] += bool(gate["passed"])
        for rule in gate["rules"]:
            for metric, value in rule["metrics"].items():
                values.setdefault(rule["rule"], {}).setdefault(metric, []).append(
                    float(value)
                )
    rule_metrics = {
        rule: {
            metric: {
                "min": min(vals),
                "mean": sum(vals) / len(vals),
                "max": max(vals),
            }
            for metric, vals in metrics.items()
        }
        for rule, metrics in values.items()
    }
    return passes, rule_metrics


def pilot_rows(plan: ScenarioPlan, n: int) -> tuple[ScenarioRow, ...]:
    """The stratified pilot slice: ``n`` rows spread over the
    (tier, class) cells — one guaranteed per nonempty cell, the remainder
    proportional to cell size (largest remainder) — taking each cell's
    first rows in uid order, so the slice is deterministic.
    ``n`` beyond the plan size clamps to the whole plan.

    Raises:
        ValueError: when ``n`` cannot cover every nonempty cell.
    """
    cells: dict[tuple[str, str], list[ScenarioRow]] = {}
    for row in plan.rows:
        cells.setdefault((row.tier, row.scenario_class), []).append(row)
    n_total = len(plan.rows)
    n = min(n, n_total)
    if n < len(cells):
        raise ValueError(
            f"pilot of {n} cannot cover every nonempty (tier, class) cell "
            f"({len(cells)} cells)"
        )
    quotas = {cell: 1 for cell in cells}
    spare = n - len(cells)
    shares = {
        cell: spare * len(rows) / n_total for cell, rows in cells.items()
    }
    for cell in quotas:
        take = min(int(shares[cell]), len(cells[cell]) - 1)
        quotas[cell] += take
        spare -= take
    # largest fractional remainder, ties broken by cell name (deterministic)
    remainders = sorted(
        cells, key=lambda c: (-(shares[c] - int(shares[c])), c)
    )
    while spare > 0:
        for cell in remainders:
            if spare == 0:
                break
            if quotas[cell] < len(cells[cell]):
                quotas[cell] += 1
                spare -= 1
    picked = [row for cell, rows in cells.items() for row in rows[: quotas[cell]]]
    return tuple(sorted(picked, key=lambda r: r.uid))


def generate_dataset(
    cfg: Config,
    plan: ScenarioPlan,
    out_dir: str | Path,
    uids: list[int] | None = None,
    workers: int | None = None,
) -> RunReport:
    """Run a plan slice to completed record folders + the run report.

    Args:
        cfg: the release configuration the plan was built from.
        plan: the scenario-plan manifest.
        out_dir: dataset folder; scenario folders land as ``uidNNNNN/``,
            beside ``plan.json`` and ``run_report.json``.
        uids: the slice to run (``None`` = every row). The pilot batch is
            ``uids=[r.uid for r in pilot_rows(plan, n)]``.
        workers: process count override of ``cfg.driver.workers``; ``0``
            runs inline in this process (debugging, tests).
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _guard_manifest(out_dir, plan)

    wanted = set(uids) if uids is not None else {r.uid for r in plan.rows}
    unknown = wanted - {r.uid for r in plan.rows}
    if unknown:
        raise ValueError(f"uids not in the plan: {sorted(unknown)}")
    done = _sweep(out_dir, wanted)
    todo = _schedule([r for r in plan.rows if r.uid in wanted - done])

    n_workers = cfg.driver.workers if workers is None else workers
    failed: dict[int, str] = {}
    core_seconds = 0.0
    completed = 0

    def account(uid: int, seconds: float, error: str | None) -> None:
        nonlocal completed, core_seconds
        core_seconds += seconds  # failed rows burned real core time too
        if error is None:
            completed += 1
        else:
            failed[uid] = error

    if n_workers == 0:  # inline mode: same path, this process
        _init_worker(cfg, plan.meta, out_dir)
        for row in todo:
            account(*_run_row(row))
    else:
        # spawn, not fork: the invoking process may be multi-threaded
        # (torch/BLAS), and forked children can deadlock; worker startup
        # cost is negligible against ~20-min rows
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(cfg, plan.meta, out_dir),
        ) as pool:
            futures = {pool.submit(_run_row, row): row.uid for row in todo}
            for future, uid in futures.items():
                try:
                    account(*future.result())
                except Exception as exc:  # noqa: BLE001 — pool infrastructure
                    failed[uid] = repr(exc)  # row errors return as data

    # cumulative accounting across resume cycles: prior invocations are
    # read back from the previous report's history
    report_path = out_dir / "run_report.json"
    history: list[dict] = []
    if report_path.exists():
        history = json.loads(report_path.read_text()).get("history", [])
    history.append(
        {
            "completed": completed,
            "skipped": len(done),
            "failed": len(failed),
            "core_seconds": core_seconds,
            "wall_seconds": time.perf_counter() - t0,
            "workers": n_workers,
        }
    )

    pass_rates, rule_metrics = _aggregate_cards(out_dir)
    report = RunReport(
        n_rows=len(wanted),
        completed=completed,
        skipped=len(done),
        failed=failed,
        scheduled_order=tuple(r.uid for r in todo),
        pass_rates=pass_rates,
        rule_metrics=rule_metrics,
        core_seconds=core_seconds,
        core_seconds_total=sum(h["core_seconds"] for h in history),
        wall_seconds=history[-1]["wall_seconds"],
        workers=n_workers,
        out_dir=str(out_dir),
    )
    report_path.write_text(
        json.dumps({**report.as_dict(), "history": history}, indent=2, sort_keys=True)
    )
    return report
