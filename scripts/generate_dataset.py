#!/usr/bin/env python
"""Generate the release dataset (or a slice of it) from the plan.

Builds the scenario-plan manifest from the config's master seed, then
drives the requested slice to record folders + the aggregated run
report. The pilot batch is `--pilot N`; a full run passes no slice
flags. Resumable: re-invoking on the same output folder skips completed
scenarios and regenerates interrupted ones. Exit code 0 iff nothing
failed and every completed scenario passed the gate.
"""

import argparse
import sys
from pathlib import Path

from dcngen.config import load_config
from dcngen.orchestrate.driver import generate_dataset, pilot_rows
from dcngen.orchestrate.sampler import build_plan
from dcngen.topology.ditec_loader import load_ditec_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="config YAML (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/dataset"),
        help="dataset output folder (default: data/dataset)",
    )
    parser.add_argument(
        "--pilot", type=int, default=None, metavar="N",
        help="run the stratified N-scenario pilot batch instead of the full plan",
    )
    parser.add_argument(
        "--uids", default=None,
        help="comma-separated scenario uids to run (overrides --pilot)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="process count (default: cfg.driver.workers; 0 = inline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    net = load_ditec_network(
        Path(cfg.paths.ditec_data) / cfg.sampler.network, scenario_id=0
    )
    plan = build_plan(cfg, net)

    if args.uids is not None:
        uids = [int(u) for u in args.uids.split(",")]
    elif args.pilot is not None:
        uids = [r.uid for r in pilot_rows(plan, args.pilot)]
    else:
        uids = None

    report = generate_dataset(cfg, plan, args.out, uids=uids, workers=args.workers)

    print(
        f"== dcngen generation: {report.n_rows} rows "
        f"({report.completed} run, {report.skipped} resumed-skip, "
        f"{len(report.failed)} failed) on {report.workers} workers =="
    )
    print(
        f"core {report.core_seconds / 3600.0:.1f} h, "
        f"wall {report.wall_seconds / 3600.0:.2f} h -> {report.out_dir}"
    )
    all_passed = True
    for cls in sorted(report.pass_rates):
        cell = report.pass_rates[cls]
        rate = cell["passed"] / cell["total"] if cell["total"] else 0.0
        all_passed &= cell["passed"] == cell["total"]
        print(f"  gate {cls:8s} {cell['passed']}/{cell['total']}  ({rate:.1%})")
    for uid, error in sorted(report.failed.items()):
        print(f"  FAILED uid {uid}: {error}")
    return 0 if not report.failed and all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
