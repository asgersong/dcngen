#!/usr/bin/env python
"""Run the PoC pair end-to-end and print the acceptance checks.

One clean-normal and one abrupt-leak scenario on the configured network
(Hanoi by default), both gated, written to records, and reloaded as typed
graphs — the first-working-pipeline definition of done. Exit code 0 iff
every check passes.
"""

import argparse
import sys
from pathlib import Path

from dcngen.config import load_config
from dcngen.orchestrate.poc import run_poc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="config YAML (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/poc"),
        help="records output folder (default: data/poc)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="horizon override in timesteps (default: cfg.poc.horizon / dt)",
    )
    args = parser.parse_args()

    run = run_poc(load_config(args.config), args.out, n_steps=args.steps)
    print("\n".join(run.lines))
    return 0 if run.passed else 1


if __name__ == "__main__":
    sys.exit(main())
