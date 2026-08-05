"""The full-horizon Hanoi PoC, as a mechanically runnable test.

Runs ~20 minutes, so it is opt-in: set ``DCN_POC_HANOI=1`` (and have the
DiTEC dump at ``cfg.paths.ditec_data``). The default suite covers the
identical code path on the mini net in ``test_run_poc.py``.
"""

import os
from pathlib import Path

import pytest

from dcngen.config import load_config

cfg = load_config()
_HANOI = Path(cfg.paths.ditec_data) / cfg.poc.network
_OPTED_IN = os.environ.get("DCN_POC_HANOI") == "1"


@pytest.mark.skipif(
    not (_OPTED_IN and _HANOI.exists()),
    reason="full-horizon Hanoi PoC (~20 min): set DCN_POC_HANOI=1 with the dump present",
)
def test_the_real_hanoi_poc_passes_the_definition_of_done(tmp_path):
    from dcngen.orchestrate.poc import run_poc

    run = run_poc(cfg, tmp_path / "poc")
    assert run.passed, "\n".join(run.lines)
