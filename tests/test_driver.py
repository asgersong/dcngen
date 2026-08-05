"""Driver seam: generate_dataset consumes a plan slice and
produces per-scenario record folders plus an aggregated run report —
process-parallel, year-tier-first, resumable (card.json is the completion
stamp; incomplete output is discarded and re-run; completed folders are
never touched, hence byte-identical across resumes). The pilot batch is
this same seam on a stratified plan slice (pilot_rows).
"""

import hashlib
import json

import pytest

from dcngen.config import load_config
from dcngen.orchestrate.driver import generate_dataset, pilot_rows
from dcngen.orchestrate.sampler import build_plan
from dcngen.records.pyg_dataset import load_scenario
from dcngen.topology.ditec_loader import load_ditec_network
from tests.conftest import (
    shrink_sampler_tiers,
    write_ditec_folder,
    write_mutated_default,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Config + plan wired so WORKER PROCESSES can load the mini network
    themselves (paths.ditec_data points at the tmp dump)."""
    nets_dir = tmp_path_factory.mktemp("nets")
    write_ditec_folder(nets_dir / "mini_net")

    def mutate(d):
        shrink_sampler_tiers(d)
        d["paths"]["ditec_data"] = str(nets_dir)
        d["driver"]["workers"] = 2

    cfg = load_config(write_mutated_default(tmp_path_factory.mktemp("cfg"), mutate))
    plan = build_plan(cfg, load_ditec_network(nets_dir / "mini_net", scenario_id=0))
    return cfg, plan


def slice_uids(plan):
    """Six rows spanning all tiers and classes (year included: scheduling)."""
    picks = [
        ("bulk", "normal"), ("bulk", "leak"), ("bulk", "fouling"),
        ("bulk", "bypass"), ("week", "normal"), ("month", "normal"),
    ]
    return [
        next(
            r.uid for r in plan.rows
            if r.tier == tier and r.scenario_class == cls
        )
        for tier, cls in picks
    ]


def folder_hashes(out_dir):
    """{scenario folder name: {file name: sha256}} over completed folders."""
    out = {}
    for folder in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        if (folder / "card.json").exists():
            out[folder.name] = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(folder.iterdir())
            }
    return out


@pytest.fixture(scope="module")
def first_run(env, tmp_path_factory):
    cfg, plan = env
    out = tmp_path_factory.mktemp("dataset")
    report = generate_dataset(cfg, plan, out, uids=slice_uids(plan))
    return cfg, plan, out, report


def test_slice_runs_to_completed_gated_folders(first_run):
    cfg, plan, out, report = first_run
    uids = slice_uids(plan)
    assert report.completed == len(uids)
    assert report.skipped == 0 and not report.failed
    for uid in uids:
        card = json.loads((out / f"uid{uid:05d}" / "card.json").read_text())
        assert card["scenario_id"] == uid
        assert "validation" in card  # gate-stamped
    assert (out / "plan.json").exists()
    assert (out / "run_report.json").exists()


def test_longest_tier_is_scheduled_first(first_run):
    _, plan, _, report = first_run
    tiers = {r.uid: r.tier for r in plan.rows}
    scheduled = [tiers[uid] for uid in report.scheduled_order]
    n_month = sum(t == "month" for t in scheduled)
    assert n_month > 0 and all(t == "month" for t in scheduled[:n_month])


def test_report_aggregates_gate_verdicts_per_class(first_run):
    _, _, out, report = first_run
    rates = report.pass_rates
    assert set(rates) == {"normal", "leak", "fouling", "bypass"}
    for cls, r in rates.items():
        assert r["total"] >= 1 and 0 <= r["passed"] <= r["total"]
    # gate-metric distributions: every rule carries min <= mean <= max
    assert report.rule_metrics
    for rule, metrics in report.rule_metrics.items():
        for stats in metrics.values():
            assert stats["min"] <= stats["mean"] <= stats["max"]
    assert report.core_seconds > 0 and report.wall_seconds > 0
    on_disk = json.loads((out / "run_report.json").read_text())
    assert on_disk["completed"] == report.completed


def test_written_scenarios_load_as_graphs(first_run):
    _, plan, out, report = first_run
    uid = slice_uids(plan)[3]  # the bypass row
    graph = load_scenario(out / f"uid{uid:05d}")
    assert graph.scenario_id == uid
    assert graph.scenario_class == "bypass"


def test_resume_skips_completed_and_rebuilds_incomplete(first_run):
    cfg, plan, out, _ = first_run
    uids = slice_uids(plan)
    before = folder_hashes(out)

    # simulate an interrupt: one scenario lost its completion stamp
    victim = out / f"uid{uids[1]:05d}"
    (victim / "card.json").unlink()

    report = generate_dataset(cfg, plan, out, uids=uids)
    assert report.skipped == len(uids) - 1
    assert report.completed == 1
    assert (victim / "card.json").exists()  # rebuilt

    after = folder_hashes(out)
    assert set(after) == set(before)
    for name in before:
        if name != victim.name:  # completed folders are byte-identical
            assert after[name] == before[name]


def test_foreign_manifest_in_out_dir_is_rejected(env, tmp_path_factory):
    cfg, plan = env

    def reseed(d):
        shrink_sampler_tiers(d)
        d["seed"] = 43

    other_cfg = load_config(
        write_mutated_default(tmp_path_factory.mktemp("cfg2"), reseed)
    )
    other_plan = build_plan(
        other_cfg,
        load_ditec_network(cfg.paths.ditec_data / "mini_net", scenario_id=0),
    )
    out = tmp_path_factory.mktemp("dataset2")
    generate_dataset(cfg, plan, out, uids=slice_uids(plan)[:1])
    with pytest.raises(ValueError, match="manifest"):
        generate_dataset(other_cfg, other_plan, out, uids=[0])


def test_pilot_rows_is_stratified_and_deterministic(env):
    _, plan = env
    rows = pilot_rows(plan, 14)
    assert len(rows) == 14
    assert rows == pilot_rows(plan, 14)  # deterministic
    cells = {(r.tier, r.scenario_class) for r in plan.rows}
    covered = {(r.tier, r.scenario_class) for r in rows}
    assert covered == cells  # every nonempty (tier, class) cell represented
    uids = [r.uid for r in rows]
    assert len(set(uids)) == len(uids)  # no duplicates
    by_uid = {r.uid: r for r in plan.rows}
    assert all(by_uid[r.uid] == r for r in rows)  # genuine plan rows


def test_pilot_smaller_than_the_cell_count_is_rejected(env):
    _, plan = env
    with pytest.raises(ValueError, match="cell"):
        pilot_rows(plan, 3)


def test_serial_mode_records_failures_without_leaving_folders(
    env, tmp_path_factory, monkeypatch
):
    cfg, plan = env
    uids = slice_uids(plan)[:2]
    doomed = uids[1]

    import dcngen.orchestrate.driver as driver_mod

    real = driver_mod.run_plan_row

    def sabotage(cfg_, row, net=None):
        if row.uid == doomed:
            raise RuntimeError("synthetic solver crash")
        return real(cfg_, row, net)

    monkeypatch.setattr(driver_mod, "run_plan_row", sabotage)
    out = tmp_path_factory.mktemp("dataset3")
    report = generate_dataset(cfg, plan, out, uids=uids, workers=0)

    assert report.completed == 1
    assert list(report.failed) == [doomed]
    assert "synthetic solver crash" in report.failed[doomed]
    assert not (out / f"uid{doomed:05d}").exists()  # no partial output
    assert (out / f"uid{uids[0]:05d}" / "card.json").exists()


def test_sweep_clears_interrupt_debris_and_ignores_foreign_folders(
    first_run,
):
    cfg, plan, out, _ = first_run
    uids = slice_uids(plan)
    # an interrupt's debris: a staging folder mid-write...
    staging = out / "uid99999.tmp"
    staging.mkdir()
    (staging / "node_dynamics.parquet").write_bytes(b"partial")
    # ...and folders that merely LOOK like ours must not be touched
    foreign = out / "uids_archive"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("precious")

    report = generate_dataset(cfg, plan, out, uids=uids)
    assert report.skipped == len(uids)  # everything already complete
    assert not staging.exists()  # debris swept
    assert (foreign / "keep.txt").read_text() == "precious"


def test_core_seconds_accumulate_across_resume_invocations(first_run):
    cfg, plan, out, _ = first_run
    report = generate_dataset(cfg, plan, out, uids=slice_uids(plan))
    # a pure-resume invocation runs nothing new but the dataset's
    # cumulative core account keeps every prior invocation's time
    assert report.completed == 0
    assert report.core_seconds == 0.0
    assert report.core_seconds_total > 0.0
    on_disk = json.loads((out / "run_report.json").read_text())
    assert len(on_disk["history"]) >= 2
    assert on_disk["core_seconds_total"] == report.core_seconds_total
