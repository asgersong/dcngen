"""A published scenario regenerates from (seed, uid).

The reproducibility claim this dataset makes is precise, and it is two
claims, not one:

* everything drawn from the RNG -- the plan row, archetypes, load series,
  heat-gain knobs, fault label and its mask -- regenerates **bitwise**;
* the **solved** fields do not, because WNTR's solver iterates in
  memory-address order, so pressures, flows and temperatures differ in the
  last bits between processes. They are compared physically instead.

Splitting the assertions that way is the point: a single "close enough"
comparison over the whole record would let a genuine sampling regression
hide inside the solver's noise floor.

Slow-marked (one bulk row is ~20 min) and skipped without the released
dataset, so this runs as release assurance rather than in the default
suite: ``pytest -m slow tests/test_v1_reproducibility.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcngen.config import load_config
from dcngen.orchestrate.plan_runner import run_plan_row
from dcngen.orchestrate.sampler import load_plan
from dcngen.records.writer import (
    plan_scenario_record_from_meta,
    write_scenario,
)

V1 = Path(__file__).resolve().parents[1] / "data" / "v1"
# uid 0 is a bulk/normal row on static draw 203 -- deliberately the draw that
# exposed the unbounded-steady-init bug, so the spot-check lands
# on a numerically awkward scenario rather than a comfortable one.
SPOT_CHECK_UID = 0

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (V1 / "plan.json").is_file(),
        reason=f"released dataset not present at {V1}",
    ),
]

# Solver noise floor. NOT a physics tolerance: it is how far two
# processes' WNTR solves drift apart on identical inputs, because the
# solver iterates in memory-address order and float addition is not
# associative. Each bound is the drift measured on this row (2026-07-27)
# with ~3 orders of headroom, which leaves every one of them 6+ orders
# below anything that could change a label, a gate verdict or a learned
# feature.
#
#   measured on uid 0:  temperatures 2.1e-12 K · pressures 1.8e-12 m
#                       flows 2.2e-13 m3/s · powers 2.3e-06 W
SOLVED_ATOL = {
    # node_dynamics
    "supply_temp": 1e-9,  # K
    "return_temp": 1e-9,
    "ets_return": 1e-9,
    "dT": 1e-9,
    "pressure_s": 1e-9,  # m
    "pressure_r": 1e-9,
    "consumer_flow": 1e-11,  # m3/s
    "delivered_load": 1e-3,  # W, against per-consumer loads of order 1e7
    # edge_dynamics
    "flow_s": 1e-11,
    "flow_r": 1e-11,
    "velocity_s": 1e-10,  # m/s
    "velocity_r": 1e-10,
    "headloss_s": 1e-11,  # m
    "headloss_r": 1e-11,
    "heat_gain_s": 1e-3,  # W, against pipe gains of order 1e4-1e5
    "heat_gain_r": 1e-3,
    # plant_dynamics
    "plant_power": 1e-2,  # W, against a plant load of order 1e8
    "plant_flow": 1e-11,
    "make_up_flow": 1e-11,  # identically zero on a clean loop
}

# Channels that are pure RNG output or pure geometry: these must be exact,
# and the whole point of splitting the assertions is that a drift here is a
# regression, never solver noise.
EXACT_TABLES = ("labels", "node_static", "edge_static")
EXACT_COLUMNS = {"cooling_load", "junction", "pipe", "leak_flow", "unmet",
                 "converged", "iterations", "scenario_id", "time_id"}


def _folder(root: Path, uid: int) -> Path:
    return root / f"uid{uid:05d}"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Re-run the spot-check row from the manifest; return both records.

    The plan is *loaded from the release* rather than rebuilt, because the
    claim under test is that the published artifact regenerates -- not that
    ``build_plan`` is deterministic (which its own tests cover).
    """
    cfg = load_config()
    plan = load_plan(V1 / "plan.json")
    row = next(r for r in plan.rows if r.uid == SPOT_CHECK_UID)

    published = _folder(V1, SPOT_CHECK_UID)
    card_published = json.loads((published / "card.json").read_text())

    scenario = run_plan_row(cfg, row)
    record = plan_scenario_record_from_meta(scenario, plan.meta)
    out = write_scenario(
        _folder(tmp_path_factory.mktemp("regen"), SPOT_CHECK_UID),
        scenario.dcn,
        scenario.config,
        record,
    )
    return published, out, card_published, json.loads((out / "card.json").read_text())


# Config blocks that cannot change a generated scenario, and why. The
# reproducibility guard below ignores these so that adding, say, a verification
# tolerance does not masquerade as a sampling regression -- but it ignores
# ONLY these, by name, so a genuine physics change still fails loudly.
GENERATION_IRRELEVANT = {
    "verification": "read only by verify/pandapipes_ref.py, never by the generator",
    "driver": "worker count — operational, cannot change a row's contents",
    "poc": "the M6b PoC pair only; plan rows do not consult it",
}


def test_the_current_config_still_matches_the_published_echo(regenerated):
    """Guard: the claim is 'regenerates under the recorded config'.

    Each card echoes the config it was generated under. If a
    generation-relevant constant has drifted, the field comparisons below
    would be ambiguous -- a sampling regression, or just a changed
    number -- so establish which it is first, and name the blocks that
    moved.
    """
    _, _, published, regen = regenerated
    blocks = (set(published["config"]) | set(regen["config"])) - set(
        GENERATION_IRRELEVANT
    )
    moved = sorted(
        block
        for block in blocks
        if published["config"].get(block) != regen["config"].get(block)
    )
    assert not moved, (
        f"generation-relevant config blocks changed since the release: {moved} "
        "-- this spot-check would be comparing against a record made under a "
        "different configuration, so its result would mean nothing"
    )


def test_seeded_draws_regenerate_bitwise(regenerated):
    """Everything the RNG produced must come back identical."""
    _, _, published, regen = regenerated

    assert regen["plan"]["row"] == published["plan"]["row"]
    assert regen["static_draw_id"] == published["static_draw_id"]
    assert regen["seed"] == published["seed"]
    assert regen["heat_gain"] == published["heat_gain"]
    assert regen["fault"] == published["fault"]
    assert regen["plan"]["master_seed"] == published["plan"]["master_seed"]
    assert regen["plan"]["bit_generator"] == published["plan"]["bit_generator"]
    # topology and the mirror's bookkeeping are functions of the draw
    assert regen["pairing"] == published["pairing"]
    assert regen["pipe_pairing"] == published["pipe_pairing"]
    assert regen["adj_list"] == published["adj_list"]


@pytest.mark.parametrize("table", EXACT_TABLES)
def test_rng_and_geometry_tables_regenerate_bitwise(regenerated, table):
    """Whole tables that contain no solved quantity must be identical.

    ``labels`` is the per-step fault mask; ``node_static`` carries the
    archetype, design load and the drawn healthy-UA margin; ``edge_static``
    carries the derived per-pipe U'. All three are pure functions of the
    draw and the seed, so any difference at all is a sampling regression.
    """
    published, regen, _, _ = regenerated
    pd.testing.assert_frame_equal(
        pd.read_parquet(published / f"{table}.parquet"),
        pd.read_parquet(regen / f"{table}.parquet"),
    )


def test_load_series_regenerates_bitwise(regenerated):
    """The cooling loads are the channel a benchmark is *about*.

    If these drifted, two runs of the same uid would pose different
    learning problems, whatever the solver did afterwards.
    """
    published, regen, _, _ = regenerated
    a = pd.read_parquet(published / "node_dynamics.parquet")
    b = pd.read_parquet(regen / "node_dynamics.parquet")
    np.testing.assert_array_equal(
        a["cooling_load"].to_numpy(), b["cooling_load"].to_numpy()
    )
    np.testing.assert_array_equal(a["junction"].to_numpy(), b["junction"].to_numpy())


@pytest.mark.parametrize(
    "table", ["node_dynamics", "edge_dynamics", "plant_dynamics"]
)
def test_solved_fields_agree_within_the_solver_noise_floor(regenerated, table):
    """Solved fields match physically, not bitwise.

    Every numeric column is checked, not a chosen few: an unlisted column
    fails the test rather than escaping it, so adding a channel to the
    writer forces a decision about which side of the exact/solved line it sits on.
    """
    published, regen, _, _ = regenerated
    a = pd.read_parquet(published / f"{table}.parquet")
    b = pd.read_parquet(regen / f"{table}.parquet")
    assert list(a.columns) == list(b.columns)

    for column in a.columns:
        x, y = a[column].to_numpy(), b[column].to_numpy()
        if column in EXACT_COLUMNS or not np.issubdtype(x.dtype, np.floating):
            np.testing.assert_array_equal(x, y, err_msg=f"{table}.{column}")
            continue
        assert column in SOLVED_ATOL, (
            f"{table}.{column} is a solved float channel with no recorded "
            "noise-floor bound -- decide whether it is exact or add its "
            "measured drift to SOLVED_ATOL"
        )
        worst = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
        assert worst <= SOLVED_ATOL[column], (
            f"{table}.{column}: regenerated run drifts by {worst:.3g} "
            f"(allowed {SOLVED_ATOL[column]:.3g}) -- beyond WNTR's "
            "cross-process noise floor, so this is a regression, not solver noise"
        )


def test_gate_verdict_and_energy_closure_reproduce(regenerated):
    """The verdict a user filters on must not depend on which run made it."""
    _, _, published, regen = regenerated
    assert regen["validation"]["passed"] == published["validation"]["passed"]

    rules_pub = {r["rule"]: r["passed"] for r in published["validation"]["rules"]}
    rules_new = {r["rule"]: r["passed"] for r in regen["validation"]["rules"]}
    assert rules_new == rules_pub

    energy = next(
        r for r in regen["validation"]["rules"] if r["rule"] == "energy_closure"
    )
    assert energy["metrics"]["residual_rel"] < 1e-12
