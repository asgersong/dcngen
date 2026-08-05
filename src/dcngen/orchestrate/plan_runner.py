"""Plan-driven scenario runner: one manifest row in,
one gated scenario out — static draw, DCN build, row-driven loads, the
row's fault, eps-NTU solve, physics gate. No pinned values remain outside
the plan row and the config.

The temperature program travels as a *row-derived config*: the row's
T_supply/T_ground replace the config's thermal block (T_return co-moves as
T_supply + dT_design), so every downstream module — plant boundary, ETS
closure via the co-move rule, thermal solver ground
temperature, validation envelope — follows the sampled program without new
plumbing.

Runtime randomness (the AR(1) load-noise path) draws from the reserved
``Channel.LOAD_PATH`` stream keyed (master seed, uid), so a scenario
regenerates from (master seed, row) alone — the master seed is
``cfg.seed``, so the caller must pair a manifest with the config that
built it; everything else the run needs is materialised in the
row and injected explicitly.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from dcngen.config import Config
from dcngen.faults.injector import (
    BypassInjection,
    LeakInjection,
    inject_abrupt_leak,
    inject_bypass,
    inject_fouling,
)
from dcngen.faults.taxonomy import FaultLabel
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.sampler import Channel, ScenarioRow, channel_rng
from dcngen.orchestrate.scenario import EpsNtuScenarioResult, run_eps_ntu_scenario
from dcngen.orchestrate.validate import GateReport, validate_scenario
from dcngen.thermal.ets import installed_ua
from dcngen.thermal.heat_gain import HeatGainKnobs, PipeHeatGain, resolve_pipe_heat_gain
from dcngen.topology.dcnifier import DCNetwork, build_dcn
from dcngen.topology.ditec_loader import DitecNetwork, load_ditec_network


@dataclass(frozen=True)
class PlanScenario:
    """One plan row, fully solved and gated."""

    row: ScenarioRow
    config: Config  # the row-derived config (temperature program applied)
    dcn: DCNetwork
    loads: dict[str, np.ndarray]  # [W] per consumer, row-driven
    ua: dict[str, float]  # [W/K] installed UA per consumer (margin x fouling)
    heat_gain: PipeHeatGain  # per-pipe U' from the row's knobs
    result: EpsNtuScenarioResult
    label: FaultLabel
    report: GateReport


def row_config(cfg: Config, row: ScenarioRow) -> Config:
    """The config as this scenario sees it: the row's temperature program
    replaces the thermal block; T_return co-moves at the Keep dT_design.

    Guards the one invariant the gate cannot self-check (its envelope is
    built from these very values): the sampled program must keep the
    heat-gain sign. Sampler bands guarantee it; edited rows do not.
    """
    if not row.T_supply < row.T_ground:
        raise ValueError(
            f"row uid {row.uid}: T_supply ({row.T_supply}) must stay below "
            f"T_ground ({row.T_ground}) — a cooling network gains heat"
        )
    t = cfg.thermal
    return replace(
        cfg,
        thermal=replace(
            t,
            T_supply=row.T_supply,
            T_return_design=row.T_supply + t.dT_design,
            T_ground=row.T_ground,
        ),
    )


def plan_row_loads(
    cfg: Config, row: ScenarioRow, dcn: DCNetwork
) -> dict[str, np.ndarray]:
    """Row-driven cooling loads [W] per consumer: archetype map, perturbed
    knots, per-consumer sigma, start weekday, and the year-tier seasonal
    draw all come from the row; the AR(1) path from the LOAD_PATH channel.
    """
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    if sorted(design) != sorted(row.archetypes):
        raise ValueError(
            f"row uid {row.uid} archetype map does not key this draw's "
            f"consumers ({sorted(row.archetypes)} vs {sorted(design)})"
        )
    times = np.arange(row.n_steps) * cfg.solver.dt
    return generate_loads(
        row.archetypes,
        design,
        times,
        cfg.loads,
        rng=channel_rng(cfg.seed, row.uid, Channel.LOAD_PATH),
        start_day=row.start_weekday,
        noise_sigma=row.noise_sigma,
        knot_factors=row.knot_jitter,
        seasonal_amplitude=row.seasonal_amplitude,
        seasonal_phase=row.seasonal_phase,
    )


def _row_injections(
    row: ScenarioRow, dcn: DCNetwork, cfg: Config, times: np.ndarray
) -> tuple[LeakInjection | None, BypassInjection | None, FaultLabel]:
    """The row's fault as injections + ground-truth label (rng-free: every
    draw is already materialised in the row)."""
    spec = row.fault
    dt = cfg.solver.dt
    if spec.kind == "none":
        return None, None, FaultLabel.no_fault(times)
    if spec.kind == "leak":
        leak = inject_abrupt_leak(
            dcn, cfg, times, None,
            junction=spec.junction, side=spec.side,
            severity=spec.severity, onset=spec.onset_step * dt,
        )
        return leak, None, leak.label
    if spec.kind == "bypass":
        bypass = inject_bypass(
            dcn, cfg, times, None,
            consumer=spec.junction, fraction=spec.severity,
            whole_horizon=spec.onset_step is None,
            onset=None if spec.onset_step is None else spec.onset_step * dt,
        )
        return None, bypass, bypass.label
    if spec.kind == "fouling":
        fouling = inject_fouling(
            dcn, cfg, times, None, consumer=spec.junction, severity=spec.severity
        )
        return None, None, fouling.label
    raise ValueError(f"unknown fault kind {spec.kind!r} on row uid {row.uid}")


def run_plan_row(
    cfg: Config, row: ScenarioRow, net: DitecNetwork | None = None
) -> PlanScenario:
    """Run one manifest row end-to-end and gate it.

    Args:
        cfg: the release configuration (``cfg.seed`` is the master seed the
            row was materialised from).
        row: one scenario-plan row (``sampler.build_plan``).
        net: the row's static draw, if the caller already loaded it (the
            driver batches loads); ``None`` loads
            ``cfg.sampler.network`` at ``row.static_draw_id``.
    """
    rcfg = row_config(cfg, row)
    if net is None:
        net = load_ditec_network(
            Path(cfg.paths.ditec_data) / cfg.sampler.network, row.static_draw_id
        )
    elif net.draw.scenario_id != row.static_draw_id:
        raise ValueError(
            f"row uid {row.uid} wants static draw {row.static_draw_id}, "
            f"got a network at draw {net.draw.scenario_id}"
        )
    dcn = build_dcn(net, rcfg)  # fresh model per scenario
    times = np.arange(row.n_steps) * rcfg.solver.dt

    heat_gain = resolve_pipe_heat_gain(
        dcn, rcfg, rng=None,
        knobs=HeatGainKnobs(
            lambda_pur=row.lambda_pur,
            lambda_soil=row.lambda_soil,
            burial_cover=row.burial_cover,
        ),
    )
    loads = plan_row_loads(rcfg, row, dcn)
    design = {j: c.design_load for j, c in dcn.consumers.items()}
    fouling_m = row.fault.severity if row.fault.kind == "fouling" else None
    ua = {
        j: installed_ua(
            design[j], rcfg,
            healthy_factor=row.healthy_ua_factor[j],
            fouling=(
                fouling_m
                if fouling_m is not None and j == row.fault.junction
                else 1.0
            ),
        )
        for j in design
    }
    leak, bypass, label = _row_injections(row, dcn, rcfg, times)

    result = run_eps_ntu_scenario(
        dcn, rcfg, loads, n_steps=row.n_steps, ua=ua,
        leak=leak, bypass=bypass, pipe_U=heat_gain.U,
    )
    if leak is not None:
        label = label.with_realized(leak_flow=result.leak_flow)
    report = validate_scenario(dcn, rcfg, result, label)
    return PlanScenario(
        row=row,
        config=rcfg,
        dcn=dcn,
        loads=loads,
        ua=ua,
        heat_gain=heat_gain,
        result=result,
        label=label,
        report=report,
    )
