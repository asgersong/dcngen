"""Typed, validated access to the project configuration.

Every physical constant, design temperature, solver tolerance, and sampling
convention lives in ``configs/default.yaml`` (SI units unless noted in the
field docs); no *tunable* physical value is hard-coded elsewhere in the
package (fixed, cited standards content — the EN 253 series-1 casing rows in
``thermal/heat_gain.py`` — is the one deliberate exception). Loading
is strict: unknown keys, missing keys, and physically inconsistent values all
raise :class:`ConfigError`.

NOTE: ``_build`` dispatches on dataclass field annotations being real classes
(``is_dataclass(f.type)``, ``f.type is Path`` ...). Do NOT add
``from __future__ import annotations`` to this module — it would turn every
``f.type`` into a string and silently break config loading.
"""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """The configuration is malformed or physically inconsistent."""


@dataclass(frozen=True)
class WaterProperties:
    cp: float  # specific heat [J/(kg K)]
    rho: float  # density [kg/m3]
    mu: float  # dynamic viscosity [Pa s]


@dataclass(frozen=True)
class ThermalDesign:
    T_supply: float  # plant supply setpoint [degC]
    T_return_design: float  # design network return temperature [degC]
    dT_design: float  # design network split [K]
    T_ground: float  # ground/ambient temperature [degC]
    init_residence_cap: float  # steady-init pre-history bound [s]:
    # caps plug residence so near-stagnant pipes hold yesterday's water


@dataclass(frozen=True)
class HeatGainModel:
    # Per-pipe linear U' [W/(m K)]; knob bounds are the cited sampling
    # ranges, drawn uniformly. The derivation itself (EN 253 series-1
    # casing table, two-resistance formula) lives in thermal/heat_gain.py.
    mode: str  # "derived" (released dataset) | "scalar" (dev/ablation fallback)
    lambda_pur_min: float  # PUR foam conductivity [W/(m K)]
    lambda_pur_max: float
    lambda_soil_min: float  # soil conductivity [W/(m K)]
    lambda_soil_max: float
    burial_cover_min: float  # ground cover above the casing [m]; H = cover + D_c/2
    burial_cover_max: float
    scalar_U: float  # fallback flat U' [W/(m K)]
    scalar_U_min: float  # fallback-mode ablation band
    scalar_U_max: float


@dataclass(frozen=True)
class ETSDesign:
    approach: float  # heat-exchanger approach [K]: the secondary program is the
    # primary shifted by this at both ends, so dT_building == dT_design
    # (derivation helpers live in thermal/ets.py)
    max_flow_factor: float  # valve limit as a multiple of design primary flow
    min_dp: float  # minimum differential pressure across the ETS at design [m]
    healthy_ua_factor_min: float  # healthy sizing-margin band on design UA:
    healthy_ua_factor_max: float  # installed UA = design UA x U(min, max)


@dataclass(frozen=True)
class SolverSettings:
    dt: float  # timestep [s]
    flow_tol_rel: float  # relative convergence tolerance on consumer flows
    relaxation: float  # under-relaxation factor for the fixed-point iteration
    max_iterations: int  # per-timestep hydraulic<->thermal iteration cap
    required_pressure: float  # PDD pressure for full demand delivery [m]
    minimum_pressure: float  # PDD pressure below which delivery is zero [m]


@dataclass(frozen=True)
class ValidationGate:
    mass_balance_rel_tol: float
    energy_rel_tol: float
    max_velocity: float  # reference velocity for exceedance metadata [m/s]
    max_unconverged_frac: float  # scenario budget of unconverged timesteps
    temperature_tol: float  # numerical slack on the temperature envelope [K]
    pressure_floor_overpressure: float  # gauge margin above the loop high point [m]
    plant_suction_min: float  # minimum pump-suction gauge pressure [m]
    pressure_ceiling: float  # continuous-operation pressure ceiling [m] (PN16)
    pressure_ceiling_pn25: float  # escalated ceiling [m] (PN25 class,
    # recorded per scenario via the pn25_escalated gate metric)
    max_clean_unmet_frac: float  # unmet budget on clean scenarios


@dataclass(frozen=True)
class VerificationHarness:
    """pandapipes cross-check.

    Thermal-strict, hydraulic-documented: the temperature and energy
    tolerances are strict, the pressure band is a documented one because
    the two engines do not share a friction law.
    """

    temperature_tol: float  # node temperature agreement [K]
    energy_rel_tol: float  # plant energy-balance residual [-]
    flow_rel_tol: float  # pipe flow deviation, per plant circulation [-]
    pressure_tol: float  # node pressure agreement [m]
    pipe_sections: int  # pandapipes internal sub-elements per pipe
    steady_turnovers: float  # slowest-pipe transits marched before comparing
    steady_max_steps: int  # cap on the derived step count
    steady_drift_tol: float  # last-step temperature drift allowed [K]


@dataclass(frozen=True)
class LoadModel:
    ar1_correlation_time: float  # [s]
    noise_sigma: float  # relative std of multiplicative AR(1) noise
    seasonal_amplitude: float  # 0 = flat (knob only)


@dataclass(frozen=True)
class FaultConventions:
    leak_severity_min: float  # leak flow as fraction of network design flow
    leak_severity_max: float
    onset_window_start: float  # onset sampling window [fraction of horizon]
    onset_window_end: float
    leak_discharge_coeff: float  # orifice C_d for the leak model
    # ETS fault severity bands
    fouling_severity_min: float  # UA multiplier m, whole-scenario
    fouling_severity_max: float
    bypass_fraction_min: float  # primary short-circuit fraction f
    bypass_fraction_max: float
    bypass_whole_horizon_prob: float  # P(whole-horizon) vs abrupt onset


@dataclass(frozen=True)
class TierPlan:
    """One horizon tier of the release: scenario length + class counts."""

    horizon: float  # scenario length [s]; must sit on the solver.dt grid
    normal: int  # clean-normal scenario count
    leak: int
    fouling: int
    bypass: int


@dataclass(frozen=True)
class SamplerSettings:
    # Scenario-plan generation. Bands are Sample/Perturb ranges drawn
    # uniformly; tier counts are the dataset shape.
    network: str  # DiTEC network folder of the release
    static_draw_count: int  # static_scenario_id ~ U{0..count-1}
    T_supply_min: float  # plant setpoint band [degC]; secondary co-moves
    T_supply_max: float
    T_ground_min: float  # ground temperature band [degC]
    T_ground_max: float
    residential_minority_min: float  # Louvain minority fraction band
    residential_minority_max: float
    noise_sigma_min: float  # per-consumer AR(1) sigma band
    noise_sigma_max: float
    knot_jitter: float  # knots x U(1-j, 1+j) per (scenario, archetype)
    seasonal_amplitude_min: float  # long-horizon (month) tier band
    seasonal_amplitude_max: float
    bulk: TierPlan
    week: TierPlan
    month: TierPlan


@dataclass(frozen=True)
class DriverSettings:
    workers: int  # process-parallel worker count; overridable per run


@dataclass(frozen=True)
class SensorPlacement:
    consumer_coverage: float  # fraction of consumers heat-metered
    pressure_coverage: float  # fraction of junctions with pressure sensors


@dataclass(frozen=True)
class DataPaths:
    ditec_data: Path  # DiTEC-WDN parquet dump (folder per network)


@dataclass(frozen=True)
class PocSettings:
    network: str  # DiTEC network folder name
    static_scenario_id: int  # pinned static parameter draw
    horizon: float  # scenario length [s]


@dataclass(frozen=True)
class Config:
    water: WaterProperties
    thermal: ThermalDesign
    heat_gain: HeatGainModel
    ets: ETSDesign
    solver: SolverSettings
    validation: ValidationGate
    verification: VerificationHarness
    loads: LoadModel
    faults: FaultConventions
    sampler: SamplerSettings
    driver: DriverSettings
    sensors: SensorPlacement
    paths: DataPaths
    poc: PocSettings
    seed: int


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

# Numerical slack (not physics) for the dT_design == T_return_design - T_supply
# consistency check: tolerates float representation error only.
_DT_CONSISTENCY_TOL = 1e-9


def _coerce(typ: type, value: object, where: str) -> object:
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"expected a number at {where}, got {value!r}")
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"expected an integer at {where}, got {value!r}")
        return value
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"expected a string at {where}, got {value!r}")
        return value
    if typ is Path:
        if not isinstance(value, str):
            raise ConfigError(f"expected a path string at {where}, got {value!r}")
        return Path(value)
    raise ConfigError(f"unsupported config field type {typ!r} at {where}")


def _build(cls: type, data: object, where: str) -> object:
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at {where}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} at {where}")
    missing = sorted(set(known) - set(data))
    if missing:
        raise ConfigError(f"missing key(s) {missing} at {where}")
    kwargs = {}
    for name, f in known.items():
        sub = f"{where}.{name}"
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, data[name], sub)
        else:
            kwargs[name] = _coerce(f.type, data[name], sub)
    return cls(**kwargs)


def _validate(cfg: Config) -> list[str]:
    bad: list[str] = []

    positive = {
        "water.cp": cfg.water.cp,
        "water.rho": cfg.water.rho,
        "water.mu": cfg.water.mu,
        "thermal.dT_design": cfg.thermal.dT_design,
        "thermal.init_residence_cap": cfg.thermal.init_residence_cap,
        "heat_gain.lambda_pur_min": cfg.heat_gain.lambda_pur_min,
        "heat_gain.lambda_soil_min": cfg.heat_gain.lambda_soil_min,
        "heat_gain.burial_cover_min": cfg.heat_gain.burial_cover_min,
        "heat_gain.scalar_U_min": cfg.heat_gain.scalar_U_min,
        "ets.approach": cfg.ets.approach,
        "ets.min_dp": cfg.ets.min_dp,
        "solver.dt": cfg.solver.dt,
        "solver.flow_tol_rel": cfg.solver.flow_tol_rel,
        "validation.mass_balance_rel_tol": cfg.validation.mass_balance_rel_tol,
        "validation.energy_rel_tol": cfg.validation.energy_rel_tol,
        "validation.max_velocity": cfg.validation.max_velocity,
        "validation.temperature_tol": cfg.validation.temperature_tol,
        "validation.pressure_floor_overpressure": cfg.validation.pressure_floor_overpressure,
        "validation.plant_suction_min": cfg.validation.plant_suction_min,
        "validation.pressure_ceiling": cfg.validation.pressure_ceiling,
        "validation.pressure_ceiling_pn25": cfg.validation.pressure_ceiling_pn25,
        "loads.ar1_correlation_time": cfg.loads.ar1_correlation_time,
        "poc.horizon": cfg.poc.horizon,
    }
    bad += [f"{k} must be positive (got {v})" for k, v in positive.items() if v <= 0]

    t = cfg.thermal
    if t.T_supply >= t.T_return_design:
        bad.append(
            f"thermal.T_supply ({t.T_supply}) must be below "
            f"thermal.T_return_design ({t.T_return_design})"
        )
    if t.T_supply >= t.T_ground:
        bad.append(
            f"thermal.T_supply ({t.T_supply}) must be below thermal.T_ground "
            f"({t.T_ground}): a cooling network gains heat from the ground"
        )
    if abs(t.T_return_design - t.T_supply - t.dT_design) > _DT_CONSISTENCY_TOL:
        bad.append(
            f"thermal.dT_design ({t.dT_design}) must equal "
            f"T_return_design - T_supply ({t.T_return_design - t.T_supply})"
        )

    hg = cfg.heat_gain
    if hg.mode not in ("derived", "scalar"):
        bad.append(
            f"heat_gain.mode ({hg.mode!r}) must be 'derived' or 'scalar'"
        )
    for knob, lo, hi in (
        ("lambda_pur", hg.lambda_pur_min, hg.lambda_pur_max),
        ("lambda_soil", hg.lambda_soil_min, hg.lambda_soil_max),
        ("burial_cover", hg.burial_cover_min, hg.burial_cover_max),
    ):
        if lo > hi:
            bad.append(f"heat_gain.{knob} range needs min ({lo}) <= max ({hi})")
    if not hg.scalar_U_min <= hg.scalar_U <= hg.scalar_U_max:
        bad.append(
            f"heat_gain.scalar_U ({hg.scalar_U}) must sit inside its ablation "
            f"band [{hg.scalar_U_min}, {hg.scalar_U_max}]"
        )
    # Structurally satisfied by the co-move rule for any positive approach,
    # but retained as an invariant check: it still fires when the approach
    # vanishes in float addition. Mirrors thermal/ets.py
    # building_hot_inlet_temp — kept inline because config cannot import ets.
    hot_inlet = t.T_return_design + cfg.ets.approach
    if hot_inlet <= t.T_return_design:
        bad.append(
            f"ETS hot inlet ({hot_inlet} = T_return_design + approach) must "
            f"exceed thermal.T_return_design ({t.T_return_design}): no finite UA "
            f"can otherwise reach the design return"
        )
    if cfg.ets.max_flow_factor <= 1.0:
        bad.append(f"ets.max_flow_factor ({cfg.ets.max_flow_factor}) must exceed 1")
    if not 1.0 <= cfg.ets.healthy_ua_factor_min <= cfg.ets.healthy_ua_factor_max:
        bad.append(
            f"healthy_ua_factor band needs 1 <= min ({cfg.ets.healthy_ua_factor_min}) "
            f"<= max ({cfg.ets.healthy_ua_factor_max}): installed UA sits at or "
            f"above the design requirement"
        )

    s = cfg.solver
    if not 0.0 < s.relaxation <= 1.0:
        bad.append(f"solver.relaxation ({s.relaxation}) must be in (0, 1]")
    if s.max_iterations < 1:
        bad.append(f"solver.max_iterations ({s.max_iterations}) must be at least 1")
    if s.minimum_pressure < 0 or s.required_pressure <= s.minimum_pressure:
        bad.append(
            f"PDD pressures need 0 <= minimum_pressure ({s.minimum_pressure}) "
            f"< required_pressure ({s.required_pressure})"
        )

    if cfg.validation.pressure_ceiling_pn25 <= cfg.validation.pressure_ceiling:
        bad.append(
            f"validation.pressure_ceiling_pn25 ({cfg.validation.pressure_ceiling_pn25}) "
            f"must exceed the PN16 ceiling ({cfg.validation.pressure_ceiling})"
        )
    if not 0.0 <= cfg.validation.max_clean_unmet_frac <= 1.0:
        bad.append(
            f"validation.max_clean_unmet_frac ({cfg.validation.max_clean_unmet_frac}) "
            f"must be in [0, 1]"
        )
    if not 0.0 <= cfg.validation.max_unconverged_frac <= 1.0:
        bad.append(
            f"validation.max_unconverged_frac ({cfg.validation.max_unconverged_frac}) "
            f"must be in [0, 1]"
        )
    if cfg.loads.noise_sigma < 0:
        bad.append(f"loads.noise_sigma ({cfg.loads.noise_sigma}) must be non-negative")
    if cfg.loads.seasonal_amplitude < 0:
        bad.append(
            f"loads.seasonal_amplitude ({cfg.loads.seasonal_amplitude}) must be non-negative"
        )

    f = cfg.faults
    if not 0.0 < f.leak_severity_min <= f.leak_severity_max < 1.0:
        bad.append(
            f"fault severities need 0 < leak_severity_min ({f.leak_severity_min}) "
            f"<= leak_severity_max ({f.leak_severity_max}) < 1"
        )
    if not 0.0 <= f.onset_window_start < f.onset_window_end <= 1.0:
        bad.append(
            f"onset window needs 0 <= start ({f.onset_window_start}) "
            f"< end ({f.onset_window_end}) <= 1"
        )
    if not 0.0 < f.fouling_severity_min <= f.fouling_severity_max < 1.0:
        bad.append(
            f"fouling band needs 0 < fouling_severity_min ({f.fouling_severity_min}) "
            f"<= fouling_severity_max ({f.fouling_severity_max}) < 1: m = 1 is the "
            f"healthy exchanger"
        )
    if not 0.0 < f.bypass_fraction_min <= f.bypass_fraction_max < 1.0:
        bad.append(
            f"bypass band needs 0 < bypass_fraction_min ({f.bypass_fraction_min}) "
            f"<= bypass_fraction_max ({f.bypass_fraction_max}) < 1: f = 0 is the "
            f"healthy crossover, f = 1 starves the exchanger"
        )
    if not 0.0 <= f.bypass_whole_horizon_prob <= 1.0:
        bad.append(
            f"faults.bypass_whole_horizon_prob ({f.bypass_whole_horizon_prob}) "
            f"must be in [0, 1]"
        )

    sp = cfg.sampler
    if not sp.network:
        bad.append("sampler.network must be a non-empty network folder name")
    if sp.static_draw_count < 1:
        bad.append(f"sampler.static_draw_count ({sp.static_draw_count}) must be >= 1")
    for band, lo, hi in (
        ("T_supply", sp.T_supply_min, sp.T_supply_max),
        ("T_ground", sp.T_ground_min, sp.T_ground_max),
        ("residential_minority", sp.residential_minority_min, sp.residential_minority_max),
        ("noise_sigma", sp.noise_sigma_min, sp.noise_sigma_max),
        ("seasonal_amplitude", sp.seasonal_amplitude_min, sp.seasonal_amplitude_max),
    ):
        if lo > hi:
            bad.append(f"sampler.{band} band needs min ({lo}) <= max ({hi})")
    if sp.T_supply_max >= sp.T_ground_min:
        bad.append(
            f"sampler.T_supply_max ({sp.T_supply_max}) must stay below "
            f"sampler.T_ground_min ({sp.T_ground_min}): every sampled scenario "
            f"must keep the heat-gain sign (supply below ground)"
        )
    if not 0.0 < sp.residential_minority_min or not sp.residential_minority_max < 1.0:
        bad.append(
            f"sampler.residential_minority band ({sp.residential_minority_min}, "
            f"{sp.residential_minority_max}) must lie inside (0, 1): the minority "
            f"class must exist without being the majority"
        )
    if sp.noise_sigma_min < 0.0:
        bad.append(f"sampler.noise_sigma_min ({sp.noise_sigma_min}) must be >= 0")
    if not 0.0 <= sp.knot_jitter < 1.0:
        bad.append(
            f"sampler.knot_jitter ({sp.knot_jitter}) must be in [0, 1): the factor "
            f"band U(1-j, 1+j) must keep knots positive"
        )
    if not 0.0 <= sp.seasonal_amplitude_min or not sp.seasonal_amplitude_max <= 1.0:
        bad.append(
            f"sampler.seasonal_amplitude band ({sp.seasonal_amplitude_min}, "
            f"{sp.seasonal_amplitude_max}) must lie in [0, 1]: amplitude beyond 1 "
            f"drives loads negative"
        )
    for tier_name, tier in (("bulk", sp.bulk), ("week", sp.week), ("month", sp.month)):
        if tier.horizon <= 0:
            bad.append(f"sampler.{tier_name}.horizon ({tier.horizon}) must be positive")
        else:
            steps = tier.horizon / cfg.solver.dt
            if abs(steps - round(steps)) > _DT_CONSISTENCY_TOL * max(steps, 1.0):
                bad.append(
                    f"sampler.{tier_name}.horizon ({tier.horizon}) must be an "
                    f"integer multiple of solver.dt ({cfg.solver.dt}): n_steps and "
                    f"onset quantisation live on the step grid"
                )
        for cls in ("normal", "leak", "fouling", "bypass"):
            n = getattr(tier, cls)
            if n < 0:
                bad.append(f"sampler.{tier_name}.{cls} ({n}) must be >= 0")

    if cfg.driver.workers < 1:
        bad.append(f"driver.workers ({cfg.driver.workers}) must be >= 1")

    for name, cov in (
        ("sensors.consumer_coverage", cfg.sensors.consumer_coverage),
        ("sensors.pressure_coverage", cfg.sensors.pressure_coverage),
    ):
        if not 0.0 <= cov <= 1.0:
            bad.append(f"{name} ({cov}) must be in [0, 1]")

    if cfg.poc.static_scenario_id < 0:
        bad.append(f"poc.static_scenario_id ({cfg.poc.static_scenario_id}) must be >= 0")
    if not cfg.poc.network:
        bad.append("poc.network must be a non-empty network folder name")
    if not cfg.paths.ditec_data:
        bad.append("paths.ditec_data must be a non-empty path")

    return bad


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a configuration file.

    Args:
        path: YAML file to load; defaults to the repository's
            ``configs/default.yaml``.

    Raises:
        ConfigError: on missing file, unknown/missing keys, wrong types, or
            physically inconsistent values.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_PATH
    try:
        raw = yaml.safe_load(cfg_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {cfg_path}") from exc

    cfg = _build(Config, raw, "config")
    problems = _validate(cfg)
    if problems:
        raise ConfigError("; ".join(problems))
    return cfg
