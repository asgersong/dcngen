"""Config seam: typed access to every constant the project has committed to.

Expected values are independent literals, never recomputed from the YAML.
Provisional defaults are pinned in a separate block so changing one is a
deliberate, visible edit rather than silent drift.
"""

from pathlib import Path

import pytest

from dcngen.config import ConfigError, load_config
from tests.conftest import write_mutated_default


def test_default_config_exposes_design_constants():
    cfg = load_config()

    # chilled-water properties
    assert cfg.water.cp == 4186.0          # J/(kg K)
    assert cfg.water.rho == 999.5          # kg/m3
    assert cfg.water.mu == 1.3e-3          # Pa s

    # thermal design regime
    assert cfg.thermal.T_supply == 6.0     # degC
    assert cfg.thermal.T_return_design == 13.0
    assert cfg.thermal.dT_design == 7.0    # K
    assert cfg.thermal.T_ground == 20.0    # degC
    assert cfg.thermal.init_residence_cap == 21600.0  # s (6 h):
    # near-stagnant water last moved at yesterday's evening shoulder

    # per-pipe heat-gain derivation knob bounds
    assert cfg.heat_gain.mode == "derived"
    assert cfg.heat_gain.lambda_pur_min == 0.023   # W/(m K), fresh conti PUR
    assert cfg.heat_gain.lambda_pur_max == 0.029   # EN 253 lambda50 limit
    assert cfg.heat_gain.lambda_soil_min == 1.0    # EN 13941 soil classes
    assert cfg.heat_gain.lambda_soil_max == 1.6
    assert cfg.heat_gain.burial_cover_min == 0.6   # m; H = cover + D_c/2
    assert cfg.heat_gain.burial_cover_max == 1.2
    assert cfg.heat_gain.scalar_U == 0.35          # W/(m K) fallback point value
    assert cfg.heat_gain.scalar_U_min == 0.2       # ablation band
    assert cfg.heat_gain.scalar_U_max == 0.6

    # ETS secondary-side closure: one
    # approach key, the secondary program derived as primary + approach at
    # both ends — no absolute secondary setpoints remain
    assert cfg.ets.approach == 1.0            # K (IEA DHC Annex VI: <= 1 degC)
    assert cfg.ets.max_flow_factor == 2.0     # valve limit, x design flow
    assert cfg.ets.min_dp == 10.0             # m (~1.0 bar), IEA DHC Annex XII 7.2.1.1

    # per-timestep solve (iteration scheme)
    assert cfg.solver.dt == 600.0             # s
    assert cfg.solver.flow_tol_rel == 1e-3
    assert cfg.solver.relaxation == 0.5
    assert cfg.solver.max_iterations == 15  # pilot-scale tail (was 10)

    # validation gate thresholds
    assert cfg.validation.mass_balance_rel_tol == 1e-6
    assert cfg.validation.max_velocity == 2.5          # m/s
    assert cfg.validation.max_unconverged_frac == 0.02  # pilot p95 1.6%
    assert cfg.validation.pressure_ceiling_pn25 == 229.5  # m, 0.9 x PN25
    assert cfg.validation.max_clean_unmet_frac == 0.005   # clean-unmet budget

    # healthy-UA sizing margin (PHE fouling margin 5-10 %, HEI #141)
    assert cfg.ets.healthy_ua_factor_min == 1.00
    assert cfg.ets.healthy_ua_factor_max == 1.10

    # fault conventions (flow-fraction severity, mid-horizon onset)
    assert cfg.faults.leak_severity_min == 0.01
    assert cfg.faults.leak_severity_max == 0.10
    assert cfg.faults.onset_window_start == 0.2
    assert cfg.faults.onset_window_end == 0.8

    # ETS fault severity bands
    assert cfg.faults.fouling_severity_min == 0.30   # worst closed-circuit m
    assert cfg.faults.fouling_severity_max == 0.85   # gap to the design margin
    assert cfg.faults.bypass_fraction_min == 0.05    # 10x FCI 70-2 Class II
    assert cfg.faults.bypass_fraction_max == 0.50    # plant dT at half design
    assert cfg.faults.bypass_whole_horizon_prob == 0.5  # temporal split

    # sampler bands + tier counts
    s = cfg.sampler
    assert s.network == "hanoi_8GB_1Y"
    assert s.static_draw_count == 1000        # DiTEC validated draws U{0..999}
    assert (s.T_supply_min, s.T_supply_max) == (5.0, 7.0)    # degC
    assert (s.T_ground_min, s.T_ground_max) == (15.0, 30.0)  # degC
    assert (s.residential_minority_min, s.residential_minority_max) == (0.25, 0.35)
    assert (s.noise_sigma_min, s.noise_sigma_max) == (0.02, 0.2)  # DiTEC band
    assert s.knot_jitter == 0.1               # +/-10 % per (scenario, archetype)
    assert (s.seasonal_amplitude_min, s.seasonal_amplitude_max) == (0.1, 0.5)
    # tier table: horizon [s] + exact class counts (dataset shape)
    assert (s.bulk.horizon, s.bulk.normal, s.bulk.leak, s.bulk.fouling, s.bulk.bypass) \
        == (86400.0, 1000, 800, 550, 450)
    assert (s.week.horizon, s.week.normal, s.week.leak, s.week.fouling, s.week.bypass) \
        == (604800.0, 80, 50, 40, 30)
    # month tier is the long-horizon record; year-long span fidelity is
    # future work
    assert (s.month.horizon, s.month.normal, s.month.leak, s.month.fouling, s.month.bypass) \
        == (2592000.0, 6, 5, 3, 2)

    # driver: conservative default; the generation box runs ~48 effective
    # workers, set per run
    assert cfg.driver.workers == 4

    # data + reproducibility
    assert cfg.paths.ditec_data == Path("/home/asgerp/ditec-wdn")
    assert cfg.seed == 42


def test_default_config_pins_provisional_defaults():
    # Starting defaults, pinned here so any change is a deliberate edit
    # (config + test together), never drift.
    cfg = load_config()

    assert cfg.solver.required_pressure == 5.0     # m, PDD full delivery
    assert cfg.solver.minimum_pressure == 0.0      # m, PDD zero delivery
    assert cfg.validation.energy_rel_tol == 0.03   # "within a few %"
    assert cfg.loads.noise_sigma == 0.05
    assert cfg.sensors.consumer_coverage == 0.3
    assert cfg.sensors.pressure_coverage == 0.1


def test_load_rejects_unphysical_water_properties(tmp_path):
    bad = write_mutated_default(tmp_path, lambda d: d["water"].__setitem__("cp", -1.0))
    with pytest.raises(ConfigError, match="cp"):
        load_config(bad)


def test_load_rejects_broken_temperature_ordering(tmp_path):
    # supply warmer than ground would invert the heat-gain physics
    bad = write_mutated_default(tmp_path, lambda d: d["thermal"].__setitem__("T_supply", 25.0))
    with pytest.raises(ConfigError, match="T_supply"):
        load_config(bad)


def test_load_rejects_non_positive_approach(tmp_path):
    # a zero/negative approach is unphysical: the secondary program would sit
    # on (or below) the primary and the heat exchanger could never transfer
    bad = write_mutated_default(tmp_path, lambda d: d["ets"].__setitem__("approach", 0.0))
    with pytest.raises(ConfigError, match="approach"):
        load_config(bad)


def test_load_rejects_thermodynamically_infeasible_ets_closure(tmp_path):
    # hot-inlet guard: an approach that vanishes in float addition leaves
    # the hot inlet at the design return — no finite UA reaches it
    bad = write_mutated_default(
        tmp_path, lambda d: d["ets"].__setitem__("approach", 1.0e-300)
    )
    with pytest.raises(ConfigError, match="hot inlet"):
        load_config(bad)


def test_load_rejects_removed_absolute_secondary_keys(tmp_path):
    # the pre-reshape absolute setpoints must not silently pass
    def reinstate_old_keys(d):
        d["ets"]["T_building_supply"] = 7.0
        d["ets"]["dT_building"] = 7.0

    bad = write_mutated_default(tmp_path, reinstate_old_keys)
    with pytest.raises(ConfigError, match="dT_building"):
        load_config(bad)


def test_load_rejects_missing_approach(tmp_path):
    bad = write_mutated_default(tmp_path, lambda d: d["ets"].pop("approach"))
    with pytest.raises(ConfigError, match="approach"):
        load_config(bad)


def test_load_rejects_unknown_keys(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["water"].__setitem__("heat_capacity", 4186.0)
    )
    with pytest.raises(ConfigError, match="heat_capacity"):
        load_config(bad)


def test_load_rejects_removed_global_pipe_u(tmp_path):
    # the removed global scalar key must not silently pass
    bad = write_mutated_default(
        tmp_path, lambda d: d["thermal"].__setitem__("pipe_U", 0.3)
    )
    with pytest.raises(ConfigError, match="pipe_U"):
        load_config(bad)


def test_load_rejects_fouling_band_reaching_one(tmp_path):
    # m = 1 is not a fault; the band must stay strictly below the healthy UA
    bad = write_mutated_default(
        tmp_path, lambda d: d["faults"].__setitem__("fouling_severity_max", 1.0)
    )
    with pytest.raises(ConfigError, match="fouling_severity"):
        load_config(bad)


def test_load_rejects_degenerate_bypass_band(tmp_path):
    # f = 0 is the healthy crossover, f = 1 starves the exchanger entirely
    for key, val in (("bypass_fraction_min", 0.0), ("bypass_fraction_max", 1.0)):
        bad = write_mutated_default(tmp_path, lambda d, k=key, v=val: d["faults"].__setitem__(k, v))
        with pytest.raises(ConfigError, match="bypass_fraction"):
            load_config(bad)


def test_load_rejects_undersized_healthy_ua_band(tmp_path):
    # installed UA sits at or above the design requirement
    bad = write_mutated_default(
        tmp_path, lambda d: d["ets"].__setitem__("healthy_ua_factor_min", 0.95)
    )
    with pytest.raises(ConfigError, match="healthy_ua_factor"):
        load_config(bad)


def test_load_rejects_bypass_variant_probability_outside_unit_interval(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["faults"].__setitem__("bypass_whole_horizon_prob", 1.5)
    )
    with pytest.raises(ConfigError, match="bypass_whole_horizon_prob"):
        load_config(bad)


def test_load_rejects_sampled_supply_band_reaching_the_ground_band(tmp_path):
    # scenario-wise heat-gain sign invariant: the warmest sampled plant
    # supply must stay below the coldest sampled ground temperature
    bad = write_mutated_default(
        tmp_path, lambda d: d["sampler"].__setitem__("T_supply_max", 16.0)
    )
    with pytest.raises(ConfigError, match="T_supply_max"):
        load_config(bad)


def test_load_rejects_inverted_sampler_band(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["sampler"].__setitem__("noise_sigma_min", 0.3)
    )
    with pytest.raises(ConfigError, match="noise_sigma"):
        load_config(bad)


def test_load_rejects_tier_horizon_off_the_step_grid(tmp_path):
    # onset quantisation and n_steps require horizon = integer x dt
    bad = write_mutated_default(
        tmp_path, lambda d: d["sampler"]["bulk"].__setitem__("horizon", 86500.0)
    )
    with pytest.raises(ConfigError, match="horizon"):
        load_config(bad)


def test_load_rejects_negative_tier_count(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["sampler"]["week"].__setitem__("leak", -1)
    )
    with pytest.raises(ConfigError, match="leak"):
        load_config(bad)


def test_load_rejects_knot_jitter_reaching_one(tmp_path):
    # a factor band U(1-j, 1+j) with j >= 1 allows non-positive knots
    bad = write_mutated_default(
        tmp_path, lambda d: d["sampler"].__setitem__("knot_jitter", 1.0)
    )
    with pytest.raises(ConfigError, match="knot_jitter"):
        load_config(bad)


def test_load_rejects_non_positive_worker_count(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["driver"].__setitem__("workers", 0)
    )
    with pytest.raises(ConfigError, match="workers"):
        load_config(bad)


def test_load_rejects_unknown_heat_gain_mode(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["heat_gain"].__setitem__("mode", "banana")
    )
    with pytest.raises(ConfigError, match="mode"):
        load_config(bad)


def test_load_rejects_inverted_heat_gain_range(tmp_path):
    def invert(d):
        d["heat_gain"]["lambda_pur_min"] = 0.031
    bad = write_mutated_default(tmp_path, invert)
    with pytest.raises(ConfigError, match="lambda_pur"):
        load_config(bad)


def test_load_rejects_scalar_u_outside_its_band(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["heat_gain"].__setitem__("scalar_U", 0.7)
    )
    with pytest.raises(ConfigError, match="scalar_U"):
        load_config(bad)


def test_load_rejects_pn25_ceiling_below_pn16(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["validation"].__setitem__("pressure_ceiling_pn25", 100.0)
    )
    with pytest.raises(ConfigError, match="pn25"):
        load_config(bad)


def test_load_rejects_clean_unmet_budget_outside_unit_interval(tmp_path):
    bad = write_mutated_default(
        tmp_path, lambda d: d["validation"].__setitem__("max_clean_unmet_frac", 1.5)
    )
    with pytest.raises(ConfigError, match="max_clean_unmet_frac"):
        load_config(bad)
