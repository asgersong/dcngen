"""Sensor-mask seam: static sensor availability — the
plant is always fully sensed; a configured fraction of consumers is
heat-metered and of junctions pressure-sensed, drawn from a passed rng.
"""

import dataclasses

import numpy as np
import pytest

from dcngen.config import load_config
from dcngen.sensors.mask import draw_sensor_mask
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

cfg = load_config()


@pytest.fixture(scope="module")
def mini_dcn(tmp_path_factory):
    from tests.conftest import write_ditec_folder

    folder = write_ditec_folder(tmp_path_factory.mktemp("nets") / "mini_net")
    return build_dcn(load_ditec_network(folder, scenario_id=0), cfg)


def with_coverage(consumer: float, pressure: float):
    sensors = dataclasses.replace(
        cfg.sensors, consumer_coverage=consumer, pressure_coverage=pressure
    )
    return dataclasses.replace(cfg, sensors=sensors)


def test_mask_is_deterministic_and_respects_coverage(mini_dcn):
    # mini net: 3 consumers, 3 junctions; 2/3 coverage -> 2 of each
    c = with_coverage(2 / 3, 2 / 3)
    a = draw_sensor_mask(mini_dcn, c, np.random.default_rng(4))
    b = draw_sensor_mask(mini_dcn, c, np.random.default_rng(4))
    assert a == b

    assert len(a.heat_metered) == 2
    assert len(a.pressure_sensed) == 2
    assert set(a.heat_metered) <= set(mini_dcn.consumers)
    assert set(a.pressure_sensed) <= set(mini_dcn.pairing)
    assert a.plant_sensed is True  # plant fully sensed, always

    other = draw_sensor_mask(mini_dcn, c, np.random.default_rng(9))
    assert isinstance(other.heat_metered, tuple)  # stable, hashable record


def test_zero_and_full_coverage_edge_cases(mini_dcn):
    none = draw_sensor_mask(mini_dcn, with_coverage(0.0, 0.0), np.random.default_rng(0))
    assert none.heat_metered == () and none.pressure_sensed == ()
    assert none.plant_sensed is True

    full = draw_sensor_mask(mini_dcn, with_coverage(1.0, 1.0), np.random.default_rng(0))
    assert set(full.heat_metered) == set(mini_dcn.consumers)
    assert set(full.pressure_sensed) == set(mini_dcn.pairing)
