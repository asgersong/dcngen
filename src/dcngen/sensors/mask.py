"""Static sensor-availability mask.

The plant is always fully sensed; a configured fraction of consumers is
heat-metered and of junctions pressure-sensed. Placement is random
(degree/betweenness strategies are future work); candidates are drawn
in sorted-id order so the same seed gives the same mask regardless of
mapping order. Counts are ``round(coverage * n)`` (Python's banker's
rounding at exact halves).
"""

from dataclasses import dataclass

import numpy as np

from dcngen.config import Config
from dcngen.topology.dcnifier import DCNetwork


@dataclass(frozen=True)
class SensorMask:
    """Which elements the sparse-sensor view may read (ground truth is
    always stored in full alongside — the mask never filters the record)."""

    heat_metered: tuple[str, ...]  # consumer junction ids with heat meters
    pressure_sensed: tuple[str, ...]  # junction ids with pressure sensors
    plant_sensed: bool  # always True: plant fully sensed


def _draw(ids: list[str], coverage: float, rng: np.random.Generator) -> tuple[str, ...]:
    n = round(coverage * len(ids))
    picked = rng.choice(len(ids), size=n, replace=False)
    return tuple(ids[i] for i in sorted(picked))


def draw_sensor_mask(
    dcn: DCNetwork, cfg: Config, rng: np.random.Generator
) -> SensorMask:
    """Draw the static mask at the configured coverages.

    Args:
        dcn: built DCN.
        cfg: ``sensors.consumer_coverage`` / ``sensors.pressure_coverage``.
        rng: seeded generator; the sole source of randomness.
    """
    return SensorMask(
        heat_metered=_draw(sorted(dcn.consumers), cfg.sensors.consumer_coverage, rng),
        pressure_sensed=_draw(sorted(dcn.pairing), cfg.sensors.pressure_coverage, rng),
        plant_sensed=True,
    )
