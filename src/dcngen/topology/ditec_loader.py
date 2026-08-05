"""Load DiTEC-WDN network folders: topology + pinned static draw.

The DiTEC dump stores each network as a folder of parquet shards: one column
per element plus a ``scenario_id`` column, with the topology embedded as an
``adj_list`` (``[start, end, link_id]`` triples) under the ``attrs`` key of
the parquet footer metadata. All values are SI (verified against Hanoi head
gradients, 2026-07-14): diameters/lengths/elevations/heads in m, demands in
m3/s, pipe roughness the Hazen-Williams C actually used by the simulation.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


class DitecFormatError(ValueError):
    """The folder does not match the DiTEC dump format."""


@dataclass(frozen=True)
class Pipe:
    """One adj_list entry: a pipe ``id`` running ``start -> end``."""

    id: str
    start: str
    end: str


@dataclass(frozen=True)
class StaticDraw:
    """One pinned static parameter draw, SI units.

    ``base_demand`` [m3/s] is the per-junction reduction of the dump's demand
    time series named by ``demand_reduction`` — the scalar that
    flow equivalence calls the DiTEC base demand. ``provenance`` maps each
    input kind to the shard file(s) it was read from (comma-joined), for the
    scenario metadata card.
    """

    scenario_id: int
    pipe_diameter: dict[str, float]  # [m]
    pipe_length: dict[str, float]  # [m]
    pipe_roughness: dict[str, float]  # Hazen-Williams C [-]
    junction_elevation: dict[str, float]  # [m]
    reservoir_head: dict[str, float]  # total head [m]
    base_demand: dict[str, float]  # [m3/s]
    demand_reduction: str
    provenance: dict[str, str]


@dataclass(frozen=True)
class DitecNetwork:
    """A DiTEC topology plus its pinned static draw."""

    name: str
    junctions: tuple[str, ...]
    reservoirs: tuple[str, ...]
    pipes: tuple[Pipe, ...]
    draw: StaticDraw


_STATIC_KINDS = (
    "pipe_diameter",
    "pipe_length",
    "pipe_roughness",
    "junction_elevation",
    "reservoir_base_head",
)

# Unit-error tripwires, not physics: bounds wide enough for any plausible
# network in SI units, narrow enough to catch a shard in the wrong unit
# (e.g. diameters in mm read as m). Physical design constants stay in
# configs/default.yaml; these guard the *format* contract of the dump.
_PLAUSIBLE_SI = {
    "pipe_diameter": (0.01, 5.0),  # m
    "pipe_length": (0.5, 100_000.0),  # m
    "pipe_roughness": (10.0, 5000.0),  # Hazen-Williams C
    "junction_elevation": (-500.0, 5000.0),  # m
    "reservoir_base_head": (-500.0, 5000.0),  # m
    "junction_base_demand": (0.0, 10.0),  # m3/s
}


def _check_plausible(kind: str, values: dict[str, float], folder: Path) -> None:
    lo, hi = _PLAUSIBLE_SI[kind]
    bad = {el: v for el, v in values.items() if not lo <= v <= hi}
    if bad:
        raise DitecFormatError(
            f"{kind} values outside plausible SI range [{lo}, {hi}] in {folder}: "
            f"{bad} — wrong units?"
        )


def _shards(folder: Path, kind: str, suffix: str) -> list[Path]:
    """All shards of one kind, in numeric shard order (-0-, -1-, ..., -10-)."""
    matches = list(folder.glob(f"{kind}-*-{suffix}.parquet"))
    if not matches:
        raise DitecFormatError(f"no {kind} {suffix} shard in {folder}")

    def index(p: Path) -> int:
        return int(p.name[len(kind) + 1 :].split("-")[0])

    return sorted(matches, key=index)


def _shard(folder: Path, kind: str, suffix: str) -> Path:
    return _shards(folder, kind, suffix)[0]


def _static_row(folder: Path, kind: str, scenario_id: int) -> tuple[dict[str, float], str]:
    """The scenario's row of a static-input kind, plus the shard it came from."""
    for path in _shards(folder, kind, "static_input"):
        table = pq.read_table(path, filters=[("scenario_id", "=", float(scenario_id))])
        if table.num_rows == 1:
            row = table.to_pydict()
            return {
                el: float(row[el][0]) for el in row if el != "scenario_id"
            }, path.name
        if table.num_rows > 1:
            raise DitecFormatError(
                f"{path} has {table.num_rows} rows for scenario_id {scenario_id}"
            )
    raise DitecFormatError(f"no scenario_id {scenario_id} in any {kind} shard of {folder}")


def _demand_time_mean(
    folder: Path, scenario_id: int
) -> tuple[dict[str, float], str]:
    """Per-junction time-mean demand [m3/s] over all demand shards."""
    sums: dict[str, float] = {}
    count = 0
    paths = _shards(folder, "junction_base_demand", "dynamic_input")
    for path in paths:
        table = pq.read_table(path, filters=[("scenario_id", "=", float(scenario_id))])
        if table.num_rows == 0:
            continue
        data = table.to_pydict()
        count += table.num_rows
        for el, values in data.items():
            if el in ("scenario_id", "time_id"):
                continue
            sums[el] = sums.get(el, 0.0) + sum(values)
    if count == 0:
        raise DitecFormatError(
            f"no scenario_id {scenario_id} in any junction_base_demand shard of {folder}"
        )
    return {el: s / count for el, s in sums.items()}, ",".join(p.name for p in paths)


def _element_columns(path: Path) -> tuple[str, ...]:
    names = pq.ParquetFile(path).schema_arrow.names
    return tuple(n for n in names if n not in ("scenario_id", "time_id"))


def _read_adj_list(path: Path) -> list[Pipe]:
    meta = pq.ParquetFile(path).metadata.metadata or {}
    if b"attrs" not in meta:
        raise DitecFormatError(f"no 'attrs' footer metadata in {path}")
    attrs = json.loads(meta[b"attrs"].decode())
    if "adj_list" not in attrs:
        raise DitecFormatError(f"footer 'attrs' of {path} has no adj_list")
    return [Pipe(id=link, start=start, end=end) for start, end, link in attrs["adj_list"]]


def summarize_network(folder: str | Path, scenario_id: int = 0) -> str:
    """Human-readable summary of a network folder and one pinned draw."""
    net = load_ditec_network(folder, scenario_id)
    d = net.draw

    def rng(values: dict[str, float], unit: str) -> str:
        return f"{min(values.values()):.3f} .. {max(values.values()):.3f} {unit}"

    n_res = len(net.reservoirs)
    lines = [
        f"{net.name} (static draw scenario_id={d.scenario_id})",
        f"  {len(net.junctions)} junctions, {n_res} reservoir{'s' if n_res != 1 else ''}, "
        f"{len(net.pipes)} pipes",
        f"  pipe diameter   {rng(d.pipe_diameter, 'm')}",
        f"  pipe length     {rng(d.pipe_length, 'm')}",
        f"  pipe roughness  {rng(d.pipe_roughness, '(Hazen-Williams C)')}",
        f"  elevation       {rng(d.junction_elevation, 'm')}",
        f"  reservoir head  {rng(d.reservoir_head, 'm')}",
        f"  base demand     {rng(d.base_demand, 'm3/s')} ({d.demand_reduction})",
    ]
    return "\n".join(lines)


def load_ditec_network(folder: str | Path, scenario_id: int) -> DitecNetwork:
    """Load one network folder pinned to one static parameter draw.

    Args:
        folder: DiTEC network folder (e.g. ``.../hanoi_8GB_1Y``).
        scenario_id: which static draw to pin (matched against the dump's
            ``scenario_id`` column, which is float-typed).
    """
    folder = Path(folder)
    elevation_shard = _shard(folder, "junction_elevation", "static_input")
    head_shard = _shard(folder, "reservoir_base_head", "static_input")

    pipes = _read_adj_list(elevation_shard)
    junctions = _element_columns(elevation_shard)
    reservoirs = _element_columns(head_shard)

    nodes = set(junctions) | set(reservoirs)
    for pipe in pipes:
        missing = {pipe.start, pipe.end} - nodes
        if missing:
            raise DitecFormatError(
                f"adj_list pipe {pipe.id} references node(s) {sorted(missing)} "
                f"absent from the junction/reservoir shards of {folder}"
            )

    values: dict[str, dict[str, float]] = {}
    provenance: dict[str, str] = {}
    for kind in _STATIC_KINDS:
        values[kind], provenance[kind] = _static_row(folder, kind, scenario_id)
        _check_plausible(kind, values[kind], folder)
    base_demand, provenance["junction_base_demand"] = _demand_time_mean(
        folder, scenario_id
    )
    _check_plausible("junction_base_demand", base_demand, folder)

    draw = StaticDraw(
        scenario_id=scenario_id,
        pipe_diameter=values["pipe_diameter"],
        pipe_length=values["pipe_length"],
        pipe_roughness=values["pipe_roughness"],
        junction_elevation=values["junction_elevation"],
        reservoir_head=values["reservoir_base_head"],
        base_demand=base_demand,
        demand_reduction="time_mean",
        provenance=provenance,
    )

    return DitecNetwork(
        name=folder.name,
        junctions=junctions,
        reservoirs=reservoirs,
        pipes=tuple(pipes),
        draw=draw,
    )


if __name__ == "__main__":  # python -m dcngen.topology.ditec_loader <folder> [scenario_id]
    print(summarize_network(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0))
