#!/usr/bin/env python
"""Dataset statistics for the thesis's dataset chapter.

Reads a generated dataset folder (the scenario cards — no Parquet loading,
so it runs in seconds even mid-generation) and emits both a human summary
and LaTeX tables ready to \\input into the thesis. Every number the
methodology chapter quotes should come from here rather than being copied
by hand, so a regenerated dataset updates the chapter mechanically.

    scripts/dataset_statistics.py [--out data/v1] [--latex tables/]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TIERS = ("bulk", "week", "month")
CLASSES = ("normal", "leak", "fouling", "bypass")


_FOLDER_RE = re.compile(r"uid(\d+)$")  # same contract as driver._sweep


def load_cards(out_dir: Path) -> list[dict]:
    """Every completed scenario card, uid-ordered.

    Published folders only: ``.tmp`` staging folders carry a full card the
    instant before their rename, so counting them double-counts a scenario
    (and an orphaned one — from a worker whose rename lost a race — would
    be counted forever).
    """
    cards = []
    for folder in sorted(out_dir.glob("uid*")):
        if not _FOLDER_RE.fullmatch(folder.name):
            continue
        card = folder / "card.json"
        if card.exists():
            cards.append(json.loads(card.read_text()))
    return cards


def folder_bytes(out_dir: Path) -> int:
    return sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())


def summarise(cards: list[dict], out_dir: Path) -> dict:
    """Every quantity the dataset chapter reports, in one dict."""
    shape: dict[tuple[str, str], int] = Counter()
    gate_pass: dict[tuple[str, str], int] = Counter()
    steps: dict[str, set[int]] = defaultdict(set)
    rule_fail: Counter = Counter()
    severity: dict[str, list[float]] = defaultdict(list)
    onset_frac: list[float] = []
    whole_horizon: Counter = Counter()
    sampled: dict[str, list[float]] = defaultdict(list)
    anchoring: Counter = Counter()
    pn25 = 0
    energy: list[float] = []
    unconverged: list[float] = []
    unconverged_ok: list[float] = []  # gate-passing scenarios only
    unmet_clean: list[float] = []
    unmet_clean_ok: list[float] = []
    draws: set[int] = set()
    archetypes: Counter = Counter()
    fault_sites: dict[str, Counter] = defaultdict(Counter)

    for c in cards:
        row = c.get("plan", {}).get("row", {})
        tier, cls = row.get("tier", "?"), row.get("scenario_class", "?")
        shape[(tier, cls)] += 1
        steps[tier].add(c["n_steps"])
        gate = c.get("validation", {})
        passed = bool(gate.get("passed"))
        gate_pass[(tier, cls)] += passed
        for r in gate.get("rules", []):
            if not r["passed"]:
                rule_fail[r["rule"]] += 1
            m = r["metrics"]
            if r["rule"] == "energy_closure":
                energy.append(m["residual_rel"])
            elif r["rule"] == "convergence_budget":
                unconverged.append(m.get("unconverged_frac", 0.0))
                # The gated dataset is the PASSING scenarios; a worst case that
                # includes rejects describes something nobody will train on, and
                # would read as though the release contains it.
                if passed:
                    unconverged_ok.append(m.get("unconverged_frac", 0.0))
            elif r["rule"] == "unmet_load":
                unmet_clean.append(m.get("unmet_frac_outside_fault", 0.0))
                if passed:
                    unmet_clean_ok.append(m.get("unmet_frac_outside_fault", 0.0))
            elif r["rule"] == "pressure_rails":
                pn25 += int(m.get("pn25_escalated", 0))

        fault = c["fault"]
        if fault["kind"] != "none":
            severity[fault["kind"]].append(fault["severity"])
            fault_sites[fault["kind"]][fault["location"]] += 1
            whole_horizon[(fault["kind"], bool(fault["whole_horizon"]))] += 1
            if not fault["whole_horizon"] and fault["onset_step"] is not None:
                onset_frac.append(fault["onset_step"] / c["n_steps"])

        draws.add(c["static_draw_id"])
        anchoring[c["plant"].get("anchoring", "ditec")] += 1
        sampled["T_supply"].append(row.get("T_supply"))
        sampled["T_ground"].append(row.get("T_ground"))
        sampled["lambda_pur"].append(row.get("lambda_pur"))
        sampled["lambda_soil"].append(row.get("lambda_soil"))
        sampled["burial_cover"].append(row.get("burial_cover"))
        for a in (row.get("archetypes") or {}).values():
            archetypes[a] += 1

    first = cards[0] if cards else {}
    return {
        "n": len(cards),
        "shape": shape,
        "gate_pass": gate_pass,
        "steps": {t: sorted(v) for t, v in steps.items()},
        "rule_fail": rule_fail,
        "severity": severity,
        "onset_frac": onset_frac,
        "whole_horizon": whole_horizon,
        "sampled": {k: [x for x in v if x is not None] for k, v in sampled.items()},
        "anchoring": anchoring,
        "pn25": pn25,
        "energy": energy,
        "unconverged": unconverged,
        "unconverged_ok": unconverged_ok,
        "unmet_clean": unmet_clean,
        "unmet_clean_ok": unmet_clean_ok,
        "draws": len(draws),
        "archetypes": archetypes,
        "fault_sites": fault_sites,
        "network": first.get("network"),
        "dt": first.get("dt"),
        "junctions": len(first.get("pairing", {})),
        "pipes": len(first.get("pipe_pairing", {})),
        # the card spreads PlanMeta directly under "plan" (writer._card)
        "consumers": len(first.get("plan", {}).get("consumers", [])),
        "bytes": folder_bytes(out_dir),
        "seed": first.get("plan", {}).get("master_seed"),
        "numpy": first.get("plan", {}).get("numpy_version"),
        "bitgen": first.get("plan", {}).get("bit_generator"),
    }


def rng(xs: list[float]) -> str:
    return f"{min(xs):.4g}--{max(xs):.4g}" if xs else "--"


def print_summary(s: dict) -> None:
    print(f"network {s['network']}  dt {s['dt']:.0f} s  seed {s['seed']}  "
          f"({s['bitgen']}, numpy {s['numpy']})")
    print(f"graph: {s['junctions']} junctions x2 (+2 plant headers), "
          f"{s['pipes']} pipes x2, {s['consumers']} ETS crossovers")
    print(f"scenarios: {s['n']}   static draws used: {s['draws']}   "
          f"size on disk: {s['bytes'] / 2**30:.1f} GiB")
    print("\nshape (completed / gate-passed):")
    for tier in TIERS:
        row = "  " + f"{tier:6s}"
        for cls in CLASSES:
            n, p = s["shape"][(tier, cls)], s["gate_pass"][(tier, cls)]
            row += f"  {cls}: {p}/{n}"
        steps = s["steps"].get(tier)
        row += f"   [{steps[0] if steps else '?'} steps]"
        print(row)
    total = sum(s["shape"].values())
    passed = sum(s["gate_pass"].values())
    print(f"  gate pass rate: {passed}/{total}"
          + (f" ({passed / total:.1%})" if total else ""))
    if s["rule_fail"]:
        print(f"  failing rules: {dict(s['rule_fail'])}")
    print("\nphysics quality:")
    print(f"  energy residual      max {max(s['energy'], default=0):.2e}")
    print(f"  unconverged fraction max {max(s['unconverged_ok'], default=0):.3%}"
          f"  (gate-passing; incl. failures {max(s['unconverged'], default=0):.3%})")
    print(f"  clean unmet fraction max {max(s['unmet_clean_ok'], default=0):.3%}"
          f"  (gate-passing; incl. failures {max(s['unmet_clean'], default=0):.3%})")
    print(f"  pressure class       PN25-escalated: {s['pn25']}")
    print(f"  plant anchoring      {dict(s['anchoring'])}")
    print("\nsampled ranges (realised):")
    for k, v in s["sampled"].items():
        print(f"  {k:14s} {rng(v)}")
    print("\nfault severities (realised):")
    for kind, v in s["severity"].items():
        wh = s["whole_horizon"][(kind, True)]
        ab = s["whole_horizon"][(kind, False)]
        print(f"  {kind:8s} {rng(v)}   whole-horizon {wh} / abrupt {ab}"
              f"   distinct sites {len(s['fault_sites'][kind])}")
    if s["onset_frac"]:
        print(f"  abrupt onsets at {rng(s['onset_frac'])} of horizon")
    print(f"\narchetype assignments (consumer-scenarios): {dict(s['archetypes'])}")


def _quality_table(s: dict) -> list[str]:
    """Gate outcome per class + the physics residuals behind it.

    The thesis reports the gate as evidence, so the numbers it quotes must
    come from here rather than from a run's console output — a regenerated
    dataset moves them.
    """
    lines = [
        "% generated by scripts/dataset_statistics.py — do not edit by hand",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Scenario class & Generated & Gate-passing & Rate \\\\",
        "\\hline",
    ]
    for cls in CLASSES:
        n = sum(s["shape"][(t, cls)] for t in TIERS)
        p = sum(s["gate_pass"][(t, cls)] for t in TIERS)
        lines.append(f"{cls} & {n} & {p} & {p / n:.1%} \\\\".replace("%", "\\%"))
    total = sum(s["shape"].values())
    passed = sum(s["gate_pass"].values())
    lines += [
        "\\hline",
        f"\\textbf{{all}} & {total} & {passed} & {passed / total:.1%} \\\\".replace(
            "%", "\\%"
        ),
        "\\hline",
        "\\end{tabular}",
    ]
    return lines


def _residual_table(s: dict) -> list[str]:
    """Worst-case physics residuals across the GATE-PASSING release.

    Budgeted quantities are reported over passing scenarios only: a worst
    case drawn from the rejects describes a scenario nobody will train on,
    and in a thesis table it would read as though the release contains it.
    The energy residual is reported over everything, because it is an
    identity that holds regardless of whether a scenario passed.
    """
    n = sum(s["shape"].values())
    rows = [
        ("First-law energy residual (all scenarios)",
         f"{max(s['energy'], default=0):.2e}", "exact to machine precision"),
        ("Unconverged timesteps", f"{max(s['unconverged_ok'], default=0):.2%}",
         "budget 2%, gate-passing only"),
        ("Unmet load, clean stretches", f"{max(s['unmet_clean_ok'], default=0):.2%}",
         "budget 0.5%, gate-passing only"),
        ("Scenarios escalated to the PN25 class", str(s["pn25"]), f"of {n}"),
        ("Plant anchored at the rail fallback",
         str(s["anchoring"].get("rail_fallback", 0)), f"of {n}"),
    ]
    lines = [
        "% generated by scripts/dataset_statistics.py — do not edit by hand",
        "\\begin{tabular}{lrl}",
        "\\hline",
        "Quantity & Worst case & Note \\\\",
        "\\hline",
    ]
    lines += [f"{a} & {b} & {c} \\\\".replace("%", "\\%") for a, b, c in rows]
    lines += ["\\hline", "\\end{tabular}"]
    return lines


def latex_tables(s: dict, out: Path) -> None:
    """Four \\input-ready tables: shape, sampling, gate outcome, residuals."""
    out.mkdir(parents=True, exist_ok=True)
    horizons = {"bulk": "24\\,h", "week": "7\\,d", "month": "30\\,d"}
    lines = [
        "% generated by scripts/dataset_statistics.py — do not edit by hand",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Tier & Horizon & Steps & Normal & Leak & Fouling & Bypass \\\\",
        "\\hline",
    ]
    for tier in TIERS:
        steps = s["steps"].get(tier)
        counts = " & ".join(str(s["shape"][(tier, c)]) for c in CLASSES)
        lines.append(
            f"{tier} & {horizons[tier]} & {steps[0] if steps else '--'} & {counts} \\\\"
        )
    totals = " & ".join(
        str(sum(s["shape"][(t, c)] for t in TIERS)) for c in CLASSES
    )
    lines += [
        "\\hline",
        f"\\textbf{{total}} & & & {totals} \\\\",
        "\\hline",
        "\\end{tabular}",
    ]
    (out / "dataset-shape.tex").write_text("\n".join(lines) + "\n")

    band = {
        "T_supply": ("Plant supply temperature", "\\si{\\degreeCelsius}"),
        "T_ground": ("Ground temperature", "\\si{\\degreeCelsius}"),
        "lambda_pur": ("PUR conductivity", "\\si{\\watt\\per\\metre\\per\\kelvin}"),
        "lambda_soil": ("Soil conductivity", "\\si{\\watt\\per\\metre\\per\\kelvin}"),
        "burial_cover": ("Burial cover", "\\si{\\metre}"),
    }
    lines = [
        "% generated by scripts/dataset_statistics.py — do not edit by hand",
        "\\begin{tabular}{llr}",
        "\\hline",
        "Quantity & Unit & Realised range \\\\",
        "\\hline",
    ]
    for key, (label, unit) in band.items():
        lines.append(f"{label} & {unit} & {rng(s['sampled'].get(key, []))} \\\\")
    for kind in ("leak", "fouling", "bypass"):
        if s["severity"].get(kind):
            lines.append(
                f"{kind.capitalize()} severity & -- & {rng(s['severity'][kind])} \\\\"
            )
    lines += ["\\hline", "\\end{tabular}"]
    (out / "dataset-sampling.tex").write_text("\n".join(lines) + "\n")
    (out / "dataset-gate.tex").write_text("\n".join(_quality_table(s)) + "\n")
    (out / "dataset-residuals.tex").write_text("\n".join(_residual_table(s)) + "\n")
    written = ("dataset-shape", "dataset-sampling", "dataset-gate", "dataset-residuals")
    print("\nwrote " + ", ".join(f"{out}/{n}.tex" for n in written))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/v1"))
    ap.add_argument("--latex", type=Path, default=None,
                    help="also write \\input-ready tables to this folder")
    args = ap.parse_args()

    cards = load_cards(args.out)
    if not cards:
        print(f"no completed scenarios in {args.out}")
        return 1
    s = summarise(cards, args.out)
    print_summary(s)
    if args.latex:
        latex_tables(s, args.latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
