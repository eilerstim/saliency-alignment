"""Seed table: mean and sample SD over seeds of one configuration pair.

Reads ``results/summary.csv`` and reports every intrinsic metric and downstream
benchmark for the 800-step aligned (kl 0.5) and control (lambda 0) runs over
all seeds present. Stdlib only.

    python scripts/python/seed_table.py results/summary.csv > results/seed_table.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import DOWN_COLS, downstream, key_of  # noqa: E402

INTR = [("AMR_mean", "AMR", 2), ("AP_mean", "AP", 2), ("NSS_mean", "NSS", 2),
        ("val_ce_loss", "val-CE", 3), ("val_accuracy", "val-acc", 3)]


def pct(v) -> float:
    x = float(v)
    return x * 100 if x <= 1.0 else x


def values(rows: list[dict], key: str) -> dict[str, tuple[dict[int, float], int]]:
    by_seed = {int(r["seed"]): r for r in rows if key_of(r) == key and r.get("seed")}
    out: dict[str, tuple[dict[int, float], int]] = {}
    for col, name, nd in INTR:
        out[name] = ({s: float(r[col]) for s, r in by_seed.items() if r.get(col)}, nd)
    if by_seed:
        seeds = sorted(by_seed)
        per_seed = [downstream(by_seed[s]) for s in seeds]
        for i, name in enumerate(DOWN_COLS):
            out[name] = ({s: pct(v[i]) for s, v in zip(seeds, per_seed) if v[i] not in (None, "")}, 1)
    return out


def mean_sd(d: dict[int, float], nd: int) -> str:
    xs = [d[s] for s in sorted(d)]
    if not xs:
        return "–"
    m = statistics.fmean(xs)
    return f"{m:.{nd}f} ± {statistics.stdev(xs):.{nd}f} (n={len(xs)})" if len(xs) >= 2 else f"{m:.{nd}f} (n=1)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path, nargs="?", default=Path("results/summary.csv"))
    ap.add_argument("--steps", default="800")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.summary.open()))
    configs = [(f"kl|0.5|lm_only|2e-5|{args.steps}|", "aligned"), (f"default|0|lm_only|2e-5|{args.steps}|", "control")]
    data = {label: values(rows, key) for key, label in configs}
    metrics = [n for _, n, _ in INTR] + DOWN_COLS

    print(f"Seed table, {args.steps} steps, LM-only, lr 2e-5: mean ± sample SD over seeds; benchmarks in %.\n")
    print("| metric | aligned (kl 0.5) | control (lambda 0) | aligned − control |")
    print("|---|---|---|---|")
    for name in metrics:
        da, nd = data["aligned"].get(name, ({}, 2))
        dc, _ = data["control"].get(name, ({}, 2))
        diff = f"{statistics.fmean(da.values()) - statistics.fmean(dc.values()):+.{nd}f}" if da and dc else "–"
        print(f"| {name} | {mean_sd(da, nd)} | {mean_sd(dc, nd)} | {diff} |")

    seeds = sorted({s for d in data.values() for v, _ in d.values() for s in v})
    print("\n| metric | " + " | ".join(f"aligned s{s}" for s in seeds) + " | " + " | ".join(f"control s{s}" for s in seeds) + " |")
    print("|---|" + "---|" * (2 * len(seeds)))
    for name in metrics:
        da, nd = data["aligned"].get(name, ({}, 2))
        dc, _ = data["control"].get(name, ({}, 2))
        cells = [f"{da[s]:.{nd}f}" if s in da else "–" for s in seeds] + [f"{dc[s]:.{nd}f}" if s in dc else "–" for s in seeds]
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
