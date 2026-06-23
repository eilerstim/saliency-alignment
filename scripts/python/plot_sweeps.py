"""Paper figures from the aggregated sweep CSV (scripts/python/aggregate_results.py).

Produces, into --out-dir (default figs/):
  dose     lambda dose-response: AMR/AP/NSS and val CE vs lambda (Experiment A)
  pareto   localization vs a downstream metric, labelled by lambda (Experiment A)
  steps    localization + downstream vs training length (Experiment B)
  rank     localization vs LoRA rank, with the full-FT reference line (Experiment C)

Means +/- std are taken across seeds. Requires matplotlib (pip install matplotlib).

Usage:
    python scripts/python/plot_sweeps.py results/summary.csv --kind all
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def load(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def agg(rows: list[dict], x_key: str, y_key: str) -> tuple[list, list, list]:
    """Group y by x across seeds; return sorted (xs, means, stds)."""
    buckets: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        x, y = fnum(r, x_key), fnum(r, y_key)
        if x is not None and y is not None:
            buckets[x].append(y)
    xs = sorted(buckets)
    means = [statistics.fmean(buckets[x]) for x in xs]
    stds = [statistics.pstdev(buckets[x]) if len(buckets[x]) > 1 else 0.0 for x in xs]
    return xs, means, stds


def _filter(rows, **eq) -> list[dict]:
    out = rows
    for k, v in eq.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _first_downstream_col(rows: list[dict]) -> str | None:
    for r in rows:
        for k in r:
            if k.startswith("down/"):
                return k
    return None


def plot_dose(plt, rows, out_dir):
    rows = _filter(rows, criterion="kl", method="full", freeze="lm_only")
    if not rows:
        print("dose: no KL/full/lm_only rows; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in ("AMR_mean", "AP_mean", "NSS_mean"):
        xs, m, s = agg(rows, "lambda", metric)
        if xs:
            ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=metric.split("_")[0])
    ax.set_xlabel("lambda")
    ax.set_ylabel("localization (mean +/- std)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.legend()
    ax.set_title("Lambda dose-response")
    ax2 = ax.twinx()
    xs, m, s = agg(rows, "lambda", "val_ce_loss")
    if xs:
        ax2.errorbar(xs, m, yerr=s, marker="s", color="gray", alpha=0.6, label="val CE")
        ax2.set_ylabel("val cross-entropy")
        ax2.legend(loc="lower right")
    _save(fig, out_dir / "dose_response.png")


def plot_pareto(plt, rows, out_dir, down_col):
    rows = _filter(rows, criterion="kl", method="full", freeze="lm_only")
    down_col = down_col or _first_downstream_col(rows)
    if not rows or not down_col:
        print("pareto: missing rows or downstream column; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = [(fnum(r, "AP_mean"), fnum(r, down_col), r.get("lambda")) for r in rows]
    pts = [(x, y, lam) for x, y, lam in pts if x is not None and y is not None]
    for x, y, lam in pts:
        ax.scatter(x, y)
        ax.annotate(f"λ={lam}", (x, y), fontsize=8)
    ax.set_xlabel("localization (AP, mean)")
    ax.set_ylabel(down_col)
    ax.set_title("Localization vs downstream (Pareto)")
    _save(fig, out_dir / "pareto.png")


def plot_steps(plt, rows, out_dir, down_col):
    rows = _filter(rows, criterion="kl", method="full", freeze="lm_only")
    down_col = down_col or _first_downstream_col(rows)
    if not rows:
        print("steps: no rows; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in ("AMR_mean", "AP_mean", "NSS_mean"):
        xs, m, s = agg(rows, "steps", metric)
        if xs:
            ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=metric.split("_")[0])
    ax.set_xlabel("training steps")
    ax.set_ylabel("localization")
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.set_title("Localization & capability vs length")
    if down_col:
        ax2 = ax.twinx()
        xs, m, s = agg(rows, "steps", down_col)
        if xs:
            ax2.errorbar(xs, m, yerr=s, marker="s", color="gray", label=down_col)
            ax2.set_ylabel(down_col)
            ax2.legend(loc="lower right")
    _save(fig, out_dir / "length_sweep.png")


def plot_rank(plt, rows, out_dir):
    lora = _filter(rows, criterion="kl", method="lora")
    full = _filter(rows, criterion="kl", method="full", freeze="lm_only")
    if not lora:
        print("rank: no LoRA rows; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in ("AMR_mean", "AP_mean", "NSS_mean"):
        xs, m, s = agg(lora, "rank", metric)
        if not xs:
            continue
        line = ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=metric.split("_")[0])
        _, fm, _ = agg(full, "lambda", metric)  # full-FT reference line
        if fm:
            ax.axhline(fm[-1], color=line[0].get_color(), ls="--", alpha=0.6)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("localization")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.set_title("LoRA rank vs full FT (dashed = full-FT reference)")
    _save(fig, out_dir / "lora_rank.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--kind", choices=("dose", "pareto", "steps", "rank", "all"), default="all"
    )
    ap.add_argument("--downstream-col", default=None, help="e.g. down/pope/acc")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.csv)
    if args.kind in ("dose", "all"):
        plot_dose(plt, rows, args.out_dir)
    if args.kind in ("pareto", "all"):
        plot_pareto(plt, rows, args.out_dir, args.downstream_col)
    if args.kind in ("steps", "all"):
        plot_steps(plt, rows, args.out_dir, args.downstream_col)
    if args.kind in ("rank", "all"):
        plot_rank(plt, rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
