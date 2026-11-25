"""Figure: attention localization across training length (aligned vs. control).

Reads ``results/summary.csv`` (from ``aggregate_results.py``) and draws AMR,
AP and NSS against training steps for the lambda=0.5 LM-only runs and the
lambda=0 controls, with the base model as a dashed reference line. Seeds
other than the first are drawn as faint markers when present.

    python scripts/python/plot_metrics_over_time.py results/summary.csv --out figures/metrics_over_time.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3
import matplotlib.pyplot as plt  # noqa: E402

METRICS = [("AMR_mean", "AMR"), ("AP_mean", "AP"), ("NSS_mean", "NSS")]
SERIES = {
    "kl|0.5": ("Alignment ($\\lambda{=}0.5$)", "tab:red", "o"),
    "default|0": ("Fine-tuning only ($\\lambda{=}0$)", "tab:blue", "s"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "summary", type=Path, nargs="?", default=Path("results/summary.csv")
    )
    ap.add_argument("--out", type=Path, default=Path("figures/metrics_over_time.pdf"))
    ap.add_argument("--freeze", default="lm_only")
    ap.add_argument("--lr", default="2e-5")
    args = ap.parse_args()

    with args.summary.open() as f:
        rows = list(csv.DictReader(f))

    base = next((r for r in rows if r["run_id"].startswith("llava-hf__")), None)
    series: dict[str, dict[int, list[tuple[int, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if (
            r.get("method") != "full"
            or r.get("freeze") != args.freeze
            or r.get("lr") != args.lr
        ):
            continue
        lam = (
            (r.get("lambda") or "").rstrip("0").rstrip(".")
            if "." in (r.get("lambda") or "")
            else r.get("lambda")
        )
        key = f"{r.get('criterion')}|{lam}"
        if key not in SERIES or not r.get("steps"):
            continue
        series[key][int(r["steps"])].append((int(r.get("seed") or 0), r))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(9.5, 2.6))
    for ax, (col, label) in zip(axes, METRICS, strict=True):
        for key, (name, color, marker) in SERIES.items():
            by_step = series.get(key, {})
            if not by_step:
                continue
            steps = sorted(by_step)
            # first seed per step as the line; other seeds as faint markers
            line_vals = [float(sorted(by_step[s])[0][1][col]) for s in steps]
            ax.plot(
                steps, line_vals, color=color, marker=marker, ms=4, lw=1.4, label=name
            )
            for s in steps:
                for _, r in sorted(by_step[s])[1:]:
                    ax.plot(
                        [s],
                        [float(r[col])],
                        color=color,
                        marker=marker,
                        ms=4,
                        alpha=0.35,
                        lw=0,
                    )
        if base is not None and base.get(col):
            ax.axhline(
                float(base[col]), color="gray", ls="--", lw=1, label="Base model"
            )
        ax.set_title(label)
        ax.set_xlabel("Training steps")
        ax.set_xticks([200, 800, 1600, 2400])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
