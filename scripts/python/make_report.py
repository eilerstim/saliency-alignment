"""Render the corrected re-run next to the paper's published numbers.

Reads ``results/summary.csv`` (written by ``aggregate_results.py``) and prints
a Markdown report: one table of intrinsic localization + validation loss and
one table of downstream benchmarks, each with the paper's value (old pipeline)
beside the corrected value. Stdlib only.

    python scripts/python/aggregate_results.py --out results/summary.csv
    python scripts/python/make_report.py results/summary.csv > results/report.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Paper values (old pipeline). Intrinsic: AMR, AP, NSS, val-CE.
# Downstream order: CountBench, CV-2D, CV-3D, MMStar, MMVP, O3, POPE,
# VLMsB accuracy, VLMsB bias rate (lower is better).
PAPER: dict[str, dict] = {
    "base": {
        "label": "Base LLaVA-1.5-7B",
        "intr": (1.25, 0.22, 0.26, None),
        "down": (41.1, 55.8, 52.3, 33.0, 62.7, 44.7, 84.3, 16.3, 41.7),
    },
    "kl|0.5|lm_only|2e-5|200|": {
        "label": "kl 0.5, LM-only, 200 st",
        "intr": (3.79, 0.48, 1.54, 1.333),
        "down": (37.3, 56.4),
    },
    "kl|0.5|lm_only|2e-5|800|": {
        "label": "kl 0.5, LM-only, 800 st",
        "intr": (6.41, 0.57, 2.02, 1.229),
        "down": (40.9, 56.7, 52.8, 35.4, 62.0, 44.4, 83.4, 16.0, 36.5),
    },
    "kl|0.5|lm_only|2e-5|1600|": {
        "label": "kl 0.5, LM-only, 1600 st",
        "intr": (7.70, 0.60, 2.19, 1.182),
        "down": (41.1, 55.4, 51.6, 35.0, 59.0, 42.9, 84.4, 15.8, 39.1),
    },
    "kl|0.5|lm_only|2e-5|2400|": {
        "label": "kl 0.5, LM-only, 2400 st",
        "intr": (8.44, 0.62, 2.27, 1.159),
        "down": (40.7, 55.8, 51.8, 35.7, 60.7, 43.3, 83.5, 15.8, 38.7),
    },
    "default|0|lm_only|2e-5|200|": {
        "label": "control (lambda 0), 200 st",
        "intr": (1.42, 0.24, 0.40, 1.284),
        "down": (36.7, 56.2),
    },
    "default|0|lm_only|2e-5|800|": {
        "label": "control (lambda 0), 800 st",
        "intr": None,
        "down": None,
    },
    "default|0|lm_only|2e-5|1600|": {
        "label": "control (lambda 0), 1600 st",
        "intr": (1.48, 0.25, 0.45, 1.135),
        "down": (41.5, 56.9, 52.9, 35.4, 59.0, 46.5, 83.8, 17.0, 40.9),
    },
    "default|0|lm_only|2e-5|2400|": {
        "label": "control (lambda 0), 2400 st",
        "intr": None,
        "down": None,
    },
    "kl|0.05|lm_only|2e-5|200|": {
        "label": "kl 0.05, 200 st",
        "intr": (1.61, 0.28, 0.67, 1.285),
        "down": (36.3, 56.1),
    },
    "kl|0.1|lm_only|2e-5|200|": {
        "label": "kl 0.1, 200 st",
        "intr": (1.81, 0.31, 0.84, 1.288),
        "down": (36.7, 54.9),
    },
    "kl|0.25|lm_only|2e-5|200|": {
        "label": "kl 0.25, 200 st",
        "intr": (2.65, 0.41, 1.21, 1.301),
        "down": (37.7, 55.0),
    },
    "kl|1|lm_only|2e-5|200|": {
        "label": "kl 1, 200 st",
        "intr": (4.96, 0.53, 1.79, 1.385),
        "down": (36.0, 55.2),
    },
    "kl|5|lm_only|2e-5|200|": {
        "label": "kl 5, 200 st",
        "intr": (6.37, 0.57, 1.99, 1.597),
        "down": (30.5, 51.8),
    },
    "kl|0.5|proj_only|2e-5|800|": {
        "label": "kl 0.5, projector-only, 800 st",
        "intr": (1.51, 0.27, 0.72, 1.808),
        "down": (33.2, 49.8, 53.8, 31.9, 52.7, 40.5, 73.9, 16.5, 36.0),
    },
    "kl|0.5|lm_proj|2e-5|800|": {
        "label": "kl 0.5, LM+projector, 800 st",
        "intr": (6.54, 0.58, 2.04, 1.228),
        "down": (40.7, 56.3, 52.0, 35.6, 60.0, 44.8, 83.5, 15.7, 39.1),
    },
    "kl|0.5|lm_only|2e-4|800|4": {
        "label": "LoRA r=4, 800 st",
        "intr": (7.30, 0.59, 2.13, 1.243),
        "down": (17.1, 50.6, 53.3, 36.7, 58.0, 45.3, 85.0, 15.6, 34.6),
    },
    "kl|0.5|lm_only|2e-4|800|16": {
        "label": "LoRA r=16, 800 st",
        "intr": (8.32, 0.61, 2.23, 1.191),
        "down": (0.4, 51.9, 54.3, 36.4, 62.0, 43.4, 85.2, 16.4, 27.3),
    },
    "kl|0.5|lm_only|2e-4|800|128": {
        "label": "LoRA r=128, 800 st",
        "intr": (9.89, 0.64, 2.38, 1.113),
        "down": (40.1, 53.4, 52.3, 36.3, 60.7, 37.9, 84.8, 13.4, 26.9),
    },
}
# Published checkpoints evaluated with the corrected metric (trained with the
# old pipeline); compared against the paper row of the same configuration.
HUB_REFERENCE = {
    "teilers__llava-1.5-7b-saliency-kl0.5-st2400": "kl|0.5|lm_only|2e-5|2400|",
}
BASE_IDS = {"llava-hf__llava-1.5-7b-hf"}

DOWN_COLS = [
    "CountBench",
    "CV-2D",
    "CV-3D",
    "MMStar",
    "MMVP",
    "O3",
    "POPE",
    "VLMsB acc",
    "VLMsB bias",
]
# lmms-eval task name -> preferred metric names (first match wins).
DOWN_TASKS = [
    ("countbench", ("exact_match", "accuracy", "acc")),
    ("cv_bench_2d", ("accuracy", "acc", "exact_match")),
    ("cv_bench_3d", ("accuracy", "acc", "exact_match")),
    ("mmstar", ("average", "accuracy", "acc")),
    ("mmvp", ("accuracy", "acc", "mmvp_score", "exact_match")),
    ("o3", ("sample_f1", "f1", "accuracy", "acc")),
    ("pope", ("pope_f1_score", "f1", "pope_accuracy", "accuracy")),
    ("vlms_are_biased", ("accuracy", "acc")),
    ("vlms_are_biased", ("bias_ratio", "bias_rate", "bias")),
]


def key_of(row: dict) -> str | None:
    rid = row["run_id"]
    if rid in BASE_IDS:
        return "base"
    if rid in HUB_REFERENCE:
        return HUB_REFERENCE[rid]
    if not row.get("criterion"):
        return None
    lam = row.get("lambda") or ""
    lam = lam.rstrip("0").rstrip(".") if "." in lam else lam
    return "|".join(
        [
            row["criterion"],
            lam,
            row.get("freeze") or "",
            row.get("lr") or "",
            row.get("steps") or "",
            row.get("rank") or "",
        ]
    )


def fnum(v, nd=2) -> str:
    if v is None or v == "":
        return "–"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return str(v)


def pct(v) -> str:
    if v is None or v == "":
        return "–"
    x = float(v)
    return f"{x * 100:.1f}" if x <= 1.0 else f"{x:.1f}"


def downstream(row: dict) -> list[str | None]:
    out: list[str | None] = []
    for task, prefs in DOWN_TASKS:
        cols = {
            k: v
            for k, v in row.items()
            if k.startswith(f"down/{task}/") and v not in ("", None)
        }
        pick = next(
            (cols[f"down/{task}/{m}"] for m in prefs if f"down/{task}/{m}" in cols),
            None,
        )
        if pick is None and cols and task != "vlms_are_biased":
            pick = next(iter(cols.values()))
        out.append(pick)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "summary", type=Path, nargs="?", default=Path("results/summary.csv")
    )
    args = ap.parse_args()
    with args.summary.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("summary.csv is empty")
        return 1

    order = list(PAPER)
    rows.sort(
        key=lambda r: (
            order.index(key_of(r)) if key_of(r) in order else len(order),
            r["run_id"],
        )
    )

    print("# Corrected pipeline re-run vs. paper\n")
    print(
        "Values: corrected (paper). Paper values come from the old pipeline; `–` = not available.\n"
    )
    print("## Intrinsic localization (held-out COCONut split) and validation loss\n")
    print("| run | AMR | AP | NSS | val-CE | val-acc |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        k = key_of(r)
        ref = PAPER.get(k, {}).get("intr") if k else None
        ref = ref or (None, None, None, None)
        label = PAPER.get(k, {}).get("label", r["run_id"]) if k else r["run_id"]
        if r["run_id"] in HUB_REFERENCE:
            label = f"published 2400-st checkpoint, corrected metric ({label})"
        cells = [
            f"{fnum(r.get('AMR_mean'))} ({fnum(ref[0])})",
            f"{fnum(r.get('AP_mean'))} ({fnum(ref[1])})",
            f"{fnum(r.get('NSS_mean'))} ({fnum(ref[2])})",
            f"{fnum(r.get('val_ce_loss'), 3)} ({fnum(ref[3], 3)})",
            fnum(r.get("val_accuracy"), 3),
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n## Downstream benchmarks (%), higher is better except VLMsB bias\n")
    print("| run | " + " | ".join(DOWN_COLS) + " |")
    print("|---|" + "---|" * len(DOWN_COLS))
    for r in rows:
        vals = downstream(r)
        if all(v is None for v in vals):
            continue
        k = key_of(r)
        ref = PAPER.get(k, {}).get("down") if k else None
        ref = list(ref or []) + [None] * (len(DOWN_COLS) - len(ref or []))
        label = PAPER.get(k, {}).get("label", r["run_id"]) if k else r["run_id"]
        cells = [f"{pct(v)} ({fnum(p, 1)})" for v, p in zip(vals, ref, strict=True)]
        print(f"| {label} | " + " | ".join(cells) + " |")

    missing = [
        k for k in PAPER if k != "base" and not any(key_of(r) == k for r in rows)
    ]
    if missing:
        print(
            "\nRuns of the paper not (yet) in this report: "
            + ", ".join(PAPER[k]["label"] for k in missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
