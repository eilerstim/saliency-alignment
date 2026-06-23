"""Collect sweep results into one tidy CSV for the paper's tables/figures.

Joins three per-run artifacts, keyed by run_id:
  * intrinsic localization  -> outputs/<run_id>/alignment_summary.json
  * language-modeling        -> outputs/<run_id>/logs/training_logs/version_*/metrics.csv
  * downstream benchmarks    -> results/lm-eval/<run_id>/**/*results*.json (lmms-eval)

The swept axes (criterion, lambda, freeze, lr, steps, seed, method, rank) are
recovered from the run_id, whose layout the sweep drivers fix:
  llava-1.5-7b_<crit>_w<lambda>_<freeze>_lr<lr>_st<steps>_seed<seed>[_lora_r<rank>]

Stdlib only. Usage:
    python scripts/python/aggregate_results.py --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CRITERIA = ("kl", "alignment", "default")
FREEZES = ("lm_proj", "proj_only", "lm_only")  # check lm_proj before lm_only
INTRINSIC = ("AMR", "AP", "NSS")
VAL_KEYS = ("val/ce_loss", "val/accuracy", "val/auxiliary_loss", "val/loss")


def parse_run_id(run_id: str) -> dict:
    """Recover swept-axis fields from a run_id (missing fields stay None)."""
    out: dict = {
        "run_id": run_id,
        "criterion": None,
        "lambda": None,
        "freeze": None,
        "lr": None,
        "steps": None,
        "seed": None,
        "method": "full",
        "rank": None,
    }
    if m := re.search(r"_(kl|alignment|default)_", run_id):
        out["criterion"] = m.group(1)
    if m := re.search(r"_w([0-9]*\.?[0-9]+)", run_id):
        out["lambda"] = float(m.group(1))
    for fr in FREEZES:
        if f"_{fr}" in run_id:
            out["freeze"] = fr
            break
    if m := re.search(r"_lr([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)", run_id):
        out["lr"] = m.group(1)
    if m := re.search(r"_st([0-9]+)", run_id):
        out["steps"] = int(m.group(1))
    if m := re.search(r"_seed([0-9]+)", run_id):
        out["seed"] = int(m.group(1))
    if m := re.search(r"_lora_r([0-9]+)", run_id):
        out["method"] = "lora"
        out["rank"] = int(m.group(1))
    return out


def read_intrinsic(run_dir: Path) -> dict:
    """Read AMR/AP/NSS mean/median/std/n from an align_eval summary."""
    path = run_dir / "alignment_summary.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text()).get("summary", {})
    row: dict = {}
    for metric in INTRINSIC:
        s = summary.get(metric, {})
        row[f"{metric}_mean"] = s.get("mean")
        row[f"{metric}_median"] = s.get("median")
        row[f"{metric}_std"] = s.get("std")
        row[f"{metric}_n"] = s.get("n_images")
    return row


def read_val_metrics(run_dir: Path) -> dict:
    """Read the last logged validation metrics from the CSV logger output."""
    candidates = sorted(run_dir.glob("logs/training_logs/version_*/metrics.csv"))
    if not candidates:
        return {}
    row: dict = {}
    with candidates[-1].open() as f:
        for rec in csv.DictReader(f):  # keep last non-empty value per key
            for k in VAL_KEYS:
                v = rec.get(k, "")
                if v not in ("", None):
                    row[k.replace("/", "_")] = float(v)
    return row


def read_downstream(run_id: str, lm_eval_dir: Path) -> dict:
    """Flatten lmms-eval task metrics for this run, if a results JSON exists."""
    if not lm_eval_dir.exists():
        return {}
    fragments = [f"/{run_id}/", f"/{run_id.replace('__', '/')}/"]
    hit = next(
        (
            p
            for p in sorted(lm_eval_dir.rglob("*results*.json"))
            if any(fr in f"{p}/" for fr in fragments)
        ),
        None,
    )
    if hit is None:
        return {}
    results = json.loads(hit.read_text()).get("results", {})
    row: dict = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, int | float):
                continue
            metric = k.split(",")[0]  # strip ",none" / filter suffix
            row[f"down/{task}/{metric}"] = v
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--lm-eval-dir", type=Path, default=Path("results/lm-eval"))
    ap.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.outputs_dir.glob("*") if p.is_dir()):
        if not (run_dir / "alignment_summary.json").exists():
            continue  # only runs we have intrinsic metrics for anchor a row
        run_id = run_dir.name
        row = parse_run_id(run_id)
        row.update(read_intrinsic(run_dir))
        row.update(read_val_metrics(run_dir))
        row.update(read_downstream(run_id, args.lm_eval_dir))
        rows.append(row)

    if not rows:
        print(f"No runs with alignment_summary.json under {args.outputs_dir}")
        return 1

    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs x {len(fieldnames)} columns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
