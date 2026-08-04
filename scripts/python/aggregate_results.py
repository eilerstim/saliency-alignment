"""Collect sweep results into one CSV for the paper's tables/figures.

Per run_id, joins intrinsic localization
(``outputs/<run_id>/alignment_summary.json``), validation metrics
(``outputs/<run_id>/logs/training_logs/version_*/metrics.csv``) and downstream
lmms-eval (``results/lm-eval/<run_id>/**/*results*.json``). Swept axes are
recovered from the run_id. Stdlib only.

    python scripts/python/aggregate_results.py --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _m(pattern: str, s: str) -> str | None:
    match = re.search(pattern, s)
    return match.group(1) if match else None


def parse_run_id(rid: str) -> dict:
    freeze = next((f for f in ("lm_proj", "proj_only", "lm_only") if f"_{f}" in rid), None)
    return {
        "run_id": rid,
        "criterion": _m(r"_(kl|alignment|default)_", rid),
        "lambda": _m(r"_w([0-9]*\.?[0-9]+)", rid),
        "freeze": freeze,
        "lr": _m(r"_lr([0-9.e-]+?)_", rid),
        "steps": _m(r"_st([0-9]+)", rid),
        "seed": _m(r"_seed([0-9]+)", rid),
        "rank": _m(r"_lora_r([0-9]+)", rid),
        "method": "lora" if "_lora_r" in rid else "full",
    }


def read_intrinsic(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "alignment_summary.json").read_text())["summary"]
    return {
        f"{name}_{stat}": summary[name][stat]
        for name in ("AMR", "AP", "NSS")
        for stat in ("mean", "median", "std")
    }


def read_val(run_dir: Path) -> dict:
    files = sorted(run_dir.glob("logs/training_logs/version_*/metrics.csv"))
    out: dict = {}
    if not files:
        return out
    with files[-1].open() as f:
        for row in csv.DictReader(f):  # keep last non-empty value per key
            for k in ("val/ce_loss", "val/accuracy"):
                if row.get(k):
                    out[k.replace("/", "_")] = float(row[k])
    return out


def read_downstream(rid: str, root: Path) -> dict:
    if not root.exists():
        return {}
    frags = (f"/{rid}/", f"/{rid.replace('__', '/')}/")
    hit = next(
        (p for p in sorted(root.rglob("*results*.json")) if any(fr in f"{p}/" for fr in frags)),
        None,
    )
    if hit is None:
        return {}
    results = json.loads(hit.read_text()).get("results", {})
    return {
        f"down/{task}/{k.split(',')[0]}": v
        for task, metrics in results.items()
        if isinstance(metrics, dict)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--lm-eval-dir", type=Path, default=Path("results/lm-eval"))
    ap.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.outputs_dir.glob("*") if p.is_dir()):
        if not (run_dir / "alignment_summary.json").exists():
            continue
        row = parse_run_id(run_dir.name)
        row.update(read_intrinsic(run_dir))
        row.update(read_val(run_dir))
        row.update(read_downstream(run_dir.name, args.lm_eval_dir))
        rows.append(row)
    if not rows:
        print(f"No runs with alignment_summary.json under {args.outputs_dir}")
        return 1

    cols: list[str] = []
    for row in rows:
        cols += [k for k in row if k not in cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs x {len(cols)} cols to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
