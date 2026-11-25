"""Paired aligned-vs-control comparison on POPE and CountBench from lmms-eval
``--log_samples`` files (``results/lm-eval/<run>_samples/``, written by
``STAGE=samples``). Items are matched by doc_id; the McNemar exact test is the
two-sided binomial test on the discordant pairs. Stdlib only.

    python scripts/python/paired_tests.py > results/paired_tests.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics

TASKS = [("countbench", "CountBench"), ("pope", "POPE")]


def load(root: str, run: str, task: str) -> dict[int, bool] | None:
    files = sorted(glob.glob(f"{root}/{run}_samples/**/*samples_{task}.jsonl", recursive=True))
    if not files:
        return None
    out: dict[int, bool] = {}
    with open(files[-1]) as f:
        for line in f:
            d = json.loads(line)
            ok = float(d["acc"]) == 1.0 if task == "countbench" else float(d["pope_accuracy"]["score"]) == 1.0
            if d["doc_id"] in out:
                raise ValueError(f"duplicate doc_id {d['doc_id']} in {files[-1]}")
            out[d["doc_id"]] = ok
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lm-eval-dir", default="results/lm-eval")
    ap.add_argument("--steps", nargs="+", type=int, default=[200, 800, 1600, 2400])
    ap.add_argument("--seed", default="42")
    args = ap.parse_args()

    print("Paired aligned (kl 0.5) vs control (lambda 0), LM-only, lr 2e-5, matched by item. "
          "POPE is per-item accuracy. McNemar exact = two-sided binomial test on the discordant pairs.\n")
    print("| benchmark | steps | n items | aligned acc | control acc | disagree | favor aligned | favor control | McNemar exact p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for task, name in TASKS:
        for st in args.steps:
            a = load(args.lm_eval_dir, f"llava-1.5-7b_kl_w0.5_lm_only_lr2e-5_st{st}_seed{args.seed}", task)
            c = load(args.lm_eval_dir, f"llava-1.5-7b_default_w0_lm_only_lr2e-5_st{st}_seed{args.seed}", task)
            if a is None or c is None:
                print(f"| {name} | {st} | missing {'aligned' if a is None else ''} {'control' if c is None else ''} | | | | | | |")
                continue
            keys = sorted(set(a) & set(c))
            b = sum(1 for k in keys if a[k] and not c[k])
            cc = sum(1 for k in keys if c[k] and not a[k])
            print(f"| {name} | {st} | {len(keys)} | {100 * statistics.fmean(a[k] for k in keys):.1f} | "
                  f"{100 * statistics.fmean(c[k] for k in keys):.1f} | {b + cc} | {b} | {cc} | {mcnemar_exact(b, cc):.3g} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
