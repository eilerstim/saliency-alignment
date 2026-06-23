# Paper sweeps

Three orchestrated sweeps over the **direct fine-tuning** regime
(LLaVA-1.5-7B on grounded COCONut captions), sharing one canonical recipe.
Run the drivers from the repo root on a login node; each loops over its grid
and submits `train -> {align-eval, lmms-eval}` with `afterok` dependencies.

## Canonical recipe

AdamW `lr=2e-5`, bf16-mixed, gradient checkpointing, **3% warmup + cosine
decay derived from `trainer.max_steps`** (`configs/scheduler/cosine.yaml`,
now in the defaults and stepped per optimizer step), FSDP across 4 GPUs,
effective batch 128, 200 steps. LM-only is the canonical trainable surface
(vision tower + projector frozen).

## Run-id convention (parsed by `aggregate_results.py`)

```
llava-1.5-7b_<crit>_w<lambda>_<freeze>_lr<lr>_st<steps>_seed<seed>[_lora_r<rank>]
```

`crit ∈ {kl, alignment, default}` (`default` = ZeroCriterion, the λ=0 control);
`freeze ∈ {lm_only, proj_only, lm_proj}`; `_lora_r<rank>` marks a LoRA run.

## Drivers

| Script | Experiment | Key env vars |
|---|---|---|
| `sweep_lambda.sh`     | A — λ dose-response (KL grid + partial MSE) | `SEEDS`, `KL_LAMBDAS`, `MSE_LAMBDAS`, `DOWNSTREAM_LAMBDAS` |
| `sweep_lr_length.sh`  | B — LR & length sensitivity at the knee λ   | `KNEE_LAMBDA`, `SEED`, `LENGTHS`, `LRS` |
| `sweep_lora.sh`       | C — LoRA vs full FT + rank sweep            | `KNEE_LAMBDA`, `SEEDS`, `RANKS`, `LORA_LR`, `LORA_LRS` |

Set `KNEE_LAMBDA` for B/C to Experiment A's knee before running them.

## Collect & plot

```bash
python scripts/python/aggregate_results.py --out results/summary.csv
python scripts/python/plot_sweeps.py results/summary.csv --kind all   # needs matplotlib
```

`aggregate_results.py` joins, per run_id: intrinsic localization
(`outputs/<run_id>/alignment_summary.json`), validation CE / accuracy
(`outputs/<run_id>/logs/training_logs/.../metrics.csv`), and downstream
lmms-eval metrics (`results/lm-eval/<run_id>/**/*results*.json`).
