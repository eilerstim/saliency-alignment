# Saliency Alignment: working notes for Claude Code

Fine-tuning LLaVA-1.5-7B with an auxiliary KL loss that aligns per-token
text-to-image attention with COCONut-PanCap panoptic masks, plus the
intrinsic (AMR/AP/NSS) and downstream (lmms-eval) evaluations of the paper.
See @README.md for the repository layout and data format.

## Where things are

- `PROJECT_DIR` is the clone you work in; every path below is relative to it.
  On CSCS it lives on scratch, e.g. `$SCRATCH/saliency-alignment-v2`.
- `data/coco/` COCO images and COCONut masks/captions (symlink from an
  existing clone instead of downloading again).
- `models/<RUN_ID>/` HF checkpoints written by training (LoRA runs also get
  `models/<RUN_ID>-merged/`).
- `outputs/<RUN_ID>/alignment_summary.json` intrinsic metrics from
  `align_eval`; `outputs/<RUN_ID>/alignment_per_image.pt` per-image scores.
- `results/lm-eval/<RUN_ID>/` downstream benchmark results.
- `logs/<jobname>_<jobid>[_<arrayid>].{out,err}` SLURM logs. Read the `.err`
  file first when a job fails.
- `scripts/python/aggregate_results.py` collects everything into a CSV.

## Cluster facts (CSCS Alps, Clariden, SLURM account aa013)

- Everything that needs a GPU or more than a few seconds of CPU runs through
  `sbatch`. Never run training, evaluation, or data preparation on the login
  node.
- Jobs run inside container environments: `--environment=saliency`
  (training, align_eval; built from `slurm/Dockerfile`) and
  `--environment=saliency_eval` (lmms-eval on vLLM; `slurm/Dockerfile.eval`).
  The EDF files `~/.edf/saliency.toml` and `~/.edf/saliency_eval.toml` point
  at the `.sqsh` images; templates are in `slurm/`. Rebuild images with
  `sbatch scripts/cscs/build_image.sbatch` / `build_eval_image.sbatch`.
- The Python venv is `.venv` inside the clone, created once inside the
  container by `sbatch scripts/cscs/arr_setup.sh` (`RESET_ENV=1` rebuilds it).
- `scripts/cscs/arr_train.sh RUN_ID CRITERION LAMBDA MODEL_SIZE "OVERRIDES"`
  trains; it skips runs whose checkpoint already exists (`FORCE_RETRAIN=1`
  overrides). `scripts/cscs/arr_align_eval.sh` and `scripts/cscs/arr_eval.sh`
  evaluate; launchers chain them with `--dependency=afterok`.
- Monitor with `squeue --me`, `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode`,
  and `tail -n 50 logs/<file>`. A finished training log ends with
  `Finished finetuning of <RUN_ID>`; a finished align_eval log contains the
  `=== Attention alignment metrics ===` table.

## Run naming

`llava-1.5-7b_<crit>_w<lambda>_<freeze>_lr<lr>_st<steps>_seed<seed>[_lora_r<r>]`
with `crit` in `{kl, default}` (`default` is the lambda=0 control, use
`w0`), `freeze` in `{lm_only, proj_only, lm_proj}`. Keep this scheme so
`aggregate_results.py` can parse the runs.

## Corrected pipeline re-run (current task)

The collator was fixed in commit d23fe50 (annotations were assigned to the
wrong caption tokens; masks were not cropped to the image processor's center
crop) and hardened afterwards; the training sequence now also ends with
`</s>` as a supervised token instead of a bare space, as in the original
LLaVA-1.5 recipe. All intrinsic numbers and all trained models of the paper
predate these changes. The re-run is driven by
`scripts/cscs/experiments/rerun_corrected.sh`:

1. Setup in a fresh clone (never reuse the old `models/`):
   `git clone -b claude/wizardly-pasteur-cg48cb <repo> $SCRATCH/saliency-alignment-v2`,
   `ln -s <old clone>/data data`, `sbatch scripts/cscs/arr_setup.sh`.
2. Smoke test on one GPU before spending compute:
   `PROJECT_DIR=$PWD srun --account=aa013 --time=00:20:00 --gpus=1 --environment=saliency .venv/bin/python scripts/python/check_alignment.py run_id=check_alignment`
   prints, per supervised token, its decoded text, mask size and AMR. The
   decoded tokens must be the words inside the annotated spans (e.g. `▁dog`
   paired with a dog-sized mask), not the words after them.
3. Pilot: `STAGE=pilot OLD_PROJECT_DIR=<old clone> sbatch scripts/cscs/experiments/rerun_corrected.sh`.
   It evaluates the base model and the original 800-step checkpoint under the
   corrected metrics and trains one corrected 800-step run with both
   evaluations.
4. Compare the pilot with the paper (old pipeline) before continuing:

   | model | AMR | AP | NSS | val-CE |
   |---|---|---|---|---|
   | base | 1.25 | 0.22 | 0.26 | – |
   | kl 0.5, LM-only, 800 steps | 6.41 | 0.57 | 2.02 | 1.229 |

   Downstream row of that 800-step model (CountBench, CV-2D, CV-3D, MMStar,
   MMVP, O3, POPE, VLMsB acc, VLMsB bias): 40.9 56.7 52.8 35.4 62.0 44.4 83.4
   16.0 36.5. Report the corrected numbers next to these; do not overwrite
   the paper's tables yourself.
5. Full sweep: `STAGE=full sbatch scripts/cscs/experiments/rerun_corrected.sh`
   (18 training runs; finished runs are skipped, so it can be resubmitted
   after failures). Then `python scripts/python/aggregate_results.py`.

## Rules

- Do not delete or overwrite anything under `data/`, `models/`, `results/`,
  or `outputs/`; do not `scancel` jobs you did not submit.
- Do not change training hyperparameters, configs, or the run naming to make
  a run "work"; report the failure instead.
- When a job fails, read its `.err` log, diagnose, and fix the cause; resubmit
  the same launcher (it is idempotent).
- Keep the login-node footprint small: no Python heavier than
  `aggregate_results.py`, no downloads outside `sbatch scripts/cscs/data.sh`.
- Commit code changes on the working branch with clear messages; never
  commit data, checkpoints, logs, or results.
