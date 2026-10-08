# Saliency Alignment: working notes for Claude Code

Fine-tuning LLaVA-1.5-7B with an auxiliary KL loss that aligns per-token
text-to-image attention with COCONut-PanCap panoptic masks, plus the
intrinsic (AMR/AP/NSS) and downstream (lmms-eval) evaluations of the paper.
See @README.md for the repository layout and data format.

## Where things are

- `PROJECT_DIR` is the clone you work in; every path below is relative to it.
  On CSCS it is `$SCRATCH/saliency-alignment`.
- `data/coco/` COCO images and COCONut masks/captions, produced by
  `sbatch scripts/cscs/data.sh` (see the data step below).
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
predate these changes. The previous clone on scratch, with its data,
container images and checkpoints, was purged, so everything starts from the
bare clone at `$SCRATCH/saliency-alignment`. The paper's 2,400-step model
survives on the Hub as `teilers/llava-1.5-7b-saliency-kl0.5-st2400` and is
the reference for "old model under the corrected metric".

Work through the steps in order; each has a check. Do not start a step
before the previous check passes.

1. **Container images.** `sbatch scripts/cscs/build_image.sbatch` and
   `sbatch scripts/cscs/build_eval_image.sbatch` (podman build + enroot
   import, up to an hour each). Check: `ls -la slurm/saliency.sqsh
   slurm/eval.sqsh`. Then confirm `~/.edf/saliency.toml` and
   `~/.edf/saliency_eval.toml` exist and their `image =` lines point at
   those two files (templates in `slurm/`; replace `<YOUR_USERNAME>`).
2. **Venv.** `sbatch scripts/cscs/arr_setup.sh`. Check: the job log ends
   with `venv ready` and `.venv/bin/python -c "import transformers, vl_saliency"`
   works inside `srun --environment=saliency`.
3. **Data.** `sbatch scripts/cscs/data.sh` downloads COCO 2017 images and
   annotations, the COCONut masks (Hub dataset `xdeng77/coconut_s`), the
   grounded captions, and converts the masks to `.npy`. The job has a
   2-hour limit and skips parts that already exist, so resubmit it until
   all checks pass: 118,287 files in `data/coco/images/train2017`, 5,000 in
   `data/coco/images/val2017`, `data/coco/annotations/panoptic_train2017.json`
   present, the same number of `.png` and `.npy` files in
   `data/coco/panoptic_train2017_masks`, and `.txt` files in
   `data/coco/panoptic_train2017_captions`. An extraction that the time
   limit interrupted leaves a partial directory that the downloader then
   skips: if a count is short, remove only that incomplete directory and
   resubmit. This is the one exception to the no-delete rule under `data/`.
4. **Credentials.** Two secrets are needed and both come from the
   environment of the shell that runs `sbatch` (sbatch forwards it to the
   job and into the container): `HF_TOKEN` for the Hub downloads in the
   data step and the model pulls, and `WANDB_API_KEY` for training logs
   (or `WANDB_MODE=offline`). Check with `test -n "$HF_TOKEN" && echo set`;
   never print a token, never write one into the repository or CLAUDE.md,
   and never pass one on a command line.
5. **Smoke test** on one GPU before spending real compute:
   `PROJECT_DIR=$PWD srun --account=aa013 --time=00:20:00 --gpus=1 --environment=saliency .venv/bin/python scripts/python/check_alignment.py run_id=check_alignment`
   prints, per supervised token, its decoded text, mask size and AMR. The
   decoded tokens must be the words inside the annotated spans (e.g. `▁dog`
   paired with a dog-sized mask), not the words after them, and no error
   from the collator's consistency checks may appear. Show the output to
   the user.
6. **Pilot.** `STAGE=pilot sbatch scripts/cscs/experiments/rerun_corrected.sh`
   submits the corrected intrinsic evaluation of the base model and of the
   published 2,400-step checkpoint, and one corrected training run at the
   paper's operating point (kl, lambda 0.5, LM-only, lr 2e-5, 800 steps)
   followed by both evaluations. Watch the jobs; when they are done run
   `python scripts/python/aggregate_results.py --out results/summary.csv`
   and `python scripts/python/make_report.py results/summary.csv`.
7. **Compare** with the paper (old pipeline) and stop for the user's
   decision before continuing:

   | model | AMR | AP | NSS | val-CE |
   |---|---|---|---|---|
   | base | 1.25 | 0.22 | 0.26 | – |
   | published 2400-step checkpoint (paper row) | 8.44 | 0.62 | 2.27 | 1.159 |
   | kl 0.5, LM-only, 800 steps (to be retrained) | 6.41 | 0.57 | 2.02 | 1.229 |

   Downstream row of the paper's 800-step model (CountBench, CV-2D, CV-3D,
   MMStar, MMVP, O3, POPE, VLMsB acc, VLMsB bias): 40.9 56.7 52.8 35.4 62.0
   44.4 83.4 16.0 36.5. Report the corrected numbers next to these; do not
   edit the paper's tables yourself.
8. **Lambda sweep first** (user's choice, cheap):
   `STAGE=lambda sbatch scripts/cscs/experiments/rerun_corrected.sh` trains
   the seven 200-step weight-sweep runs (lambda 0 to 5) with both
   evaluations. When their align-evals are in, report AMR/AP/NSS and
   val-CE per lambda; the operating point stays 0.5 unless the user
   decides otherwise from that table (downstream is not used for the
   choice). Wait for the user's go before the full stage.
9. **Full sweep** only after the user says so:
   `STAGE=full sbatch scripts/cscs/experiments/rerun_corrected.sh` submits
   the remaining 17 training runs with both evaluations plus the base
   model's downstream suite. Finished runs are skipped, so resubmit the
   same command after failures. Expect a day or more of wall time; check
   back periodically rather than polling every minute.
10. **Seeds and per-sample outputs** (after the full sweep, on the user's
   go): `STAGE=seeds sbatch scripts/cscs/experiments/rerun_corrected.sh`
   trains seeds 43 and 44 of the 800-step aligned and control runs with
   both evaluations; `STAGE=samples sbatch scripts/cscs/experiments/rerun_corrected.sh`
   re-runs lm-eval with `--log_samples` for the base model and the eight
   length-series runs into `results/lm-eval/<run>_samples/`, leaving the
   original results untouched. Both stages are idempotent.
11. **Final report.** When every job has finished or failed for good, run
   `python scripts/python/aggregate_results.py --out results/summary.csv`
   and `python scripts/python/make_report.py results/summary.csv > results/report.md`,
   then give the user the full content of `results/report.md` verbatim,
   followed by a list of runs that are missing or failed and why. That
   text is pasted into the paper discussion, so keep it complete and
   unedited.

## Status updates and accounting

- Every message to the user ends with a short status block: jobs running /
  pending / finished / failed (from `squeue --me` and `sacct`), the current
  step of the runbook, the next thing you will check and when, and the
  compute used so far from `scripts/cscs/gpu_hours.sh <start date>`
  (GPU-hours and node-hours since the re-run began; CSCS bills node-hours,
  one GH200 node is 4 GPUs). Keep the block under ten lines.
- A `Stop` hook (`scripts/cscs/notify.py`, configured in
  `.claude/settings.json`) forwards each of your messages to the user's
  phone when `CLAUDE_NOTIFY_URL` is set, and a `Notification` hook tells
  them when you are waiting for input. Write status messages so they make
  sense on a phone screen: outcome first, numbers in a short table, no
  long log dumps (quote at most the decisive lines of a log).
- Report compute in the final report as well: total GPU-hours and
  node-hours, split by training, align-eval, and lm-eval job names.

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
