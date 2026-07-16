#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_models
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-5
# Cross-architecture test at the operating point (KL, lm_only, 1600 steps), each
# model paired with its lambda=0 (CE-only) control. Each model's freeze list is
# read from its own configs/model/<cfg>.yaml (selected via MODEL_CFG), so the
# correct submodules are frozen per architecture:
#   llava-1.5-13b / gemma-3-4b -> freeze [vision_tower, multi_modal_projector]
#   qwen2.5-vl-7b              -> freeze [visual]   (no separate projector)
# LR is matched across models (isolates architecture; Qwen may want retuning).
# Submits training + alignment eval (AMR/AP/NSS). Downstream lmms-eval is left as
# a manual follow-up -- it needs per-architecture vLLM/tokenizer validation (run
# arr_eval.sh with TOKENIZER=models/<run_id> once confirmed).
# LLaVA-13B trains on 2 nodes (8 GPUs); the 7B/4B models on 1 node (4 GPUs).
# Layout (6 tasks = 3 models x {kl@0.5, default@0}):
#   0,1 = llava-1.5-13b   2,3 = qwen2.5-vl-7b   4,5 = gemma-3-4b
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

LR=${LR:-2e-5}          # matched recipe across models
STEPS=${STEPS:-1600}    # operating point for this test
SEED=${SEED:-42}

CFGS=(llava-1.5-13b qwen2.5-vl-7b gemma-3-4b)
NODES=(2            1             1)   # 13B needs more GPUs
CRITS=(kl default)
LAMS=(0.5 0)

M=$(( SLURM_ARRAY_TASK_ID / 2 ))
L=$(( SLURM_ARRAY_TASK_ID % 2 ))

CFG=${CFGS[$M]}
NNODES=${NODES[$M]}
CRIT=${CRITS[$L]}
LAM=${LAMS[$L]}

RUN_ID="${CFG}_${CRIT}_w${LAM}_lm_only_lr${LR}_st${STEPS}_seed${SEED}"
OVERRIDES="optim.lr=${LR} trainer.max_steps=${STEPS} seed=${SEED}"

# MODEL_CFG selects the config group (and freeze list); propagated to the job via
# sbatch's default --export=ALL. MODEL_SIZE positional is unused when MODEL_CFG set.
TRAIN_JOBID=$(MODEL_CFG=${CFG} sbatch --parsable --nodes=${NNODES} \
    scripts/cscs/arr_train.sh "$RUN_ID" "$CRIT" "$LAM" na "$OVERRIDES")

sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} (${NNODES} node(s)) → ALIGN-EVAL for ${RUN_ID}"
