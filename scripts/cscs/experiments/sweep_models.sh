#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_models
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# NOTE: --array is applied by the bootstrap below (after the venv setup job),
# not as a header directive, so the initial submission runs the bootstrap once.
# Cross-architecture test at the operating point (KL, lm_only, 1600 steps), each
# model paired with its lambda=0 (CE-only) control. Each model's freeze list is
# read from its own configs/model/<cfg>.yaml (selected via MODEL_CFG), so the
# correct submodules are frozen per architecture:
#   llava-1.5-13b / gemma-3-4b -> freeze [vision_tower, multi_modal_projector]
#   qwen2.5-vl-7b              -> freeze [visual]   (no separate projector)
# LR is matched across models (isolates architecture; Qwen may want retuning).
# Submits training + alignment eval (AMR/AP/NSS) + downstream lmms-eval. The vLLM
# tokenizer is set per model: LLaVA uses the hub tokenizer (its saved
# tokenizer_class is patched for vLLM), Qwen/Gemma use their self-contained
# checkpoint. Needs a saliency_eval vLLM new enough to support these arches.
# LLaVA-13B trains on 2 nodes (8 GPUs); the 7B/4B models on 1 node (4 GPUs).
# Layout (6 tasks = 3 models x {kl@0.5, default@0}):
#   0,1 = llava-1.5-13b   2,3 = qwen2.5-vl-7b   4,5 = gemma-3-4b
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Bootstrap: build the venv once, then fan out (avoids a 6-way race on
# $PROJECT_DIR/.venv). The initial submission has no array context, so submit the
# one-shot setup job and resubmit this script as an array that waits for it. Once
# setup completes the venv exists on disk, so each training job's env.sh just
# activates it -- no concurrent creation.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SETUP_JOBID=$(sbatch --parsable scripts/cscs/arr_setup.sh)
    sbatch --dependency=afterok:${SETUP_JOBID} --array=0-5 \
        scripts/cscs/experiments/sweep_models.sh
    echo "Submitted SETUP=${SETUP_JOBID} → sweep_models array (afterok:${SETUP_JOBID})"
    exit 0
fi

LR=${LR:-2e-5}          # matched recipe across models
STEPS=${STEPS:-1600}    # operating point for this test
SEED=${SEED:-42}

CFGS=(llava-1.5-13b qwen2.5-vl-7b gemma-3-4b)
NODES=(2            1             1)   # 13B needs more GPUs
# vLLM tokenizer per model for downstream eval (__checkpoint__ -> the run's own).
TOKENIZERS=(llava-hf/llava-1.5-13b-hf __checkpoint__ __checkpoint__)
CRITS=(kl default)
LAMS=(0.5 0)

M=$(( SLURM_ARRAY_TASK_ID / 2 ))
L=$(( SLURM_ARRAY_TASK_ID % 2 ))

CFG=${CFGS[$M]}
NNODES=${NODES[$M]}
CRIT=${CRITS[$L]}
LAM=${LAMS[$L]}

RUN_ID="${CFG}_${CRIT}_w${LAM}_lm_only_lr${LR}_st${STEPS}_seed${SEED}"
# trainer.num_nodes must match the sbatch --nodes below, or ranks on extra nodes
# fall outside Lightning's computed world size (devices x num_nodes).
OVERRIDES="optim.lr=${LR} trainer.max_steps=${STEPS} trainer.num_nodes=${NNODES} seed=${SEED}"

TOK=${TOKENIZERS[$M]}
[ "$TOK" = "__checkpoint__" ] && TOK="models/${RUN_ID}"

# MODEL_CFG selects the config group (and freeze list); propagated to the job via
# sbatch's default --export=ALL. MODEL_SIZE positional is unused when MODEL_CFG set.
TRAIN_JOBID=$(MODEL_CFG=${CFG} sbatch --parsable --nodes=${NNODES} \
    scripts/cscs/arr_train.sh "$RUN_ID" "$CRIT" "$LAM" na "$OVERRIDES")

sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
TOKENIZER="$TOK" sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_eval.sh "$RUN_ID"

echo "Submitted TRAIN=${TRAIN_JOBID} (${NNODES} node(s)) → ALIGN-EVAL + LM-EVAL for ${RUN_ID}"
