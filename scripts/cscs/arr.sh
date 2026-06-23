#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=submit_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-6%4
# Array layout (kl x single lambda x freeze, plus default-per-freeze, plus baseline):
#   0..2 = kl x lambda=0.5 x {3 freezes}
#   3..5 = default x {3 freezes}
#   6    = baseline eval only

set -euo pipefail
mkdir -p logs

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY=${EVAL_ONLY:-false}

MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

CRITERION="kl"
LAMBDAS=(0.5)
FREEZE_NAMES=("lm_only" "proj_only" "lm_proj")
FREEZE_OVERRIDES=(
    "model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
    "model.freeze=[all] model.unfreeze=[multi_modal_projector]"
    "model.freeze=[vision_tower] model.unfreeze=[]"
)

NUM_LAMBDAS=${#LAMBDAS[@]}
NUM_FREEZES=${#FREEZE_NAMES[@]}
NUM_KL_TASKS=$((NUM_LAMBDAS * NUM_FREEZES))
DEFAULT_START_ID=${NUM_KL_TASKS}
BASELINE_ID=$((NUM_KL_TASKS + NUM_FREEZES))

# ---- BASELINE: eval only ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${BASELINE_ID}" ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true"
    # sbatch scripts/cscs/count/eval.sh "${BASE_MODEL}" "true"
    echo "Submitted EVAL only for baseline model ${BASE_MODEL} at $(date)"
    exit 0
fi

# ---- Resolve criterion / lambda / freeze ----
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${DEFAULT_START_ID}" ]; then
    CRITERION="default"
    LAMBDA=0.0
    FREEZE_IDX=$((SLURM_ARRAY_TASK_ID - DEFAULT_START_ID))
else
    LAMBDA_IDX=$((SLURM_ARRAY_TASK_ID / NUM_FREEZES))
    FREEZE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_FREEZES))
    LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
fi

FREEZE_NAME=${FREEZE_NAMES[$FREEZE_IDX]}
FREEZE_OVERRIDE=${FREEZE_OVERRIDES[$FREEZE_IDX]}

RUN_ID="llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_${FREEZE_NAME}"

echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Check if only evaluation is requested ----
if [ "${EVAL_ONLY}" = "true" ]; then
    sbatch scripts/cscs/arr_eval.sh "$RUN_ID"
    # sbatch scripts/cscs/count/eval.sh "$RUN_ID" "false"
    echo "Submitted EVAL only for ${RUN_ID}"
    exit 0
fi

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE" "$FREEZE_OVERRIDE")

# ---- Submit evaluation job dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "$RUN_ID"

# sbatch --dependency=afterok:${TRAIN_JOBID} \
#     scripts/cscs/count/eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"
