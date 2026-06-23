#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lambda
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-33
# Experiment A: loss-weight (lambda) dose-response. LM-only, canonical recipe.
# Layout (matches arr.sh style): 11 (criterion,lambda) combos x 3 seeds = 33
# training tasks (0..32), plus baseline eval-only at task 33.
#   combos: default@0 (ZeroCriterion control), kl@{0.05,0.1,0.25,0.5,1,2,5},
#           alignment(MSE)@{0.1,0.5,2}
# Submit:  sbatch scripts/cscs/experiments/sweep_lambda.sh

set -euo pipefail
mkdir -p logs

export EVAL_ONLY=${EVAL_ONLY:-false}

MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

COMBO_CRIT=(default kl kl kl kl kl kl kl alignment alignment alignment)
COMBO_LAM=( 0       0.05 0.1 0.25 0.5 1 2 5 0.1       0.5       2)
SEEDS=(42 43 44)

N_COMBO=${#COMBO_CRIT[@]}
N_SEED=${#SEEDS[@]}
N_RUN=$((N_COMBO * N_SEED))
BASELINE_ID=${N_RUN}

# ---- BASELINE: eval only ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${BASELINE_ID}" ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true"
    sbatch scripts/cscs/arr_align_eval.sh "${BASE_MODEL}" "true"
    echo "Submitted baseline eval for ${BASE_MODEL} at $(date)"
    exit 0
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_RUN}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} is out of range (N_RUN=${N_RUN}); nothing to do"
    exit 0
fi

# ---- Resolve criterion / lambda / seed ----
COMBO_IDX=$((SLURM_ARRAY_TASK_ID / N_SEED))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEED))
CRITERION=${COMBO_CRIT[$COMBO_IDX]}
LAMBDA=${COMBO_LAM[$COMBO_IDX]}
SEED=${SEEDS[$SEED_IDX]}

RUN_ID="llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_lm_only_lr2e-5_st200_seed${SEED}"
echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Check if only evaluation is requested ----
if [ "${EVAL_ONLY}" = "true" ]; then
    sbatch scripts/cscs/arr_eval.sh "$RUN_ID"
    sbatch scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
    echo "Submitted EVAL only for ${RUN_ID}"
    exit 0
fi

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE" "$FREEZE seed=${SEED}")

# ---- Submit evaluation jobs dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "$RUN_ID"

sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"
