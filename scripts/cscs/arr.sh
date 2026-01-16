#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-5%4
# Array: 0-3 = criterion x lambdas; 4 = default (no regularization); 5 = baseline eval only

set -euo pipefail
mkdir -p logs

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY=${EVAL_ONLY:-false}

MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

CRITERION="kl"
LAMBDAS=(0.0001 0.001 0.01 0.1 1.0)

NUM_LAMBDAS=${#LAMBDAS[@]}
DEFAULT_ID=${NUM_LAMBDAS}
BASELINE_ID=$((NUM_LAMBDAS + 1))

# ---- BASELINE: eval only ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${BASELINE_ID}" ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true"
    sbatch scripts/cscs/count/eval.sh "${BASE_MODEL}" "true"
    echo "Submitted EVAL only for baseline model ${BASE_MODEL} at $(date)"
    exit 0
fi

# ---- Resolve lambda (or default) ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${DEFAULT_ID}" ]; then
    CRITERION="default"
    LAMBDA=0.0
else
    LAMBDA_INDEX=${SLURM_ARRAY_TASK_ID}   # 0..NUM_LAMBDAS-1
    LAMBDA=${LAMBDAS[$LAMBDA_INDEX]}
fi

RUN_ID="llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}"

echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Check if only evaluation is requested ----
if [ "${EVAL_ONLY}" = "true" ]; then
    sbatch scripts/cscs/arr_eval.sh "$RUN_ID"
    sbatch scripts/cscs/count/eval.sh "$RUN_ID" "false"
    echo "Submitted EVAL only for ${RUN_ID}"
    exit 0
fi

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE")

# ---- Submit evaluation job dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "$RUN_ID"

sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/count/eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"