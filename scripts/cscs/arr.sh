#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-7
# Array: 0-7: combinations of 6 lambdas + default + eval only (baseline)

set -euo pipefail
mkdir -p logs

MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

LAMBDAS=(0.01 0.1 0.3 0.5 1.0 2.0)

# ---- BASELINE: eval only ----
if [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}"
    exit 0
fi

# ---- Resolve criterion / lambda ----
if [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]; then
    CRITERION="default"
    LAMBDA=0.0
else
    CRITERION="alignment"
    LAMBDA=${LAMBDAS[${SLURM_ARRAY_TASK_ID}]}
fi

RUN_ID="llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}"

echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE")

# ---- Submit evaluation job dependent on training ----
sbatch \
    --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "models/${RUN_ID}"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"