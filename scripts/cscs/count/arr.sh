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
if [ ${SLURM_ARRAY_TASK_ID} -eq 7 ]; then
    sbatch scripts/cscs/count/eval.sh "${BASE_MODEL}" "true"
    echo "Submitted EVAL only for baseline model ${BASE_MODEL} at $(date)"
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

# ---- submit evaluation job ----
sbatch scripts/cscs/count/eval.sh "$RUN_ID"
echo "Submitted EVAL only for ${RUN_ID}"
