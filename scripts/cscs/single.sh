#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
mkdir -p logs

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY=${EVAL_ONLY:-false}

MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

CRITERION="default"
LAMBDA=0.0

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