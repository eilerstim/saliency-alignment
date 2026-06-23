#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lr_length
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-7
# Experiment B: learning-rate & training-length sensitivity at the knee lambda.
# LM-only, KL. One axis at a time (parallel LR_LIST / ST_LIST):
#   tasks 0..4 = length {50,100,200,400,800} at lr=2e-5
#   tasks 5..7 = lr {5e-6,1e-5,5e-5} at 200 steps  (2e-5@200 is task 2)
# The cosine schedule re-derives warmup/decay from trainer.max_steps per length.
# Submit:  KNEE_LAMBDA=0.5 sbatch scripts/cscs/experiments/sweep_lr_length.sh

set -euo pipefail
mkdir -p logs

export EVAL_ONLY=${EVAL_ONLY:-false}
KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
SEED=${SEED:-42}

MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

LR_LIST=(2e-5 2e-5 2e-5 2e-5 2e-5 5e-6 1e-5 5e-5)
ST_LIST=(50   100  200  400  800  200  200  200)

LR=${LR_LIST[$SLURM_ARRAY_TASK_ID]}
STEPS=${ST_LIST[$SLURM_ARRAY_TASK_ID]}

RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st${STEPS}_seed${SEED}"
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
    "$RUN_ID" "kl" "$KNEE_LAMBDA" "$MODEL_SIZE" \
    "$FREEZE optim.lr=${LR} trainer.max_steps=${STEPS} seed=${SEED}")

# ---- Submit evaluation jobs dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "$RUN_ID"

sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"
