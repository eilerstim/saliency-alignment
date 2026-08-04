#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=run_mse
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# Squared-error (MSE) alignment variant, run once at the knee lambda (Exp. A)
# and the best lr/length (Exp. B) so the paper can report KL-vs-MSE at a matched
# operating point. Submit after A and B:
#   KNEE_LAMBDA=0.5 MSE_LR=2e-5 MSE_STEPS=200 sbatch scripts/cscs/experiments/run_mse.sh
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}
MSE_LR=${MSE_LR:-2e-5}
MSE_STEPS=${MSE_STEPS:-200}
SEED=${SEED:-42}
MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

RUN_ID="llava-1.5-${MODEL_SIZE}_alignment_w${KNEE_LAMBDA}_lm_only_lr${MSE_LR}_st${MSE_STEPS}_seed${SEED}"

JID=$(sbatch --parsable scripts/cscs/arr_train.sh \
    "$RUN_ID" alignment "$KNEE_LAMBDA" "$MODEL_SIZE" \
    "$FREEZE optim.lr=${MSE_LR} trainer.max_steps=${MSE_STEPS} seed=${SEED}")
sbatch --dependency=afterok:$JID scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
echo "Submitted ${RUN_ID} (train ${JID})"
