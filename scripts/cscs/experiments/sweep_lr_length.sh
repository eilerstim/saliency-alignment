#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lr_length
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-8
# Experiment B: LR & length at the knee lambda (LM-only, KL). Parallel LRS/STS.
# Effective batch 128; ~98k images/epoch => 1 epoch ~= 766 steps. Base recipe
# lr 2e-5 follows LLaVA-1.5 (Liu et al. 2024, arXiv:2310.03744). The length axis
# and the low-LR-for-longer block jointly cover the LR x length interaction;
# the pure LR-axis at 200 steps is dropped as redundant.
#   0..4 length axis {100,200,400,800,1600} @ lr 2e-5   (~0.13 .. ~2.1 epochs)
#   5..8 low-LR-for-longer: {1e-5,5e-6} x {800,1600} steps
set -euo pipefail
mkdir -p logs

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
SEED=${SEED:-42}
MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
LRS=(2e-5 2e-5 2e-5 2e-5 2e-5 1e-5 1e-5 5e-6 5e-6)
STS=(100  200  400  800  1600 800  1600 800  1600)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
ST=${STS[$SLURM_ARRAY_TASK_ID]}
RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st${ST}_seed${SEED}"

JID=$(sbatch --parsable scripts/cscs/arr_train.sh \
    "$RUN_ID" kl "$KNEE_LAMBDA" "$MODEL_SIZE" \
    "$FREEZE optim.lr=${LR} trainer.max_steps=${ST} seed=${SEED}")
sbatch --dependency=afterok:$JID scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
echo "Submitted ${RUN_ID} (train ${JID})"
