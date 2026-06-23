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
# Experiment A: lambda dose-response (LM-only, canonical recipe).
# 11 (criterion,lambda) combos x 3 seeds = tasks 0..32; task 33 = baseline eval.
set -euo pipefail
mkdir -p logs

MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
CRITS=(default kl kl kl kl kl kl kl alignment alignment alignment)
LAMS=(0 0.05 0.1 0.25 0.5 1 2 5 0.1 0.5 2)
SEEDS=(42 43 44)
N=$(( ${#CRITS[@]} * ${#SEEDS[@]} ))

if [ "${SLURM_ARRAY_TASK_ID}" -eq "$N" ]; then
    BASE="llava-hf/llava-1.5-${MODEL_SIZE}-hf"
    sbatch scripts/cscs/arr_eval.sh "$BASE" "true"
    sbatch scripts/cscs/arr_align_eval.sh "$BASE" "true"
    exit 0
fi

C=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
CRIT=${CRITS[$C]}
LAM=${LAMS[$C]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))]}
RUN_ID="llava-1.5-${MODEL_SIZE}_${CRIT}_w${LAM}_lm_only_lr2e-5_st200_seed${SEED}"

JID=$(sbatch --parsable scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRIT" "$LAM" "$MODEL_SIZE" "$FREEZE seed=${SEED}")
sbatch --dependency=afterok:$JID scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
echo "Submitted ${RUN_ID} (train ${JID})"
