#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lora
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-2
# Experiment C: LoRA rank sweep at the knee lambda (LM-only, KL), single seed.
# The full-FT reference is Experiment A's knee run (kl@<knee>, lm_only, lr 2e-5,
# st 200) -- reused, not retrained here. LoRA LR 2e-4 and alpha = 2*rank match
# the LLaVA-1.5 / VIRAL finetune_lora recipe (lr 2e-4, r 128, alpha 256;
# haotian-liu/LLaVA & cvlab-kaist/VIRAL, arXiv:2310.03744 / 2509.07979).
# Layout (3 tasks):
#   0..2 = LoRA rank {4,16,128} @ LORA_LR
# For error bars, add seeds to SEEDS and bump --array accordingly.
set -euo pipefail
mkdir -p logs

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
LORA_LR=${LORA_LR:-2e-4}          # LLaVA/VIRAL canonical
MODEL_SIZE=7b

SEEDS=(42)   # single seed; add 43 44 for error bars (and bump --array)
RANKS=(4 16 128)
NUM_SEEDS=${#SEEDS[@]}

RANK=${RANKS[$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))]}

# alpha = 2*rank keeps alpha/r constant across the sweep.
RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LORA_LR}_st200_seed${SEED}_lora_r${RANK}"
OVERRIDES="lora.enabled=true lora.r=${RANK} lora.lora_alpha=$(( 2 * RANK )) optim.lr=${LORA_LR} seed=${SEED}"

# ---- Submit training job + dependent evals ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh "$RUN_ID" kl "$KNEE_LAMBDA" "$MODEL_SIZE" "$OVERRIDES")

sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok) for ${RUN_ID}"
