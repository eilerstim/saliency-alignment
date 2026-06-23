#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lora
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-8
# Experiment C: LoRA vs full FT (+ rank sweep) at the knee lambda, LM-only, KL,
# projector frozen in both arms. LoRA LR defaults to 2e-4 and alpha = 2*rank
# (constant alpha/r), matching the LLaVA-1.5 / VIRAL finetune_lora recipe
# (lr 2e-4, r 128, alpha 256; haotian-liu/LLaVA & cvlab-kaist/VIRAL
# scripts/v1_5/finetune_lora.sh, arXiv:2310.03744 / 2509.07979). Layout (9 tasks):
#   0    = full-FT reference @ lr 2e-5
#   1..6 = LoRA rank {4,8,16,32,64,128} @ LORA_LR
#   7..8 = LoRA-LR sub-sweep {1e-4,3e-4} @ rank 8
# For error bars, add seeds to SEEDS below and bump --array accordingly.
set -euo pipefail
mkdir -p logs

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
LORA_LR=${LORA_LR:-2e-4}          # LLaVA/VIRAL canonical; confirm via sub-sweep
MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

SEEDS=(42)   # single seed; add 43 44 for error bars (and bump --array)
RANKS=(4 8 16 32 64 128)
SUB_LRS=(1e-4 3e-4)

NUM_SEEDS=${#SEEDS[@]}
NUM_FULL=${NUM_SEEDS}
NUM_RANK=$(( ${#RANKS[@]} * NUM_SEEDS ))
SUB_START=$(( NUM_FULL + NUM_RANK ))

# ---- Resolve method / lr / rank / seed from the task id ----
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${NUM_FULL}" ]; then
    METHOD="full"
    LR="2e-5"
    RANK=0
    SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
elif [ "${SLURM_ARRAY_TASK_ID}" -lt "${SUB_START}" ]; then
    J=$(( SLURM_ARRAY_TASK_ID - NUM_FULL ))
    METHOD="lora"
    LR=${LORA_LR}
    RANK=${RANKS[$(( J / NUM_SEEDS ))]}
    SEED=${SEEDS[$(( J % NUM_SEEDS ))]}
else
    J=$(( SLURM_ARRAY_TASK_ID - SUB_START ))
    METHOD="lora"
    LR=${SUB_LRS[$J]}
    RANK=8
    SEED=42
fi

# ---- Build run_id and overrides (alpha = 2*rank keeps alpha/r constant) ----
if [ "${METHOD}" = "lora" ]; then
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}_lora_r${RANK}"
    OVERRIDES="lora.enabled=true lora.r=${RANK} lora.lora_alpha=$(( 2 * RANK )) optim.lr=${LR} seed=${SEED}"
else
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}"
    OVERRIDES="$FREEZE optim.lr=${LR} seed=${SEED}"
fi

# ---- Submit training job + dependent evals ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh "$RUN_ID" kl "$KNEE_LAMBDA" "$MODEL_SIZE" "$OVERRIDES")

sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:${TRAIN_JOBID} scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok) for ${RUN_ID}"
