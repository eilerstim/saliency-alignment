#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=sweep_lora
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-19
# Experiment C: LoRA vs full FT (+ rank sweep) at the knee lambda, LM-only, KL,
# projector frozen in both arms. Layout (20 tasks):
#   0..2   = full-FT reference @ lr 2e-5 x {3 seeds}
#   3..17  = LoRA rank {4,8,16,32,64} @ LORA_LR x {3 seeds}
#   18..19 = LoRA-LR sub-sweep {5e-5,3e-4} @ rank 8, seed 42
set -euo pipefail
mkdir -p logs

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
LORA_LR=${LORA_LR:-1e-4}          # LR for the rank sweep (pick via sub-sweep)
MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

SEEDS=(42 43 44)
RANKS=(4 8 16 32 64)
SUB_LRS=(5e-5 3e-4)

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

# ---- Build run_id and overrides ----
if [ "${METHOD}" = "lora" ]; then
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}_lora_r${RANK}"
    OVERRIDES="lora.enabled=true lora.r=${RANK} optim.lr=${LR} seed=${SEED}"
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
