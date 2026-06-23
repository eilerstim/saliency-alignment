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
# Experiment C: LoRA vs full fine-tuning (+ rank sweep) at the knee lambda.
# LM-only, KL, projector frozen in both arms (full-FT lm_only freezes the
# projector; PEFT freezes the whole base, so LoRA only adapts the LM).
# Layout (parallel arrays, built below): default grid = 20 tasks
#   - full-FT reference (lr=2e-5)            x 3 seeds            = 3
#   - LoRA rank sweep (lr=LORA_LR) r in {4,8,16,32,64} x 3 seeds = 15
#   - LoRA-LR sub-sweep (rank 8, seed 42) lr in {5e-5,3e-4}      = 2
# arr_train.sh merges the adapter after training so the evals see a full ckpt.
# Submit:  KNEE_LAMBDA=0.5 LORA_LR=1e-4 sbatch scripts/cscs/experiments/sweep_lora.sh

set -euo pipefail
mkdir -p logs

export EVAL_ONLY=${EVAL_ONLY:-false}
KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
LORA_LR=${LORA_LR:-1e-4}          # LR for the rank sweep (pick via sub-sweep)
FULL_LR=2e-5

MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

SEEDS=(42 43 44)
RANKS=(4 8 16 32 64)
SUB_LRS=(5e-5 3e-4)               # LoRA-LR sub-sweep (excludes LORA_LR)

# Build parallel arrays: METHOD / LR / RANK / SEED.
M_METHOD=(); M_LR=(); M_RANK=(); M_SEED=()
for s in "${SEEDS[@]}"; do
    M_METHOD+=("full"); M_LR+=("$FULL_LR"); M_RANK+=("0"); M_SEED+=("$s")
done
for r in "${RANKS[@]}"; do
    for s in "${SEEDS[@]}"; do
        M_METHOD+=("lora"); M_LR+=("$LORA_LR"); M_RANK+=("$r"); M_SEED+=("$s")
    done
done
for lr in "${SUB_LRS[@]}"; do
    M_METHOD+=("lora"); M_LR+=("$lr"); M_RANK+=("8"); M_SEED+=("42")
done

N_RUN=${#M_METHOD[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_RUN}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} is out of range (N_RUN=${N_RUN}); nothing to do"
    exit 0
fi

METHOD=${M_METHOD[$SLURM_ARRAY_TASK_ID]}
LR=${M_LR[$SLURM_ARRAY_TASK_ID]}
RANK=${M_RANK[$SLURM_ARRAY_TASK_ID]}
SEED=${M_SEED[$SLURM_ARRAY_TASK_ID]}

if [ "${METHOD}" = "lora" ]; then
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}_lora_r${RANK}"
    OVERRIDES="lora.enabled=true lora.r=${RANK} optim.lr=${LR} seed=${SEED}"
else
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}"
    OVERRIDES="$FREEZE optim.lr=${LR} seed=${SEED}"
fi
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
    "$RUN_ID" "kl" "$KNEE_LAMBDA" "$MODEL_SIZE" "$OVERRIDES")

# ---- Submit evaluation jobs dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "$RUN_ID"

sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"
