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
# projector frozen in both arms. Default grid (20 tasks): full-FT ref @lr2e-5 x3
# seeds; LoRA r{4,8,16,32,64}@LORA_LR x3 seeds; LoRA-LR sub-sweep r8 @{5e-5,3e-4}.
set -euo pipefail
mkdir -p logs

KNEE_LAMBDA=${KNEE_LAMBDA:-0.5}   # set to Experiment A's knee
LORA_LR=${LORA_LR:-1e-4}          # LR for the rank sweep (pick via sub-sweep)
MODEL_SIZE=7b
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

# Parallel arrays: method / lr / rank / seed.
MS=(); LRS=(); RS=(); SS=()
for s in 42 43 44; do MS+=(full); LRS+=(2e-5); RS+=(0); SS+=("$s"); done
for r in 4 8 16 32 64; do for s in 42 43 44; do MS+=(lora); LRS+=("$LORA_LR"); RS+=("$r"); SS+=("$s"); done; done
for lr in 5e-5 3e-4; do MS+=(lora); LRS+=("$lr"); RS+=(8); SS+=(42); done

i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}; LR=${LRS[$i]}; R=${RS[$i]}; SEED=${SS[$i]}

if [ "$M" = lora ]; then
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}_lora_r${R}"
    OV="lora.enabled=true lora.r=${R} optim.lr=${LR} seed=${SEED}"
else
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LR}_st200_seed${SEED}"
    OV="$FREEZE optim.lr=${LR} seed=${SEED}"
fi

JID=$(sbatch --parsable scripts/cscs/arr_train.sh "$RUN_ID" kl "$KNEE_LAMBDA" "$MODEL_SIZE" "$OV")
sbatch --dependency=afterok:$JID scripts/cscs/arr_eval.sh "$RUN_ID"
sbatch --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN_ID" "false"
echo "Submitted ${RUN_ID} (train ${JID})"
