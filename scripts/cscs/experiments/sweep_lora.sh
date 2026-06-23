#!/bin/bash
# Experiment C — LoRA vs full fine-tuning (+ rank sweep).  [run from repo root]
#
# At the knee lambda, LM-only, projector frozen in BOTH arms (full-FT lm_only
# freezes the projector; PEFT freezes the whole base, so LoRA only adapts the
# LM attention/MLP) — matched trainable surface. The rank sweep turns the
# paper's asserted "low-rank, head-concentrated" adaptation (Appendix B) into a
# measurement: the rank at which LoRA matches full FT ~ the intrinsic rank.
#
# LoRA needs its own (higher) LR; a small LoRA-LR sub-sweep picks it before the
# rank sweep. After training, arr_train.sh merges the adapter so align-eval,
# lmms-eval and compare_drift.py all see a full checkpoint.
#
#   KNEE_LAMBDA=0.5 LORA_LR=1e-4 bash scripts/cscs/experiments/sweep_lora.sh
#
# Env overrides: KNEE_LAMBDA, SEEDS, BASE_SEED, RANKS, LORA_LR, LORA_LRS,
#                FULL_LR, SUBSWEEP_RANK.

source "$(dirname "$0")/_lib.sh"

KNEE_LAMBDA="${KNEE_LAMBDA:-0.5}"
SEEDS=(${SEEDS:-42 43 44})
BASE_SEED="${BASE_SEED:-42}"
RANKS=(${RANKS:-4 8 16 32 64})
LORA_LR="${LORA_LR:-1e-4}"          # LR for the rank sweep (pick via sub-sweep below)
LORA_LRS=(${LORA_LRS:-5e-5 1e-4 3e-4})
SUBSWEEP_RANK="${SUBSWEEP_RANK:-8}"
FULL_LR="${FULL_LR:-2e-5}"          # full-FT reference LR (canonical)

lora_overrides() { echo "lora.enabled=true lora.r=$1 optim.lr=$2 seed=$3"; }

echo "== Experiment C: LoRA vs full FT (lambda=${KNEE_LAMBDA}) =="

# --- Full fine-tuning reference (LM-only). Matches Experiment A's run_id when
#     KNEE_LAMBDA/seed/LR coincide; harmless to re-submit, kept so C stands alone.
echo "-- full-FT LM-only reference (lr=${FULL_LR}) --"
for seed in "${SEEDS[@]}"; do
    run_id="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${FULL_LR}_st200_seed${seed}"
    submit_run "$run_id" kl "$KNEE_LAMBDA" "$FREEZE_LM_ONLY optim.lr=${FULL_LR} seed=${seed}" "true"
done

# --- LoRA-LR sub-sweep at a fixed rank (base seed only) to choose LORA_LR.
echo "-- LoRA-LR sub-sweep (r=${SUBSWEEP_RANK}, seed=${BASE_SEED}) --"
for lr in "${LORA_LRS[@]}"; do
    [ "$lr" = "$LORA_LR" ] && continue   # the chosen LR is covered by the rank sweep
    run_id="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${lr}_st200_seed${BASE_SEED}_lora_r${SUBSWEEP_RANK}"
    submit_run "$run_id" kl "$KNEE_LAMBDA" "$(lora_overrides "$SUBSWEEP_RANK" "$lr" "$BASE_SEED")" "true"
done

# --- Rank sweep at the chosen LoRA LR, across seeds.
echo "-- LoRA rank sweep (lr=${LORA_LR}) --"
for r in "${RANKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_id="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LORA_LR}_st200_seed${seed}_lora_r${r}"
        submit_run "$run_id" kl "$KNEE_LAMBDA" "$(lora_overrides "$r" "$LORA_LR" "$seed")" "true"
    done
done

echo "Done submitting Experiment C."
echo "After runs finish, compare drift/heads, e.g.:"
echo "  python scripts/python/compare_drift.py --per-head \\"
echo "    models/llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${FULL_LR}_st200_seed${BASE_SEED} \\"
echo "    models/llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${LORA_LR}_st200_seed${BASE_SEED}_lora_r${SUBSWEEP_RANK}-merged"
