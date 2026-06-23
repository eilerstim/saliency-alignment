#!/bin/bash
# Experiment B — learning-rate & training-length sensitivity.  [run from repo root]
#
# Around the operating point (KL, LM-only, knee lambda), vary one axis at a time:
#   * length: tests whether 200 steps is enough and whether longer training
#     erodes capability (the "captioning-drift" hypothesis, Section 6.3) —
#     downstream is tracked at every length for exactly this reason.
#   * learning rate: robustness, and whether LR governs LM-vs-projector drift.
# The cosine schedule re-derives warmup/decay from trainer.max_steps, so each
# length trains under a correct schedule.
#
#   KNEE_LAMBDA=0.5 bash scripts/cscs/experiments/sweep_lr_length.sh
#
# Env overrides: KNEE_LAMBDA, SEED, LENGTHS, LRS.

source "$(dirname "$0")/_lib.sh"

KNEE_LAMBDA="${KNEE_LAMBDA:-0.5}"   # set to Experiment A's knee
SEED="${SEED:-42}"
LENGTHS=(${LENGTHS:-50 100 200 400 800})
LRS=(${LRS:-5e-6 1e-5 2e-5 5e-5})

CENTER_LR="2e-5"
CENTER_STEPS="200"

run_one() {
    local lr="$1" steps="$2"
    # The center (2e-5, 200 steps) is already produced by Experiment A at this
    # lambda/seed; skip it to avoid a duplicate run.
    if [ "$lr" = "$CENTER_LR" ] && [ "$steps" = "$CENTER_STEPS" ]; then
        echo "  skip center lr=${lr} st=${steps} (produced by Experiment A)"
        return
    fi
    local run_id="llava-1.5-${MODEL_SIZE}_kl_w${KNEE_LAMBDA}_lm_only_lr${lr}_st${steps}_seed${SEED}"
    submit_run "$run_id" kl "$KNEE_LAMBDA" \
        "$FREEZE_LM_ONLY optim.lr=${lr} trainer.max_steps=${steps} seed=${SEED}" "true"
}

echo "== Experiment B: LR & length sensitivity (lambda=${KNEE_LAMBDA}, seed=${SEED}) =="

echo "-- length sweep (lr=${CENTER_LR}) --"
for steps in "${LENGTHS[@]}"; do
    run_one "$CENTER_LR" "$steps"
done

echo "-- learning-rate sweep (st=${CENTER_STEPS}) --"
for lr in "${LRS[@]}"; do
    run_one "$lr" "$CENTER_STEPS"
done

echo "Done submitting Experiment B."
