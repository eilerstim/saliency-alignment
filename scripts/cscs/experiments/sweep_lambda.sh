#!/bin/bash
# Experiment A — loss-weight (lambda) sweep.  [run from repo root]
#
# Dose-response of the alignment loss's central knob on LLaVA-1.5-7B, LM-only,
# under the canonical recipe (cosine schedule, 200 steps). Produces the
# localization-vs-fluency curve and the localization-vs-downstream Pareto that
# defend the paper's central claim (Section 6, λ-ablation).
#
# Eval tiering (cost control): every (lambda, seed) gets the cheap intrinsic
# align-eval; the full downstream lmms-eval runs for the base seed at all
# lambdas, plus all seeds at the decisive lambdas (knee + endpoints).
#
#   bash scripts/cscs/experiments/sweep_lambda.sh
#
# Env overrides: SEEDS, KL_LAMBDAS, MSE_LAMBDAS, DOWNSTREAM_LAMBDAS, BASE_SEED.

source "$(dirname "$0")/_lib.sh"

SEEDS=(${SEEDS:-42 43 44})
BASE_SEED="${BASE_SEED:-42}"

# Primary (forward-KL) grid, plus a partial squared-error (MSE) grid so the
# paper can report the variant instead of deferring it.
KL_LAMBDAS=(${KL_LAMBDAS:-0 0.05 0.1 0.25 0.5 1 2 5})
MSE_LAMBDAS=(${MSE_LAMBDAS:-0.1 0.5 2})

# Lambdas that get the full downstream suite at *every* seed (others: base seed
# only). Defaults to the endpoints + the provisional knee; update post-hoc.
DOWNSTREAM_LAMBDAS=(${DOWNSTREAM_LAMBDAS:-0 0.5 5})

in_list() { local x="$1"; shift; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }

echo "== Experiment A: lambda sweep =="
submit_baseline

submit_grid() {
    local crit_label="$1" crit="$2"; shift 2
    local lambdas=("$@")
    for lam in "${lambdas[@]}"; do
        # lambda=0 is the no-alignment control: use the ZeroCriterion (loss=default)
        # so the KL upsample/softmax never runs (cleaner and cheaper than weight=0).
        local c="$crit"
        [ "$lam" = "0" ] && c="default"
        for seed in "${SEEDS[@]}"; do
            local down="false"
            if [ "$seed" = "$BASE_SEED" ] || in_list "$lam" "${DOWNSTREAM_LAMBDAS[@]}"; then
                down="true"
            fi
            local run_id="llava-1.5-${MODEL_SIZE}_${c}_w${lam}_lm_only_lr2e-5_st200_seed${seed}"
            submit_run "$run_id" "$c" "$lam" "$FREEZE_LM_ONLY seed=${seed}" "$down"
        done
    done
}

echo "-- KL grid --"
submit_grid kl kl "${KL_LAMBDAS[@]}"
echo "-- MSE grid (alignment) --"
submit_grid mse alignment "${MSE_LAMBDAS[@]}"

echo "Done submitting Experiment A."
