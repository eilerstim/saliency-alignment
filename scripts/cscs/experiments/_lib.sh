#!/bin/bash
# Shared submission helpers for the paper's sweep drivers (lambda, lr/length,
# lora). Source this from a driver, then call submit_run / submit_baseline.
#
# These drivers are plain bash: run them from the repo root on a login node;
# they loop over the grid and issue sbatch calls (train -> dependent evals).
# Every swept axis is encoded in the run_id so checkpoints/results never
# collide and scripts/python/aggregate_results.py can recover the axes.
#
# Run-id convention (parsed by aggregate_results.py):
#   llava-1.5-7b_<crit>_w<lambda>_<freeze>_lr<lr>_st<steps>_seed<seed>[_lora_r<rank>]

set -euo pipefail
mkdir -p logs

MODEL_SIZE="${MODEL_SIZE:-7b}"
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

# Full fine-tuning, language-model only (vision tower + projector frozen). This
# is the decisive component (Table 1) and the canonical surface for the sweeps.
FREEZE_LM_ONLY="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"

# submit_run <run_id> <criterion> <lambda> <extra_overrides> [downstream=true]
#   Trains one config, then submits the intrinsic align-eval (always) and the
#   downstream lmms-eval (only when <downstream>=true) with afterok deps.
submit_run() {
    local run_id="$1" crit="$2" lam="$3" extra="$4" downstream="${5:-true}"
    local jid
    jid=$(sbatch --parsable scripts/cscs/arr_train.sh \
        "$run_id" "$crit" "$lam" "$MODEL_SIZE" "$extra")
    echo "  TRAIN ${run_id}  (job ${jid}, downstream=${downstream})"

    sbatch --dependency="afterok:${jid}" \
        scripts/cscs/arr_align_eval.sh "$run_id" "false" >/dev/null

    if [ "$downstream" = "true" ]; then
        sbatch --dependency="afterok:${jid}" \
            scripts/cscs/arr_eval.sh "$run_id" >/dev/null
    fi
}

# submit_baseline — eval the untrained base model (intrinsic + downstream).
# Idempotent anchor; safe to call once per sweep.
submit_baseline() {
    echo "  BASELINE eval ${BASE_MODEL}"
    sbatch scripts/cscs/arr_align_eval.sh "${BASE_MODEL}" "true" >/dev/null
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true" >/dev/null
}
