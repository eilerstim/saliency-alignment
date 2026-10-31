#!/bin/bash
# Reproduce the paper's remaining experiments with the fixed alignment and
# tokenizer (run from the repo root on a node with sbatch):
#
#   bash scripts/cscs/experiments/reproduce_rest.sh
#
#   1. Component ablation (Tables 2 and 4): projector only, LM + projector;
#      lambda 0.5, 800 steps.
#   2. Loss-weight sensitivity (Table 3): lambda 0.05, 0.1, 0.25, 1, 5 at
#      200 steps (0 and 0.5 come from reproduce_lm_only.sh). Downstream only
#      CountBench and CV-2D, the two benchmarks the table reports.
#   3. LoRA rank sweep (Table 5): r 4, 16, 128 (alpha 2r), lr 2e-4, 800 steps.
#   4. Seed variation (Appendix D): lambda 0.5 and 0, 800 steps, seeds 43, 44.
#   5. Parameter drift (Appendix E) of the 800-step checkpoints, CPU only,
#      once the runs above have trained. DRIFT_AFTER (colon-separated job ids)
#      adds dependencies, e.g. on the LM-only 800-step run submitted elsewhere.
#
# Every training run is followed by its alignment eval and downstream eval.
# Jobs are named after their run so they are easy to find in squeue.
set -euo pipefail
mkdir -p logs

# sbatch from inside an allocation hands that job's SLURM_* settings (e.g. its
# CPU binding mask) to the new jobs, whose srun then fails at launch.
for v in $(compgen -e | grep -E "^(SLURM|SRUN|SBATCH|SALLOC)_" | grep -v "^SLURM_CONF$"); do
    unset "$v"
done

export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL_SIZE=7b
RUN_TAG=${RUN_TAG:-fixalign}
LM_ONLY="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
PROJ_ONLY="model.freeze=[all] model.unfreeze=[multi_modal_projector]"
LM_PROJ="model.freeze=[vision_tower] model.unfreeze=[]"

# submit RUN_ID CRITERION LAMBDA OVERRIDES [EVAL_TASKS]: training plus its
# alignment and downstream evals; prints the training job id
submit() {
    local run_id="$1" crit="$2" lam="$3" overrides="$4" tasks="${5:-}"
    local jid
    jid=$(sbatch --parsable -J "train_${run_id}" scripts/cscs/arr_train.sh \
        "$run_id" "$crit" "$lam" "$MODEL_SIZE" "$overrides")
    sbatch --parsable -J "align_${run_id}" --dependency=afterok:$jid \
        scripts/cscs/arr_align_eval.sh "$run_id" "false" > /dev/null
    # TASKS goes through the environment: --export=...,TASKS=a,b would split
    # the comma-separated task list into separate variables
    TASKS="${tasks:-${TASKS:-}}" sbatch --parsable -J "eval_${run_id}" --dependency=afterok:$jid \
        scripts/cscs/arr_eval.sh "$run_id" > /dev/null
    echo "Submitted ${run_id} (train ${jid})" >&2
    echo "$jid"
}

# 1. Component ablation
PROJ_ID="llava-1.5-${MODEL_SIZE}_kl_w0.5_proj_only_lr2e-5_st800_seed42_${RUN_TAG}"
LMPROJ_ID="llava-1.5-${MODEL_SIZE}_kl_w0.5_lm_proj_lr2e-5_st800_seed42_${RUN_TAG}"
PROJ_JID=$(submit "$PROJ_ID" kl 0.5 "$PROJ_ONLY optim.lr=2e-5 trainer.max_steps=800 seed=42")
LMPROJ_JID=$(submit "$LMPROJ_ID" kl 0.5 "$LM_PROJ optim.lr=2e-5 trainer.max_steps=800 seed=42")

# 2. Loss-weight sensitivity
for LAM in 0.05 0.1 0.25 1 5; do
    submit "llava-1.5-${MODEL_SIZE}_kl_w${LAM}_lm_only_lr2e-5_st200_seed42_${RUN_TAG}" kl "$LAM" \
        "$LM_ONLY optim.lr=2e-5 trainer.max_steps=200 seed=42" "countbench,cv_bench_2d" > /dev/null
done

# 3. LoRA rank sweep
LORA_DEPS=()
LORA_CKPTS=()
for R in 4 16 128; do
    RUN_ID="llava-1.5-${MODEL_SIZE}_kl_w0.5_lm_only_lr2e-4_st800_seed42_lora_r${R}_${RUN_TAG}"
    LORA_DEPS+=("$(submit "$RUN_ID" kl 0.5 "$LM_ONLY lora.enabled=true lora.r=${R} lora.lora_alpha=$(( 2 * R )) optim.lr=2e-4 trainer.max_steps=800 seed=42")")
    LORA_CKPTS+=("models/${RUN_ID}-merged")
done

# 4. Seed variation
for SEED in 43 44; do
    for CRIT_LAMBDA in "kl 0.5" "default 0.0"; do
        read -r CRITERION LAMBDA <<< "${CRIT_LAMBDA}"
        # SKIP_LAMBDA0_SEEDS=1: the lambda = 0 seed runs are already submitted
        [ "$CRITERION" = default ] && [ "${SKIP_LAMBDA0_SEEDS:-0}" = 1 ] && continue
        submit "llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_lm_only_lr2e-5_st800_seed${SEED}_${RUN_TAG}" \
            "$CRITERION" "$LAMBDA" "$LM_ONLY optim.lr=2e-5 trainer.max_steps=800 seed=${SEED}" > /dev/null
    done
done

# 5. Parameter drift: component ablation (LM only first, as in the paper's
# comparisons) and full fine-tuning vs LoRA
LM_ONLY_CKPT="models/llava-1.5-${MODEL_SIZE}_kl_w0.5_lm_only_lr2e-5_st800_seed42_${RUN_TAG}"
DRIFT_A=$(sbatch --parsable -J "drift_components_${RUN_TAG}" \
    --dependency=afterok:${PROJ_JID}:${LMPROJ_JID}${DRIFT_AFTER:+:$DRIFT_AFTER} scripts/cscs/arr_drift.sh \
    "components_${RUN_TAG}" "$LM_ONLY_CKPT" "models/${LMPROJ_ID}" "models/${PROJ_ID}")
DRIFT_B=$(sbatch --parsable -J "drift_lora_${RUN_TAG}" \
    --dependency=afterok:$(IFS=:; echo "${LORA_DEPS[*]}")${DRIFT_AFTER:+:$DRIFT_AFTER} scripts/cscs/arr_drift.sh \
    "lora_${RUN_TAG}" "$LM_ONLY_CKPT" "${LORA_CKPTS[@]}")
echo "Submitted drift analyses (${DRIFT_A}, ${DRIFT_B})"
