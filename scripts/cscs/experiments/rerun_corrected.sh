#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=rerun_corrected
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# Re-run of the paper's experiments with the corrected data pipeline
# (annotations mapped to caption tokens by character offsets; masks cropped to
# the image processor's center crop). Submit from a FRESH clone / PROJECT_DIR so
# RUN_IDs cannot collide with the original checkpoints, which arr_train.sh
# would otherwise skip as already trained.
#
# Stages (STAGE env var):
#   pilot  (a) corrected intrinsic metrics for the base model and for the
#              paper's published 2,400-step checkpoint on Hugging Face
#              (REF_MODELS, space separated; trained with the OLD pipeline,
#              so this shows what the old models score under the corrected
#              metric: paper values AMR 8.44 / AP 0.62 / NSS 2.27);
#          (b) one corrected training run at the paper's operating point
#              (kl, lambda 0.5, LM-only, lr 2e-5, 800 steps) with both
#              evaluations. Compare with the paper before launching `full`.
#   lambda the loss-weight sweep at 200 steps (lambda 0, 0.05, 0.1, 0.25,
#          0.5, 1, 5), to confirm the operating point under the corrected
#          signal before the rest is trained at lambda 0.5. Selection rule:
#          localization (AMR/AP/NSS) against validation loss; downstream
#          columns are reported but not used.
#   lora   the LoRA rank sweep alone (a subset of `full`; see the case below).
#   full   every remaining run: length series, lambda=0 controls (now also
#          duration-matched at 800 and 2400 steps), lambda sweep, component
#          ablation, LoRA rank sweep. Idempotent: finished runs are skipped.
#
# The base model's downstream (lmms-eval) numbers do not depend on the
# collator; `full` re-runs them anyway so the final report is self-contained.
#
# Usage:
#   STAGE=pilot sbatch scripts/cscs/experiments/rerun_corrected.sh
#   STAGE=full  sbatch scripts/cscs/experiments/rerun_corrected.sh
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:-pilot}"
SEED="${SEED:-42}"
MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"
REF_MODELS="${REF_MODELS:-teilers/llava-1.5-7b-saliency-kl0.5-st2400}"

# lmms-eval results exist when a results JSON was written under
# results/lm-eval/<name>/ (lmms-eval nests it in a model/timestamp directory).
has_lm_eval_results() {
    find "${PROJECT_DIR}/results/lm-eval/$1" -name '*results*.json' 2>/dev/null | grep -q .
}

# Evaluation jobs for a Hugging Face model id (no training). The align_eval
# output lands in outputs/<id with / replaced by __>/.
eval_hub_model() {
    local model="$1"
    local tag="${model//\//__}"
    if [ -f "${PROJECT_DIR}/outputs/${tag}/alignment_summary.json" ]; then
        echo "[skip align-eval] ${model}"
    else
        sbatch scripts/cscs/arr_align_eval.sh "$model" "true" >/dev/null
        echo "[align-eval] ${model}"
    fi
    if [ "${2:-}" = "with-downstream" ]; then
        if has_lm_eval_results "${model}"; then
            echo "[skip lm-eval] ${model}"
        else
            sbatch scripts/cscs/arr_eval.sh "$model" "true" >/dev/null
            echo "[lm-eval] ${model}"
        fi
    fi
}

LM_ONLY="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
PROJ_ONLY="model.freeze=[all] model.unfreeze=[multi_modal_projector]"
LM_PROJ="model.freeze=[vision_tower] model.unfreeze=[]"

# submit RUN_ID CRITERION LAMBDA OVERRIDES
#   Training is skipped by arr_train.sh when the checkpoint exists; the two
#   evaluations are skipped here when their result files exist.
submit() {
    local run_id="$1" crit="$2" lam="$3" overrides="$4"
    local model_dir="${PROJECT_DIR}/models/${run_id}"
    local dep=""
    if [ -f "${model_dir}/config.json" ] || [ -d "${model_dir}-merged" ]; then
        echo "[skip train] ${run_id}"
    else
        local jid
        jid=$(sbatch --parsable scripts/cscs/arr_train.sh \
            "$run_id" "$crit" "$lam" "$MODEL_SIZE" "$overrides")
        dep="--dependency=afterok:${jid}"
        echo "[train ${jid}] ${run_id}"
    fi
    if [ -f "${PROJECT_DIR}/outputs/${run_id}/alignment_summary.json" ]; then
        echo "[skip align-eval] ${run_id}"
    else
        sbatch ${dep} scripts/cscs/arr_align_eval.sh "$run_id" "false" >/dev/null
    fi
    if has_lm_eval_results "${run_id}"; then
        echo "[skip lm-eval] ${run_id}"
    else
        sbatch ${dep} scripts/cscs/arr_eval.sh "$run_id" >/dev/null
    fi
}

run_id() {  # run_id CRIT LAMBDA FREEZE_NAME LR STEPS [SUFFIX]
    echo "llava-1.5-${MODEL_SIZE}_$1_w$2_$3_lr$4_st$5_seed${SEED}${6:-}"
}

case "$STAGE" in
pilot)
    # (a) corrected intrinsic metrics for existing models (evaluation only)
    eval_hub_model "$BASE_MODEL"
    for ref in $REF_MODELS; do
        eval_hub_model "$ref"
    done
    # (b) one corrected training run at the operating point
    submit "$(run_id kl 0.5 lm_only 2e-5 800)" kl 0.5 \
        "$LM_ONLY optim.lr=2e-5 trainer.max_steps=800 seed=${SEED}"
    ;;
lambda)
    submit "$(run_id default 0 lm_only 2e-5 200)" default 0 \
        "$LM_ONLY optim.lr=2e-5 trainer.max_steps=200 seed=${SEED}"
    for lam in 0.05 0.1 0.25 0.5 1 5; do
        submit "$(run_id kl "$lam" lm_only 2e-5 200)" kl "$lam" \
            "$LM_ONLY optim.lr=2e-5 trainer.max_steps=200 seed=${SEED}"
    done
    ;;
lora)
    # LoRA rank sweep alone (same runs as in `full`), for resubmitting it
    # while other `full` trainings are still running: `full` would submit
    # those again because their checkpoints do not exist yet.
    for r in 4 16 128; do
        submit "$(run_id kl 0.5 lm_only 2e-4 800 "_lora_r${r}")" kl 0.5 \
            "lora.enabled=true lora.r=${r} lora.lora_alpha=$(( 2 * r )) optim.lr=2e-4 trainer.max_steps=800 seed=${SEED}"
    done
    ;;
full)
    # Base model: intrinsic metrics (skipped if the pilot did them) + downstream
    eval_hub_model "$BASE_MODEL" with-downstream
    # Length series, lambda = 0.5, LM-only (800 is the pilot run; skipped if done)
    for st in 200 800 1600 2400; do
        submit "$(run_id kl 0.5 lm_only 2e-5 "$st")" kl 0.5 \
            "$LM_ONLY optim.lr=2e-5 trainer.max_steps=${st} seed=${SEED}"
    done
    # Fine-tuning-only controls, duration-matched to every aligned length
    for st in 200 800 1600 2400; do
        submit "$(run_id default 0 lm_only 2e-5 "$st")" default 0 \
            "$LM_ONLY optim.lr=2e-5 trainer.max_steps=${st} seed=${SEED}"
    done
    # Loss-weight sweep at 200 steps (0 and 0.5 are covered above)
    for lam in 0.05 0.1 0.25 1 5; do
        submit "$(run_id kl "$lam" lm_only 2e-5 200)" kl "$lam" \
            "$LM_ONLY optim.lr=2e-5 trainer.max_steps=200 seed=${SEED}"
    done
    # Component ablation at 800 steps (LM-only is the pilot run)
    submit "$(run_id kl 0.5 proj_only 2e-5 800)" kl 0.5 \
        "$PROJ_ONLY optim.lr=2e-5 trainer.max_steps=800 seed=${SEED}"
    submit "$(run_id kl 0.5 lm_proj 2e-5 800)" kl 0.5 \
        "$LM_PROJ optim.lr=2e-5 trainer.max_steps=800 seed=${SEED}"
    # LoRA rank sweep at 800 steps, lr 2e-4, alpha = 2r
    for r in 4 16 128; do
        submit "$(run_id kl 0.5 lm_only 2e-4 800 "_lora_r${r}")" kl 0.5 \
            "lora.enabled=true lora.r=${r} lora.lora_alpha=$(( 2 * r )) optim.lr=2e-4 trainer.max_steps=800 seed=${SEED}"
    done
    ;;
*)
    echo "Unknown STAGE='${STAGE}' (expected pilot, lambda, lora, or full)" >&2
    exit 1
    ;;
esac
echo "Submitted stage '${STAGE}' at $(date)"
