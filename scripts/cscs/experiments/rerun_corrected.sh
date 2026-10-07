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
#   pilot  (a) corrected intrinsic metrics for the base model and, when
#              OLD_PROJECT_DIR is set, for the ORIGINAL 800-step checkpoint;
#          (b) one corrected training run at the paper's operating point
#              (kl, lambda 0.5, LM-only, lr 2e-5, 800 steps) with both
#              evaluations. Compare with the paper before launching `full`.
#   full   every remaining run: length series, lambda=0 controls (now also
#          duration-matched at 800 and 2400 steps), lambda sweep, component
#          ablation, LoRA rank sweep. Idempotent: finished runs are skipped.
#
# Downstream (lmms-eval) numbers of the base model do not depend on the
# collator and are not re-run here; reuse them from the original results.
#
# Usage:
#   STAGE=pilot [OLD_PROJECT_DIR=/path/to/original/clone] sbatch scripts/cscs/experiments/rerun_corrected.sh
#   STAGE=full sbatch scripts/cscs/experiments/rerun_corrected.sh
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:-pilot}"
SEED="${SEED:-42}"
MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

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
    if [ -d "${PROJECT_DIR}/results/lm-eval/${run_id}" ]; then
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
    sbatch scripts/cscs/arr_align_eval.sh "$BASE_MODEL" "true" >/dev/null
    echo "[align-eval] ${BASE_MODEL} (corrected metrics)"
    if [ -n "${OLD_PROJECT_DIR:-}" ]; then
        OLD_CKPT="${OLD_PROJECT_DIR}/models/llava-1.5-${MODEL_SIZE}_kl_w0.5_lm_only_lr2e-5_st800_seed${SEED}"
        if [ -f "${OLD_CKPT}/config.json" ]; then
            sbatch scripts/cscs/arr_align_eval.sh "$OLD_CKPT" "true" >/dev/null
            echo "[align-eval] original 800-step checkpoint under corrected metrics: ${OLD_CKPT}"
        else
            echo "WARNING: ${OLD_CKPT} not found; skipping corrected eval of the original checkpoint"
        fi
    fi
    # (b) one corrected training run at the operating point
    submit "$(run_id kl 0.5 lm_only 2e-5 800)" kl 0.5 \
        "$LM_ONLY optim.lr=2e-5 trainer.max_steps=800 seed=${SEED}"
    ;;
full)
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
    echo "Unknown STAGE='${STAGE}' (expected pilot or full)" >&2
    exit 1
    ;;
esac
echo "Submitted stage '${STAGE}' at $(date)"
