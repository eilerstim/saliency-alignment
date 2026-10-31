#!/bin/bash
# Rerun the paper reproduction with masks cropped like the images (LLaVA-1.5
# center-crops to 336x336; masks used to cover the full uncropped image). Run
# from the repo root on a node with sbatch:
#
#   bash scripts/cscs/experiments/reproduce_fixcrop.sh
#
# What the crop fix changes, and what is reused from the "fixalign" runs:
#   - lambda > 0 training uses the masks, so every such run is retrained.
#   - lambda = 0 training never reads masks: the seed-42 checkpoints, their
#     val-CE and downstream results are reused (linked/copied under the new run
#     id) and only their alignment eval is redone.
#   - Base model: downstream results stand; alignment eval is redone.
# Submits: LM-only lambda 0.5 at 200/800/1600/2400 steps (Figure 3, Tables 1-2),
# counting correlation (Section 5.3) for the base and the 800-step aligned
# model, and everything in reproduce_rest.sh (ablations, lambda and LoRA
# sweeps, seeds, drift).
set -euo pipefail
mkdir -p logs

# sbatch from inside an allocation hands that job's SLURM_* settings (e.g. its
# CPU binding mask) to the new jobs, whose srun then fails at launch.
for v in $(compgen -e | grep -E "^(SLURM|SRUN|SBATCH|SALLOC)_" | grep -v "^SLURM_CONF$"); do
    unset "$v"
done

export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OLD_TAG=fixalign
export RUN_TAG=fixcrop
BASE_MODEL="llava-hf/llava-1.5-7b-hf"
LM_ONLY="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
run_id() { echo "llava-1.5-7b_$1_lm_only_lr2e-5_st$2_seed42_$3"; }

# RETRAIN_ONLY=1 resubmits only the lambda > 0 work (base model and lambda = 0
# reuse already done)
RETRAIN_ONLY=${RETRAIN_ONLY:-0}

# Base model: alignment eval and counting
[ "$RETRAIN_ONLY" = 1 ] || sbatch -J "align_${BASE_MODEL//\//__}_${RUN_TAG}" scripts/cscs/arr_align_eval.sh "${BASE_MODEL}" "true" > /dev/null
[ "$RETRAIN_ONLY" = 1 ] || sbatch -J "counting_${BASE_MODEL//\//__}" scripts/cscs/arr_counting.sh "${BASE_MODEL}" "true" > /dev/null
[ "$RETRAIN_ONLY" = 1 ] || echo "Submitted base model alignment eval and counting"

for ST in 200 800 1600 2400; do
    # lambda = 0: reuse the checkpoint, logs and downstream results
    if [ "$RETRAIN_ONLY" != 1 ]; then
    OLD=$(run_id default_w0.0 "$ST" "$OLD_TAG")
    NEW=$(run_id default_w0.0 "$ST" "$RUN_TAG")
    ln -sfn "${PROJECT_DIR}/models/${OLD}" "models/${NEW}"
    mkdir -p "outputs/${NEW}" "results/lm-eval"
    cp -r "outputs/${OLD}/logs" "outputs/${NEW}/"
    [ -d "results/lm-eval/${OLD}" ] && cp -r "results/lm-eval/${OLD}" "results/lm-eval/${NEW}"
    sbatch -J "align_${NEW}" scripts/cscs/arr_align_eval.sh "${NEW}" "false" > /dev/null
    echo "Reused ${OLD} as ${NEW} (alignment eval submitted)"
    fi

    # lambda = 0.5: retrain, then alignment and downstream evals
    RUN=$(run_id kl_w0.5 "$ST" "$RUN_TAG")
    JID=$(sbatch --parsable -J "train_${RUN}" scripts/cscs/arr_train.sh \
        "$RUN" kl 0.5 7b "$LM_ONLY optim.lr=2e-5 trainer.max_steps=${ST} seed=42")
    sbatch -J "align_${RUN}" --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN" "false" > /dev/null
    sbatch -J "eval_${RUN}" --dependency=afterok:$JID scripts/cscs/arr_eval.sh "$RUN" > /dev/null
    if [ "$ST" = 800 ]; then
        KL800_JID=$JID
        sbatch -J "counting_${RUN}" --dependency=afterok:$JID scripts/cscs/arr_counting.sh "$RUN" > /dev/null
    fi
    echo "Submitted ${RUN} (train ${JID})"
done

# Ablations, lambda and LoRA sweeps, seeds and drift (drift also needs the
# LM-only 800-step run above)
DRIFT_AFTER="${KL800_JID}" SKIP_LAMBDA0_SEEDS="$RETRAIN_ONLY" bash scripts/cscs/experiments/reproduce_rest.sh
