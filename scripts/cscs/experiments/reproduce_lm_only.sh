#!/bin/bash
# Reproduce the paper's LM-only results: Figure 3 (AMR/AP/NSS vs. training
# length), Table 1 (downstream suite) and the Base / LM-only rows of Table 2.
# Plain submitter (run from the repo root on a node with sbatch):
#
#   bash scripts/cscs/experiments/reproduce_lm_only.sh
#
# Runs, all LM-only (vision tower + projector frozen), lr 2e-5, seed 42:
#   lambda in {0.5 (kl), 0 (default)} x steps in {200, 800, 1600, 2400}
# Each training run is followed by the alignment eval (Figure 3, Table 2).
# Downstream lmms-eval runs for 800/1600/2400 steps (Table 1). The base model
# gets both evals. Trained-model downstream evals wait for the base one so the
# benchmark datasets are downloaded into the shared HF cache only once.
# SKIP_BASE=1 skips the base model evals (e.g. when they were run elsewhere).
#
# Requires: data downloaded, slurm/saliency.sqsh + slurm/eval.sqsh built, and
# .venv created inside the container (scripts/cscs/arr_setup.sh).
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-$PWD}"

SEED=${SEED:-42}
LR=${LR:-2e-5}
MODEL_SIZE=7b
BASE_MODEL="llava-hf/llava-1.5-${MODEL_SIZE}-hf"
FREEZE="model.freeze=[vision_tower,multi_modal_projector] model.unfreeze=[]"
# Suffix keeps these runs distinct from earlier ones (checkpoints, W&B names)
RUN_TAG=${RUN_TAG:-fixalign}
STEPS=(200 800 1600 2400)
DOWNSTREAM_STEPS=" 800 1600 2400 "

EVAL_DEP=""
if [ "${SKIP_BASE:-0}" != "1" ]; then
    BASE_EVAL=$(sbatch --parsable scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true")
    BASE_ALIGN=$(sbatch --parsable scripts/cscs/arr_align_eval.sh "${BASE_MODEL}" "true")
    EVAL_DEP=",afterany:${BASE_EVAL}"
    echo "Submitted base model evals (downstream ${BASE_EVAL}, alignment ${BASE_ALIGN})"
fi

for CRIT_LAMBDA in "kl 0.5" "default 0.0"; do
    read -r CRITERION LAMBDA <<< "${CRIT_LAMBDA}"
    for ST in "${STEPS[@]}"; do
        RUN_ID="llava-1.5-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_lm_only_lr${LR}_st${ST}_seed${SEED}_${RUN_TAG}"

        JID=$(sbatch --parsable scripts/cscs/arr_train.sh \
            "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE" \
            "$FREEZE optim.lr=${LR} trainer.max_steps=${ST} seed=${SEED}")
        sbatch --dependency=afterok:$JID scripts/cscs/arr_align_eval.sh "$RUN_ID" "false" > /dev/null
        if [[ "${DOWNSTREAM_STEPS}" == *" ${ST} "* ]]; then
            sbatch --dependency=afterok:${JID}${EVAL_DEP} \
                scripts/cscs/arr_eval.sh "$RUN_ID" > /dev/null
        fi
        echo "Submitted ${RUN_ID} (train ${JID})"
    done
done
