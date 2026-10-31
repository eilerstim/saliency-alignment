#!/bin/bash
# TEMPORARY: submit the downstream (lmms-eval) evals of the LM-only reproduction
# (reproduce_lm_only.sh) once a Hugging Face token is in $PROJECT_DIR/.hf/token.
# MMVP, MMStar and POPE set `token: True` in their lmms-eval task configs, so
# these evals fail without one. Run from the repo root on any node with sbatch:
#
#   bash scripts/cscs/experiments/downstream_evals_after_hf_login.sh
#
# Submits the base model eval and one eval per 800/1600/2400-step run. A run
# whose training job is still queued or running gets an afterok dependency on
# it; a finished run needs its checkpoint in models/. Trained-model evals wait
# for the base eval so benchmark datasets are downloaded only once. Models that
# already have lmms-eval results are skipped, so re-running is safe.
set -euo pipefail
mkdir -p logs

# sbatch from inside an allocation hands that job's SLURM_* settings (e.g. its
# CPU binding mask) to the new jobs, whose srun then fails at launch.
for v in $(compgen -e | grep -E "^(SLURM|SRUN|SBATCH|SALLOC)_" | grep -v "^SLURM_CONF$"); do
    unset "$v"
done

export PROJECT_DIR="${PROJECT_DIR:-$PWD}"

TOKEN_FILE="${PROJECT_DIR}/.hf/token"
if [ ! -s "${TOKEN_FILE}" ]; then
    echo "No Hugging Face token at ${TOKEN_FILE}; add it first." >&2
    exit 1
fi
if ! curl -sf -o /dev/null -H "Authorization: Bearer $(cat "${TOKEN_FILE}")" \
    https://huggingface.co/api/whoami-v2; then
    echo "The token in ${TOKEN_FILE} was rejected by the Hugging Face Hub." >&2
    exit 1
fi

has_results() {
    [ -n "$(find "${PROJECT_DIR}/results/lm-eval/$1" -name '*results*.json' -print -quit 2>/dev/null)" ]
}
# Eval jobs are named after their model, so a re-run can see which are queued
job_name() { echo "eval_${1//\//__}"; }
queued_job() { squeue -u "$USER" -h -n "$(job_name "$1")" -o %i 2>/dev/null | head -n 1; }

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
BASE_EVAL=""
if has_results "${BASE_MODEL}"; then
    echo "Skipping ${BASE_MODEL}: results exist"
elif BASE_EVAL=$(queued_job "${BASE_MODEL}") && [ -n "${BASE_EVAL}" ]; then
    echo "Skipping ${BASE_MODEL}: eval ${BASE_EVAL} already queued"
else
    BASE_EVAL=$(sbatch --parsable -J "$(job_name "${BASE_MODEL}")" scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true")
    echo "Submitted ${BASE_MODEL} (eval ${BASE_EVAL})"
fi

# "<run id> <training job id>" as submitted by reproduce_lm_only.sh
RUNS=(
    "llava-1.5-7b_kl_w0.5_lm_only_lr2e-5_st800_seed42_fixalign 3750945"
    "llava-1.5-7b_kl_w0.5_lm_only_lr2e-5_st1600_seed42_fixalign 3750948"
    "llava-1.5-7b_kl_w0.5_lm_only_lr2e-5_st2400_seed42_fixalign 3750951"
    "llava-1.5-7b_default_w0.0_lm_only_lr2e-5_st800_seed42_fixalign 3750956"
    "llava-1.5-7b_default_w0.0_lm_only_lr2e-5_st1600_seed42_fixalign 3750959"
    "llava-1.5-7b_default_w0.0_lm_only_lr2e-5_st2400_seed42_fixalign 3750962"
)

for RUN in "${RUNS[@]}"; do
    read -r RUN_ID TRAIN_JID <<< "${RUN}"
    if has_results "${RUN_ID}"; then
        echo "Skipping ${RUN_ID}: results exist"
        continue
    fi
    # Set while an eval of this run is going on elsewhere (e.g. an interactive node)
    if [ -e "${PROJECT_DIR}/results/lm-eval/${RUN_ID}.running" ]; then
        echo "Skipping ${RUN_ID}: being evaluated elsewhere (${RUN_ID}.running)"
        continue
    fi
    if [ -n "$(queued_job "${RUN_ID}")" ]; then
        echo "Skipping ${RUN_ID}: eval $(queued_job "${RUN_ID}") already queued"
        continue
    fi

    DEPS=()
    if [ -n "$(squeue -h -j "${TRAIN_JID}" -o %T 2>/dev/null || true)" ]; then
        DEPS+=("afterok:${TRAIN_JID}")
    elif [ ! -f "${PROJECT_DIR}/models/${RUN_ID}/config.json" ]; then
        echo "WARNING: skipping ${RUN_ID}: training job ${TRAIN_JID} is gone but there is" \
            "no checkpoint (see logs/saliency-finetune_${TRAIN_JID}_*.err)" >&2
        continue
    fi
    [ -n "${BASE_EVAL}" ] && DEPS+=("afterany:${BASE_EVAL}")

    DEP_ARGS=()
    if [ ${#DEPS[@]} -gt 0 ]; then
        DEP_ARGS=(--dependency="$(IFS=,; echo "${DEPS[*]}")" --kill-on-invalid-dep=yes)
    fi
    JID=$(sbatch --parsable -J "$(job_name "${RUN_ID}")" "${DEP_ARGS[@]}" scripts/cscs/arr_eval.sh "${RUN_ID}")
    echo "Submitted ${RUN_ID} (eval ${JID}${DEPS:+, after ${DEPS[*]}})"
done
