# Source this from an eval script to materialize a merged HF checkpoint
# next to a LoRA adapter. Updates $MODEL_PATH in place when an adapter is
# detected (`adapter_config.json`).
#
# Concurrent-safe: jobs that arrive together (e.g. arr_eval + count/eval
# both dispatched with `afterok:$TRAIN_JOBID`) serialize via flock, and
# the merger writes into `<dst>.tmp` and atomic-renames into place — so
# the visible presence of `<dst>` always means "fully merged". A partial
# write is never observable at the target path.

if [ -f "${MODEL_PATH}/adapter_config.json" ]; then
    MERGED_PATH="${MODEL_PATH}-merged"
    if [ ! -d "${MERGED_PATH}" ]; then
        (
            flock 9
            if [ ! -d "${MERGED_PATH}" ]; then
                echo "LoRA adapter detected; merging ${MODEL_PATH} -> ${MERGED_PATH}"
                TMP_PATH="${MERGED_PATH}.tmp"
                rm -rf "${TMP_PATH}"
                "$PROJECT_DIR/.venv/bin/python" -m finetune.merge \
                    "${MODEL_PATH}" --output "${TMP_PATH}"
                mv "${TMP_PATH}" "${MERGED_PATH}"
            fi
        ) 9>"${MODEL_PATH}.merge.lock"
    fi
    MODEL_PATH="${MERGED_PATH}"
fi
