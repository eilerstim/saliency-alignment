#!/bin/bash
#SBATCH --account=aa013 
#SBATCH --job-name=saliency-eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

MODEL_NAME="$1"

if [ "${2:-false}" = "true" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="models/${MODEL_NAME}"
fi
[ -d "${MODEL_PATH}-merged" ] && MODEL_PATH="${MODEL_PATH}-merged"

echo "Starting LM-eval of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

# LLaVA checkpoints patch tokenizer_class for vLLM, so they point at the hub
# tokenizer by default; for other architectures set TOKENIZER to the checkpoint's
# own tokenizer (e.g. TOKENIZER="${MODEL_PATH}").
TOKENIZER="${TOKENIZER:-llava-hf/llava-1.5-7b-hf}"
MODEL_ARGS="model=${MODEL_PATH},tokenizer=${TOKENIZER},tensor_parallel_size=1,dtype=bfloat16,trust_remote_code=True"

srun --environment=saliency_eval bash -c '
    set -euo pipefail
    uv pip install --force-reinstall numpy scipy --system
    # lmms-eval logs its evaluation-tracker args at INFO level, and they
    # include HF_TOKEN. Scrub tokens from both streams so none lands in logs/
    # (pipefail keeps a failing lmms_eval failing the job).
    scrub() { sed -u -E "s/hf_[A-Za-z0-9]{20,}/hf_<redacted>/g"; }
    python3 -m lmms_eval \
        --model vllm \
        --model_args "'"${MODEL_ARGS}"'" \
        --output_path "'"${PROJECT_DIR}"'/results/lm-eval/'"${MODEL_NAME}"'" \
        --include_path "'"${PROJECT_DIR}"'/eval/lmms_eval/tasks" \
        --tasks o3,vlms_are_biased,cv_bench_2d,cv_bench_3d,mmvp,mmstar,pope,countbench \
        2> >(scrub >&2) | scrub
'

echo "Finished LM-eval evaluation at $(date)"