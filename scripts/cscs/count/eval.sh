#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=saliency-count
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=saliency_eval
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail
mkdir -p logs

MODEL_NAME="$1"

# if second arg is set to true, it's a hf model name
# Otherwise, it's a path under models/
if [ "${2:-false}" = "true" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="${PROJECT_DIR}/models/${MODEL_NAME}"
fi

source ./scripts/cscs/count/env.sh

echo "Starting Count evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

# Extract model name for output directory
MODEL_DIR_NAME=$(basename "$MODEL_PATH" | sed 's/\//-/g')
OUTPUT_DIR="$COUNT_DIR/results/${MODEL_DIR_NAME}"

$COUNT_DIR/.count_venv/bin/python -m evaluator \
    ++root_dir=$COUNT_DIR \
    ++output_dir=$OUTPUT_DIR \
    ++model.engine=vllm_generate \
    ++model.model=$MODEL_PATH \
    ++model.vllm_params.tensor_parallel_size=4 \
    ++model.vllm_params.dtype=bfloat16 \
    ++model.vllm_params.trust_remote_code=true \
    ++model.vllm_params.tokenizer=llava-hf/llava-1.5-7b-hf \

echo "Finished LM-eval evaluation at $(date)"