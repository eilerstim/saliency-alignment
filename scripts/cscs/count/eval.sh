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

$COUNT_DIR/.count_venv/bin/python -m evaluator \
    +root_dir $COUNT_DIR \
    +model.engine vllm_generate \
    +model.name $MODEL_PATH \
    +model.vllm_params "{tensor_parallel_size:4,dtype:bfloat16,trust_remote_code:True}"

echo "Finished LM-eval evaluation at $(date)"