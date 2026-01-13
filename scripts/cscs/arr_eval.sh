#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=saliency-eval
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
    MODEL_PATH="models/${MODEL_NAME}"
fi

echo "Starting LM-eval of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

MODEL_ARGS="model=${MODEL_PATH},tokenizer=llava-hf/llava-1.5-7b-hf,tensor_parallel_size=4,dtype=bfloat16,trust_remote_code=True"
# WANDB_ARGS="project=alignment-eval,entity=teilers-eth-z-rich,name=${MODEL_NAME},dir=${PROJECT_DIR}/outputs/lmms_eval/"

python3 -m lmms_eval \
    --model vllm \
    --model_args "${MODEL_ARGS}" \
    --output_path "${PROJECT_DIR}/results/lm-eval/${MODEL_NAME}" \
    --include_path $PROJECT_DIR/eval/lmms_eval/tasks \
    --tasks mmbench,mmerealworld,gqa,pope,mmvetv2
    # --wandb_args "${WANDB_ARGS}" \

echo "Finished LM-eval evaluation at $(date)"
