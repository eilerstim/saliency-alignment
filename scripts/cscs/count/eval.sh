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
#SBATCH --environment=saliency
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

echo "Starting Count evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

$COUNT_DIR/.count_venv/bin/python -m vlm-counting \
    +root_dir $COUNT_DIR \
    +model.engine huggingface \
    +model.name $MODEL_PATH

echo "Finished LM-eval evaluation at $(date)"