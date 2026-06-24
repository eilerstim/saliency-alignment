#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=align-eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

MODEL_NAME="$1"

if [ "${2:-false}" = "true" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="${PROJECT_DIR}/models/${MODEL_NAME}"
fi

[ -d "${MODEL_PATH}-merged" ] && MODEL_PATH="${MODEL_PATH}-merged"

source ./scripts/cscs/env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_IB_DISABLE=1

echo "Beginning alignment eval of ${MODEL_NAME} at $(date)"

srun \
    --environment=saliency \
    $PROJECT_DIR/.venv/bin/python -m align_eval.eval \
    run_id="${MODEL_NAME//\//__}" \
    +model_path="${MODEL_PATH}"

echo "Finished alignment eval of ${MODEL_NAME} at $(date)"
