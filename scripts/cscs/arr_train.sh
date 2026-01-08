#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=saliency-finetune
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=saliency
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail

RUN_ID="$1"
CRITERION="$2"
LAMBDA="$3"
MODEL_SIZE="$4"

export CRITERION LAMBDA MODEL_SIZE

source ./scripts/cscs/env.sh

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_IB_DISABLE=1

echo "Beginning finetuning of ${RUN_ID} at $(date)"
echo "CRITERION=${CRITERION} LAMBDA=${LAMBDA} MODEL_SIZE=${MODEL_SIZE}"

srun $PROJECT_DIR/.venv/bin/python -m finetune \
    run_id="${RUN_ID}" \
    loss="${CRITERION}" \
    loss.weight="${LAMBDA}" \
    model.name="llava-hf/llava-1.5-${MODEL_SIZE}-hf"

echo "Finished finetuning of ${RUN_ID} at $(date)"