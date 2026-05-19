#!/bin/bash
#SBATCH --account=aa013 
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
MODEL="$4"
EXTRA_OVERRIDES="${5:-}"

export CRITERION LAMBDA MODEL

source ./scripts/cscs/env.sh

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_IB_DISABLE=1

export CUDA_LAUNCH_BLOCKING=1 
export HYDRA_FULL_ERROR=1

echo "Beginning finetuning of ${RUN_ID} at $(date)"
echo "CRITERION=${CRITERION} LAMBDA=${LAMBDA} MODEL=${MODEL} EXTRA_OVERRIDES=${EXTRA_OVERRIDES}"

# EXTRA_OVERRIDES is intentionally unquoted so multiple Hydra overrides split into separate args.
srun $PROJECT_DIR/.venv/bin/python -m finetune \
    run_id="${RUN_ID}" \
    loss="${CRITERION}" \
    loss.weight="${LAMBDA}" \
    model="${MODEL}" \
    ${EXTRA_OVERRIDES}

# If we trained with LoRA, materialize a merged HF checkpoint for downstream eval.
MODEL_DIR="${PROJECT_DIR}/models/${RUN_ID}"
if [ -f "${MODEL_DIR}/adapter_config.json" ]; then
    echo "Merging LoRA adapter for ${RUN_ID} at $(date)"
    $PROJECT_DIR/.venv/bin/python -m finetune.merge \
        "${MODEL_DIR}" --output "${MODEL_DIR}-merged"
fi

echo "Finished finetuning of ${RUN_ID} at $(date)"