#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=saliency
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning finetuning at $(date)"

source ./scripts/cscs/env.sh

export CRITERION=alignment
export LAMBDA=0.5
export MODEL=llava-v1.6-mistral-7b-hf

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun $PROJECT_DIR/.venv/bin/python -m finetune \
    loss=$CRITERION \
    loss.weight=$LAMBDA \
    model=$MODEL

echo "Finished finetuning at $(date)"
