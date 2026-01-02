#!/bin/bash
#SBATCH --job-name=saliency-alignment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=a100_80gb:4
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Beginning finetuning at $(date)"

source scripts/env.sh

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks

uv run -m finetune

echo "Finished finetuning at $(date)"
