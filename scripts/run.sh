#!/bin/bash
#SBATCH --job-name=saliency-alignment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpumem:64g
#SBATCH --mail-type=END,FAIL

echo "Beginning finetuning at $(date)"

source scripts/env.sh

uv run -m finetune