#!/bin/bash
#SBATCH --job-name=saliency-alignment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:05:00
#SBATCH --gpus=a100_80gb:4
#SBATCH --mail-type=END,FAIL

source scripts/euler/env.sh

export CRITERION=alignment
export MODEL=llava-1.5-7b
export LAMBDA=0.5

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_IB_DISABLE=1

echo "Beginning finetuning of $MODEL with $CRITERION loss (w=$LAMBDA) at $(date)"

srun $PROJECT_DIR/.venv/bin/python -m finetune \
    loss=$CRITERION \
    loss.weight=$LAMBDA \
    model=$MODEL

echo "Finished finetuning at $(date)"
