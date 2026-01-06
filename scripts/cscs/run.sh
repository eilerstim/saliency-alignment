#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=saliency
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
#SBATCH --array=0-2

source ./scripts/cscs/env.sh

lambda_values=(0.3 0.0 2.0)
export LAMBDA=${lambda_values[$SLURM_ARRAY_TASK_ID]}


export CRITERION=alignment
export MODEL=llava-1.5-7b

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
