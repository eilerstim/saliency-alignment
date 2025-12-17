#!/bin/bash
#SBATCH --account=a163
#SBATCH --time=12:00:00
#SBATCH --job-name=saliency-alignment
#SBATCH --output=/iopsstor/scratch/cscs/teilers/saliency-alignment/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/teilers/saliency-alignment/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=/users/teilers/saliency-alignment.toml
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled


echo "Beginning finetuning at $(date)"

cd /iopsstor/scratch/cscs/teilers/saliency-alignment

pip uninstall -y torchao

python -c "from transformers import AutoProcessor; print('ok')"

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks

python -m finetune

echo "Finished finetuning at $(date)"
