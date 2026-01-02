#!/bin/bash
#SBATCH --account=a163
#SBATCH --time=12:00:00
#SBATCH --job-name=data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=saliency
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

source ./scripts/cscs/env.sh

echo "Beginning downloading data at $(date)"

python -m finetune.data.coconut.download \
    run_id=${SLURM_JOB_NAME}_${SLURM_JOB_ID}

echo "Finished downloading data at $(date)"
