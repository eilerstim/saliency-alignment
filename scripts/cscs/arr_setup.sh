#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# One-shot creation of $PROJECT_DIR/.venv INSIDE the saliency container (which
# provides python3.12), so parallel training jobs never race to build it. env.sh
# is idempotent -- it skips creation if the venv already exists -- so re-running
# this is safe. Set RESET_ENV=1 to force a clean rebuild.
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

srun --environment=saliency bash -c 'source ./scripts/cscs/env.sh'
echo "venv ready at ${PROJECT_DIR}/.venv"
