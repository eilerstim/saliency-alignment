#!/bin/bash
#SBATCH --job-name=data_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

echo "Beginning downloading data at $(date)"

source scripts/env.sh

uv run -m finetune.data.grand.download

echo "Finished downloading data at $(date)"
