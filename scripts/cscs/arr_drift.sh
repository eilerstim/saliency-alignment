#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=drift
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
# Parameter-drift analysis (paper Appendix E) of fine-tuned checkpoints against
# the base model: per-component, per-layer and per-head relative drift, Gini
# concentration and cross-run top-k Jaccard. Runs on CPU.
#
#   sbatch scripts/cscs/arr_drift.sh OUT_NAME CKPT [CKPT ...]
#
# Checkpoint paths are relative to the repo root; a LoRA run passes its
# "<run>-merged" directory. The report is written to results/drift/OUT_NAME.txt.
set -euo pipefail
mkdir -p logs results/drift

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_NAME="$1"
shift

source ./scripts/cscs/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Drift analysis of $* at $(date)"
# Top-5 layers per projection (q, v) = the 10 layer-projection cells, top-8 heads
srun --environment=saliency \
    "$PROJECT_DIR/.venv/bin/python" scripts/python/compare_drift.py \
    --base llava-hf/llava-1.5-7b-hf \
    --per-layer --per-head --per-head-layers-k 5 --per-head-top-k 8 \
    --save-reports "$@" > "results/drift/${OUT_NAME}.txt"
echo "Finished drift analysis at $(date); report in results/drift/${OUT_NAME}.txt"
