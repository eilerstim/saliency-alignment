#!/bin/bash
#SBATCH --account=aa013
#SBATCH --job-name=counting
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Counting-correlation analysis (paper Section 5.3) for one model:
#
#   sbatch scripts/cscs/arr_counting.sh MODEL_NAME [is_hub_model]
#
# MODEL_NAME is a run id under models/ or, with "true", a hub id. Questions
# come from the cat_bench repo next to this one (CAT_BENCH_DATA to override).
# Results go to results/counting/<MODEL_NAME>/.
set -euo pipefail
mkdir -p logs

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_NAME="$1"
if [ "${2:-false}" = "true" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="${PROJECT_DIR}/models/${MODEL_NAME}"
fi
[ -d "${MODEL_PATH}-merged" ] && MODEL_PATH="${MODEL_PATH}-merged"
DATA="${CAT_BENCH_DATA:-${PROJECT_DIR}/../cat_bench/data/test.json}"

source ./scripts/cscs/env.sh

echo "Counting analysis of ${MODEL_NAME} at $(date)"
srun --environment=saliency "$PROJECT_DIR/.venv/bin/python" -m align_eval.counting \
    --model "${MODEL_PATH}" --data "${DATA}" \
    --out "results/counting/${MODEL_NAME//\//__}"
echo "Finished counting analysis at $(date)"
