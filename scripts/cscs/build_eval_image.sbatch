#!/bin/sh
#SBATCH --account=large-sc-2
#SBATCH --job-name=eval_docker_build
#SBATCH --output=logs/docker_build_%j.out
#SBATCH --error=logs/docker_build_%j.err
#SBATCH --time=01:00:00

set -eu

mkdir -p logs

podman build \
  --build-arg PROJECT_DIR="$SCRATCH/saliency-alignment" \
  -t eval:0.1-py3 \
  -f slurm/Dockerfile.eval \
  .

rm -f slurm/eval.sqsh
enroot import -o slurm/eval.sqsh podman://eval:0.1-py3