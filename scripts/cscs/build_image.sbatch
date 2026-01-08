#!/bin/sh
#SBATCH --account=a163
#SBATCH --job-name=docker_build
#SBATCH --output=logs/docker_build_%j.out
#SBATCH --error=logs/docker_build_%j.err
#SBATCH --time=01:00:00

set -eu

mkdir -p logs

podman build \
  --build-arg PROJECT_DIR="$SCRATCH/saliency-alignment" \
  -t saliency:0.1-py3 \
  -f slurm/Dockerfile \
  .

rm -f slurm/saliency.sqsh
enroot import -o slurm/saliency.sqsh podman://saliency:0.1-py3