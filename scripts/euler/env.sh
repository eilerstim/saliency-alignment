#!/bin/bash
# This script sets up the virtual environment
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# You may need to adjust the module load commands based on your environment.

set -euo pipefail

dir="$SCRATCH/saliency-alignment"
export HF_HOME="$dir/.hf/"
export UV_CACHE_DIR="$dir/.uv/"
export UV_PROJECT_ENVIRONMENT="$dir/.venv"

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --quiet

source "$UV_PROJECT_ENVIRONMENT/bin/activate"

echo "Environment ready at $UV_PROJECT_ENVIRONMENT"
