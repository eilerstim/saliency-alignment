#!/bin/bash
# This script sets up the virtual environment
# Designed for use on the CSCS Alps clariden cluster, in the saliency environment.

export PIP_CACHE_DIR="$PROJECT_DIR/.pip-cache"

python -m pip install -e "${PROJECT_DIR}" --no-build-isolation --quiet

echo "Environment ready at $PROJECT_DIR/.venv"
