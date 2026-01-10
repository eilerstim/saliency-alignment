#!/bin/bash
# This script sets up the virtual environment
# Designed for use on the CSCS Alps clariden cluster, in the saliency environment.

export COUNT_DIR="$PROJECT_DIR/../grounding-vlms"
export RESET_ENV=${RESET_ENV:-0}

# If reset_env is set, delete existing virtual environment
if [ "$RESET_ENV" == "1" ]; then
    rm -rf "$COUNT_DIR/.count_venv"
fi

# If venv does not exist, create it
if [ ! -d "$COUNT_DIR/.count_venv" ]; then
    RESET_ENV=1
    python3.12 -m venv "$COUNT_DIR/.count_venv" --system-site-packages
    echo "Environment created at $COUNT_DIR/.count_venv"
else
    echo "Environment already exists at $COUNT_DIR/.count_venv"
fi

# Activate the virtual environment
source "$COUNT_DIR/.count_venv/bin/activate"
if [ "$RESET_ENV" == "1" ]; then
    python -m pip install -e "${COUNT_DIR}" --no-build-isolation --quiet
fi

echo "Environment ready at $COUNT_DIR/.count_venv"
