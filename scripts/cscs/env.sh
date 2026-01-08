#!/bin/bash
# This script sets up the virtual environment
# Designed for use on the CSCS Alps clariden cluster, in the saliency environment.

export RESET_ENV=${RESET_ENV:-0}

# If reset_env is set, delete existing virtual environment
if [ "$RESET_ENV" == "1" ]; then
    rm -rf "$PROJECT_DIR/.venv"
fi

# If venv does not exist, create it
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    RESET_ENV=1
    python3.12 -m venv "$PROJECT_DIR/.venv" --system-site-packages
    echo "Environment created at $PROJECT_DIR/.venv"
else
    echo "Environment already exists at $PROJECT_DIR/.venv"
fi

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

if [ "$RESET_ENV" == "1" ]; then
    python -m pip install -e "${PROJECT_DIR}" --no-build-isolation --quiet
fi

echo "Environment ready at $PROJECT_DIR/.venv"
