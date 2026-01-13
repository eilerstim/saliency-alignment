#!/bin/bash
# This script sets up the virtual environment
# Designed for use on the CSCS Alps clariden cluster, in the saliency environment.

export COUNT_DIR="$PROJECT_DIR/../grounding-vlms"
export RESET_ENV=${RESET_ENV:-0}
export VENV_DIR="$COUNT_DIR/.count_venv"

# If reset_env is set, delete existing virtual environment
if [ "$RESET_ENV" = "1" ]; then
    rm -rf "$VENV_DIR"
fi

# If venv does not exist, create it
if [ ! -d "$VENV_DIR" ]; then
    RESET_ENV=1
    uv venv "$VENV_DIR" --python 3.12 --system-site-packages
    echo "Environment created at $VENV_DIR"
else
    echo "Environment already exists at $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install package if environment was reset or newly created
if [ "$RESET_ENV" = "1" ]; then
    uv pip install -e "$COUNT_DIR" --no-build-isolation --quiet
fi

echo "Environment ready at $COUNT_DIR/.count_venv"
