# List available just commands
[private]
list:
    @just --list --unsorted

# Run the training script using sbatch
run:
    sbatch scripts/cscs/run.sh

# Follow the output logs of the latest run
[group('logs')]
follow-out:
    -@tail -F "$(ls -1t logs/*.out | head -n1)"

# Follow the error logs of the latest run
[group('logs')]
follow-err:
    -@tail -F "$(ls -1t logs/*.err | head -n1)"

# Format and lint
[group('code quality')]
ruff:
    ruff format .
    ruff check . --fix

# Type checking
[group('code quality')]
mypy:
    mypy .

# Watch current user's squeue output
[group('slurm')]
squeue:
    -watch squeue --me

# Simple interactive session
[group('slurm')]
srun:
    srun --pty --account=aa013 --environment=saliency bash

# GPU interactive session
[group('slurm')]
srun-gpu:
    srun --pty --account=aa013 --environment=saliency --gpus-per-node=4 --cpus-per-task=16 bash

# Delete logs directory
[group('clean')]
clean-logs:
    rm -rf logs/*

# Delete outputs directory
[group('clean')]
clean-outputs:
    rm -rf outputs/*

# Delete logs and outputs
[group('clean')]
clean: clean-logs clean-outputs
