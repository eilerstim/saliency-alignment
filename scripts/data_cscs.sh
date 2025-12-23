#!/bin/bash
#!/bin/bash
#SBATCH --account=a163
#SBATCH --time=12:00:00
#SBATCH --job-name=data_download
#SBATCH --output=/iopsstor/scratch/cscs/teilers/saliency-alignment/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/teilers/saliency-alignment/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=/users/teilers/saliency-alignment.toml
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning downloading data at $(date)"

cd /iopsstor/scratch/cscs/teilers/saliency-alignment

python -m finetune.data.coconut.download

echo "Finished downloading data at $(date)"
