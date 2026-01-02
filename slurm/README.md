# SLURM Environment Setup

For running jobs on clusters using SLURM's environment modules, e.g. the CSCS clariden cluster, run the Dockerfile. A script to do so is provided in `scripts/cscs/build_image.sbatch`.

Once the image is built, copy the `saliency.toml` file from `scratch/saliency-alignment/slurm/` to your `.edf/` root directory. Make sure to adjust any paths in the `saliency.toml` file as necessary for your environment. Then, in your SLURM job submission scripts, specify the environment using the following directive:

```bash
#SBATCH --environment=saliency
```