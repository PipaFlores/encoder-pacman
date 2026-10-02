#!/bin/bash
#SBATCH --job-name=render_gifs
#SBATCH --partition=small
#SBATCH --account=project_2012947
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4000

# Render the hover GIFs for the augmented interactive plot into gifs/.
# All arguments go straight to render_gifs.py, e.g.
#   sbatch render_gifs.sh --config experiments/general_training.yaml
#   sbatch render_gifs.sh --sequence-type pacman_attack --context 20
# Already-rendered GIFs are skipped, so resubmitting after a timeout just continues.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load python-pytorch/2.13
srun python render_gifs.py "$@"
