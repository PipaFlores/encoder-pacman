#!/bin/bash
#SBATCH --partition=gputest
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch

srun python ae_pytorch_comparative_analysis.py

module load tensorflow

srun python ae_keras_comparative_analysis.py