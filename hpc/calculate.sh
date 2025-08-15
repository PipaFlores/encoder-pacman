#!/bin/bash
#SBATCH --partition=small
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --time=03:00:00
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch
srun python affinity_cal.py