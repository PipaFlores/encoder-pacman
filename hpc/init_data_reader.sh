#!/bin/bash
#SBATCH --partition=small
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=32000
#SBATCH --time=02:00:00

# Set the number of threads based on cpus-per-task
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} # This is for OpemMP multiprocessing but I use multiprocessing.Pool in python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


module load python-pytorch/2.13

srun python init_data_reader.py