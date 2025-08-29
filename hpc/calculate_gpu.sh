#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


### PYTORCH MODELS
# module load pytorch

# srun python pytorch_pacman.py --sequence-type 'first_5_seconds' --n-epochs 1000
# srun python pytorch_pacman.py --sequence-type 'last_5_seconds' --n-epochs 1000
# srun python pytorch_pacman.py --sequence-type 'whole_level' --n-epochs 1000

### AEON/KERAS MODELS
module load tensorflow

srun python keras_pacman.py --model 'DRNN' --sequence-type 'first_5_seconds' --n-epochs 500

srun python keras_pacman.py --model 'ResNet' --sequence-type 'first_5_seconds' --n-epochs 500

srun python keras_pacman.py --model 'DRNN' --sequence-type 'whole_level' --n-epochs 500

srun python keras_pacman.py --model 'ResNet' --sequence-type 'whole_level' --n-epochs 500

