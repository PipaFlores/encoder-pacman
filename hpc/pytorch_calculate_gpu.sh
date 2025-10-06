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
module load pytorch

FEATURE_SETS=("Ghost_Distances")
# ("Pacman" "Pacman_Ghosts" "Ghost_Distances")
SEQUENCE_TYPES=("pacman_attack")
#("first_5_seconds" "last_5_seconds")

for FEATURES in "${FEATURE_SETS[@]}"; do
    for SEQ_TYPE in "${SEQUENCE_TYPES[@]}"; do
        srun python pytorch_pacman.py --sequence-type "$SEQ_TYPE" --n-epochs 500 --features "$FEATURES" --context 20
    done
done

