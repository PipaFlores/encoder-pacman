#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### AEON/KERAS MODELS
module load tensorflow

# Define arrays for different parameters
MODELS=("DRNN" "ResNet" "DCNN")
SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds")
FEATURES_SETS=("Pacman" "Pacman_Ghosts")

N_EPOCHS=500
CONTEXT=20
VALIDATION_SPLIT=0.3
LOGGING_COMMENT="sorted distances"

# Loop through models, sequence types, and feature sets
for MODEL in "${MODELS[@]}"; do
    for SEQ_TYPE in "${SEQUENCE_TYPES[@]}"; do
        for FEATURES in "${FEATURES_SETS[@]}"; do
            srun python keras_pacman.py --model "$MODEL" \
            --sequence-type "$SEQ_TYPE" \
            --n-epochs "$N_EPOCHS" \
            --features "$FEATURES" \
            --context "$CONTEXT" \
            --validation-split "$VALIDATION_SPLIT" \
            --logging-comment "$LOGGING_COMMENT"
        done
    done
done
