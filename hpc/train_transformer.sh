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


module load pytorch
 
## ALL OPTIONS
FEATURE_SETS=("Pacman" "Pacman_Ghosts" "Ghost_Distances")
SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds" "pacman_attack")

## SPECIFIC (COMMENT OUT ABOVE)

# FEATURE_SETS=("Pacman" "Pacman_Ghosts")
# FEATURE_SETS=("Pacman_Ghosts")
# FEATURE_SETS=("Ghost_Distances")

# SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds")
# SEQUENCE_TYPES=("pacman_attack")

EMBEDDER="Transformer"
CLUSTERER="hdbscan"
REDUCER="umap"

N_EPOCHS=500
LATENT_SPACE=128
BATCH_SIZE=32
VALIDATION_SPLIT=0.3
CONTEXT=20
NORMALIZATION="global"
LOGGING_COMMENT="First transformer run"
EXTRA_FLAGS=(
    --using-hpc
    --verbose
)

# Test run
srun python train_model.py \
    --test-dataset \
    --embedder Transformer \
    --latent-space 128 \
    --validation-split 0.3 \
    --normalization "global" \
    --logging-comment "TestDataset" \
    --using-hpc \
    --verbose

# for FEATURES in "${FEATURE_SETS[@]}"; do
#     for SEQ_TYPE in "${SEQUENCE_TYPES[@]}"; do
#         FEATURE_FLAGS=("${EXTRA_FLAGS[@]}")
#         if [[ "$FEATURES" == "Ghost_Distances" ]]; then
#             FEATURE_FLAGS+=("--sort-ghost-distances")
#         fi

#         srun python train_model.py \
#             --sequence-type "$SEQ_TYPE" \
#             --feature-set "$FEATURES" \
#             --embedder "$EMBEDDER" \
#             --clusterer "$CLUSTERER" \
#             --reducer "$REDUCER" \
#             --n-epochs "$N_EPOCHS" \
#             --latent-space "$LATENT_SPACE" \
#             --batch-size "$BATCH_SIZE" \
#             --validation-split "$VALIDATION_SPLIT" \
#             --context "$CONTEXT" \
#             --normalization "$NORMALIZATION" \
#             --logging-comment "$LOGGING_COMMENT" \
#             "${FEATURE_FLAGS[@]}"
#     done
# done
