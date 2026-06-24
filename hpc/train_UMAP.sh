#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


module load pytorch/2.7
 
## ALL OPTIONS
# FEATURE_SETS=("Pacman" "Pacman_Ghosts" "Ghost_Distances")
# SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds" "pacman_attack")

## SPECIFIC (COMMENT OUT ABOVE)

FEATURE_SETS=("Experimental2")
# FEATURE_SETS=("all_features")
# FEATURE_SETS=("Pacman" "Pacman_Ghosts")
# FEATURE_SETS=("Pacman_Ghosts")
# FEATURE_SETS=("Ghost_Distances")

# SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds")
SEQUENCE_TYPES=("pacman_attack")
# SEQUENCE_TYPES=("fixed_blocks")

EMBEDDER="None"
CLUSTERER="hdbscan"
REDUCER="umap"

CONTEXT=20
FILTER_BY_PILL="None"

REDUCER_COMPONENTS=2
UMAP_NEIGHBORS=15
UMAP_MIN_DIST=0.1
UMAP_METRIC="euclidean"

NORMALIZATION="global"
LOGGING_COMMENT="testing Umap embedding"
EXTRA_FLAGS=(
    --using-hpc
    --verbose
    --elementwise-masking
)


for FEATURES in "${FEATURE_SETS[@]}"; do
    for SEQ_TYPE in "${SEQUENCE_TYPES[@]}"; do
        FEATURE_FLAGS=("${EXTRA_FLAGS[@]}")
        if [[ "$FEATURES" == "Ghost_Distances" ]]; then
            FEATURE_FLAGS+=("--sort-ghost-distances")
        fi

        srun python train_model.py \
            --sequence-type "$SEQ_TYPE" \
            --feature-set "$FEATURES" \
            --embedder "$EMBEDDER" \
            --clusterer "$CLUSTERER" \
            --reducer "$REDUCER" \
            --reducer-components "$REDUCER_COMPONENTS" \
            --umap-neighbors "$UMAP_NEIGHBORS" \
            --umap-min-dist "$UMAP_MIN_DIST" \
            --umap-metric "$UMAP_METRIC" \
            --context "$CONTEXT" \
            --filter-by-pill "$FILTER_BY_PILL" \
            --normalization "$NORMALIZATION" \
            --logging-comment "$LOGGING_COMMENT" \
            "${FEATURE_FLAGS[@]}"
    done
done
