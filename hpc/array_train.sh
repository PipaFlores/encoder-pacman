#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --array=1-5 ## Sweeping over different latent dimensions (5 for transformer, 7 for LSTM)
#SBATCH --account=project_2012947   # replace <project> with your CSC project, e.g. project_2001234
#SBATCH --nodes=1            # replace <N> with the number of nodes to run on
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  # Mahti has 128 CPU cores per node, Puhti has 40
#SBATCH --mem-per-cpu=32000 # 8000 is enough but generalist models might need much more (used 64 gb before)
#SBATCH --gres=gpu:v100:1
#SBATCH --time=4:00:00

# Set the number of threads based on cpus-per-task
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} # This is for OpemMP multiprocessing but I use multiprocessing.Pool in python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


module load python-pytorch/2.13
############################################################## NOTES ###################################################
LOGGING_COMMENT="latent dimension sweep -- all features" 
########################################################################################################################
## ALL OPTIONS
# FEATURE_SETS=("Pacman" "Pacman_Ghosts" "Ghost_Distances" "Experimental2")
# SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds" "pacman_attack")

## SPECIFIC (COMMENT OUT ABOVE)

# FEATURE_SETS=("Experimental2")
# FEATURE_SETS=("Experimental2" "Pacman_Ghosts")
# FEATURE_SETS=("Experimental2")
# FEATURE_SETS=("Pacman_Ghosts")
# FEATURE_SETS=("Ghost_Distances")
FEATURE_SETS=("all_features")

# SEQUENCE_TYPES=("first_5_seconds" "last_5_seconds")
SEQUENCE_TYPES=("pacman_attack")
# SEQUENCE_TYPES=("sliding_window")
# SEQUENCE_TYPES=("fixed_blocks")
# SEQUENCE_TYPES=("first_5_seconds" "pacman_attack")

EMBEDDER="Transformer" # Or "Transformer", or "None"
CLUSTERER="hdbscan"
REDUCER="umap"

N_EPOCHS=100
# LATENT_SPACE=32
# LATENT_SPACE_LIST=(2 16 32 64 128 256 512) # for LSTM
LATENT_SPACE_LIST=(32 64 128 256 512) # for transformer

## Select latent space size based on SLURM array task id (array indices start at 1)
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    # convert to zero-based index
    idx=$((SLURM_ARRAY_TASK_ID-1))
    LATENT_SPACE=${LATENT_SPACE_LIST[$idx]}
else
    # fallback to first element if not running under SLURM
    LATENT_SPACE=${LATENT_SPACE_LIST[0]}
fi

BATCH_SIZE=32
VALIDATION_SPLIT=0.3
CONTEXT=20
FILTER_BY_PILL="None"  # or "None"
NORMALIZATION="global"
EXTRA_FLAGS=(
    --using-hpc
    --verbose
    # --elementwise-masking   # Always use for LSTM (or better said, for first 5 seconds where there is a LOT of infs)
)


# Test run
# srun python train_model.py \
#     --test-dataset \
#     --embedder Transformer \
#     --latent-space 128 \
#     --validation-split 0.3 \
#     --normalization "global" \
#     --logging-comment "TestDataset" \
#     --using-hpc \
#     --verbose

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
            --n-epochs "$N_EPOCHS" \
            --latent-space "$LATENT_SPACE" \
            --batch-size "$BATCH_SIZE" \
            --validation-split "$VALIDATION_SPLIT" \
            --context "$CONTEXT" \
            --filter-by-pill "$FILTER_BY_PILL" \
            --normalization "$NORMALIZATION" \
            --logging-comment "$LOGGING_COMMENT" \
            "${FEATURE_FLAGS[@]}"
    done
done