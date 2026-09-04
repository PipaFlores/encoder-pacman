#!/bin/bash
# Run this script on the HPC scheduler. It executes the same benchmark
# configuration for each dataset listed in DATASETS below.
#SBATCH --partition=gpumedium
#SBATCH --account=project_2012947
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=8  # The product should be 72 if requesting 1 GPU per node
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node
#SBATCH --time=04:00:00

# Set the number of threads based on cpus-per-task
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Keep the dataset list small enough for one scheduled job, or split it into arrays.
DATASETS=("BasicMotions" "PenDigits")

for DATASET in "${DATASETS[@]}"; do
    # Shared across both passes so their results land in the same results.csv/json.
    RUN_ID=$(date +%Y%m%d-%H%M%S)

    # Pass 1: repo pytorch architectures plus the no-training baselines.
    # UMAP reduces to --clustering-dim before HDBSCAN; plots are always 2D.
    # Transformer benchmark follows the repo masked-imputation objective.
    module load python-pytorch/2.13
    python train_benchmark_autoencoders.py \
    --dataset "$DATASET" \
    --run-id "$RUN_ID" \
    --architectures LSTM Transformer VanillaVAE TimeVAE UMAP RandomProjection \
    --latent-space 64 \
    --n-epochs 100 \
    --batch-size 32 \
    --validation-split 0.2 \
    --clustering-dim 2 \
    --transformer-masking-ratio 0.15 \
    --transformer-mean-mask-length 3 \
    --transformer-mask-mode separate \
    --transformer-mask-distribution geometric \
    --continue-on-error \
    --verbose
    module unload python-pytorch/2.13

    # Pass 2: aeon's Keras/tensorflow-backed deep clusterers.
    module load python-tensorflow/2.21
    python train_benchmark_autoencoders.py \
    --dataset "$DATASET" \
    --run-id "$RUN_ID" \
    --architectures DRNN DCNN ResNet \
    --latent-space 64 \
    --n-epochs 100 \
    --batch-size 32 \
    --validation-split 0.2 \
    --clustering-dim 2 \
    --continue-on-error \
    --verbose
    module unload python-tensorflow/2.21
done
