#!/bin/bash
# Run this script on the HPC scheduler. It executes the same benchmark
# configuration for each dataset listed in DATASETS below.
# Keras/tensorflow-only variant of smoke_benchmark_autoencoders.sh, for
# isolating and re-testing the DRNN/DCNN/ResNet transpose fix without paying
# for the pytorch pass.
#SBATCH --partition=gputest
#SBATCH --account=project_2012947
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=8  # The product should be 72 if requesting 1 GPU per node
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node
#SBATCH --time=00:15:00

# Set the number of threads based on cpus-per-task
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Keep the dataset list small enough for one scheduled job, or split it into arrays.
DATASETS=("PenDigits" "NATOPS" "Worms")

for DATASET in "${DATASETS[@]}"; do
    RUN_ID=$(date +%Y%m%d-%H%M%S)

    # aeon's Keras/tensorflow-backed deep clusterers.
    module load python-tensorflow/2.21
    python train_benchmark_autoencoders.py \
    --dataset "$DATASET" \
    --run-id "$RUN_ID" \
    --architectures DRNN DCNN ResNet \
    --latent-space 64 \
    --n-epochs 2 \
    --batch-size 32 \
    --validation-split 0.2 \
    --clustering-dim 2 \
    --continue-on-error \
    --verbose
    module unload python-tensorflow/2.21
done
