#!/bin/bash
# Run this script on the HPC scheduler. It executes one configuration of an experiment file
# per array task, or the whole expansion in order when submitted without an array.
# Not usually submitted by hand - use submit_experiment.sh, which sizes the array, applies
# the experiment's own slurm block, and selects the right environment module.
#
#   ./submit_experiment.sh experiments/latent_sweep_transformer.yaml
#
# The values below are fallbacks only: sbatch command-line flags override the header, and
# submit_experiment.sh passes whatever the experiment file's `slurm:` block specifies, so
# this header should not need editing per experiment.
#SBATCH --partition=gpumedium
#SBATCH --account=project_2012947
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72  # 72 cores per reserved GPU; the product is 72 per GPU
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node
# No --mem/--mem-per-cpu on purpose: Roihu hands over the reserved GH200's full 217 GiB
# (95 GiB HBM3 + 122 GiB LPDDR5) automatically, and that 217 GiB is also the hard QOS
# ceiling - 8 x 32000 MiB asked for 250 GiB and was rejected as a size-limit violation.
#SBATCH --time=04:00:00

# Set the number of threads based on cpus-per-task
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The module is a property of the experiment, not of this script; submit_experiment.sh
# exports the choice (see experiment_module() in experiment.py).
module load "${EXPERIMENT_MODULE:-python-pytorch/2.13}"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}" || exit 1

CONFIG="$1"
# Optional comma-separated index list for a sequential (non-array) submission.
# Passed as an argument rather than an environment variable because sbatch --export
# splits its value on commas: "EXPERIMENT_INDICES=0,1,2" arrived as just "0".
INDEX_LIST="${2:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: sbatch run_experiment.sh <experiment.yaml> [index,list]" >&2
    exit 2
fi

# Three ways in, in priority order:
#  - an array task runs exactly the one configuration at its index;
#  - a sequential submission runs the index list it was given, in order (this is how an
#    experiment reaches a partition whose job limit will not tolerate an array);
#  - neither, and the job runs the whole expansion.
INDEX_FLAGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    INDEX_FLAGS=(--index "$SLURM_ARRAY_TASK_ID")
elif [[ -n "$INDEX_LIST" ]]; then
    INDEX_FLAGS=(--indices "$INDEX_LIST")
fi

echo "run_experiment.sh: config=$CONFIG selection=${INDEX_FLAGS[*]:-<whole expansion>}"

# Every task of one array must agree on the output directory, so the run id is decided
# at submit time rather than per job.
RUN_ID_FLAGS=()
if [[ -n "${EXPERIMENT_RUN_ID:-}" ]]; then
    RUN_ID_FLAGS=(--run-id "$EXPERIMENT_RUN_ID")
fi

srun python run_experiment.py \
    --config "$CONFIG" \
    "${INDEX_FLAGS[@]}" \
    "${RUN_ID_FLAGS[@]}" \
    --continue-on-error
