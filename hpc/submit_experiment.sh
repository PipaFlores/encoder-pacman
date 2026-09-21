#!/bin/bash
# Submit an experiment as one SLURM array, one task per configuration.
#
#   ./submit_experiment.sh experiments/latent_sweep_transformer.yaml
#
# The array size comes from the experiment file rather than from a hand-edited
# `#SBATCH --array` line, which is the thing that used to need updating (and the arithmetic
# that used to be done in bash) every time a sweep changed shape.
set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: $0 <experiment.yaml> [run-id]" >&2
    echo "  run-id: reuse an existing run directory, e.g. when re-submitting after" >&2
    echo "          a failed sbatch. Defaults to a fresh timestamp." >&2
    exit 2
fi

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HPC_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "No such experiment file: $CONFIG" >&2
    exit 2
fi

# Expanding the grid is stdlib-only (see experiment.py), so this stays cheap on a login node.
N=$(python run_experiment.py --config "$CONFIG" --count)
SLURM_FLAGS=$(python run_experiment.py --config "$CONFIG" --slurm-flags)
SUBMIT_MODE=$(python run_experiment.py --config "$CONFIG" --submit-mode)
NAME=$(basename "${CONFIG%.*}")
RUN_ID="${2:-$(date +%Y%m%d-%H%M%S)}"

if [[ "$N" -lt 1 ]]; then
    echo "Experiment expanded to $N configurations, nothing to submit." >&2
    exit 1
fi

mkdir -p "logs/$NAME"

echo "Experiment:     $NAME"
echo "Configurations: $N"
echo "Run id:         $RUN_ID"
echo "sbatch flags:   ${SLURM_FLAGS:-<none, using run_experiment.sh header>}"
echo "Submit mode:    $SUBMIT_MODE"
python run_experiment.py --config "$CONFIG" --dry-run
echo

# Strip a trailing CR: harmless on the cluster, but Python on Windows prints CRLF and the
# carriage return would end up inside the sbatch --export value.
MODULE=$(python run_experiment.py --config "$CONFIG" --module)
MODULE="${MODULE%$'\r'}"

echo "Submitting ${MODULE} for ${N} configuration(s)"

# Slurm counts every array task as its own job against the project limit, so an
# experiment can ask to be one sequential job instead - see submit_mode().
MODE_FLAGS=()
if [[ "$SUBMIT_MODE" == "sequential" ]]; then
    # No index list: run_experiment.sh runs the whole expansion when given none.
    LOG_PATTERN="logs/$NAME/%j.out"
else
    # "array" or "array%N"; the throttle caps concurrently running tasks.
    THROTTLE="${SUBMIT_MODE#array}"
    MODE_FLAGS=(--array="0-$((N - 1))${THROTTLE}")
    LOG_PATTERN="logs/$NAME/%A_%a.out"
fi

# SLURM_FLAGS is unquoted on purpose: it is a list of flags, not a single argument.
# shellcheck disable=SC2086
if ! sbatch \
    --job-name="$NAME" \
    "${MODE_FLAGS[@]}" \
    --output="$LOG_PATTERN" \
    --export=ALL,EXPERIMENT_MODULE="$MODULE",EXPERIMENT_RUN_ID="$RUN_ID" \
    $SLURM_FLAGS \
    run_experiment.sh "$CONFIG"
then
    echo
    echo "sbatch failed." >&2
    echo >&2
    echo "A 'job submit limit' or accounting/QOS rejection usually means the project has" >&2
    echo "too many jobs queued or running - and Slurm counts EVERY array task as a job." >&2
    echo "Check the limit with: sacctmgr show qos format=name,maxjobspu,maxsubmitjobspu" >&2
    echo "then set array_throttle: N, or use_array: false, in the experiment slurm block." >&2
    echo >&2
    echo "sbatch errors mentioning sinfo, argos, or a socket timeout are instead the" >&2
    echo "scheduler being unreachable, not a problem with the experiment. Check with:" >&2
    echo "  sinfo -p \$(python run_experiment.py --config $CONFIG --slurm-flags | tr ' ' '\\n' | sed -n 's/^--partition=//p')" >&2
    echo >&2
    echo "To retry into the SAME run directory once the scheduler responds:" >&2
    echo "  $0 $CONFIG $RUN_ID" >&2
    exit 1
fi

echo "Submitted run $RUN_ID."

# run_experiment.py collects on its own whenever one process ran every configuration,
# which is true of a sequential submission but not of an array - an array task only ever
# sees its own result.
if [[ "$SUBMIT_MODE" == "sequential" ]]; then
    echo "Results are merged into results.csv automatically when the job finishes."
else
    echo "Each task writes its own results/<index>.json. Once they have all finished,"
    echo "merge them into results.csv with:"
    echo "  python run_experiment.py --config $CONFIG --run-id $RUN_ID --collect"
fi
