#!/bin/bash
# Submit an experiment as one or more SLURM arrays, one task per configuration.
#
#   ./submit_experiment.sh experiments/latent_sweep_transformer.yaml
#
# The array size comes from the experiment file rather than from a hand-edited
# `#SBATCH --array` line, which is the thing that used to need updating (and the arithmetic
# that used to be done in bash) every time a sweep changed shape.
#
# An experiment that mixes torch and aeon/keras embedders is submitted as one array per
# module, each with an explicit index list - the array equivalent of the two sequential
# passes in train_benchmark_autoencoders.sh. All of them share one run id, so the whole
# experiment still collects into a single results directory.
set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: $0 <experiment.yaml> [run-id]" >&2
    echo "  run-id: reuse an existing run directory, e.g. when re-submitting a group" >&2
    echo "          whose sbatch failed. Defaults to a fresh timestamp." >&2
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

# One array per environment module. Most experiments produce a single line here; an
# experiment spanning torch and keras embedders produces two.
#
# A failed sbatch must not abort the loop: with several groups, stopping at the first
# failure would leave the experiment half-submitted with no record of which half. Each
# outcome is tracked and reported at the end instead.
FAILED_GROUPS=()
SUBMITTED=0
GROUP_COUNT=$(python run_experiment.py --config "$CONFIG" --module-groups | grep -c .)

while IFS=$'\t' read -r MODULE INDICES; do
    # Strip a trailing CR: harmless on the cluster, but Python on Windows prints CRLF
    # and the carriage return would end up inside the sbatch --array value.
    MODULE="${MODULE%$'\r'}"; INDICES="${INDICES%$'\r'}"
    [[ -z "$MODULE" ]] && continue
    echo "Submitting ${MODULE} for indices ${INDICES}"

    # Slurm counts every array task as its own job against the project limit, so an
    # experiment can ask to be one sequential job instead - see submit_mode().
    MODE_FLAGS=()
    SCRIPT_ARGS=("$CONFIG")
    if [[ "$SUBMIT_MODE" == "sequential" ]]; then
        # As a script argument, not --export: that flag splits its value on commas,
        # so an index list like 0,1,2 arrived at the job as just "0".
        SCRIPT_ARGS+=("$INDICES")
        LOG_PATTERN="logs/$NAME/%j.out"
    else
        # "array" or "array%N"; the throttle caps concurrently running tasks.
        THROTTLE="${SUBMIT_MODE#array}"
        MODE_FLAGS=(--array="${INDICES}${THROTTLE}")
        LOG_PATTERN="logs/$NAME/%A_%a.out"
    fi

    # SLURM_FLAGS is unquoted on purpose: it is a list of flags, not a single argument.
    # shellcheck disable=SC2086
    if sbatch \
        --job-name="$NAME" \
        "${MODE_FLAGS[@]}" \
        --output="$LOG_PATTERN" \
        --export=ALL,EXPERIMENT_MODULE="$MODULE",EXPERIMENT_RUN_ID="$RUN_ID" \
        $SLURM_FLAGS \
        run_experiment.sh "${SCRIPT_ARGS[@]}"
    then
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ^ sbatch failed for ${MODULE} (indices ${INDICES})" >&2
        FAILED_GROUPS+=("${MODULE}\t${INDICES}")
    fi
done < <(python run_experiment.py --config "$CONFIG" --module-groups)

echo
if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "${#FAILED_GROUPS[@]} group(s) failed to submit:" >&2
    for GROUP in "${FAILED_GROUPS[@]}"; do
        printf "  %b\n" "$GROUP" >&2
    done
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
    echo "(already-submitted groups would be submitted again - cancel them first with" >&2
    echo " scancel, or sbatch just the failed group by hand.)" >&2
    exit 1
fi

echo "Submitted $SUBMITTED job(s)/array(s) for run $RUN_ID."

# run_experiment.py collects on its own whenever one process ran every configuration.
# That is only true of a sequential submission with a single module group; an array
# task sees just its own result, and a two-module experiment is split across two jobs.
if [[ "$SUBMIT_MODE" == "sequential" && "$GROUP_COUNT" -eq 1 ]]; then
    echo "Results are merged into results.csv automatically when the job finishes."
else
    echo "Each task writes its own results/<index>.json. Once they have all finished,"
    echo "merge them into results.csv with:"
    echo "  python run_experiment.py --config $CONFIG --run-id $RUN_ID --collect"
fi
