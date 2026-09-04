#!/bin/bash
# Convenience wrapper for running the benchmark from WSL/Codex while using
# the Windows Conda environment that contains the project ML dependencies.
set -euo pipefail

PYTHON_EXE="/mnt/c/Users/Localadmin_pabflore/miniconda3/envs/pacman_encoder/python.exe"

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Could not find pacman_encoder Python at: $PYTHON_EXE" >&2
    exit 1
fi

# Forward every user-provided argument to the Python benchmark runner.
"$PYTHON_EXE" hpc/train_benchmark_autoencoders.py "$@"
