"""Single entry point for Pacman PatternAnalysis experiments.

An experiment is a YAML file describing a `defaults` block and a `grid` to sweep over
(see hpc/experiments/). This script expands it into an ordered list of configurations and
runs them - all of them in sequence locally, or exactly one per SLURM array task:

    python run_experiment.py --config experiments/latent_sweep.yaml --dry-run
    python run_experiment.py --config experiments/latent_sweep.yaml --index 3
    python run_experiment.py --config experiments/latent_sweep.yaml --collect

Results land under hpc/runs/<experiment>/<run-id>/. Each configuration writes its own
results/<index>.json rather than a shared CSV, because array tasks finish concurrently and
would otherwise race each other's writes; `--collect` merges them into results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiment import (  # noqa: E402
    HPC_DIR,
    RunConfig,
    RunResult,
    expand,
    load_spec,
    module_groups,
    run_one,
    run_one_guarded,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Pacman PatternAnalysis experiment (a defaults block plus a grid).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment file (.yaml, or .json as a fallback).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only this configuration, by position in the expansion. This is what a SLURM array task passes.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help=(
            "Comma-separated configuration indices to run in sequence within this one "
            "process. Used instead of --index when an experiment is submitted without an "
            "array (see use_array in the slurm block)."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identifier for this execution of the experiment, used as the output subdirectory. "
            "Defaults to the SLURM array job id so every task of one array shares a directory, "
            "falling back to a timestamp off the cluster."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(HPC_DIR, "runs"),
        help="Directory that holds per-experiment output.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration key for every run in the expansion. Repeatable, e.g. --set n_epochs=2.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failing configuration as an error row and carry on, instead of stopping.",
    )

    # Inspection modes: these print and exit without running anything.
    parser.add_argument("--dry-run", action="store_true", help="Print the expansion and exit.")
    parser.add_argument("--count", action="store_true", help="Print the number of configurations and exit.")
    parser.add_argument(
        "--slurm-flags",
        action="store_true",
        help="Print the experiment's slurm block as sbatch flags and exit (used by submit_experiment.sh).",
    )
    parser.add_argument(
        "--module-groups",
        action="store_true",
        help=(
            "Print one line per environment module needed by this experiment, as "
            "<module><TAB><comma-separated indices>, and exit. submit_experiment.sh "
            "submits one array per line."
        ),
    )
    parser.add_argument(
        "--submit-mode",
        action="store_true",
        help=(
            "Print how this experiment wants to be submitted - 'array', 'array%%N' or "
            "'sequential' - and exit (used by submit_experiment.sh)."
        ),
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Merge the per-configuration result files of a run into results.csv/results.json and exit.",
    )

    return parser.parse_args()


def parse_override(text: str) -> tuple[str, object]:
    """Parse a --set KEY=VALUE pair, typing the value the way the experiment file would."""
    if "=" not in text:
        raise SystemExit(f"--set expects KEY=VALUE, got {text!r}")
    key, _, raw = text.partition("=")
    key = key.strip()

    try:
        import yaml

        value = yaml.safe_load(raw)
    except ImportError:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
    return key, value


def default_run_id() -> str:
    """Every task of one array must agree on the output directory, so prefer the array job
    id (shared by all tasks) over a per-process timestamp."""
    for env_var in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        value = os.environ.get(env_var)
        if value:
            return value
    return time.strftime("%Y%m%d-%H%M%S")


def flatten_result(result: RunResult) -> dict[str, object]:
    """One CSV row: the measures first, then the full configuration as cfg_* columns, so
    results.csv can be read on its own without cross-referencing configs.json."""
    row = {k: v for k, v in vars(result).items() if k != "config"}
    row.update({f"cfg_{key}": value for key, value in sorted(result.config.items())})
    return row


def collect(run_root: Path) -> int:
    """Merge results/<index>.json into results.csv and results.json."""
    results_dir = run_root / "results"
    if not results_dir.is_dir():
        raise SystemExit(f"No results directory at {results_dir} - has anything run yet?")

    rows = []
    for path in sorted(results_dir.glob("*.json"), key=lambda p: int(p.stem)):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No result files found in {results_dir}.")

    flat = [flatten_result(RunResult(**row)) for row in rows]

    # Configurations can differ in which keys they carry, so take the union in a stable order.
    fieldnames: list[str] = []
    for row in flat:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with (run_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)

    (run_root / "results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Collected {len(rows)} result(s) ({ok} ok, {len(rows) - ok} error) into {run_root / 'results.csv'}")
    return 0


# Keys of the slurm block that steer how submission happens rather than naming an
# sbatch flag. See submit_mode() below.
SUBMIT_CONTROL_KEYS = frozenset({"use_array", "array_throttle"})


def slurm_flags(spec: dict) -> str:
    """Render the experiment's slurm block as sbatch flags.

    Keeping these in the experiment file rather than in the sbatch header is what lets one
    generic run_experiment.sh serve every experiment: sbatch command-line flags override
    the header, and submit_experiment.sh passes whatever this prints.
    """
    mapping = {
        "partition": "--partition",
        "account": "--account",
        "time": "--time",
        "gres": "--gres",
        "nodes": "--nodes",
        "ntasks": "--ntasks",
        "ntasks_per_node": "--ntasks-per-node",
        "cpus_per_task": "--cpus-per-task",
        "mem_per_cpu": "--mem-per-cpu",
    }
    block = spec.get("slurm") or {}
    unknown = sorted(set(block) - set(mapping) - SUBMIT_CONTROL_KEYS)
    if unknown:
        valid = sorted(set(mapping) | SUBMIT_CONTROL_KEYS)
        raise SystemExit(f"Unknown slurm key(s): {', '.join(unknown)}. Valid: {', '.join(valid)}")
    return " ".join(f"{mapping[key]}={block[key]}" for key in mapping if key in block)


def submit_mode(spec: dict) -> str:
    """How this experiment should be submitted: "array", "array%N", or "sequential".

    Slurm counts each array task as its own job against the project limit, so an
    experiment aimed at a partition with a tight limit can ask not to be an array at
    all - its configurations then run in sequence inside a single job.
    """
    block = spec.get("slurm") or {}
    if block.get("use_array") is False:
        return "sequential"
    throttle = block.get("array_throttle")
    return f"array%{int(throttle)}" if throttle else "array"


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    spec = load_spec(config_path)

    if args.slurm_flags:
        print(slurm_flags(spec))
        return 0

    experiment_name = spec.get("name") or config_path.stem
    configs = expand(spec)

    for override in args.overrides:
        key, value = parse_override(override)
        if key not in RunConfig.field_names():
            raise SystemExit(f"--set key {key!r} is not a configuration field.")
        configs = [RunConfig.from_dict({**cfg.to_dict(), key: value}) for cfg in configs]

    if args.count:
        print(len(configs))
        return 0

    if args.submit_mode:
        print(submit_mode(spec))
        return 0

    if args.module_groups:
        for module, indices in module_groups(configs, spec.get("modules")).items():
            print(f"{module}\t{','.join(str(i) for i in indices)}")
        return 0

    if args.dry_run:
        print(f"{experiment_name}: {len(configs)} configuration(s)")
        for index, config in enumerate(configs):
            print(f"  [{index}] {config.label()}  ({config.config_hash()})")
        return 0

    run_root = Path(args.output_root) / experiment_name / (args.run_id or default_run_id())

    if args.collect:
        return collect(run_root)

    (run_root / "results").mkdir(parents=True, exist_ok=True)

    # Record what was asked for alongside what came out of it. Written by every task, which
    # is harmless - they all write identical content from the same source file.
    shutil.copyfile(config_path, run_root / config_path.name)
    (run_root / "configs.json").write_text(
        json.dumps(
            [{"index": i, "config_hash": c.config_hash(), "config": c.to_dict()} for i, c in enumerate(configs)],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if args.index is not None and args.indices is not None:
        raise SystemExit("Pass either --index or --indices, not both.")

    if args.index is not None:
        wanted = [args.index]
    elif args.indices is not None:
        wanted = [int(part) for part in args.indices.split(",") if part.strip()]
    else:
        wanted = list(range(len(configs)))

    for index in wanted:
        if not 0 <= index < len(configs):
            raise SystemExit(f"Index {index} out of range: this experiment has {len(configs)} configuration(s).")
    selected = [(index, configs[index]) for index in wanted]

    # Group the sweep in wandb without touching PatternAnalysis: wandb reads both of these
    # from the environment at init time.
    os.environ.setdefault("WANDB_RUN_GROUP", f"{experiment_name}_{run_root.name}")

    print(f"Experiment: {experiment_name} ({len(configs)} configuration(s))")
    print(f"Output:     {run_root}")

    runner = run_one_guarded if args.continue_on_error else run_one
    failures = 0

    for index, config in selected:
        run_dir = run_root / f"{index:03d}_{config.config_hash()}"
        os.environ["WANDB_JOB_TYPE"] = str(config.embedder or "NoEmbedder")
        # Name the wandb run after its output directory, so a run in the UI leads
        # straight to its config.json, plots and validation_measures.csv on disk.
        # Set per iteration, since sequential mode runs several configs in one process.
        os.environ["WANDB_NAME"] = f"{run_dir.name}_{config.label()}"

        result = runner(config, index=index, run_dir=run_dir)
        (run_root / "results" / f"{index}.json").write_text(
            json.dumps(vars(result), indent=2, default=str), encoding="utf-8"
        )
        if result.status != "ok":
            failures += 1
            print(f"[{index}] FAILED: {result.error}")

    if len(selected) == len(configs):
        collect(run_root)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
