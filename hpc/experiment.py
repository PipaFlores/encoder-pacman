"""Experiment definition and execution for the Pacman PatternAnalysis pipeline.

This is the one place that knows how to turn a configuration into a finished
PatternAnalysis run. Both entry points go through it:

- `run_experiment.py` expands an experiment file (a `defaults` block plus a `grid`)
  into many `RunConfig`s and runs them by index, one per SLURM array task;
- `train_model.py` builds a single `RunConfig` from its command line and runs that.

Keeping `run_one()` as the only code path that touches PatternAnalysis is the point:
the previous setup had the sbatch scripts and the CLI drift apart from the library
(embedders that existed in `PatternAnalysis` but not in the CLI, pipeline arguments
that were never exposed), and two parallel implementations would do it again.

Checkpoints stay in the shared `hpc/trained_models/` tree and every run trains from
scratch (`force_training=True`). That means configurations differing only in something
absent from the checkpoint name - feature set with the same column count, normalization,
context, masking - overwrite each other's saved weights. That is deliberate: the latent
space and the validation measures are the deliverable here, not the weights.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

HPC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HPC_DIR, ".."))

sys.path.append(REPO_ROOT)

# torch, umap and the pipeline itself are imported lazily inside the builders below:
# expanding a grid (--dry-run, --count, --slurm-flags) happens on a login node, where
# paying for a torch import to print a list of configurations would be absurd.

if TYPE_CHECKING:  # annotations only - never imported at runtime
    from src.analysis import PatternAnalysis


# Which environment module an embedder needs. The aeon/keras architectures import
# tensorflow and the rest import torch, and the two modules cannot be loaded at once -
# which is why train_benchmark_autoencoders.sh runs them as two sequential passes. For
# an array the equivalent is one array per module over the same experiment file (see
# module_groups() and submit_experiment.sh), so a single experiment can still sweep
# every embedder and land in one results directory.
KERAS_EMBEDDERS = frozenset({"DRNN", "DCNN", "ResNet"})
DEFAULT_MODULES = {
    "torch": "python-pytorch/2.13",
    "keras": "python-tensorflow/2.21",
}


# Keys that describe where a run reads and writes, or how loudly it talks - not what it
# does. Excluded from the config hash so the same experimental condition hashes the same
# on the cluster and on a laptop.
_HASH_EXCLUDE = frozenset(
    {
        "data_folder",
        "hpc_folder",
        "cache_folder",
        "ignore_cache",
        "disable_wandb",
        "logging_comment",
        "verbose",
        "using_hpc",
    }
)


@dataclass
class RunConfig:
    """One fully-resolved pipeline configuration.

    Field names are the vocabulary of the experiment files: a key under `defaults:` or
    `grid:` is valid exactly when it names a field here, which is also what makes a typo
    in a sweep fail at expansion time rather than three hours into an array job.

    Booleans are stored in their positive form (`rebase_score`, `use_best`) even though
    the CLI spells them as negations (`--no-rebase-score`, `--use-last-model`), so an
    experiment file never asks the reader to think in double negatives.
    """

    # Data / IO
    data_folder: str = os.path.join(REPO_ROOT, "data")
    hpc_folder: str = HPC_DIR
    cache_folder: str = os.path.join(REPO_ROOT, "cache")
    ignore_cache: bool = False

    # Slicing / features
    sequence_type: str = "first_5_seconds"
    context: int = 20
    filter_by_pill: Optional[int] = None
    rebase_score: bool = True
    feature_set: str = "Pacman"
    normalization: Optional[str] = None
    sort_ghost_distances: bool = False

    # Embedding
    embedder: Optional[str] = "LSTM"
    latent_space: int = 256
    n_epochs: int = 500
    batch_size: int = 32
    validation_split: float = 0.3
    dropout: float = 0.1
    elementwise_masking: bool = False
    use_best: bool = True

    # Reducer
    reducer: Optional[str] = "umap"
    reducer_components: int = 2
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # Clustering
    clusterer: str = "hdbscan"
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    kmeans_k: int = 6
    similarity_measure: str = "euclidean"
    geom_similarity: str = "dtw"

    # Validation / logging
    validation_method: Optional[str] = "Behavlets"
    disable_wandb: bool = False
    logging_comment: str = ""

    # Execution
    test_dataset: bool = False
    test_run: bool = False
    max_samples: Optional[int] = None
    using_hpc: bool = False
    seed: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        # Experiment files and shell arguments both spell "no embedder" as the string
        # "none"; the pipeline wants an actual None.
        for key in ("embedder", "reducer", "normalization", "validation_method"):
            value = getattr(self, key)
            if isinstance(value, str) and value.lower() == "none":
                setattr(self, key, None)

        # HDBSCAN's own default for min_samples is None, and the CLI spells that -1.
        if self.min_samples is not None and self.min_samples < 0:
            self.min_samples = None

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunConfig":
        """Build a config, naming every unknown key at once rather than one per attempt."""
        unknown = sorted(set(values) - cls.field_names())
        if unknown:
            raise ValueError(
                f"Unknown configuration key(s): {', '.join(unknown)}. "
                f"Valid keys are: {', '.join(sorted(cls.field_names()))}"
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """Short stable digest of the experimental condition (see _HASH_EXCLUDE)."""
        payload = {k: v for k, v in sorted(self.to_dict().items()) if k not in _HASH_EXCLUDE}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def label(self) -> str:
        """Human-readable identity for logs and run directories."""
        return f"{self.sequence_type}_{self.feature_set}_{self.embedder or 'NoEmbedder'}_h{self.latent_space}"


@dataclass
class RunResult:
    """One row of an experiment's results.csv / results.json."""

    index: int
    config_hash: str
    label: str
    status: str  # "ok" | "error"
    config: dict[str, Any] = field(default_factory=dict)
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    embedding_dim: Optional[int] = None
    reduced_dim: Optional[int] = None
    n_clusters: Optional[int] = None
    n_noise: Optional[int] = None
    mean_ari: Optional[float] = None
    mean_ami: Optional[float] = None
    mean_nmi: Optional[float] = None
    mean_neigh_hit: Optional[float] = None
    best_ari: Optional[float] = None
    best_ari_set: Optional[str] = None
    n_validation_sets: Optional[int] = None
    wandb_run_id: Optional[str] = None
    wandb_url: Optional[str] = None
    fit_seconds: Optional[float] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------------------
# Experiment files
# --------------------------------------------------------------------------------------


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read an experiment file. YAML by default, JSON accepted as a fallback so a compute
    node without PyYAML can still run a converted experiment rather than stranding a job."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"PyYAML is not available in this environment, so {path.name} cannot be read. "
            "Convert the experiment file to .json and pass that instead."
        ) from exc

    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{path} does not contain a mapping at the top level.")
    return spec


def _matches(config: dict[str, Any], rule: dict[str, Any]) -> bool:
    """True when every key/value in `rule` is present in `config` - a partial match, so an
    exclude rule can name just the two fields that make a combination invalid."""
    return all(config.get(key) == value for key, value in rule.items())


def expand(spec: dict[str, Any]) -> list[RunConfig]:
    """Expand an experiment file into an ordered list of configurations.

    The order is the cartesian product of `grid` in the key order written in the file,
    followed by any `include` entries. It is deterministic, which is what lets an array
    task trust `--index`: editing the experiment file renumbers the indices, so a running
    array should be left alone until it finishes.
    """
    known = {"name", "description", "slurm", "modules", "defaults", "grid", "include", "exclude"}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise ValueError(
            f"Unknown top-level key(s) in experiment file: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(known))}"
        )

    defaults = dict(spec.get("defaults") or {})
    grid = dict(spec.get("grid") or {})
    description = spec.get("description")

    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid entry '{key}' must be a non-empty list, got {values!r}")

    includes = list(spec.get("include") or [])
    grid_keys = list(grid)
    combos: Iterable[dict[str, Any]]
    if grid_keys:
        combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*(grid[k] for k in grid_keys))]
    elif includes:
        # An include-only experiment lists its configurations explicitly, so `defaults`
        # is a base for them rather than a configuration in its own right.
        combos = []
    else:
        combos = [{}]

    raw = [{**defaults, **combo} for combo in combos]
    raw += [{**defaults, **dict(entry)} for entry in includes]

    for rule in spec.get("exclude") or []:
        raw = [cfg for cfg in raw if not _matches(cfg, dict(rule))]

    if not raw:
        raise ValueError("Experiment expanded to zero configurations - check grid/exclude.")

    # The experiment's description doubles as the wandb comment, unless a config sets its own.
    if description:
        for cfg in raw:
            cfg.setdefault("logging_comment", description)

    configs = [RunConfig.from_dict(cfg) for cfg in raw]

    seen: set[str] = set()
    unique: list[RunConfig] = []
    for cfg in configs:
        digest = cfg.config_hash()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(cfg)
    return unique


def required_module(config: RunConfig, modules: Optional[dict[str, str]] = None) -> str:
    """The environment module this configuration has to run under."""
    table = {**DEFAULT_MODULES, **(modules or {})}
    return table["keras" if config.embedder in KERAS_EMBEDDERS else "torch"]


def module_groups(configs: list[RunConfig], modules: Optional[dict[str, str]] = None) -> dict[str, list[int]]:
    """Group configuration indices by the module they need, in first-appearance order.

    submit_experiment.sh turns each group into its own `sbatch --array=<indices>`, which
    is how one experiment file sweeping both torch and keras embedders still produces a
    single results directory with no index collisions.
    """
    groups: dict[str, list[int]] = {}
    for index, config in enumerate(configs):
        groups.setdefault(required_module(config, modules), []).append(index)
    return groups


# --------------------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------------------


def build_reducer(config: RunConfig):
    if config.reducer is None:
        return None
    name = config.reducer.lower()
    if name == "umap":
        from umap import UMAP

        return UMAP(
            n_neighbors=config.umap_neighbors,
            n_components=config.reducer_components,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_state=config.seed,
        )
    if name == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=config.reducer_components, random_state=config.seed)
    raise ValueError(f"Reducer {config.reducer} is not supported.")


def build_clusterer(config: RunConfig):
    if config.clusterer == "hdbscan":
        from hdbscan import HDBSCAN

        return HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            metric="euclidean",
        )
    if config.clusterer == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=config.kmeans_k, random_state=config.seed, n_init="auto")
    if config.clusterer == "geom":
        from src.analysis import GeomClustering

        return GeomClustering(
            similarity_measure=config.geom_similarity,
            verbose=config.verbose,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
        )
    raise ValueError(f"Clusterer {config.clusterer} is not supported.")


def build_analysis(config: RunConfig) -> "PatternAnalysis":
    from src.analysis import PatternAnalysis

    return PatternAnalysis(
        data_folder=config.data_folder,
        hpc_folder=config.hpc_folder,
        cache_folder=config.cache_folder,
        embedder=config.embedder,
        reducer=build_reducer(config),
        clusterer=build_clusterer(config),
        similarity_measure=config.similarity_measure,
        sequence_type=config.sequence_type,
        context=config.context,
        rebase_score=config.rebase_score,
        filter_by_pill=config.filter_by_pill,
        validation_method=config.validation_method,
        feature_set=config.feature_set,
        augmented_visualization=False,
        batch_size=config.batch_size,
        normalization=config.normalization,
        max_epochs=config.n_epochs,
        latent_dimension=config.latent_space,
        validation_data_split=config.validation_split,
        elementwise_masking=config.elementwise_masking,
        use_best=config.use_best,
        max_samples=config.max_samples,
        dropout=config.dropout,
        sort_distances=config.sort_ghost_distances,
        using_hpc=config.using_hpc,
        random_seed=config.seed,
        verbose=config.verbose,
        wandb_logging=not config.disable_wandb,
        wandb_logging_comment=config.logging_comment,
    )


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def _summarize_validation(analysis: "PatternAnalysis", result: RunResult, run_dir: Optional[Path]) -> None:
    """Fold the per-validation-set measures into the handful of numbers that belong in
    results.csv, and keep the full table beside the run for anything finer-grained.

    `validation_measures` is None whenever validation was skipped or had no labels to
    compare against, so nothing here may assume it exists - see the same guard in main().
    """
    measures = analysis.validation_measures
    if measures is None or len(measures) == 0:
        return

    result.n_validation_sets = int(len(measures))
    result.mean_ari = float(measures["ARI"].mean(skipna=True))
    result.mean_ami = float(measures["AMI"].mean(skipna=True))
    result.mean_nmi = float(measures["NMI"].mean(skipna=True))
    result.mean_neigh_hit = float(measures["neigh_hit"].mean(skipna=True))

    if measures["ARI"].notna().any():
        best = measures.loc[measures["ARI"].idxmax()]
        result.best_ari = float(best["ARI"])
        result.best_ari_set = str(best["validation_set"])

    if run_dir is not None:
        measures.to_csv(run_dir / "validation_measures.csv", index=False)


def run_one(config: RunConfig, index: int = 0, run_dir: Optional[Path] = None) -> RunResult:
    """Run the full pipeline for one configuration and report it as a RunResult.

    Exceptions are allowed to propagate; the caller decides whether one bad configuration
    should stop a sweep (see `--continue-on-error` in run_experiment.py).
    """
    result = RunResult(
        index=index,
        config_hash=config.config_hash(),
        label=config.label(),
        status="ok",
        config=config.to_dict(),
    )

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")

    print(f"[{index}] {config.label()} ({result.config_hash})")
    for key, value in sorted(config.to_dict().items()):
        print(f"    {key}: {value}")

    started = time.perf_counter()
    analysis = build_analysis(config)

    # force_training is not configurable on purpose: checkpoints are shared across
    # configurations (see the module docstring), so reusing one risks loading weights
    # trained under a different feature set or normalization.
    analysis.fit(
        force_training=True,
        ignore_cache=config.ignore_cache,
        test_dataset=config.test_dataset,
        test_run=config.test_run,
        close_wandb_logger=False,
    )
    analysis.summarize()
    result.fit_seconds = time.perf_counter() - started

    result.n_samples = analysis.results.get("n_samples") if isinstance(analysis.results.get("n_samples"), int) else None
    result.n_features = len(analysis.features_columns)
    result.embedding_dim = int(analysis.embeddings.shape[1]) if analysis.embeddings is not None else None
    result.reduced_dim = int(analysis.reduced_embeddings.shape[1]) if analysis.reduced_embeddings is not None else None
    result.n_clusters = analysis.results.get("n_clusters")
    result.n_noise = analysis.results.get("n_noise_points")

    _summarize_validation(analysis, result, run_dir)

    try:
        fig = analysis.plot_latent_space_overview(
            validation_set="all",
            black_background=True,
            colormap="viridis",
            dotsize=0.2,
            pretty_val_title=True,
        )
    except Exception as exc:
        fig = None
        print(f"Error in creating validation plots: {exc}")

    if fig is not None and run_dir is not None:
        fig.savefig(run_dir / "latent_space_overview.png", dpi=150, bbox_inches="tight")

    if analysis.wandbrun is not None:
        import wandb

        result.wandb_run_id = analysis.wandbrun.id
        result.wandb_url = getattr(analysis.wandbrun, "url", None)

        log_payload = {}
        if fig is not None:
            log_payload["validation_plot"] = wandb.Image(fig)

        # fit() only fills validation_measures when a validation method ran, and
        # calculate_validation_measures returns None when there are no validation labels
        # to compare against. Neither is a reason to lose a finished training run.
        if analysis.validation_measures is not None:
            log_payload["validation_measures"] = wandb.Table(dataframe=analysis.validation_measures)
        else:
            print("No validation measures to log: validation was skipped or produced no labels.")

        if log_payload:
            analysis.wandbrun.log(log_payload)
        analysis.wandbrun.finish()

    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return result


def run_one_guarded(config: RunConfig, index: int = 0, run_dir: Optional[Path] = None) -> RunResult:
    """`run_one` that turns a failure into an "error" result instead of an exception, so
    one bad configuration doesn't take the rest of a sweep with it."""
    try:
        return run_one(config, index=index, run_dir=run_dir)
    except Exception as exc:
        traceback.print_exc()
        return RunResult(
            index=index,
            config_hash=config.config_hash(),
            label=config.label(),
            status="error",
            config=config.to_dict(),
            error=f"{type(exc).__name__}: {exc}",
        )
