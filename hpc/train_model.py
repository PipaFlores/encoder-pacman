"""Single-configuration entry point for the PatternAnalysis pipeline.

This is a command-line front end over `experiment.run_one()` - the same function that
run_experiment.py calls for every configuration of a sweep. It exists for one-off and
debugging runs, where writing an experiment file for a single configuration would be
ceremony; anything that sweeps, or that should leave a results.csv behind, belongs in
run_experiment.py.

Nothing here constructs a pipeline of its own. That is deliberate: this CLI and the
library had drifted apart before (embedders reachable from PatternAnalysis but not
from here, pipeline arguments never exposed), and a second implementation of "build a
reducer, a clusterer and a PatternAnalysis" is exactly how that happens. RunConfig is
the shared vocabulary, and `RunConfig.from_dict` below rejects any argparse
destination that does not name one of its fields - so the drift fails at startup
rather than silently.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiment import HPC_DIR as _HPC_DIR, REPO_ROOT as _REPO_ROOT, RunConfig, run_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PatternAnalysis pipeline end-to-end."
    )

    # Data / IO -----------------------------------------------------------------
    # Defaults resolve relative to this script rather than the cwd, so a job does not
    # depend on being submitted from hpc/ (they point at the same folders when it is).
    parser.add_argument(
        "--data-folder",
        type=str,
        default=os.path.join(_REPO_ROOT, "data"),
        help="Base folder that stores Pacman CSV and processed artifacts.",
    )
    parser.add_argument(
        "--hpc-folder",
        type=str,
        default=_HPC_DIR,
        help="Folder to store affinity matrices, trained models and plots.",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=os.path.join(_REPO_ROOT, "cache"),
        help=(
            "Folder for cached pipeline intermediates (make_data, validation encodings). "
            "Kept separate from --hpc-folder since it is local to each machine/environment."
        ),
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Bypass the on-disk cache and recompute make_data and validation encodings from scratch (the fresh result still overwrites the cache).",
    )
    parser.add_argument(
        "--sequence-type",
        type=str,
        default="first_5_seconds",
        help="Slice type to use when building sequences (see PacmanDataReader).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=20,
        help="Number of frames of context to include for attack-mode slices. (Default to 20)",
    )

    parser.add_argument(
        "--filter-by-pill",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        help="If set (1-4), only use sequences associated with this power pill index. 1 is upper left, then clockwise. Use 'none' to disable.",
    )

    parser.add_argument(
        "--no-rebase-score",
        action="store_true",
        help="Keep the raw score instead of rebasing each sequence so its score starts at 0.",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="Pacman",
        help="Feature bundle to feed into the pipeline.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["global", "sequence", "sample", "none", "None"],
        help="Normalization strategy to apply before training.",
    )
    parser.add_argument(
        "--sort-ghost-distances",
        action="store_true",
        help="Sort ghost distance channels per frame when using Ghost_Distances.",
    )

    # Embedding -----------------------------------------------------------------
    parser.add_argument(
        "--embedder",
        type=str,
        default="LSTM",
        choices=["LSTM", "MLP", "Transformer", "VAE", "TimeVAE", "none", "None"],
        help=(
            "Deep embedder to use: LSTM, MLP, Transformer, VAE or TimeVAE. 'none' skips "
            "embedding and uses the reducer instead (e.g., UMAP)."
        ),
    )
    parser.add_argument(
        "--latent-space",
        type=int,
        default=256,
        help="Latent space size for autoencoders.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=500,
        help="Maximum number of epochs for embedding training.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for autoencoder training.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.3,
        help="Fraction of data set aside for validation during training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for the autoencoder embedder (default: 0.1).",
    )

    parser.add_argument(
        "--kld-weight",
        type=float,
        default=0.00025,
        help=(
            "Weight on the KL term of the VAE objective (VAE/TimeVAE only; ignored by the "
            "other embedders). The effective beta scales with seq_len x n_features because "
            "the two loss terms are reduced differently, so unit-variance ELBO is roughly "
            "2 / (seq_len * n_features) rather than 1 (default: 0.00025)."
        ),
    )

    parser.add_argument(
        "--elementwise-masking",
        action="store_true",
        help="Enable elementwise masking (obs. mask) during training of pytorch models.",
    )

    parser.add_argument(
        "--use-last-model",
        action="store_true",
        help="Use the last checkpoint instead of the best one when loading a trained embedder.",
    )

    # Reducer -------------------------------------------------------------------
    parser.add_argument(
        "--reducer",
        type=str,
        default="umap",
        choices=["umap", "pca", "none"],
        help="Dimensionality reducer applied to embeddings.",
    )
    parser.add_argument(
        "--reducer-components",
        type=int,
        default=2,
        help="Output dimensionality of reducer.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP nearest neighbors (only if reducer=umap).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance (only if reducer=umap).",
    )
    parser.add_argument(
        "--umap-metric",
        type=str,
        default="euclidean",
        help="UMAP metric (only if reducer=umap).",
    )

    # Clustering ----------------------------------------------------------------
    parser.add_argument(
        "--clusterer",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "kmeans"],
        help="Clustering algorithm for the reduced latent space.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=-1,
        help="HDBSCAN min_samples. Use -1 to keep the library default.",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="Optional cluster_selection_epsilon for HDBSCAN.",
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=6,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--similarity-measure",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Similarity metric for the affinity matrix of reduced embeddings.",
    )

    # Validation / logging ------------------------------------------------------
    parser.add_argument(
        "--validation-method",
        type=str,
        default="Behavlets",
        choices=["Behavlets", "none"],
        help="Validation labels to compute after clustering.",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if the package is available.",
    )

    parser.add_argument(
        "--logging-comment",
        type=str,
        default="",
        help="Custom comment or remarks to be added in the wandb logger"
    )

    # Execution -----------------------------------------------------------------
    parser.add_argument(
        "--test-dataset",
        action="store_true",
        help="Run against the PenDigits benchmark dataset (debug mode).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run the pipeline on a subset of at most 500 samples (debug mode).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of sequences fed to the pipeline. For fast debugging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity for the pipeline.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # argparse spells two booleans as negations for a better command line; RunConfig keeps
    # them positive. Every other destination name matches a RunConfig field one-for-one,
    # and from_dict() raises on any that does not.
    values = vars(args).copy()
    values["rebase_score"] = not values.pop("no_rebase_score")
    values["use_best"] = not values.pop("use_last_model")

    config = RunConfig.from_dict(values)
    result = run_one(config)

    if result.status != "ok":
        print(f"Run failed: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
