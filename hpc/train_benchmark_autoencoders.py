"""Benchmark time-series representation models on labeled aeon datasets.

The experiment mirrors the Pacman `PatternAnalysis` latent-space flow while using
benchmark datasets with known labels:

1. Load an aeon classification dataset. Aeon arrays are channels-first, so they
   are converted to this repo's [samples, time, features] convention.
2. Train a deep autoencoder-style architecture, or fit a no-training baseline.
3. Extract one latent vector per test sample.
4. Reduce latent vectors with UMAP, cluster with HDBSCAN, and score clusters
   against the dataset labels with ARI/AMI/NMI plus neighborhood hit.
5. Save a two-panel latent-space plot colored by HDBSCAN labels and true labels.

`UMAP`, `RandomProjection`, and `RandomNoise` are baselines: they do not train
a reconstruction model, so their reconstruction losses are recorded as NA.
`RandomProjection` is a genuine (if crude) Gaussian random projection of the
data, while `RandomNoise` is a null baseline of uniform noise unrelated to
the data, matching `src.utils.utils.random_projection_measures`.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Result:
    """Single architecture result row written to CSV/JSON after each run."""
    dataset: str
    architecture: str
    status: str
    train_loss: float | None = None
    val_loss: float | None = None
    test_loss: float | None = None
    ari: float | None = None
    ami: float | None = None
    nmi: float | None = None
    neigh_hit: float | None = None
    n_clusters: int | None = None
    plot_path: str | None = None
    fit_seconds: float | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse experiment, architecture, masking, UMAP, and HDBSCAN settings."""
    parser = argparse.ArgumentParser(
        description="Train and compare autoencoder architectures on an aeon time-series benchmark dataset."
    )
    parser.add_argument(
        "--dataset",
        default="BasicMotions",
        help="aeon dataset name. BasicMotions is a small multivariate default.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print dataset names exposed by the installed aeon.datasets.tsc_datasets module and exit.",
    )
    parser.add_argument(
        "--list-datasets-filter",
        default=None,
        help="Optional case-insensitive substring filter used with --list-datasets.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["LSTM", "Transformer", "VanillaVAE", "TimeVAE", "DRNN", "DCNN", "ResNet", "UMAP", "RandomProjection", "RandomNoise"],
        choices=["LSTM", "Transformer", "VanillaVAE", "TimeVAE", "DRNN", "DCNN", "ResNet", "UMAP", "RandomProjection", "RandomNoise"],
        help="Architectures to run.",
    )
    parser.add_argument("--output-dir", default=os.path.join("benchmark_results", "autoencoders"))
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier used as the output subdirectory name. Defaults to the current "
            "timestamp. Pass the same run-id across multiple invocations (e.g. a pytorch pass "
            "followed by a tensorflow pass) to accumulate their results in one results.csv/json."
        ),
    )
    parser.add_argument("--latent-space", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--kld-weight", type=float, default=0.00025)
    parser.add_argument("--elementwise-masking", action="store_true")
    parser.add_argument("--transformer-masking-ratio", type=float, default=0.15)
    parser.add_argument("--transformer-mean-mask-length", type=int, default=3)
    parser.add_argument("--transformer-mask-mode", default="separate", choices=["separate", "concurrent"])
    parser.add_argument("--transformer-mask-distribution", default="geometric", choices=["geometric", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--clustering-dim", type=int, default=2)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--neighborhood-k", type=int, default=5)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and keep running remaining architectures.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()



def list_aeon_datasets(name_filter: str | None = None) -> None:
    """Print dataset names exposed by the installed aeon version."""
    from aeon.datasets import tsc_datasets

    seen = set()
    for attr_name, value in sorted(vars(tsc_datasets).items()):
        if attr_name.startswith("_"):
            continue
        if isinstance(value, dict):
            names = sorted(str(name) for name in value.keys())
        elif isinstance(value, (list, tuple, set)) and all(isinstance(item, str) for item in value):
            names = sorted(value)
        else:
            continue

        if name_filter is not None:
            names = [name for name in names if name_filter.lower() in name.lower()]
        if not names:
            continue

        print(f"\n{attr_name} ({len(names)})")
        for name in names:
            print(name)
            seen.add(name)

    print(f"\nUnique dataset names shown: {len(seen)}")


def load_aeon_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test splits when available, otherwise create a deterministic split."""
    from aeon.datasets import load_classification

    try:
        X_train, y_train = load_classification(name, split="train")
        X_test, y_test = load_classification(name, split="test")
    except TypeError:
        X, y = load_classification(name)
        rng = np.random.default_rng(42)
        order = rng.permutation(len(X))
        split_at = int(0.8 * len(X))
        train_idx, test_idx = order[:split_at], order[split_at:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


def to_repo_shape(X: np.ndarray) -> np.ndarray:
    """aeon uses [samples, channels, time]; repo torch models use [samples, time, features]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D time-series array, got shape {X.shape}")
    return np.transpose(X, (0, 2, 1))


def standardize_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each feature using train-set statistics only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def cap_samples(X: np.ndarray, y: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Optionally truncate train/test arrays for quick smoke runs."""
    if limit is None:
        return X, y
    return X[:limit], y[:limit]


def evaluate_torch_model(model, X_test: np.ndarray, args: argparse.Namespace) -> float:
    """Evaluate reconstruction loss for torch models on held-out test data.

    LSTM and VAE variants use full reconstruction MSE. The transformer uses the
    same masked-imputation objective as `Transformer_Trainer`: mask test inputs,
    reconstruct the original values, and score only the masked positions.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.datahandlers import ImputationDataset, PacmanDataset, collate_dynamic_padding
    from src.models import AELSTM, TSTransformerEncoder
    from src.models.loss import MaskedMSELoss, VAELoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if isinstance(model, TSTransformerEncoder):
        dataset = ImputationDataset(
            X_test,
            mean_mask_length=args.transformer_mean_mask_length,
            masking_ratio=args.transformer_masking_ratio,
            mode=args.transformer_mask_mode,
            distribution=args.transformer_mask_distribution,
            elementwise_masking=args.elementwise_masking,
        )
    else:
        dataset = PacmanDataset(X_test, elementwise_masking=args.elementwise_masking)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dynamic_padding)
    mse_loss = MaskedMSELoss()
    vae_loss = VAELoss(kld_weight=0.0)
    total = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["data"].to(device)
            padding_mask = batch.get("padding_mask")
            obs_mask = batch.get("obs_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            if obs_mask is not None:
                obs_mask = obs_mask.to(device)

            if isinstance(model, TSTransformerEncoder):
                noise_mask = batch["noise_mask"].to(device)
                # Zero out masked positions, per Zerveas et al. 2020, rather than a learned mask token.
                x_masked = x * noise_mask
                recon = model(x_masked, padding_mask.bool())
                loss_mask = 1 - noise_mask
                loss = mse_loss(recon, x, padding_mask=padding_mask, obs_mask=obs_mask, loss_mask=loss_mask)
            elif isinstance(model, AELSTM):
                lengths = batch["lengths"].to(device)
                recon = model(x, lengths=lengths)
                loss = mse_loss(recon, x, padding_mask=padding_mask, obs_mask=obs_mask)
            else:
                recon, mu, log_var = model(x, padding_mask=padding_mask)
                loss, _ = vae_loss(recon, x, mu, log_var, padding_mask=padding_mask, obs_mask=obs_mask)

            total += float(loss.item()) * len(x)
            count += len(x)

    return total / max(count, 1)


def train_lstm(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train the repo LSTM autoencoder and return train/val/test losses."""
    from src.datahandlers import PacmanDataset
    from src.models import AELSTM, AE_Trainer

    model = AELSTM(
        input_size=X_train.shape[-1],
        hidden_size=args.latent_space,
        dropout=args.dropout,
        seq_length=X_train.shape[1],
    )
    trainer = AE_Trainer(
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        lr=args.learning_rate,
        verbose=args.verbose,
    )
    trainer.fit(model, PacmanDataset(X_train, elementwise_masking=args.elementwise_masking))
    return model, trainer.train_loss_list[-1], trainer.val_loss_list[-1], evaluate_torch_model(model, X_test, args)


def train_transformer(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train the repo transformer with dynamic noise masks for imputation."""
    from src.datahandlers import ImputationDataset
    from src.models import TSTransformerEncoder, Transformer_Trainer

    model = TSTransformerEncoder(
        feat_dim=X_train.shape[-1],
        max_len=X_train.shape[1],
        d_model=args.latent_space,
        dropout=args.dropout,
        pos_encoding="fixed",
        pooling_method="mean",
        n_heads=8,
        num_layers=3,
        dim_feedforward=256,
    )
    trainer = Transformer_Trainer(
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        use_imputation=True,
        lr=args.learning_rate,
        seed=args.seed,
        verbose=args.verbose,
    )
    trainer.fit(
        model,
        ImputationDataset(
            X_train,
            mean_mask_length=args.transformer_mean_mask_length,
            masking_ratio=args.transformer_masking_ratio,
            mode=args.transformer_mask_mode,
            distribution=args.transformer_mask_distribution,
            elementwise_masking=args.elementwise_masking,
        ),
    )
    return model, trainer.train_loss_list[-1], trainer.val_loss_list[-1], evaluate_torch_model(model, X_test, args)


def train_vanilla_vae(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train the repo 1D convolutional VAE."""
    from src.datahandlers import PacmanDataset
    from src.models import VanillaVAE, VAE_Trainer

    dataset = PacmanDataset(X_train, elementwise_masking=args.elementwise_masking)
    # Only pool (padding-safe, position-agnostic) when this dataset actually has padded
    # samples; a genuinely fixed-length dataset keeps the reference flatten+Linear encoder,
    # which is strictly more expressive when there's no padding to protect against.
    has_padding = bool(dataset.lengths.min().item() < X_train.shape[1])

    model = VanillaVAE(
        input_dim=X_train.shape[-1],
        seq_len=X_train.shape[1],
        latent_dim=args.latent_space,
        pooling=has_padding,
    )
    trainer = VAE_Trainer(
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        kld_weight=args.kld_weight,
        verbose=args.verbose,
    )
    trainer.fit(model, dataset)
    return model, trainer.train_loss_list[-1], trainer.val_loss_list[-1], evaluate_torch_model(model, X_test, args)


def train_time_vae(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train TimeVAE with the same VAE trainer/loss used by VanillaVAE."""
    from src.datahandlers import PacmanDataset
    from src.models import TimeVAE, VAE_Trainer

    model = TimeVAE(
        input_dim=X_train.shape[-1],
        seq_len=X_train.shape[1],
        latent_dim=args.latent_space,
    )
    trainer = VAE_Trainer(
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        kld_weight=args.kld_weight,
        verbose=args.verbose,
    )
    trainer.fit(model, PacmanDataset(X_train, elementwise_masking=args.elementwise_masking))
    return model, trainer.train_loss_list[-1], trainer.val_loss_list[-1], evaluate_torch_model(model, X_test, args)



def extract_torch_embeddings(name: str, model, X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Extract one latent vector per sample from torch-backed architectures."""
    import torch
    from torch.utils.data import DataLoader
    from src.datahandlers import ImputationDataset, PacmanDataset, collate_dynamic_padding
    from src.models import TSTransformerEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if isinstance(model, TSTransformerEncoder):
        dataset = ImputationDataset(X, masking_ratio=0.0, elementwise_masking=args.elementwise_masking)
    else:
        dataset = PacmanDataset(X, elementwise_masking=args.elementwise_masking)

    embeddings = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dynamic_padding):
            batch_x = batch["data"].to(device)
            if isinstance(model, TSTransformerEncoder):
                padding_mask = batch["padding_mask"].to(device)
                batch_z = model.encode(batch_x, padding_mask, pooling=True)
            elif name == "LSTM":
                batch_lengths = batch["lengths"].to(device)
                batch_z = model.encode(batch_x, lengths=batch_lengths)
            else:
                batch_padding_mask = batch["padding_mask"].to(device)
                encoded = model.encode(batch_x, padding_mask=batch_padding_mask)
                batch_z = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
            embeddings.append(batch_z.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def extract_aeon_embeddings(model, X: np.ndarray) -> np.ndarray:
    """Extract latent vectors from aeon/Keras clusterer internals."""
    X_channels_first = np.transpose(X, (0, 2, 1))
    return model.model_.layers[1].predict(X_channels_first)


def extract_embeddings(name: str, model, X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Dispatch latent extraction for trained models and no-training baselines."""
    if name in {"UMAP", "RandomProjection", "RandomNoise"}:
        return model.transform(X)
    if name in {"LSTM", "Transformer", "VanillaVAE", "TimeVAE"}:
        return extract_torch_embeddings(name, model, X, args)
    return extract_aeon_embeddings(model, X)


def reduce_for_clustering(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Reduce model embeddings with UMAP before HDBSCAN clustering."""
    from umap import UMAP

    return UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.clustering_dim,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    ).fit_transform(embeddings)


def reduce_for_visualization(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Ensure scatter plots are 2D even when clustering uses more dimensions."""
    from umap import UMAP

    if embeddings.shape[1] == 2:
        return embeddings
    return UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=2,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    ).fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Cluster the UMAP latent space with HDBSCAN, as in PatternAnalysis."""
    from hdbscan import HDBSCAN

    return HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    ).fit_predict(embeddings)


def encode_labels_for_plot(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map arbitrary true/cluster labels onto compact colorbar indices."""
    labels = np.asarray(labels)
    unique_labels = sorted(np.unique(labels), key=lambda value: str(value))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_to_index[label] for label in labels])
    return encoded, [str(label) for label in unique_labels]


def add_scatter(ax, points: np.ndarray, labels: np.ndarray, title: str):
    """Draw one labeled latent-space scatter subplot."""
    encoded, tick_labels = encode_labels_for_plot(labels)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=encoded, s=10, cmap="tab20", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(tick_labels) <= 20:
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(len(tick_labels)))
        cbar.set_ticklabels(tick_labels)
    return scatter


def latent_title(architecture: str, args: argparse.Namespace) -> str:
    """Build the plot title with architecture, latent dimensionality, and epochs."""
    if architecture == "UMAP":
        return f"UMAP latent dim={args.clustering_dim}, epochs=0"
    if architecture in {"RandomProjection", "RandomNoise"}:
        return f"{architecture} latent dim={args.clustering_dim}, epochs=0"
    return f"{architecture} latent h={args.latent_space}, epochs={args.n_epochs}"


def plot_filename(architecture: str, args: argparse.Namespace) -> str:
    """Use stable plot filenames that encode the architecture configuration."""
    if architecture == "UMAP":
        return f"UMAP_d{args.clustering_dim}.png"
    if architecture in {"RandomProjection", "RandomNoise"}:
        return f"{architecture}_d{args.clustering_dim}.png"
    return f"{architecture}_h{args.latent_space}_e{args.n_epochs}.png"


def save_latent_space_plot(
    visual_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    architecture: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    """Save a two-panel scatter plot: predicted clusters beside true labels."""
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "latent_space_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / plot_filename(architecture, args)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(latent_title(architecture, args))
    add_scatter(axes[0], visual_embeddings, cluster_labels, "HDBSCAN clusters")
    add_scatter(axes[1], visual_embeddings, true_labels, "True labels")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return str(plot_path)


def score_latent_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    architecture: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | int | str]:
    """Run UMAP, HDBSCAN, validation metrics, and plot generation.

    Neural architectures first reduce their learned embeddings with UMAP. The
    UMAP and random-projection baselines already produce the clustering space, so
    they skip the second reduction.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
    from src.utils.utils import neighborhood_hit

    labels = np.asarray(labels)
    if architecture in {"UMAP", "RandomProjection", "RandomNoise"}:
        clustering_embeddings = embeddings
    else:
        clustering_embeddings = reduce_for_clustering(embeddings, args)
    cluster_labels = cluster_embeddings(clustering_embeddings, args)
    visual_embeddings = reduce_for_visualization(clustering_embeddings, args)
    plot_path = save_latent_space_plot(
        visual_embeddings=visual_embeddings,
        cluster_labels=cluster_labels,
        true_labels=labels,
        architecture=architecture,
        args=args,
        output_dir=output_dir,
    )
    non_noise = set(cluster_labels.tolist()) - {-1}
    return {
        "ari": float(adjusted_rand_score(labels, cluster_labels)),
        "ami": float(adjusted_mutual_info_score(labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(labels, cluster_labels)),
        "neigh_hit": float(
            neighborhood_hit(
                clustering_embeddings,
                labels,
                n_neighbors=args.neighborhood_k,
                metric="euclidean",
                no_null_instances=False,
            )
        ),
        "n_clusters": len(non_noise),
        "plot_path": plot_path,
    }


class RandomProjectionBaseline:
    """Seeded Gaussian random projection over flattened time series.

    Unlike `RandomNoiseBaseline`, this is a genuine (if crude) linear
    dimensionality reduction of the input: by the Johnson-Lindenstrauss
    lemma it approximately preserves pairwise distances, so it can surface
    real cluster structure correlated with the true labels.

    References:
        Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
            mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
        Achlioptas, D. (2003). Database-friendly random projections:
            Johnson-Lindenstrauss with binary coins. Journal of Computer and
            System Sciences, 66(4), 671-687. (Source of the 1/sqrt(k) Gaussian
            scaling used below.)
        Bingham, E., & Mannila, H. (2001). Random projection in dimensionality
            reduction: Applications to image and text data. Proceedings of the
            7th ACM SIGKDD International Conference on Knowledge Discovery and
            Data Mining, 245-250.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.projection: np.ndarray | None = None

    @staticmethod
    def flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray):
        X_flat = self.flatten(X)
        rng = np.random.default_rng(self.args.seed)
        self.projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.args.clustering_dim),
            size=(X_flat.shape[1], self.args.clustering_dim),
        ).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.projection is None:
            raise RuntimeError("RandomProjectionBaseline must be fit before transform.")
        return self.flatten(X) @ self.projection


def train_random_projection_baseline(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    model = RandomProjectionBaseline(args).fit(X_train)
    return model, None, None, None


class RandomNoiseBaseline:
    """Pure random-noise null baseline, matching
    `src.utils.utils.random_projection_measures`: embeddings are uniform
    noise independent of the input values (only their sample count is
    used), so they carry no signal about the true labels. Useful as a
    lower-bound sanity check against `RandomProjectionBaseline`.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def fit(self, X: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.random.random(size=(len(X), self.args.clustering_dim))


def train_random_noise_baseline(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    model = RandomNoiseBaseline(args).fit(X_train)
    return model, None, None, None


class UMAPBaseline:
    """No-training baseline: fit UMAP on train data and transform test data."""
    def __init__(self, args: argparse.Namespace):
        from umap import UMAP

        self.reducer = UMAP(
            n_neighbors=args.umap_neighbors,
            n_components=args.clustering_dim,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )

    @staticmethod
    def flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray):
        self.reducer.fit(self.flatten(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.transform(self.flatten(X))


def train_umap_baseline(X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    model = UMAPBaseline(args).fit(X_train)
    return model, None, None, None


def train_aeon(name: str, X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train aeon/Keras autoencoder clusterers and evaluate reconstruction MSE."""
    from aeon.clustering import DummyClusterer
    from aeon.clustering.deep_learning import AEDCNNClusterer, AEDRNNClusterer, AEResNetClusterer

    constructors: dict[str, Callable[[], object]] = {
        "DRNN": lambda: AEDRNNClusterer(
            estimator=DummyClusterer(),
            latent_space_dim=args.latent_space,
            n_epochs=args.n_epochs,
            validation_split=args.validation_split,
            verbose=args.verbose,
        ),
        "DCNN": lambda: AEDCNNClusterer(
            estimator=DummyClusterer(),
            latent_space_dim=args.latent_space,
            n_epochs=args.n_epochs,
            validation_split=args.validation_split,
            verbose=args.verbose,
        ),
        "ResNet": lambda: AEResNetClusterer(
            estimator=DummyClusterer(),
            n_epochs=args.n_epochs,
            validation_split=args.validation_split,
            verbose=args.verbose,
        ),
    }
    model = constructors[name]()
    X_train_channels_first = np.transpose(X_train, (0, 2, 1))
    X_test_channels_first = np.transpose(X_test, (0, 2, 1))
    model.fit(X_train_channels_first)

    recon = model.model_(X_test_channels_first, training=False).numpy()
    test_loss = float(np.mean((recon - X_test_channels_first) ** 2))
    summary = model.summary()
    train_loss = float(summary["loss"][-1]) if "loss" in summary else None
    val_loss = float(summary["val_loss"][-1]) if "val_loss" in summary else None
    return model, train_loss, val_loss, test_loss


def run_architecture(name: str, X_train: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    """Train or fit the requested architecture/baseline."""
    trainers = {
        "LSTM": train_lstm,
        "Transformer": train_transformer,
        "VanillaVAE": train_vanilla_vae,
        "TimeVAE": train_time_vae,
        "UMAP": train_umap_baseline,
        "RandomProjection": train_random_projection_baseline,
        "RandomNoise": train_random_noise_baseline,
    }
    if name in trainers:
        return trainers[name](X_train, X_test, args)
    return train_aeon(name, X_train, X_test, args)


def load_existing_results(output_dir: Path) -> list[Result]:
    """Load a prior run's results.json, if present, so a later invocation with a
    shared --run-id (e.g. a tensorflow pass after a pytorch pass) appends to it
    instead of overwriting it."""
    json_path = output_dir / "results.json"
    if not json_path.exists():
        return []
    with json_path.open() as f:
        rows = json.load(f)
    return [Result(**row) for row in rows]


def write_results(results: list[Result], output_dir: Path, args: argparse.Namespace) -> None:
    """Persist incremental results so long benchmark runs keep partial output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    config_path = output_dir / "config.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Result.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    with json_path.open("w") as f:
        json.dump([row.__dict__ for row in results], f, indent=2)

    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)


def main():
    """Run the requested benchmark architectures on one dataset."""
    args = parse_args()
    if args.list_datasets:
        list_aeon_datasets(args.list_datasets_filter)
        return

    np.random.seed(args.seed)

    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    X_train_raw, y_train, X_test_raw, y_test = load_aeon_dataset(args.dataset)
    X_train = to_repo_shape(X_train_raw)
    X_test = to_repo_shape(X_test_raw)
    X_train, y_train = cap_samples(X_train, y_train, args.max_train_samples)
    X_test, y_test = cap_samples(X_test, y_test, args.max_test_samples)
    X_train, X_test = standardize_from_train(X_train, X_test)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / args.dataset / run_id
    results: list[Result] = load_existing_results(output_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Train shape: {X_train.shape}; test shape: {X_test.shape}")
    print(f"Writing results to {output_dir}")

    for architecture in args.architectures:
        print(f"\n=== {architecture} ===")
        started = time.perf_counter()
        try:
            model, train_loss, val_loss, test_loss = run_architecture(architecture, X_train, X_test, args)
            embeddings = extract_embeddings(architecture, model, X_test, args)
            latent_scores = score_latent_space(embeddings, y_test, architecture, args, output_dir)
            elapsed = time.perf_counter() - started
            result = Result(
                dataset=args.dataset,
                architecture=architecture,
                status="ok",
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                ari=latent_scores["ari"],
                ami=latent_scores["ami"],
                nmi=latent_scores["nmi"],
                neigh_hit=latent_scores["neigh_hit"],
                n_clusters=latent_scores["n_clusters"],
                plot_path=latent_scores["plot_path"],
                fit_seconds=elapsed,
            )
            test_loss_text = "NA" if test_loss is None else f"{test_loss:.6f}"
            print(
                f"test_loss={test_loss_text} "
                f"ARI={latent_scores['ari']:.4f} "
                f"AMI={latent_scores['ami']:.4f} "
                f"neigh_hit={latent_scores['neigh_hit']:.4f} "
                f"clusters={latent_scores['n_clusters']} "
                f"fit_seconds={elapsed:.1f}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            result = Result(
                dataset=args.dataset,
                architecture=architecture,
                status="failed",
                fit_seconds=elapsed,
                error=repr(exc),
            )
            print(f"failed: {exc!r}")
            if not args.continue_on_error:
                results.append(result)
                write_results(results, output_dir, args)
                raise
        results.append(result)
        write_results(results, output_dir, args)

    print("\nSummary")
    for result in results:
        if result.status == "ok":
            test_loss_text = "NA" if result.test_loss is None else f"{result.test_loss:.6f}"
            print(
                f"{result.architecture}: test_loss={test_loss_text} "
                f"ARI={result.ari:.4f} AMI={result.ami:.4f} "
                f"neigh_hit={result.neigh_hit:.4f}"
            )
        else:
            print(f"{result.architecture}: failed ({result.error})")


if __name__ == "__main__":
    main()
