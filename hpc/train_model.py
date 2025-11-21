import argparse
import os
import sys
from typing import Optional

from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.analysis import GeomClustering, PatternAnalysis 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PatternAnalysis pipeline end-to-end."
    )

    # Data / IO -----------------------------------------------------------------
    parser.add_argument(
        "--data-folder",
        type=str,
        default=os.path.join("..", "data"),
        help="Base folder that stores Pacman CSV and processed artifacts.",
    )
    parser.add_argument(
        "--hpc-folder",
        type=str,
        default=".",
        help="Folder to store affinity matrices, trained models and plots.",
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
        "--feature-set",
        type=str,
        default="Pacman",
        choices=["Pacman", "Pacman_Ghosts", "Ghost_Distances"],
        help="Feature bundle to feed into the pipeline.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["global", "sequence", "sample", "none"],
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
        choices=["LSTM", "DRNN", "DCNN", "ResNet", "none"],
        help="Deep embedder to use. 'none' skips embedding (geom clustering only).",
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
        choices=["hdbscan", "kmeans", "geom"],
        help="Clustering algorithm for latent space or trajectories.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="HDBSCAN min_cluster_size / geom clustering min cluster size.",
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
    parser.add_argument(
        "--geom-similarity",
        type=str,
        default="dtw",
        help="Similarity measure for geometric clustering (only if clusterer=geom).",
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
        "--using-hpc",
        action="store_true",
        help="Enable HPC-friendly settings (mostly affects GeomClustering affinity).",
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


def build_reducer(args: argparse.Namespace):
    reducer_name = args.reducer.lower()
    if reducer_name == "umap":
        return UMAP(
            n_neighbors=args.umap_neighbors,
            n_components=args.reducer_components,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
    if reducer_name == "pca":
        return PCA(n_components=args.reducer_components, random_state=args.seed)
    if reducer_name == "none":
        return None
    raise ValueError(f"Reducer {args.reducer} is not supported.")


def build_clusterer(args: argparse.Namespace):
    min_samples: Optional[int]
    if args.min_samples is None or args.min_samples < 0:
        min_samples = None
    else:
        min_samples = args.min_samples

    if args.clusterer == "hdbscan":
        return HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            metric="euclidean",
        )
    if args.clusterer == "kmeans":
        return KMeans(
            n_clusters=args.kmeans_k,
            random_state=args.seed,
            n_init="auto",
        )
    if args.clusterer == "geom":
        return GeomClustering(
            similarity_measure=args.geom_similarity,
            verbose=args.verbose,
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
        )
    raise ValueError(f"Clusterer {args.clusterer} is not supported.")


def main():
    args = parse_args()

    if not args.disable_wandb:
        try:
            import wandb
            WANDB_AVAILABLE = True
        except ImportError:
            WANDB_AVAILABLE = False
    else:
        WANDB_AVAILABLE = False

    print("Running PatternAnalysis with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    embedder = None if args.embedder.lower() == "none" else args.embedder
    normalization = (
        None if args.normalization.lower() == "none" else args.normalization
    )
    validation_method = (
        None if args.validation_method.lower() == "none" else args.validation_method
    )

    reducer = build_reducer(args)
    clusterer = build_clusterer(args)

    analysis = PatternAnalysis(
        data_folder=args.data_folder,
        hpc_folder=args.hpc_folder,
        embedder=embedder,
        reducer=reducer,
        clusterer=clusterer,
        similarity_measure=args.similarity_measure,
        sequence_type=args.sequence_type,
        context=args.context,
        validation_method=validation_method,
        feature_set=args.feature_set,
        augmented_visualization=False,
        batch_size=args.batch_size,
        normalization=normalization,
        max_epochs=args.n_epochs,
        latent_dimension=args.latent_space,
        validation_data_split=args.validation_split,
        sort_distances=args.sort_ghost_distances,
        using_hpc=args.using_hpc,
        random_seed=args.seed,
        verbose=args.verbose,
        wandb_logging=not args.disable_wandb,
        wandb_logging_comment=args.logging_comment,
    )

    analysis.fit(
        force_training=True,
        test_dataset=args.test_dataset,
        close_wandb_logger=False
    )
    analysis.summarize()

    ## VISUALIZE (
    # With old style way instead of pa.plot_latent_space_overview() to keep current wandb workspace
    import matplotlib.pyplot as plt


    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    axs.scatter(analysis.reduced_embeddings[:,0], analysis.reduced_embeddings[:,1], s=2, cmap="tab20", c=analysis.labels)
    axs.set_title(f"{args.sequence_type}_{args.feature_set}_{embedder}_{reducer}_{clusterer}", size=8)

    if WANDB_AVAILABLE:
        analysis.wandbrun.log({"latent_space_plot": fig})  
        
        artifact = wandb.Artifact('model', type='model')
        
        # Extract dir and filename stem for best/last construction
        import os
        model_dir = os.path.dirname(analysis.model_path)
        model_name = os.path.basename(analysis.model_path)

        if model_name.endswith('_last.pth'):
            model_base = model_name[:-9]  # Remove '_last.pth'
        elif model_name.endswith('_best.pth'):
            model_base = model_name[:-9]  # Remove '_best.pth'
        else:
            model_base = os.path.splitext(model_name)[0]

        model_best = os.path.join(model_dir, model_base + '_best.pth')
        model_last = os.path.join(model_dir, model_base + '_last.pth')

        artifact.add_file(model_best)
        artifact.add_file(model_last)

        analysis.wandbrun.log_artifact(artifact)
        analysis.wandbrun.finish()


if __name__ == "__main__":
    main()
