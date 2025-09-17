import numpy as np
import random
from typing import List
import tqdm
from src.utils.similarity_measures import SimilarityMeasures
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
import hdbscan
from src.visualization.cluster_visualizer import ClusterVisualizer
from src.datahandlers import Trajectory

from src.utils import setup_logger
from bokeh.plotting import show, row
import time
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager
import threading

# Initialize module-level logger
logger = setup_logger(__name__)


def _compute_distance_chunk(args):
    """Helper function to compute distances for a chunk of trajectory pairs."""
    similarity_measures, trajectories, indices = args
    results = []
    for i, j in indices:
        distance = similarity_measures.calculate_distance(
            trajectories[i], trajectories[j]
        )
        results.append((i, j, distance))
    return results

class GeomClustering:
    """
    A class for clustering and analyzing geometric trajectories.
    FIXME : Move all analyzes and vis to pattern_analysis.py

    This class provides functionality to cluster trajectories based on their geometric similarity
    using either DBSCAN or HDBSCAN algorithms. It calculates affinity matrices between trajectories
    and provides various visualization methods for analyzing the clustering results.

    The class is particularly useful for analyzing movement patterns in game AI, such as Pac-Man
    trajectories, by grouping similar movement patterns together.

    Attributes:
        trajectories (np.ndarray): Array of trajectory data, shape (num_trajectories, num_timesteps, 2)
        trajectories_centroids (np.ndarray): Centroids of each trajectory
        similarity_measures (SimilarityMeasures): Object for calculating distances between trajectories
        cluster_method (str): Method used for clustering ('DBSCAN' or 'HDBSCAN')
        affinity_matrix (np.ndarray): Matrix of distances between all trajectory pairs
        clusterer: The fitted clustering model (DBSCAN or HDBSCAN)
        labels (np.ndarray): Cluster labels for each trajectory
        cluster_centroids (np.ndarray): Centroids of each cluster
        cluster_sizes (np.ndarray): Number of trajectories in each cluster
        vis (ClusterVisualizer): Visualizer for clustering results
    """

    def __init__(
        self,
        similarity_measure: str = "euclidean",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the GeomClustering object.

        Args:
            similarity_measure (str, optional): Method for calculating distances between trajectories.
                Defaults to "euclidean".
            cluster_method (str, optional): Algorithm to use for clustering. Options are "DBSCAN" or "HDBSCAN".
                Defaults to "HDBSCAN".
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        """
        if verbose:
            logger.setLevel("INFO")
        logger.info(
            f"Initializing GeomClustering with similarity measure: {similarity_measure}"
        )
        self.trajectories = None
        self.trajectories_centroids = np.array([])

        self.similarity_measures = SimilarityMeasures(similarity_measure)
        self.affinity_matrix = np.array([])

        self.clusterer = hdbscan.HDBSCAN(metric="precomputed", **kwargs)
        self.labels = np.array([])
        self.cluster_centroids = np.array([])
        self.cluster_sizes = np.array([])

    def fit(
        self,
        trajectories: List[Trajectory] | np.ndarray | List[np.ndarray],
        recalculate_affinity_matrix: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit the clustering model to the provided trajectories.

        This method performs the following steps:
        1. Converts trajectories to numpy array (if not already)
        2. Calculates the affinity matrix between all trajectory pairs
        3. Applies the specified clustering algorithm
        4. Sorts and remaps cluster labels by size
        5. Initializes the visualizer

        Args:
            trajectories (List[np.ndarray]): List of trajectory arrays, each with shape (num_timesteps, 2)
            cluster_method (str | None, optional): Override the clustering method for this fit.
                If None, uses the method specified at initialization. Defaults to None.
            recalculate_affinity_matrix (bool, optional): Whether to recalculate the affinity matrix.
            **kwargs: Additional parameters to pass to the clustering algorithm.(i.e., min_cluster_size, min_samples for HDBSCAN)

        Returns:
            np.ndarray: Array of cluster labels for each trajectory
        """

        logger.info(f"Fitting clustering model with {len(trajectories)} trajectories")

        self.trajectories = trajectories

        if recalculate_affinity_matrix or self.affinity_matrix.size == 0:
            self.affinity_matrix = self.calculate_affinity_matrix(trajectories)
        else:
            logger.info("Using existing affinity matrix")

        logger.info(f"clustering with {self.clusterer.__class__.__name__}")
        self.labels = self.clusterer.fit_predict(self.affinity_matrix)

        self.cluster_centroids, self.cluster_sizes = (
            np.array([]),
            np.array([]),
        )  # Reset cluster centroids and sizes
        self.labels = self._sort_labels()
        logger.info(f"Clustering complete. Found {len(set(self.labels)) - 1} clusters")

        self.cluster_vis = ClusterVisualizer(
        )  # Init visualizer

        return self.labels

    def calculate_affinity_matrix(
        self, trajectories: List[Trajectory] | np.ndarray | List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate affinity matrix between all trajectories in the list.

        The affinity matrix is a symmetric matrix where each element (i,j) represents
        the distance between trajectory i and trajectory j according to the specified
        similarity measure.

        Args:
            trajectories (np.ndarray | List[np.ndarray]): List of trajectory arrays to compare

        Returns:
            np.ndarray: Affinity matrix of shape (num_trajectories, num_trajectories)
        """
        logger.info("Calculating affinity matrix")
        time_start = time.time()
        num_trajectories = len(trajectories)
        affinity_matrix = np.zeros((num_trajectories, num_trajectories))

        total_pairs = num_trajectories * (num_trajectories - 1) // 2

        with tqdm.tqdm(total=total_pairs, desc="Calculating affinity matrix") as pbar:
            for i in range(num_trajectories):
                for j in range(i + 1, num_trajectories):
                    affinity_matrix[i, j] = (
                        self.similarity_measures.calculate_distance(
                            trajectories[i], trajectories[j]
                        )
                    )
                    affinity_matrix[j, i] = affinity_matrix[i, j]
                    pbar.update(1)

        logger.info(
            f"Affinity matrix calculation complete in {round(time.time() - time_start, 2)} seconds"
        )
        return affinity_matrix


    def get_cluster_elements(
        self, cluster_id, type: str = "trajectory"
    ) -> list[Trajectory]:
        if type == "trajectory":
            cluster_elements = [
                traj
                for traj, label in zip(self.trajectories, self.labels)
                if label == cluster_id
            ]
        elif type == "timevalues":
            if self.trajectories[0].timevalues is not None:
                cluster_elements = [
                    traj.timevalues
                    for traj, label in zip(self.trajectories, self.labels)
                    if label == cluster_id
                ]
            else:
                raise AttributeError("Timevalues not in trajectories")

        else:
            cluster_elements = [
                traj.metadata[type]
                for traj, label in zip(self.trajectories, self.labels)
                if label == cluster_id
            ]

        return cluster_elements


    def plot_affinity_matrix_overview(self, axs: np.ndarray[plt.Axes] | None = None):

        """
        Plot a comprehensive overview of the affinity matrix.

        Creates a figure with 4 subplots showing different aspects of the affinity matrix:
        a) The affinity matrix heatmap
        b) Histogram of distances in the matrix
        c) Bar chart of non-repetitive distance values
        d) Average column values

        Args:
            axs (np.ndarray[plt.Axes] | None, optional): Array of 4 axes objects to plot on.
                If None, a new figure and axes are created. Defaults to None.
        """
        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            show_plot = True
        else:
            show_plot = False

        self.cluster_vis.plot_affinity_matrix(self.affinity_matrix,measure_type=self.similarity_measures.measure_type,ax=axs[0])
        axs[0].set_title("a) " + axs[0].get_title())
        self.cluster_vis.plot_distance_matrix_histogram(self.affinity_matrix,ax=axs[1])
        axs[1].set_title("b) " + axs[1].get_title())
        self.cluster_vis.plot_non_repetitive_distances_values_barchart(self.affinity_matrix,ax=axs[2])
        axs[2].set_title("c) " + axs[2].get_title())
        self.cluster_vis.plot_average_column_value(self.affinity_matrix,ax=axs[3])
        axs[3].set_title("d) " + axs[3].get_title())

        # Add title to the figure
        if axs is None:
            fig.suptitle(
                f"Affinity Matrix Overview - {self.similarity_measures.measure_type.capitalize()} measure"
            )
            fig.tight_layout()

        if show_plot:
            plt.show()

    def plot_interactive_overview(self):

        """
        Create an interactive visualization of the affinity matrix and trajectories embeddings.

        Uses Bokeh to create an interactive plot with:
        - Affinity matrix heatmap
        - Trajectory 2D-embeddings visualization

        The plots are displayed side by side in a row.
        """
        p1 = self.cluster_vis.plot_affinity_matrix_bokeh(
            affinity_matrix=self.affinity_matrix, 
            measure_type=self.similarity_measures.measure_type)

        if self.trajectories_centroids.size > 0:
            p2 = self.cluster_vis.plot_trajectories_embedding_bokeh(
                traj_centroids=self.trajectories_centroids,
                labels=self.labels
            )
        else:
            self.trajectories_centroids = self._calculate_trajectory_centroids()
            p2 = self.cluster_vis.plot_trajectories_embedding_bokeh(
                traj_centroids=self.trajectories_centroids,
                labels=self.labels
            )

        show(row(p1, p2))

    def plot_latent_space_overview(self,
                                   axs: np.ndarray[plt.Axes] | None = None,
                                   frame_to_maze: bool = False):

        """
        Plot an overview of the latent space, including trajectory embeddings and cluster centroids.

        This function creates a two-panel matplotlib figure:
        - The first panel shows the 2D embedding of all trajectory centroids, colored by cluster label.
        - The second panel shows the centroids of each cluster, with marker size proportional to the number of trajectories in the cluster.

        Args:
            axs (np.ndarray[plt.Axes] | None, optional): Array of two matplotlib axes to plot on. If None, creates a new figure and axes.
            frame_to_maze (bool, optional): Whether to set axis limits to the maze boundaries. Defaults to False.

        Returns:
            None. Displays the plot or adds to the provided axes.
        """

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            show_plot = True
        else:
            show_plot = False


        if self.trajectories_centroids.size == 0:
            self.trajectories_centroids = self._calculate_trajectory_centroids()

        if self.cluster_centroids.size == 0:
            self.cluster_centroids, self.cluster_sizes = (
                self._calculate_cluster_centroids()
            )

        self.cluster_vis.plot_trajectories_embedding(
            self.trajectories_centroids, 
            self.labels,
            ax=axs[0], 
            frame_to_maze=frame_to_maze
        )
        axs[0].set_title("a) " + axs[0].get_title())

        self.cluster_vis.plot_clusters_centroids(
            self.cluster_centroids,
            self.cluster_sizes,
            self.labels,
            ax=axs[1],
            frame_to_maze=frame_to_maze,
        )
        axs[1].set_title("b) " + axs[0].get_title())

        if axs is None:
            fig.suptitle(
                f"Latent Space Overview"
            )
            fig.tight_layout()

        if show_plot:
            plt.show()

    def plot_cluster_overview(
        self,
        cluster_id: int,
        figsize: tuple[int, int] = (18, 6),
        seed: int = 42,
        darkmode: bool = True,
    ):
        """
        Plot an overview of a specific cluster showing velocity grid, heatmap and sample trajectories.

        This method creates a figure with 6 subplots arranged in a 2x4 grid:
        - Left column (2 rows): Velocity grid and heatmap of the entire cluster
        - Right column (4 rows): 4 randomly selected sample trajectories from the cluster

        Args:
            cluster_id (int): ID of the cluster to visualize
            figsize (tuple[int, int], optional): Figure size as (width, height). Defaults to (18, 6).
            seed (int): seed for randomly selecting cluster trajectories
        """
        cluster_trajectories = [
            traj for traj, l in zip(self.trajectories, self.labels) if l == cluster_id
        ]
        # Create a local random number generator with a fixed seed for reproducibility
        rng = random.Random(42)
        cluster_size = len(cluster_trajectories)
        subset = rng.sample(cluster_trajectories, min(4, cluster_size))

        from src.visualization.game_visualizer import GameVisualizer  # Lazy import

        game_viz = GameVisualizer(darkmode=darkmode)
        fig = plt.figure(figsize=figsize)
        G = GridSpec(2, 4, width_ratios=[2, 2, 1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(G[:, 0])
        ax2 = fig.add_subplot(G[:, 1])
        ax3 = fig.add_subplot(G[0, 2])
        ax4 = fig.add_subplot(G[0, 3])
        ax5 = fig.add_subplot(G[1, 2])
        ax6 = fig.add_subplot(G[1, 3])

        game_viz.plot_velocity_grid(
            trajectory=cluster_trajectories,
            normalize=True,
            ax=ax1,
            title_id=f"Cluster {cluster_id} (n = {cluster_size})",
        )
        game_viz.plot_heatmap(
            trajectory=cluster_trajectories,
            normalize=True,
            ax=ax2,
            title_id=f"Cluster {cluster_id}",
        )
        game_viz.plot_multiple_trajectories(
            trajectories=subset,
            plot_type="line",
            axs=[ax3, ax4, ax5, ax6],
            show_maze=False,
            metadata_label="level_id",
        )

    def _sort_labels(self) -> np.ndarray:
        """
        Sort cluster labels based on the number of trajectories in each cluster and
        remap labels so that the largest cluster has label 0, second largest has label 1, etc.
        Noise points (label -1) remain unchanged.

        Returns:
            np.ndarray: New array of remapped labels
        """
        logger.debug("Sorting and remapping cluster labels by size")

        if self.labels.size == 0:
            logger.warning("No clusters to sort")
            return np.array([])

        unique_labels = np.unique(self.labels)
        cluster_sizes = []

        # Calculate size of each cluster (excluding noise)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_size = np.sum(self.labels == label)
                cluster_sizes.append((label, cluster_size))

        # Sort clusters by size in descending order
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Create mapping from old labels to new labels
        new_labels = np.copy(self.labels)
        for new_label, (old_label, size) in enumerate(cluster_sizes):
            logger.debug(
                f"Remapping cluster {old_label} (size {size}) to label {new_label}"
            )
            new_labels[self.labels == old_label] = new_label

        logger.debug(f"Remapped {len(cluster_sizes)} clusters")
        return new_labels

    def _calculate_trajectory_centroids(self) -> List[np.ndarray]:
        """
        Calculate the geometrical centroids of all trajectories for dimensionality reduction and plotting.

        Returns:
            List[np.ndarray]: List of centroids for each trajectory
        """
        logger.debug("Calculating trajectory centroids")
        centroids = [np.mean(trajectory, axis=0) for trajectory in self.trajectories]
        centroids_array = np.array(centroids)
        if centroids_array.ndim == 1:
            centroids_array = centroids_array.reshape(
                -1, 2
            )  # Reshape to (n_trajectories, 2)
        logger.debug("Trajectory centroids calculated")
        return centroids_array

    def _calculate_cluster_centroids(self) -> np.ndarray:
        """
        Calculate the centroids of all clusters by first calculating the mean of each trajectory,
        then calculating the mean of those means for each cluster.

        Returns:
            np.ndarray: Array of centroids for each cluster, shape (n_clusters, 2)
            np.ndarray: Array of sizes for each cluster, shape (n_clusters,)
        """
        logger.debug("Calculating cluster centroids")
        cluster_centroids = []
        cluster_sizes = []
        for label in np.unique(self.labels):
            if label != -1:  # Skip noise points
                # Get all trajectories in this cluster
                cluster_trajectories = [
                    traj
                    for traj, l in zip(self.trajectories, self.labels)
                    if l == label
                ]
                cluster_sizes.append(len(cluster_trajectories))
                # Calculate mean for each trajectory first
                trajectory_means = np.array(
                    [np.mean(traj, axis=0) for traj in cluster_trajectories]
                )
                # Then calculate the mean of all trajectory means
                cluster_centroid = np.mean(trajectory_means, axis=0)
                cluster_centroids.append(cluster_centroid)

        logger.debug("Cluster centroids calculated")
        return np.array(cluster_centroids), np.array(cluster_sizes)

    def calculate_affinity_matrix_parallel_cpu(
        self,
        trajectories: List[Trajectory] | np.ndarray | List[np.ndarray],
        n_jobs: int = None,
        chunk_size_multiplier: int = 1,
    ) -> np.ndarray:
        """
        Calculate affinity matrix using CPU multiprocessing with optimized chunking for DTW.

        returns affinity matrix
        """
        logger.info("Calculating affinity matrix with CPU parallelization")
        time_start = time.time()

        if n_jobs is None:
            n_jobs = cpu_count()

        num_trajectories = len(trajectories)
        self.affinity_matrix = np.zeros((num_trajectories, num_trajectories))

        # Generate all upper triangular indices
        indices = [
            (i, j)
            for i in range(num_trajectories)
            for j in range(i + 1, num_trajectories)
        ]
        total_pairs = len(indices)

        # Optimize chunk size based on similarity measure
        if self.similarity_measures.measure_type in [
            "dtw",
            "dtw_optimized",
            "EDR",
            "LCSS",
            "frechet",
        ]:
            # For expensive operations, use larger chunks to reduce overhead
            base_chunk_size = max(10, total_pairs // (n_jobs * chunk_size_multiplier))
        else:
            # For cheap operations, use smaller chunks for better load balancing
            base_chunk_size = max(1, total_pairs // (n_jobs * 4))

        chunks = [
            indices[i : i + base_chunk_size]
            for i in range(0, len(indices), base_chunk_size)
        ]

        # Prepare arguments for each chunk
        chunk_args = [
            (self.similarity_measures, trajectories, chunk) for chunk in chunks
        ]

        logger.info(
            f"Processing {total_pairs} pairs using {n_jobs} cores in {len(chunks)} chunks (chunk_size: {base_chunk_size}), with similarity measure: {self.similarity_measures.measure_type}"
        )

        # Process chunks in parallel
        with Pool(n_jobs) as pool:
            with tqdm.tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                chunk_results = []
                for result in pool.map(_compute_distance_chunk, chunk_args):
                    chunk_results.append(result)
                    pbar.update(1)

        # Fill the affinity matrix
        for chunk_result in chunk_results:
            for i, j, distance in chunk_result:
                self.affinity_matrix[i, j] = distance
                self.affinity_matrix[j, i] = distance  # Symmetric matrix

        logger.info(
            f"Parallel affinity matrix calculation complete in {round(time.time() - time_start, 2)} seconds"
        )
        return self.affinity_matrix
