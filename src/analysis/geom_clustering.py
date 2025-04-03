import numpy as np
from typing import List
from src.analysis.distance_measures import SimilarityMeasures
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from src.visualization.cluster_visualizer import ClusterVisualizer
from src.utils.utils import timer
from src.utils import setup_logger
from bokeh.plotting import show, row
import time

# Initialize module-level logger
logger = setup_logger(__name__)

class GeomClustering:
    def __init__(self, 
                 similarity_measure: str = 'euclidean',
                 cluster_method: str = 'HDBSCAN',
                 verbose: bool = False):
        if verbose:
            logger.setLevel("INFO")
        logger.info(f"Initializing GeomClustering with similarity measure: {similarity_measure}")
        self.trajectories = np.array([])
        self.labels = np.array([])

        self.similarity_measures = SimilarityMeasures(similarity_measure)
        self.cluster_method = cluster_method
        self.affinity_matrix = np.array([])

        self.trajectories_centroids = np.array([])
        self.cluster_centroids = np.array([])
        self.cluster_sizes = np.array([])

    def fit(self, 
            trajectories: List[np.ndarray], 
            cluster_method: str | None = None, 
            **kwargs) -> np.ndarray:
        if cluster_method is None:
            cluster_method = self.cluster_method
        else:
            self.cluster_method = cluster_method

        logger.info(f"Fitting clustering model with {len(trajectories)} trajectories")
        self.trajectories = np.array(trajectories) # Array is (num_trajectories, num_timesteps, 2)
        self.affinity_matrix = self.calculate_affinity_matrix(trajectories)
        self.labels = self.cluster_trajectories(cluster_method, **kwargs)
        self.labels = self._sort_labels()
        logger.info(f"Clustering complete. Found {len(set(self.labels))} clusters")

        self.vis = ClusterVisualizer(self.affinity_matrix, self.labels, self.trajectories, self.similarity_measures.measure_type) # Init visualizer
        return self.labels

    def calculate_affinity_matrix(self, 
                                  trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Calculate affinity matrix between all trajectories in the list.
        
        Args:
            trajectories: List of arrays (x, y) representing trajectories

        Returns:
            np.ndarray: Affinity matrix between all trajectories
        """
        logger.info("Calculating affinity matrix")
        time_start = time.time()
        num_trajectories = len(trajectories)
        self.affinity_matrix = np.zeros((num_trajectories, num_trajectories))

        for i in range(num_trajectories):
            for j in range(i + 1, num_trajectories):
                self.affinity_matrix[i, j] = self.similarity_measures.calculate_distance(trajectories[i], trajectories[j])
                self.affinity_matrix[j, i] = self.affinity_matrix[i, j]

        logger.info(f"Affinity matrix calculation complete in {round(time.time() - time_start, 2)} seconds")
        return self.affinity_matrix
    
    def cluster_trajectories(self,
                             cluster_method: str = 'HDBSCAN', 
                             **kwargs) -> List[np.ndarray]:
        if cluster_method == 'DBSCAN':
            return self._DBSCAN_fit(**kwargs)
        elif cluster_method == 'HDBSCAN':
            return self._HDBSCAN_fit(**kwargs)
        else:
            raise ValueError(f"Invalid cluster method: {cluster_method}")

    def _DBSCAN_fit(self, **kwargs):
        dbscan = DBSCAN(eps=0.5, min_samples=15, metric='precomputed', **kwargs)
        logger.info(f"Starting DBSCAN clustering with eps={dbscan.eps} and min_samples={dbscan.min_samples}")
        time_start = time.time()
        dbscan.fit(self.affinity_matrix)
        logger.info(f"DBSCAN clustering complete in {round(time.time() - time_start, 2)} seconds")
        
        return dbscan.labels_
    
    def _HDBSCAN_fit(self, **kwargs):
        hdbscan = HDBSCAN(min_cluster_size=15, metric='precomputed', **kwargs)
        time_start = time.time()
        logger.info(f"Starting HDBSCAN clustering with min_cluster_size={hdbscan.min_cluster_size} and min_samples={hdbscan.min_samples}")
        hdbscan.fit(self.affinity_matrix)
        logger.info(f"HDBSCAN clustering complete in {round(time.time() - time_start, 2)} seconds")
        return hdbscan.labels_

    ### Affinity Matrix Visualization

    def plot_affinity_matrix_overview(self, axs: np.ndarray[plt.Axes] | None = None):
        """
        Plot a summary of the affinity matrix.
        
        Args:
            axs: Axes object to plot on. If None, a new figure and axes are created.
            array of 4 axes objects.
        """
        if axs is None:
            fig, axs = plt.subplots(1,4, figsize=(24,6))
            show_plot = True
        else:
            show_plot = False
        
        self.plot_affinity_matrix(ax=axs[0])
        axs[0].set_title('a) ' + axs[0].get_title())
        self.plot_distance_matrix_histogram(ax=axs[1])
        axs[1].set_title('b) ' + axs[1].get_title())
        self.plot_non_repetitive_distances_values_barchart(ax=axs[2])
        axs[2].set_title('c) ' + axs[2].get_title())
        self.plot_average_column_value(ax=axs[3])
        axs[3].set_title('d) ' + axs[3].get_title())

        # Add title to the figure
        fig.suptitle(f'Affinity Matrix Overview - {self.similarity_measures.measure_type.capitalize()} measure')
        fig.tight_layout()

        if show_plot:
            plt.show()

    def plot_interactive_overview(self):
        p1 = self.vis.plot_affinity_matrix_bokeh()

        if self.trajectories_centroids.size > 0:
            p2 = self.vis.plot_trajectories_bokeh(traj_centroids=self.trajectories_centroids)
        else:
            self._calculate_trajectory_centroids()
            p2 = self.vis.plot_trajectories_bokeh(traj_centroids=self.trajectories_centroids)

        show(row(p1, p2))

    def plot_affinity_matrix(self, ax: plt.Axes | None = None):
        self.vis.plot_affinity_matrix(ax=ax)

    def plot_distance_matrix_histogram(self, ax: plt.Axes | None = None, **kwargs):
        self.vis.plot_distance_matrix_histogram(ax=ax, **kwargs)

    def plot_non_repetitive_distances_values_barchart(self, ax: plt.Axes | None = None, **kwargs):
        self.vis.plot_non_repetitive_distances_values_barchart(ax=ax, **kwargs)

    def plot_average_column_value(self, ax: plt.Axes | None = None):
        self.vis.plot_average_column_value(ax=ax)


    ### Clustering Results Visualization

    def plot_trajectories(self, ax: plt.Axes | None = None, frame_to_maze: bool = True):
        if self.trajectories_centroids.size == 0:
            self._calculate_trajectory_centroids()
        self.vis.plot_trajectories(self.trajectories_centroids, ax=ax, frame_to_maze=frame_to_maze)

    def plot_cluster_centroids(self, ax: plt.Axes | None = None, frame_to_maze: bool = True):
        if self.cluster_centroids.size == 0:
            self._calculate_cluster_centroids()
        self.vis.plot_cluster_centroids(self.cluster_centroids, self.cluster_sizes, ax=ax, frame_to_maze=frame_to_maze)





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
            logger.debug(f"Remapping cluster {old_label} (size {size}) to label {new_label}")
            new_labels[self.labels == old_label] = new_label
            
        logger.debug(f"Remapped {len(cluster_sizes)} clusters")
        return new_labels
    
    def _calculate_trajectory_centroids(self) -> List[np.ndarray]:
        """
        Calculate the centroids of all trajectories for dimensionality reduction and plotting.
        
        Returns:
            List[np.ndarray]: List of centroids for each trajectory
        """
        logger.debug("Calculating trajectory centroids")
        centroids = [np.mean(trajectory, axis=0) for trajectory in self.trajectories]
        centroids_array = np.array(centroids)
        if centroids_array.ndim == 1:
            centroids_array = centroids_array.reshape(-1, 2)  # Reshape to (n_trajectories, 2)
        logger.debug("Trajectory centroids calculated")
        self.trajectories_centroids = centroids_array
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
                cluster_trajectories = self.trajectories[self.labels == label]
                cluster_sizes.append(len(cluster_trajectories))
                # Calculate mean for each trajectory first
                trajectory_means = np.array([np.mean(traj, axis=0) for traj in cluster_trajectories])
                # Then calculate the mean of all trajectory means
                cluster_centroid = np.mean(trajectory_means, axis=0)
                cluster_centroids.append(cluster_centroid)

        logger.debug("Cluster centroids calculated")
        self.cluster_centroids = np.array(cluster_centroids)
        self.cluster_sizes = np.array(cluster_sizes)
        return np.array(cluster_centroids), np.array(cluster_sizes)
    


