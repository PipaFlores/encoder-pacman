import numpy as np
from typing import List
from src.analysis.distance_measures import SimilarityMeasures
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.utils.utils import timer
from src.utils import setup_logger

# Initialize module-level logger
logger = setup_logger(__name__)

class GeomClustering:
    def __init__(self, similarity_measure: str = 'euclidean', verbose: bool = False):
        if verbose:
            logger.setLevel("DEBUG")
        logger.info(f"Initializing GeomClustering with similarity measure: {similarity_measure}")
        self.trajectories = np.array([])
        self.labels = np.array([])

        self.similarity_measures = SimilarityMeasures(similarity_measure)
        self.distance_matrix = np.array([])

    def fit(self, trajectories: List[np.ndarray]):
        self.trajectories = np.array(trajectories)
        logger.info(f"Fitting clustering model with {len(trajectories)} trajectories")
        self.distance_matrix = self.calculate_distance_matrix(trajectories)
        self.labels = self.cluster_trajectories()
        self.labels = self._sort_labels()
        logger.info(f"Clustering complete. Found {len(set(self.labels))} clusters")
        return self.labels

    def calculate_distance_matrix(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Calculate distance matrix between all trajectories in the list.
        
        Args:
            trajectories: List of arrays (x, y) representing trajectories

        Returns:
            np.ndarray: Distance matrix between all trajectories
        """
        logger.debug("Calculating distance matrix")
        num_trajectories = len(trajectories)
        self.distance_matrix = np.zeros((num_trajectories, num_trajectories))

        for i in range(num_trajectories):
            for j in range(i + 1, num_trajectories):
                self.distance_matrix[i, j] = self.similarity_measures.calculate_distance(trajectories[i], trajectories[j])
                self.distance_matrix[j, i] = self.distance_matrix[i, j]

        logger.debug("Distance matrix calculation complete")
        return self.distance_matrix
    
    def cluster_trajectories(self) -> List[np.ndarray]:
        logger.debug("Starting DBSCAN clustering")
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
        dbscan.fit(self.distance_matrix)
        logger.debug("DBSCAN clustering complete")
        
        return dbscan.labels_

    def plot_distance_matrix(self):
        if self.distance_matrix.size == 0:
            logger.error("Distance matrix is not calculated")
            raise ValueError("Distance matrix is not calculated. Please call calculate_distance_matrix first.")
        
        logger.debug("Plotting distance matrix")
        plt.imshow(self.distance_matrix, cmap = 'viridis')
        plt.colorbar(label=f'{self.similarity_measures.measure_type.capitalize()} Distance')
        plt.title('Distance Matrix')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Trajectory Index')
        plt.show()

    def plot_distance_matrix_histogram(self, **kwargs):
        logger.debug("Plotting distance matrix histogram")
        plt.figure(figsize=(10, 6))
        distances = self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)]
        plt.hist(distances, bins=200, edgecolor='black', **kwargs)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Matrix Histogram')
        plt.show()

    def plot_trajectories(self):
        logger.debug("Plotting trajectories")
        centroids = self._calculate_trajectory_centroids()
        # Create a custom colormap that maps -1 to gray
        cmap = plt.cm.viridis
        cmap.set_under('gray')
        scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=self.labels, cmap=cmap, vmin=0)
        plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        plt.title('Trajectory Clusters')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    def plot_cluster_centroids(self):
        logger.debug("Plotting cluster centroids")
        centroids, sizes = self._calculate_cluster_centroids()
        # Create a custom colormap that maps -1 to gray
        unique_labels = np.unique(self.labels)
        scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=unique_labels[1:], s=sizes)
        plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        handle, labels = scatter.legend_elements(prop = "sizes", alpha = 0.5)
        plt.legend(handle, labels, title="Cluster Sizes", loc="lower left")
        plt.title('Cluster Centroids')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()


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
        logger.debug("Trajectory centroids calculated")
        return np.array(centroids)
    
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
        return np.array(cluster_centroids), np.array(cluster_sizes)
    


