import numpy as np
from typing import List
from src.analysis.distance_measures import DistanceMeasures
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class GeomClustering:
    def __init__(self, distance_measure: str = 'euclidean'):
        self.distance_measures = DistanceMeasures(distance_measure)
        self.distance_matrix = np.array([])

    def fit(self, trajectories: List[np.ndarray]):
        self.distance_matrix = self.calculate_distance_matrix(trajectories)
        self.clusters = self.cluster_trajectories()
        return self.clusters

    def calculate_distance_matrix(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Calculate distance matrix between all trajectories in the list.
        
        Args:
            trajectories: List of arrays (x, y) representing trajectories

        Returns:
            np.ndarray: Distance matrix between all trajectories
        """
        num_trajectories = len(trajectories)
        self.distance_matrix = np.zeros((num_trajectories, num_trajectories))

        for i in range(num_trajectories):
            for j in range(i + 1, num_trajectories):
                self.distance_matrix[i, j] = self.distance_measures.calculate_distance(trajectories[i], trajectories[j])
                self.distance_matrix[j, i] = self.distance_matrix[i, j]

        return self.distance_matrix
    
    def cluster_trajectories(self) -> List[np.ndarray]:
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
        dbscan.fit(self.distance_matrix)
        return dbscan.labels_

    def plot_distance_matrix(self):
        if self.distance_matrix.size == 0:
            raise ValueError("Distance matrix is not calculated. Please call calculate_distance_matrix first.")
        
        plt.imshow(self.distance_matrix, cmap = 'viridis')
        plt.colorbar()
        plt.show()
