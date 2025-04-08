import numpy as np
from typing import Dict, Callable
import matplotlib.pyplot as plt

from src.utils import calculate_velocities
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimilarityMeasures:
    """
    A class for calculating different similarity measures between trajectories.
    """

    MEASURES = ["euclidean", "manhattan", "dtw", "EDR", "LCSS", "hausdorff", "frechet"]

    @property
    def MEASURE_FUNCTIONS(self) -> Dict[str, Callable]:
        """Dictionary mapping measure names to their calculation functions."""
        return {
            "euclidean": self.calculate_euclidean_distance,
            "manhattan": self.calculate_manhattan_distance,
            "dtw": self.calculate_dtw_distance,
            "EDR": self.calculate_edr_distance,
            "LCSS": self.calculate_lcss_distance,
            "hausdorff": self.calculate_hausdorff_distance,
            "frechet": self.calculate_frechet_distance,
        }

    def __init__(self, measure_type: str = "euclidean"):
        """
        Initialize the SimilarityMeasures class.

        Args:
            measure_type: Type of distance measure to use

        Raises:
            ValueError: If measure_type is not a valid measure
        """
        if measure_type not in self.MEASURES:
            raise ValueError(
                f"Invalid measure type: {measure_type}. Must be one of: {', '.join(self.MEASURES)}"
            )
        self.measure_type = measure_type

    def calculate_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> float:
        """
        Calculate distance between two trajectories using the specified measure.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            float: Distance between trajectory1 and trajectory2
        """
        return self.MEASURE_FUNCTIONS[self.measure_type](trajectory1, trajectory2)

    def calculate_euclidean_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> float:
        """
        Calculate Euclidean distance between corresponding points in two trajectories.

        Args:
            trajectory1: Tuple of arrays (x1, y1) representing the first trajectory
            trajectory2: Tuple of arrays (x2, y2) representing the second trajectory

        Returns:
            float: Euclidean distance between trajectory1 and trajectory2
        """
        x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
        x2, y2 = trajectory2[:, 0], trajectory2[:, 1]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).sum()

    def calculate_manhattan_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> float:
        """
        Calculate Manhattan distance between corresponding points in two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            float: Manhattan distance between trajectory1 and trajectory2
        """
        x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
        x2, y2 = trajectory2[:, 0], trajectory2[:, 1]
        return (np.abs(x2 - x1) + np.abs(y2 - y1)).sum()

    def calculate_dtw_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray, _return_matrix: bool = False
    ) -> float:
        """
        Calculate Dynamic Time Warping (DTW) distance between two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            _return_matrix: Whether to return the DTW matrix instead of the distance

        Returns:
            float: DTW distance between trajectory1 and trajectory2
        """
        n = len(trajectory1)
        m = len(trajectory2)
    
        # Create cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill the cost matrix with cumulative distances
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Calculate Euclidean distance between points
                cost = np.sqrt((trajectory1[i-1, 0] - trajectory2[j-1, 0])**2 + (trajectory1[i-1, 1] - trajectory2[j-1, 1])**2)
                # Update DTW matrix
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        if _return_matrix:
            return dtw_matrix
        else:
            return dtw_matrix[-1, -1] # last element of the matrix, total distance
        

    def _plot_dtw_path(self, trajectory1, trajectory2, ax: plt.Axes | None = None):
        """
        Plot the DTW path between two trajectories, for illustration purposes.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            ax: Axes object to plot the path on

        Returns:
            None
        """
        DTW_matrix = self.calculate_dtw_distance(trajectory1, trajectory2, _return_matrix=True)

        n = DTW_matrix.shape[0] - 1
        m = DTW_matrix.shape[1] - 1

        Path = [(n,m)]

        while n > 0 or m > 0:
            val = min(DTW_matrix[n-1,m-1], DTW_matrix[n-1,m], DTW_matrix[n, m-1])
            if val == DTW_matrix[n-1,m-1]:
                min_val = (n-1, m-1)
            elif val == DTW_matrix[n-1,m]:
                min_val = (n-1, m)
            elif val == DTW_matrix[n,m-1]:
                min_val = (n,m-1)
            (n,m) = min_val
            if n > 0 and m > 0:
                Path.append(min_val)

        Path.reverse()
        Path = np.array(Path)

        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
        else:
            show_plot = False

        ax.set_title("DTW Path")
        ax.set_xlabel("Trajectory 2")
        ax.set_ylabel("Trajectory 1")

        for i in range(DTW_matrix.shape[0]):
            for j in range(DTW_matrix.shape[1]):
                text = ax.text(j, i, f"{DTW_matrix[i,j]:.1f}", ha="center", va="center", color="white")
        
        ax.imshow(DTW_matrix, cmap='viridis')
        ax.plot(Path[:, 1], Path[:, 0], 'r-', label='DTW Path') # Rows are vertical movement, columns are horizontal movement

        if show_plot:
            plt.show()
        

    def calculate_edr_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Edit Distance on Real sequences (EDR) between two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            np.ndarray: Array of EDR distances between corresponding points in the trajectories
        """
        return NotImplementedError

    def calculate_lcss_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Longest Common Subsequence (LCSS) distance between two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            np.ndarray: Array of LCSS distances between corresponding points in the trajectories
        """
        return NotImplementedError

    def calculate_hausdorff_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Hausdorff distance between two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            np.ndarray: Array of Hausdorff distances between corresponding points in the trajectories
        """
        return NotImplementedError

    def calculate_frechet_distance(
        self, trajectory1: np.ndarray, trajectory2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Fréchet distance between two trajectories.

        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory

        Returns:
            np.ndarray: Array of Fréchet distances between corresponding points in the trajectories
        """
        return NotImplementedError

    def calculate_path_length(self, trajectory: np.ndarray) -> float:
        """
        Calculate the total path length of a trajectory.

        Args:
            trajectory: Array of x-coordinates and y-coordinates

        Returns:
            float: Total path length
        """
        dx, dy = calculate_velocities(trajectory)
        return np.sum(np.sqrt(dx**2 + dy**2))
