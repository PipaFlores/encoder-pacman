import numpy as np
from typing import Tuple, List, Optional, Literal, Dict, Callable

from src.utils import calculate_velocities
from src.config.defaults import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SimilarityMeasures:
    """
    A class for calculating different similarity measures between trajectories.
    """
    MEASURES = ['euclidean', 'manhattan', 'dtw', 'EDR', 'LCSS', 'hausdorff', 'frechet']

    @property
    def MEASURE_FUNCTIONS(self) -> Dict[str, Callable]:
        """Dictionary mapping measure names to their calculation functions."""
        return {
            'euclidean': self.calculate_euclidean_distance,
            'manhattan': self.calculate_manhattan_distance,
            'dtw': self.calculate_dtw_distance,
            'EDR': self.calculate_edr_distance,
            'LCSS': self.calculate_lcss_distance,
            'hausdorff': self.calculate_hausdorff_distance,
            'frechet': self.calculate_frechet_distance,
        }
    
    def __init__(self, measure_type: str = 'euclidean'):
        """
        Initialize the SimilarityMeasures class.
        
        Args:
            measure_type: Type of distance measure to use
            
        Raises:
            ValueError: If measure_type is not a valid measure
        """
        if measure_type not in self.MEASURES:
            raise ValueError(f"Invalid measure type: {measure_type}. Must be one of: {', '.join(self.MEASURES)}")
        self.measure_type = measure_type

    def calculate_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate distance between two trajectories using the specified measure.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of distances between corresponding points in the trajectories
        """
        return self.MEASURE_FUNCTIONS[self.measure_type](trajectory1, trajectory2)

    def calculate_euclidean_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between corresponding points in two trajectories.
        
        Args:
            trajectory1: Tuple of arrays (x1, y1) representing the first trajectory
            trajectory2: Tuple of arrays (x2, y2) representing the second trajectory
            
        Returns:
            np.ndarray: Array of Euclidean distances between corresponding points in the trajectories
        """
        x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
        x2, y2 = trajectory2[:, 0], trajectory2[:, 1]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2).sum()
    

    def calculate_manhattan_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Manhattan distance between corresponding points in two trajectories.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of Manhattan distances between corresponding points in the trajectories
        """
        x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
        x2, y2 = trajectory2[:, 0], trajectory2[:, 1]
        return (np.abs(x2 - x1) + np.abs(y2 - y1)).sum()
    
    def calculate_dtw_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Dynamic Time Warping (DTW) distance between two trajectories.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of DTW distances between corresponding points in the trajectories
        """
        return NotImplementedError
    
    def calculate_edr_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Edit Distance on Real sequences (EDR) between two trajectories.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of EDR distances between corresponding points in the trajectories
        """
        return NotImplementedError
    
    def calculate_lcss_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Longest Common Subsequence (LCSS) distance between two trajectories.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of LCSS distances between corresponding points in the trajectories
        """
        return NotImplementedError
    
    def calculate_hausdorff_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
        """
        Calculate Hausdorff distance between two trajectories.
        
        Args:
            trajectory1: Array of x-coordinates and y-coordinates representing the first trajectory
            trajectory2: Array of x-coordinates and y-coordinates representing the second trajectory
            
        Returns:
            np.ndarray: Array of Hausdorff distances between corresponding points in the trajectories
        """
        return NotImplementedError
    
    def calculate_frechet_distance(self, trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
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
