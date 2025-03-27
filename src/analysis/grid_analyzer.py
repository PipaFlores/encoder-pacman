import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

import src.utils as utils
from src.visualization.base_visualizer import BaseVisualizer
from src.config.defaults import config

class GridAnalyzer:
    """
    A class for analyzing and visualizing grid-based data.
    """
    def __init__(self, 
                 figsize: Tuple[int, int] = config.figsize):
        
        # Initialize the visualizer, which contains the maze boundaries and grid size
        self.visualizer = BaseVisualizer(figsize=figsize)
        
        # Create coordinate grids
        self.x_grid = np.linspace(self.visualizer.MAZE_X_MIN, self.visualizer.MAZE_X_MAX, self.visualizer.GRID_SIZE_X)
        self.y_grid = np.linspace(self.visualizer.MAZE_Y_MIN, self.visualizer.MAZE_Y_MAX, self.visualizer.GRID_SIZE_Y)

        # Initialize grids
        self.recurrence_idx_grid = np.empty((self.visualizer.GRID_SIZE_Y, self.visualizer.GRID_SIZE_X), dtype=object)
        self.recurrence_count_grid = np.zeros((self.visualizer.GRID_SIZE_Y, self.visualizer.GRID_SIZE_X))
        self.velocity_grid = np.zeros((self.visualizer.GRID_SIZE_Y, self.visualizer.GRID_SIZE_X, 2))
        



        
    def calculate_recurrence_grid(self,
                               trajectory: np.ndarray,
                               calculate_velocities: bool = True,
                               aggregate: bool = False,
                               normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates a grid storing movement recurrences, i.e. the number of times a position has been visited and in which steps.
        If aggregate is True, the calculations are aggregated over previous calculated grids.
        
        Args:
            trajectory: Array of shape (N, 2) containing x,y coordinates of the trajectory
            calculate_velocities: Whether to calculate velocities
            aggregate: Whether to aggregate over previous calculated grids
            normalize: Whether to normalize the grid
            
        Returns:
            results: Tuple containing:
                recurrence_count_grid: Grid containing number of trajectories per cell
                recurrence_idx_grid: Grid containing position indeces that passed through the cell
                velocity_grid: Grid containing velocities (if calculate_velocities is True)
        
        """
        # Initialize grids if not aggregating or if not already initialized
        if not aggregate:
            self._reset_grids()
            
        x, y = trajectory[:, 0], trajectory[:, 1]
        
        if calculate_velocities:
            dx, dy = utils.calculate_velocities(trajectory=trajectory)

        # Assign vectors to grid cells
        for i in range(len(x)):
            x_idx = np.argmin(np.abs(self.x_grid - x[i]))
            y_idx = np.argmin(np.abs(self.y_grid - y[i]))
            # Accumulate vectors only if the positions are non-consecutive in the idx_grid
            if i > 0:
                if (not self.recurrence_idx_grid[y_idx, x_idx]) or (i - self.recurrence_idx_grid[y_idx, x_idx][-1] > 4):
                    self.recurrence_count_grid[y_idx, x_idx] += 1
                    self.recurrence_idx_grid[y_idx, x_idx].append(i)
                    if calculate_velocities:
                        # because vel. are calculated with prepended x and y values, we need to subtract 1 from the index to get the correct velocity
                        self.velocity_grid[y_idx, x_idx] += np.array([dx[i-1], dy[i-1]]) 

        if normalize:
            self.recurrence_count_grid = self.recurrence_count_grid / self.recurrence_count_grid.max()
            
        results = (self.recurrence_count_grid, self.recurrence_idx_grid)
        if calculate_velocities:
            if normalize:
                self.velocity_grid = self.velocity_grid / self.velocity_grid.max()
            results += (self.velocity_grid,)

        
        
        return results

    def plot_heatmap(self, trajectory: np.ndarray | None = None,
                     aggregate: bool = False, 
                     normalize: bool = False,
                     walls: bool = True,
                     pellets: bool = False):
        """Plot the movement recurrence heatmap.
        Calculates the recurrence grid if trajectory data is provided.
        
        Args:
            trajectory: Array of shape (N, 2) containing x,y coordinates of the trajectory
            aggregate: Whether to aggregate over previous calculated grids
            normalize: Whether to normalize the grid
            walls: Whether to show walls
            pellets: Whether to show pellets
        """
        if trajectory is not None:
            self.calculate_recurrence_grid(trajectory, calculate_velocities=False, aggregate=aggregate, normalize=normalize)

        self.visualizer.plot_heatmap(self.recurrence_count_grid, walls=walls, pellets=pellets)

    def plot_count_grid(self, trajectory: np.ndarray | None = None,
                        aggregate: bool = False,
                        normalize: bool = False,
                        walls: bool = True,
                        pellets: bool = False):
        """Plot both recurrence count grids.
        Calculates the recurrence grid if trajectory data is provided.
        
        Args:
            trajectory: Array of shape (N, 2) containing x,y coordinates of the trajectory
            aggregate: Whether to aggregate over previous calculated grids
            normalize: Whether to normalize the grid
            walls: Whether to show walls
            pellets: Whether to show pellets
        """
        if trajectory is not None:
            self.calculate_recurrence_grid(trajectory, calculate_velocities=False, aggregate=aggregate, normalize=normalize)

        self.visualizer.plot_count_grid(self.recurrence_count_grid, walls=walls, pellets=pellets)

    def plot_velocity_grid(self, trajectory: np.ndarray | None = None,
                           aggregate: bool = False,
                           normalize: bool = False,
                           walls: bool = True,
                           pellets: bool = False):
        """Plot both velocity grids.
        
        Args:
            trajectory: Array of shape (N, 2) containing x,y coordinates of the trajectory
            aggregate: Whether to aggregate over previous calculated grids
            normalize: Whether to normalize the grid
            walls: Whether to show walls
            pellets: Whether to show pellets
        """
        if trajectory is not None or self.velocity_grid.sum() == 0: # Only calculate if trajectory is provided or if the grid is empty
            self.calculate_recurrence_grid(trajectory, calculate_velocities=True, aggregate=aggregate, normalize=normalize)

        self.visualizer.plot_velocity_grid(self.velocity_grid, walls=walls, pellets=pellets)
    
    def _reset_grids(self):
        """Reset all grids to their initial state"""
        self.velocity_grid.fill(0)
        self.recurrence_count_grid.fill(0)
        self._initialize_idx_grid()

    def _initialize_idx_grid(self):
        """Helper method to initialize the idx grid with empty lists"""
        for i in range(self.visualizer.GRID_SIZE_Y):
            for j in range(self.visualizer.GRID_SIZE_X):
                self.recurrence_idx_grid[i, j] = []



