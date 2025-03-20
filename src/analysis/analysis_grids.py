import numpy as np
from typing import Tuple

import src.utils as utils


class GridAnalyzer:
    def __init__(self, 
                 grid_size_x: int = 28,
                 grid_size_y: int = 31,
                 maze_x_min: int = -14,
                 maze_x_max: int = 14,
                 maze_y_min: int = -17,
                 maze_y_max: int = 14):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.maze_x_min = maze_x_min
        self.maze_x_max = maze_x_max
        self.maze_y_min = maze_y_min
        self.maze_y_max = maze_y_max
        
        # Create coordinate grids
        self.x_grid = np.linspace(self.maze_x_min + 1, self.maze_x_max - 1, self.grid_size_x - 2)
        self.y_grid = np.linspace(self.maze_y_min + 1, self.maze_y_max - 1, self.grid_size_y - 2)

        # Initialize grids
        self.recurrence_idx_grid = np.empty((self.grid_size_y - 2, self.grid_size_x - 2), dtype=object)
        self.recurrence_count_grid = np.zeros((self.grid_size_y - 2, self.grid_size_x - 2))
        self.velocity_grid = np.zeros((self.grid_size_y - 2, self.grid_size_x - 2, 2))
        



        
    def calculate_recurrence_grid(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               calculate_velocities: bool = True,
                               aggregate: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates a grid storing movement recurrences, i.e. the number of times a position has been visited and in which steps.
        If aggregate is True, the calculations are aggregated over previous calculated grids.
        
        Args:
            x, y: Position arrays
            timesteps: Optional array of timesteps
            calculate_velocities: Whether to calculate velocities
            aggregate: Whether to aggregate over previous calculated grids
        
        Returns:
            results: Tuple containing:
                recurrence_count_grid: Grid containing number of trajectories per cell
                recurrence_idx_grid: Grid containing position indeces that passed through the cell
                velocity_grid: Grid containing velocities (if calculate_velocities is True)
        
        """
        # Initialize grids if not aggregating or if not already initialized
        if not aggregate:
            self.reset_grids()
            
        if calculate_velocities:
            dx, dy = utils.calculate_velocities(x, y)

        # Assign vectors to grid cells
        for i in range(len(x)):
            x_idx = np.argmin(np.abs(self.x_grid - x[i]))
            y_idx = np.argmin(np.abs(self.y_grid - y[i]))
            # Accumulate vectors only if the positions are non-consecutive in the idx_grid
            if i > 0:
                if (not self.recurrence_idx_grid[y_idx, x_idx]) or (self.recurrence_idx_grid[y_idx, x_idx][-1] < i-10):
                    self.recurrence_count_grid[y_idx, x_idx] += 1
                    self.recurrence_idx_grid[y_idx, x_idx].append(i)
                    if calculate_velocities:
                        self.velocity_grid[y_idx, x_idx] += np.array([dx[i], dy[i]])

        results = (self.recurrence_count_grid, self.recurrence_idx_grid)
        if calculate_velocities:
            results += (self.velocity_grid,)
        
        return results


    def calculate_velocity_grid(self, x: np.ndarray, y: np.ndarray, aggregate: bool = False) -> np.ndarray:
        """
        Create a grid storing movement vectors.
        If aggregate is True, the calculations are aggregated over previous calculated grids.

        Args:
            x, y: Position arrays
            aggregate: Whether to aggregate over previous calculated grids
        
        Returns:
            velocity_grid: Grid containing velocities

            
        Example:
            ```python
            analyzer = GridAnalyzer()
            x = np.array([1, 2, 3, 4, 5])
            y = np.array([1, 2, 3, 4, 5])
            aggregate = False
            velocity_grid = analyzer.calculate_velocity_grid(x, y, aggregate)

            #If aggregate is True, the calculations are aggregated over previous calculated grids.
            #This is useful for calculating the vector grid for a moving average of the velocities.

            # Example for a batch of trajectories
            x_batch = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            y_batch = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            for x, y in zip(x_batch, y_batch):
                velocity_grid = analyzer.calculate_velocity_grid(x, y, aggregate)
            ```

        """
        if not aggregate:
            self.reset_grids()

        dx, dy = utils.calculate_velocities(x, y)

        for i in range(0, len(x), 10): # Only every 10th step to avoid cluttering the grid
            x_idx = np.argmin(np.abs(self.x_grid - x[i]))
            y_idx = np.argmin(np.abs(self.y_grid - y[i]))
            self.velocity_grid[y_idx, x_idx] += np.array([dx[i], dy[i]])

        return self.velocity_grid

    def reset_grids(self):
        """Reset all grids to their initial state"""
        self.velocity_grid.fill(0)
        self.recurrence_count_grid.fill(0)
        self._initialize_idx_grid()

    def _initialize_idx_grid(self):
        """Helper method to initialize the idx grid with empty lists"""
        for i in range(self.grid_size_y - 2):
            for j in range(self.grid_size_x - 2):
                self.recurrence_idx_grid[i, j] = []


    def analyze_grid_patterns(self, vector_grid, count_grid):
        # Additional analysis methods
        pass