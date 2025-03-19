import numpy as np
from typing import Tuple

from src.utils.utils import calculate_velocities


def create_vector_grid(x: np.ndarray,
                        y: np.ndarray,
                        GRID_SIZE_X: int = 28,
                        GRID_SIZE_Y: int = 31,
                        MAZE_X_MIN: int = -14,
                        MAZE_X_MAX: int = 14,
                        MAZE_Y_MIN: int = -17,
                        MAZE_Y_MAX: int = 14,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a grid storing movement vectors and timesteps.
        
        Args:
            x, y: Position arrays
            timesteps: Optional array of timesteps
        
        Returns:
            vector_grid: Grid containing movement vectors (dx, dy)
            count_grid: Grid containing number of trajectories per cell
            idx_grid: Grid containing position indeces that passed through the cell
        """
        # Initialize grids
        vector_grid = np.zeros((GRID_SIZE_Y - 2, GRID_SIZE_X - 2, 2))  # Store (dx, dy)
        count_grid = np.zeros((GRID_SIZE_Y - 2, GRID_SIZE_X - 2))
        idx_grid = np.empty((GRID_SIZE_Y - 2, GRID_SIZE_X - 2), dtype=object)  # Store position indices that passed through the cell
        for i in range(GRID_SIZE_Y - 2):
            for j in range(GRID_SIZE_X - 2):
                idx_grid[i, j] = []
        
        # Create coordinate grids
        x_grid = np.linspace(MAZE_X_MIN + 1, MAZE_X_MAX - 1, GRID_SIZE_X - 2)
        y_grid = np.linspace(MAZE_Y_MIN + 1, MAZE_Y_MAX - 1, GRID_SIZE_Y - 2)
        
        
        dx, dy = calculate_velocities(x, y)

        # Assign vectors to grid cells
        for i in range(len(x)):
            x_idx = np.argmin(np.abs(x_grid - x[i]))
            y_idx = np.argmin(np.abs(y_grid - y[i]))
            # Accumulate vectors only if the positions are non-consecutive in the idx_grid
            if i > 0:
                if (not idx_grid[y_idx, x_idx]) or (idx_grid[y_idx, x_idx][-1] < i-10):
                    vector_grid[y_idx, x_idx] += np.array([dx[i], dy[i]])
                    count_grid[y_idx, x_idx] += 1
                    idx_grid[y_idx, x_idx].append(i)
        # Average vectors and times
        mask = count_grid > 0
        vector_grid[mask] /= count_grid[mask, np.newaxis]
            
        return vector_grid, count_grid, idx_grid