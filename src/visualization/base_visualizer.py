import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import src.utils as utils
from src.config.defaults import config

class BaseVisualizer:
    def __init__(self, 
                 maze_x_min: int = -13.5,
                 maze_x_max: int = 13.5,
                 maze_y_min: int = -16.5,
                 maze_y_max: int = 13.5,
                 grid_size_x: int = 28,
                 grid_size_y: int = 31,
                 figsize: Tuple[int, int] = config.figsize):
        self.MAZE_X_MIN = maze_x_min
        self.MAZE_X_MAX = maze_x_max
        self.MAZE_Y_MIN = maze_y_min
        self.MAZE_Y_MAX = maze_y_max
        self.GRID_SIZE_X = grid_size_x
        self.GRID_SIZE_Y = grid_size_y
        self.figsize = figsize

        # Create coordinate grids
        self.x_grid = np.linspace(self.MAZE_X_MIN, self.MAZE_X_MAX, self.GRID_SIZE_X)
        self.y_grid = np.linspace(self.MAZE_Y_MIN, self.MAZE_Y_MAX, self.GRID_SIZE_Y)

    def plot_count_grid(self, count_grid: np.ndarray, walls = True, pellets = False) -> None:
        """Plot the count grid."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.MAZE_X_MIN - 0.5, self.MAZE_X_MAX + 0.5)
        ax.set_ylim(self.MAZE_Y_MIN - 0.5, self.MAZE_Y_MAX + 0.5)
        
        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                if count_grid[i, j] != 0:
                    ax.text(self.x_grid[j], self.y_grid[i], int(count_grid[i, j]), 
                            ha='center', va='center', color='black', 
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory Count Grid')

        self._plot_walls_and_pellets(walls, pellets)
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_heatmap(self, count_grid: np.ndarray, cmap: str = 'YlOrRd', walls = True, pellets = False) -> None:
        """Plot the heatmap."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(count_grid, extent=[self.MAZE_X_MIN - 0.5, self.MAZE_X_MAX + 0.5,
                                      self.MAZE_Y_MIN - 0.5, self.MAZE_Y_MAX + 0.5],
                                      origin='lower',
                                      cmap= cmap,
                                      aspect='equal')

        self._plot_walls_and_pellets(walls, pellets)
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_velocity_grid(self, velocity_grid: np.ndarray, walls = True, pellets = False) -> None:
        """Plot the vector grid."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.MAZE_X_MIN - 0.5, self.MAZE_X_MAX + 0.5)
        ax.set_ylim(self.MAZE_Y_MIN - 0.5, self.MAZE_Y_MAX + 0.5)
        
        walls_positions, pellets_positions = self._plot_walls_and_pellets(walls, pellets, return_transformed_positions=True)

        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                if (self.x_grid[j],self.y_grid[i]) not in walls_positions:
                    ax.arrow(self.x_grid[j], self.y_grid[i], 
                         velocity_grid[i, j, 0], velocity_grid[i, j, 1], 
                         head_width=0.2, head_length=0.2, fc='red', ec='red')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Velocity Grid')
    
        plt.grid(True, alpha=0.3)
        plt.show()

    
    def _plot_walls_and_pellets(self, walls = True, pellets = False, return_transformed_positions = False):
        """
        Plots the walls and pellets on the maze.

        Args:
            walls: Boolean indicating whether to plot walls.
            pellets: Boolean indicating whether to plot pellets.
            return_transformed_positions: Boolean indicating whether to return the transformed positions.
            useful for plotting the velocity grid with masked positions.

        Note:
            The wall positions are transformed by adding 0.5 to the x-coordinates and subtracting 0.5 from the y-coordinates.
            The pellet positions are transformed by adding 0.5 to the x-coordinates and subtracting 0.5 from the y-coordinates.
        """
        wall_positions, pellet_positions = utils.load_maze_data()

        if walls:
            # Plot walls
            wall_x, wall_y = zip(*wall_positions)
            plt.scatter(np.array(wall_x) + 0.5, np.array(wall_y) - 0.5, c='gray', marker='s', s=100, alpha=0.5, label='Walls')
        
        if pellets:
            # Plot pellets
            pellet_x, pellet_y = zip(*pellet_positions)
            plt.scatter(np.array(pellet_x) + 0.5, np.array(pellet_y) - 0.5 , c='blue', marker='o', s=20, alpha=0.5, label='Pellets')

        if return_transformed_positions:
            wall_positions = [(x + 0.5, y - 0.5) for x, y in wall_positions]
            pellet_positions = [(x + 0.5, y - 0.5) for x, y in pellet_positions]

        return wall_positions, pellet_positions
    