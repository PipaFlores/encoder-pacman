import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import src.utils as utils
from src.config.defaults import config


class BaseVisualizer:
    def __init__(
        self,
        maze_x_min: int = -13.5,
        maze_x_max: int = 13.5,
        maze_y_min: int = -16.5,
        maze_y_max: int = 13.5,
        grid_size_x: int = 28,
        grid_size_y: int = 31,
        figsize: Tuple[int, int] = config.figsize,
    ):
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



    def _plot_walls_and_pellets(
        self,
        walls=True,
        pellets=False,
        return_transformed_positions=False,
        ax: plt.Axes | None = None,
    ):
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
            if ax is None:
                plt.scatter(
                    np.array(wall_x) + 0.5,
                    np.array(wall_y) - 0.5,
                    c="gray",
                    marker="s",
                    s=100,
                    alpha=0.5,
                    label="Walls",
                )
            else:
                ax.scatter(
                    np.array(wall_x) + 0.5,
                    np.array(wall_y) - 0.5,
                    c="gray",
                    marker="s",
                    s=100,
                    alpha=0.5,
                    label="Walls",
                )

        if pellets:
            # Plot pellets
            pellet_x, pellet_y = zip(*pellet_positions)
            if ax is None:
                plt.scatter(
                    np.array(pellet_x) + 0.5,
                    np.array(pellet_y) - 0.5,
                    c="blue",
                    marker="o",
                    s=20,
                    alpha=0.5,
                    label="Pellets",
                )
            else:
                ax.scatter(
                    np.array(pellet_x) + 0.5,
                    np.array(pellet_y) - 0.5,
                    c="blue",
                    marker="o",
                    s=20,
                    alpha=0.5,
                    label="Pellets",
                )

        if return_transformed_positions:
            wall_positions = [(x + 0.5, y - 0.5) for x, y in wall_positions]
            pellet_positions = [(x + 0.5, y - 0.5) for x, y in pellet_positions]

        return wall_positions, pellet_positions
