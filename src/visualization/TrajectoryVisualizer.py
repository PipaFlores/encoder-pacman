import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import pandas as pd
from src.utils.utils import load_maze_data, read_data

class TrajectoryVisualizer:
    def __init__(self, data_folder: str):
        """
        Initialize the trajectory visualizer.
        
        Args:
            data_folder: Path to the data folder containing game data
        """
        self.data_folder = data_folder
        # Load maze layout data
        self.wall_positions, self.pellet_positions = load_maze_data()
        
        # Define maze boundaries based on your game data
        self.MAZE_X_MIN, self.MAZE_X_MAX = -14, 14  # Adjust these based on your game coordinates
        self.MAZE_Y_MIN, self.MAZE_Y_MAX = -17, 14

        # Grid resolution for heatmap (higher number = more detailed heatmap)
        self.GRID_SIZE_Y = 31
        self.GRID_SIZE_X = 28
        
 
    def create_trajectory_heatmap(self, 
                                game_id: int,
                                show_maze: bool = True,
                                show_pellet: bool = False,
                                normalize: bool = True,
                                cmap: str = 'YlOrRd') -> None:
        """
        Create a heatmap visualization of a game trajectory.
        
        Args:
            game_id: The ID of the game to visualize
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
            cmap: Colormap to use for the heatmap
        """
        # Load game data
        _, _, _, _, game_data, _ = read_data(self.data_folder, game_list=[game_id])

        if len(game_data) == 0:
            print(f"No data found for game {game_id}")
            return

        # Create grid for heatmap
        x_grid = np.linspace(self.MAZE_X_MIN, self.MAZE_X_MAX, self.GRID_SIZE_X)
        y_grid = np.linspace(self.MAZE_Y_MIN, self.MAZE_Y_MAX, self.GRID_SIZE_Y)
        grid = np.zeros((len(y_grid), len(x_grid)))
        
        # Create wall mask
        wall_mask = np.zeros_like(grid)
        for wall_x, wall_y in self.wall_positions:
            x_idx = np.argmin(np.abs(x_grid - (wall_x + 0.5)))
            y_idx = np.argmin(np.abs(y_grid - (wall_y - 0.5)))
            wall_mask[y_idx, x_idx] = 1
    
        
        # Create heatmap data
        for x, y in zip(game_data['Pacman_X'], game_data['Pacman_Y']):
            x_idx = np.argmin(np.abs(x_grid - x))
            y_idx = np.argmin(np.abs(y_grid - y))
            if not wall_mask[y_idx, x_idx]:  # Only add visits to non-wall cells
                grid[y_idx, x_idx] += 1

        # Set wall areas to minimum value
            grid = np.ma.masked_array(grid, mask=wall_mask)

        # Normalize if requested
        if normalize:
            grid = grid / grid.max()
            
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Plot heatmap
        plt.imshow(grid, extent=[self.MAZE_X_MIN, self.MAZE_X_MAX, 
                               self.MAZE_Y_MIN, self.MAZE_Y_MAX],
                  origin='lower', cmap=cmap, aspect='equal')
        
        if show_maze:
            # Plot walls
            wall_x, wall_y = zip(*self.wall_positions)
            plt.scatter(np.array(wall_x) + 0.5, np.array(wall_y) - 0.5, c='gray', marker='s', s=100, alpha=0.5, label='Walls')
        
        if show_pellet:
            # Plot pellets
            pellet_x, pellet_y = zip(*self.pellet_positions)
            plt.scatter(np.array(pellet_x) + 0.5, np.array(pellet_y) - 0.5 , c='blue', marker='o', s=20, alpha=0.5, label='Pellets')
        
        plt.colorbar(label='Normalized visit frequency' if normalize else 'Visit count')
        plt.title(f'Y max = {self.MAZE_Y_MAX}, Y min = {self.MAZE_Y_MIN}')
        # plt.title(f'Pacman Trajectory Heatmap - Game {game_id}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def create_multi_game_heatmap(self,
                                 game_ids: List[int],
                                 show_maze: bool = True,
                                 show_pellet: bool = False,
                                 normalize: bool = True,
                                 cmap: str = 'YlOrRd') -> None:
        """
        Create a combined heatmap visualization for multiple games.
        
        Args:
            game_ids: List of game IDs to visualize
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
            cmap: Colormap to use for the heatmap
        """
        # Create grid for combined heatmap
        x_grid = np.linspace(self.MAZE_X_MIN, self.MAZE_X_MAX, self.GRID_SIZE)
        y_grid = np.linspace(self.MAZE_Y_MIN, self.MAZE_Y_MAX, self.GRID_SIZE)
        combined_grid = np.zeros((len(y_grid), len(x_grid)))
        
        # Accumulate data from all games
        for game_id in game_ids:
            game_data = self.load_game_context(game_id)
            for x, y in zip(game_data['Pacman_X'], game_data['Pacman_Y']):
                x_idx = np.argmin(np.abs(x_grid - x))
                y_idx = np.argmin(np.abs(y_grid - y))
                combined_grid[y_idx, x_idx] += 1
                
        if normalize:
            combined_grid = combined_grid / combined_grid.max()
            
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_grid, extent=[self.MAZE_X_MIN, self.MAZE_X_MAX, 
                                        self.MAZE_Y_MIN, self.MAZE_Y_MAX],
                  origin='lower', cmap=cmap, aspect='equal')
        
        if show_maze:
            wall_x, wall_y = zip(*self.wall_positions)
            plt.scatter(wall_x, wall_y, c='gray', marker='s', s=100, alpha=0.5, label='Walls')
        if show_pellet:
            pellet_x, pellet_y = zip(*self.pellet_positions)
            plt.scatter(pellet_x, pellet_y, c='white', marker='o', s=20, alpha=0.5, label='Pellets')
            
        plt.colorbar(label='Normalized visit frequency' if normalize else 'Visit count')
        plt.title(f'Combined Pacman Trajectory Heatmap\n({len(game_ids)} games)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()