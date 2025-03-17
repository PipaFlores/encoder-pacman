import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import pandas as pd
import torch
from src.utils.utils import load_maze_data, read_data

class TrajectoryVisualizer:
    def __init__(self, data_folder: str = None):
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
        
 
    def trajectory_heatmap(self, 
                          game_id: int = None,
                          trajectory: torch.Tensor | None = None,
                          show_maze: bool = True,
                          show_pellet: bool = False,
                          normalize: bool = True,
                          cmap: str = 'YlOrRd') -> None:
        """
        Create a heatmap visualization of a game trajectory.
        
        Args:
            game_id: The ID of the game to visualize
            trajectory: The trajectory to visualize, should be a tensor of shape (N, 2) where N is the number of steps in the trajectory
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
            cmap: Colormap to use for the heatmap
        """
        # Load game data

        if self.data_folder is not None and game_id is not None:
            _, _, _, _, game_data, _ = read_data(self.data_folder, game_list=[game_id])

        elif isinstance(trajectory, torch.Tensor):
            game_data = trajectory.cpu().numpy()
            game_data = pd.DataFrame(game_data, columns=['Pacman_X', 'Pacman_Y'])

        else:
            raise ValueError("Either game_id or trajectory must be provided")

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
        
        self.plot_walls_and_pellets_(walls=show_maze, pellets=show_pellet)
        
        plt.colorbar(label='Normalized visit frequency' if normalize else 'Visit count')
        
        plt.title(f'Pacman Trajectory Heatmap - Game {game_id}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def multi_game_heatmap(self,
                                 game_ids: List[int] = None,
                                 trajectories: torch.Tensor | None = None,
                                 show_maze: bool = True,
                                 show_pellet: bool = False,
                                 normalize: bool = True,
                                 cmap: str = 'YlOrRd') -> None:
        """
        Create a combined heatmap visualization for multiple games.
        
        Args:
            game_ids: List of game IDs to visualize
            trajectories: List of trajectories to visualize, should be a tensor of shape (M, N, 2) where M is the number of trajectories and N is the number of steps in the trajectory
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
            cmap: Colormap to use for the heatmap
        """

        if self.data_folder is not None and game_ids is not None:
            _, _, _, _, game_data, _ = read_data(self.data_folder, game_list=game_ids)

        elif isinstance(trajectories, torch.Tensor):
            game_data = trajectories.cpu().numpy()
            game_data = pd.DataFrame(game_data.reshape(-1, 2), columns=['Pacman_X', 'Pacman_Y'])
            # Remove rows where both Pacman_X and Pacman_Y are 0
            game_data = game_data[(game_data['Pacman_X'] != 0) | (game_data['Pacman_Y'] != 0)]

        else:
            raise ValueError("Either game_ids or trajectories must be provided")

        # Create grid for combined heatmap
        x_grid = np.linspace(self.MAZE_X_MIN, self.MAZE_X_MAX, self.GRID_SIZE_X)
        y_grid = np.linspace(self.MAZE_Y_MIN, self.MAZE_Y_MAX, self.GRID_SIZE_Y)
        combined_grid = np.zeros((len(y_grid), len(x_grid)))
        
        # Accumulate data from all games
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
        
        self.plot_walls_and_pellets_(walls=show_maze, pellets=show_pellet)

        n_games = len(game_data['game_id'].unique()) if game_ids is not None else trajectories.shape[0]
            
        plt.colorbar(label='Normalized visit frequency' if normalize else 'Visit count')
        plt.title(f'Combined Pacman Trajectory Heatmap\n({n_games} games)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def trajectory_line_plot(self,
                           game_id: int = None,
                           trajectory: torch.Tensor | None = None,
                           time_step_delta: int = 1,
                           show_maze: bool = True,
                           show_pellet: bool = False,
                           arrow_spacing: int = 5) -> None:
        """
        Create a line plot of trajectories with time-colored segments and directional arrows.
        
        Args:
            game_id: The ID of the game to visualize
            trajectory: Tensor of shape (N, 2) containing x,y coordinates
            time_step_delta: Number of time steps between points to connect (each time step is 50ms)
            show_maze: Whether to show maze layout
            show_pellet: Whether to show pellets
            arrow_spacing: Place arrows every N segments
        """
        # Load game data (similar to heatmap)
        if self.data_folder is not None and game_id is not None:
            _, _, _, _, game_data, _ = read_data(self.data_folder, game_list=[game_id])
        elif isinstance(trajectory, torch.Tensor):
            game_data = trajectory.cpu().numpy()
            game_data = pd.DataFrame(game_data, columns=['Pacman_X', 'Pacman_Y'])
        else:
            raise ValueError("Either game_id or trajectory must be provided")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axis explicitly
        
        # Get positions
        x = game_data['Pacman_X'].values
        y = game_data['Pacman_Y'].values
        
        # Create segments with time_step_delta
        points = np.array([x, y]).T
        segments = np.array([points[i:i+time_step_delta+1] 
                           for i in range(0, len(points)-time_step_delta, time_step_delta)])
        
        # Create a colormap based on time
        norm = plt.Normalize(0, len(segments))
        colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        
        # Plot line segments with arrows
        for i, segment in enumerate(segments):
            ax.plot(segment[:, 0], segment[:, 1], color=colors[i], linewidth=2)
            if i % arrow_spacing == 0 and len(segment) > 1:
                mid = min(len(segment) // 2, len(segment) - 2)
                dx = segment[mid+1, 0] - segment[mid, 0]
                dy = segment[mid+1, 1] - segment[mid, 1]
                ax.arrow(segment[mid, 0], segment[mid, 1], dx/2, dy/2,
                        head_width=0.3, head_length=0.5, fc=colors[i], ec=colors[i])
        
        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])  # Set array for the scalar mappable
        plt.colorbar(sm, ax=ax, label='Time')  # Specify the axis for the colorbar
        
        # Plot maze layout
        self.plot_walls_and_pellets_(walls=show_maze, pellets=show_pellet)
        
        ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)
        ax.set_title(f'Pacman Trajectory - Game {game_id}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        plt.show()

    def plot_walls_and_pellets_(self, walls = True, pellets = False):
        """
        Plots the walls and pellets on the maze.

        Args:
            walls: Boolean indicating whether to plot walls.
            pellets: Boolean indicating whether to plot pellets.

        Note:
            The wall positions are transformed by adding 0.5 to the x-coordinates and subtracting 0.5 from the y-coordinates.
            The pellet positions are transformed by adding 0.5 to the x-coordinates and subtracting 0.5 from the y-coordinates.
        """

        if walls:
            # Plot walls
            wall_x, wall_y = zip(*self.wall_positions)
            plt.scatter(np.array(wall_x) + 0.5, np.array(wall_y) - 0.5, c='gray', marker='s', s=100, alpha=0.5, label='Walls')
        
        if pellets:
            # Plot pellets
            pellet_x, pellet_y = zip(*self.pellet_positions)
            plt.scatter(np.array(pellet_x) + 0.5, np.array(pellet_y) - 0.5 , c='blue', marker='o', s=20, alpha=0.5, label='Pellets')
