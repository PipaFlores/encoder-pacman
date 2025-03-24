import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import pandas as pd
import torch
import src.analysis as analysis
import src.utils as utils
import logging
from src.visualization.base_visualizer import BaseVisualizer
from src.utils import DataReader



class GameVisualizer(BaseVisualizer):
    """
    Visualizes trajectories and heatmaps of Pacman games, either from a single game or a list of games.
    """
    def __init__(self, data_folder: str, verbose: bool = False):
        """
        Initialize the trajectory visualizer.
        
        Args:
            data_folder: Path to the data folder containing game data
        """
        super().__init__()
        self.logger = logging.getLogger('TrajectoryVisualizer')
        logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

        self.analyzer = analysis.GridAnalyzer()

        self.datareader = DataReader(data_folder=data_folder, read_games_only=True, verbose=verbose)
        

    def game_heatmap(self, 
                          game_id: int | list[int] | None = None,
                          trajectory: torch.Tensor | None = None,
                          show_maze: bool = True,
                          show_pellet: bool = False,
                          normalize: bool = True) -> None:
        """
        Create a heatmap visualization of a game trajectory.
        If multiple games are provided, the heatmap will be a combination of all games.
        
        Args:
            game_id: The ID of the game to visualize
            trajectory: The trajectory to visualize, should be a tensor of shape (N, 2) where N is the number of steps in the trajectory
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
        """

        game_data = self._read_game_data(game_id, trajectory)
        
        self.analyzer.plot_heatmap(x=game_data['Pacman_X'].values, y=game_data['Pacman_Y'].values, 
                                   aggregate=False, walls=show_maze, pellets=show_pellet, normalize=normalize)

    
 

    def trajectory_line_plot(self,
                           game_id: int = None,
                           trajectory: torch.Tensor | None = None,
                           time_step_delta: int = 1,
                           show_maze: bool = True,
                           show_pellet: bool = False,
                           arrow_spacing: int = 5,
                           offset_scale: float = 0.5) -> None:
        """
        Create a line plot with smart trajectory offsetting based on movement direction.
        """
        game_data = self._read_game_data(game_id, trajectory)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axis explicitly
        
        x, y = game_data['Pacman_X'].values, game_data['Pacman_Y'].values
        # Calculate instantaneous velocities 
        dx, dy = utils.calculate_velocities(x, y)

        # Create vector grid
        _, idx_grid = self.analyzer.calculate_recurrence_grid(x, y, calculate_velocities=False)
        
        # Calculate offsets based on movement direction
        x_grid = np.linspace(self.MAZE_X_MIN, self.MAZE_X_MAX, self.GRID_SIZE_X)
        y_grid = np.linspace(self.MAZE_Y_MIN, self.MAZE_Y_MAX, self.GRID_SIZE_Y)
        
        x_offset = np.zeros_like(x)
        y_offset = np.zeros_like(y)
        
        # Calculate offsets to positions
        for i in range(len(x)):
            x_idx = np.argmin(np.abs(x_grid - x[i]))
            y_idx = np.argmin(np.abs(y_grid - y[i]))
            dx_i, dy_i = dx[i], dy[i]
            
            # Apply perpendicular offset based on movement direction
            if len(idx_grid[y_idx, x_idx]) > 0:
                traj_n = sum(i >= idx for idx in idx_grid[y_idx, x_idx])
                if abs(dx_i) > abs(dy_i):  # Mainly horizontal movement
                    y_offset[i] = offset_scale * (traj_n - 1)
                elif abs(dx_i) < abs(dy_i):  # Mainly vertical movement
                    x_offset[i] = offset_scale * (traj_n - 1)


        # Apply offsets
        x = x + x_offset
        y = y + y_offset
        
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
        self._plot_walls_and_pellets(walls=show_maze, pellets=show_pellet)
        
        ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)
        ax.set_title(f'Pacman Trajectory - Game {game_id}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        plt.show()

    def plot_velocity_grid(self,
                           game_id: int = None,
                           trajectory: torch.Tensor | None = None,
                           show_maze: bool = True,
                           show_pellet: bool = False,
                           normalize: bool = False) -> None:
        """
        Plot the velocity grid for a game trajectory.
        """
        # Load game data (similar to heatmap)
        game_data =self._read_game_data(game_id=game_id, trajectory=trajectory)
        
        self.analyzer.plot_velocity_grid(x=game_data['Pacman_X'].values, y=game_data['Pacman_Y'].values, 
                                         aggregate=False, walls=show_maze, pellets=show_pellet, normalize=normalize)

    
    def plot_count_grid(self,
                        game_id: int = None,
                        trajectory: torch.Tensor | None = None,
                        show_maze: bool = True,
                        show_pellet: bool = False,
                        normalize: bool = False) -> None:
        
        """
        Plot the count grid for a game trajectory.
        """
        # Load game data (similar to heatmap)
        game_data = self._read_game_data(game_id=game_id, trajectory=trajectory)

        self.analyzer.plot_count_grid(x=game_data['Pacman_X'].values, y=game_data['Pacman_Y'].values, aggregate=False,
                                      walls=show_maze, pellets=show_pellet, normalize=normalize)

        
        
    def _read_game_data(self, game_id: int | list[int] | None = None, trajectory: torch.Tensor | None = None):
        """
        Read game data from a game ID or trajectory tensor.
        """

        if game_id is None and trajectory is None:
            raise ValueError("Either game_id or trajectory must be provided")

        if game_id is not None:
            game_data = self.datareader.filter_gamestate_data(game_id=game_id)

    
        elif isinstance(trajectory, torch.Tensor):
            game_data = trajectory.cpu().numpy()
            if game_data.ndim == 2:
                game_data = pd.DataFrame(game_data, columns=['Pacman_X', 'Pacman_Y'])
            elif game_data.ndim == 3: # Multiple trajectories
                game_data = pd.DataFrame(game_data.reshape(-1, game_data.shape[-1]), columns=['Pacman_X', 'Pacman_Y'])
            else:
                raise ValueError("Trajectory tensor must have 2 or 3 dimensions")

        return game_data

