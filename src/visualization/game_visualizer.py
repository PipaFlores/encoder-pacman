import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from src.analysis.utils.grid_analyzer import GridAnalyzer
import src.utils as utils
import logging
from src.visualization.base_visualizer import BaseVisualizer
from src.datahandlers import PacmanDataReader, Trajectory
from src.config.defaults import config


class GameVisualizer(BaseVisualizer):
    """
    Visualizes trajectories and heatmaps of Pacman games, either from a single game or a list of games.
    """

    def __init__(
        self,
        data_folder: str = None,
        verbose: bool = False,
        figsize: Tuple[int, int] = config.figsize,
    ):
        """
        Initialize the trajectory visualizer.

        Args:
            data_folder: Path to the data folder containing game data, if visualizing by game or user id.
            verbose: Whether to print verbose output
            figsize: Size of the figure (width, height). Default is (6, 6).
            if multiplot is used, the figsize will be multiplied by the number of plots.
        """
        super().__init__(figsize=figsize)
        self.logger = logging.getLogger("GameVisualizer")
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

        self.analyzer = GridAnalyzer(
            MAZE_X_MIN=self.MAZE_X_MIN,
            MAZE_X_MAX=self.MAZE_X_MAX,
            MAZE_Y_MIN=self.MAZE_Y_MIN,
            MAZE_Y_MAX=self.MAZE_Y_MAX,
            GRID_SIZE_X=self.GRID_SIZE_X,
            GRID_SIZE_Y=self.GRID_SIZE_Y,
        )

        if data_folder is None:
            self.datareader = None
        else:
            self.datareader = PacmanDataReader(
                data_folder=data_folder, read_games_only=True, verbose=verbose
            )

    def plot_heatmap(
        self,
        game_id: int | list[int] | None = None,
        trajectory: Trajectory | torch.Tensor | np.ndarray | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = True,
        ax: plt.Axes | None = None,
        title_id: int | None = None,
        metadata_label: str |list[str] | None = None
    ) -> None:
        """
        Create a heatmap visualization of a game trajectory.
        If multiple games are provided, the heatmap will be a combination of all games.

        Args:
            game_id: The ID of the game to visualize
            trajectory: The trajectory to visualize, should be a tensor/array of shape (N, 2) where N is the number of steps
            show_maze: Whether to show maze walls and pellets
            normalize: Whether to normalize the heatmap values
        """
        if game_id is not None:
            trajectory = self.datareader.get_trajectory(game_id=game_id)
            title_id = f"game {game_id}"
        elif trajectory is not None:
            trajectory = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either game_id or trajectory must be provided")


        self.analyzer.calculate_recurrence_grid(
            trajectory=trajectory,
            calculate_velocities=False,
            aggregate=False,
            normalize=normalize,
        )

        """Plot the heatmap."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        ax.imshow(
            self.analyzer.recurrence_count_grid,
            extent=[
                self.MAZE_X_MIN - 0.5,
                self.MAZE_X_MAX + 0.5,
                self.MAZE_Y_MIN - 0.5,
                self.MAZE_Y_MAX + 0.5,
            ],
            origin="lower",
            cmap="YlOrRd",
            aspect="equal",
        )

        self._plot_walls_and_pellets(show_maze, show_pellet, ax=ax)
        ax.grid(True, alpha=0.3)

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = [metadata_label] if isinstance(metadata_label, str) else metadata_label
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(
            f"Trajectory Heatmap - {title_id if title_id is not None else ' '}"
        )
        if show_plot:
            plt.show()



    def plot_trajectory_line(
        self,
        game_id: int | None = None,
        trajectory: Trajectory | torch.Tensor | np.ndarray | None = None,
        time_step_delta: int = 1,
        show_maze: bool = True,
        show_pellet: bool = False,
        arrow_spacing: int = 5,
        offset_scale: float = 0.5,
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None
    ) -> None:
        """
        Create a line plot with smart trajectory offsetting based on movement direction.
        """
        if game_id is not None:
            trajectory = self.datareader.get_trajectory(game_id=game_id)
            if title_id is None:
                title_id = f"game {game_id}"
        elif trajectory is not None:
            trajectory = self._format_trajectory_data(trajectory=trajectory)
        else:
            raise ValueError("Either game_id or trajectory must be provided")

        # Create figure and axis
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self.figsize
            )  # Create figure and axis explicitly
            show_plot = True
        else:
            show_plot = False

        x, y = trajectory[:, 0], trajectory[:, 1]
        # Calculate instantaneous velocities
        dx, dy = utils.calculate_velocities(trajectory=trajectory)

        # Create vector grid
        _, idx_grid = self.analyzer.calculate_recurrence_grid(
            trajectory=trajectory, calculate_velocities=False
        )

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
        segments = np.array(
            [
                points[i : i + time_step_delta + 1]
                for i in range(0, len(points) - time_step_delta, time_step_delta)
            ]
        )

        # Create a colormap based on time
        norm = plt.Normalize(0, len(segments))
        colormap_str = "Spectral"
        colors = plt.colormaps[colormap_str](np.linspace(0, 1, len(segments)))

        # Plot line segments with arrows
        for i, segment in enumerate(segments):
            ax.plot(segment[:, 0], segment[:, 1], color=colors[i], linewidth=2)
            if i % arrow_spacing == 0 and len(segment) > 1:
                mid = min(len(segment) // 2, len(segment) - 2)
                dx = segment[mid + 1, 0] - segment[mid, 0]
                dy = segment[mid + 1, 1] - segment[mid, 1]
                ax.arrow(
                    segment[mid, 0],
                    segment[mid, 1],
                    dx / 2,
                    dy / 2,
                    head_width=0.3,
                    head_length=0.5,
                    fc=colors[i],
                    ec=colors[i],
                )

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=colormap_str, norm=norm)
        sm.set_array([])  # Set array for the scalar mappable
        plt.colorbar(sm, ax=ax, label="Timestep")  # Specify the axis for the colorbar

        # Plot maze layout
        self._plot_walls_and_pellets(walls=show_maze, pellets=show_pellet, ax=ax)

        ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = [metadata_label] if isinstance(metadata_label, str) else metadata_label
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(f"Pacman Trajectory - {title_id}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_velocity_grid(
        self,
        game_id: int | None = None,
        trajectory: torch.Tensor | np.ndarray | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = False,
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None
    ) -> None:
        """
        Plot the velocity grid for a game trajectory.
        """
        if game_id is not None:
            trajectory_array = self.datareader.get_trajectory(game_id=game_id)
            title_id = f"game {game_id}"
        elif trajectory is not None:
            trajectory_array = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either game_id or trajectory must be provided")

        self.analyzer.calculate_recurrence_grid(
            trajectory=trajectory_array,
            calculate_velocities=True,
            aggregate=False,
            normalize=normalize,
        )



        """Plot the vector grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        ax.set_xlim(self.MAZE_X_MIN - 0.5, self.MAZE_X_MAX + 0.5)
        ax.set_ylim(self.MAZE_Y_MIN - 0.5, self.MAZE_Y_MAX + 0.5)

        walls_positions, _ = self._plot_walls_and_pellets(
            show_maze, show_pellet, ax=ax, return_transformed_positions=True
        )

        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                if (self.x_grid[j], self.y_grid[i]) not in walls_positions:
                    ax.arrow(
                        self.x_grid[j],
                        self.y_grid[i],
                        self.analyzer.velocity_grid[i, j, 0],
                        self.analyzer.velocity_grid[i, j, 1],
                        head_width=0.2,
                        head_length=0.2,
                        fc="red",
                        ec="red",
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = [metadata_label] if isinstance(metadata_label, str) else metadata_label
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(f"Velocity Grid - {title_id if title_id is not None else ' '}")

        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_count_grid(
        self,
        game_id: int | None = None,
        trajectory: torch.Tensor | np.ndarray | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = False,
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None
    ) -> None:
        """
        Plot the count grid for a game trajectory.
        """
        if game_id is not None:
            trajectory_array = self.datareader.get_trajectory(game_id=game_id)
            title_id = f"game {game_id}"
        elif trajectory is not None:
            trajectory_array = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either game_id or trajectory must be provided")

        self.analyzer.calculate_recurrence_grid(
            trajectory=trajectory_array,
            calculate_velocities=False,
            aggregate=False,
            normalize=normalize,
        )

        """Plot the count grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        ax.set_xlim(self.MAZE_X_MIN - 0.5, self.MAZE_X_MAX + 0.5)
        ax.set_ylim(self.MAZE_Y_MIN - 0.5, self.MAZE_Y_MAX + 0.5)

        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                if self.analyzer.recurrence_count_grid[i, j] != 0:
                    ax.text(
                        self.x_grid[j],
                        self.y_grid[i],
                        int(self.analyzer.recurrence_count_grid[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        bbox=dict(
                            facecolor="white",
                            edgecolor="black",
                            boxstyle="round,pad=0.3",
                        ),
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = [metadata_label] if isinstance(metadata_label, str) else metadata_label
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(
            f"Trajectory Count Grid - {title_id if title_id is not None else ' '}"
        )

        self._plot_walls_and_pellets(show_maze, show_pellet, ax=ax)
        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_trajectory_scatter(
        self,
        game_id: int | None = None,
        trajectory: torch.Tensor | np.ndarray | None = None,
        show_maze: bool = False,
        show_pellet: bool = False,
        ax: plt.Axes | None = None,
        title_id: int = None,
    ) -> None:
        """
        Plot the trajectory points as a scatter plot, without the use of grids.
        """
        if game_id is not None:
            trajectory = self.datareader.get_trajectory(game_id=game_id)
            title_id = f"game {game_id}"
        elif trajectory is not None:
            trajectory = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either game_id or trajectory must be provided")

        # Create figure and axis
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self.figsize
            )  # Create figure and axis explicitly
            show_plot = True
        else:
            show_plot = False

        ax.scatter(trajectory[:, 0], trajectory[:, 1], color="blue", alpha=0.5)

        # Plot maze layout
        self._plot_walls_and_pellets(walls=show_maze, pellets=show_pellet, ax=ax)

        ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)

        ax.set_title(f"Pacman Trajectory Scatter Plot - {title_id}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_multiple_trajectories(
        self,
        game_ids: List[int] | None = None,
        trajectories: List[torch.Tensor | np.ndarray | Trajectory] | None = None,
        plot_type: str = "line",
        n_cols: int = 4,
        show_maze: bool = True,
        show_pellet: bool = False,
        axs: list[plt.Axes] | None = None,
        metadata_label: str | list[str] | None = "game_id",
        **kwargs,
    ) -> None:

        """
        Create subplots of multiple trajectories.

        Args:
            game_ids: List of game IDs to visualize
            trajectories: List of trajectories to visualize
            plot_type: Type of plot ('line', 'heatmap', 'scatter', 'velocity', 'count')
            n_cols: Number of columns in the subplot grid
            show_maze: Whether to show maze walls
            show_pellet: Whether to show pellets
            axs: Passed axes, usually 4
            **kwargs: Additional arguments to pass to the individual plotting functions
        """
        if game_ids is not None:
            n_plots = len(game_ids)
            data_source = game_ids
        elif trajectories is not None:
            n_plots = len(trajectories)
            data_source = trajectories
        else:
            raise ValueError("Either game_ids or trajectories must be provided")

        if axs is None:
            n_cols = min(n_plots, n_cols)  # Adjust columns if fewer plots than columns
            n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
            fig = plt.figure(
                figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows)
            )
            axs = [plt.subplot(n_rows, n_cols, i + 1) for i in range(n_plots)]

        # Add show_maze and show_pellet to kwargs if not already present
        if "show_maze" not in kwargs:
            kwargs["show_maze"] = show_maze
        if "show_pellet" not in kwargs:
            kwargs["show_pellet"] = show_pellet

        for i, data in enumerate(data_source):
            ax = axs[i]
            game_id = game_ids[i] if game_ids is not None else None
            trajectory = trajectories[i] if trajectories is not None else None
            title_id = f"Game {game_id}" if game_id is not None else f"Trajectory {i}"

            # Select the appropriate plotting function
            if plot_type == "line":
                self.plot_trajectory_line(
                    ax=ax,
                    game_id=game_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    metadata_label=metadata_label,
                    **kwargs,
                )
            elif plot_type == "heatmap":
                self.plot_heatmap(
                    ax=ax,
                    game_id=game_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "scatter":
                self.plot_trajectory_scatter(
                    ax=ax,
                    game_id=game_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "velocity":
                self.plot_velocity_grid(
                    ax=ax,
                    game_id=game_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "count":
                self.plot_count_grid(
                    ax=ax,
                    game_id=game_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

        plt.tight_layout()
        plt.show()

    def _format_trajectory_data(
        self, trajectory: Trajectory | torch.Tensor | np.ndarray
    ) -> np.ndarray | Trajectory:
        """
        Format trajectory data from a trajectory tensor or numpy array of shape (N, 2) where N is the number of timesteps
        If there are multiple feeded, it ill be reshaped to (N, 2). If there is a Trajectory object, it will be returned as is.
        Don't feed lists of `Trajectory` dataclass, as they wont be reshaped.
        """

        
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        elif isinstance(trajectory, np.ndarray):
            pass
        elif isinstance(trajectory, Trajectory):
            return trajectory
        else:
            raise ValueError(
                f"trajectory must be either a Trajectory, torch.Tensor or np.ndarry. Type:{type(trajectory)}")

        if trajectory.ndim == 2:
            return trajectory
        elif trajectory.ndim == 3:  # Multiple trajectories
            return trajectory.reshape(-1, trajectory.shape[-1])
        else:
            raise ValueError("Trajectory must have 2 or 3 dimensions")
