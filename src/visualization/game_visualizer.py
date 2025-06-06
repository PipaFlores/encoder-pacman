import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from src.utils.grid_analyzer import GridAnalyzer
import src.utils as utils
import logging
from src.visualization.base_visualizer import BaseVisualizer
from src.datahandlers import PacmanDataReader, Trajectory
from src.config.defaults import config


class GameVisualizer(BaseVisualizer):
    """
    Visualizes trajectories and heatmaps of Pacman levels, either from a single game or a list of levels.
    """

    def __init__(
        self,
        data_folder: str = None,
        verbose: bool = False,
        figsize: Tuple[int, int] = config.figsize,
        darkmode: bool = True,
    ):
        """
        Initialize the trajectory visualizer.

        Args:
            data_folder: Path to the data folder containing game data, if visualizing by game or user id.
            verbose: Whether to print verbose output
            figsize: Size of the figure (width, height). Default is (6, 6).
            if multiplot is used, the figsize will be multiplied by the number of plots.
            darkmode: Wether to add black background and light contrast to visualizations (easier for spotting non-recurrent deviations from patterns in aggregated plots).
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
        self.darkmode = darkmode

        if data_folder is None:
            self.datareader = None
        else:
            self.datareader = PacmanDataReader(
                data_folder=data_folder, read_games_only=True, verbose=verbose
            )

    def plot_heatmap(
        self,
        level_id: int | list[int] | None = None,
        trajectory: Trajectory
        | list[Trajectory]
        | torch.Tensor
        | np.ndarray
        | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = True,
        ax: plt.Axes | None = None,
        title_id: int | None = None,
        metadata_label: str | list[str] | None = None,
    ) -> None:
        """
        Create a heatmap visualization of a game trajectory.
        If multiple levels are provided, the heatmap will be a combination of all levels.

        Args:
            level_id (int | list[int] | None): The ID of the game to visualize. If None, trajectory must be provided.
            trajectory (Trajectory | list[Trajectory] | torch.Tensor | np.ndarray | None): The trajectory data to visualize.
                Can be a single trajectory or a list of trajectories. For single trajectories, should be a tensor/array of shape (N, 2)
                where N is the number of steps.
            show_maze (bool): Whether to show maze walls. Defaults to True.
            show_pellet (bool): Whether to show pellet positions. Defaults to False.
            normalize (bool): Whether to normalize the heatmap values. If True, values are scaled to [0,1]. Defaults to True.
            ax (plt.Axes | None): Matplotlib axes to plot on. If None, a new figure is created.
            title_id (int | None): ID to display in the plot title.
            metadata_label (str | list[str] | None): Metadata labels to display in the title.
        """
        if level_id is not None:
            trajectory = self.datareader.get_trajectory(level_id=level_id)
            title_id = f"game {level_id}"
        elif trajectory is not None and not isinstance(trajectory, list):
            trajectory = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        elif isinstance(trajectory, list):
            title_id = "aggregated" if not title_id else title_id
            if not isinstance(trajectory[0], Trajectory):
                raise ValueError(
                    "When a list of trajectories is provided, all elements need to be of the Trajectory dataclass"
                )
        else:
            raise ValueError("Either level_id or trajectory must be provided")

        if isinstance(trajectory, list):
            for traj in trajectory:
                self.analyzer._initialize_idx_grid()
                self.analyzer.calculate_recurrence_grid(
                    trajectory=traj,
                    calculate_velocities=False,
                    aggregate=True,
                    normalize=False,
                )

            if normalize:
                max_recurrence = self.analyzer.recurrence_count_grid.max()
                if max_recurrence > 0:
                    self.analyzer.recurrence_count_grid = (
                        self.analyzer.recurrence_count_grid / max_recurrence
                    )
        else:
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

        if self.darkmode:
            # Create a copy of the YlOrRd colormap and set the color for 0 to black
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_under("black")

        # Find the minimum nonzero value in the grid (if any)
        grid = self.analyzer.recurrence_count_grid
        nonzero = grid[grid > 0]
        vmin = nonzero.min() if nonzero.size > 0 else 1  # avoid vmin=0

        # Create an alpha (transparency) mask that fades with value
        # Normalize grid to [0, 1] for alpha, with 0 mapped to 0 (fully transparent)
        norm_grid = (
            (grid - vmin) / (grid.max() - vmin)
            if grid.max() > vmin
            else np.zeros_like(grid)
        )
        alpha_min, alpha_max = 0.2, 1.0  # Minimum and maximum alpha values
        alpha = np.where(grid > 0, alpha_min + (alpha_max - alpha_min) * norm_grid, 0.0)

        # Plot the heatmap with fading (alpha)
        im = ax.imshow(
            grid,
            extent=[
                self.MAZE_X_MIN - 0.5,
                self.MAZE_X_MAX + 0.5,
                self.MAZE_Y_MIN - 0.5,
                self.MAZE_Y_MAX + 0.5,
            ],
            origin="lower",
            cmap=cmap if self.darkmode else "YlOrRd",
            aspect="equal",
            vmin=vmin,
            alpha=alpha if alpha.shape == grid.shape else 1.0,
        )

        if self.darkmode:
            # Set background style to black
            ax.set_facecolor("black")
            if ax.figure is not None:
                ax.figure.set_facecolor("black")

            # Set text color to white for all relevant elements
            text_color = "white"
            ax.tick_params(colors=text_color, which="both")  # Tick labels
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            # Also set spine colors to white for better contrast
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

        self._plot_walls_and_pellets(show_maze, show_pellet, ax=ax)
        ax.grid(True, alpha=0.3)

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = (
                [metadata_label] if isinstance(metadata_label, str) else metadata_label
            )
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(
            f"Trajectory Heatmap - {title_id if title_id is not None else ' '}",
            color=text_color if self.darkmode else "black",
        )
        if show_plot:
            plt.show()

    def plot_trajectory_line(
        self,
        level_id: int | None = None,
        trajectory: Trajectory | torch.Tensor | np.ndarray | None = None,
        time_step_delta: int = 1,
        show_maze: bool = True,
        show_pellet: bool = False,
        arrow_spacing: int = 5,
        offset_scale: float = 0.5,
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None,
    ) -> None:
        """
        Create a line plot with smart trajectory offsetting based on movement direction.
        """
        if level_id is not None:
            trajectory = self.datareader.get_trajectory(level_id=level_id)
            if title_id is None:
                title_id = f"game {level_id}"
        elif trajectory is not None:
            trajectory = self._format_trajectory_data(trajectory=trajectory)
        else:
            raise ValueError("Either level_id or trajectory must be provided")

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

        if self.darkmode:
            cbar = plt.colorbar(
                sm, ax=ax, label="Timestep"
            )  # Specify the axis for the colorbar
            cbar.set_label("Timestep", color="white")  # Set label color to white
            cbar.ax.yaxis.set_tick_params(color="white")  # Set tick color to white
            plt.setp(
                plt.getp(cbar.ax.axes, "yticklabels"), color="white"
            )  # Set tick label color to white
            # Set background style to black
            ax.set_facecolor("black")
            if ax.figure is not None:
                ax.figure.set_facecolor("black")

            # Set text color to white for all relevant elements
            text_color = "white"
            ax.tick_params(colors=text_color, which="both")  # Tick labels
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            # Also set spine colors to white for better contrast
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)
        else:
            text_color = "black"
            plt.colorbar(
                sm, ax=ax, label="Timestep"
            )  # Specify the axis for the colorbar

        # Plot maze layout
        self._plot_walls_and_pellets(walls=show_maze, pellets=show_pellet, ax=ax)

        ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = (
                [metadata_label] if isinstance(metadata_label, str) else metadata_label
            )
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(f"Pacman Trajectory - {title_id}", color=text_color)
        ax.set_xlabel("X Position", color=text_color)
        ax.set_ylabel("Y Position", color=text_color)
        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_velocity_grid(
        self,
        level_id: int | None = None,
        trajectory: torch.Tensor
        | np.ndarray
        | Trajectory
        | list[Trajectory]
        | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = False,  #
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None,
        min_alpha: float = 0.2,  # Minimum alpha value for the least frequent movements
        max_alpha: float = 1.0,  # Maximum alpha value for the most frequent movements
    ) -> None:
        """
        Plot the velocity grid for a game trajectory. The vectors are normalized to unit length
        and colored according to the recurrence count at each position. The transparency (alpha)
        of each vector is also scaled based on its recurrence count.

        Args:
            level_id (int | None): ID of the game to plot. If None, trajectory must be provided.
            trajectory (torch.Tensor | np.ndarray | Trajectory | list[Trajectory] | None): 
                Trajectory data to plot. Can be a single trajectory or a list of trajectories.
            show_maze (bool): Whether to display the maze walls. Defaults to True.
            show_pellet (bool): Whether to display the pellet positions. Defaults to False.
            normalize (bool): Whether to normalize the velocity vectors and recurrence counts. Defaults to False. \
            If true it returns the unitary vector. For list of levels (aggregated plot) this is forced to True
            ax (plt.Axes | None): Matplotlib axes to plot on. If None, a new figure is created.
            title_id (int): ID to display in the plot title.
            metadata_label (str | list[str] | None): Metadata labels to display in the title.
            min_alpha (float): Minimum transparency value for vectors. Defaults to 0.2.
            max_alpha (float): Maximum transparency value for vectors. Defaults to 1.0.
        """

        aggregate_plot = (
            False  # Flag for aggregate visualization (encode recurrence in color)
        )
        """Input Check"""
        if level_id is not None:
            trajectory_array = self.datareader.get_trajectory(level_id=level_id)
            title_id = f"game {level_id}"
        elif trajectory is not None and not isinstance(trajectory, list):
            trajectory_array = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        elif isinstance(trajectory, list):
            aggregate_plot = True
            normalize = True
            title_id = "aggregated" if not title_id else title_id
            if not isinstance(trajectory[0], Trajectory):
                raise ValueError(
                    "When a list of trajectories is provided, all elements need to be of the Trajectory dataclass"
                )
        else:
            raise ValueError("Either level_id or trajectory must be provided")

        """Calculate the recurrence and velocity grid"""
        if isinstance(trajectory, list):
            self.analyzer._reset_grids()  # In case previous aggregate calculations were performed
            # First aggregate all trajectories without normalization
            for traj in trajectory:
                self.analyzer._initialize_idx_grid()  # reset idx grid (otherwise overlaps with close timing wont be accounted for due to recurrence algorithm)
                self.analyzer.calculate_recurrence_grid(
                    trajectory=traj,
                    calculate_velocities=True,
                    aggregate=True,
                    normalize=False,  # Don't normalize individual trajectories during aggregation
                )
            # Normalize the final aggregated grid
            if normalize and self.analyzer.recurrence_count_grid is not None:
                # Only normalize if there are non-zero values
                max_recurrence = self.analyzer.recurrence_count_grid.max()
                if max_recurrence > 0:
                    self.analyzer.recurrence_count_grid = (
                        self.analyzer.recurrence_count_grid / max_recurrence
                    )

                # For velocity grid, we want to maintain relative magnitudes
                # So we normalize each vector individually to unit length
                if self.analyzer.velocity_grid is not None:
                    # Calculate magnitude for each vector
                    magnitudes = np.sqrt(np.sum(self.analyzer.velocity_grid**2, axis=2))
                    # Avoid division by zero
                    non_zero_mask = magnitudes > 0
                    # Normalize only non-zero vectors
                    self.analyzer.velocity_grid[non_zero_mask] = (
                        self.analyzer.velocity_grid[non_zero_mask]
                        / magnitudes[non_zero_mask, np.newaxis]
                    )
        else:
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

        # Create a colormap for the vectors based on recurrence counts
        if aggregate_plot:
            cmap = plt.cm.YlOrRd
            norm = plt.Normalize(vmin=0, vmax=self.analyzer.recurrence_count_grid.max())

        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                if (self.x_grid[j], self.y_grid[i]) not in walls_positions:
                    # Get the recurrence count for this position
                    recurrence = self.analyzer.recurrence_count_grid[i, j]
                    if aggregate_plot:
                        # Get the color based on recurrence
                        color = cmap(norm(recurrence))
                        # Calculate alpha based on recurrence
                        alpha = min_alpha + (max_alpha - min_alpha) * norm(recurrence)
                    else:
                        color = "red"
                        alpha = 1

                    # Only plot vectors with non-zero velocity
                    if np.any(self.analyzer.velocity_grid[i, j] != 0):
                        ax.arrow(
                            self.x_grid[j],
                            self.y_grid[i],
                            self.analyzer.velocity_grid[i, j, 0] * 0.5,
                            self.analyzer.velocity_grid[i, j, 1] * 0.5,
                            head_width=0.2,
                            head_length=0.2,
                            fc=color,
                            ec=color,
                            alpha=alpha,  # Add transparency
                        )

        if aggregate_plot:
            # Add colorbar to show recurrence scale, with white text for contrast
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            if self.darkmode:
                cbar = plt.colorbar(sm, ax=ax, label="Normalized Recurrence")
                cbar.set_label(
                    "Normalized Recurrence", color="white"
                )  # Set label color to white
                cbar.ax.yaxis.set_tick_params(color="white")  # Set tick color to white
                plt.setp(
                    plt.getp(cbar.ax.axes, "yticklabels"), color="white"
                )  # Set tick label color to white
            else:
                plt.colorbar(sm, ax=ax, label="Normalized Recurrence")

        if self.darkmode:
            # Set background style to black
            ax.set_facecolor("black")
            if ax.figure is not None:
                ax.figure.set_facecolor("black")

            # Set text color to white for all relevant elements
            text_color = "white"
            ax.tick_params(colors=text_color, which="both")  # Tick labels
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            # Also set spine colors to white for better contrast
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)
        else:
            text_color = "black"

        ax.set_xlabel("X", color=text_color)
        ax.set_ylabel("Y", color=text_color)

        # Set metadata on title
        if isinstance(trajectory, Trajectory) and metadata_label:
            title_id = ""
            metadata_label = (
                [metadata_label] if isinstance(metadata_label, str) else metadata_label
            )
            for column in metadata_label:
                title_id += f"{column} {trajectory.metadata[column]} "

        ax.set_title(
            f"Velocity Grid - {title_id if title_id is not None else ' '}",
            color=text_color,
        )

        ax.grid(True, alpha=0.3)
        if show_plot:
            plt.show()

    def plot_count_grid(
        self,
        level_id: int | None = None,
        trajectory: torch.Tensor | np.ndarray | None = None,
        show_maze: bool = True,
        show_pellet: bool = False,
        normalize: bool = False,
        ax: plt.Axes | None = None,
        title_id: int = None,
        metadata_label: str | list[str] | None = None,
    ) -> None:
        """
        Plot the count grid for a game trajectory.
        """
        if level_id is not None:
            trajectory_array = self.datareader.get_trajectory(level_id=level_id)
            title_id = f"game {level_id}"
        elif trajectory is not None:
            trajectory_array = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either level_id or trajectory must be provided")

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
            metadata_label = (
                [metadata_label] if isinstance(metadata_label, str) else metadata_label
            )
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
        level_id: int | None = None,
        trajectory: torch.Tensor | np.ndarray | None = None,
        show_maze: bool = False,
        show_pellet: bool = False,
        ax: plt.Axes | None = None,
        title_id: int = None,
    ) -> None:
        """
        Plot the trajectory points as a scatter plot, without the use of grids.
        """
        if level_id is not None:
            trajectory = self.datareader.get_trajectory(level_id=level_id)
            title_id = f"game {level_id}"
        elif trajectory is not None:
            trajectory = self._format_trajectory_data(trajectory=trajectory)
            title_id = f"trajectory {title_id}"
        else:
            raise ValueError("Either level_id or trajectory must be provided")

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
        level_ids: List[int] | None = None,
        trajectories: List[torch.Tensor | np.ndarray | Trajectory] | None = None,
        plot_type: str = "line",
        n_cols: int = 4,
        show_maze: bool = True,
        show_pellet: bool = False,
        axs: list[plt.Axes] | None = None,
        metadata_label: str | list[str] | None = "level_id",
        **kwargs,
    ) -> None:
        """
        Create subplots of multiple trajectories.

        Args:
            level_ids: List of game IDs to visualize
            trajectories: List of trajectories to visualize
            plot_type: Type of plot ('line', 'heatmap', 'scatter', 'velocity', 'count')
            n_cols: Number of columns in the subplot grid
            show_maze: Whether to show maze walls
            show_pellet: Whether to show pellets
            axs: Passed axes, usually 4
            **kwargs: Additional arguments to pass to the individual plotting functions
        """
        if level_ids is not None:
            n_plots = len(level_ids)
            data_source = level_ids
        elif trajectories is not None:
            n_plots = len(trajectories)
            data_source = trajectories
        else:
            raise ValueError("Either level_ids or trajectories must be provided")

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
            level_id = level_ids[i] if level_ids is not None else None
            trajectory = trajectories[i] if trajectories is not None else None
            title_id = f"Game {level_id}" if level_id is not None else f"Trajectory {i}"

            # Select the appropriate plotting function
            if plot_type == "line":
                self.plot_trajectory_line(
                    ax=ax,
                    level_id=level_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    metadata_label=metadata_label,
                    **kwargs,
                )
            elif plot_type == "heatmap":
                self.plot_heatmap(
                    ax=ax,
                    level_id=level_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "scatter":
                self.plot_trajectory_scatter(
                    ax=ax,
                    level_id=level_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "velocity":
                self.plot_velocity_grid(
                    ax=ax,
                    level_id=level_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            elif plot_type == "count":
                self.plot_count_grid(
                    ax=ax,
                    level_id=level_id,
                    trajectory=trajectory,
                    title_id=title_id,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

        plt.tight_layout()
        # plt.show()

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
                f"trajectory must be either a Trajectory, torch.Tensor or np.ndarry. Type:{type(trajectory)}"
            )

        if trajectory.ndim == 2:
            return trajectory
        elif trajectory.ndim == 3:  # Multiple trajectories
            return trajectory.reshape(-1, trajectory.shape[-1])
        else:
            raise ValueError("Trajectory must have 2 or 3 dimensions")
