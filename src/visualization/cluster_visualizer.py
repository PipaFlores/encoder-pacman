import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import setup_logger
from src.visualization.base_visualizer import BaseVisualizer
from bokeh.plotting import figure
from bokeh.models import (
    ColorBar,
    LinearColorMapper,
    CategoricalColorMapper,
    ColumnDataSource,
)

from bokeh.palettes import Viridis256


logger = setup_logger(__name__)


class ClusterVisualizer(BaseVisualizer):
    def __init__(
        self,
        affinity_matrix: np.ndarray,
        labels: np.ndarray,
        trajectories: np.ndarray,
        measure_type: str,
    ):
        super().__init__()
        self.affinity_matrix = affinity_matrix
        self.labels = labels
        self.trajectories = trajectories
        self.measure_type = measure_type

    def plot_affinity_matrix(self, ax: plt.Axes | None = None):
        if self.affinity_matrix.size == 0:
            logger.error("Affinity matrix is not calculated")
            raise ValueError(
                "Affinity matrix is not calculated. Please call calculate_affinity_matrix first."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        logger.debug("Plotting Affinity Matrix")
        ax.imshow(self.affinity_matrix, cmap="viridis")
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array(self.affinity_matrix)  # Set array for the scalar mappable
        plt.colorbar(sm, ax=ax, label=f"{self.measure_type.capitalize()} Distance")
        ax.set_title("Affinity Matrix")
        ax.set_xlabel("Trajectory Index")
        ax.set_ylabel("Trajectory Index")
        if show_plot:
            plt.show()

    def plot_affinity_matrix_bokeh(self):
        shape = self.affinity_matrix.shape
        row_indices, col_indices = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        # TODO include trajectory metadata in tooltips

        source = ColumnDataSource(
            data=dict(
                image=[self.affinity_matrix],
                x_index=[col_indices],
                y_index=[row_indices],
                x=[0],
                y=[0],
                dw=[shape[0]],
                dh=[shape[1]],
            )
        )

        p = figure(
            title="Affinity Matrix",
            x_range=(0, shape[0]),
            y_range=(shape[1], 0),
            tooltips=[
                ("x_index", "@x_index"),
                ("y_index", "@y_index"),
                ("value", "@image"),
            ],
            x_axis_label="Trajectory Index",
            y_axis_label="Trajectory Index",
        )

        p.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            palette="Viridis256",
            level="image",
            source=source,
        )

        color_mapper = LinearColorMapper(
            palette="Viridis256", low=0, high=np.max(self.affinity_matrix)
        )
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            title=f"{self.measure_type.capitalize()} Distance",
        )
        p.add_layout(color_bar, "right")

        p.title.text = "Affinity Matrix"
        return p

    def plot_distance_matrix_histogram(self, ax: plt.Axes | None = None, **kwargs):
        logger.debug("Plotting affinity matrix histogram")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances = self.affinity_matrix[
            np.triu_indices_from(self.affinity_matrix, k=1)
        ]
        ax.hist(distances, bins=200, **kwargs)
        ax.set_xlabel("Distance Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Affinity Matrix Histogram")
        if show_plot:
            plt.show()

    def plot_non_repetitive_distances_values_barchart(
        self, ax: plt.Axes | None = None, **kwargs
    ):
        logger.debug("Plotting non repetitive distances values barchart")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances_values = np.sort(
            self.affinity_matrix[np.triu_indices_from(self.affinity_matrix, k=1)]
        )[::-1]
        ax.plot(distances_values, **kwargs)
        ax.set_xlabel("Distance Index (sorted)")
        ax.set_ylabel("Distance Value")
        ax.set_title("Non Repetitive Distances Values")
        if show_plot:
            plt.show()

    def plot_average_column_value(self, ax: plt.Axes | None = None):
        logger.debug("Plotting average column value")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        average_column_value = np.sort(np.mean(self.affinity_matrix, axis=0))[::-1]
        ax.plot(average_column_value)
        ax.set_xlabel("Trajectory Index (sorted)")
        ax.set_ylabel("Average Column Value")
        ax.set_title("Average Column Value")
        if show_plot:
            plt.show()


    ### Cluster Visualization

    def plot_trajectories(
        self,
        traj_centroids: np.ndarray,
        ax: plt.Axes | None = None,
        frame_to_maze: bool = True,
    ):
        logger.debug("Plotting trajectories")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        if frame_to_maze:
            ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)
            ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)
        # Create a custom colormap that maps -1 to gray
        cmap = plt.cm.viridis
        cmap.set_under("gray")
        scatter = ax.scatter(
            traj_centroids[:, 0], traj_centroids[:, 1], c=self.labels, cmap=cmap, vmin=0
        )
        ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.set_title("Trajectory Clusters")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if show_plot:
            plt.show()

    def plot_trajectories_bokeh(self, traj_centroids: np.ndarray):
        # TODO include trajectory metadata in tooltips
        source = ColumnDataSource(
            data=dict(
                x=traj_centroids[:, 0],
                y=traj_centroids[:, 1],
                cluster=self.labels.astype(str),
                traj_idx=np.arange(len(traj_centroids[:, 0])),
            )
        )

        logger.debug("Plotting trajectories")
        p = figure(
            title="Trajectories",
            x_range=(self.MAZE_X_MIN, self.MAZE_X_MAX),
            y_range=(self.MAZE_Y_MIN, self.MAZE_Y_MAX),
            tooltips=[("traj_idx", "@traj_idx"), ("cluster", "@cluster")],
            x_axis_label="X Coordinate",
            y_axis_label="Y Coordinate",
        )

        # Create color mapper for clusters
        unique_labels = np.sort(np.unique(self.labels))
        unique_labels_str = [str(label) for label in unique_labels]
        n_clusters = len(unique_labels)

        # If we have noise points (-1), we need a special color for them
        if -1 in unique_labels:
            step = max(1, 256 // (n_clusters - 1))
            colors = ["gray"] + list(Viridis256[::step][: n_clusters - 1])
        else:
            step = max(1, 256 // n_clusters)
            colors = list(Viridis256[::step][:n_clusters])

        color_mapper = CategoricalColorMapper(factors=unique_labels_str, palette=colors)

        # Add scatter plot with color mapping
        scatter = p.scatter(
            x="x",
            y="y",
            color={"field": "cluster", "transform": color_mapper},
            legend_group="cluster",
            source=source,
        )

        # Add legend
        sorted_legend_items = sorted(p.legend.items, key=lambda x: int(x.label.value))
        p.legend.items = sorted_legend_items

        p.legend.title = "Clusters"
        p.legend.location = "top_right"
        return p

    def plot_clustering_centroids(
        self,
        cluster_centroids: np.ndarray,
        cluster_sizes: np.ndarray,
        frame_to_maze: bool = True,
        ax: plt.Axes | None = None,
    ):
        logger.debug("Plotting cluster centroids")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        if frame_to_maze:
            ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)
            ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)

        # Create a custom colormap that maps -1 to gray
        unique_labels = np.unique(self.labels)
        scatter = ax.scatter(
            cluster_centroids[:, 0],
            cluster_centroids[:, 1],
            c=unique_labels[1:],
            s=cluster_sizes,
        )
        ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        handle, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
        ax.legend(handle, labels, title="Cluster Sizes", loc="lower left")
        ax.set_title("Cluster Centroids")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if show_plot:
            plt.show()
