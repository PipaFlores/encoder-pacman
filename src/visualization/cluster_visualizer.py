import matplotlib.pyplot as plt
import numpy as np
from src.config.defaults import config
from src.utils.logger import setup_logger
from src.visualization.base_visualizer import BaseVisualizer
from bokeh.plotting import figure, show
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource

logger = setup_logger(__name__)

class ClusterVisualizer(BaseVisualizer):
    def __init__(self,
                 affinity_matrix: np.ndarray,
                 labels: np.ndarray,
                 trajectories: np.ndarray,
                 measure_type: str):
        super().__init__()
        self.affinity_matrix = affinity_matrix
        self.labels = labels
        self.trajectories = trajectories
        self.measure_type = measure_type

    def plot_affinity_matrix(self,
                             ax: plt.Axes | None = None):
        if self.affinity_matrix.size == 0:
            logger.error("Affinity matrix is not calculated")
            raise ValueError("Affinity matrix is not calculated. Please call calculate_affinity_matrix first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        logger.debug("Plotting Affinity Matrix")
        ax.imshow(self.affinity_matrix, cmap = 'viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(self.affinity_matrix)  # Set array for the scalar mappable
        plt.colorbar(sm, ax=ax, label=f'{self.measure_type.capitalize()} Distance')
        ax.set_title('Affinity Matrix')
        ax.set_xlabel('Trajectory Index')
        ax.set_ylabel('Trajectory Index')
        if show_plot:
            plt.show()
        
    def plot_affinity_matrix_bokeh(self):

        p = figure(title="Affinity Matrix",
                   x_range= (0, self.affinity_matrix.shape[0]),
                   y_range= (self.affinity_matrix.shape[1], 0),
                   tooltips=[("x_index", "$x"), ("y_index", "$y"), ("value", "@image")])

        p.image(image=[self.affinity_matrix],
                x=0, y=0,
                dw=self.affinity_matrix.shape[0],
                dh=self.affinity_matrix.shape[1],
                palette="Viridis256",
                level="image")

        color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=np.max(self.affinity_matrix))
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, title=f'{self.measure_type.capitalize()} Distance')
        p.add_layout(color_bar, 'right')
        
        p.title.text = 'Affinity Matrix'
        show(p)

    def plot_distance_matrix_histogram(self, ax: plt.Axes | None = None, **kwargs):
        logger.debug("Plotting affinity matrix histogram")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances = self.affinity_matrix[np.triu_indices_from(self.affinity_matrix, k=1)]
        ax.hist(distances, bins=200, **kwargs)
        ax.set_xlabel('Distance Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Affinity Matrix Histogram')
        if show_plot:
            plt.show()

    def plot_non_repetitive_distances_values_barchart(self,
                                                      ax: plt.Axes | None = None,
                                                      **kwargs):
        logger.debug("Plotting non repetitive distances values barchart")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances_values = np.sort(self.affinity_matrix[np.triu_indices_from(self.affinity_matrix, k=1)])[::-1]
        ax.plot(distances_values, **kwargs)
        ax.set_xlabel('Distance Index (sorted)')
        ax.set_ylabel('Distance Value')
        ax.set_title('Non Repetitive Distances Values Barchart')
        if show_plot:
            plt.show()

    def plot_average_column_value(self, 
                                  ax: plt.Axes | None = None):
        logger.debug("Plotting average column value")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False
        
        average_column_value = np.sort(np.mean(self.affinity_matrix, axis=0))[::-1]
        ax.plot(average_column_value)
        ax.set_xlabel('Trajectory Index (sorted)')
        ax.set_ylabel('Average Column Value')
        ax.set_title('Average Column Value')
        if show_plot:
            plt.show()

    def plot_trajectories(self,
                          traj_centroids: np.ndarray,
                          ax: plt.Axes | None = None,
                          frame_to_maze: bool = True):
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
        cmap.set_under('gray')
        scatter = ax.scatter(traj_centroids[:, 0], traj_centroids[:, 1], c=self.labels, cmap=cmap, vmin=0)
        ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.set_title('Trajectory Clusters')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        if show_plot:
            plt.show()

    def plot_cluster_centroids(self, 
                               cluster_centroids: np.ndarray,
                               cluster_sizes: np.ndarray,
                               frame_to_maze: bool = True,
                               ax: plt.Axes | None = None):
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
        scatter = ax.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c=unique_labels[1:], s=cluster_sizes)
        ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        handle, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
        ax.legend(handle, labels, title="Cluster Sizes", loc="lower left")
        ax.set_title('Cluster Centroids')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        if show_plot:
            plt.show()