import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import setup_logger
from src.visualization.base_visualizer import BaseVisualizer
from src.datahandlers import Trajectory
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
    """
    A class for visualizing clustering results and related metrics.

    This class provides methods to visualize various aspects of trajectory clustering:
    - Affinity matrix showing pairwise distances between trajectories
    - Distance matrix histograms and barcharts
    - Clustering results including trajectories and centroids
    - Interactive visualizations using Bokeh

    The visualizer is initialized with:
    - An affinity matrix containing pairwise distances between trajectories
    - Cluster labels for each trajectory
    - The original trajectories
    - The type of similarity measure used

    Methods are provided for both static matplotlib plots and interactive Bokeh visualizations.
    """

    def __init__(
        self,
        cmap = plt.cm.tab20,
        ):
        """
        Initialize the ClusterVisualizer with clustering data.

        Args:
            affinity_matrix (np.ndarray): A square matrix containing pairwise distances between trajectories
            labels (np.ndarray): Array of cluster labels for each trajectory
            trajectories (np.ndarray | list[Trajectory]): The original trajectory data
            measure_type (str): Type of similarity measure used (e.g., 'euclidean', 'dtw')
        """
        super().__init__()
        self.cmap = cmap
    def plot_affinity_matrix(self, 
                             affinity_matrix: np.ndarray,
                             measure_type:str = "",
                             ax: plt.Axes | None = None):
        """
        Plot the affinity matrix using matplotlib.

        This method creates a heatmap visualization of the pairwise distances between trajectories.
        The visualization includes a colorbar indicating the distance values.

        Args:
            affinity_matrix (np.ndarray): The affinity matrix to plot.
            measure_type (str): The measure type used to calculate the distances between elements
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.

        Raises:
            ValueError: If affinity matrix is not calculated
        """
        if affinity_matrix.size == 0:
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
        ax.imshow(affinity_matrix, cmap="viridis")
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array(affinity_matrix)  # Set array for the scalar mappable
        plt.colorbar(sm, ax=ax, label=f"{measure_type.capitalize()} Distance")
        ax.set_title("Affinity Matrix")
        ax.set_xlabel("Trajectory Index")
        ax.set_ylabel("Trajectory Index")
        if show_plot:
            plt.show()

    def plot_affinity_matrix_bokeh(self,
                                   affinity_matrix: np.ndarray,
                                   measure_type: str = ""):
        """
        Create an interactive Bokeh plot of the affinity matrix.

        Args:
            affinity_matrix (np.ndarray): The affinity matrix to plot
            measure_type (str): The measure type used to calculate the distance between elements

        Returns:
            bokeh.plotting.figure: A Bokeh figure containing the interactive affinity matrix visualization
        """
        shape = affinity_matrix.shape
        row_indices, col_indices = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        # TODO include trajectory metadata in tooltips

        source = ColumnDataSource(
            data=dict(
                image=[affinity_matrix],
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
            palette="Viridis256", low=0, high=np.max(affinity_matrix)
        )
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            title=f"{measure_type.capitalize()} Distance",
        )
        p.add_layout(color_bar, "right")

        p.title.text = "Affinity Matrix"
        return p

    def plot_distance_matrix_histogram(self, 
                                       affinity_matrix: np.ndarray,
                                       ax: plt.Axes | None = None, 
                                       **kwargs):
        """
        Plot a histogram of the distance values from the affinity matrix.

        This method visualizes the distribution of pairwise distances between trajectories,
        excluding self-distances (diagonal elements).

        Args:
            affinity_matrix (np.ndarray): The affinity matrix to plot
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.
            **kwargs: Additional arguments to pass to matplotlib's hist function
        """
        logger.debug("Plotting affinity matrix histogram")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances = affinity_matrix[
            np.triu_indices_from(affinity_matrix, k=1)
        ]
        ax.hist(distances, bins=200, **kwargs)
        ax.set_xlabel("Distance Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Affinity Matrix Histogram")
        if show_plot:
            plt.show()

    def plot_non_repetitive_distances_values_barchart(
        self, 
        affinity_matrix: np.ndarray,
        ax: plt.Axes | None = None, **kwargs
    ):
        """
        Plot a sorted bar chart of unique distance values from the affinity matrix.

        This method shows the distribution of unique distance values in descending order,
        providing insight into the range and distribution of trajectory similarities.

        Args:
            affinity_matrix (np.ndarray): Affinity matrix to plot
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.
            **kwargs: Additional arguments to pass to matplotlib's plot function
        """
        logger.debug("Plotting non repetitive distances values barchart")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        distances_values = np.sort(
            affinity_matrix[np.triu_indices_from(affinity_matrix, k=1)]
        )[::-1]
        ax.plot(distances_values, **kwargs)
        ax.set_xlabel("Distance Index (sorted)")
        ax.set_ylabel("Distance Value")
        ax.set_title("Non Repetitive Distances Values")
        if show_plot:
            plt.show()

    def plot_average_column_value(self, 
                                  affinity_matrix: np.ndarray,
                                  ax: plt.Axes | None = None):
        """
        Plot the average distance value for each trajectory.

        This method calculates and visualizes the mean distance of each trajectory to all others,
        sorted in descending order. This can help identify outliers or particularly distinct trajectories.

        Args:
            affinity_matrix (np.ndarray): The affinity matrix to plot
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.
        """
        logger.debug("Plotting average column value")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        average_column_value = np.sort(np.mean(affinity_matrix, axis=0))[::-1]
        ax.plot(average_column_value)
        ax.set_xlabel("Trajectory Index (sorted)")
        ax.set_ylabel("Average Column Value")
        ax.set_title("Average Column Value")
        if show_plot:
            plt.show()

    ### Cluster Visualization

    def plot_trajectories_embedding(
        self,
        traj_embeddings: np.ndarray,
        labels: np.ndarray | None = None,
        ax: plt.Axes | None = None,
        frame_to_maze: bool = True,
    ):
        """
        Plot the trajectory embeddings (or geometrical centroids) colored by their cluster assignments.

        This method visualizes the spatial distribution of trajectory centroids,
        with each point colored according to its cluster membership.

        Args:
            traj_embeddings (np.ndarray): Array of trajectory centroid coordinates or 2D embedding (n,2)
            labels (np.ndarray): Labels of the trajectories
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.
            show_maze (bool, optional): Whether to set axis limits to maze boundaries.
                Defaults to True.
        """
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
        cmap = self.cmap
        cmap.set_under("gray")

        scatter = ax.scatter(
            traj_embeddings[:, 0],
            traj_embeddings[:, 1],
            c=labels,
            cmap=cmap,
            vmin=0,
            s=3,
        )

        # Create custom legend with cluster sizes
        # Instead of using matplotlib's automatic size legend
        # Create a legend mapping cluster label to color (colored 'o' marker next to the cluster number)
        import matplotlib.lines as mlines

        # Get unique cluster labels (excluding noise if present)
        if labels is not None:
            unique_labels = np.unique(labels)[:8]
        else:
            unique_labels = np.array([0])

        legend_handles = []
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points, use gray
                color = "gray"
            else:
                color = cmap(label)
            handle = mlines.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor=color, 
                markeredgecolor=color, markersize=8, linestyle='None', label=str(label)
            )
            legend_handles.append(handle)

        ax.legend(handles=legend_handles, title="Cluster (first 8)", loc="upper left", frameon=True)



        ax.set_title("Reduced latent space, individual data points.")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if show_plot:
            plt.show()

    def plot_trajectories_embedding_bokeh(self, 
                                          traj_centroids: np.ndarray,
                                          labels: np.ndarray | None = None):
        """
        Create an interactive Bokeh plot of trajectory centroids.

        This method provides an interactive visualization of trajectory centroids,
        with tooltips showing trajectory index and cluster assignment.

        Args:
            traj_centroids (np.ndarray): Array of trajectory centroid coordinates
            labels (np.ndarray): Array of trajectory labels

        Returns:
            bokeh.plotting.figure: A Bokeh figure containing the interactive trajectory visualization
        """
        data=dict(
            x=traj_centroids[:, 0],
            y=traj_centroids[:, 1],
            traj_idx=np.arange(len(traj_centroids[:, 0])),
        )
        if labels is not None:
            data["cluster"] = labels.astype(str)
            

        source = ColumnDataSource(data)

        logger.debug("Plotting trajectories")
        p = figure(
            title="Trajectories",
            x_range=(self.MAZE_X_MIN, self.MAZE_X_MAX),
            y_range=(self.MAZE_Y_MIN, self.MAZE_Y_MAX),
            tooltips=[("traj_idx", "@traj_idx"), ("cluster", "@cluster")] if labels is not None else [("traj_idx", "@traj_idx")],
            x_axis_label="X Coordinate",
            y_axis_label="Y Coordinate",
        )

        # Create color mapper for clusters
        if labels is not None:
            unique_labels = np.sort(np.unique(labels))
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

            scatter = p.scatter(
                x="x",
                y="y",
                color={"field": "cluster", "transform": color_mapper},
                legend_group="cluster",
                source=source,
            )
            sorted_legend_items = sorted(p.legend.items, key=lambda x: int(x.label.value))
            p.legend.items = sorted_legend_items
            # Add legend
            p.legend.title = "Clusters"
            p.legend.location = "top_right"
        
        else:
            scatter = p.scatter(
                x="x",
                y="y",       
                source=source,
                )


        return p

    def plot_clusters_centroids(
        self,
        cluster_centroids: np.ndarray,
        cluster_sizes: np.ndarray,
        labels: np.ndarray | None = None,
        frame_to_maze: bool = True,
        ax: plt.Axes | None = None,
    ):
        """
        Plot the centroids of each cluster with size proportional to cluster size.

        This method visualizes the center points of each cluster, with the size of each
        point indicating the number of trajectories in that cluster.

        Args:
            cluster_centroids (np.ndarray): Array of cluster centroid coordinates (without the -1 labelled noise cluster)
            cluster_sizes (np.ndarray): Array of sizes for each cluster (without the -1 labelled noise cluster)
            labels (np.ndarray) : Array of datapoints labels
            show_maze (bool, optional): Whether to set axis limits to maze boundaries.
                Defaults to True.
            ax (plt.Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
                Defaults to None.
        """
        logger.debug("Plotting cluster centroids")
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            show_plot = True
        else:
            show_plot = False

        if frame_to_maze:
            ax.set_ylim(self.MAZE_Y_MIN, self.MAZE_Y_MAX)
            ax.set_xlim(self.MAZE_X_MIN, self.MAZE_X_MAX)

        unique_labels = np.unique(labels)[1:] if labels is not None else None

        cmap = self.cmap

        scatter = ax.scatter(
            cluster_centroids[:, 0],
            cluster_centroids[:, 1],
            cmap=cmap,
            c=unique_labels,
            s=cluster_sizes,
        )
        # Create custom legend with cluster sizes
        # Instead of using matplotlib's automatic size legend
        size_legend_handles = []
        size_legend_labels = []
        max_size = max(cluster_sizes)
        
        # Create legend entries for cluster sizes
        for i, size in enumerate(cluster_sizes[:8]):
            # Get the color for this cluster from the colormap
            if unique_labels is not None:
                cluster_color = cmap(unique_labels[i])
            else:
                cluster_color = cmap(i)

            # Create a scatter point with size
            handle = plt.Line2D([0], [0], marker='o', color=cluster_color, 
                              markerfacecolor=cluster_color, markersize= 10 * (size/max_size), 
                              alpha=0.7, linestyle='None')
            size_legend_handles.append(handle)
            size_legend_labels.append(f'{int(size)}')
        
        # Add the size legend
        size_legend = ax.legend(size_legend_handles, size_legend_labels, 
                               title="Cluster Sizes", loc="upper left", 
                               frameon=True, fancybox=True, shadow=True)
        
        # Add the size legend to the plot (matplotlib removes the first legend when adding a second)
        ax.add_artist(size_legend)
        
        ax.set_title("Cluster Centroids")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if show_plot:
            plt.show()
