import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from typing import Literal
from sklearn.neighbors import KernelDensity


def calculate_latent_distances(latent_points, ax):

    diff_vectors = np.diff(latent_points, axis = 0)

    path_distances = np.linalg.norm(diff_vectors, axis=1, ord=2) # euclidean distance
    # np.sqrt(np.sum(np.square(diff_vectors), axis=1)) # euclidean distance
    
    return path_distances

## Entropy-based 
def overall_novelty(latent_points, 
                       bandwith: Literal["silverman", "scott"] | int = 1, 
                       return_density = False):
    """
    Calculate a per-step novelty of a latent point considering the whole latent points distribution
    
    The novelty is computed as the negative log-likelihood of each point under the
    kernel density estimate (KDE) of the full set of latent points. If $p(x)$ is the
    estimated density at point $x$, the per-point novelty is

        `text{novelty}(x) = -\log p(x)`

    The KDE provides $\hat p(x)$ and the function returns $-\log \hat p(x)$ for each
    latent point. If return_density is True, the function also returns $\hat p(x)$.
    """
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwith)
    kde.fit(latent_points)

    points_likelihood = kde.score_samples(latent_points)

    points_novelty = points_likelihood * -1

    if return_density:
        points_density = np.exp(points_likelihood)
        return points_novelty, points_density

    return points_novelty

def step_novelty(latent_points, 
                    min_history=2, 
                    rolling_window=None, 
                    bandwidth: Literal["scott", "silverman"] | int = 1,
                    return_density = False):
    """
    Calculate the per-step likelihood of a latent point based on the past history.

    For each step :math:`t \geq \text{min\_history}`, a KDE is fit on past latent
    points :math:`\{x_0, \dots, x_{t-1}\}` and evaluated at :math:`x_t`:

        :math:`\hat p_t(x_t) = \mathrm{KDE}(x_t \mid x_0, \dots, x_{t-1})`

    The per-step novelty is then defined as the negative log-likelihood:

        :math:`\mathrm{novelty}_t = -\log \hat p_t(x_t)`

    and, when ``return_density=True``, the function also returns
    :math:`\hat p_t(x_t) = \exp(\log \hat p_t(x_t))`.

    For :math:`t < \text{min\_history}`, novelty and density are ``NaN``.
    """
    
    steps = len(latent_points)
    novelty = np.full(steps, np.nan)
    densities = np.full(steps, np.nan)

    for step in range(steps):
        
        if step < min_history:
            continue

        past = latent_points[:step]
        kde = KernelDensity(kernel = "gaussian",
                            bandwidth=bandwidth)
        kde.fit(past)
        log_likelihood = kde.score_samples(latent_points[step].reshape(1,-1))
        
        novelty[step] = log_likelihood[0] * -1
        densities[step] = np.exp(log_likelihood)[0]

    if return_density:
        return novelty, densities
    return novelty


def plot_latent_navigation(latent_points, ax):
    # Draw directed lines between points following the order of their indexes, 
    # colored by temporal progression
    if len(latent_points) > 1:
        # Build segments connecting consecutive points
        points = latent_points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a color mapping from start (early) to end (late)
        norm = plt.Normalize(0, len(latent_points) - 1 if len(latent_points) > 1 else 1)
        color_indices = np.arange(len(latent_points))
        colors = cm.Spectral(norm(color_indices))

        # Color the connecting segments according to time
        seg_colors = cm.Spectral(norm(np.arange(len(segments))))
        lc = LineCollection(segments, colors=seg_colors, linewidths=1, alpha=0.8, zorder=1)
        ax.add_collection(lc)

        # Optionally add a colorbar to show the temporal progression
        sm = plt.cm.ScalarMappable(cmap=cm.Spectral, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Temporal progression (earlier → later)", pad=0.01)

        # Scatter points colored by the same progression (First and last point have a special icon)
        # Middle points
        if len(latent_points) > 2:
            ax.scatter(
                latent_points[1:-1, 0], latent_points[1:-1, 1],
                c=color_indices[1:-1], cmap=cm.Spectral, s=6, zorder=3, norm=norm, label="middle"
            )
        # Special icon for first point
        ax.scatter(
            latent_points[0, 0], latent_points[0, 1],
            c=[color_indices[0]], cmap=cm.Spectral, s=180, zorder=4, norm=norm,
            # edgecolor="white",
            marker="*",
            label="start"
        )
        # Special icon for last point
        ax.scatter(
            latent_points[-1, 0], latent_points[-1, 1],
            c=[color_indices[-1]], cmap=cm.Spectral, s=60, zorder=4, norm=norm,
            # edgecolor="white", 
            marker="X",
            label="end"
        )
    else:
        # Only one point, plot as red
        ax.scatter(x=latent_points[:, 0], y=latent_points[:, 1], s=6, color="red", zorder=3)


def plot_latent_navigation_animated(latent_points, ax=None, interval=80, repeat=True):
    """
    Animate the progression along latent_points: draw segments one by one,
    with a moving "current" marker, then show start (star) and end (X).
    """
    if ax is None:
        ax = plt.gca()
    if len(latent_points) < 2:
        ax.scatter(latent_points[:, 0], latent_points[:, 1], s=6, color="red", zorder=3)
        return None

    n = len(latent_points)
    norm = plt.Normalize(0, n - 1)
    cmap = cm.Spectral

    # Fixed elements: start marker (star), optional colorbar
    start_plot = ax.scatter(
        latent_points[0, 0], latent_points[0, 1],
        c=[0], cmap=cmap, s=180, zorder=4, norm=norm, marker="*", label="start"
    )

    # LineCollection for segments drawn so far (updated each frame)
    lc = LineCollection([], colors=[], linewidths=1, alpha=0.8, zorder=1, cmap=cmap, norm=norm)
    ax.add_collection(lc)

    # Moving "current position" dot
    current_plot = ax.scatter(
        latent_points[0, 0], latent_points[0, 1],
        c=[0], cmap=cmap, s=40, zorder=5, norm=norm
    )

    # End marker (X) – hidden until last frame
    end_plot = ax.scatter(
        latent_points[-1, 0], latent_points[-1, 1],
        c=[n - 1], cmap=cmap, s=60, zorder=4, norm=norm, marker="X", label="end",
        visible=False
    )

    def init():
        lc.set_segments([])
        lc.set_colors([])
        current_plot.set_offsets(latent_points[0:1])
        current_plot.set_array(np.array([0]))
        end_plot.set_visible(False)
        return lc, current_plot, end_plot

    def update(frame):
        # frame 0: only start visible
        # frame 1..n-1: draw segment from frame-1 to frame, move current dot
        # frame n: show end marker
        if frame == 0:
            lc.set_segments([])
            lc.set_colors([])
            current_plot.set_offsets(latent_points[0:1])
            current_plot.set_array(np.array([0]))
            end_plot.set_visible(False)
        elif frame < n:
            # segments from 0..1, 1..2, ..., frame-1..frame
            segments = [
                latent_points[i : i + 2].reshape(1, 2, 2)
                for i in range(frame)
            ]
            if segments:
                segs = np.concatenate(segments, axis=0)
                lc.set_segments(segs)
                lc.set_colors(cmap(norm(np.arange(frame))))
            current_plot.set_offsets(latent_points[frame : frame + 1])
            current_plot.set_array(np.array([frame]))
            end_plot.set_visible(False)
        else:
            # final frame: full path + end marker
            segments = [
                latent_points[i : i + 2].reshape(1, 2, 2)
                for i in range(n - 1)
            ]
            segs = np.concatenate(segments, axis=0)
            lc.set_segments(segs)
            lc.set_colors(cmap(norm(np.arange(n - 1))))
            current_plot.set_offsets(latent_points[-1:])  # or hide it
            current_plot.set_array(np.array([n - 1]))
            end_plot.set_visible(True)
        return lc, current_plot, end_plot

    total_frames = n + 1  # 0, 1, ..., n (last frame shows end marker)
    anim = FuncAnimation(
        ax.figure, update, init_func=init,
        frames=total_frames, interval=interval, repeat=repeat, blit=False
    )
    return anim

def plot_latent_kde(
    latent_points,
    ax=None,
    cmap="Blues",
    shade=True,
    n_levels=50,
    bw_method=None,
    kernel=None,  # Not used with scikit-learn KernelDensity
    gridsize=100,
    cut=3,
    thresh=0.05,
    **kwargs,
):
    """
    Plot a 2D kernel density estimation (KDE) of the given latent points using scikit-learn.

    Parameters:
    -----------
    latent_points : np.ndarray
        Array of shape (N, 2), where each row corresponds to a point in 2D latent space.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, otherwise creates a new one.
    cmap : str, optional
        Colormap to use for KDE.
    shade : bool, optional
        If True, fill the density area.
    n_levels : int, optional
        Number of contour levels.
    bw_method : float or str, optional
        Bandwidth for scikit-learn KernelDensity. If None, will use sklearn default.
    gridsize : int, optional
        Number of points along each dimension of the evaluation grid.
    cut : float, optional
        How many bandwidths to extend the KDE past the extreme values.
    thresh : float, optional
        Threshold level to fill the density areas.
    **kwargs : 
        Additional keyword arguments passed to contourf/contour.
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis with the KDE plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity

    latent_points = np.asarray(latent_points)
    if latent_points.ndim != 2 or latent_points.shape[1] != 2:
        raise ValueError("latent_points must be a (N, 2) array of 2D points.")

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 6)))

    # 1. Fit KDE
    if bw_method is not None:
        bandwidth = bw_method if isinstance(bw_method, (float, int)) else 1.0
    else:
        bandwidth = 1.0

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(latent_points)

    # 2. Make grid to evaluate KDE
    xmin, xmax = latent_points[:, 0].min(), latent_points[:, 0].max()
    ymin, ymax = latent_points[:, 1].min(), latent_points[:, 1].max()
    # Apply cut
    dx = (xmax - xmin) * cut / gridsize
    dy = (ymax - ymin) * cut / gridsize
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    eval_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # 3. Evaluate the density model on the grid
    Z = np.exp(kde.score_samples(eval_points))
    Z = Z.reshape(xx.shape)

    # 4. Plot density
    cmap_instance = plt.get_cmap(cmap)
    contour_method = ax.contourf if shade else ax.contour
    # Avoid very low density for visualization
    levels = np.linspace(thresh * Z.max(), Z.max(), n_levels)
    cset = contour_method(
        xx, yy, Z, levels=levels, cmap=cmap_instance, **kwargs
    )

    # 5. Optionally overlay sample points
    ax.scatter(latent_points[:, 0], latent_points[:, 1], s=8, c='k', alpha=0.2, label='points')

    ax.set_title("2D Kernel Density Estimation of Latent Space (sklearn)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    return ax

