import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from typing import Literal
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer


def calculate_latent_distances(latent_points):
    """
    Calculate Euclidean distances between consecutive latent points.

    The input is expected to be ordered by gameplay progression for one
    participant or sequence of interest. The returned array has one fewer value
    than the input because each value describes movement from one point to the
    next.
    """

    diff_vectors = np.diff(latent_points, axis = 0)

    path_distances = np.linalg.norm(diff_vectors, axis=1, ord=2) # euclidean distance
    # np.sqrt(np.sum(np.square(diff_vectors), axis=1)) # euclidean distance
    
    return path_distances


def shannon_entropy(labels, normalize: bool = False, ignore_noise: bool = False):
    """
    Calculate Shannon entropy for a sequence of discrete labels.

    Entropy is low when most observations belong to the same label and high when
    observations are spread across many labels. When ``normalize`` is True, the
    value is scaled to the observed number of labels, giving 0 for a single
    repeated label and 1 for a perfectly even distribution.
    """
    labels = np.asarray(labels)
    labels = labels[~pd.isnull(labels)]
    if ignore_noise:
        labels = labels[labels != -1]

    if labels.size == 0:
        return np.nan

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))

    if normalize:
        if len(probabilities) <= 1:
            return 0.0
        entropy = entropy / np.log(len(probabilities))

    return float(entropy)


def summarize_continuous_progression(
    latent_points,
    population_embeddings,
    bandwidth: Literal["silverman", "scott"] | int | float = 1,
    min_history: int = 2,
    global_step_normalizer = None ## (e.g, QuantileTransformer)
):
    """
    Reduce an ordered participant trajectory in latent space to scalar measures.

    These summaries describe how much the participant moves through latent space
    and how novel each point is under KDE-based density estimates.
    """
    latent_points = np.asarray(latent_points)
    n_points = len(latent_points)

    summary = {
        "n_sequences": n_points,
        "latent_path_total_distance": np.nan,
        "latent_path_mean_distance": np.nan,
        "latent_path_std_distance": np.nan,
        "latent_centroid_mean_distance": np.nan,
        "kde_step_novelty_mean": np.nan,
        "kde_step_novelty_std": np.nan,
        "kde_pop_relative_novelty_mean": np.nan,
        "kde_pop_relative_novelty_std": np.nan
    }

    if n_points == 0:
        return summary

    centroid = latent_points.mean(axis=0)
    centroid_distances = np.linalg.norm(latent_points - centroid, axis=1)
    summary["latent_centroid_mean_distance"] = float(np.mean(centroid_distances))

    if n_points > 1:
        path_distances = calculate_latent_distances(latent_points)
        total_distance = float(np.sum(path_distances))
        net_displacement = float(np.linalg.norm(latent_points[-1] - latent_points[0]))

        summary["latent_path_total_distance"] = total_distance
        summary["latent_path_mean_distance"] = float(np.mean(path_distances))
        summary["latent_path_std_distance"] = float(np.std(path_distances))

    if n_points >= 2:
        pop_relative = population_relative_novelty(latent_points, population_embeddings, bandwidth=bandwidth, normalize=True)
        summary["kde_pop_relative_novelty_mean"] = float(np.nanmean(pop_relative))
        summary["kde_pop_relative_novelty_std"] = float(np.nanstd(pop_relative))

    if n_points > min_history:
        step = step_novelty(
            latent_points,
            min_history=min_history,
            bandwidth=bandwidth,
            normalize=False, ## For cross-comparison normalization is applied after (through QuantileTransform)
            global_normalizer=global_step_normalizer
        )
        valid_step = step[~np.isnan(step)]
        if len(valid_step) > 0:
            summary["kde_step_novelty_mean"] = float(np.nanmean(valid_step))
            summary["kde_step_novelty_std"] = float(np.nanstd(valid_step))


    return summary


def summarize_cluster_progression(
    cluster_labels,
    ignore_noise_for_entropy: bool = False,
):
    """
    Reduce an ordered sequence of cluster (discrete) labels to scalar progression measures.

    High entropy and many visited clusters indicate broader exploration. High
    dominant-cluster fraction and repeated-cluster transitions indicate stronger
    exploitation/repetition.
    """
    cluster_labels = np.asarray(cluster_labels)
    n_points = len(cluster_labels)

    summary = {
        "visited_clusters": np.nan,
        "Shannon_entropy": np.nan,
        "Shannon_entropy_normalized": np.nan,
        "cluster_dominant_fraction": np.nan,
        "cluster_noise_fraction": np.nan,
        "cluster_transition_rate": np.nan,
        "cluster_repeat_fraction": np.nan,
    }

    if n_points == 0:
        return summary

    non_noise = cluster_labels[cluster_labels != -1]
    labels_for_counts = non_noise if ignore_noise_for_entropy else cluster_labels
    if len(labels_for_counts) == 0:
        labels_for_counts = cluster_labels

    unique_labels, counts = np.unique(labels_for_counts, return_counts=True)
    summary["visited_clusters"] = int(len(unique_labels))
    summary["Shannon_entropy"] = shannon_entropy(
        cluster_labels,
        normalize=False,
        ignore_noise=ignore_noise_for_entropy,
    )
    summary["Shannon_entropy_normalized"] = shannon_entropy(
        cluster_labels,
        normalize=True,
        ignore_noise=ignore_noise_for_entropy,
    )
    summary["cluster_dominant_fraction"] = float(np.max(counts) / np.sum(counts))
    summary["cluster_noise_fraction"] = float(np.mean(cluster_labels == -1))

    if n_points > 1:
        transitions = cluster_labels[1:] != cluster_labels[:-1]
        summary["cluster_transition_rate"] = float(np.mean(transitions))
        summary["cluster_repeat_fraction"] = float(1 - np.mean(transitions))

    return summary




def Recurrence_Quantification_Analysis(latent_points,
                                       epsilon: float = 0.65,
                                       return_recurrence_matrix: bool = False):
    import pyrqa
    from pyrqa.settings import Settings
    from pyrqa.time_series import EmbeddedSeries
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    from pyrqa.computation import RPComputation

    summary = {
            "Recurrence_Rate": np.nan,
            "Laminarity": np.nan,
            "Trapping_Time": np.nan,
            "V_max": np.nan,
        }
    
    # Set up variables and run computation
    time_series = EmbeddedSeries(latent_points)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(epsilon),   ## TODO This is an important Parameter to tune
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    
    rp_computation = RPComputation.create(settings)
    RP = rp_computation.run()

    recurrence_matrix = RP.recurrence_matrix

    recurrence_matrix = np.asarray(recurrence_matrix, dtype=int)

    rqa_computation = RQAComputation.create(settings, verbose=False)
    rqa = rqa_computation.run()

    # Extract relevant measures
    summary["Recurrence_Rate"] = rqa.recurrence_rate
    summary["Laminarity"] = rqa.laminarity
    summary["Trapping_Time"] = rqa.trapping_time
    summary["V_max"] = rqa.longest_vertical_line

    if return_recurrence_matrix:
        return summary, recurrence_matrix
    else:
        return summary


## Entropy-based 

def population_relative_novelty(
    latent_points,
    population_latent_set,
    bandwidth: Literal["scott","silverman"] | int = "scott",
    normalize=False,
    return_density = False
):
    
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(population_latent_set)

    pop_novelty = kde.score_samples(population_latent_set) * -1

    points_likelihood = kde.score_samples(latent_points)

    points_novelty = points_likelihood * -1

    if normalize:
        transformer = QuantileTransformer(
                n_quantiles=min(1000, len(pop_novelty)),
                output_distribution="normal",
                random_state=0,
            )
        transformer.fit(pop_novelty.reshape(-1, 1))
        points_novelty = transformer.transform(
            points_novelty.reshape(-1, 1)
        ).ravel()

        
    if return_density:
        points_density = np.exp(points_likelihood)
        return points_novelty, points_density

    return points_novelty

    
def step_novelty(latent_points, 
                    min_history=2, 
                    rolling_window=None, 
                    bandwidth: Literal["scott", "silverman"] | int = "scott",
                    normalize = False,
                    global_normalizer = None,
                    return_density = False):
    """
    Calculates novelty (similar to surprisal but in a continuous probability space) of each step.
    Estimate how unusual each latent point is relative to previous points.

    For each point after ``min_history``, a kernel density estimate is fit on the
    participant's earlier latent points and then evaluated at the current point.
    KDE estimates the log likelihood of probability density of a latent point given the fit. 
    The inverse of this log likelihood is novelty.

    Higher novelty means the current behavior is unlike the participant's prior
    trajectory; lower novelty means it falls in a region they have already
    occupied.

    Values before ``min_history`` are returned as ``NaN`` because there is not
    enough history to define a personal baseline. If ``return_density`` is True,
    the function also returns the estimated density 
    (exponential of log likelihood = likelihood/prob. density) for each point.

    Normalization does min-max scaling. Do not use to comparo across participants.
    For that provide a global_normalizer (e.g, fitted QuantileTransformer)
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

    # Optionally normalize novelty values (min-max scaling) while preserving NaNs
    # Only useful for within participant observations. For cross-comparisons, normalization needs to be done after all
    # raw novelties have been calculated.
    valid = np.isfinite(novelty)
    if global_normalizer is not None:
        if not hasattr(global_normalizer, "transform"):
            raise TypeError("global_normalizer must provide a transform method")

        if np.any(valid):
            novelty[valid] = global_normalizer.transform(
                novelty[valid].reshape(-1, 1)
            ).ravel()
    elif normalize:
        if np.any(valid):
            v = novelty[valid]
            vmin = np.min(v)
            vmax = np.max(v)
            if vmax > vmin:
                novelty_scaled = (v - vmin) / (vmax - vmin)
            else:
                # all values equal -> map to zeros
                novelty_scaled = np.zeros_like(v)
            novelty = novelty.copy()
            novelty[valid] = novelty_scaled


    if return_density:
        return novelty, densities
    return novelty

def fit_global_step_novelty_transform(
    participant_latent_points,
    min_history=2,
    bandwidth="scott",
    output_distribution="normal",
):
    """
    Fit a cross-participant calibration transform.

    participant_latent_points should be an iterable containing one latent
    trajectory per participant.
    """
    raw_values = []

    for latent_points in participant_latent_points:
        novelty = step_novelty(
            latent_points,
            min_history=min_history,
            bandwidth=bandwidth,
            normalize=False,
        )

        valid = novelty[np.isfinite(novelty)]
        raw_values.extend(valid)

    if not raw_values:
        raise ValueError("No valid step novelty values were found.")

    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(raw_values)),
        output_distribution=output_distribution,
        random_state=0,
    )
    transformer.fit(np.asarray(raw_values).reshape(-1, 1))

    return transformer

def heuristic_bandwidth(
    latent_points,
    method="scott",
    scale="mean_std",
):
    """Return one scalar bandwidth for sklearn KernelDensity."""
    x = np.asarray(latent_points, dtype=float)
    n, dimensions = x.shape

    if n < 2:
        raise ValueError("At least two latent points are required.")

    if scale == "mean_std":
        spread = np.mean(np.std(x, axis=0, ddof=1))
    elif scale == "median_std":
        spread = np.median(np.std(x, axis=0, ddof=1))
    else:
        raise ValueError("Unknown scale method.")

    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0

    if method == "scott":
        factor = n ** (-1.0 / (dimensions + 4))
    elif method == "silverman":
        factor = (n * (dimensions + 2) / 4.0) ** (
            -1.0 / (dimensions + 4)
        )
    else:
        raise ValueError("method must be 'scott' or 'silverman'.")

    return float(factor * spread)

def plot_latent_navigation(latent_points, ax, **config):
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

def plot_latent_distances(latent_points, ax):
    """
    Plot the euclidean distances between latent points
    """

    diff_vectors = np.diff(latent_points, axis = 0)
    # diff_vectors

    path_distances = np.linalg.norm(diff_vectors, axis=1, ord=2) # euclidean distance
    # np.sqrt(np.sum(np.square(diff_vectors), axis=1)) # euclidean distance
    # path_distances

    ax.plot(list(range(len(path_distances))) ,path_distances)
    ax.set_title("Euclidean Distance between sequences")
    ax.set_xlabel("Sequence ID")
    ax.set_ylabel("Euclidean distance")

    # ## Get session shifts
    # user_subset_sessions_num = pa.metadata_dictionary["session_number"][user_mask]
    # sessions_diffs = np.diff(user_subset_sessions_num)
    # np.unique_counts(sessions_diffs)
    # # Plot a vertical line whenever sessions_diffs == 1, using the index as the x value
    # for idx, diff in enumerate(sessions_diffs):
    #     if diff == 1:
    #         ax.axvline(x=idx, color='black', linestyle=':', alpha=0.3)
    
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
    kernel=None,
    gridsize=100,
    cut=3,
    thresh=0.05,
    use_seaborn=False,
    **config,
):
    """
    Plot a 2D kernel density estimation (KDE) of the given latent points.

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
        Bandwidth for the KDE. If None, will use the underlying library default.
    kernel : str, optional
        Kernel name for scikit-learn. Ignored when use_seaborn=True.
    gridsize : int, optional
        Number of points along each dimension of the evaluation grid.
    cut : float, optional
        How many bandwidths to extend the KDE past the extreme values.
    thresh : float, optional
        Threshold level to fill the density areas.
    use_seaborn : bool, optional
        If True, use seaborn.kdeplot instead of the scikit-learn implementation.
    **config :
        Additional keyword arguments passed to the plotting function.
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis with the KDE plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    latent_points = np.asarray(latent_points)
    if latent_points.ndim != 2 or latent_points.shape[1] != 2:
        raise ValueError("latent_points must be a (N, 2) array of 2D points.")

    if ax is None:
        _, ax = plt.subplots(figsize=config.pop("figsize", (8, 6)))

    if use_seaborn:
        
        import seaborn as sns
        # sns.set_theme(style="darkgrid")
        # print("caca")


        kde_kwargs = {
            "ax": ax,
            "fill": shade,
            "cmap": cmap,
            "levels": n_levels,
            "thresh": thresh,
            "cut": cut,
        }
        if bw_method is not None:
            bandwidth = bw_method if isinstance(bw_method, (float, int)) else 1.0
            kde_kwargs["bw_adjust"] = bandwidth

        sns.kdeplot(
            x=latent_points[:, 0],
            y=latent_points[:, 1],
            **kde_kwargs,
            **config,
        )
    else:
        from sklearn.neighbors import KernelDensity

        if bw_method is not None:
            bandwidth = bw_method if isinstance(bw_method, (float, int)) else 1.0
        else:
            bandwidth = 1.0

        kde = KernelDensity(kernel=kernel or "gaussian", bandwidth=bandwidth)
        kde.fit(latent_points)

        xmin, xmax = latent_points[:, 0].min(), latent_points[:, 0].max()
        ymin, ymax = latent_points[:, 1].min(), latent_points[:, 1].max()
        dx = (xmax - xmin) * cut / gridsize
        dy = (ymax - ymin) * cut / gridsize
        xmin -= dx
        xmax += dx
        ymin -= dy
        ymax += dy

        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, gridsize),
            np.linspace(ymin, ymax, gridsize),
        )
        eval_points = np.vstack([xx.ravel(), yy.ravel()]).T

        Z = np.exp(kde.score_samples(eval_points))
        Z = Z.reshape(xx.shape)

        cmap_instance = plt.get_cmap(cmap)
        contour_method = ax.contourf if shade else ax.contour
        levels = np.linspace(thresh * Z.max(), Z.max(), n_levels)
        contour_method(xx, yy, Z, levels=levels, cmap=cmap_instance, **config)

    ax.scatter(
        latent_points[:, 0],
        latent_points[:, 1],
        s=1,
        c="k",
        alpha=0.2,
        label="points",
    )

    ax.set_title("2D Kernel Density Estimation of Latent Space")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    return ax


def plot_recurrence_plot(
        recurrence_matrix,
        ax = None,
        **kwargs
):

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 6)))

    ax.imshow(recurrence_matrix, cmap="binary", origin="lower", interpolation="nearest")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Trial")
    ax.set_title("Recurrence Matrix")

    return ax


def plot_correlation_matrix(
        df,
        p_threshold = None,
        title = "Correlation Matrix.",
        ax= None,
        **kwargs
):
    """
    Plot correlation matrix with significance thresholds in upper triangle 
    """
    import seaborn as sns
    from scipy.stats import pearsonr

    if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (16, 14)))

    # Calculate correlation matrix and p-values
    corr_matrix = df.corr()
    p_values = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
    if p_threshold:
        title = title + f"significance threshold p < {p_threshold}."

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                x = df.iloc[:, i]
                y = df.iloc[:, j]
                mask = (~x.isna()) & (~y.isna())
                if mask.sum() >= 2:
                    _, p_val = pearsonr(x[mask], y[mask])
                else:
                    p_val = np.nan
                p_values.iloc[i, j] = p_val

    # Create significance markers for upper triangle
    significance_markers = np.empty_like(corr_matrix, dtype=object)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            p_val = p_values.iloc[i, j]
            if p_threshold is not None:
                if pd.isna(p_val):
                    significance_markers[i,j] = ''
                elif p_val <= p_threshold:
                    significance_markers[i,j] = '*'
                else:
                    significance_markers[i,j] = 'ns'
            else:
                if pd.isna(p_val):
                    significance_markers[i, j] = ''
                elif p_val < 0.001:
                    significance_markers[i, j] = '***'
                elif p_val < 0.01:
                    significance_markers[i, j] = '**'
                elif p_val < 0.05:
                    significance_markers[i, j] = '*'
                else:
                    significance_markers[i, j] = 'ns'

    # Prepare annotation: show correlation values only in the lower triangle (remove upper triangle values)
    annot_df = corr_matrix.round(2).astype(str)
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    annot_df.values[mask_upper] = ''

    # Plot heatmap with only lower-triangle numeric annotations
    ax = sns.heatmap(corr_matrix, annot=annot_df, fmt='', cmap='coolwarm', center=0,
                    cbar_kws={'label': 'Correlation'}, square=True)

    # Add significance markers to upper triangle
    for i, j in zip(*np.where(mask_upper)):
        sig = significance_markers[i, j]
        if sig:
            ax.text(j + 0.5, i + 0.5, sig, ha='center', va='center', fontsize=8, color='black')

    ax.set_title(title)
    return ax
