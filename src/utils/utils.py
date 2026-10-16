import functools
import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time:.4f} seconds")
        return value

    return wrapper_timer


def get_relative_path(*path_parts):
    """
    Returns an absolute path relative to the caller script location.

    Example:
        get_relative_path('..', 'data', 'file.txt')
    """
    caller_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(caller_dir, *path_parts)


def load_maze_data():
    """
    Load wall and pellet positions from Unity tilemap files and return them as a tuple of two lists.


    Returns:
        tuple: (wall_positions, pellet_positions) where each is a list of (x,y) tuples
    """
    # Get the directory where utils.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to utils.py location
    walls_file = os.path.join(current_dir, "grid", "walls.unity")
    pellets_file = os.path.join(current_dir, "grid", "pellets.unity")

    with open(walls_file, "r") as f:
        walls_content = f.read()
    with open(pellets_file, "r") as f:
        pellets_content = f.read()

    wall_positions = parse_unity_tilemap_(walls_content)
    pellet_positions = parse_unity_tilemap_(pellets_content)

    return wall_positions, pellet_positions


def parse_unity_tilemap_(file_content):
    """
    Parse Unity tilemap file content to extract tile positions.

    Args:
        file_content (str): Content of the Unity tilemap file

    Returns:
        list: List of (x, y) tuples representing tile positions
    """
    positions = []
    current_pos = None

    for line in file_content.split("\n"):
        line = line.strip()

        # Look for position declarations
        if line.startswith("- first:"):
            # Reset current position
            current_pos = None

        # Extract x coordinate
        if "x:" in line:
            x = int(line.split("x:")[1].split(",")[0].strip())

        # Extract y coordinate
        if "y:" in line:
            y = int(line.split("y:")[1].split(",")[0].strip())
            current_pos = (x, y)

        if "m_TileIndex:" in line and current_pos:
            positions.append(current_pos)
            current_pos = None

    return positions


def pos_mirroring(df, return_quadrant=False):
    """
    Mirror the positions of Pacman
    on each quadrant of the maze. Each quadrant
    will mimic the first quadrant (upper right).
    If return_quadrant is True, add a column
    'quadrant' to the dataframe with the quadrant
    that the Pacman is in.
    """
    MIRROR_X = 0.0
    MIRROR_Y = -0.5

    mirrored_df = df.copy()
    if return_quadrant:
        mirrored_df["quadrant"] = np.float64(0)
    for i, row in mirrored_df.iterrows():
        if row["Pacman_X"] < MIRROR_X:
            mirrored_df.loc[i, "Pacman_X"] = (MIRROR_X - row["Pacman_X"]) + MIRROR_X
        if row["Pacman_Y"] < MIRROR_Y:
            mirrored_df.loc[i, "Pacman_Y"] = (MIRROR_Y - row["Pacman_Y"]) + MIRROR_Y
        if return_quadrant:
            if row["Pacman_X"] >= MIRROR_X and row["Pacman_Y"] >= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 1.0
            elif row["Pacman_X"] <= MIRROR_X and row["Pacman_Y"] >= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 2.0
            elif row["Pacman_X"] <= MIRROR_X and row["Pacman_Y"] <= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 3.0
            elif row["Pacman_X"] >= MIRROR_X and row["Pacman_Y"] <= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 4.0

    return mirrored_df


def calculate_velocities(
    trajectory: np.ndarray, round: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate velocities from position data, it rounds and removes signed zeros to avoid noise issues.

    Args:
        trajectory: Array of shape (N, 2) containing x,y coordinates
        round: Whether to round velocities to 0.5 to remove small noise in direction changes

    Returns:
        dx: Array of x-velocities
        dy: Array of y-velocities
    """
    x, y = trajectory[:, 0], trajectory[:, 1]

    # Calculate velocities
    if round:
        dx = (
            np.round(np.diff(x, prepend=x[0]) * 2) / 2
        )  # round to 0.5 to remove small noise in direction changes
        dy = np.round(np.diff(y, prepend=y[0]) * 2) / 2
    else:
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])

    dx = pd.Series(dx).replace(0, 0).values  # remove signed zeros using .loc
    dy = pd.Series(dy).replace(0, 0).values

    dx = np.nan_to_num(dx, nan=0)
    dy = np.nan_to_num(dy, nan=0)

    return dx, dy



def replace_inf_with_feature_max(data: np.ndarray, padding_value: float | None = None) -> np.ndarray:
    """
    Replace infinite values with the largest finite value of their own feature.

    Some features are legitimately infinite for part of a sequence - a ghost distance is inf
    while that ghost isn't reachable (still in the house, or eaten) - and models can't train on
    that: it makes the reconstruction loss inf/NaN on the first batch and every weight NaN from
    there on. The torch datasets already handle this themselves (PacmanDataset/ImputationDataset
    build an obs_mask from isfinite and call replace_inf, see src/datahandlers/datamodule.py);
    this is the equivalent for the code paths that feed numpy arrays straight to a model or a
    reducer.

    A feature that is infinite everywhere across the sample has no finite maximum to fall back
    on, so it gets 0 - the same "no information" placeholder the feature normalization uses.

    Args:
        data: Array whose last axis is the feature axis, e.g. [n_samples, seq_len, n_features].
        padding_value: Padding sentinel to leave out of the per-feature maximum. It is finite, so
            without this an all-inf feature would be filled with the sentinel instead of 0.

    Returns:
        np.ndarray: The input unchanged when it holds no infinities, otherwise a copy with them
            replaced per feature.
    """
    if not np.any(np.isinf(data)):
        return data

    cleaned = data.copy()
    for feat_idx in range(data.shape[-1]):
        feat_data = data[..., feat_idx]
        candidates = np.isfinite(feat_data)
        if padding_value is not None:
            candidates &= feat_data != padding_value
        finite_values = feat_data[candidates]
        fill_value = finite_values.max() if finite_values.size else 0.0
        cleaned[..., feat_idx] = np.where(np.isinf(feat_data), fill_value, feat_data)

    return cleaned


def neighborhood_hit(X_embedded,
                    labels, 
                    n_neighbors=5, 
                    no_null_instances=True,
                    metric='euclidean'):
    """
    Compute the neighborhood hit ratio for a low-dimensional embedding.

    The neighborhood hit (Paulovich et al., 2008) is the mean fraction of each point's
    k nearest neighbors that share its label. Its value under no structure at all depends
    on the label's class balance, so raw values are not comparable across label sets -
    see adjusted_neighborhood_hit() for the chance-corrected version.

    References:
        Paulovich, F. V., Nonato, L. G., Minghim, R., & Levkowitz, H. (2008). Least Square
            Projection: A fast high-precision multidimensional projection technique and its
            application to document mapping. IEEE Transactions on Visualization and
            Computer Graphics, 14(3), 564-575.
        Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S. T., & Telea, A. C. (2021).
            Toward a quantitative survey of dimension reduction techniques. IEEE
            Transactions on Visualization and Computer Graphics, 27(3), 2153-2173.

    Args:
        X_embedded: Array of shape (n_samples, n_components) with embedded points.
        labels: Array of shape (n_samples,) with class labels.
        n_neighbors: Number of nearest neighbors to consider.
        no_null_instances: If True, exclude samples labeled as -1 from the final average.
        metric: Distance metric used by ``NearestNeighbors``.

    Returns:
        float: Mean fraction of same-labeled neighbors.
    """

    assert len(labels.shape) == 1
    assert len(X_embedded.shape) == 2
    X_embedded = np.asarray(X_embedded)
    labels = np.asarray(labels)
    n_samples = X_embedded.shape[0]
    if labels.shape[0] != n_samples:
        raise ValueError("labels must have same length as X_embedded")

    # +1 because we get the point itself as neighbor at distance 0
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(X_embedded)
    distances, indices = nn.kneighbors(X_embedded)

    # Exclude self (first column)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = labels[neighbor_indices]

    same_label = (neighbor_labels == labels[:, None])

    if no_null_instances:
        no_null_mask = labels != -1
        result = same_label[no_null_mask].mean()
    else:
        # Average fraction of same-labeled neighbors
        result = same_label.mean()

    return result


def neighborhood_hit_chance(labels, no_null_instances=True) -> float:
    """
    Expected neighborhood hit when labels carry no neighborhood structure.

    Exact expectation of neighborhood_hit() under a random permutation of the labels over
    a fixed neighbor graph: a query point of class c has each neighbor (any other point,
    including absent ones) in class c with probability (n_c - 1) / (n - 1), so

        E = sum_c (n_c / n_query) * (n_c - 1) / (n - 1)

    over the query classes - every class except -1 when no_null_instances is True (absent
    points still count as neighbors, as in neighborhood_hit), every class otherwise. It
    does not depend on the number of neighbors. For large classes this is close to the
    Simpson index sum_c p_c^2; the (n_c - 1) / (n - 1) term is what keeps it exact for small
    samples and rare classes. With a single query class (a presence/absence label with
    absences excluded) it is that class's prevalence among the other points.

    Args:
        labels: Array of shape (n_samples,) with class labels.
        no_null_instances: If True, samples labeled -1 are not queried (matching
            neighborhood_hit's argument of the same name).

    Returns:
        float: Expected neighborhood hit, or NaN when it is undefined (fewer than two
            samples, or no query samples).
    """
    labels = np.asarray(labels)
    n_samples = labels.shape[0]
    query_labels = labels[labels != -1] if no_null_instances else labels
    if n_samples < 2 or query_labels.size == 0:
        return float("nan")

    _, counts = np.unique(query_labels, return_counts=True)
    return float(np.sum((counts / query_labels.size) * (counts - 1) / (n_samples - 1)))


def chance_adjusted_neighborhood_hit(hit: float, chance: float) -> float:
    """
    Rescale a neighborhood hit so chance level maps to 0 and a perfect hit to 1:
    (hit - chance) / (1 - chance). Negative values mean neighbors share labels less often
    than chance. NaN when chance is 1 (every point in one class), where there is nothing
    left to separate.

    See adjusted_neighborhood_hit() for the rationale and references.
    """
    if np.isnan(hit) or np.isnan(chance) or chance >= 1.0:
        return float("nan")
    return float((hit - chance) / (1.0 - chance))


def adjusted_neighborhood_hit(X_embedded,
                              labels,
                              n_neighbors=5,
                              no_null_instances=True,
                              metric='euclidean') -> float:
    """
    Chance-adjusted neighborhood hit: (NH - E[NH]) / (1 - E[NH]).

    The raw neighborhood hit (Paulovich et al., 2008) has a chance level that depends on
    class balance: with one dominant class, random neighbors already agree most of the time,
    while with ten balanced classes they agree about 10% of the time. That makes raw values
    incomparable across validation label sets. Subtracting the expected value under random
    labels (neighborhood_hit_chance) and dividing by its maximum attainable gain puts every
    label on the same scale: 0 at chance, 1 when every neighbor shares the label, negative
    when neighbors share labels less than chance.

    This is the same chance correction as Cohen's kappa, and the one behind the Adjusted
    Rand Index (Hubert & Arabie, 1985) and Adjusted Mutual Information (Vinh et al., 2010)
    used for the clusterings, so it reads on the same scale as those. Computed on the
    k-nearest-neighbor graph, it has the form of Newman's (2003) assortativity coefficient
    for categorical node attributes, with the expected term taken under label permutation
    rather than from edge-end marginals.

    References:
        Paulovich, F. V., Nonato, L. G., Minghim, R., & Levkowitz, H. (2008). Least Square
            Projection: A fast high-precision multidimensional projection technique and its
            application to document mapping. IEEE Transactions on Visualization and
            Computer Graphics, 14(3), 564-575.
        Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and
            Psychological Measurement, 20(1), 37-46.
        Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification,
            2(1), 193-218.
        Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for
            clusterings comparison: Variants, properties, normalization and correction for
            chance. Journal of Machine Learning Research, 11, 2837-2854.
        Newman, M. E. J. (2003). Mixing patterns in networks. Physical Review E, 67(2),
            026126.

    Args:
        X_embedded: Array of shape (n_samples, n_components) with embedded points.
        labels: Array of shape (n_samples,) with class labels.
        n_neighbors: Number of nearest neighbors to consider.
        no_null_instances: If True, samples labeled -1 are not queried (they still count
            as neighbors), as in neighborhood_hit.
        metric: Distance metric used by ``NearestNeighbors``.

    Returns:
        float: Chance-adjusted neighborhood hit, NaN when undefined (see
            chance_adjusted_neighborhood_hit).
    """
    hit = neighborhood_hit(X_embedded, labels, n_neighbors=n_neighbors,
                           no_null_instances=no_null_instances, metric=metric)
    chance = neighborhood_hit_chance(labels, no_null_instances=no_null_instances)
    return chance_adjusted_neighborhood_hit(hit, chance)


def nearest_neighbor_indices(X_embedded, n_neighbors: int, metric: str = "euclidean") -> np.ndarray:
    """
    Each point's n_neighbors nearest other points, nearest first, shape (n_samples, n_neighbors).

    Drops the first column of the search as the point itself, the same convention as
    neighborhood_hit(), so hits computed from these indices match it exactly.

    Args:
        X_embedded: Array of shape (n_samples, n_dims).
        n_neighbors: Number of neighbors per point, at most n_samples - 1.
        metric: Distance metric used by ``NearestNeighbors``.
    """
    X_embedded = np.asarray(X_embedded)
    if not 1 <= n_neighbors < X_embedded.shape[0]:
        raise ValueError(f"n_neighbors must be between 1 and {X_embedded.shape[0] - 1}, got {n_neighbors}")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(X_embedded)
    return nn.kneighbors(X_embedded, return_distance=False)[:, 1:]


def neighborhood_hit_profile(neighbor_indices: np.ndarray,
                             labels,
                             ks,
                             no_null_instances: bool = True) -> pd.DataFrame:
    """
    Neighborhood hit at several neighborhood sizes k, from one neighbor search.

    The hit at k = 5 measures purely local agreement; raising k widens the neighborhood until,
    at k = n - 1, it holds every other point and the adjusted hit is 0 by definition. The
    shape of the curve shows how far a label's structure reaches: a flat stretch ending in a
    drop marks the size of a group, a steady decline a gradient, an immediate fall structure
    that is only local. This follows the multi-scale evaluation of embeddings, where quality
    is reported as a function of neighborhood size (Lee & Verleysen, 2009; Lee et al., 2015),
    here with the labels' agreement instead of neighborhood preservation.

    Neighbors come sorted by distance, so the k nearest are a prefix of the k_max nearest:
    one search at max(ks) (nearest_neighbor_indices) serves every k, and a running count
    along each row gives every hit at once. The chance level does not depend on k
    (neighborhood_hit_chance), so the adjusted values are comparable along the curve.

    References:
        Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction:
            Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
        Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities
            in stochastic neighbour embedding: Reducing dimensionality while preserving both
            local and global structure. Neurocomputing, 169, 246-261.

    Args:
        neighbor_indices: Array of shape (n_samples, k_max) from nearest_neighbor_indices(),
            with k_max >= max(ks).
        labels: Array of shape (n_samples,) with class labels.
        ks: Neighborhood sizes to report, each between 1 and k_max.
        no_null_instances: If True, samples labeled -1 are not queried (they still count as
            neighbors), as in neighborhood_hit().

    Returns:
        pd.DataFrame: One row per k, columns 'k', 'neigh_hit', 'neigh_hit_chance',
            'neigh_hit_adj'. Hits are NaN when no sample is queried.
    """
    labels = np.asarray(labels)
    ks = np.asarray(sorted(set(int(k) for k in ks)))
    if ks.min() < 1 or ks.max() > neighbor_indices.shape[1]:
        raise ValueError(f"ks must lie between 1 and {neighbor_indices.shape[1]} (the neighbors searched)")

    query = labels != -1 if no_null_instances else np.ones(labels.shape[0], dtype=bool)
    chance = neighborhood_hit_chance(labels, no_null_instances=no_null_instances)
    if query.any():
        same = labels[neighbor_indices[query, :ks.max()]] == labels[query, None]
        counts = same.cumsum(axis=1, dtype=np.int32)[:, ks - 1]
        hits = counts.mean(axis=0) / ks
    else:
        hits = np.full(len(ks), np.nan)

    return pd.DataFrame({
        "k": ks,
        "neigh_hit": hits,
        "neigh_hit_chance": chance,
        "neigh_hit_adj": [chance_adjusted_neighborhood_hit(h, chance) for h in hits],
    })


def hold_last_step_padding(data: np.ndarray, padding_value: float = -999.0) -> np.ndarray:
    """
    Fill each sequence's padded steps with its last valid step.

    For models that need a real value in every column, such as a linear probe on the
    flattened sequences: left at the sentinel, the padded columns hand them the sequence
    length for free. A step is valid when any of its features differs from padding_value.

    Args:
        data: Padded array of shape [n_samples, seq_len, n_features].
        padding_value: Padding sentinel.

    Returns:
        np.ndarray: A copy with the padding replaced. Sequences with no valid step keep
            their padding.
    """
    filled = data.copy()
    lengths = (data != padding_value).any(axis=-1).sum(axis=1)
    for i, length in enumerate(lengths):
        if 0 < length < data.shape[1]:
            filled[i, length:] = data[i, length - 1]
    return filled


def ridge_probe_r2(X: np.ndarray,
                   y: np.ndarray,
                   n_folds: int = 5,
                   alphas: np.ndarray | None = None,
                   random_state: int = 42,
                   constant_spread: float = 1e-3,
                   clip: float | None = 10.0) -> float:
    """
    Linear probe: out-of-fold R² of a standardised ridge regression from X to y.

    Asks whether a space carries a quantity along a direction, which the neighborhood hit
    (local agreement of binned labels) does not: a label can sit in tight but scattered
    islands (high hit, low R²) or vary smoothly along an axis that neighborhoods ignore
    (low hit, high R²). The regularization strength is picked by RidgeCV inside each
    training fold, and every prediction comes from a model that never saw that sample.

    On a 2-D UMAP map it is near zero by construction (a non-linear map holds no global
    linear gradient), so it is mainly a probe of full-dimensional spaces: a latent, or the
    flattened input (see hold_last_step_padding for its padding).

    Columns whose spread is below constant_spread x the median column's are dropped first:
    standardised, their numerically constant values become huge z-scores on held-out
    samples, which no ridge penalty fixes. The drop uses no labels, so nothing leaks.

    Args:
        X: Array of shape (n_samples, n_dims).
        y: Continuous target of shape (n_samples,), finite.
        n_folds: Cross-validation folds.
        alphas: Candidate ridge penalties, default 10^-2 ... 10^6.
        random_state: Seed for the fold split.
        constant_spread: Relative spread below which a column counts as constant.
        clip: Bound on each standardised feature (None for no bound). A column that is nearly
            constant in a training fold but not in the held-out one (a rare value) gives a
            held-out sample a z-score in the hundreds, and with it a prediction far outside
            the target's range: one such sample can take R² from 0.9 to below 0. Clipping at
            ±10 leaves ordinary values untouched and caps that.

    Returns:
        float: 1 - SS_res / SS_tot over the out-of-fold predictions. 1 is a perfect linear
            read-out, 0 no better than predicting the mean, negative worse than that.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if alphas is None:
        alphas = np.logspace(-2, 6, 17)

    spread = X.std(axis=0)
    X = X[:, spread >= constant_spread * np.median(spread)]

    steps = [StandardScaler()]
    if clip is not None:
        steps.append(FunctionTransformer(np.clip, kw_args={"a_min": -clip, "a_max": clip}))
    model = make_pipeline(*steps, RidgeCV(alphas=alphas))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    predicted = cross_val_predict(model, X, y, cv=folds)
    residual = np.sum((y - predicted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return float(1 - residual / total)


def make_synth_data(n_per_category: int | tuple[int] = (1000, 1000),
              n_noise: int = 1000,
              seed = 42):
    """
    Generate a synthetic 2D dataset with clustered categories and uniform noise.

    Helper method, not called by the main pipeline: kept for ad-hoc use (e.g. sanity-checking
    clustering or validation measures on data with a known ground truth).

    Args:
        n_per_category: Number of samples per category.
        n_noise: Number of noise points to generate.
        seed: Random seed for reproducibility.

    Returns:
        pandas.DataFrame: DataFrame with columns ``x``, ``y``, and ``label``.
    """

    np.random.seed(seed) 
    dataframe_list = []

    for i, cat_size in enumerate(n_per_category):
        centroid =(np.random.random(size=2) * 20) - 10
        spread = np.random.random(size=2)

        dataframe_list.append(pd.DataFrame({
            "x": np.random.normal(loc= centroid[0], scale = spread[0], size= cat_size),
            "y": np.random.normal(loc= centroid[1], scale = spread[1], size= cat_size),
            "label": i
        })
        )

    df = pd.concat(dataframe_list, ignore_index=True)

    max = (df["x"].max(), df["y"].max())
    min = (df["x"].min(),df["y"].min())

    noise = pd.DataFrame({
        "x": np.random.uniform(low= min[0], high = max[0], size= n_noise),
        "y": np.random.uniform(low= min[1], high = max[1], size= n_noise),
        "label" : -1
    })
    
    df = pd.concat([df, noise], ignore_index=True)


    return df


def random_projection_measures(processed_sequence_data):
    ## Create random projection from the numpy array [n_samples, seq_len, features]
    random_embeddings = np.random.random(size=(len(processed_sequence_data), 2)) # Random projection to 2D
    from hdbscan import HDBSCAN
    random_labels = HDBSCAN().fit_predict(random_embeddings)

    # res_ran = pa.calculate_validation_measures(
    #     random_labels,
    #     pa.validation_labels,
    #     embeddings=random_embeddings,
    #     no_null_instances=True,
    #     neighborhood_k=5
    # )

    return random_embeddings, random_labels