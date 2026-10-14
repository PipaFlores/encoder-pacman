"""Probes of where a latent space's structure comes from, for the report's RQ1.

The sweep scores every space after UMAP has reduced it to two dimensions, so a high score
cannot tell the encoder's work from UMAP's. These probes score the same validation labels
in the spaces on either side of UMAP:

- **Neighbourhood hit** with a full-dimensional distance: the chance-adjusted share of each
  sequence's five nearest neighbours that carry its label, exactly as
  `PatternAnalysis.calculate_validation_measures()` computes it, but with the neighbours
  found in the encoder's latent space or in the raw input instead of the 2-D map.
- **Ridge regression**, out-of-fold R² from the space to the label's continuous value: does
  the space carry the quantity linearly? On a 2-D map this is near zero by construction, so
  it is mainly a probe of the full-dimensional spaces.

Distances. Encoder latents use the metric the sweep's UMAP used (cosine), with Euclidean as
a check. Raw sequences are padded with -999.0, so a plain metric on them is dominated by
sequence length; they are probed with the padding-aware version of the same metric (what
"Masked UMAP" uses, src/utils/masked_distance.py) and, to show that leak, with the plain
metric on the padded input (what plain UMAP sees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import MASKED_METRICS, replace_inf_with_feature_max
from src.utils.masked_distance import PADDING_VALUE
from src.utils.utils import chance_adjusted_neighborhood_hit, neighborhood_hit_chance

N_NEIGHBORS = 5  # as in calculate_validation_measures
RIDGE_ALPHAS = np.logspace(-2, 6, 17)
# Dimensions whose spread is below this share of the median dimension's are numerically
# constant. Standardised, their tiny differences become z-scores in the tens, and a held-out
# sequence lands on a prediction in the thousands (an untrained Transformer gave R² of -80
# with five such dimensions); no ridge penalty fixes that, so they are dropped first.
CONSTANT_SPREAD = 1e-3
RIDGE_FOLDS = 5
MIN_RIDGE_SAMPLES = 50


# --------------------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------------------


def flat_padded(processed: np.ndarray) -> np.ndarray:
    """The flattened, padded sequences exactly as the no-encoder UMAP embeds them."""
    data = replace_inf_with_feature_max(processed, padding_value=PADDING_VALUE)
    return data.reshape(len(data), -1)


def flat_hold_last(processed: np.ndarray) -> np.ndarray:
    """Flattened sequences with each sequence's last valid step repeated over its padding.

    For the ridge probe, which needs a value in every column: -999 there would hand it the
    sequence length for free, and repeating the last step is the neutral alternative.
    """
    data = replace_inf_with_feature_max(processed, padding_value=PADDING_VALUE).copy()
    valid = (data != PADDING_VALUE).all(axis=2)
    lengths = valid.sum(axis=1)
    for i, length in enumerate(lengths):
        if 0 < length < data.shape[1]:
            data[i, length:] = data[i, length - 1]
    return data.reshape(len(data), -1)


# --------------------------------------------------------------------------------------
# Neighbourhood hit
# --------------------------------------------------------------------------------------


def neighbor_indices(
    X: np.ndarray, metric: str, padding_aware: bool = False, seed: int = 42
) -> np.ndarray:
    """Each point's N_NEIGHBORS nearest other points, shape (n, N_NEIGHBORS).

    Plain metrics use an exact search. Padding-aware ones are numba functions that sklearn
    can only call one pair at a time, so they go through pynndescent, the approximate
    nearest-neighbour search UMAP itself uses, with a wider candidate list for accuracy.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if padding_aware:
        from pynndescent import NNDescent

        index = NNDescent(
            X,
            metric=MASKED_METRICS[metric],
            n_neighbors=N_NEIGHBORS + 10,
            random_state=seed,
            n_jobs=1,
        )
        candidates, _ = index.neighbor_graph
    else:
        from sklearn.neighbors import NearestNeighbors

        search = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric=metric).fit(X)
        _, candidates = search.kneighbors(X)

    # Drop each point itself wherever the search put it (ties can move it off column 0).
    out = np.empty((len(X), N_NEIGHBORS), dtype=np.int64)
    for i, row in enumerate(candidates):
        others = row[row != i]
        out[i] = others[:N_NEIGHBORS]
    return out


def neighborhood_hits(
    indices: np.ndarray, validation_labels: pd.DataFrame
) -> pd.DataFrame:
    """Neighbourhood hit per validation set from a neighbour graph, with the same label
    handling as calculate_validation_measures(): sets with no signal are skipped, sequences
    without an instance (-1) are not queried but still count as neighbours."""
    rows = []
    for column in validation_labels.columns:
        labels = validation_labels[column].to_numpy()
        if pd.isnull(labels).all():
            continue
        query = labels != -1
        if not query.any():
            continue
        same = labels[indices] == labels[:, None]
        hit = float(same[query].mean())
        chance = neighborhood_hit_chance(labels, no_null_instances=True)
        rows.append(
            {
                "validation_set": column,
                "neigh_hit": hit,
                "neigh_hit_chance": chance,
                "neigh_hit_adj": chance_adjusted_neighborhood_hit(hit, chance),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------------------


def ridge_r2(
    X: np.ndarray,
    validation_encodings: pd.DataFrame,
    validation_labels: pd.DataFrame,
    targets: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Out-of-fold R² of a standardised ridge regression from X to each target's value.

    Fitted on the same sequences the neighbourhood hit queries: those whose label is not
    -1 (e.g. attacks with at least one ghost kill for Aggression3).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    spread = X.std(axis=0)
    X = X[:, spread >= CONSTANT_SPREAD * np.median(spread)]  # label-free, so no leakage

    rows = []
    for target in targets:
        if target not in validation_encodings:
            continue
        y = pd.to_numeric(validation_encodings[target], errors="coerce").to_numpy(float)
        keep = np.isfinite(y)
        if target in validation_labels:
            keep &= validation_labels[target].to_numpy() != -1
        if keep.sum() < MIN_RIDGE_SAMPLES or np.std(y[keep]) == 0:
            continue
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        folds = KFold(n_splits=RIDGE_FOLDS, shuffle=True, random_state=seed)
        predicted = cross_val_predict(model, X[keep], y[keep], cv=folds)
        residual = np.sum((y[keep] - predicted) ** 2)
        total = np.sum((y[keep] - y[keep].mean()) ** 2)
        rows.append(
            {
                "validation_set": target,
                "ridge_r2": 1 - residual / total,
                "n": int(keep.sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# One rebuilt space
# --------------------------------------------------------------------------------------


PROBE_KINDS = ("latent", "plain_umap", "masked_umap")


def probe_space(
    analysis, metric: str, kind: str, targets: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All probes of one rebuilt space, as long tables (one row per validation set).

    `kind` says what the rebuilt space is: "latent" (an encoder), "plain_umap" (UMAP on the
    padded input) or "masked_umap" (UMAP with the padding-aware metric). The raw input's
    full-dimensional probes are the same for both UMAP baselines, so they run once, with
    "plain_umap".

    Output columns: `representation` is "latent", "raw" (the input, padding-aware) or
    "raw_padded" (the input as plain UMAP sees it); `space` is "full" or "umap2d"; `metric`
    is the distance the neighbours were found with.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"kind must be one of {PROBE_KINDS}, got {kind!r}")
    labels = analysis.validation_labels
    encodings = analysis.validation_encodings
    reduced = np.asarray(analysis.reduced_embeddings)
    metrics = (metric, "euclidean") if metric != "euclidean" else (metric,)
    hits, ridges = [], []

    def add_hits(representation, space, metric_name, indices):
        frame = neighborhood_hits(indices, labels)
        hits.append(
            frame.assign(representation=representation, space=space, metric=metric_name)
        )

    def add_ridge(representation, space, X):
        frame = ridge_r2(X, encodings, labels, targets)
        ridges.append(frame.assign(representation=representation, space=space))

    # After UMAP: the 2-D map the sweep scores, with the pipeline's Euclidean neighbours.
    mapped = {"latent": "latent", "plain_umap": "raw_padded", "masked_umap": "raw"}[
        kind
    ]
    add_hits(mapped, "umap2d", "euclidean", neighbor_indices(reduced, "euclidean"))
    add_ridge(mapped, "umap2d", reduced)

    if kind == "latent":
        latent = np.asarray(analysis.embeddings)
        for name in metrics:
            add_hits("latent", "full", name, neighbor_indices(latent, name))
        add_ridge("latent", "full", latent)
    elif kind == "plain_umap":
        padded = flat_padded(analysis.processed_sequence_data)
        for name in metrics:
            add_hits(
                "raw", "full", name, neighbor_indices(padded, name, padding_aware=True)
            )
            add_hits("raw_padded", "full", name, neighbor_indices(padded, name))
        add_ridge("raw", "full", flat_hold_last(analysis.processed_sequence_data))

    return pd.concat(hits, ignore_index=True), pd.concat(ridges, ignore_index=True)
