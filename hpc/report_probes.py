"""Probes of where a latent space's structure comes from, for the report's RQ1.

The sweep scores every space after UMAP has reduced it to two dimensions, so a high score
cannot tell the encoder's work from UMAP's. These probes score the same validation labels
in the spaces on either side of the reduction (UMAP or PCA after an encoder; UMAP, Masked
UMAP, PCA or a random projection straight from the raw input; to 2-D, 5-D or the latent's
own width):

- **Neighbourhood hit** with a full-dimensional distance: the chance-adjusted share of each
  sequence's five nearest neighbours that carry its label, exactly as
  `PatternAnalysis.calculate_validation_measures()` computes it, but with the neighbours
  found in the encoder's latent space or in the raw input instead of the 2-D map.
- **Neighbourhood-hit profile**: the same chance-adjusted hit at neighbourhood sizes from 5
  to 1000 (`PatternAnalysis.calculate_neighborhood_hit_profile()`), which shows how far each
  label's structure reaches: local islands fall off quickly, gradients and large groups
  slowly.
- **Ridge regression**, out-of-fold R² from the space to the label's continuous value: does
  the space carry the quantity linearly? This is the pipeline's own linear probe,
  `PatternAnalysis.calculate_probe_measures()`, run on the same three representations. On a
  2-D map it is near zero by construction, so it is mainly a probe of the full-dimensional
  spaces. Unlike the two above it sees the labels, so it asks whether the information is
  there at all, not whether the space makes it salient.
- **Length-only ridge**: the same probe from sequence length alone, a one-column space. Many
  Behavlets grow or shrink with how long the attack lasts, so this is the R² any space gets
  for free by knowing the length.

Distances. Encoder latents use the metric the sweep's UMAP used (cosine), with Euclidean as
a check. Raw sequences are padded with -999.0, so a plain metric on them is dominated by
sequence length; they are probed with the padding-aware version of the same metric (what
"Masked UMAP" uses, src/utils/masked_distance.py) and, to show that leak, with the plain
metric on the padded input (what plain UMAP sees).
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from src.utils import MASKED_METRICS, replace_inf_with_feature_max
from src.utils.masked_distance import PADDING_VALUE
from src.utils.utils import (
    chance_adjusted_neighborhood_hit,
    nearest_neighbor_indices,
    neighborhood_hit_chance,
    neighborhood_hit_profile,
    ridge_probe_r2,
)

N_NEIGHBORS = 5  # as in calculate_validation_measures
# Neighbourhood sizes of the hit profile (calculate_neighborhood_hit_profile's default).
PROFILE_KS = (5, 10, 20, 50, 100, 200, 500, 1000)


# --------------------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------------------


def flat_padded(processed: np.ndarray) -> np.ndarray:
    """The flattened, padded sequences exactly as the no-encoder UMAP embeds them."""
    data = replace_inf_with_feature_max(processed, padding_value=PADDING_VALUE)
    return data.reshape(len(data), -1)


# --------------------------------------------------------------------------------------
# Neighbourhood hit
# --------------------------------------------------------------------------------------


_MASKED_SEARCHES = {}


def _masked_search(metric: str):
    """An exact, parallel k-nearest-neighbour search under one padding-aware metric.

    sklearn can only call a numba metric one pair at a time, and an approximate search
    (pynndescent, what UMAP uses) does not hold up at k in the hundreds, so this compares
    every pair: n² distances, a minute or so for the attack sequences.
    """
    if metric not in _MASKED_SEARCHES:
        distance = MASKED_METRICS[metric]

        @numba.njit(parallel=True)
        def search(X, k):
            n = X.shape[0]
            out = np.empty((n, k), dtype=np.int64)
            for i in numba.prange(n):
                d = np.empty(n)
                for j in range(n):
                    d[j] = np.inf if j == i else distance(X[i], X[j])
                out[i] = np.argsort(d)[:k]
            return out

        _MASKED_SEARCHES[metric] = search
    return _MASKED_SEARCHES[metric]


def neighbor_indices(
    X: np.ndarray, metric: str, padding_aware: bool = False, k: int = N_NEIGHBORS
) -> np.ndarray:
    """Each point's k nearest other points, nearest first, shape (n, k).

    Plain metrics use the pipeline's own search (nearest_neighbor_indices), so hits match
    calculate_validation_measures() exactly. Padding-aware ones compare every pair.
    """
    if padding_aware:
        X = np.ascontiguousarray(X, dtype=np.float64)
        return _masked_search(metric)(X, k)
    return nearest_neighbor_indices(X, k, metric=metric)


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


def hit_profiles(
    indices: np.ndarray, validation_labels: pd.DataFrame, sets: list[str], ks
) -> pd.DataFrame:
    """Neighbourhood-hit profile of each label set from one neighbour graph (k_max wide),
    as calculate_neighborhood_hit_profile() builds it for the latent and reduced spaces."""
    frames = []
    for column in sets:
        if column not in validation_labels:
            continue
        labels = validation_labels[column].to_numpy()
        if pd.isnull(labels).all():
            continue
        frames.append(
            neighborhood_hit_profile(indices, labels, ks).assign(validation_set=column)
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# One rebuilt space
# --------------------------------------------------------------------------------------


# What a rebuilt space was reduced from: an encoder's latent, the padded input (what plain
# UMAP, PCA and the random projection see) or the input under the padding-aware metric
# (what Masked UMAP sees).
PROBE_KINDS = ("latent", "raw_padded", "raw_masked")


def length_ridge(analysis, targets: list[str], n_folds: int = 5) -> pd.DataFrame:
    """Ridge R² from sequence length alone to each target, on the sequences and folds
    calculate_probe_measures() uses: what knowing the length is worth to a linear probe."""
    encodings = analysis.validation_encodings
    labels = analysis.validation_labels
    length = pd.to_numeric(encodings["sequence_length"], errors="coerce").to_numpy(float)
    seed = analysis.random_seed if analysis.random_seed is not None else 42
    rows = []
    for target in targets:
        if target == "sequence_length" or target not in encodings:
            continue
        y = pd.to_numeric(encodings[target], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(length)
        if labels is not None and target in labels:
            keep &= labels[target].to_numpy() != -1
        rows.append(
            {
                "validation_set": target,
                "ridge_r2": ridge_probe_r2(
                    length[keep, None], y[keep], n_folds=n_folds, random_state=seed
                ),
                "n": int(keep.sum()),
                "dims": 1,
            }
        )
    return pd.DataFrame(rows)


def probe_space(
    analysis,
    metric: str,
    kind: str,
    targets: list[str],
    full: bool = True,
    ks=PROFILE_KS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All probes of one rebuilt space, as long tables: neighbourhood hits (one row per
    validation set), ridge R² (one per target) and hit profiles (one per target and k).

    `kind` is one of PROBE_KINDS. The reduced space (2-D, 5-D or as wide as the latent,
    whatever the reducer produced) is always probed. With `full`, so is the representation
    it was reduced from: the encoder's latent, or for "raw_padded" the raw input both
    padding-aware and plain. Spaces that share that representation with another (the same
    encoder under another reducer, every no-encoder baseline) pass full=False and leave it
    to one of them.

    Output columns: `representation` is "latent", "raw" (the input, padding-aware),
    "raw_padded" (the input as it is, padding included) or, for ridge only, "length"
    (sequence length alone, with the raw input); `space` is "full" or "reduced";
    `dims` is the space's width; `metric` is the distance the neighbours were found with.
    Profiles cover `targets` only, at the sizes in `ks` below the number of sequences.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"kind must be one of {PROBE_KINDS}, got {kind!r}")
    labels = analysis.validation_labels
    reduced = np.asarray(analysis.reduced_embeddings)
    metrics = (metric, "euclidean") if metric != "euclidean" else (metric,)
    ks = [k for k in ks if k < len(reduced)]
    hits, ridges, profiles = [], [], []

    def add_hits(representation, space, metric_name, X, padding_aware=False):
        indices = neighbor_indices(
            X, metric_name, padding_aware=padding_aware, k=max(ks)
        )
        # Neighbours come nearest first, so the five nearest are a prefix of the k_max.
        frame = neighborhood_hits(indices[:, :N_NEIGHBORS], labels)
        if padding_aware or representation != "latent" and space == "full":
            # No pipeline method for the raw input; the same helpers on the same graph.
            add_profile(
                representation,
                space,
                metric_name,
                hit_profiles(indices, labels, targets, ks),
            )
        hits.append(
            frame.assign(
                representation=representation,
                space=space,
                dims=X.shape[1],
                metric=metric_name,
            )
        )

    def add_profile(representation, space, metric_name, frame):
        profiles.append(
            frame[["validation_set", "k", "neigh_hit_adj"]].assign(
                representation=representation, space=space, metric=metric_name
            )
        )

    def add_pipeline_profile(representation, space, probed, metric_name):
        # `probed` is calculate_neighborhood_hit_profile()'s name for the representation.
        frame = analysis.calculate_neighborhood_hit_profile(
            space=probed, ks=tuple(ks), metric=metric_name, validation_sets=targets
        )
        add_profile(representation, space, metric_name, frame)

    def add_ridge(representation, space, probed):
        # `probed` is calculate_probe_measures()'s name for the same representation.
        frame = analysis.calculate_probe_measures(space=probed, targets=targets)
        ridges.append(
            frame.drop(columns="space").assign(
                representation=representation, space=space
            )
        )

    # The reduced space the pipeline scores, with its Euclidean neighbours.
    mapped = {"latent": "latent", "raw_padded": "raw_padded", "raw_masked": "raw"}[kind]
    add_hits(mapped, "reduced", "euclidean", reduced)
    add_pipeline_profile(mapped, "reduced", "reduced", "euclidean")
    add_ridge(mapped, "reduced", "reduced")

    if full and kind == "latent":
        latent = np.asarray(analysis.embeddings)
        for name in metrics:
            add_hits("latent", "full", name, latent)
            add_pipeline_profile("latent", "full", "latent", name)
        add_ridge("latent", "full", "latent")
    elif full and kind == "raw_padded":
        padded = flat_padded(analysis.processed_sequence_data)
        for name in metrics:
            add_hits("raw", "full", name, padded, padding_aware=True)
            add_hits("raw_padded", "full", name, padded)
        add_ridge("raw", "full", "raw")
        ridges.append(
            length_ridge(analysis, targets).assign(
                representation="length", space="full"
            )
        )

    return (
        pd.concat(hits, ignore_index=True),
        pd.concat(ridges, ignore_index=True),
        pd.concat(profiles, ignore_index=True),
    )
