"""Padding-aware distances between flattened, padded sequences.

`processed_sequence_data` pads every sequence to the longest one with the sentinel -999.0.
Flattened and compared with a plain metric, two sequences that differ in length by a single
step are already ~999 * sqrt(n_features) apart, which drowns out every behavioural
difference: a UMAP of the raw sequences then orders them by length before anything else.

These metrics skip every element that is padding in either vector, so two sequences are
compared over the steps both of them have. All sequences of a sequence type start at the
same point (e.g. `pacman_attack` 20 steps before the pill), so that is the shared prefix,
i.e. the shorter sequence against the start of the longer one.

They are numba-compiled so UMAP can take them as `metric=` directly. Neither is a true
metric (a short sequence can be close to two long ones whose tails differ), which UMAP's
nearest-neighbour graph tolerates.
"""

import numba
import numpy as np

PADDING_VALUE = -999.0

# Returned when two vectors share no unpadded element. Cannot happen for sequences that are
# padded at the end only (every sequence has a first step); kept finite for UMAP's sake.
_NO_OVERLAP_EUCLIDEAN = 1e6
_NO_OVERLAP_COSINE = 2.0


@numba.njit(fastmath=True)
def masked_euclidean(x, y):
    """Root-mean-square difference over the elements unpadded in both `x` and `y`.

    Averaged rather than summed, so the distance does not grow with how many steps two
    sequences share - a sum would make long pairs far apart for being long, which is the
    length dependence this metric exists to remove.
    """
    total = 0.0
    count = 0
    for i in range(x.shape[0]):
        if x[i] != PADDING_VALUE and y[i] != PADDING_VALUE:
            diff = x[i] - y[i]
            total += diff * diff
            count += 1
    if count == 0:
        return _NO_OVERLAP_EUCLIDEAN
    return np.sqrt(total / count)


@numba.njit(fastmath=True)
def masked_cosine(x, y):
    """Cosine distance (1 - cosine similarity) over the elements unpadded in both."""
    dot = 0.0
    norm_x = 0.0
    norm_y = 0.0
    count = 0
    for i in range(x.shape[0]):
        if x[i] != PADDING_VALUE and y[i] != PADDING_VALUE:
            dot += x[i] * y[i]
            norm_x += x[i] * x[i]
            norm_y += y[i] * y[i]
            count += 1
    if count == 0:
        return _NO_OVERLAP_COSINE
    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    if norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    return 1.0 - dot / np.sqrt(norm_x * norm_y)


# The padding-aware counterpart of each UMAP metric name that has one.
MASKED_METRICS = {
    "euclidean": masked_euclidean,
    "cosine": masked_cosine,
}
