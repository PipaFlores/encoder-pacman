"""The padding-aware metrics behind embedder="Masked UMAP" (src/utils/masked_distance.py).

Plain vectors, no dataset: these pin down what the metrics compare, which the integration
tests in test_pattern_analysis_fit.py only exercise end to end.
"""

import numpy as np
import pytest
from umap import UMAP

from src.utils import MASKED_METRICS, masked_cosine, masked_euclidean

PAD = -999.0


def padded(values, length):
    """A flattened sequence padded with the sentinel up to `length` elements."""
    out = np.full(length, PAD)
    out[: len(values)] = values
    return out


def test_padding_never_enters_the_distance():
    """Two sequences identical over the steps both have are at distance 0, however much
    longer one of them is - whereas the plain flattened distance is dominated by the
    -999s (the sequence-length leak of the UMAP baseline)."""
    short = padded([0.5, -1.0, 2.0], 10)
    long = padded([0.5, -1.0, 2.0, 1.0, 1.0, 1.0, 3.0], 10)

    assert masked_euclidean(short, long) == pytest.approx(0.0)
    assert masked_cosine(short, long) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(short - long) > 1000


def test_masked_euclidean_is_a_mean_not_a_sum():
    """The same per-element difference gives the same distance at any overlap, so long
    pairs are not pushed apart just for sharing more steps."""
    few = masked_euclidean(padded(np.zeros(3), 20), padded(np.ones(3), 20))
    many = masked_euclidean(padded(np.zeros(15), 20), padded(np.ones(15), 20))

    assert few == pytest.approx(1.0)
    assert many == pytest.approx(1.0)


def test_unpadded_vectors_match_the_plain_metrics():
    """Without padding the masked metrics reduce to RMS difference and cosine distance."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=12), rng.normal(size=12)

    assert masked_euclidean(x, y) == pytest.approx(np.sqrt(np.mean((x - y) ** 2)))
    cosine = 1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
    assert masked_cosine(x, y) == pytest.approx(cosine)


def test_umap_accepts_the_masked_metrics():
    """UMAP takes them as `metric=` (they are numba-compiled) and embeds padded input."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 10, size=40)
    X = np.stack([padded(rng.normal(size=n), 10) for n in lengths])

    for metric in MASKED_METRICS.values():
        embedding = UMAP(n_neighbors=5, metric=metric, random_state=0).fit_transform(X)
        assert embedding.shape == (40, 2)
        assert np.isfinite(embedding).all()
