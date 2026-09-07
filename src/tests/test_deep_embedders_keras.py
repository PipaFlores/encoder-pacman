"""Smoke tests for the aeon/Keras deep clustering embedders (DRNN, DCNN, ResNet).

Exercises each clusterer the way `PatternAnalysis`/hpc/keras_pacman.py do:

- `model.fit(X)` (the aeon/sklearn API) wants channels-first X - shape
  [n_samples, n_channels, series_length], aeon's own array convention - and
  transposes it back to channels-last internally before building/feeding the
  actual Keras graph.
- `model_.layers[1].predict(X)` (reaching directly into the raw Keras
  encoder submodel to pull out embeddings, since aeon's public API only
  exposes cluster labels) bypasses that preprocessing and talks to the
  already-transposed graph directly, so it wants channels-last X - shape
  [n_samples, series_length, n_channels] - the same "repo shape" used
  everywhere else in this codebase (padded_sequence_data, the torch models,
  etc.), with no transpose needed.

Mixing these two up is exactly what caused the DCNN/ResNet input-shape
ValueErrors seen in HPC runs - see the issue tracking that bug. These tests
pin down the correct convention for each call so a regression fails here
instead of only surfacing hours into an HPC job.

Skipped entirely when tensorflow/aeon are not installed, mirroring the
module-load split used on the HPC cluster (see
hpc/smoke_benchmark_autoencoders.sh, which loads python-tensorflow
separately from python-pytorch for exactly these architectures).
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
aeon = pytest.importorskip("aeon")

from aeon.clustering import DummyClusterer  # noqa: E402
from aeon.clustering.deep_learning import (  # noqa: E402
    AEDCNNClusterer,
    AEDRNNClusterer,
    AEResNetClusterer,
)

N_SAMPLES = 8
SERIES_LENGTH = 16
N_CHANNELS = 2
LATENT_DIM = 4


def make_synthetic_series(n=N_SAMPLES, series_length=SERIES_LENGTH, n_channels=N_CHANNELS, seed=0):
    """Channels-last: [n_samples, series_length, n_channels] - repo shape."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, series_length, n_channels)).astype(np.float32)


@pytest.mark.parametrize(
    "clusterer_cls, extra_kwargs, expected_latent_dim",
    [
        (AEDRNNClusterer, {"latent_space_dim": LATENT_DIM}, LATENT_DIM),
        (AEDCNNClusterer, {"latent_space_dim": LATENT_DIM}, LATENT_DIM),
        # ResNet's latent dim is fixed internally, not configurable here.
        (AEResNetClusterer, {}, None),
    ],
    ids=["DRNN", "DCNN", "ResNet"],
)
def test_fit_and_embed_shapes(clusterer_cls, extra_kwargs, expected_latent_dim):
    data = make_synthetic_series()
    model = clusterer_cls(
        estimator=DummyClusterer(),
        n_epochs=1,
        validation_split=0.0,
        verbose=False,
        **extra_kwargs,
    )

    # model.fit() wants channels-first, unlike the channels-last data everywhere else.
    model.fit(np.transpose(data, (0, 2, 1)))

    # model_.layers[1] bypasses aeon's preprocessing and wants channels-last
    # directly - i.e. `data` as-is, no transpose.
    embeddings = model.model_.layers[1].predict(data, verbose=0)

    assert embeddings.ndim == 2
    assert embeddings.shape[0] == N_SAMPLES
    if expected_latent_dim is not None:
        assert embeddings.shape[1] == expected_latent_dim

    # Calling model_ directly (e.g. for a reconstruction-error check) is the
    # other place this matters: it also wants channels-last, matching `data`.
    recon = model.model_(data, training=False).numpy()
    assert recon.shape == data.shape
