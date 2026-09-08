"""Integration test for PatternAnalysis.fit() end-to-end, on real data.

Exercises the full pipeline (build_data -> train the embedder -> embed ->
reduce -> cluster) for the two main sequence types ("first_5_seconds" and
"pacman_attack") on the "all_features" feature set. Uses the real "data"
folder (like test_data_reader.py / test_pattern_analysis_cache.py) but keeps
`max_samples` small and `max_epochs=2` so this stays a matter of seconds
rather than a full training run - this is still a slow-ish integration test
(the sequence slicing in build_data() runs over the whole dataset regardless
of max_samples, so most of the runtime is a one-time, session-cached data
load), not a unit test.

`validation_method` is left at None: the default "Behavlets" validation
currently raises `AssertionError: <col> has missing values` for some
Aggression behavlets that are legitimately NaN for sequences without a
pill/ghost-kill event (recode_validation_labels/calculate_validation_measures
don't handle that), independent of sample size - a separate, pre-existing
issue from what's being tested here.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hdbscan import HDBSCAN  # noqa: E402

from src.analysis.pattern_analysis import PatternAnalysis  # noqa: E402
from src.datahandlers import PacmanDataReader  # noqa: E402

MAX_SAMPLES = 30
FEATURE_SET = "all_features"
LATENT_DIM = 2  # <= 2 so the reducer (UMAP) is skipped - keeps this robust to tiny sample sizes


@pytest.fixture(scope="module")
def reader():
    """`PacmanDataReader` is a singleton per process (see test_data_reader.py),
    so loading the real gamestate data happens once regardless of how many
    times/where it's requested - this fixture just names that instance."""
    return PacmanDataReader(data_folder="data")


def make_pattern_analysis(reader, tmp_path, sequence_type):
    return PatternAnalysis(
        reader=reader,
        cache_folder=str(tmp_path / "cache"),
        hpc_folder=str(tmp_path / "hpc"),
        embedder="MLP",
        clusterer=HDBSCAN(min_cluster_size=5),
        sequence_type=sequence_type,
        feature_set=FEATURE_SET,
        max_samples=MAX_SAMPLES,
        max_epochs=2,
        latent_dimension=LATENT_DIM,
        batch_size=8,
        validation_method=None,
        verbose=False,
        random_seed=0,
    )


@pytest.mark.parametrize("sequence_type", ["first_5_seconds", "pacman_attack"])
def test_fit_runs_end_to_end(reader, tmp_path, sequence_type):
    pa = make_pattern_analysis(reader, tmp_path, sequence_type)

    pa.fit()

    assert len(pa.raw_sequence_data) == MAX_SAMPLES
    assert pa.processed_sequence_data.shape[0] == MAX_SAMPLES

    assert pa.embeddings.shape == (MAX_SAMPLES, LATENT_DIM)
    assert np.isfinite(pa.embeddings).all()

    # latent_dim <= 2, so reduced_embeddings should be the raw embeddings, untouched by UMAP
    assert pa.reduced_embeddings is pa.embeddings

    assert pa.labels.shape == (MAX_SAMPLES,)

    # The embedder actually trained for the requested number of epochs
    assert len(pa.embedder.loss_history) == 2
    assert all(np.isfinite(pa.embedder.loss_history))
