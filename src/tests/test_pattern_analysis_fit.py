"""Integration test for PatternAnalysis.fit() end-to-end, on real data.

Exercises the full pipeline (build_data -> train the embedder -> embed ->
reduce -> cluster) across the main variations of the analysis pipeline:

- the standard flow (deep embedder -> UMAP down to 2 dimensions -> HDBSCAN) for
  the two main sequence types ("first_5_seconds" and "pacman_attack");
- every supported embedder (LSTM, MLP, Transformer, VAE, TimeVAE);
- the no-deep-embedder configuration, where the reducer embeds the flat raw
  data itself instead of a trained model's latent space;
- "Behavlets" validation on top of the clustering.

Uses the real "data" folder (like test_data_reader.py /
test_pattern_analysis_cache.py) but keeps `max_samples` small and `max_epochs=2`
so each case is a matter of seconds rather than a full training run. This is
still a slow-ish integration test, not a unit test: the sequence slicing in
build_data() runs over the whole dataset regardless of max_samples, so most of
the runtime is data loading - paid once per data configuration thanks to the
module-scoped `cache_folder` fixture below.

The embedder variations all run on "pacman_attack" alone: it's the more complex
of the two slicing schemes (variable-length sequences, hence padding, hence the
masked-pooling encoder variants), so it exercises more of each model than
"first_5_seconds" would, at half the runtime of covering both.

`validation_method` is left at None everywhere except
test_fit_with_behavlets_validation, so the rest of the cases stay about fit()'s
own pipeline rather than the cost and quirks of behavlet encoding on top of it.
"""

import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hdbscan import HDBSCAN  # noqa: E402

from src.analysis.pattern_analysis import PatternAnalysis  # noqa: E402
from src.datahandlers import PacmanDataReader  # noqa: E402

MAX_SAMPLES = 30
MAX_EPOCHS = 2
# The small feature set used for fast analysis (4 ghost distances + score). "all_features"
# expands to 267 columns, which makes these cases heavy enough to exhaust memory partway
# through the module, and puts TimeVAE out of reach entirely - its residual decoder ends in
# Linear(feat_dim * L_final, seq_len * feat_dim) (see ResidualConnection in src/models/TVAE.py),
# a 17GB weight at 267 features and pacman_attack's seq_len of 247.
FEATURE_SET = "Experimental2"
NORMALIZATION = "global"
REDUCED_DIM = 2  # what the reducer (UMAP) takes the embeddings down to, for clustering/visualization

# Latent size for the embedder variations: > 2 so the reducer actually runs (the
# standard flow), and a multiple of 8 because TSTransformerEncoder uses d_model=latent
# with n_heads=8, which torch requires to divide evenly.
LATENT_DIM = 128
# The end-to-end test uses a realistic latent size instead, as in a real run.
CANONICAL_LATENT_DIM = 256
# The more complex of the two sequence types - see module docstring.
EMBEDDER_SEQUENCE_TYPE = "pacman_attack"

TORCH_EMBEDDERS = ["MLP", "Transformer", "VAE", "TimeVAE"]  # "LSTM" is covered end-to-end below


@pytest.fixture(scope="module")
def reader():
    """`PacmanDataReader` is a singleton per process (see test_data_reader.py),
    so loading the real gamestate data happens once regardless of how many
    times/where it's requested - this fixture just names that instance."""
    return PacmanDataReader(data_folder="data")


@pytest.fixture(scope="module")
def cache_folder(tmp_path_factory):
    """One build_data() cache shared by every test in this module.

    That cache is keyed by a fingerprint of the data configuration (sequence_type,
    feature_set, max_samples, source dataset, ... - see _compute_data_fingerprint), so
    sharing it across tests is safe: each distinct config still gets its own entry. It
    matters because reader.make_data() re-slices the *whole* dataset regardless of
    max_samples - doing that once per test instead of once per config is slow enough to
    exhaust memory partway through the module. Caching behaviour itself is covered
    separately, on its own tmp_path, in test_pattern_analysis_cache.py.
    """
    return tmp_path_factory.mktemp("pattern_analysis_data_cache")


@pytest.fixture(autouse=True)
def release_memory_between_tests():
    """Drop each case's model and data copy before the next one starts.

    Every case holds a trained model plus its own copy of the sliced data. This module runs
    close enough to the memory ceiling on a 16GB machine that build_data()'s full-dataset
    padding array (~190MB for "first_5_seconds", on top of torch) has failed to allocate
    partway through a run.
    """
    yield
    gc.collect()


def make_pattern_analysis(
    reader,
    cache_folder,
    tmp_path,
    sequence_type=EMBEDDER_SEQUENCE_TYPE,
    embedder="LSTM",
    latent_dimension=LATENT_DIM,
    validation_method=None,
):
    return PatternAnalysis(
        reader=reader,
        cache_folder=str(cache_folder),
        # Per-test, unlike the data cache: a shared one would let a test load the model a
        # previous test trained (see _check_model_training_status) instead of training its own.
        hpc_folder=str(tmp_path / "hpc"),
        embedder=embedder,
        clusterer=HDBSCAN(min_cluster_size=5),
        sequence_type=sequence_type,
        feature_set=FEATURE_SET,
        max_samples=MAX_SAMPLES,
        max_epochs=MAX_EPOCHS,
        # What every HPC training job runs with. Unnormalized, raw scores run into the
        # thousands right next to the -999 padding value, which diverges to NaN in TimeVAE.
        normalization=NORMALIZATION,
        latent_dimension=latent_dimension,
        batch_size=8,
        validation_method=validation_method,
        verbose=False,
        random_seed=0,
    )


def assert_pipeline_outputs(pa, latent_dimension):
    """The outputs every deep-embedder run of the standard flow must produce."""
    assert len(pa.raw_sequence_data) == MAX_SAMPLES
    assert pa.processed_sequence_data.shape[0] == MAX_SAMPLES

    assert pa.embeddings.shape == (MAX_SAMPLES, latent_dimension)
    assert np.isfinite(pa.embeddings).all()

    # latent_dim > 2, so the reducer takes the embeddings down to REDUCED_DIM for clustering
    assert pa.reduced_embeddings is not pa.embeddings
    assert pa.reduced_embeddings.shape == (MAX_SAMPLES, REDUCED_DIM)
    assert np.isfinite(pa.reduced_embeddings).all()

    assert pa.labels.shape == (MAX_SAMPLES,)


@pytest.mark.parametrize("sequence_type", ["first_5_seconds", "pacman_attack"])
def test_fit_runs_end_to_end(reader, cache_folder, tmp_path, sequence_type):
    """The standard flow, at a realistic latent size: LSTM -> UMAP -> HDBSCAN."""
    pa = make_pattern_analysis(
        reader, cache_folder, tmp_path, sequence_type=sequence_type,
        embedder="LSTM", latent_dimension=CANONICAL_LATENT_DIM,
    )

    pa.fit()

    assert_pipeline_outputs(pa, CANONICAL_LATENT_DIM)

    # The embedder actually trained for the requested number of epochs
    assert len(pa.embedder.loss_history) == MAX_EPOCHS
    assert all(np.isfinite(pa.embedder.loss_history))


@pytest.mark.parametrize("embedder", TORCH_EMBEDDERS)
def test_fit_with_torch_embedders(reader, cache_folder, tmp_path, embedder):
    pa = make_pattern_analysis(reader, cache_folder, tmp_path, embedder=embedder)

    pa.fit()

    assert_pipeline_outputs(pa, LATENT_DIM)
    assert len(pa.embedder.loss_history) == MAX_EPOCHS
    assert all(np.isfinite(pa.embedder.loss_history))


def test_fit_without_deep_embedder(reader, cache_folder, tmp_path):
    """embedder=None skips training entirely: the reducer embeds the flattened raw
    sequences straight down to 2 dimensions, and those are clustered as-is."""
    pa = make_pattern_analysis(reader, cache_folder, tmp_path, embedder=None)

    pa.fit()

    assert pa.embedder is None
    assert len(pa.raw_sequence_data) == MAX_SAMPLES

    # The reducer is the embedder here, so it already outputs REDUCED_DIM and the
    # separate reduction step is skipped.
    assert pa.embeddings.shape == (MAX_SAMPLES, REDUCED_DIM)
    assert np.isfinite(pa.embeddings).all()
    assert pa.reduced_embeddings is pa.embeddings

    assert pa.labels.shape == (MAX_SAMPLES,)


def test_fit_with_behavlets_validation(reader, cache_folder, tmp_path):
    """Validation runs as the last pipeline step, scoring the clustering against
    behavlet/metadata labels."""
    pa = make_pattern_analysis(
        reader, cache_folder, tmp_path, embedder="MLP", validation_method="Behavlets"
    )

    pa.fit()

    assert_pipeline_outputs(pa, LATENT_DIM)

    assert len(pa.validation_encodings) == MAX_SAMPLES
    assert len(pa.validation_labels) == MAX_SAMPLES

    # One row of measures per validation label set, minus any that carried no signal at this
    # sample size and were skipped (see calculate_validation_measures)
    assert 0 < len(pa.validation_measures) <= len(pa.validation_labels.columns)
    assert set(pa.validation_measures.columns) == {"validation_set", "ARI", "AMI", "NMI", "neigh_hit"}
