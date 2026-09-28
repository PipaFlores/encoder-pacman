"""Unit tests for the hpc/ configuration layer (RunConfig, build_reducer).

Deliberately data-free and fast, unlike the rest of src/tests: everything here is about
the vocabulary shared between experiment YAML files and train_model.py's flags, which is
exactly the part that fails silently. A key that exists in RunConfig but never reaches
PatternAnalysis produces a sweep whose points are all the same run.
"""

import pytest

from hpc.experiment import RunConfig, build_reducer


def test_pca_reducer_is_built_from_config():
    """`reducer: pca` gives a PCA with the configured width and seed."""
    PCA = pytest.importorskip("sklearn.decomposition").PCA

    reducer = build_reducer(RunConfig(reducer="pca", reducer_components=3, seed=7))

    assert isinstance(reducer, PCA)
    assert reducer.n_components == 3
    assert reducer.random_state == 7


def test_umap_reducer_is_built_from_config():
    UMAP = pytest.importorskip("umap").UMAP

    reducer = build_reducer(
        RunConfig(reducer="umap", reducer_components=2, umap_neighbors=10, umap_min_dist=0.3)
    )

    assert isinstance(reducer, UMAP)
    assert reducer.n_components == 2
    assert reducer.n_neighbors == 10
    assert reducer.min_dist == 0.3


def test_reducer_none_skips_reduction():
    """__post_init__ turns the YAML/CLI spelling "none" into an actual None."""
    assert build_reducer(RunConfig(reducer="none")) is None
    assert build_reducer(RunConfig(reducer=None)) is None


def test_unknown_reducer_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        build_reducer(RunConfig(reducer="tsne"))


def test_kld_weight_is_a_configuration_key():
    """kld_weight is sweepable from an experiment file, not just a constructor default."""
    config = RunConfig.from_dict({"embedder": "VAE", "kld_weight": 0.002})

    assert config.kld_weight == 0.002
    assert config.to_dict()["kld_weight"] == 0.002
    # Two points of a sweep over it must be distinguishable in results.csv
    assert config.config_hash() != RunConfig(embedder="VAE").config_hash()


def test_kld_weight_distinguishes_sweep_labels():
    """A kld_weight sweep must not label all of its points identically.

    `label` is the results.csv column and the wandb run name (run directories are named by
    index and hash, so they stay unique regardless). Same rule as the checkpoint stem: VAE
    family only, non-default values only, so labels from earlier runs are unchanged.
    """
    default = RunConfig(embedder="VAE", latent_space=64)
    assert default.label() == "first_5_seconds_Pacman_VAE_h64"

    swept = RunConfig(embedder="VAE", latent_space=64, kld_weight=0.0016)
    assert swept.label() == "first_5_seconds_Pacman_VAE_h64_kld0.0016"

    # 0.0 is a meaningful point of the sweep (a plain autoencoder), not "unset"
    assert RunConfig(embedder="VAE", kld_weight=0.0).label().endswith("_kld0")

    # The embedders without a KL term never carry it, whatever the value
    assert "kld" not in RunConfig(embedder="LSTM", kld_weight=0.0016).label()
    assert "kld" not in RunConfig(embedder=None, kld_weight=0.0016).label()


def test_unknown_configuration_key_is_named():
    with pytest.raises(ValueError, match="kl_weight"):
        RunConfig.from_dict({"kl_weight": 0.002})
