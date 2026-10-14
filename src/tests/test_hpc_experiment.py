"""Unit tests for the hpc/ configuration layer (RunConfig, build_reducer, expand).

Deliberately data-free and fast, unlike the rest of src/tests: everything here is about
the vocabulary shared between experiment YAML files and train_model.py's flags, which is
exactly the part that fails silently. A key that exists in RunConfig but never reaches
PatternAnalysis produces a sweep whose points are all the same run.
"""

import pytest

from hpc.experiment import RunConfig, build_reducer, expand


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


def test_initialized_model_is_its_own_run():
    """The untrained control must not collide with its trained counterpart in results.csv,
    the run directories (named by hash) or wandb (named by label)."""
    trained = RunConfig(embedder="Transformer", latent_space=128)
    control = RunConfig.from_dict({"embedder": "Transformer", "latent_space": 128, "test_initialized_model": True})

    assert control.config_hash() != trained.config_hash()
    assert control.label() == trained.label() + "_init"


def test_initialized_model_grid_has_one_no_embedder_run():
    """Gridding the flag against `embedder: [..., none]` keeps a single no-embedder run:
    there are no weights to leave untrained, so both points are the same configuration."""
    configs = expand({
        "defaults": {"latent_space": 128},
        "grid": {"embedder": ["MLP", "none"], "test_initialized_model": [False, True]},
    })

    assert [(c.embedder, c.test_initialized_model) for c in configs] == [
        ("MLP", False), ("MLP", True), (None, False),
    ]


def test_umap_embedders_are_configurations_without_weights():
    """"UMAP" and "Masked UMAP" are no-model baselines like `none`: no untrained control
    for them, distinct hashes, and a label without the space."""
    configs = expand({
        "defaults": {"latent_space": 128},
        "grid": {"embedder": ["UMAP", "Masked UMAP"], "test_initialized_model": [False, True]},
    })

    assert [(c.embedder, c.test_initialized_model) for c in configs] == [
        ("UMAP", False), ("Masked UMAP", False),
    ]
    assert configs[0].config_hash() != configs[1].config_hash()
    assert "_MaskedUMAP_" in configs[1].label()


def test_unknown_configuration_key_is_named():
    with pytest.raises(ValueError, match="kl_weight"):
        RunConfig.from_dict({"kl_weight": 0.002})


def test_resample_stride_is_a_configuration_key():
    config = RunConfig.from_dict({"embedder": "MLP", "resample_stride": 2})

    assert config.resample_stride == 2
    assert config.config_hash() != RunConfig(embedder="MLP").config_hash()


def test_resample_stride_distinguishes_labels_and_leaves_default_ones_alone():
    """A stride sweep must not label all of its points identically; stride 1 keeps the old label."""
    plain = RunConfig(embedder="MLP", latent_space=128)

    assert "stride" not in plain.label()
    assert RunConfig(embedder="MLP", latent_space=128, resample_stride=4).label() == plain.label() + "_stride4"


def test_default_resample_stride_leaves_the_config_hash_unchanged():
    """Run directories are named by hash: configurations from before the key existed keep theirs."""
    import hashlib
    import json

    from hpc.experiment import _HASH_EXCLUDE

    config = RunConfig(embedder="MLP")
    payload = {
        k: v for k, v in sorted(config.to_dict().items())
        if k not in _HASH_EXCLUDE and k != "resample_stride"
    }
    before_the_key = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]

    assert config.config_hash() == before_the_key


def test_resampling_keys_reach_pattern_analysis(monkeypatch):
    """The failure this module exists for: a key in RunConfig that never reaches the pipeline
    makes every point of a sweep the same run. PatternAnalysis is swapped for a recorder, so
    nothing is loaded."""
    import sys
    import types

    from hpc.experiment import build_analysis

    received = {}
    recorder = types.ModuleType("src.analysis")
    recorder.PatternAnalysis = lambda **kwargs: received.update(kwargs)
    monkeypatch.setitem(sys.modules, "src.analysis", recorder)

    build_analysis(RunConfig.from_dict(
        {"embedder": "MLP", "reducer": "none", "speed_correction": True, "resample_stride": 4}
    ))

    assert received["speed_correction"] is True
    assert received["resample_stride"] == 4


def test_train_model_flag_matches_the_configuration_key(monkeypatch):
    import sys

    monkeypatch.syspath_prepend("hpc")
    import train_model

    monkeypatch.setattr(sys, "argv", ["train_model.py", "--resample-stride", "2"])
    assert train_model.parse_args().resample_stride == 2

    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    assert train_model.parse_args().resample_stride == RunConfig().resample_stride == 1


def test_train_model_defaults_match_run_config(monkeypatch):
    """train_model.py with no flags must run the same configuration as an experiment file that
    sets nothing: both read RunConfig, so a flag default that drifts from it silently splits the
    two paths (speed correction, normalization, ghost-distance sorting and the UMAP metric once
    did)."""
    import sys

    monkeypatch.syspath_prepend("hpc")
    import train_model

    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    from_cli = train_model.config_from_args(train_model.parse_args()).to_dict()

    assert from_cli == train_model.RunConfig().to_dict()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (True, True)),
        (["--no-speed-correction"], (False, True)),
        (["--speed-correction"], (True, True)),
        (["--no-sort-ghost-distances"], (True, False)),
        (["--sort-ghost-distances"], (True, True)),  # the spelling the train_*.sh scripts use
    ],
)
def test_on_by_default_flags_accept_both_spellings(monkeypatch, flags, expected):
    import sys

    monkeypatch.syspath_prepend("hpc")
    import train_model

    monkeypatch.setattr(sys, "argv", ["train_model.py", *flags])
    config = train_model.config_from_args(train_model.parse_args())

    assert (config.speed_correction, config.sort_ghost_distances) == expected


def test_runs_without_speed_correction_get_their_own_label():
    """A speed_correction: [true, false] sweep must not give both points the same label (the
    results.csv label column and the wandb run name). The suffix marks the exception, so runs
    at the default keep their labels."""
    corrected = RunConfig(embedder="LSTM", latent_space=128)
    uncorrected = RunConfig(embedder="LSTM", latent_space=128, speed_correction=False)

    assert "speedcorr" not in corrected.label()
    assert uncorrected.label() == corrected.label() + "_nospeedcorr"


def test_grid_bundles_pair_each_embedder_with_its_own_latent_size():
    """A bundle axis sets several keys together, so each embedder keeps its own latent size
    while every other axis is still crossed with it."""
    configs = expand({
        "grid": {
            "model": [
                {"embedder": "LSTM", "latent_space": 256},
                {"embedder": "MLP", "latent_space": 64},
                {"embedder": "none"},
            ],
            "resample_stride": [1, 2],
        },
    })

    assert [(c.embedder, c.latent_space, c.resample_stride) for c in configs] == [
        ("LSTM", 256, 1), ("LSTM", 256, 2),
        ("MLP", 64, 1), ("MLP", 64, 2),
        (None, RunConfig().latent_space, 1), (None, RunConfig().latent_space, 2),
    ]


def test_grid_bundles_combine_with_defaults_and_exclude():
    configs = expand({
        "defaults": {"latent_space": 32, "n_epochs": 5},
        "grid": {"model": [{"embedder": "LSTM", "latent_space": 256}, {"embedder": "MLP"}]},
        "exclude": [{"embedder": "MLP"}],
    })

    assert [(c.embedder, c.latent_space, c.n_epochs) for c in configs] == [("LSTM", 256, 5)]


def test_grid_axis_cannot_mix_bundles_and_plain_values():
    with pytest.raises(ValueError, match="mixes bundles"):
        expand({"grid": {"model": [{"embedder": "LSTM"}, "MLP"]}})


def test_grid_cannot_set_a_key_on_two_axes():
    with pytest.raises(ValueError, match="'latent_space' on more than one axis"):
        expand({"grid": {
            "model": [{"embedder": "LSTM", "latent_space": 256}],
            "latent_space": [64],
        }})


def test_grid_bundle_keys_are_checked_like_any_other():
    with pytest.raises(ValueError, match="latent_size"):
        expand({"grid": {"model": [{"embedder": "LSTM", "latent_size": 256}]}})
