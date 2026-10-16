"""Unit tests for the probe helpers in src/utils/utils.py (synthetic data, no dataset)."""

import numpy as np
import pytest

from src.utils.utils import (
    hold_last_step_padding,
    nearest_neighbor_indices,
    neighborhood_hit,
    neighborhood_hit_chance,
    neighborhood_hit_profile,
    ridge_probe_r2,
)

PAD = -999.0


def test_hold_last_step_padding_repeats_last_valid_step():
    data = np.full((3, 4, 2), PAD)
    data[0, :2] = [[1, 2], [3, 4]]          # length 2
    data[1] = np.arange(8).reshape(4, 2)    # full length, nothing to fill
    # data[2] has no valid step and keeps its padding

    filled = hold_last_step_padding(data, padding_value=PAD)

    np.testing.assert_array_equal(filled[0], [[1, 2], [3, 4], [3, 4], [3, 4]])
    np.testing.assert_array_equal(filled[1], data[1])
    assert (filled[2] == PAD).all()
    assert (data[0, 2:] == PAD).all(), "the input must not be modified"


def test_ridge_probe_r2_reads_a_linear_direction_and_not_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    linear = X @ rng.normal(size=20)
    noise = rng.normal(size=400)

    assert ridge_probe_r2(X, linear + 0.01 * noise) > 0.99
    assert ridge_probe_r2(X, noise) < 0.05


def test_ridge_probe_r2_survives_a_rare_value_in_a_near_constant_column():
    """A column that is near-constant except in one sample: in the folds that do not train
    on it, that sample's standardised value is in the thousands. The clip keeps it from
    wrecking the prediction (without it, R² falls to about 0.6 here)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    y = X @ rng.normal(size=20) + 0.1 * rng.normal(size=400)
    X = np.c_[X, 1e-4 * rng.normal(size=400)]
    X[7, -1] = 1.0

    assert ridge_probe_r2(X, y) > 0.99
    assert ridge_probe_r2(X, y, clip=None) < 0.9


@pytest.fixture
def islands():
    """Four islands of 50 points; the label is the island, with -1 (absent) for one of them."""
    rng = np.random.default_rng(0)
    island = np.repeat(np.arange(4), 50)
    X = 50 * np.eye(4)[island] + rng.normal(size=(200, 4))  # islands ~70 apart, spread 1
    labels = np.where(island == 3, -1, island)
    return X, labels


def test_neighborhood_hit_profile_matches_one_search_per_k(islands):
    """One search at the largest k gives the same hits as a separate search per k."""
    X, labels = islands
    ks = [1, 5, 20, 49, 120]
    profile = neighborhood_hit_profile(nearest_neighbor_indices(X, max(ks)), labels, ks)

    assert list(profile["k"]) == ks
    expected = [neighborhood_hit(X, labels, n_neighbors=k) for k in ks]
    np.testing.assert_allclose(profile["neigh_hit"], expected)
    assert (profile["neigh_hit_chance"] == neighborhood_hit_chance(labels)).all()


def test_neighborhood_hit_profile_tracks_the_size_of_a_group(islands):
    """Flat at 1 while k fits inside an island of 50, then falling, and 0 at k = n - 1."""
    X, labels = islands
    n = len(labels)
    profile = neighborhood_hit_profile(nearest_neighbor_indices(X, n - 1), labels, [5, 49, 100, n - 1])
    adjusted = profile["neigh_hit_adj"].to_numpy()

    np.testing.assert_allclose(adjusted[:2], 1.0)
    assert 0 < adjusted[2] < 1
    assert adjusted[3] == pytest.approx(0.0, abs=1e-12)


def test_neighborhood_hit_profile_rejects_k_beyond_the_search(islands):
    X, labels = islands
    with pytest.raises(ValueError):
        neighborhood_hit_profile(nearest_neighbor_indices(X, 10), labels, [5, 20])
