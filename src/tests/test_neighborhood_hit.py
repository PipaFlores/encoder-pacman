"""Unit tests for the neighborhood hit and its chance adjustment (src/utils/utils.py).

Synthetic data only, so these run in about a second and need no `data/` folder.
"""

import numpy as np

from src.utils.utils import (
    adjusted_neighborhood_hit,
    chance_adjusted_neighborhood_hit,
    neighborhood_hit,
    neighborhood_hit_chance,
)


def separated_blobs(n_per_class, n_classes, rng):
    """Tight, far-apart 2-D blobs - every point's nearest neighbors share its class."""
    centers = np.arange(n_classes)[:, None] * 100.0 * np.array([[1.0, 0.0]])
    points = np.concatenate([c + rng.normal(0, 0.1, (n_per_class, 2)) for c in centers])
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return points, labels


def test_chance_matches_the_mean_hit_over_label_permutations():
    """The analytic chance level is the expected hit when labels are shuffled over a fixed
    neighbor graph, including absent (-1) points that are neighbors but never queried."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(60, 2))
    labels = np.array([0] * 30 + [1] * 12 + [2] * 3 + [-1] * 15)

    permuted_hits = [neighborhood_hit(points, rng.permutation(labels)) for _ in range(3000)]

    assert np.isclose(np.mean(permuted_hits), neighborhood_hit_chance(labels), atol=0.01)


def test_chance_is_simpson_index_for_large_samples():
    labels = np.repeat(np.arange(10), 1000)  # ten balanced classes

    assert np.isclose(neighborhood_hit_chance(labels), 0.1, atol=1e-3)


def test_chance_of_a_presence_label_is_its_prevalence():
    """With absences excluded, a presence/absence behavlet has a single query class, and a
    random neighbor of a present point is present with probability (n_present - 1) / (n - 1)."""
    labels = np.array([1] * 20 + [-1] * 80)

    assert np.isclose(neighborhood_hit_chance(labels), 19 / 99)


def test_including_null_instances_treats_absence_as_a_class():
    labels = np.array([1] * 20 + [-1] * 80)
    expected = 0.2 * 19 / 99 + 0.8 * 79 / 99

    assert np.isclose(neighborhood_hit_chance(labels, no_null_instances=False), expected)


def test_adjusted_hit_is_one_for_perfectly_separated_labels():
    points, labels = separated_blobs(n_per_class=50, n_classes=4, rng=np.random.default_rng(1))

    assert np.isclose(adjusted_neighborhood_hit(points, labels), 1.0)


def test_adjusted_hit_is_near_zero_without_structure_whatever_the_class_balance():
    """Raw hits differ with class balance even when the embedding knows nothing about the
    labels; the adjusted hit puts both at chance (0)."""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(4000, 2))
    balanced = rng.integers(0, 10, size=4000)
    dominant = np.where(rng.random(4000) < 0.9, 0, 1)

    assert neighborhood_hit(points, dominant) > 0.75
    assert neighborhood_hit(points, balanced) < 0.15
    assert abs(adjusted_neighborhood_hit(points, balanced)) < 0.03
    assert abs(adjusted_neighborhood_hit(points, dominant)) < 0.03


def test_adjusted_hit_is_undefined_when_there_is_nothing_to_separate():
    points = np.random.default_rng(3).normal(size=(30, 2))

    assert np.isnan(adjusted_neighborhood_hit(points, np.zeros(30)))  # one class only
    assert np.isnan(neighborhood_hit_chance(np.full(30, -1)))  # nothing to query
    assert np.isnan(chance_adjusted_neighborhood_hit(0.5, np.nan))
