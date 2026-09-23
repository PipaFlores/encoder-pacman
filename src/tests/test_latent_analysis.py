import math

import numpy as np

from src.analysis.latent_analysis import (
    shannon_entropy,
    summarize_cluster_progression,
    summarize_continuous_progression,
)


def test_shannon_entropy_is_zero_for_single_repeated_label():
    assert shannon_entropy([2, 2, 2], normalize=True) == 0.0


def test_cluster_progression_separates_repetition_from_exploration():
    repetitive = summarize_cluster_progression([0, 0, 0, 0])
    exploratory = summarize_cluster_progression([0, 1, 2, 3])

    assert repetitive["Shannon_entropy_normalized"] == 0.0
    assert repetitive["cluster_repeat_fraction"] == 1.0
    assert np.isclose(exploratory["Shannon_entropy_normalized"], 1.0)
    assert exploratory["cluster_transition_rate"] == 1.0


def test_progression_summaries_separate_repetitive_from_exploratory_participant():
    repetitive_points = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    exploratory_points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    population = np.vstack([repetitive_points, exploratory_points])

    repeated = {
        **summarize_cluster_progression([0, 0, 0]),
        **summarize_continuous_progression(repetitive_points, population, min_history=1),
    }
    exploratory = {
        **summarize_cluster_progression([0, 1, 2]),
        **summarize_continuous_progression(exploratory_points, population, min_history=1),
    }

    assert repeated["Shannon_entropy_normalized"] == 0.0
    assert repeated["cluster_repeat_fraction"] == 1.0
    assert np.isclose(exploratory["Shannon_entropy_normalized"], 1.0)
    assert exploratory["latent_path_total_distance"] > repeated["latent_path_total_distance"]
    assert math.isfinite(exploratory["kde_step_novelty_mean"])
    assert math.isfinite(exploratory["kde_pop_relative_novelty_mean"])
