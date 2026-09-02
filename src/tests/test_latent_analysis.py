import math

import numpy as np

from src.analysis.latent_analysis import (
    participant_progression_table,
    shannon_entropy,
    summarize_cluster_progression,
)


def test_shannon_entropy_is_zero_for_single_repeated_label():
    assert shannon_entropy([2, 2, 2], normalize=True) == 0.0


def test_cluster_progression_separates_repetition_from_exploration():
    repetitive = summarize_cluster_progression([0, 0, 0, 0])
    exploratory = summarize_cluster_progression([0, 1, 2, 3])

    assert repetitive["cluster_entropy_normalized"] == 0.0
    assert repetitive["cluster_repeat_fraction"] == 1.0
    assert np.isclose(exploratory["cluster_entropy_normalized"], 1.0)
    assert exploratory["cluster_transition_rate"] == 1.0


def test_participant_progression_table_returns_one_row_per_participant():
    latent_points = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    participant_ids = np.array([1, 1, 1, 2, 2, 2])
    cluster_labels = np.array([0, 0, 0, 0, 1, 2])

    table = participant_progression_table(
        latent_points,
        participant_ids=participant_ids,
        cluster_labels=cluster_labels,
        min_history=1,
    )

    assert table.shape[0] == 2
    assert set(table["user_id"]) == {1, 2}

    repeated = table.loc[table["user_id"] == 1].iloc[0]
    exploratory = table.loc[table["user_id"] == 2].iloc[0]

    assert repeated["cluster_entropy_normalized"] == 0.0
    assert repeated["cluster_repeat_fraction"] == 1.0
    assert np.isclose(exploratory["cluster_entropy_normalized"], 1.0)
    assert exploratory["latent_path_total_distance"] > repeated["latent_path_total_distance"]
    assert math.isfinite(exploratory["kde_overall_novelty_mean"])
