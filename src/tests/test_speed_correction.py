from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.datahandlers.pacman_data_reader import PacmanDataReader

positions = PacmanDataReader.resampling_positions
SPEEDS = [0.80, 0.90, 1.00]  # level 1, levels 2-4, levels 5-20


def _reader_stub(levels: dict[int, int]):
    """
    The attributes _resample_sequences reads, without loading the dataset: PacmanDataReader
    is a singleton, so a bare instance here would replace the real one for other tests.
    """
    return SimpleNamespace(
        level_df=pd.DataFrame({"level": levels}),
        SPEED_CORRECTION_ANCHORS=PacmanDataReader.SPEED_CORRECTION_ANCHORS,
        SPEED_CORRECTION_INTERPOLATED=PacmanDataReader.SPEED_CORRECTION_INTERPOLATED,
        SPEED_CORRECTION_ACTORS=PacmanDataReader.SPEED_CORRECTION_ACTORS,
        MAX_FRAME_MOVE=PacmanDataReader.MAX_FRAME_MOVE,
        SLOWEST_PACMAN_SPEED=PacmanDataReader.SLOWEST_PACMAN_SPEED,
        pacman_speed=PacmanDataReader.pacman_speed,
        resampling_positions=PacmanDataReader.resampling_positions,
    )


def _moving_sequence(level_id, n_frames, step, **extra_columns):
    """Pac-Man moving right by `step` maze units per frame, one level."""
    return pd.DataFrame(
        {
            "level_id": level_id,
            "Pacman_X": -10.0 + step * np.arange(n_frames),
            "Pacman_Y": -9.5,
            **extra_columns,
        }
    )


def test_pacman_speed_table():
    assert [PacmanDataReader.pacman_speed(level) for level in [1, 2, 4, 5, 20, 21]] == [
        0.80,
        0.90,
        0.90,
        1.00,
        1.00,
        0.90,
    ]


def test_full_speed_keeps_every_frame():
    np.testing.assert_array_equal(positions(100, 1.0), np.arange(100))


def test_level_one_samples_every_1_25_frames():
    t = positions(100, 0.80, anchor="start")
    assert len(t) == 80
    np.testing.assert_allclose(np.diff(t), 1.25)
    assert t[0] == 0


@pytest.mark.parametrize("n_frames", [1, 2, 37, 50, 100, 200])
@pytest.mark.parametrize("anchor", ["start", "end"])
def test_anchored_length_does_not_depend_on_level(n_frames, anchor):
    lengths = {len(positions(n_frames, speed, anchor=anchor)) for speed in SPEEDS}
    assert len(lengths) == 1


def test_end_anchor_keeps_the_last_frame():
    for speed in SPEEDS:
        t = positions(100, speed, anchor="end")
        assert t[-1] == 99
        assert len(t) == 80


def test_unanchored_windows_keep_their_full_distance():
    # event-defined windows are not cut: a faster level keeps more steps
    assert [len(positions(100, speed)) for speed in SPEEDS] == [80, 90, 100]


@pytest.mark.parametrize("anchor", [None, "start", "end"])
def test_positions_stay_inside_the_sequence(anchor):
    for n_frames in range(1, 300):
        for speed in SPEEDS:
            t = positions(n_frames, speed, anchor=anchor)
            assert t.min() >= 0 and t.max() <= n_frames - 1
            assert np.all(np.diff(t) > 0)


def test_every_level_moves_the_same_distance_per_step():
    # Pac-Man's per-frame movement scales with level speed; after correction it must not
    speeds = {1: 0.80, 2: 0.90, 5: 1.00}
    raw = [
        _moving_sequence(level_id, 100, 0.47 * speed)
        for level_id, speed in speeds.items()
    ]
    stub = _reader_stub({1: 1, 2: 2, 5: 5})

    corrected = PacmanDataReader._resample_sequences(stub, raw, raw, "first_5_seconds")

    for sequence in corrected:
        assert len(sequence) == 80
        # same step at every level, and no rhythm: every step identical
        np.testing.assert_allclose(np.diff(sequence["Pacman_X"]), 0.47)


def test_physics_step_alternation_leaves_no_level_signature():
    """
    Each logged frame holds 2 or 3 of Unity's 0.02 s physics steps, so raw positions advance in
    a short/long alternation at every level. After correction every level must move the same,
    constant distance per step: no leftover rhythm that differs between levels.
    """
    raw = []
    for level_id, speed in {1: 0.80, 2: 0.90, 5: 1.00}.items():
        tick = 9.47 * speed * 0.02  # distance per physics step
        steps = np.tile([2 * tick, 3 * tick], 50)
        raw.append(
            pd.DataFrame(
                {
                    "level_id": level_id,
                    "Pacman_X": -10.0 + np.r_[0, np.cumsum(steps)][:100],
                    "Pacman_Y": -9.5,
                }
            )
        )

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1, 2: 2, 5: 5}), raw, raw, "first_5_seconds"
    )

    for sequence in corrected:
        interior = np.diff(sequence["Pacman_X"].to_numpy())[2:-2]
        np.testing.assert_allclose(interior, 2.5 * 9.47 * 0.02, rtol=1e-9)


# At level 1 the samples fall at 0, 1.25, 2.5, 3.75, 5.0, 6.25, ...: the wrap below sits
# between frames 6 and 7, so the sample at 6.25 lands on it.
TUNNEL_X = np.r_[np.linspace(-11.0, -13.5, 7), np.linspace(13.5, 11.5, 5)]


def test_tunnel_wrap_is_not_interpolated():
    x = TUNNEL_X
    raw = [pd.DataFrame({"level_id": 1, "Pacman_X": x, "Pacman_Y": -0.5})]

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1}), raw, raw, "whole_level"
    )

    # no position between the two tunnel ends, i.e. nothing invented in the middle of the maze
    assert not np.any(
        (corrected[0]["Pacman_X"] > -11.0) & (corrected[0]["Pacman_X"] < 11.0)
    )


def test_infinite_distances_are_not_interpolated():
    # the sample at 5.0 has an inf neighbour (frame 6): interpolating would give 0 * inf = nan
    distance = np.r_[np.linspace(20, 15, 6), np.full(14, np.inf)]
    ghost_x = 5.0 - 0.3 * np.arange(
        20
    )  # a ghost moving normally, so only `inf` blocks blending
    raw = [
        _moving_sequence(
            1, 20, 0.38, Ghost1_X=ghost_x, Ghost1_Y=-9.5, Ghost1_distance=distance
        )
    ]

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1}), raw, raw, "whole_level"
    )

    values = corrected[0]["Ghost1_distance"].to_numpy()
    assert not np.isnan(values).any()
    assert values[4] == 15  # the sample at 5.0, taken from frame 5 as recorded
    assert np.isinf(values[5:]).all()


def test_other_columns_take_the_nearest_frame():
    states = np.repeat([1, 2, 3, 4], 25)
    raw = [
        _moving_sequence(1, 100, 0.38, ghost1_state=states, score=np.arange(100) * 10)
    ]

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1}), raw, raw, "first_5_seconds"
    )

    assert set(corrected[0]["ghost1_state"]) <= {1, 2, 3, 4}
    assert set(corrected[0]["score"]) <= set(np.arange(100) * 10)


def test_respawn_is_never_blended_one_axis_at_a_time():
    """
    Regression: a respawn after a death moved Pac-Man from (-0.5, ~10) to the spawn (0, -9.5).
    x changed by only 0.5, so a per-column jump test blended x across the teleport while y took
    a real frame, putting points off the corridor and inside walls. The move is now judged in 2D.
    """
    n_before, n_after = 7, 13
    x = np.r_[np.full(n_before, -0.5), 0.0 + 0.38 * np.arange(n_after)]
    y = np.r_[12.5 - 0.38 * np.arange(n_before), np.full(n_after, -9.5)]
    raw = [pd.DataFrame({"level_id": 1, "Pacman_X": x, "Pacman_Y": y})]

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1}), raw, raw, "whole_level"
    )

    # the level-1 sample at 6.25 falls between frame 6 (before) and frame 7 (after the
    # respawn): it must be frame 6 itself, both coordinates, not a blend of the two
    step = corrected[0].iloc[5]
    assert (step["Pacman_X"], step["Pacman_Y"]) == (x[6], y[6])


def test_interpolation_is_decided_on_raw_values():
    # normalized values are on another scale (here /100, so the wrap is a jump of only 0.27);
    # the wrap must still be detected, from the raw values
    raw = [pd.DataFrame({"level_id": 1, "Pacman_X": TUNNEL_X, "Pacman_Y": -0.5})]
    normalized = [raw[0].assign(Pacman_X=TUNNEL_X / 100)]

    corrected = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1}), raw, normalized, "whole_level"
    )

    assert not np.any(np.abs(corrected[0]["Pacman_X"]) < 0.11)


def test_rejects_multi_level_sequences_and_unknown_types():
    stub = _reader_stub({1: 1, 2: 2})
    mixed = [pd.concat([_moving_sequence(1, 10, 0.38), _moving_sequence(2, 10, 0.43)])]
    with pytest.raises(ValueError, match="single-level"):
        PacmanDataReader._resample_sequences(stub, mixed, mixed, "whole_level")

    single = [_moving_sequence(1, 10, 0.38)]
    with pytest.raises(ValueError, match="no window anchor"):
        PacmanDataReader._resample_sequences(
            stub, single, single, "not_a_sequence_type"
        )


# ---- resample_stride: compression by a fixed factor --------------------------------------


def test_stride_spaces_steps_by_that_many_full_speed_frames():
    t = positions(100, 0.80, anchor="start", stride=2)
    assert len(t) == 40
    np.testing.assert_allclose(np.diff(t), 2.5)


@pytest.mark.parametrize("stride", [2, 3, 4, 8])
@pytest.mark.parametrize("n_frames", [37, 100, 200])
@pytest.mark.parametrize("anchor", ["start", "end"])
def test_anchored_length_with_stride_does_not_depend_on_level(stride, n_frames, anchor):
    lengths = {
        len(positions(n_frames, speed, anchor=anchor, stride=stride))
        for speed in SPEEDS
    }
    assert len(lengths) == 1


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_end_anchor_keeps_the_last_frame_with_stride(stride):
    for speed in SPEEDS:
        assert positions(100, speed, anchor="end", stride=stride)[-1] == 99


@pytest.mark.parametrize("stride", [2, 4])
def test_every_level_moves_the_same_distance_per_strided_step(stride):
    speeds = {1: 0.80, 2: 0.90, 5: 1.00}
    raw = [
        _moving_sequence(level_id, 100, 0.47 * speed)
        for level_id, speed in speeds.items()
    ]

    resampled = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1, 2: 2, 5: 5}), raw, raw, "first_5_seconds", stride=stride
    )

    for sequence in resampled:
        assert len(sequence) == len(positions(100, 0.80, anchor="start", stride=stride))
        np.testing.assert_allclose(np.diff(sequence["Pacman_X"]), 0.47 * stride)


def test_stride_without_speed_correction_keeps_every_nth_frame():
    # levels no longer matter: both sequences keep frames 0, 2, 4, ... as recorded
    raw = [_moving_sequence(1, 100, 0.38), _moving_sequence(2, 100, 0.43)]

    resampled = PacmanDataReader._resample_sequences(
        _reader_stub({1: 1, 2: 2}),
        raw,
        raw,
        "whole_level",
        speed_correction=False,
        stride=2,
    )

    for before, after in zip(raw, resampled, strict=True):
        assert len(after) == 50
        np.testing.assert_allclose(
            after["Pacman_X"], before["Pacman_X"].to_numpy()[::2]
        )


def test_strided_positions_stay_on_the_corridors():
    """
    Values are read at each step, not averaged over it: averaging the positions a step spans
    would cut across the corner, by more the larger the stride.
    """
    step = 0.38
    along = np.arange(0, 4.0, step)
    x = np.r_[
        -3.5 + along, np.full(len(along), 0.5)
    ]  # right along y = -9.5, then up x = 0.5
    y = np.r_[np.full(len(along), -9.5), -9.5 + step + along]
    raw = [pd.DataFrame({"level_id": 1, "Pacman_X": x, "Pacman_Y": y})]

    for stride in [2, 4, 8]:
        resampled = PacmanDataReader._resample_sequences(
            _reader_stub({1: 1}), raw, raw, "whole_level", stride=stride
        )
        xy = resampled[0][["Pacman_X", "Pacman_Y"]].to_numpy()
        off_centreline = np.min(np.abs(xy - np.floor(xy) - 0.5), axis=1)
        assert off_centreline.max() <= 0.25, stride


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_rejects_strides_that_are_not_whole_numbers_of_at_least_one(stride):
    raw = [_moving_sequence(1, 10, 0.38)]
    with pytest.raises(ValueError, match="resample_stride"):
        PacmanDataReader._resample_sequences(
            _reader_stub({1: 1}), raw, raw, "whole_level", stride=stride
        )
