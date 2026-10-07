import math
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.datahandlers.pacman_data_reader import PacmanDataReader
from src.utils import Astar, MazeDistances
from src.utils.utils import load_maze_data


@pytest.fixture(scope="module")
def walls():
    return Astar.generate_squared_walls(load_maze_data()[0])


@pytest.fixture(scope="module")
def maze_distances(walls):
    return MazeDistances(walls)


@pytest.fixture(scope="module")
def lattice(walls):
    """Free lattice points, and wall points whose free neighbours are all in the maze."""
    xs = [p[0] for p in walls]
    ys = [p[1] for p in walls]
    points = [
        (x / 2, y / 2)
        for x in range(round(min(xs) * 2), round(max(xs) * 2) + 1)
        for y in range(round(min(ys) * 2), round(max(ys) * 2) + 1)
    ]
    free = [p for p in points if p not in walls]
    free_set = set(free)
    # A* from a wall point with a free neighbour *outside* the outer wall never terminates
    # when the goal is unreachable, so those are not comparable; positions never land there.
    wall_starts = []
    for p in points:
        if p in free_set:
            continue
        neighbours = Astar.get_neighbors(p, walls, 0.5)
        if neighbours and all(q in free_set for q in neighbours):
            wall_starts.append(p)
    return free, wall_starts


def _astar(start, goal, walls):
    return Astar.calculate_path_and_distance(start, goal, walls)[1]


def _jitter(points, rng):
    """
    Off-lattice positions, as logged by the game: up to half a step from a free point, so they
    snap to tile centres, tile edges and wall faces alike. x is clipped inside the tunnel ends
    (logged positions stay within +-13.52), since the seal points at +-14.0 border open space
    outside the maze.
    """
    jittered = points + rng.uniform(-0.49, 0.49, size=points.shape)
    jittered[:, 0] = np.clip(jittered[:, 0], -13.74, 13.74)
    return jittered


def test_matches_astar(walls, maze_distances, lattice):
    free, wall_starts = lattice
    rng = random.Random(0)
    pairs = [(rng.choice(free), rng.choice(free)) for _ in range(300)]
    pairs += [(rng.choice(wall_starts), rng.choice(free)) for _ in range(100)]
    pairs += [
        (rng.choice(free), rng.choice(wall_starts)) for _ in range(50)
    ]  # goal on a wall

    for start, goal in pairs:
        assert maze_distances.distance(start, goal) == _astar(start, goal, walls), (
            start,
            goal,
        )


def test_raw_positions_snap_like_astar(walls, maze_distances, lattice):
    free, _ = lattice
    rng = np.random.default_rng(0)
    starts = _jitter(
        np.array([free[i] for i in rng.integers(len(free), size=300)]), rng
    )
    goals = _jitter(np.array([free[i] for i in rng.integers(len(free), size=300)]), rng)

    table = maze_distances.distances(starts, goals)
    reference = [_astar(tuple(s), tuple(g), walls) for s, g in zip(starts, goals)]
    np.testing.assert_array_equal(table, reference)


def test_tunnel_route(maze_distances):
    # same pair as the A* regression test for #50
    assert maze_distances.distance((-9.5, -0.5), (12.5, -0.5)) == 4.0 + 0.5 + 1.0


def test_unreachable_and_invalid_positions(maze_distances):
    ghost_house = (0.5, -0.5)  # a ghost waiting in the house cannot reach Pacman
    spawn = (0.5, -9.5)
    assert maze_distances.distance(ghost_house, spawn) == math.inf
    assert maze_distances.distance((np.nan, 0.0), spawn) == math.inf
    assert maze_distances.distance((100.0, 100.0), spawn) == math.inf


def test_reader_distances_match_astar(walls):
    rng = np.random.default_rng(1)
    free = [
        p
        for p in ((x + 0.5, y + 0.5) for x in range(-14, 14) for y in range(-17, 14))
        if p not in walls
    ]
    columns = {}
    for name in ["Pacman", "Ghost1", "Ghost2", "Ghost3", "Ghost4"]:
        pos = _jitter(
            np.array([free[i] for i in rng.integers(len(free), size=40)]), rng
        )
        columns[f"{name}_X"], columns[f"{name}_Y"] = pos[:, 0], pos[:, 1]
    gamestate_df = pd.DataFrame(
        columns, index=pd.RangeIndex(1000, 1040, name="game_state_id")
    )

    reader = SimpleNamespace(gamestate_df=gamestate_df)
    result = PacmanDataReader._calculate_astar_distances(reader)

    for row in result.itertuples():
        pacman = (row.Pacman_X, row.Pacman_Y)
        for i in range(1, 5):
            ghost = (getattr(row, f"Ghost{i}_X"), getattr(row, f"Ghost{i}_Y"))
            assert getattr(row, f"Ghost{i}_distance") == _astar(ghost, pacman, walls)
