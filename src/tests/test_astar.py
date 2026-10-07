import math
import random
from collections import deque

import pytest

from src.utils import Astar
from src.utils.utils import load_maze_data

STEP = 0.5  # lattice resolution used by Astar.calculate_path_and_distance
SPAWN = (0.5, -9.5)  # Pacman's start tile, inside the main maze


@pytest.fixture(scope="module")
def walls():
    return Astar.generate_squared_walls(load_maze_data()[0])


def _exhaustive_distances(start, walls):
    """
    Reference shortest-path distances from `start` to every reachable lattice node.

    Every move, the tunnel wrap included, costs one STEP, so breadth-first search is exact
    and needs no heuristic: it is what A* has to agree with.
    """
    dist = {start: 0.0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in Astar.get_neighbors(current, walls, STEP):
            if nxt not in dist:
                dist[nxt] = dist[current] + STEP
                queue.append(nxt)
    return dist


@pytest.fixture(scope="module")
def maze_tiles(walls):
    """Walkable tile centres reachable from the spawn, i.e. the maze outside the ghost house."""
    reachable = _exhaustive_distances(SPAWN, walls)
    return sorted(p for p in reachable if p[0] % 1 == 0.5 and p[1] % 1 == 0.5)


def test_route_through_tunnel_is_found(walls):
    """Regression test for #50: plain Manhattan made A* go the long way round."""
    # both on the tunnel row: four tiles from the left end, one tile from the right end
    left, right = (-9.5, -0.5), (12.5, -0.5)
    assert left not in walls and right not in walls

    path, distance = Astar.calculate_path_and_distance(left, right, walls)

    # the unfixed heuristic returned 28.0 here, the way round through the maze
    assert distance == 4.0 + STEP + 1.0
    assert Astar.TUNNEL_POS[0] in path and Astar.TUNNEL_POS[1] in path


def test_distances_match_exhaustive_search(walls, maze_tiles):
    rng = random.Random(0)
    left_side = [p for p in maze_tiles if p[0] < -8]
    right_side = [p for p in maze_tiles if p[0] > 8]
    # random pairs anywhere, plus pairs on opposite sides of the maze where the tunnel matters
    pairs = [tuple(rng.sample(maze_tiles, 2)) for _ in range(150)]
    pairs += [(rng.choice(left_side), rng.choice(right_side)) for _ in range(100)]

    by_start = {}
    for start, goal in pairs:
        if start not in by_start:
            by_start[start] = _exhaustive_distances(start, walls)
        path, distance = Astar.calculate_path_and_distance(start, goal, walls)

        assert distance == by_start[start][goal], (start, goal)
        assert path[0] == start and path[-1] == goal
        assert (len(path) - 1) * STEP == distance


def test_ghost_house_is_unreachable(walls, maze_tiles):
    reachable = _exhaustive_distances(SPAWN, walls)
    # walkable tile centres the spawn can't reach: the ghost house interior
    house = [
        (x + 0.5, y + 0.5)
        for x in range(-14, 14)
        for y in range(-17, 14)
        if (x + 0.5, y + 0.5) not in walls and (x + 0.5, y + 0.5) not in reachable
    ]
    assert house, "expected a sealed ghost house"

    path, distance = Astar.calculate_path_and_distance(house[0], SPAWN, walls)
    assert path == [] and distance == math.inf
