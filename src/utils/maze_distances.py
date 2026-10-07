import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from . import Astar


class MazeDistances:
    """
    Shortest-path distances between every pair of points on the A* lattice, computed once.

    The maze never changes, so the distance between two positions is the same in every frame
    of every game. Instead of running A* per query, this precomputes the whole table on the
    same 0.5-unit lattice and neighbour rule A* uses (`Astar.get_neighbors`, tunnel wrap
    included) and answers queries by lookup.

    `distance(start, goal)` returns exactly what
    `Astar.calculate_path_and_distance(start, goal, wall_grid)` returns, with the same
    conventions: positions snap with `Astar.transform_to_grid`, a goal on a wall is
    unreachable (inf), and a start on a wall first steps onto its free neighbours.

    Only for the static maze: queries with `blocked_positions` (e.g. ghosts treated as walls)
    change the graph per query and still need `Astar.calculate_path_and_distance`.
    """

    STEP = 0.5

    def __init__(self, wall_grid: set[tuple[float, float]]):
        self.wall_grid = wall_grid

        # Lattice bounding box of the walls, in half-units. The outer wall encloses the maze,
        # so every free point inside it is in the maze or the ghost house.
        xs = [p[0] for p in wall_grid]
        ys = [p[1] for p in wall_grid]
        self._x0, self._y0 = round(min(xs) * 2), round(min(ys) * 2)
        width = round(max(xs) * 2) - self._x0 + 1
        height = round(max(ys) * 2) - self._y0 + 1

        points = [
            ((self._x0 + i) / 2, (self._y0 + j) / 2)
            for i in range(width)
            for j in range(height)
        ]
        nodes = [p for p in points if p not in wall_grid]
        node_idx = {p: k for k, p in enumerate(nodes)}

        # Every move costs one STEP, the wrap included, so unweighted search is exact.
        src, dst = [], []
        for p, k in node_idx.items():
            for q in Astar.get_neighbors(p, wall_grid, self.STEP):
                if q in node_idx:
                    src.append(k)
                    dst.append(node_idx[q])
        adjacency = coo_matrix(
            (np.ones(len(src)), (src, dst)), shape=(len(nodes), len(nodes))
        ).tocsr()
        node_dist = shortest_path(adjacency, directed=True, unweighted=True) * self.STEP

        # A start on a wall is not a node, but A* still expands from it onto its free
        # neighbours: one STEP plus the best of their rows. Walls without free neighbours
        # keep no row, so queries from them are unreachable.
        wall_rows = []
        wall_starts = []
        for p in points:
            if p in node_idx:
                continue
            free = [
                node_idx[q]
                for q in Astar.get_neighbors(p, wall_grid, self.STEP)
                if q in node_idx
            ]
            if free:
                wall_starts.append(p)
                wall_rows.append(self.STEP + node_dist[free].min(axis=0))

        # rows: starts (nodes, then walls with free neighbours); columns: goals (nodes only)
        self._table = np.vstack([node_dist, *wall_rows]) if wall_rows else node_dist
        self._row = np.full((width, height), -1, dtype=np.int64)
        self._col = np.full((width, height), -1, dtype=np.int64)
        for k, p in enumerate(nodes):
            i, j = self._cell(p)
            self._row[i, j] = k
            self._col[i, j] = k
        for k, p in enumerate(wall_starts, start=len(nodes)):
            i, j = self._cell(p)
            self._row[i, j] = k

    def _cell(self, point):
        return round(point[0] * 2) - self._x0, round(point[1] * 2) - self._y0

    def _lookup(self, positions, index):
        """Map raw positions to table indices; -1 for non-finite or out-of-maze positions."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        out = np.full(len(positions), -1, dtype=np.int64)
        finite = np.isfinite(positions).all(axis=1)
        # np.round rounds half to even, like the built-in round in transform_to_grid
        i = np.round(positions[finite, 0] * 2).astype(np.int64) - self._x0
        j = np.round(positions[finite, 1] * 2).astype(np.int64) - self._y0
        inside = (i >= 0) & (i < index.shape[0]) & (j >= 0) & (j < index.shape[1])
        found = np.full(len(i), -1, dtype=np.int64)
        found[inside] = index[i[inside], j[inside]]
        out[finite] = found
        return out

    def distances(self, starts, goals) -> np.ndarray:
        """
        Vectorized distances from each start to the goal in the same row.

        Args:
            starts: (n, 2) array of raw (x, y) positions.
            goals: (n, 2) array of raw (x, y) positions.

        Returns:
            np.ndarray: (n,) float distances in maze units; inf where no path exists or a
            position is non-finite or outside the maze.
        """
        rows = self._lookup(starts, self._row)
        cols = self._lookup(goals, self._col)
        out = np.full(len(rows), np.inf)
        ok = (rows >= 0) & (cols >= 0)
        out[ok] = self._table[rows[ok], cols[ok]]
        return out

    def distance(self, start, goal) -> float:
        """Distance from one raw (x, y) position to another; see `distances`."""
        return float(self.distances([start], [goal])[0])
