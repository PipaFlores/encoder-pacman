import math


def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def transform_to_grid(pos):
    """Transform a pacman, or ghost, position to the grid frame of reference.
    It rounds the position to the nearest 0.5 unit.
    This ensures that Astar pathfinding works accurately with 0.5 unit steps.
    """
    return (round(pos[0] * 2) / 2, round(pos[1] * 2) / 2)
    # return (round(pos[0] - 0.5), round(pos[1] + 0.5))


def generate_squared_walls(wall_positions):
    """
    Generate a grid of walls by adding the vertices of the squares,
    The inputted positions are the upper left corner of the squares as given by unity maze data.
    This ensures that Astar pathfinding works accurately with 0.5 unit steps.
    """

    squared_walls = set(wall_positions)

    for wall in wall_positions:
        for direction in [
            (0.5, 0),
            (0, -0.5),
            (0.5, -0.5),
        ]:  # Fill the adjacent sides and diagonal.
            squared_walls.add((wall[0] + direction[0], wall[1] + direction[1]))
            squared_walls.add(
                (wall[0] + (direction[0] * 2), wall[1] + (direction[1] * 2))
            )
        for direction in [
            (1, -0.5),
            (0.5, -1),
        ]:  # fill the opposite sides of the square.
            squared_walls.add((wall[0] + direction[0], wall[1] + direction[1]))
    return squared_walls


def get_neighbors(pos, wall_positions, step, blocked_positions=None):
    """Get valid neighboring positions (up, right, down, left)."""
    TUNNEL_POS = [(-13.5, -0.5), (13.5, -0.5)]
    x, y = pos
    neighbors = [
        (x, y + step),  # up
        (x + step, y),  # right
        (x, y - step),  # down
        (x - step, y),  # left
    ]
    if blocked_positions is None:
        valid_neighbors = [
            (x, y) for (x, y) in neighbors if (x, y) not in wall_positions
        ]
    else:
        valid_neighbors = [
            (x, y)
            for (x, y) in neighbors
            if (x, y) not in wall_positions and (x, y) not in blocked_positions
        ]

    # Add the other tunnel position as a neighbor if in tunnel
    if pos == TUNNEL_POS[0]:
        valid_neighbors.append(TUNNEL_POS[1])
    elif pos == TUNNEL_POS[1]:
        valid_neighbors.append(TUNNEL_POS[0])

    return valid_neighbors


def calculate_path_and_distance(start, goal, grid, blocked_positions=None):
    """
    Calculate shortest path and distance between two points using A* algorithm.

    Args:
        start (tuple): Starting position (x, y)
        goal (tuple): Goal position (x, y)
        grid (set): Set of wall positions, discretized to 0.5 units.
        blocked_positions (set): Set of positions that are blocked. (e.g., ghost positions)

    Returns:
        tuple: (path, distance) where path is a list of positions and distance is the path length.
        Returns `[], math.inf` if no path is found.
    """
    from heapq import heappush, heappop

    STEP = 0.5
    if blocked_positions is not None:
        if (
            goal in blocked_positions
        ):  # if the goal is blocked, return an empty path and infinity distance
            return [], math.inf

    frontier = []
    heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = heappop(frontier)[1]

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, cost_so_far[goal]

        for next_pos in get_neighbors(current, grid, STEP, blocked_positions):
            new_cost = cost_so_far[current] + STEP

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan_distance(next_pos, goal)
                heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current

    return [], math.inf  # No path found


def calculate_ghost_paths_and_distances(
    pacman_pos, ghost_positions, grid, blocked_positions=None
):
    """
    Calculate the shortest paths and distances between Pacman and all ghosts.

    Args:
        pacman_pos (tuple): Pacman's position (x, y)
        ghost_positions (list): List of ghost positions [(x, y), ...]
        grid (set): Set of wall positions, discretized to 0.5 units.
        blocked_positions (set): Set of positions that are blocked.

    Returns:
        list: Tuples of (path, distance) to each ghost, in the same order as ghost_positions
    """
    # Round and transform positions to grid frame of reference.
    pacman_pos_transformed = transform_to_grid(pacman_pos)
    ghost_pos_transformed = [
        transform_to_grid(ghost_pos) for ghost_pos in ghost_positions
    ]

    results = []
    for ghost_pos in ghost_pos_transformed:
        path, distance = calculate_path_and_distance(
            ghost_pos, pacman_pos_transformed, grid, blocked_positions
        )
        results.append(([(pos[0], pos[1]) for pos in path], distance))
    return results


if __name__ == "__main__":
    import utils
    import pandas as pd
    import matplotlib.pyplot as plt

    FRAME = 214062  ## game_state_id of the frame to visualize

    gamestate_df = pd.read_csv(
        "data/gamestate.csv", converters={"user_id": lambda x: int(x)}
    )
    row = gamestate_df.loc[gamestate_df["game_state_id"] == FRAME].iloc[0]
    # row = gamestate_df.iloc[0]
    pacman_pos = (row["Pacman_X"], row["Pacman_Y"])
    print(f"Pacman position: {pacman_pos}")

    ghost_pos = [(row[f"Ghost{i + 1}_X"], row[f"Ghost{i + 1}_Y"]) for i in range(3)]
    print(f"Red ghost position: {ghost_pos[0]}")
    wall_pos = generate_squared_walls(utils.load_maze_data()[0])

    results = calculate_ghost_paths_and_distances(pacman_pos, ghost_pos, wall_pos)
    print(f"Distance to red: {results[0][1]}")
    print(f"Path to red: {results[0][0]}")

    print(f"Pacman: {pacman_pos}")
    print(f"Ghost: {ghost_pos[0]}")
    if results[0][0]:
        print(f"Path: {results[0][0]}")
    if results[0][1]:
        print(f"Distance: {results[0][1]}")

    plt.scatter(*zip(*wall_pos))
    if results[0][0]:
        plt.scatter(*zip(*results[0][0]))
    plt.scatter(*pacman_pos)
    plt.scatter(*ghost_pos[0])
    # Generate a grid with 0.5 units
    grid_size = 0.5
    x_min, x_max = (
        min(pacman_pos[0], min(x for x, y in wall_pos)),
        max(pacman_pos[0], max(x for x, y in wall_pos)),
    )
    y_min, y_max = (
        min(pacman_pos[1], min(y for x, y in wall_pos)),
        max(pacman_pos[1], max(y for x, y in wall_pos)),
    )

    x_ticks = [
        x_min + i * grid_size for i in range(int((x_max - x_min) / grid_size) + 1)
    ]
    y_ticks = [
        y_min + i * grid_size for i in range(int((y_max - y_min) / grid_size) + 1)
    ]

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True)

    plt.show()
