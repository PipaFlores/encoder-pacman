from utils import timer

def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def transform_to_grid(pos):
    """Transform a pacman, or ghost, position to the grid frame of reference."""
    return (round(pos[0] - 0.5), round(pos[1] + 0.5))

@timer
def generate_intermediate_walls(wall_positions):
    """Generate additional wall positions between close wall points."""
    intermediate_walls = set()
    expanded_walls = set()
    corner_walls = set()
    wall_list = list(wall_positions)


    # Add intermediate points between adjacent points
    for i, pos1 in enumerate(wall_list):
        for pos2 in wall_list[i+1:]:
            # Calculate distance between points
            dist = manhattan_distance(pos1, pos2)
            if dist == 1:  # If points are directly adjacent (not diagonal)
                # Generate intermediate point
                if pos1[0] == pos2[0]:  # Same x-coordinate, vertical adjacency
                    y = (pos1[1] + pos2[1]) / 2
                    intermediate_walls.add((pos1[0], y))
                elif pos1[1] == pos2[1]:  # Same y-coordinate, horizontal adjacency
                    x = (pos1[0] + pos2[0]) / 2
                    intermediate_walls.add((x, pos1[1]))
    
    filled_walls = intermediate_walls.union(wall_positions)

    # TODO : Figure out how to fill corners

    # # Add the four adjacent points to the set
    # for pos in filled_walls:
    #     expanded_walls.add((pos[0] + .5, pos[1]))
    #     expanded_walls.add((pos[0] - .5, pos[1]))
    #     expanded_walls.add((pos[0], pos[1] + .5))
    #     expanded_walls.add((pos[0], pos[1] - .5))

    # filled_walls = filled_walls.union(expanded_walls)

    # # Fill corners
    # for pos in filled_walls:
    #     for direction in [(0.5,0), (0,-0.5), (-0.5,0), (0,0.5)]:
    #         for step in range(1,4):
    #             if (pos[0] + direction[0]*step, pos[1] + direction[1]*step) in filled_walls:
    #                 if step < 3:
    #                     break  # Skip to the next direction if a wall is found at step 1 or 2
    #                 elif step == 3:
    #                     corner_walls.add((pos[0] + direction[0], pos[1] + direction[1]))

        # counter = 0
        # for quadrant in [(0.5,0.5), (0.5,-0.5), (-0.5,0.5), (-0.5,-0.5)]:
        #     if (pos[0] + quadrant[0], pos[1] + quadrant[1]) in filled_walls:
        #         counter += 1
        #     else:
        #         unfilled_corner = (pos[0] + quadrant[0], pos[1] + quadrant[1])
        # if counter == 3:
        #     corner_walls.add(unfilled_corner)

    # filled_walls = filled_walls.union(corner_walls)

    return filled_walls

def get_neighbors(pos, wall_positions, step=1):
    """Get valid neighboring positions (up, right, down, left)."""
    x, y = pos
    neighbors = [
        (x, y+step),  # up
        (x+step, y),  # right
        (x, y-step),  # down
        (x-step, y)   # left
    ]
    return [(x, y) for (x, y) in neighbors if (x, y) not in wall_positions]

def calculate_path_and_distance(start, goal, wall_positions):
    """
    Calculate shortest path and distance between two points using A* algorithm.
    
    Args:
        start (tuple): Starting position (x, y)
        goal (tuple): Goal position (x, y)
        wall_positions (set): Set of wall positions to avoid
        
    Returns:
        tuple: (path, distance) where path is a list of positions and distance is the path length
    """
    from heapq import heappush, heappop
    
    wall_positions = set(wall_positions)
    step = 1
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
            
        for next_pos in get_neighbors(current, wall_positions, step):
            new_cost = cost_so_far[current] + step
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan_distance(next_pos, goal)
                heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current
    
    return [], float('inf')  # No path found


def calculate_ghost_paths_and_distances(pacman_pos, ghost_positions, wall_positions):
    """
    Calculate the shortest paths and distances between Pacman and all ghosts.
    
    Args:
        pacman_pos (tuple): Pacman's position (x, y)
        ghost_positions (list): List of ghost positions [(x, y), ...]
        wall_positions (set): Set of wall positions
        
    Returns:
        list: Tuples of (path, distance) to each ghost, in the same order as ghost_positions
    """
    # Round and transform positions to grid frame of reference.
    pacman_pos_scaled = transform_to_grid(pacman_pos)
    ghost_pos_scaled = [transform_to_grid(ghost_pos) for ghost_pos in ghost_positions]

    results = []
    for ghost_pos in ghost_pos_scaled:
        path, distance = calculate_path_and_distance(ghost_pos, pacman_pos_scaled, wall_positions)
        results.append(([(pos[0], pos[1]) for pos in path], distance)) 
    return results


if __name__ == "__main__":
    import utils
    import pandas as pd
    import matplotlib.pyplot as plt

    FRAME = 214062 ## game_state_id of the frame to visualize

    gamestate_df = pd.read_csv('data/gamestate.csv', converters={'user_id': lambda x: int(x)})
    row = gamestate_df.loc[gamestate_df['game_state_id'] == FRAME].iloc[0]
    # row = gamestate_df.iloc[0]
    pacman_pos = (row['Pacman_X'], row['Pacman_Y'])
    print(f'Pacman position: {pacman_pos}')

    ghost_pos = [(row[f'Ghost{i+1}_X'], row[f'Ghost{i+1}_Y']) for i in range(3)]
    print(f'Red ghost position: {ghost_pos[0]}')
    wall_pos, _ = utils.load_maze_data()
    
    results = calculate_ghost_paths_and_distances(pacman_pos, ghost_pos, wall_pos)
    print(f'Distance to red: {results[0][1]}')
    print(f'Path to red: {results[0][0]}')

    print(f'Pacman: {pacman_pos}')
    print(f'Ghost: {ghost_pos[0]}')
    if results[0][0]:
        print(f'Path: {results[0][0]}')
    if results[0][1]:
        print(f'Distance: {results[0][1]}')

    plt.scatter(*zip(*wall_pos))
    if results[0][0]:
        plt.scatter(*zip(*results[0][0]))
    plt.scatter(*pacman_pos)
    plt.scatter(*ghost_pos[0])
    # Generate a grid with 0.5 units
    grid_size = 0.5
    x_min, x_max = min(pacman_pos[0], min(x for x, y in wall_pos)), max(pacman_pos[0], max(x for x, y in wall_pos))
    y_min, y_max = min(pacman_pos[1], min(y for x, y in wall_pos)), max(pacman_pos[1], max(y for x, y in wall_pos))

    x_ticks = [x_min + i * grid_size for i in range(int((x_max - x_min) / grid_size) + 1)]
    y_ticks = [y_min + i * grid_size for i in range(int((y_max - y_min) / grid_size) + 1)]

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True)

    plt.show()

    


