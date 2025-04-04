from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import functools
import time
import numpy as np
import os

## TODO Implement a logger class


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time:.4f} seconds")
        return value

    return wrapper_timer


def read_data(data_folder, game_list=None, user_list=None):
    """
    Read all data from CSV files.

    Args:
        data_folder: path to data folder
        game_list: list of game IDs to filter, default is None
        user_list: list of user IDs to filter, default is None

    Returns:
        tuple: (user_df, ip_df, redcap_df, game_df, gamestate_df, psychometrics_df)
        user_df: DataFrame with user data
        ip_df: DataFrame with IP data
        redcap_df: DataFrame with REDCap data
        game_df: DataFrame with game data
        gamestate_df: DataFrame with gamestate data
        psychometrics_df: DataFrame with psychometrics data
    """
    ## Read tables from csv
    BANNED_USERS = [42]  # Myself

    user_df = pd.read_csv(os.path.join(data_folder, "user.csv"))
    ip_df = pd.read_csv(os.path.join(data_folder, "userip.csv"))
    redcap_df = pd.read_csv(os.path.join(data_folder, "redcapdata.csv"))
    game_df = pd.read_csv(
        os.path.join(data_folder, "game.csv"),
        converters={"date_played": lambda x: pd.to_datetime(x)},
    )
    banned_game_ids = game_df.loc[game_df["user_id"].isin(BANNED_USERS), "game_id"]
    game_df = game_df[~game_df["game_id"].isin(banned_game_ids)]

    gamestate_df = pd.read_csv(
        os.path.join(data_folder, "gamestate.csv"),
        converters={
            "user_id": lambda x: int(x),
            "Pacman_X": lambda x: round(float(x), 2),
            "Pacman_Y": lambda x: round(float(x), 2),
        },
    )
    gamestate_df = gamestate_df[~gamestate_df["game_id"].isin(banned_game_ids)]
    psychometrics_df = pd.read_csv(
        os.path.join(data_folder, "psych\AiPerCogPacman_DATA_2025-03-03_0927.csv")
    )

    # Filter dataframes based on game_list and user_list if provided
    if game_list is not None:
        game_df = game_df[game_df["game_id"].isin(game_list)]
        gamestate_df = gamestate_df[gamestate_df["game_id"].isin(game_list)]

    if user_list is not None:
        user_df = user_df[user_df["user_id"].isin(user_list)]
        game_df = game_df[game_df["user_id"].isin(user_list)]
        gamestate_df = gamestate_df[gamestate_df["user_id"].isin(user_list)]

    return user_df, ip_df, redcap_df, game_df, gamestate_df, psychometrics_df


def load_maze_data():
    """
    Load wall and pellet positions from Unity tilemap files and return them as a tuple of two lists.


    Returns:
        tuple: (wall_positions, pellet_positions) where each is a list of (x,y) tuples
    """
    # Get the directory where utils.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to utils.py location
    walls_file = os.path.join(current_dir, "grid", "walls.unity")
    pellets_file = os.path.join(current_dir, "grid", "pellets.unity")

    with open(walls_file, "r") as f:
        walls_content = f.read()
    with open(pellets_file, "r") as f:
        pellets_content = f.read()

    wall_positions = parse_unity_tilemap_(walls_content)
    pellet_positions = parse_unity_tilemap_(pellets_content)

    return wall_positions, pellet_positions


def parse_unity_tilemap_(file_content):
    """
    Parse Unity tilemap file content to extract tile positions.

    Args:
        file_content (str): Content of the Unity tilemap file

    Returns:
        list: List of (x, y) tuples representing tile positions
    """
    positions = []
    current_pos = None

    for line in file_content.split("\n"):
        line = line.strip()

        # Look for position declarations
        if line.startswith("- first:"):
            # Reset current position
            current_pos = None

        # Extract x coordinate
        if "x:" in line:
            x = int(line.split("x:")[1].split(",")[0].strip())

        # Extract y coordinate
        if "y:" in line:
            y = int(line.split("y:")[1].split(",")[0].strip())
            current_pos = (x, y)

        if "m_TileIndex:" in line and current_pos:
            positions.append(current_pos)
            current_pos = None

    return positions


def plot_ts(ts, title):
    fig, axs = plt.subplots(ts.shape[1], sharex=True, gridspec_kw={"hspace": 0})
    plt.suptitle(title, fontsize="30")

    for i in range(ts.shape[1]):
        axs[i].set_ylabel(f"{ts.columns[i]}", fontsize="8")
        axs[i].set_xlabel("Step", fontsize="20")
        axs[i].plot(ts.iloc[:, i])

    plt.show()


def pos_mirroring(df, return_quadrant=False):
    """
    Mirror the positions of Pacman
    on each quadrant of the maze. Each quadrant
    will mimic the first quadrant (upper right).
    If return_quadrant is True, add a column
    'quadrant' to the dataframe with the quadrant
    that the Pacman is in.
    """
    MIRROR_X = 0.0
    MIRROR_Y = -0.5

    mirrored_df = df.copy()
    if return_quadrant:
        mirrored_df["quadrant"] = np.float64(0)
    for i, row in mirrored_df.iterrows():
        if row["Pacman_X"] < MIRROR_X:
            mirrored_df.loc[i, "Pacman_X"] = (MIRROR_X - row["Pacman_X"]) + MIRROR_X
        if row["Pacman_Y"] < MIRROR_Y:
            mirrored_df.loc[i, "Pacman_Y"] = (MIRROR_Y - row["Pacman_Y"]) + MIRROR_Y
        if return_quadrant:
            if row["Pacman_X"] >= MIRROR_X and row["Pacman_Y"] >= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 1.0
            elif row["Pacman_X"] <= MIRROR_X and row["Pacman_Y"] >= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 2.0
            elif row["Pacman_X"] <= MIRROR_X and row["Pacman_Y"] <= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 3.0
            elif row["Pacman_X"] >= MIRROR_X and row["Pacman_Y"] <= MIRROR_Y:
                mirrored_df.loc[i, "quadrant"] = 4.0

    return mirrored_df


def calculate_velocities(
    trajectory: np.ndarray, round: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate velocities from position data, it rounds and removes signed zeros to avoid noise issues.

    Args:
        trajectory: Array of shape (N, 2) containing x,y coordinates
        round: Whether to round velocities to 0.5 to remove small noise in direction changes

    Returns:
        dx: Array of x-velocities
        dy: Array of y-velocities
    """
    x, y = trajectory[:, 0], trajectory[:, 1]

    # Calculate velocities
    if round:
        dx = (
            np.round(np.diff(x, prepend=x[0]) * 2) / 2
        )  # round to 0.5 to remove small noise in direction changes
        dy = np.round(np.diff(y, prepend=y[0]) * 2) / 2
    else:
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])

    dx = pd.Series(dx).replace(0, 0).values  # remove signed zeros using .loc
    dy = pd.Series(dy).replace(0, 0).values

    dx = np.nan_to_num(dx, nan=0)
    dy = np.nan_to_num(dy, nan=0)

    return dx, dy
