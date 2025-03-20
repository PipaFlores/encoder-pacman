from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import functools
import time
import numpy as np
import torch
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
    BANNED_USERS = [42] # Myself

    user_df = pd.read_csv(os.path.join(data_folder, 'user.csv'))
    ip_df = pd.read_csv(os.path.join(data_folder, 'userip.csv'))
    redcap_df = pd.read_csv(os.path.join(data_folder, 'redcapdata.csv'))
    game_df = pd.read_csv(os.path.join(data_folder, 'game.csv'), converters={'date_played': lambda x: pd.to_datetime(x)})
    banned_game_ids = game_df.loc[game_df['user_id'].isin(BANNED_USERS), 'game_id']
    game_df = game_df[~game_df['game_id'].isin(banned_game_ids)]
    
    gamestate_df = pd.read_csv(os.path.join(data_folder, 'gamestate.csv'), converters={'user_id': lambda x: int(x),
                                                                                      'Pacman_X': lambda x: round(float(x), 2),
                                                                                      'Pacman_Y': lambda x: round(float(x), 2)
                                                                                      })
    gamestate_df = gamestate_df[~gamestate_df['game_id'].isin(banned_game_ids)]
    psychometrics_df = pd.read_csv(os.path.join(data_folder, 'psych\AiPerCogPacman_DATA_2025-03-03_0927.csv'))

    # Filter dataframes based on game_list and user_list if provided
    if game_list is not None:
        game_df = game_df[game_df['game_id'].isin(game_list)]
        gamestate_df = gamestate_df[gamestate_df['game_id'].isin(game_list)]

    if user_list is not None:
        user_df = user_df[user_df['user_id'].isin(user_list)]
        game_df = game_df[game_df['user_id'].isin(user_list)]
        gamestate_df = gamestate_df[gamestate_df['user_id'].isin(user_list)]
        

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
    walls_file = os.path.join(current_dir, 'grid', 'walls.unity')
    pellets_file = os.path.join(current_dir, 'grid', 'pellets.unity')

    with open(walls_file, 'r') as f:
        walls_content = f.read()
    with open(pellets_file, 'r') as f:
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
    
    for line in file_content.split('\n'):
        line = line.strip()

        # Look for position declarations
        if line.startswith('- first:'):
            # Reset current position
            current_pos = None

        # Extract x coordinate
        if 'x:' in line:
            x = int(line.split('x:')[1].split(',')[0].strip())

            
        # Extract y coordinate
        if 'y:' in line:
            y = int(line.split('y:')[1].split(',')[0].strip())
            current_pos = (x, y)

            
        if 'm_TileIndex:' in line and current_pos:
            positions.append(current_pos)
            current_pos = None
            
    
    return positions

def plot_ts(ts, title):
    fig, axs = plt.subplots(ts.shape[1], sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle(title, fontsize='30')

    for i in range(ts.shape[1]):
        axs[i].set_ylabel(f'{ts.columns[i]}', fontsize='8')
        axs[i].set_xlabel('Step', fontsize ='20')
        axs[i].plot(ts.iloc[:,i])
        
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
        mirrored_df['quadrant'] = np.float64(0)
    for i, row in mirrored_df.iterrows():
        if row['Pacman_X'] < MIRROR_X:
            mirrored_df.loc[i, 'Pacman_X'] = (MIRROR_X - row['Pacman_X']) + MIRROR_X
        if row['Pacman_Y'] < MIRROR_Y:
            mirrored_df.loc[i, 'Pacman_Y'] = (MIRROR_Y - row['Pacman_Y']) + MIRROR_Y
        if return_quadrant:
            if row['Pacman_X'] >= MIRROR_X and row['Pacman_Y'] >= MIRROR_Y:
                mirrored_df.loc[i, 'quadrant'] = 1.0
            elif row['Pacman_X'] <= MIRROR_X and row['Pacman_Y'] >= MIRROR_Y:
                mirrored_df.loc[i, 'quadrant'] = 2.0
            elif row['Pacman_X'] <= MIRROR_X and row['Pacman_Y'] <= MIRROR_Y:
                mirrored_df.loc[i, 'quadrant'] = 3.0
            elif row['Pacman_X'] >= MIRROR_X and row['Pacman_Y'] <= MIRROR_Y:
                mirrored_df.loc[i, 'quadrant'] = 4.0

    return mirrored_df

def create_game_trajectory_tensor(processed_df, max_sequence_length=None):
    """
    Creates a tensor of shape (num_games, sequence_length, num_features)
    where num_features = 4 (Pacman_X, Pacman_Y, score, powerPellets) for autoencoder training.
    
    Args:
        processed_df: DataFrame containing all games' preprocessed data with game_id column
        max_sequence_length: Optional, pad/truncate all sequences to this length
        
    Returns:
        tensor: A tensor of shape (num_games, max_sequence_length, num_features)
        mask: A mask tensor of shape (num_games, max_sequence_length) indicating valid timesteps
        game_ids: A list of unique game IDs
    """
    # If max_sequence_length isn't provided, use the longest game
    if max_sequence_length is None:
        max_sequence_length = processed_df.groupby('game_id').size().max()
    
    game_ids = processed_df['game_id'].unique()
    # Determine the number of features from the DataFrame
    num_features = processed_df.shape[1] - 1  # Subtract 1 for the 'game_id' column
    num_games = len(game_ids)
    
    # Initialize tensor with zeros
    # Shape: (num_games, max_sequence_length, num_features)
    tensor = torch.zeros((num_games, max_sequence_length, num_features))
    
    # Create mask to track actual sequence lengths
    mask = torch.zeros((num_games, max_sequence_length), dtype=torch.bool)
    
    # Create dictionary to store game_id to index mapping
    game_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}
    
    for game_id in game_ids:
        # Get game data excluding 'game_id' column
        game_data = processed_df[processed_df['game_id'] == game_id].iloc[:, 1:].values
        seq_len = min(len(game_data), max_sequence_length)
        game_idx = game_to_idx[game_id]
        
        # Fill tensor and mask
        tensor[game_idx, :seq_len, :] = torch.FloatTensor(game_data[:seq_len])
        mask[game_idx, :seq_len] = 1
    
    # The resulting tensor will have:
    # - First dimension: different games
    # - Second dimension: timesteps in the game
    # - Third dimension: features (X, Y, score, powerPellets)

    return tensor, mask, game_ids


def preprocess_game_data(gamestate_df, series_type=['position'], include_game_state_vars= False, include_timesteps = True):
    """
    Preprocess gamestates' data before converting to tensor for Autoencoder training.
    
    Args:
        gamestate_df: DataFrame containing raw game data
        series_type: List of series types to include in the preprocessing (e.g., ['position', 'movements', 'input'])
        include_game_state_vars: Boolean indicating whether to include game state variables (score, powerPellets)
        include_timesteps: Boolean indicating whether to include time elapsed in the features
        
    Returns:
        processed_df: DataFrame containing preprocessed game data with selected features
    """
    
    GAME_STATE_VARS = ['score', 'powerPellets'] if include_game_state_vars else []

    features = ['game_id'] + GAME_STATE_VARS

    if include_timesteps:
        features.extend(['time_elapsed'])

    if 'position' in series_type:
        features.extend(['Pacman_X', 'Pacman_Y']) # No preprocessing here as it is done in the read_data() function
        
    
    if 'movements' in series_type:
        # Convert movement directions to dx, dy components
        direction_mapping = {
            'right': (1, 0),
            'left': (-1, 0),
            'up': (0, 1),
            'down': (0, -1),
            'none': (0, 0)
        }
        gamestate_df['movement_dx'] = gamestate_df['movement_direction'].map(lambda d: direction_mapping[d][0])
        gamestate_df['movement_dy'] = gamestate_df['movement_direction'].map(lambda d: direction_mapping[d][1])
        features.extend(['movement_dx', 'movement_dy'])
    
    if 'input' in series_type:
        # Similarly for input directions
        gamestate_df['input_dx'] = gamestate_df['input_direction'].map(lambda d: direction_mapping[d][0])
        gamestate_df['input_dy'] = gamestate_df['input_direction'].map(lambda d: direction_mapping[d][1])
        features.extend(['input_dx', 'input_dy'])
    
    processed_df = gamestate_df[features].copy()
    
    # Convert to float
    for col in processed_df.columns:
        if col != 'game_id':  # Skip game_id conversion
            processed_df[col] = processed_df[col].astype(float)
    
    return processed_df


def calculate_velocities(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate velocities from position data, it rounds and removes signed zeros to avoid noise issues.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        dx: Array of x-velocities
        dy: Array of y-velocities
    """
    # Calculate velocities
    dx = np.round(np.diff(x, prepend=x[0]) * 2) / 2 # round to 0.5 to remove small noise in direction changes
    dy = np.round(np.diff(y, prepend=y[0]) * 2) / 2

    dx = pd.Series(dx).replace(0, 0).values # remove signed zeros using .loc
    dy = pd.Series(dy).replace(0, 0).values

    dx = np.nan_to_num(dx, nan=0)
    dy = np.nan_to_num(dy, nan=0)

    return dx, dy
