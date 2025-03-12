import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox
import functools
import time
import stumpy
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

def parse_game_table(sql_file):
    columns = ['game_id', 'user_id', 'session_number', 'game_in_session', 
               'total_games_played', 'source', 'date_played', 'game_duration', 
               'win', 'level']
    
    values = []
    with open(sql_file, 'r') as file:
        for line in file:
            # Look for lines containing value tuples
            if line.strip().startswith('('):
                # Remove parentheses, comma, and semicolon
                row = line.strip().strip('(),;')
                # Split the values and handle different data types
                row_values = []
                for val in row.split(','):
                    val = val.strip()
                    # Remove any trailing semicolon
                    val = val.rstrip(');')
                    if val.startswith("'"):  # String value
                        row_values.append(val.strip("'"))
                    else:  # Numeric value
                        try:
                            row_values.append(float(val) if '.' in val else int(val))
                        except ValueError:
                            continue  # Skip invalid rows
                
                # Only add rows that have the correct number of columns
                if len(row_values) == len(columns):
                    values.append(row_values)
    
    # Create DataFrame
    df = pd.DataFrame(values, columns=columns)
    return df


def parse_sql_table(sql_file, table_name):
    """
    Parse a specific table from a SQL dump file.
    """
    # First, extract column names from INSERT statement
    columns = []
    values = []
    with open(sql_file, 'r') as file:
        for line in file:
            if f'INSERT INTO `{table_name}`' in line:
                # Extract column names from the INSERT statement
                columns_start = line.index('(') + 1
                columns_end = line.index(')', columns_start)
                columns_str = line[columns_start:columns_end]
                columns = [col.strip('` ') for col in columns_str.split(',')]
                continue
            
            # Look for lines containing value tuples
            if line.strip().startswith('('):
                # Remove parentheses and trailing comma/semicolon
                row = line.strip()
                if row.endswith('),'):
                    row = row[:-2]
                elif row.endswith(');'):
                    row = row[:-2]
                else:
                    row = row.strip('()')
                
                # Split into values
                row_values = []
                current_value = ''
                in_quotes = False
                
                for char in row:
                    if char == ',' and not in_quotes:
                        val = current_value.strip()
                        row_values.append(val)
                        current_value = ''
                    elif char == "'":
                        in_quotes = not in_quotes
                        current_value += char
                    else:
                        current_value += char
                
                # Add the last value
                if current_value:
                    row_values.append(current_value.strip())
                
                # Convert values to appropriate types
                processed_values = []
                for val in row_values:
                    val = val.strip()
                    if val.startswith("'") and val.endswith("'"):
                        processed_values.append(val.strip("'"))
                    elif val.startswith('0x'):
                        processed_values.append(val)
                    else:
                        try:
                            if '.' in val:
                                processed_values.append(float(val))
                            else:
                                processed_values.append(int(val))
                        except ValueError:
                            processed_values.append(val)
                
                if len(processed_values) == len(columns):
                    values.append(processed_values)
    
    # Create DataFrame
    df = pd.DataFrame(values, columns=columns)
    return df

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
def parse_unity_tilemap(file_content):
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

def load_maze_data():
    """
    Load wall and pellet positions from Unity tilemap files.
    
        
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
        
    wall_positions = parse_unity_tilemap(walls_content)
    pellet_positions = parse_unity_tilemap(pellets_content)
    
    return wall_positions, pellet_positions


def plot_ts(ts, title):
    fig, axs = plt.subplots(ts.shape[1], sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle(title, fontsize='30')

    for i in range(ts.shape[1]):
        axs[i].set_ylabel(f'{ts.columns[i]}', fontsize='8')
        axs[i].set_xlabel('Step', fontsize ='20')
        axs[i].plot(ts.iloc[:,i])
        
    plt.show()

def find_motif(ts, m, plot=True):
    """
    Find motifs and plot themin a time series.
    
    Args:
        ts (pd.DataFrame): Time series data
        m (int): Window size for motif detection
        plot (bool): Whether to plot the motifs
        
    Returns:
        tuple: (mps, motifs_idx) where mps is a dictionary of matrix profiles and motifs_idx is a dictionary of motif index locations
    """
    mps = {}  # Store the 1-dimensional matrix profiles
    motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)
    for dim_name in ts.columns:
        mps[dim_name] = stumpy.stump(ts[dim_name], m)
        motif_distance = np.round(mps[dim_name][:, 0].astype(float).min(), 1)
        print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
        motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]

    if plot:
        fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})
        for i, dim_name in enumerate(list(mps.keys())):
            axs[i].set_ylabel(dim_name, fontsize='8')
            axs[i].plot(ts[dim_name])
            axs[i].set_xlabel('Step', fontsize ='20')
            for idx in motifs_idx[dim_name]:
                axs[i].plot(ts[dim_name].iloc[idx:idx+m], c='red', linewidth=4)
                axs[i].axvline(x=idx, linestyle="dashed", c='black')
            
        plt.show()
    return mps, motifs_idx


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
    where num_features = 4 (Pacman_X, Pacman_Y, score, powerPellets)
    
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
    
    return tensor, mask, game_ids


# The resulting tensor will have:
# - First dimension: different games
# - Second dimension: timesteps in the game
# - Third dimension: features (X, Y, score, powerPellets)

def preprocess_game_data(df, series_type=['position'], include_game_state_vars= False, include_timesteps = True):
    """
    Preprocess gamestates' data before converting to tensor.
    
    Args:
        df: DataFrame containing raw game data
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
        df['movement_dx'] = df['movement_direction'].map(lambda d: direction_mapping[d][0])
        df['movement_dy'] = df['movement_direction'].map(lambda d: direction_mapping[d][1])
        features.extend(['movement_dx', 'movement_dy'])
    
    if 'input' in series_type:
        # Similarly for input directions
        df['input_dx'] = df['input_direction'].map(lambda d: direction_mapping[d][0])
        df['input_dy'] = df['input_direction'].map(lambda d: direction_mapping[d][1])
        features.extend(['input_dx', 'input_dy'])
    
    processed_df = df[features].copy()
    
    # Convert to float
    for col in processed_df.columns:
        if col != 'game_id':  # Skip game_id conversion
            processed_df[col] = processed_df[col].astype(float)
    
    return processed_df

