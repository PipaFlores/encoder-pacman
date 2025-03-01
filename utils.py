import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox
import functools
import time
import stumpy
import numpy as np

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

def read_data():
    """
    Read all data from CSV files.

    Returns:
        tuple: (user_df, ip_df, redcap_df, game_df, gamestate_df, psychometrics_df)
        user_df: DataFrame with user data
        ip_df: DataFrame with IP data
        redcap_df: DataFrame with REDCap data
        game_df: DataFrame with game data
        gamestate_df: DataFrame with gamestate data
        psychometrics_df: DataFrame with psychometrics data
    """
    user_df = pd.read_csv('data/user.csv')
    ip_df = pd.read_csv('data/userip.csv')
    redcap_df = pd.read_csv('data/redcapdata.csv')
    game_df = pd.read_csv('data/game.csv', converters={'date_played': lambda x: pd.to_datetime(x)})
    game_df = game_df[game_df['user_id'] != 42] # Remove user 42 (myself)
    # game_df = game_df[game_df['user_id'] != 47]
    gamestate_df = pd.read_csv('data/gamestate.csv', converters={'user_id': lambda x: int(x)})
    gamestate_df = gamestate_df[~gamestate_df['game_id'].isin(game_df.loc[game_df['user_id'] == 42, 'game_id'])] # Remove games associated with userid 42 (myself)
    psychometrics_df = pd.read_csv('data/psych/psych.csv')

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

def load_maze_data(walls_file='grid/walls.unity', pellets_file= 'grid/pellets.unity'):
    """
    Load wall and pellet positions from Unity tilemap files.
    
    Args:
        walls_file (str): Path to walls.unity file
        pellets_file (str): Path to pellets.unity file
        
    Returns:
        tuple: (wall_positions, pellet_positions) where each is a list of (x,y) tuples
    """
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
