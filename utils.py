import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox


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


class SessionVisualizer:
    def __init__(self, data, playback_speed=1.0, columns=None):
        """
        Initialize the visualizer with data.
        
        Args:
            data (DataFrame): Game state data with columns [time, Pacman_X, Pacman_Y, Ghost1_X, Ghost1_Y, ...].
            playback_speed (float): Initial speed multiplier for the playback.
            columns (list): List of columns to use for the animation. If None, the default columns will be used.
        """
        self.full_data = data
        self.default_columns = ['game_id','time_elapsed', 'Pacman_X', 'Pacman_Y', 'Ghost1_X', 'Ghost1_Y',
                              'Ghost2_X', 'Ghost2_Y', 'Ghost3_X', 'Ghost3_Y', 'Ghost4_X', 'Ghost4_Y']
        self.columns = columns if columns else self.default_columns
        self.playback_speed = playback_speed
        self.anim = None
        self.is_playing = False
        self.wall_positions, self.pellet_positions = load_maze_data()
        self.pellet_objects = []  # Store pellet objects
        self.eaten_pellets = set()  # Track eaten pellets



    def restart_animation(self):
        if self.anim:
            self.anim.frame_seq = self.anim.new_frame_seq()
            self.anim.event_source.start()
        
    def animate_session(self):
        """
        Creates an interactive animation window with controls.
        """
        # Create the main figure and grid layout
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[10, 1], hspace=0.1)
        
        # Game display axis
        ax_game = fig.add_subplot(gs[0])
        ax_game.set_xlim(-15, 15)
        ax_game.set_ylim(-18, 15)
        ax_game.set_aspect('equal')
        ax_game.set_facecolor('black')

        ax_game.set_xticks([])
        ax_game.set_yticks([])

        # Controls axis
        ax_controls = fig.add_subplot(gs[1])
        ax_controls.set_visible(False)  # Hide the actual axis
        
         # Initialize plot elements
         # Add walls and pellets
        for x, y in self.wall_positions:
            ax_game.add_patch(plt.Rectangle((x-0, y-1), 1, 1, 
                                          color='blue', alpha=0.5))
        
        self.pellet_objects = []
        for x, y in self.pellet_positions:
            pellet = ax_game.plot(x+0.5, y-0.5, 'o', color='white', markersize=2)[0]
            self.pellet_objects.append((pellet, (x, y)))

        pacman_dot, = ax_game.plot([], [], 'o', color='yellow', label='Pac-Man')
        ghost_dots = [ax_game.plot([], [], 'o', label=f'Ghost {i+1}')[0] for i in range((len(self.columns) - 4) // 2)]


        
        # Add game ID text label
        game_id_text = ax_game.text(0.02, 0.98, '', transform=ax_game.transAxes,
                                   verticalalignment='top',
                                   fontsize=10,
                                   color='white')
        
        # Add time elapsed text label
        time_elapsed_text = ax_game.text(0.02, 0.92, '', transform=ax_game.transAxes,
                                   verticalalignment='top',
                                   fontsize=10,
                                   color='white')
        
        # Game Selector
        game_ids = sorted(self.full_data['game_id'].unique())
        game_selector_ax = plt.axes([0.2, 0.15, 0.2, 0.03])  # Adjust size for text box
        game_selector = TextBox(game_selector_ax, 'Game ID: ', 
                              initial=str(game_ids[0]))  # Start with first game ID
        
        # Add play/pause and restart buttons
        play_button_ax = plt.axes([0.5, 0.15, 0.1, 0.04])
        play_button = plt.Button(play_button_ax, 'Play/Pause')
        
        restart_button_ax = plt.axes([0.65, 0.15, 0.1, 0.04])
        restart_button = plt.Button(restart_button_ax, 'Restart')
        
        def init():
            pacman_dot.set_data([], [])
            for ghost_dot in ghost_dots:
                ghost_dot.set_data([], [])
            game_id_text.set_text('')
            time_elapsed_text.set_text('')
            return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text]
        
        def update(frame):
            if not self.is_playing:
                return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, *[p[0] for p in self.pellet_objects]]
            
            try:
                row = self.data.iloc[frame]
                pacman_x, pacman_y = row['Pacman_X'], row['Pacman_Y']
                pacman_dot.set_data(pacman_x, pacman_y)
                
                # Check for pellet collisions
                for pellet, (x, y) in self.pellet_objects:
                    pellet_x = x + 0.5
                    pellet_y = y - 0.5
                    
                    distance = ((pellet_x - pacman_x)**2 + (pellet_y - pacman_y)**2)**0.5
                    if distance < 1.0 and (x, y) not in self.eaten_pellets:
                        print(f"Eating pellet at {x}, {y}. Distance: {distance:.2f}")
                        pellet.set_visible(False)
                        self.eaten_pellets.add((x, y))
                
                for i, ghost_dot in enumerate(ghost_dots):
                    ghost_dot.set_data(row[f'Ghost{i+1}_X'], row[f'Ghost{i+1}_Y'])

                current_game_id = row['game_id']
                game_id_text.set_text(f'Game ID: {current_game_id}')

                time_elapsed = row['time_elapsed']
                time_elapsed_text.set_text(f'Time Elapsed: {time_elapsed:.2f}')
                
                # Return all artists that need to be redrawn
                return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, *[p[0] for p in self.pellet_objects]]

            except IndexError:
                print(f"Frame {frame} out of bounds for data length {len(self.data)}")
                self.is_playing = False
                return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, *[p[0] for p in self.pellet_objects]]
        
        def on_game_select(text):
            try:
                game_id = int(text)
                if game_id in game_ids:
                    self.data = self.full_data[self.columns].loc[self.full_data['game_id'] == game_id]
                    # Reset pellets for new game
                    self.eaten_pellets.clear()
                    for pellet, _ in self.pellet_objects:
                        pellet.set_visible(True)
                    self.restart_animation()
                else:
                    print(f"Game ID {game_id} not found. Available IDs: {game_ids}")
            except ValueError:
                print("Please enter a valid game ID number")
        
        
        def on_play_pause(event):
            self.is_playing = not self.is_playing
        
        def on_restart(event):
            self.eaten_pellets.clear()
            for pellet, _ in self.pellet_objects:
                pellet.set_visible(True)
            self.restart_animation()
        
        def restart_animation():
            if self.anim:
                self.anim.frame_seq = self.anim.new_frame_seq()
                self.anim.event_source.start()
        
        # Connect callbacks
        game_selector.on_submit(on_game_select)
        play_button.on_clicked(on_play_pause)
        restart_button.on_clicked(on_restart)
        
        # Start with the first game
        self.data = self.full_data[self.columns].loc[self.full_data['game_id'] == game_ids[0]]
        interval = (self.data['time_elapsed'].diff().mean() / self.playback_speed) * 1000
        self.anim = FuncAnimation(fig, update, frames=len(self.data), init_func=init,
                                interval=interval, blit=True, repeat=True)
        plt.show()
