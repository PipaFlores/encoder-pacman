from utils import load_maze_data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation
import argparse
import Astar
import logging

class SessionVisualizer:
    def __init__(self, data, game_id=None, playback_speed=1.0, columns=None, verbose = False, pathfinding=True, pellets=True, show_grid=False):
        """
        Initialize the visualizer with data.
        
        Args:
            data (DataFrame): Game state data with columns [time, Pacman_X, Pacman_Y, Ghost1_X, Ghost1_Y, ...].
            game_id (int): Game ID to visualize. If None, the first game ID will be used.
            playback_speed (float): Initial speed multiplier for the playback.
            columns (list): List of columns to use for the animation. If None, the default columns will be used.
            verbose (bool): Whether to print verbose logging.
            pathfinding (bool): Whether to calculate A* pathfinding.
            pellets (bool): Whether to show pellets.
            show_grid (bool): Whether to show the grid.
        """
        self.full_data = data
        self.game_id = game_id
        self.default_columns = ['game_state_id', 'game_id', 'time_elapsed', 'Pacman_X', 'Pacman_Y', 'Ghost1_X', 'Ghost1_Y',
                              'Ghost2_X', 'Ghost2_Y', 'Ghost3_X', 'Ghost3_Y', 'Ghost4_X', 'Ghost4_Y', 'score', 'lives', 'level']
        self.columns = columns if columns else self.default_columns

        # Animation
        self.playback_speed = playback_speed
        self.anim = None
        self.is_playing = True
        self.finalized = False

        # Maze
        self.show_grid = show_grid
        self.pellets = pellets
        self.wall_positions, self.pellet_positions = load_maze_data()
        self.wall_grid = Astar.generate_squared_walls(self.wall_positions)
        self.pellet_objects = []  # Store pellet objects
        self.eaten_pellets = set()  # Track eaten pellets

        # Astar paths
        self.pathfinding = pathfinding

        # Logging
        self.logger = logging.getLogger('SessionVisualizer')
        logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)




    def restart_animation(self):
        if self.anim:
            self.is_finalized = False
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

        
        if self.show_grid:
            ax_game.grid(True, color='white', alpha=0.3, linestyle='-', linewidth=1)
            ax_game.set_xticks(range(-15, 16))
            ax_game.set_yticks(range(-18, 16))


        # Controls axis
        ax_controls = fig.add_subplot(gs[1])
        ax_controls.set_visible(False)  # Hide the actual axis
        
         # Initialize plot elements
         # Add walls and pellets
        for x, y in self.wall_positions:
            ax_game.add_patch(plt.Rectangle((x, y), 1, -1,  # Negative height to assimilate unity's drawing logic.
                                          color='blue', alpha=0.5)) 
        
        if self.pellets:
            self.pellet_objects = []
            for x, y in self.pellet_positions:
                x = x + 0.5 # Translation to the center of the grid cell.
                y = y - 0.5
                pellet = ax_game.plot([x], [y], 'o', color='white', markersize=2)[0]
                self.pellet_objects.append((pellet, (x, y)))

        pacman_dot, = ax_game.plot([], [], 'o', color='yellow', label='Pac-Man')

        ghost_colors = ['red', 'pink', 'cyan', 'orange']
        ghost_dots = [ax_game.plot([], [], 'o', color=ghost_colors[i], label=f'Ghost {i+1}')[0] for i in range(4)]

        path_lines = [ax_game.plot([], [], 'o', color=ghost_colors[i], alpha=0.5, markersize=2)[0] for i in range(len(ghost_dots))]
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

        score_text = ax_game.text(0.02, 0.86, '', transform=ax_game.transAxes,
                                   verticalalignment='top',
                                   fontsize=10,
                                   color='white')
        
        lives_text = ax_game.text(0.02, 0.80, '', transform=ax_game.transAxes,
                                   verticalalignment='top',
                                   fontsize=10,
                                   color='white')
        
        level_text = ax_game.text(0.02, 0.74, '', transform=ax_game.transAxes,
                                   verticalalignment='top',
                                   fontsize=10,
                                   color='white')
        

        
        # Game Selector
        game_ids = sorted(self.full_data['game_id'].unique())
        game_selector_ax = plt.axes([0.2, 0.1, 0.2, 0.03])  # Adjust size for text box
        game_selector = TextBox(game_selector_ax, 'Game ID: ', 
                              initial=str(self.game_id if self.game_id else game_ids[0]))  # Start with first game ID if none is provided
        
        # Add play/pause and restart buttons
        play_button_ax = plt.axes([0.5, 0.1, 0.1, 0.04])
        play_button = plt.Button(play_button_ax, 'Play/Pause')
        
        restart_button_ax = plt.axes([0.65, 0.1, 0.1, 0.04])
        restart_button = plt.Button(restart_button_ax, 'Restart')
        
        def init():
            pacman_dot.set_data([], [])
            for ghost_dot in ghost_dots:
                ghost_dot.set_data([], [])

            for path_line in path_lines:
                path_line.set_data([], [])

            game_id_text.set_text('')
            time_elapsed_text.set_text('')
            score_text.set_text('')
            lives_text.set_text('')
            level_text.set_text('')

            return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, score_text, lives_text, level_text, *path_lines]
        
        def update(frame):
            if not self.is_playing:
                return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, score_text, lives_text, level_text, *path_lines,*[p[0] for p in self.pellet_objects]]
            

            try:


                row = self.data.iloc[frame]
                pacman_x, pacman_y = row['Pacman_X'], row['Pacman_Y']
                pacman_dot.set_data([pacman_x], [pacman_y])
                ghost_positions = [(row[f'Ghost{i+1}_X'], row[f'Ghost{i+1}_Y']) for i in range(4)]

                if self.pathfinding:
                    self.logger.debug(f'Calculating paths for gamestate {row["game_state_id"]} at frame {frame}')
                    self.logger.debug(f'Pacman position: {pacman_x}, {pacman_y}')
                    self.logger.debug(f'Ghost positions: {ghost_positions}')
                    results = Astar.calculate_ghost_paths_and_distances((pacman_x, pacman_y), ghost_positions, self.wall_grid)
                    self.logger.debug(f'Distance to red: {results[0][1]}')
                    paths = [result[0] for result in results]
                    # Draw paths
                    for path_line, path in zip(path_lines, paths):
                        if path:
                            path_x, path_y = zip(*path)
                            path_x = [x for x in path_x] # Transform to unity reference.
                            path_y = [y for y in path_y]
                            path_line.set_data(path_x, path_y) 
                        else:
                            path_line.set_data([], [])

                if self.pellets:
                    # Check for pellet collisions
                    for pellet, (x, y) in self.pellet_objects:

                        distance = ((x - pacman_x)**2 + (y - pacman_y)**2)**0.5
                        if distance < 0.625 and (x, y) not in self.eaten_pellets:
                            self.logger.debug(f"Eating pellet at {x}, {y}. Distance: {distance:.2f}")
                            pellet.set_visible(False)
                            self.eaten_pellets.add((x, y))
                
                for i, ghost_dot in enumerate(ghost_dots):
                    ghost_dot.set_data([row[f'Ghost{i+1}_X']], [row[f'Ghost{i+1}_Y']])

                current_game_id = row['game_id']
                game_id_text.set_text(f'Game ID: {int(current_game_id)}')

                time_elapsed = row['time_elapsed']
                time_elapsed_text.set_text(f'Time Elapsed: {time_elapsed:.2f}')

                score_text.set_text(f'Score: {int(row["score"])}')
                lives_text.set_text(f'Lives: {int(row["lives"])}')

                level_text.set_text(f'Level: {int(row["level"])}')

                if frame == len(self.data) - 2:
                    self.anim.event_source.stop()
                    self.finalized = True
                    self.is_playing = False
                # Return all artists that need to be redrawn
                artists = [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, score_text, lives_text, level_text, *[p[0] for p in self.pellet_objects]]
                for path_line in path_lines:
                    if path_line is not None:
                        artists.append(path_line)
                return artists

            except IndexError:
                print(f"Frame {frame} out of bounds for data length {len(self.data)} - Probably repeat set to False")
                self.anim.event_source.stop()
                self.is_playing = False
                self.finalized = True
                return [pacman_dot, *ghost_dots, game_id_text, time_elapsed_text, score_text, lives_text, level_text, *[p[0] for p in self.pellet_objects]]
        
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
            if self.is_playing:
                self.anim.event_source.stop()
            elif not self.finalized:
                self.anim.event_source.start()
            else:
                self.eaten_pellets.clear()
                for pellet, _ in self.pellet_objects:
                    pellet.set_visible(True)
                self.restart_animation()
            self.is_playing = not self.is_playing

        def on_restart(event):
            self.eaten_pellets.clear()
            for pellet, _ in self.pellet_objects:
                pellet.set_visible(True)
            self.restart_animation()
        
        
        # Connect callbacks
        game_selector.on_submit(on_game_select)
        play_button.on_clicked(on_play_pause)
        restart_button.on_clicked(on_restart)
        
        # Start with the game_id provided by the user, else start with the first game
        self.data = self.full_data[self.columns].loc[self.full_data['game_id'] == (self.game_id if self.game_id else game_ids[0])]
        interval = (self.data['time_elapsed'].diff().mean() / self.playback_speed) * 1000
        self.anim = FuncAnimation(fig, update, frames=len(self.data), init_func=init,
                                interval=interval, blit=True, repeat=False)

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pac-Man game states")
    parser.add_argument('-g', '--game-id', type=int, default=None, help="Game ID to visualize")
    parser.add_argument('--playback-speed', type=float, default=2.0, help="Playback speed multiplier")
    parser.add_argument('--data-path', type=str, default='data/gamestate.csv', help="Path to the game state data")
    parser.add_argument('--gamedata-path', type=str, default='data/game.csv', help="Path to the game data")
    parser.add_argument('--no-pellets', action='store_false', dest='pellets', help="Disable pellets")
    parser.add_argument('--no-pathfinding', action='store_false', dest='pathfinding', help="Disable A* path finding calculation")
    parser.add_argument('--verbose', action='store_true', default=False, help="Print verbose logging")
    parser.add_argument('--grid',action= 'store_true', default=False, help="Show grid")
    args = parser.parse_args()
    gamestate_df = pd.read_csv(args.data_path, converters={'user_id': lambda x: int(x)})
    game_df = pd.read_csv(args.gamedata_path, converters={'user_id': lambda x: int(x)})
    gamestate_df = pd.merge(gamestate_df, game_df, on='game_id', how='left')
    visualizer = SessionVisualizer(gamestate_df, game_id=args.game_id, playback_speed=args.playback_speed, verbose=args.verbose, pellets=args.pellets, pathfinding=args.pathfinding, show_grid=args.grid)
    visualizer.animate_session()
