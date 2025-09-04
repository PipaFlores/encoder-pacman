from src.utils import utils
from src.utils import Astar
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation

import logging
import os
import subprocess

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class GameReplayer:
    def __init__(
        self,
        data: pd.DataFrame = None,
        level_id=None,
        playback_speed=1.0,
        verbose=False,
        pathfinding=True,
        pellets=True,
        show_grid=False,
    ):
        """
        Initialize the visualizer with data.

        Args:
            data (DataFrame): Game state data with columns ['game_state_id','Pacman_X', 'Pacman_Y', 'Ghost1_X', 'Ghost1_Y',
                              'Ghost2_X', 'Ghost2_Y', 'Ghost3_X', 'Ghost3_Y', 'Ghost4_X', 'Ghost4_Y',
                              'level_id', 'time_elapsed', 'score', 'lives', 'level', 'movement_direction', 'input_direction'].
            level_id (int): Level ID to visualize. If None, the first level ID will be used.
            playback_speed (float): Initial speed multiplier for the playback.
            stats_columns (list): List of columns to use for the stats panel. If None, the default columns will be used.
            verbose (bool): Whether to print verbose logging.
            pathfinding (bool): Whether to calculate A* pathfinding.
            pellets (bool): Whether to show pellets.
            show_grid (bool): Whether to show the grid.

        Example:
            The class will work with only pacman data (without ghosts or other metadata). However, the full visualization
            includes all 4 ghosts and the level metadata. For this, the ideal data input is the merge betwheen gamestates_df and
            level_df from the PacmanDataReader class
            ```python
                reader= PacmanDataReader(data='..data/')

                data = pd.merge(reader.gamestate_df, reader.level_df, left_on='level_id', right_index=True)

                ## include data filtering and/or slicing as desired (i.e., data.loc on user, level, or slice of game)

                Replayer = GameReplayer(
                data
                )

                Replayer.animate_session() # for interactive visualization, executing a python script from terminal
                Replayer.animate_session(save_path="replay.mp4") # For notebooks or other modular calls in analysis pipeline.


            ```
        """
        self.full_data = data
        self.level_id = level_id

        # Logging
        self.logger = logging.getLogger("SessionVisualizer")
        logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

        if data is not None:
            if all(col in data.columns for col in ["Pacman_X", "Pacman_Y"]):
                self.columns = ["Pacman_X", "Pacman_Y"]
            else:
                raise ValueError("Pacman_X and Pacman_Y columns must be in data")

            self.game_state_id_column = (
                ["game_state_id"] if "game_state_id" in data.columns else []
            )
            self.logger.debug(f"Game state id column, {self.game_state_id_column}")
            self.columns.extend(self.game_state_id_column)

            self.ghosts_columns = (
                [
                    "Ghost1_X",
                    "Ghost1_Y",
                    "Ghost2_X",
                    "Ghost2_Y",
                    "Ghost3_X",
                    "Ghost3_Y",
                    "Ghost4_X",
                    "Ghost4_Y",
                ]
                if "Ghost1_X" in data.columns
                else []
            )
            self.columns.extend(self.ghosts_columns)
            self.logger.debug(f"Ghosts columns, {self.ghosts_columns}")

            self.pellet_positions_column = (
                ["available_pellets", "available_powerpills"]
                if "available_pellets" in data.columns
                else None
            )
            self.columns.extend(self.pellet_positions_column)

            self.pacman_attack_column = (
                ["pacman_attack"] if "pacman_attack" in data.columns else None
            )
            self.columns.extend(self.pacman_attack_column)

            self.stats_columns = [
                col
                for col in [
                    "level_id",
                    "time_elapsed",
                    "score",
                    "lives",
                    "level",
                    "movement_direction",
                    "input_direction",
                ]
                if all(
                    col in data.columns
                    for col in [
                        "level_id",
                        "time_elapsed",
                        "score",
                        "lives",
                        "level",
                        "movement_direction",
                        "input_direction",
                    ]
                )
            ]
            self.columns = self.columns + self.stats_columns
            self.logger.debug(f"Stats columns, {self.stats_columns}")
            self.logger.debug(f"Columns, {self.columns}")

        # Animation
        self.playback_speed = playback_speed
        self.anim = None
        self.is_playing = True
        self.finalized = False

        # Maze
        self.show_grid = show_grid
        self.pellets = pellets
        self.wall_positions, self.pellet_positions = utils.load_maze_data()
        self.wall_grid = Astar.generate_squared_walls(self.wall_positions)

        # Astar paths
        self.pathfinding = pathfinding

    def restart_animation(self):
        if self.anim:
            self.finalized = False
            self.is_playing = True
            self.anim.frame_seq = self.anim.new_frame_seq()
            self.anim.event_source.start()

    def animate_session(
        self, save_path: str = None, save_format: str = "mp4", title: str = None
    ):
        """
        Creates an interactive animation window with controls.
        """
        # Start with the level_id provided by the user, else start with the first level
        if self.level_id:
            self.data = self.full_data[self.columns].loc[
                self.full_data["level_id"] == self.level_id
            ]
        else:
            self.data = self.full_data[self.columns]

        # Create the main figure and grid layout
        if save_path:
            # For saving, use a simpler layout without UI controls
            fig = plt.figure(figsize=(9, 6))
            if title:
                fig.suptitle(title, color="black", fontsize=12)
            ax_game = fig.add_subplot(111)  # Single subplot for game display
            ax_game.set_xlim(-15, 15)
            ax_game.set_ylim(-18, 15)
            ax_game.set_aspect("equal")
            ax_game.set_facecolor("black")

            if self.show_grid:
                ax_game.grid(True, color="white", alpha=0.3, linestyle="-", linewidth=1)
                ax_game.set_xticks(range(-15, 16))
                ax_game.set_yticks(range(-18, 16))

            # Initialize empty lists for UI elements that won't be used
            stats_text_objects = []
            game_selector = None
            play_button = None
            restart_button = None
        else:
            # Original interactive layout with UI controls
            fig = plt.figure(figsize=(9, 6))
            gs = fig.add_gridspec(
                2, 2, width_ratios=[3, 1], height_ratios=[10, 1], hspace=0.1
            )

            # Game display axis
            ax_game = fig.add_subplot(gs[0, 0])
            ax_game.set_xlim(-15, 15)
            ax_game.set_ylim(-18, 15)
            ax_game.set_aspect("equal")
            ax_game.set_facecolor("black")

            # Stats panel (right side)
            ax_stats = fig.add_subplot(gs[0, 1])
            ax_stats.set_facecolor("black")
            ax_stats.set_xticks([])
            ax_stats.set_yticks([])

            if self.show_grid:
                ax_game.grid(True, color="white", alpha=0.3, linestyle="-", linewidth=1)
                ax_game.set_xticks(range(-15, 16))
                ax_game.set_yticks(range(-18, 16))

            # Controls axis
            ax_controls = fig.add_subplot(gs[1, :])
            ax_controls.set_visible(False)  # Hide the actual axis

            # Stats panel
            stats_text_objects = []
            if self.stats_columns:
                for i, column in enumerate(self.stats_columns):
                    stats_text_objects.append(
                        ax_stats.text(
                            0.1, 0.9 - i * 0.1, "", color="white", fontsize=10
                        )
                    )

            # Game Selector
            if "level_id" in self.full_data.columns:
                level_ids = sorted(self.full_data["level_id"].unique())
                game_selector_ax = plt.axes([0.2, 0.05, 0.2, 0.03])  # Move down
                game_selector = TextBox(
                    game_selector_ax,
                    "Level ID: ",
                    initial=str(self.level_id if self.level_id else level_ids[0]),
                )  # Start with first game ID if none is provided
            else:
                game_selector = None

            # Add play/pause and restart buttons
            play_button_ax = plt.axes([0.5, 0.1, 0.1, 0.04])
            play_button = plt.Button(play_button_ax, "Play/Pause")

            restart_button_ax = plt.axes([0.65, 0.1, 0.1, 0.04])
            restart_button = plt.Button(restart_button_ax, "Restart")

        # Initialize plot elements
        # Add walls and pellets
        for x, y in self.wall_positions:
            ax_game.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    -1,  # Negative height to assimilate unity's drawing logic.
                    color="blue",
                    alpha=0.5,
                )
            )

        if self.pellets:
            (remaining_pellets,) = ax_game.plot(
                [], [], "o", color="white", markersize=2
            )
            (remaining_powerpills,) = ax_game.plot(
                [], [], "o", color="white", markersize=5
            )

        (pacman_dot,) = ax_game.plot([], [], "o", color="yellow", label="Pac-Man")

        if self.ghosts_columns:
            ghost_colors = ["red", "pink", "cyan", "orange"]
            self.flash_counter = 0
            self.flash = False
            self.flash_color = "white"
            ghost_dots = [
                ax_game.plot(
                    [], [], "o", color=ghost_colors[i], label=f"Ghost {i + 1}"
                )[0]
                for i in range(4)
            ]

            path_lines = [
                ax_game.plot(
                    [], [], "o", color=ghost_colors[i], alpha=0.5, markersize=2
                )[0]
                for i in range(len(ghost_dots))
            ]
        else:
            ghost_dots = []
            path_lines = []

        def init():
            pacman_dot.set_data([], [])
            remaining_pellets.set_data([], [])
            remaining_powerpills.set_data([], [])
            for ghost_dot in ghost_dots:
                ghost_dot.set_data([], [])

            for path_line in path_lines:
                path_line.set_data([], [])

            for text_object in stats_text_objects:
                text_object.set_text("")

            return [
                pacman_dot,
                remaining_pellets,
                remaining_powerpills,
                *ghost_dots,
                *path_lines,
                *stats_text_objects,
            ]

        def update(frame):
            if not self.is_playing:
                return [
                    pacman_dot,
                    remaining_pellets,
                    remaining_powerpills,
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                    # *[p[0] for p in self.pellet_objects],
                ]

            try:
                row = self.data.iloc[frame]
                pacman_x, pacman_y = row["Pacman_X"], row["Pacman_Y"]
                pacman_dot.set_data([pacman_x], [pacman_y])

                available_pellets = row["available_pellets"]
                remaining_pellets.set_data(
                    available_pellets[:, 0], available_pellets[:, 1]
                )

                available_powerpills = row["available_powerpills"]
                remaining_powerpills.set_data(
                    [p[0] for p in available_powerpills],
                    [p[1] for p in available_powerpills],  # to solve weird TypeError
                )

                if self.ghosts_columns:
                    ghost_positions = [
                        (row[f"Ghost{i + 1}_X"], row[f"Ghost{i + 1}_Y"])
                        for i in range(4)
                    ]

                if self.pathfinding and self.ghosts_columns:
                    if "game_state_id" in self.data.columns:
                        self.logger.debug(
                            f"Calculating paths for gamestate {row['game_state_id']} at frame {frame}"
                        )
                    else:
                        self.logger.debug(f"Calculating paths for row index {frame}")
                    self.logger.debug(f"Pacman position: {pacman_x}, {pacman_y}")
                    self.logger.debug(f"Ghost positions: {ghost_positions}")
                    results = Astar.calculate_ghost_paths_and_distances(
                        (pacman_x, pacman_y), ghost_positions, self.wall_grid
                    )
                    self.logger.debug(f"Distance to red: {results[0][1]}")
                    paths = [result[0] for result in results]
                    # Draw paths
                    for path_line, path in zip(path_lines, paths):
                        if path:
                            path_x, path_y = zip(*path)
                            path_x = [
                                x for x in path_x
                            ]  # Transform to unity reference.
                            path_y = [y for y in path_y]
                            path_line.set_data(path_x, path_y)
                        else:
                            path_line.set_data([], [])

                if self.ghosts_columns:
                    if row["pacman_attack"] == 1:
                        self.flash_counter += 1
                        self.flash = True
                        if self.flash_counter % 4 == 0:
                            self.flash_color = (
                                "blue" if self.flash_color == "white" else "white"
                            )
                    else:
                        self.flash = False

                    for i, ghost_dot in enumerate(ghost_dots):
                        ghost_dot.set_data(
                            [row[f"Ghost{i + 1}_X"]], [row[f"Ghost{i + 1}_Y"]]
                        )
                        ghost_dot.set_color(
                            self.flash_color if self.flash else ghost_colors[i]
                        )

                if (
                    self.stats_columns and not save_path
                ):  # Only update stats if not saving
                    current_game_id = row["level_id"]
                    stats_text_objects[0].set_text(f"Level ID: {int(current_game_id)}")

                    time_elapsed = row["time_elapsed"]
                    stats_text_objects[1].set_text(f"Time Elapsed: {time_elapsed:.2f}")

                    stats_text_objects[2].set_text(f"Score: {int(row['score'])}")
                    stats_text_objects[3].set_text(f"Lives: {int(row['lives'])}")

                    stats_text_objects[4].set_text(f"Level: {int(row['level'])}")

                    for i, column in enumerate(self.stats_columns[5:]):
                        stats_text_objects[i + 5].set_text(f"{column}: {row[column]}")

                if frame == len(self.data) - 2:
                    self.logger.debug(f"Final frame {frame}")
                    self.anim.event_source.stop()
                    self.finalized = True
                    self.is_playing = False

                return [
                    pacman_dot,
                    remaining_pellets,
                    remaining_powerpills,
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                ]

            except IndexError:
                print(
                    f"Frame {frame} out of bounds for data length {len(self.data)} - Probably repeat set to False"
                )
                self.anim.event_source.stop()
                self.is_playing = False
                self.finalized = True
                return [
                    pacman_dot,
                    remaining_pellets,
                    remaining_powerpills,
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                ]

        # Only set up callbacks if not saving
        if not save_path:

            def on_game_select(text):
                try:
                    level_id = int(text)
                    if level_id in level_ids:
                        self.data = self.full_data[self.columns].loc[
                            self.full_data["level_id"] == level_id
                        ]

                        self.restart_animation()
                    else:
                        print(
                            f"Level ID {level_id} not found. Available IDs: {level_ids}"
                        )
                except ValueError:
                    print("Please enter a valid level ID number")

            def on_play_pause(event):
                if self.is_playing:
                    self.anim.event_source.stop()
                    self.is_playing = False
                elif not self.finalized:
                    self.anim.event_source.start()
                    self.is_playing = True
                else:
                    self.restart_animation()

            def on_restart(event):
                self.restart_animation()

            # Connect callbacks
            if game_selector:
                game_selector.on_submit(on_game_select)
            play_button.on_clicked(on_play_pause)
            restart_button.on_clicked(on_restart)

        interval = (
                50 / self.playback_speed
            )  # Assume 50 milliseconds. Note: There is some sampling noise in real data, but should be neglectable.

        self.anim = FuncAnimation(
            fig,
            update,
            frames=len(self.data),
            init_func=init,
            interval=interval,
            blit=True,
            repeat=False,
        )

        if save_path:
            if save_format.lower() == "mp4":
                self.anim.save(
                    save_path,
                    writer="ffmpeg",
                    fps=20,
                    dpi=300,
                    bitrate=2000,
                    codec="libx264",
                    # progress_callback=lambda i, n: print(f'Saving frame {i}/{n}',
                )
            elif save_format.lower() == "gif":
                self.anim.save(save_path, writer="pillow", fps=20)
        else:
            plt.show()


    def animate_session_compact(self, save_path: str, save_format: str = "mp4", title: str = None):
        """
        Creates a compact video focused only on gameplay without UI elements or whitespace.
        Preserves all original animation features including ghost flashing.
        """
        # Prepare data
        if self.level_id:
            self.data = self.full_data[self.columns].loc[
                self.full_data["level_id"] == self.level_id
            ]
        else:
            self.data = self.full_data[self.columns]

        # Create compact figure - ONLY CHANGE: remove margins and axis decorations
        fig = plt.figure(figsize=(6, 6))
        ax_game = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
        
        ax_game.set_xlim(-15, 15)
        ax_game.set_ylim(-18, 15)
        ax_game.set_aspect("equal")
        ax_game.set_facecolor("black")
        
        # Remove all visual elements except game content
        ax_game.set_xticks([])
        ax_game.set_yticks([])
        ax_game.axis('off')
        
        # Set figure background
        fig.patch.set_facecolor('black')
        
        # Add title if specified
        if title:
            ax_game.text(0, 14, title, color="white", fontsize=10, 
                        ha='center', va='top', transform=ax_game.transData)
        
        # Initialize plot elements (same as original method)
        # Add walls and pellets
        for x, y in self.wall_positions:
            ax_game.add_patch(
                plt.Rectangle(
                    (x, y), 1, -1, color="blue", alpha=0.5
                )
            )

        if self.pellets:
            (remaining_pellets,) = ax_game.plot([], [], "o", color="white", markersize=2)
            (remaining_powerpills,) = ax_game.plot([], [], "o", color="white", markersize=5)

        (pacman_dot,) = ax_game.plot([], [], "o", color="yellow", markersize=8)

        if self.ghosts_columns:
            ghost_colors = ["red", "pink", "cyan", "orange"]
            self.flash_counter = 0
            self.flash = False
            self.flash_color = "white"
            ghost_dots = [
                ax_game.plot([], [], "o", color=ghost_colors[i], markersize=6)[0]
                for i in range(4)
            ]

            path_lines = [
                ax_game.plot([], [], "o", color=ghost_colors[i], alpha=0.5, markersize=2)[0]
                for i in range(len(ghost_dots))
            ]
        else:
            ghost_dots = []
            path_lines = []

        # EXACT COPY of original init function
        def init():
            pacman_dot.set_data([], [])
            remaining_pellets.set_data([], [])
            remaining_powerpills.set_data([], [])
            for ghost_dot in ghost_dots:
                ghost_dot.set_data([], [])
            for path_line in path_lines:
                path_line.set_data([], [])
            return [
                pacman_dot,
                remaining_pellets,
                remaining_powerpills,
                *ghost_dots,
                *path_lines,
            ]

        def update(frame):
            try:
                row = self.data.iloc[frame]
                pacman_x, pacman_y = row["Pacman_X"], row["Pacman_Y"]
                pacman_dot.set_data([pacman_x], [pacman_y])

                available_pellets = row["available_pellets"]
                remaining_pellets.set_data(
                    available_pellets[:, 0], available_pellets[:, 1]
                )

                available_powerpills = row["available_powerpills"]
                remaining_powerpills.set_data(
                    [p[0] for p in available_powerpills],
                    [p[1] for p in available_powerpills],
                )

                if self.ghosts_columns:
                    ghost_positions = [
                        (row[f"Ghost{i + 1}_X"], row[f"Ghost{i + 1}_Y"])
                        for i in range(4)
                    ]

                if self.pathfinding and self.ghosts_columns:
                    if "game_state_id" in self.data.columns:
                        self.logger.debug(
                            f"Calculating paths for gamestate {row['game_state_id']} at frame {frame}"
                        )
                    else:
                        self.logger.debug(f"Calculating paths for row index {frame}")
                    self.logger.debug(f"Pacman position: {pacman_x}, {pacman_y}")
                    self.logger.debug(f"Ghost positions: {ghost_positions}")
                    results = Astar.calculate_ghost_paths_and_distances(
                        (pacman_x, pacman_y), ghost_positions, self.wall_grid
                    )
                    self.logger.debug(f"Distance to red: {results[0][1]}")
                    paths = [result[0] for result in results]
                    # Draw paths
                    for path_line, path in zip(path_lines, paths):
                        if path:
                            path_x, path_y = zip(*path)
                            path_x = [x for x in path_x]
                            path_y = [y for y in path_y]
                            path_line.set_data(path_x, path_y)
                        else:
                            path_line.set_data([], [])

                if self.ghosts_columns:
                    # CRITICAL: Ghost flashing logic preserved!
                    if row["pacman_attack"] == 1:
                        self.flash_counter += 1
                        self.flash = True
                        if self.flash_counter % 4 == 0:
                            self.flash_color = (
                                "blue" if self.flash_color == "white" else "white"
                            )
                    else:
                        self.flash = False

                    for i, ghost_dot in enumerate(ghost_dots):
                        ghost_dot.set_data(
                            [row[f"Ghost{i + 1}_X"]], [row[f"Ghost{i + 1}_Y"]]
                        )
                        # CRITICAL: Ghost color flashing preserved!
                        ghost_dot.set_color(
                            self.flash_color if self.flash else ghost_colors[i]
                        )

                if frame == len(self.data) - 2:
                    self.logger.debug(f"Final frame {frame}")
                    self.anim.event_source.stop()
                    self.finalized = True
                    self.is_playing = False

                return [
                    pacman_dot,
                    remaining_pellets,
                    remaining_powerpills,
                    *ghost_dots,
                    *path_lines,
                ]

            except IndexError:
                print(
                    f"Frame {frame} out of bounds for data length {len(self.data)} - Probably repeat set to False"
                )
                self.anim.event_source.stop()
                self.is_playing = False
                self.finalized = True
                return [
                    pacman_dot,
                    remaining_pellets,
                    remaining_powerpills,
                    *ghost_dots,
                    *path_lines,
                ]

        
        interval = 50 / self.playback_speed

        self.anim = FuncAnimation(
            fig, update, frames=len(self.data), init_func=init,
            interval=interval, blit=True, repeat=False
        )

        # Save the animation - SAME AS ORIGINAL
        if save_format.lower() == "mp4":
            self.anim.save(
                save_path, writer="ffmpeg", fps=20,
                dpi=150, bitrate=2000, codec="libx264"
            )
        elif save_format.lower() == "gif":
            self.anim.save(save_path, writer="pillow", fps=20)
        
        plt.close(fig)  # Clean up


    def extract_gamestate_subsequence_ffmpeg(self, video_path: str, start_gamestate: int, end_gamestate: int, 
                                           output_path: str, output_format: str = "gif"):
        """
        Extract a subsequence of frames from a video file and save it as a GIF or other format using FFmpeg.

        This method leverages FFmpeg directly to achieve high-quality output, accurate frame rates, and correct color handling,
        which can be challenging with some Python-based video processing libraries.

        Args:
            video_path (str): Path to the source video file (e.g., .mp4) to extract from.
            start_gamestate (int): The starting frame index (0-based, corresponds to the game's frame number).
            end_gamestate (int): The ending frame index (exclusive).
            output_path (str): Path to save the extracted subsequence (e.g., .gif).
            output_format (str, optional): Output format, e.g., "gif". Default is "gif".

        Notes:
            - Assumes the video is recorded at 20 frames per second (fps).
            - Requires FFmpeg to be installed and available in the system PATH.
            - This method first generates a color palette for optimal GIF quality, then applies it to the output.
            - The resulting GIF will be scaled to 640 pixels wide (height is adjusted to maintain aspect ratio).
            - Temporary files (e.g., palette.png) are cleaned up automatically.
        """
        

        
        if not os.path.exists(video_path):
            raise ValueError(f"Source video file not found: {video_path}")
        
        # Calculate time offsets (assuming 20 fps)
        start_time = start_gamestate / 20.0
        duration = (end_gamestate - start_gamestate) / 20.0
        
        try:
            if output_format.lower() == "gif":
                # FFmpeg command for high-quality GIF with proper fps
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output file
                    '-ss', str(start_time),  # Start time
                    '-t', str(duration),     # Duration
                    '-i', video_path,        # Input file
                    '-vf', 'fps=20,scale=640:-1:flags=lanczos,palettegen=stats_mode=diff',  # Generate palette
                    '-f', 'image2pipe', '-vcodec', 'png', '-'  # Output to pipe
                ]
                
                # Generate palette first
                palette_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-i', video_path,
                    '-vf', 'fps=20,scale=640:-1:flags=lanczos,palettegen=stats_mode=diff',
                    'palette.png'
                ]
                
                subprocess.run(palette_cmd, check=True, capture_output=True)
                
                # Create GIF with custom palette
                gif_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-i', video_path,
                    '-i', 'palette.png',
                    '-lavfi', 'fps=20,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
                    output_path
                ]
                
                subprocess.run(gif_cmd, check=True, capture_output=True)
                
                # Clean up palette file
                if os.path.exists('palette.png'):
                    os.remove('palette.png')
                    
            elif output_format.lower() == "mp4":
                # FFmpeg command for MP4 with proper fps
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-i', video_path,
                    '-c:v', 'libx264',  # H.264 codec
                    '-r', '20',         # Output fps
                    '-crf', '23',       # Quality (lower = better)
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
            
            self.logger.info(f"Created {output_format.upper()} using FFmpeg: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed: {e}")
            raise ValueError(f"FFmpeg extraction failed. Make sure FFmpeg is installed.")
        except FileNotFoundError:
            self.logger.error("FFmpeg not found in PATH")
            raise ValueError("FFmpeg not found. Please install FFmpeg or use the OpenCV method.")

    def extract_multiple_subsequences_ffmpeg(self, video_path: str, gamestate_ranges: list, 
                                           output_dir: str, output_format: str = "gif", 
                                           name_prefix: str = "sequence"):
        """
        Extract multiple subsequences from a video in batch using FFmpeg.
        
        Args:
            video_path: Path to the source MP4 file
            gamestate_ranges: List of (start, end) tuples for gamestate indices
            output_dir: Directory to save output files
            output_format: Output format ("gif" or "mp4")
            name_prefix: Prefix for output filenames
            
        Returns:
            list: Paths to created subsequence videos
            
        Example:
            # Extract multiple interesting segments as GIFs
            ranges = [(0, 100), (200, 300), (500, 600)]
            paths = replayer.extract_multiple_subsequences_ffmpeg("replay.mp4", ranges, "segments/")
            
            # Extract as MP4s with custom naming
            paths = replayer.extract_multiple_subsequences_ffmpeg(
                "replay.mp4", ranges, "clips/", "mp4", "clip"
            )
        """

        
        if not os.path.exists(video_path):
            raise ValueError(f"Source video file not found: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        failed_extractions = []
        
        self.logger.info(f"Starting batch extraction of {len(gamestate_ranges)} subsequences...")
        
        for i, (start, end) in enumerate(gamestate_ranges):
            extension = "gif" if output_format.lower() == "gif" else "mp4"
            output_path = os.path.join(output_dir, f"{name_prefix}_{start:06d}_{end:06d}.{extension}")
            
            try:
                self.logger.info(f"Processing subsequence {i+1}/{len(gamestate_ranges)}: frames {start}-{end}")
                result_path = self.extract_gamestate_subsequence_ffmpeg(
                    video_path, start, end, output_path, output_format
                )
                created_files.append(result_path)
                self.logger.info(f"✓ Created: {os.path.basename(result_path)}")
                
            except Exception as e:
                error_msg = f"Failed to create subsequence {start}-{end}: {e}"
                self.logger.error(f"✗ {error_msg}")
                failed_extractions.append((start, end, str(e)))
        
        # Summary
        self.logger.info(f"Batch extraction complete:")
        self.logger.info(f"  ✓ Successfully created: {len(created_files)} files")
        if failed_extractions:
            self.logger.warning(f"  ✗ Failed extractions: {len(failed_extractions)}")
            for start, end, error in failed_extractions:
                self.logger.warning(f"    - Frames {start}-{end}: {error}")
        
        return created_files
  