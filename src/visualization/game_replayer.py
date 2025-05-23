from src.utils import utils
from src.utils import Astar
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation

import logging
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class GameReplayer:
    def __init__(
        self,
        data,
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
        self.pellet_objects = []  # Store pellet objects
        self.eaten_pellets = set()  # Track eaten pellets

        # Astar paths
        self.pathfinding = pathfinding

    def restart_animation(self):
        if self.anim:
            self.is_finalized = False
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
            self.pellet_objects = []
            for x, y in self.pellet_positions:
                x = x + 0.5  # Translation to the center of the grid cell.
                y = y - 0.5
                pellet = ax_game.plot([x], [y], "o", color="white", markersize=2)[0]
                self.pellet_objects.append((pellet, (x, y)))

        (pacman_dot,) = ax_game.plot([], [], "o", color="yellow", label="Pac-Man")

        if self.ghosts_columns:
            ghost_colors = ["red", "pink", "cyan", "orange"]
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
            for ghost_dot in ghost_dots:
                ghost_dot.set_data([], [])

            for path_line in path_lines:
                path_line.set_data([], [])

            for text_object in stats_text_objects:
                text_object.set_text("")

            return [pacman_dot, *ghost_dots, *path_lines, *stats_text_objects]

        def update(frame):
            if not self.is_playing:
                return [
                    pacman_dot,
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                    *[p[0] for p in self.pellet_objects],
                ]

            try:
                row = self.data.iloc[frame]
                pacman_x, pacman_y = row["Pacman_X"], row["Pacman_Y"]
                pacman_dot.set_data([pacman_x], [pacman_y])

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

                if self.pellets:
                    # Check for pellet collisions
                    # TODO Change logic. Do pellet preprocessing and get values from dataframe object.
                    for pellet, (x, y) in self.pellet_objects:
                        distance = ((x - pacman_x) ** 2 + (y - pacman_y) ** 2) ** 0.5
                        if distance < 0.625 and (x, y) not in self.eaten_pellets:
                            self.logger.debug(
                                f"Eating pellet at {x}, {y}. Distance: {distance:.2f}"
                            )
                            pellet.set_visible(False)
                            self.eaten_pellets.add((x, y))

                if self.ghosts_columns:
                    for i, ghost_dot in enumerate(ghost_dots):
                        ghost_dot.set_data(
                            [row[f"Ghost{i + 1}_X"]], [row[f"Ghost{i + 1}_Y"]]
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
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                    *[p[0] for p in self.pellet_objects],
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
                    *ghost_dots,
                    *stats_text_objects,
                    *path_lines,
                    *[p[0] for p in self.pellet_objects],
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
                        # Reset pellets for new game
                        self.eaten_pellets.clear()
                        for pellet, _ in self.pellet_objects:
                            pellet.set_visible(True)
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
            if game_selector:
                game_selector.on_submit(on_game_select)
            play_button.on_clicked(on_play_pause)
            restart_button.on_clicked(on_restart)

        # Calculate the interval based on the time elapsed between frames
        if (
            "time_elapsed" in self.data.columns
            and not self.data["time_elapsed"].isnull().all()
        ):
            interval = (
                self.data["time_elapsed"].diff().mean() / self.playback_speed
            ) * 1000
        else:
            interval = (
                50 / self.playback_speed
            )  # Assume 50 milliseconds between rows if time_elapsed is not in the data

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
                    fps=1000 / interval,
                    dpi=300,
                    bitrate=2000,
                    codec="libx264",
                )
            elif save_format.lower() == "gif":
                self.anim.save(save_path, writer="pillow", fps=1000 / interval)
        else:
            plt.show()
