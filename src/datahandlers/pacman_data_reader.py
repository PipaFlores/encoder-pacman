import pandas as pd
import json
import os
import numpy as np
import time
from src.datahandlers.trajectory import Trajectory
from src.utils import setup_logger, load_maze_data
from typing import Tuple, Dict

logger = setup_logger(__name__)


class PacmanDataReader:
    """
    A class for reading and processing Pacman game data.

    This class provides functionality to:
    - Read and filter Pacman game data from CSV files
    - Extract trajectories and game states
    - Filter data by game ID, user ID, and timesteps
    - Handle metadata and banned users

    The class implements the Singleton pattern to ensure only one instance exists,
    which helps manage memory usage when dealing with large datasets.

    Attributes:
        data_folder (str): Path to the folder containing game data files
        verbose (bool): Whether to enable verbose logging
        read_games_only (bool): Whether to read only game data or include additional information
        BANNED_USERS (list): List of user IDs to exclude from the data
        game_df (pd.DataFrame): DataFrame containing game metadata (encompassing one or more levels)
        level_df (pd.DataFrame): DataFrame containing level metadata
        gamestate_df (pd.DataFrame): DataFrame containing game state data
        user_df (pd.DataFrame): DataFrame containing user metadata
        session_df (pd.DataFrame): DataFrame containing session metadata

    Example:
    ```python
    # In .ipynb at ./notebooks/
    data = PacmanDataReader(data_folder="../data/")

    ## To get all trajectories as a list[Trajectory]
    all_trajs = []
    for game in data.game_df["level_id"].tolist():
        traj = data.get_trajectory(level_id=game, include_metadata=True)
        all_trajs.append(traj)

    ```

    """

    _instance = None
    BANNED_USERS = [42]
    BANNED_GAMES = [419]  ## Game with 600 second idle duration (bug)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PacmanDataReader, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        data_folder: str,
        read_games_only: bool = True,
        verbose: bool = False,
        debug: bool = False,
        process_pellet: bool = True,
    ):
        # Initialize basic attributes if not already initialized
        if not hasattr(self, "initialized"):
            self.data_folder = data_folder
            self.verbose = verbose
            self.initialized = True
            self.read_games_only = read_games_only
            self.process_pellet = process_pellet
            logger.info(
                f"Initializing PacmanDataReader with read_games_only: {read_games_only}"
            )
            self._init_logger(verbose, debug)
            self._read_data(read_games_only)  # Initial load with BANNED_USERS = [42]
        # If already initialized but requesting more data, load it
        elif not self.read_games_only and read_games_only:
            logger.info(
                "Warning: Instance already loaded with full data. Ignoring read_games_only=True."
            )
        elif self.read_games_only and not read_games_only:
            self._init_logger(verbose, debug)
            logger.info("Loading additional data as requested...")
            self._read_data(read_games_only)
            self.read_games_only = read_games_only

        elif not self.process_pellet and process_pellet:
            self.process_pellet = True
            self.gamestate_df["available_pellets"] = self.gamestate_df[
                "available_pellets"
            ].apply(lambda s: np.array(json.loads(s)))

    def _init_logger(self, verbose, debug):
        if verbose:
            logger.setLevel("INFO")
        elif debug:
            logger.setLevel("DEBUG")
            logger.debug("Debug mode enabled")
        else:
            logger.setLevel("WARNING")

    def _read_data(self, read_games_only: bool):
        """
        Initialize the dataframes.
        Reads the data from the data folder and filters out banned users.
        """
        time_start = time.time()
        logger.info(f"Reading game data in '{self.data_folder}'")
        self.game_df = pd.read_csv(
            os.path.join(self.data_folder, "game.csv"),
            converters={"date_played": lambda x: pd.to_datetime(x)},
        )
        gamestate_pkl_path = os.path.join(self.data_folder, "gamestate.pkl")
        gamestate_csv_path = os.path.join(self.data_folder, "gamestate.csv")
        if os.path.exists(gamestate_pkl_path):
            self.gamestate_df = pd.read_pickle(gamestate_pkl_path)
        else:
            self.gamestate_df = pd.read_csv(
                gamestate_csv_path,
                converters={
                    "user_id": lambda x: int(x)
                    #  , 'Pacman_X': lambda x: round(float(x), 2),
                    #   'Pacman_Y': lambda x: round(float(x), 2)
                },
            )

        self.gamestate_df.set_index(keys="game_state_id", drop=False, inplace=True)
        logger.info(f"Time taken to read game data: {time.time() - time_start} seconds")
        ## Filter banned users and games
        self.banned_game_ids = pd.concat(
            [
                self.game_df.loc[
                    self.game_df["user_id"].isin(self.BANNED_USERS), "game_id"
                ],
                pd.Series(self.BANNED_GAMES),
            ]
        ).unique()

        self.game_df = self.game_df[~self.game_df["game_id"].isin(self.banned_game_ids)]
        self.gamestate_df = self.gamestate_df[
            ~self.gamestate_df[
                "game_id" if "game_id" in self.gamestate_df.columns else "level_id"
            ].isin(self.banned_game_ids)
        ]

        ## Refactor game_df for analysis consistency.
        self.game_df, self.level_df, self.gamestate_df = self._restructure_game_data()

        ## If pellet positions have not been processed (using the .csv)
        if "available_pellets" not in self.gamestate_df.columns:
            logger.warning(
                "Calculating pellet positions for each game state in pacman game, estimated time of 10-15 minutes"
            )
            self.gamestate_df = self._process_pellet_positions()

            logger.warning("Processing logging bugs")
            self.gamestate_df = (
                self._process_logging_bugs()
            )  # Whatever bugs are encountered (e.g., attackmode at initial gamestates)
            self.gamestate_df.to_pickle(os.path.join(self.data_folder, "gamestate.pkl"))

        if not read_games_only:
            self.user_df = pd.read_csv(os.path.join(self.data_folder, "user.csv"))
            self.ip_df = pd.read_csv(os.path.join(self.data_folder, "userip.csv"))
            self.redcap_df = pd.read_csv(
                os.path.join(self.data_folder, "redcapdata.csv")
            )
            self.psychometrics_df = pd.read_csv(
                os.path.join(
                    self.data_folder, r"psych\AiPerCogPacman_DATA_2025-06-03_1037.csv"
                )
            )

            # Process psych data
            self.game_flow_df = self._process_flow()
            self.bisbas_df = self._process_bisbas()

    def _process_pellet_positions(self):
        """
        Processes and reconstructs the available pellet positions for each game state in the Pacman game.

        Intended for a single time use and then save it to a csv

        This method uses the initial pellet layout from the maze data and, for each game state,
        determines which pellets are still available based on Pacman's position and the number of pellets left.
        It creates a new column 'available_pellets' in the gamestate DataFrame, which contains the positions
        of all pellets that have not yet been eaten at each game state.

        Returns:
            pd.DataFrame: A copy of the gamestate DataFrame with an added 'available_pellets' column,
                          where each entry is an array of (x, y) positions of remaining pellets for that state.
        """

        MAZE_X_MIN: int = -13.5
        MAZE_X_MAX: int = 13.5
        MAZE_Y_MIN: int = -16.5
        MAZE_Y_MAX: int = 13.5
        GRID_SIZE_X: int = 28
        GRID_SIZE_Y: int = 31

        _, pellet_positions = load_maze_data()

        gamestate_df = self.gamestate_df.copy()

        pellet_positions = [
            (pellet[0] + 0.5, pellet[1] - 0.5) for pellet in pellet_positions
        ]

        x_grid = np.linspace(MAZE_X_MIN, MAZE_X_MAX, GRID_SIZE_X)

        y_grid = np.linspace(MAZE_Y_MAX, MAZE_Y_MIN, GRID_SIZE_Y)

        pellet_states = np.zeros(shape=(GRID_SIZE_Y, GRID_SIZE_X))

        for pellet in pellet_positions:
            x_idx = np.argmin(np.abs(x_grid - pellet[0]))
            y_idx = np.argmin(np.abs(y_grid - pellet[1]))
            pellet_states[y_idx, x_idx] = 1

        initial_available_pellet_pos = np.array(
            [
                (x_grid[idx[1]], y_grid[idx[0]])
                for idx, pellet_state in np.ndenumerate(pellet_states)
                if pellet_state == 1
            ]
        )

        initial_powerpill_pos = [
            [12.5, -9.5],
            [-12.5, -9.5],
            [-12.5, 10.5],
            [12.5, 10.5],
        ]

        gamestate_df["available_pellets"] = [
            initial_available_pellet_pos.copy() for _ in range(len(gamestate_df))
        ]
        gamestate_df["available_powerpills"] = [
            initial_powerpill_pos.copy() for _ in range(len(gamestate_df))
        ]

        for level in gamestate_df["level_id"].unique():
            gamestates = gamestate_df.loc[gamestate_df["level_id"] == level]
            logger.info(f"processing level {level}")

            for i, gamestate in enumerate(gamestates.itertuples()):
                if i == 0:
                    pacman_pos = np.array([gamestate.Pacman_X, gamestate.Pacman_Y])
                    distances = np.linalg.norm(
                        gamestate.available_pellets - pacman_pos, axis=1
                    )
                    if gamestate.pellets == 244:
                        available_pellet_pos = initial_available_pellet_pos
                    elif gamestate.pellets == 243:
                        closest_pellet_idx = np.argmin(distances)
                        available_pellet_pos = np.delete(
                            gamestate.available_pellets, closest_pellet_idx, axis=0
                        )

                    elif gamestate.pellets == 242:
                        closest_pellets_indices = np.argsort(distances)[:2]
                        available_pellet_pos = np.delete(
                            gamestate.available_pellets, closest_pellets_indices, axis=0
                        )

                    gamestate_df.at[gamestate.Index, "available_pellets"] = (
                        available_pellet_pos
                    )

                elif i > 0:
                    pacman_pos = np.array([gamestate.Pacman_X, gamestate.Pacman_Y])
                    prev_pellets = gamestate_df.at[
                        gamestate.Index - 1, "available_pellets"
                    ]
                    if len(prev_pellets) == 0:
                        break
                    distances = np.linalg.norm(prev_pellets - pacman_pos, axis=1)
                    closest_pellet_idx = np.argmin(distances)
                    pellet_counter_change = (
                        gamestate.pellets
                        - gamestate_df.at[gamestate.Index - 1, "pellets"]
                    )

                    # Double check for distance collision or counter change to be secure.
                    if (
                        distances[closest_pellet_idx] <= 0.50
                        or pellet_counter_change == -1
                    ):
                        available_pellet_pos = np.delete(
                            prev_pellets, closest_pellet_idx, axis=0
                        )

                        powerpill_mask = np.any(
                            np.all(
                                available_pellet_pos[:, None] == initial_powerpill_pos,
                                axis=2,
                            ),
                            axis=1,
                        )
                        available_powerpills = available_pellet_pos[powerpill_mask]

                        gamestate_df.at[gamestate.Index, "available_pellets"] = (
                            available_pellet_pos
                        )
                        gamestate_df.at[gamestate.Index, "available_powerpills"] = (
                            available_powerpills
                        )
                    else:
                        gamestate_df.at[gamestate.Index, "available_pellets"] = (
                            prev_pellets
                        )
                        gamestate_df.at[gamestate.Index, "available_powerpills"] = (
                            gamestate_df.at[gamestate.Index - 1, "available_powerpills"]
                        )

        return gamestate_df

    def _restructure_game_data(self):
        """
        Restructures the raw data to be more consistent in its naming and structure.

        The raw game_df is refactored to level_df. Game_df contains metadata regarding a single
        run of Pacman (which may encompass several levels, if the players are good). level_df contains
        the data as originally recorded, per level. and gamestate_df points to level_df
        """
        # Create level_df by renaming game_id to level_id and setting it as index
        level_df = self.game_df.rename(
            columns={
                "game_id": "level_id",
                "game_duration": "duration",
                "game_in_session": "level_in_session",
                "total_games_played": "total_levels_played",
            }
        ).set_index(keys="level_id", drop=False)

        # Rename game_id to level_id in gamestate_df for consistency
        if "game_id" in self.gamestate_df.columns:
            gamestate_df = self.gamestate_df.rename(columns={"game_id": "level_id"})
        else:
            gamestate_df = self.gamestate_df

        # Calculate max score for each level
        scores = gamestate_df.groupby("level_id").agg({"score": "max"})
        level_df["max_score"] = scores["score"]

        # Create game_df starting from lost levels (win=0)
        game_df = level_df.loc[level_df["win"] == 0].copy()

        # Calculate game counts
        game_df["total_games_played"] = game_df.groupby("user_id").cumcount() + 1
        game_df.drop(columns=("level_in_session"), inplace=True)
        game_df["game_in_session"] = (
            game_df.groupby(["user_id", "session_number"]).cumcount() + 1
        )

        # Rename columns for clarity
        game_df.rename(
            columns={
                "level_id": "game_id",
                "level": "max_level",
                "duration": "game_duration",
            },
            inplace=True,
        )

        game_df.index.set_names("game_id", inplace=True)

        # Remove win column as it's no longer needed
        game_df.drop(columns=["win"], inplace=True)

        # # Cross-reference between dataframes
        game_df["level_ids"] = None
        # Add game_id column to level_df
        level_df["game_id"] = None

        # For each game, collect all related level IDs and update game_id in level_df
        for _, game in game_df.iterrows():
            level_ids = []
            n_levels = game["max_level"]
            game_id = game.name
            user_id = game["user_id"]
            level_ids.append(game_id)
            level_df.at[game.name, "game_id"] = game_id
            game_duration = game["game_duration"]

            # Search for previous levels by the same user
            search = 1
            while len(level_ids) < n_levels:
                row_search = level_df.loc[game_id - search]
                if row_search["user_id"] == user_id:
                    level_ids.append(int(row_search.name))
                    game_duration += row_search["duration"]

                    level_df.at[game_id - search, "game_id"] = game_id

                search += 1

            # Sort level IDs and update game metadata
            level_ids.sort()
            game_df.at[game.name, "date_played"] = level_df.loc[
                level_ids[0], "date_played"
            ]
            game_df.at[game.name, "game_duration"] = game_duration
            game_df.at[game.name, "level_ids"] = level_ids

        return game_df, level_df, gamestate_df

    def _process_flow(self):
        """
        Process psychometrics to pair flow measures with their respective games.
        """
        flow_items = [
            "fss_1",
            "fss_2",
            "fss_3",
            "fss_4",
            "fss_5",
            "fss_6",
            "fss_7",
            "fss_8",
        ]

        flow = self.psychometrics_df.loc[
            self.psychometrics_df["redcap_repeat_instrument"] == "flow"
        ].loc[
            :, ["record_id", "total_games_flow", "redcap_repeat_instance"] + flow_items
        ]

        flow = flow.rename(
            columns={"record_id": "user_id", "total_games_flow": "total_levels_played"}
        )

        flow["FLOW"] = flow.iloc[:, 3:].sum(axis=1)

        game_psych_df = (
            pd.merge(
                flow,
                self.game_df[
                    [
                        "user_id",
                        "total_levels_played",
                        "total_games_played",
                        "max_score",
                    ]
                ],
                on=["user_id", "total_levels_played"],
                how="right",
            )
            .dropna()
            .drop(
                columns=flow_items + ["total_levels_played", "redcap_repeat_instance"]
            )
        )

        game_psych_df["log(max_score)"] = np.log(game_psych_df["max_score"])
        game_psych_df["inv(max_score)"] = (
            game_psych_df["max_score"].max() - game_psych_df["max_score"] + 1
        )  # Avoid zeros
        game_psych_df["log(inv(max_score))"] = np.log(game_psych_df["inv(max_score)"])

        game_psych_df["log(total_games_played)"] = np.log(
            game_psych_df["total_games_played"]
        )  ## i.e., cum trials

        game_psych_df["flow_z_score"] = game_psych_df.groupby("user_id")[
            "FLOW"
        ].transform(lambda x: (x - x.mean()) / x.std())

        game_psych_df["cum_score"] = game_psych_df.groupby("user_id")[
            "max_score"
        ].cumsum()
        game_psych_df["log(cum_score)"] = np.log(game_psych_df["cum_score"])

        # Calculate deviation from linear regression for each participant
        from sklearn.linear_model import LinearRegression

        # Initialize lists to store results
        user_ids = []
        deviations = []

        # Group by user_id and calculate deviation for each user
        for user_id, user_data in game_psych_df.groupby("user_id"):
            # Prepare data
            X = user_data[["total_games_played"]]
            y = user_data["log(inv(max_score))"]

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate predicted values
            y_pred = model.predict(X)

            # Calculate deviation (real - predicted)
            deviation = y - y_pred

            # Store results
            user_ids.extend([user_id] * len(deviation))
            deviations.extend(deviation)

        # Add deviation column to dataframe
        game_psych_df["score_deviation"] = deviations

        return game_psych_df

    def _process_bisbas(self):
        bisbas_df = self.psychometrics_df.loc[
            ~self.psychometrics_df["redcap_repeat_instrument"].isin(["flow", "sam"])
            & self.psychometrics_df["consent_timestamp"].notna()
            & (self.psychometrics_df["record_id"] > 60)
        ].copy()

        bisbas_df["BIS"] = bisbas_df.loc[
            :, ["bis_1", "bis_2", "bis_3", "bis_4", "bis_5", "bis_6", "bis_7"]
        ].sum(axis=1)
        bisbas_df["REW"] = bisbas_df.loc[
            :, ["rew_1", "rew_2", "rew_3", "rew_4", "rew_5"]
        ].sum(axis=1)
        bisbas_df["DRIVE"] = bisbas_df.loc[
            :, ["drive_1", "drive_2", "drive_3", "drive_4"]
        ].sum(axis=1)
        bisbas_df["FUN"] = bisbas_df.loc[:, ["fun_1", "fun_2", "fun_3", "fun_4"]].sum(
            axis=1
        )

        bisbas_df = bisbas_df.loc[
            :,
            [
                "record_id",
                "age",
                "gender",
                "nationality",
                "edu",
                "BIS",
                "REW",
                "DRIVE",
                "FUN",
            ],
        ]

        bisbas_df = bisbas_df.rename(columns={"record_id": "user_id"})
        return bisbas_df

    def _process_logging_bugs(self):
        """
        Preprocessing of identified bugs in the data

        Attack mode bug: In several levels, pacman starts in attack mode.
        This happens when the player wins the previous game shortly after eating a powerpill.
        """
        gamestate_df = self.gamestate_df.copy()
        HOUSE_X = [-3.5, 3.5]
        HOUSE_Y = [-2.5, 1.5]

        def is_in_house(state, ghost_number):
            return (
                getattr(state, f"Ghost{ghost_number}_X") > HOUSE_X[0]
                and getattr(state, f"Ghost{ghost_number}_X") < HOUSE_X[1]
                and getattr(state, f"Ghost{ghost_number}_Y") > HOUSE_Y[0]
                and getattr(state, f"Ghost{ghost_number}_Y") < HOUSE_Y[1]
                and getattr(state, "powerPellets") == 4
            )

        # Fix attack mode bug at start of levels
        for level_id in gamestate_df["level_id"].unique():
            level_data = gamestate_df[gamestate_df["level_id"] == level_id]
            first_350_states = level_data.iloc[
                :350
            ]  # First 350 are the first 17.5 seconds

            for state in first_350_states.itertuples():
                # Fix attack mode bug at start of levels
                if state.powerPellets == 4 and state.pacman_attack == 1:
                    gamestate_df.loc[state.Index, "pacman_attack"] = 0

                # Fix ghost in house with hunted state bug (ghosts are in house but not hunted)
                if state.ghost1_state == 4:
                    if is_in_house(state, 1):
                        gamestate_df.loc[state.Index, "ghost1_state"] = 0
                if state.ghost2_state == 4:
                    if is_in_house(state, 2):
                        gamestate_df.loc[state.Index, "ghost2_state"] = 0
                if state.ghost3_state == 4:
                    if is_in_house(state, 3):
                        gamestate_df.loc[state.Index, "ghost3_state"] = 0
                if state.ghost4_state == 4:
                    if is_in_house(state, 4):
                        gamestate_df.loc[state.Index, "ghost4_state"] = 0

        return gamestate_df

    def _filter_gamestate_data(
        self,
        level_id: int | list[int] | None = None,
        user_id: int | list[int] | None = None,
        game_states: tuple[int, int] | None = None,
        include_metadata: bool = False,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter gamestate data from the dataframes. Includes trajectories and game state variables.
        Args:
            level_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            include_metadata: Boolean indicating whether to include metadata in the filtered dataframe.
        Returns:
            filtered_df: tuple[pd.DataFrame, dict]: A tuple containing:
                - DataFrame containing the filtered gamestate data
                - Dictionary with metadata about the filtered data
        """

        if level_id is None and user_id is None and game_states is None:
            raise ValueError("Either level_id, user_id, or gamestates must be provided")

        time_start = time.time()

        if level_id is not None:
            logger.debug(f"Filtering gamestate data for game {level_id}...")
            level_id = [level_id] if isinstance(level_id, (int, np.int64)) else level_id

            filtered_df = self.gamestate_df[
                self.gamestate_df["level_id"].isin(level_id)
            ]
            n_games = len(
                self.level_df[self.level_df["level_id"].isin(level_id)][
                    "level_id"
                ].unique()
            )
            if n_games == 0:
                logger.info("No games found")
                raise ValueError("No game founds")

        elif user_id is not None:
            logger.debug(f"Filtering gamestate data for user {user_id}...")
            user_id = [user_id] if isinstance(user_id, (int, np.int64)) else user_id
            user_games_list = self.level_df[self.level_df["user_id"].isin(user_id)][
                "level_id"
            ].unique()
            n_games = len(user_games_list)

            if len(user_games_list) > 0:
                filtered_df = self.gamestate_df[
                    self.gamestate_df["level_id"].isin(user_games_list)
                ]
            else:
                logger.info("No games found")
                raise ValueError("No game founds")
        elif game_states is not None:
            logger.debug(f"filtering gamestate data for idx {game_states}")
            filtered_df = self.gamestate_df.loc[game_states[0] : game_states[1]]
            if len(filtered_df) == 0:
                raise ValueError("No gamestates found in the specified index range")
            if len(filtered_df["level_id"].unique()) > 1:
                raise ValueError(
                    f"The specified index range {game_states} includes more than one level_id. Please specify a range within a single level. {filtered_df['level_id'].unique()}"
                )
            level_id = filtered_df["level_id"].unique()

        if include_metadata:
            levels_meta_df = (
                self.level_df[self.level_df["level_id"].isin(level_id)]
                if level_id is not None
                else self.level_df[self.level_df["user_id"].isin(user_id)]
            )
            metadata = {
                "level_id": levels_meta_df["level_id"].unique().tolist()
                if len(levels_meta_df["level_id"].unique()) > 1
                else levels_meta_df["level_id"].unique()[0],
                "game_id": levels_meta_df["game_id"].unique().tolist()
                if len(levels_meta_df["game_id"].unique()) > 1
                else levels_meta_df["game_id"].unique()[0],
                "user_id": levels_meta_df["user_id"].unique().tolist()
                if len(levels_meta_df["user_id"].unique()) > 1
                else levels_meta_df["user_id"].unique()[0],
                "session_number": levels_meta_df["session_number"].unique().tolist()
                if len(levels_meta_df["session_number"].unique()) > 1
                else levels_meta_df["session_number"].unique()[0],
                "level_in_session": levels_meta_df["level_in_session"].unique().tolist()
                if len(levels_meta_df["level_in_session"].unique()) > 1
                else levels_meta_df["level_in_session"].unique()[0],
                "total_levels_played": levels_meta_df["total_levels_played"]
                .unique()
                .tolist()
                if len(levels_meta_df["total_levels_played"].unique()) > 1
                else levels_meta_df["total_levels_played"].unique()[0],
                "duration": levels_meta_df["duration"].unique().tolist()
                if len(levels_meta_df["duration"].unique()) > 1
                else levels_meta_df["duration"].unique()[0],
                "win": levels_meta_df["win"].unique().tolist()
                if len(levels_meta_df["win"].unique()) > 1
                else levels_meta_df["win"].unique()[0],
                "level": levels_meta_df["level"].unique().tolist()
                if len(levels_meta_df["level"].unique()) > 1
                else levels_meta_df["level"].unique()[0],
            }
        else:
            metadata = {}

        # logger.info(
        #     f"Found {n_games} levels for {'user' if user_id is not None else 'level'} {level_id if level_id is not None else user_id}"
        # )
        logger.debug(
            f"Time taken to filter gamestate data: {time.time() - time_start} seconds"
        )

        return filtered_df, metadata

    def get_trajectory(
        self,
        level_id: int | None = None,
        game_states: tuple[int, int] | None = None,
        get_timevalues: bool = False,
        include_metadata: bool = True,
    ) -> Trajectory:
        """
        Get Pacman trajectory data from the dataframes, without any metadata.
        Args:
            level_id: level id to filter the data by. If None, game_states must be provided.
            game_states: Tuple of gamestates defining the start and end of a trajectory (based on game_state_id indexing)
            get_timevalues: Boolean indicating whether to include time values in the trajectory.
            include_metadata: Boolean indicating whether to include metadata in the trajectory.
        Returns:
            trajectory: `Trajectory` object containing Pacman trajectory data (x,y) coordinates and metadata.

        Example:
        ```python
        # Get trajectory for a specific level
        trajectory = data_reader.get_trajectory(level_id=123)

        # Get trajectory with time values
        trajectory = data_reader.get_trajectory(level_id=123, get_timevalues=True)

        # Get trajectory without metadata
        trajectory = data_reader.get_trajectory(level_id=123, include_metadata=False)

        # Access trajectory data
        coordinates = trajectory.coordinates  # numpy array of (x,y) coordinates
        timevalues = trajectory.timevalues  # numpy array of time values (if get_timevalues=True)
        metadata = trajectory.metadata  # dictionary of metadata (if include_metadata=True)
        ```
        """
        time_start = time.time()
        logger.debug(f"Getting trajectory for game {level_id}...")
        if level_id is None and game_states is None:
            raise ValueError("Either level_id or gamestates must be provided")

        filtered_df, metadata = self._filter_gamestate_data(
            level_id=level_id,
            game_states=game_states,
            include_metadata=include_metadata,
        )

        if filtered_df is None:
            return None

        if get_timevalues:
            logger.debug(f"Trajectory retrieved in {time.time() - time_start} seconds")
            return Trajectory(
                coordinates=np.array(filtered_df[["Pacman_X", "Pacman_Y"]].values),
                timevalues=np.array(filtered_df["time_elapsed"].values),
                metadata=metadata,
            )
        else:
            logger.debug(f"Trajectory retrieved in {time.time() - time_start} seconds")
            return Trajectory(
                coordinates=np.array(filtered_df[["Pacman_X", "Pacman_Y"]].values),
                metadata=metadata,
            )

    def get_partial_trajectory(
        self,
        level_id: int | list[int] | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
        get_timevalues: bool = False,
    ) -> Trajectory:
        """
        Get a partial trajectory from the dataframes.
        Args:
            level_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            start_timestep: Start timestep of the trajectory.
            end_timestep: End timestep of the trajectory.
        Returns:
            trajectory: `Trajectory` object containing Pacman trajectory data (x,y) coordinates.
        """
        trajectory = self.get_trajectory(
            level_id=level_id, get_timevalues=get_timevalues
        )

        partial_trajectory = trajectory.get_segment(start_timestep, end_timestep)

        return partial_trajectory

    def get_trajectory_dataframe(
        self,
        series_type=["position"],
        include_game_state_vars=False,
        include_timesteps=True,
        include_game_id=True,
        level_id: int | list[int] = None,
        user_id: int | list[int] = None,
    ) -> pd.DataFrame:
        """
        For use in `datamodule`.
        Preprocess gamestates' data before converting to tensor for Autoencoder training.

        Args:
            gamestate_df: DataFrame containing raw game data
            series_type: List of series types to include in the preprocessing (e.g., ['position', 'movement', 'input'])
            include_game_state_vars: Boolean indicating whether to include game state variables (score, powerPellets)
            include_timesteps: Boolean indicating whether to include time elapsed in the features
            include_game_id: Boolean indicating whether to include game id in the features
            level_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.

        Returns:
            processed_df: DataFrame containing preprocessed game data with selected features

        Note:
            - `movement` is the direction Pacman is moving in the current timestep. Is an unitary vector, and should not
            be confused with instant/average velocities (as calculated in `GridAnalyzer.py`).
            - `input` is the direction Pacman is moving in the next timestep.

        """
        if user_id is not None:
            dataframe, _ = self._filter_gamestate_data(user_id=user_id)
        elif level_id is not None:
            dataframe, _ = self._filter_gamestate_data(level_id=level_id)
        else:
            dataframe = self.gamestate_df

        features = []

        if include_game_id:
            features.extend(["level_id"])

        if include_game_state_vars:
            features.extend(["score", "powerPellets"])

        if include_timesteps:
            features.extend(["time_elapsed"])

        if "position" in series_type:
            features.extend(
                ["Pacman_X", "Pacman_Y"]
            )  # No preprocessing here as it is done in the read_data() function

        direction_mapping = {
            "right": (1, 0),
            "left": (-1, 0),
            "up": (0, 1),
            "down": (0, -1),
            "none": (0, 0),
            np.nan: (0, 0),
            "w": (0, 1),
            "a": (-1, 0),
            "s": (0, -1),
            "d": (1, 0),
        }

        if "movement" in series_type:
            # Convert movement directions to dx, dy components
            dataframe["movement_dx"] = dataframe["movement_direction"].map(
                lambda d: direction_mapping[d][0]
            )
            dataframe["movement_dy"] = dataframe["movement_direction"].map(
                lambda d: direction_mapping[d][1]
            )
            features.extend(["movement_dx", "movement_dy"])

        if "input" in series_type:
            # Similarly for input directions
            dataframe["input_dx"] = dataframe["input_direction"].map(
                lambda d: direction_mapping[d][0]
            )
            dataframe["input_dy"] = dataframe["input_direction"].map(
                lambda d: direction_mapping[d][1]
            )
            features.extend(["input_dx", "input_dy"])

        processed_df = dataframe[features].copy()

        # Convert to float
        for col in processed_df.columns:
            if col != "level_id":  # Skip level_id conversion
                processed_df[col] = processed_df[col].astype(float)

        return processed_df
