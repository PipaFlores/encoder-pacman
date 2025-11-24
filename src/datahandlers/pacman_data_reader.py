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

    FEATURE_SETS: dict[str, list[str]] = {
    "Pacman": ["Pacman_X", "Pacman_Y"],
    "Pacman_Ghosts": [
        "Pacman_X", "Pacman_Y",
        "Ghost1_X", "Ghost1_Y",
        "Ghost2_X", "Ghost2_Y",
        "Ghost3_X", "Ghost3_Y",
        "Ghost4_X", "Ghost4_Y",
    ],
    "Ghost_Distances": [
        "Ghost1_distance", "Ghost2_distance",
        "Ghost3_distance", "Ghost4_distance",
    ],
    }



    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PacmanDataReader, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        data_folder: str,
        read_games_only: bool = True,
        rebase_score_per_level: bool = True,
        process_pellet: bool = True,
        force_preprocess: bool = False,
        verbose: bool = False,
        debug: bool = False,

    ):
        # Initialize basic attributes if not already initialized
        if not hasattr(self, "initialized"):
            self.data_folder = data_folder
            self.verbose = verbose
            self.initialized = True
            self.read_games_only = read_games_only
            self.rebase_score_per_level = rebase_score_per_level
            self.process_pellet = process_pellet
            logger.info(
                f"Initializing PacmanDataReader with read_games_only: {read_games_only}"
            )
            self._init_logger(verbose, debug)
            self._read_data(read_games_only, force_preprocess=force_preprocess)  # Initial load with BANNED_USERS = [42]
        # If already initialized but requesting more data, load it
        elif not self.read_games_only and read_games_only:
            logger.info(
                "Warning: Instance already loaded with full data. Ignoring read_games_only=True."
            )
        elif self.read_games_only and not read_games_only:
            self._init_logger(verbose, debug)
            logger.info("Loading additional data as requested...")
            self._read_data(read_games_only, force_preprocess=force_preprocess)
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

    def _read_data(self, read_games_only: bool, force_preprocess: bool = False):
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


        # Check that for each level_id in gamestate_df there is an analogue in level_df. Otherwise raise error
        missing_level_ids = set(self.gamestate_df["level_id"].unique()) - set(self.level_df["level_id"].unique())
        if missing_level_ids:
            print(
                f"ERROR: The following level_id(s) in gamestate_df are missing from level_df: {missing_level_ids}. Please check that 'game.csv' and 'gamestate.csv' are aligned and collected at the same time."
            )
            del self
            return

        ## If pellet positions have not been processed (using the .csv)
        if "available_pellets" not in self.gamestate_df.columns or force_preprocess:
            logger.warning(
                "Calculating pellet positions for each game state in pacman game, estimated time of 10-15 minutes"
            )
            self.gamestate_df = self._process_pellet_positions()

            logger.warning("Processing final gamestates")
            self.gamestate_df = self._process_final_gamestate()

            logger.warning("Processing logging bugs")
            self.gamestate_df = (
                self._process_logging_bugs()
            )  # Whatever bugs are encountered (e.g., attackmode at initial gamestates)

            logger.warning("Calculating and inserting Astar distances")
            self.gamestate_df = self._calculate_astar_distances()

            self.gamestate_df.to_pickle(os.path.join(self.data_folder, "gamestate.pkl"))

        if self.rebase_score_per_level:
                # For each level_id, rebase the scores so that the first score in the level becomes zero.
                # This guarantees that 'score' starts from 0 at every level, by subtracting the level's first score from each row in that level.
                first_scores = self.gamestate_df.groupby("level_id")["score"].transform("first")
                self.gamestate_df["score"] = self.gamestate_df["score"] - first_scores

        if not read_games_only:
            self.user_df = pd.read_csv(os.path.join(self.data_folder, "user.csv"))
            self.ip_df = pd.read_csv(os.path.join(self.data_folder, "userip.csv"))
            self.redcap_df = pd.read_csv(
                os.path.join(self.data_folder, "redcapdata.csv")
            )
            self.psychometrics_df = pd.read_csv(
                os.path.join(
                    self.data_folder, r"psych\AiPerCogPacman_DATA_2025-09-19_1324.csv"
                )
            )

            # Process psych data
            self.game_flow_df = self._process_flow()
            self.bisbas_df = self._process_bisbas()

    def make_data(
        self,
        feature_set: str = "Pacman",
        sequence_type: str = "first_5_seconds",
        context: int = 20,
        padding_value: float = -999.0,
        sort_ghost_distances: bool = True,
        normalization: str | None = None,
        make_gif: bool = False,
        return_raw_sequences: bool = False,
        max_samples : int | None = None
    ):
        """
        Assemble processed game data sequences, returning normalized and padded feature arrays.

        Args:
            feature_set (str): Name of the feature set to extract columns from for analysis. Must match keys in self.FEATURE_SETS.
            sequence_type (str): The method or rule for slicing sequences (e.g., 'first_5_seconds', 'start_to_first_death').
            context (int): Number of timesteps to include before/after the main slicing window, for context.
            padding_value (float): Value used to pad variable-length sequences to uniform length.
            sort_ghost_distances (bool): Whether to sort all columns ending in '_distance' for ghosts in ascending order per timestep/sample.
            normalization (str|None): Can be 'global', 'sequence', 'sample', or None. Specifies normalization strategy for features.
            make_gif (bool): If True, generates GIFs (visualizations) for each sequence.
            as_pytorch_dataset (bool): If True, returns a PyTorch-ready dataset (not implemented/used here).
            return_raw_sequences (bool): If True, returns raw/from-info dataframes for each sliced sequence. Used in `patern_analysis` for validation

        Returns:
            If return_raw_sequences is True:
                tuple: (raw_sequences, X_padded, gif_paths, features)
                - raw_sequences: list[pd.DataFrame] of each sliced sequence.
                - X_padded: np.ndarray, shape (n_sequences, sequence_length, n_features), normalized and padded data.
                - gif_paths: list[str], paths to any generated GIFs (if make_gif is True).
                - features: list[str], feature column names used for extraction.
            Otherwise:
                (Not used here; function expects return_raw_sequences and will return above tuple.)
        """
        
        seq_type = sequence_type
        try:
            features = self.FEATURE_SETS[feature_set]
        except KeyError as exc:
            raise ValueError(f"Unknown features selection: {feature_set}") from exc
        
        ## Initialize Normalizer class
        
        if isinstance(normalization, str) and normalization.lower() == "none":
            normalization = None
            
        if normalization is not None:
            from src.datahandlers import FeatureNormalizer
            Normalizer = FeatureNormalizer()

        # Slice sequences by type
        raw_sequences, gif_paths = self._slice_by_sequence_type(seq_type, context, make_gif=make_gif)

        ## If global normalization, then normalize before slicing
        if normalization == "global": ## full dataset normalization
            non_normalized_gamestate_df = self.gamestate_df.copy()
            self.gamestate_df = Normalizer.normalize(self.gamestate_df) # normalize raw dataframe for slicing method
            normalized_sequences, gif_paths = self._slice_by_sequence_type(seq_type, context, make_gif=make_gif)
            self.gamestate_df = non_normalized_gamestate_df # return to original raw dataframe


        # Local normalizations
        if normalization == "sequence": ## across sequence subset (e.g., across first 5 seconds)
            df = Normalizer.normalize(pd.concat(raw_sequences))
            normalized_sequences = [df.iloc[start:end] for start, end in zip(
                np.cumsum([0] + [len(seq) for seq in raw_sequences[:-1]]),
                np.cumsum([len(seq) for seq in raw_sequences])
            )]
        elif normalization == "sample": # Per sample
            normalized_sequences = [Normalizer.normalize(seq) for seq in raw_sequences]
        
        elif normalization is None:
            normalized_sequences = raw_sequences


        filtered = [sequence[features].to_numpy() for sequence in normalized_sequences]
        X_padded = self.padding_sequences(filtered, padding_value=padding_value)

        if sort_ghost_distances:
            ghost_idx = [
                i for i, col in enumerate(features)
                if col.startswith("Ghost") and col.endswith("_distance")
            ]
            if ghost_idx:
                X_padded[..., ghost_idx] = np.sort(X_padded[..., ghost_idx], axis=-1)


        if max_samples:
            raw_sequences = raw_sequences[:max_samples]
            X_padded = X_padded[:max_samples]
            gif_paths = gif_paths[:max_samples]
            features = features[:max_samples]

        if return_raw_sequences:
            return raw_sequences, X_padded, gif_paths, features

        return X_padded, gif_paths, features
    
    ### PRE-PROCESSING METHODS
    def _calculate_astar_distances(self):
        """
        Calculates and inserts A* (Astar) distances between Pacman and each ghost for every game state.

        This method iterates through each row in the gamestate DataFrame, extracts the positions of Pacman and all ghosts,
        and uses the Astar algorithm to compute the shortest path distance from each ghost to Pacman, taking into account
        the maze's wall layout. The resulting distances are inserted as new columns (Ghost1_distance, Ghost2_distance, etc.)
        in the gamestate DataFrame.

        Returns:
            pd.DataFrame: The updated gamestate DataFrame with Astar distances for each ghost.
        """
        from src.utils import Astar

        gamestate_df = self.gamestate_df.copy()
        wall_grid = Astar.generate_squared_walls(load_maze_data()[0])


        for state in gamestate_df.itertuples():
            pac_pos = (state.Pacman_X, state.Pacman_Y)
            ghost_positions = [
                (getattr(state, f"Ghost{i + 1}_X"), getattr(state, f"Ghost{i + 1}_Y"))
                for i in range(4)
            ]
            results = Astar.calculate_ghost_paths_and_distances(
                pacman_pos=pac_pos,
                ghost_positions=ghost_positions,
                grid=wall_grid
            )
            for idx, result in enumerate(results):
                gamestate_df.at[state.game_state_id, f"Ghost{idx+1}_distance"] = result[1]

        return gamestate_df


    def _process_final_gamestate(self):
        """
        Ensures that for each level where the player loses (i.e., does not win), the final gamestate reflects that all lives have been lost (lives = 0).

        This is necessary because, in some cases, the last recorded gamestate for a lost level may still show lives = 1, which can cause issues for downstream analysis (e.g., behavlets logic that expects a terminal state with lives = 0).

        The method updates the final gamestate of each non-winning level to set lives = 0, ensuring consistency and correctness for further processing.
        """
        gamestate_df = self.gamestate_df.copy()

        for level_id in gamestate_df["level_id"].unique():
            if self.level_df.loc[level_id].win == 0:
                gamestate = self.gamestate_df.loc[self.gamestate_df["level_id"] == level_id]
                last_index = gamestate.iloc[-1].game_state_id
                gamestate_df.at[last_index, "lives"] = 0
        
        return gamestate_df


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
        ).set_index(keys="level_id", drop=False).sort_index()

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
            game_states: Tuple of game_states to slice the data by. Must be states from the same level_id
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

### TRAJECTORY METHODS
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
        start_step: int = 0,
        end_step: int = -1,
        get_timevalues: bool = False,
    ) -> Trajectory:
        """
        Get a partial trajectory from the dataframes.
        Args:
            level_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            start_step: Start step (index) of the trajectory.
            end_step: End step (index) of the trajectory.
        Returns:
            trajectory: `Trajectory` object containing Pacman trajectory data (x,y) coordinates.
        """
        trajectory = self.get_trajectory(
            level_id=level_id, get_timevalues=get_timevalues
        )

        partial_trajectory = trajectory.get_segment(start_step, end_step)

        return partial_trajectory


### Tensor pre-processing
    def padding_sequences(self,
                          sequence_list:list[np.ndarray], 
                          padding_value = -999.0) -> np.ndarray:

        # Find the maximum sequence length
        max_seq_length = max(x.shape[0] for x in sequence_list)
        n = len(sequence_list)
        n_features = sequence_list[0].shape[1]

        # Pad sequences with np.nan (or 0, or any value) to max_seq_length
        padded_sequence_list = np.full((n, max_seq_length, n_features), padding_value , dtype=np.float32)
        for i, sequence in enumerate(sequence_list):
            seq_len = sequence.shape[0]
            padded_sequence_list[i, :seq_len, :] = sequence

        return padded_sequence_list

### SLICING METHODS. 
# One method iterates all levels and returns slice type.
# If interactive plotting, include gif_path
# self.slicing_method(**slicing params, 
#                       FEATURES, 
#                       make_gif
#                            ) -> [sequence_list,  // CORE -> This could be padded (maybe slicing parameter)
#                                   metadata_list, 
#                                   traj_list, 
#                                   gif_path_list] // CORE-Optional

    def _slice_by_sequence_type(
        self,
        sequence_type: str,
        context: int = 20,
        make_gif = False,
        videos_directory= "../hpc/videos/",
        gifs_directory = "./Results/subsequences/"
    ) -> Tuple[list[pd.DataFrame], list[str]]:
        """
        Load and slice data based on the specified sequence type.

        Parameters
        ----------
        sequence_type : str
            Type of sequence to extract. Valid options include:
                - "first_5_seconds"
                - "whole_level"
                - "last_5_seconds"
                - "pacman_attack"
                - "first_50_steps" (for debugging)
        context : int, optional
            Context window (in steps) for special slicing modes, such as "pacman_attack". Default is 20.
        make_gif : bool, optional
            Whether to generate GIFs for the sequences. Default is False.
        videos_directory : str, optional
            Directory containing input videos for GIF generation.
        gifs_directory : str, optional
            Directory to save generated GIFs.

        Returns
        -------
        raw_sequence_list : pd.DataFrame
            Extracted raw sequences for the selected sequence_type, containing all collected features.
        gif_path_list: list[str]
            If `make_gif = True`, returns a list of paths to each sequence's rendered .gif animation, for augmented visualization.
        """

        if sequence_type == "first_5_seconds":
            raw_sequences, gif_paths = self.slice_seq_of_each_level(
                start_step=0,
                end_step=100,
                make_gif=make_gif,
                videos_directory=videos_directory,
                gifs_directory=gifs_directory
            )
        elif sequence_type == "whole_level":
            raw_sequences, gif_paths = self.slice_seq_of_each_level(
                start_step=0,
                end_step=-1,
                make_gif=make_gif,
                videos_directory=videos_directory,
                gifs_directory=gifs_directory
            )
        elif sequence_type == "last_5_seconds":
            raw_sequences, gif_paths = self.slice_seq_of_each_level(
                start_step=-100,
                end_step=-1,
                make_gif=make_gif,
                videos_directory=videos_directory,
                gifs_directory=gifs_directory
            )
        elif sequence_type == "pacman_attack":
            raw_sequences, gif_paths = self.slice_attack_modes(
                CONTEXT=context,
                make_gif=make_gif,
                videos_directory=videos_directory,
                gifs_directory=gifs_directory
            )
        elif sequence_type == "first_50_steps": # For fast debugging
            raw_sequences, gif_paths = self.slice_seq_of_each_level(
                start_step=0, 
                end_step=50,
                make_gif=make_gif,
                videos_directory=videos_directory,
                gifs_directory=gifs_directory
            )
        else:    
            raise ValueError(f"Sequence type ({sequence_type}) not valid")
        
        return raw_sequences, gif_paths

    def slice_seq_of_each_level(
            self,
            start_step=0,
            end_step=-1,
            make_gif=False,
            videos_directory= "../hpc/videos/",
            gifs_directory = "./subsequences/"
        )-> tuple[list[pd.DataFrame], list[str]]:
        """
        Extracts a slice (subsequence) of game states for each level, from `start_step` to `end_step`.
        These are iloc type of indices, not "game_state_id" indexs.

        Args:
            start_step (int, optional): The starting index (inclusive) of the slice for each level. Defaults to 0.
            end_step (int, optional): The ending index (exclusive) of the slice for each level. If -1, includes all steps to the end. Defaults to -1.
            make_gif (bool, optional): If True, generates a GIF for each level's subsequence and returns the file paths. Defaults to False.
            videos_directory (str, optional): Directory containing the videos for each level. Defaults to "../hpc/videos/".
        Returns:
            tuple:
                raw_sequences : list[pd.DataFrame] List of DataFrames, each containing all, unfiltered, features for the sliced sequence of a level, with their named columns (useful for behavlets calculations)
                gif_path_list: list[str] List of GIF file paths (empty if make_gif is False).
        """
        raw_sequences = []

        gif_path_list = []

        if make_gif:
            from src.visualization import GameReplayer
            replayer = GameReplayer()
            logger.info("Using augmented visualization, checking for .gif or creating (can take long)")

        for level_id in self.level_df["level_id"].unique():
            gamestates, _ = self._filter_gamestate_data(level_id=level_id, include_metadata=False)

            start_step_ = start_step
            end_step_ = end_step

            end_step_ = min(end_step, len(gamestates)) if end_step != -1 else len(gamestates)

            # Handle negative indices first
            if start_step < 0:
                start_step_ = len(gamestates) + start_step
            if end_step < 0 and end_step != -1:
                end_step_ = len(gamestates) + end_step
            elif end_step == -1:
                end_step_ = len(gamestates)  # or just use None/default pandas behavior
            else:
                end_step_ = min(end_step, len(gamestates))  # Remove the -1

            gamestates = gamestates.iloc[start_step_:end_step_]

            raw_sequences.append(gamestates)


            ## and create video_sequence
            if make_gif:
                gif_path = os.path.join(gifs_directory, f"level_{level_id}_{start_step_:06d}_{end_step_:06d}.gif")
                gif_path_list.append(gif_path)
                if not os.path.exists(gif_path):
                    replayer.extract_gamestate_subsequence_ffmpeg(
                        video_path=os.path.join(videos_directory, f"{level_id}.mp4"),
                        start_gamestate=start_step_, 
                        end_gamestate=end_step_,
                        output_path=gif_path)
                    
                else:
                    # print(f"sequence for level_id {level_id} already exists, skipping")
                    pass

        return raw_sequences, gif_path_list
    

    def slice_attack_modes(self,
                           CONTEXT: int = 0):
        """
        Extracts and slices sequences of game states where Pac-Man is in "attack mode".

        This method identifies contiguous intervals in each level where the "pacman_attack"
        column is active (i.e., equals 1). For each such interval, it extracts the corresponding
        subsequence of game states, optionally including a context window (not yet implemented).

        Args:
            CONTEXT (int, optional): Number of extra frames to include before and after each
                attack mode interval. Default is 0 (no extra context).

        Returns:
            raw_sequences (list): List of DataFrames, each containing a sequence of game states
                where Pac-Man is in attack mode for a given level.
            gif_path_list (list): Empty list (reserved for future use, e.g., GIF generation).
        """
        ## TODO continue here, fine-tune and abstact for any binary column event slicing.

        raw_sequences = []
        gif_path_list = []

        for level_id in self.level_df["level_id"].unique():
            gamestates = self._filter_gamestate_data(level_id=level_id)[0]
            # Find indices where "pacman_attack" changes value
            attack_col = gamestates["pacman_attack"].values
            change_indices = np.where(attack_col[1:] != attack_col[:-1])[0] + 1  # +1 to get the index where the change happened [if changed to attack, the value at this state, and thereafter, will be 1]
            
            # Find (start_index, end_index) tuples where value switches from 1 to 0.
            # If a 1 is not followed by a 0, use the last index in gamestates as end_index.
            intervals = []
            start_index = None

            for idx, value in zip(change_indices, attack_col[change_indices]):
                if value == 1:
                    start_index = max(idx - CONTEXT , 0)
                elif value == 0 and start_index is not None:
                    end_index = min(idx + CONTEXT, len(gamestates) - 1)
                    intervals.append((start_index, end_index))
                    start_index = None

            # If we end with a 1 and no following 0, close the interval at the last index
            if start_index is not None:
                end_index = gamestates.index[-1]
                intervals.append((start_index, end_index))

            for (start_index, end_index) in intervals:
                raw_sequences.append(gamestates.iloc[start_index:end_index+1])

        return raw_sequences, gif_path_list
        
