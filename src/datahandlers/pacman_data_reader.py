import pandas as pd
import os
import numpy as np
import time
from src.datahandlers.trajectory import Trajectory
from src.utils import setup_logger

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
        game_df (pd.DataFrame): DataFrame containing game metadata
        gamestate_df (pd.DataFrame): DataFrame containing game state data
        user_df (pd.DataFrame): DataFrame containing user metadata
        session_df (pd.DataFrame): DataFrame containing session metadata
    
    Example:
    ```python
    data = PacmanDataReader(data_folder="../data/")

    ## To get all trajectories as a list[Trajectory]
    all_trajs = []
    for game in data.game_df["game_id"].tolist():
        traj = data.get_trajectory(game_id=game, include_metadata=True)
        all_trajs.append(traj)
        
    ```
    
    """
    _instance = None
    BANNED_USERS = [42]
    BANNED_GAMES = [419] ## Game with 600 second idle duration (bug)

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
    ):
        # Initialize basic attributes if not already initialized
        if not hasattr(self, "initialized"):
            self.data_folder = data_folder
            self.verbose = verbose
            self.initialized = True
            self.read_games_only = read_games_only
            logger.info(
                f"Initializing PacmanDataReader with read_games_only: {read_games_only}"
            )
            self._read_data(read_games_only)  # Initial load with BANNED_USERS = [42]
        # If already initialized but requesting more data, load it
        elif not self.read_games_only and read_games_only:
            logger.info(
                "Warning: Instance already loaded with full data. Ignoring read_games_only=True."
            )
        elif self.read_games_only and not read_games_only:
            logger.info("Loading additional data as requested...")
            self._read_data(read_games_only)
            self.read_games_only = read_games_only

        if verbose:
            logger.setLevel("INFO")
        elif debug:
            logger.setLevel("DEBUG")
            logger.debug(f"Debug mode enabled")
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
        self.gamestate_df = pd.read_csv(
            os.path.join(self.data_folder, "gamestate.csv"),
            converters={
                "user_id": lambda x: int(x)
                #  , 'Pacman_X': lambda x: round(float(x), 2),
                #   'Pacman_Y': lambda x: round(float(x), 2)
            },
        )
        logger.info(f"Time taken to read game data: {time.time() - time_start} seconds")
        ## Filter banned users and games
        self.banned_game_ids = pd.concat([
            self.game_df.loc[self.game_df["user_id"].isin(self.BANNED_USERS), "game_id"],
            pd.Series(self.BANNED_GAMES)
        ]).unique()

        self.game_df = self.game_df[~self.game_df["game_id"].isin(self.banned_game_ids)]
        self.gamestate_df = self.gamestate_df[
            ~self.gamestate_df["game_id"].isin(self.banned_game_ids)
        ]

        # Create merged dataframe with game metadata
        self.gamestate_w_metadata_df = pd.merge(
            self.gamestate_df, self.game_df, on="game_id", how="left"
        )

        if not read_games_only:
            self.user_df = pd.read_csv(os.path.join(self.data_folder, "user.csv"))
            self.ip_df = pd.read_csv(os.path.join(self.data_folder, "userip.csv"))
            self.redcap_df = pd.read_csv(
                os.path.join(self.data_folder, "redcapdata.csv")
            )
            self.psychometrics_df = pd.read_csv(
                os.path.join(
                    self.data_folder, "psych\AiPerCogPacman_DATA_2025-04-01_1343.csv"
                )
            )
            
    def _filter_gamestate_data(
        self,
        game_id: int | list[int] | None = None,
        user_id: int | list[int] | None = None,
        include_metadata: bool = False,
    ) -> pd.DataFrame:
        """
        Filter gamestate data from the dataframes. Includes trajectories and game state variables.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            include_metadata: Boolean indicating whether to include metadata in the filtered dataframe.
        Returns:
            filtered_df: DataFrame containing the filtered gamestate data.
        """

        if game_id is None and user_id is None:
            raise ValueError("Either game_id or user_id must be provided")

        time_start = time.time()

        if game_id is not None:
            logger.debug(f"Filtering gamestate data for game {game_id}...")
            game_id = [game_id] if isinstance(game_id, int) else game_id

            filtered_df = self.gamestate_df[self.gamestate_df["game_id"].isin(game_id)]
            n_games = len(
                self.game_df[self.game_df["game_id"].isin(game_id)]["game_id"].unique()
            )
            if n_games == 0:
                logger.info("No games found")
                return None

        elif user_id is not None:
            logger.debug(f"Filtering gamestate data for user {user_id}...")
            user_id = [user_id] if isinstance(user_id, int) else user_id
            user_games_list = self.game_df[self.game_df["user_id"].isin(user_id)][
                "game_id"
            ].unique()
            n_games = len(user_games_list)

            if len(user_games_list) > 0:
                filtered_df = self.gamestate_df[
                    self.gamestate_df["game_id"].isin(user_games_list)
                ]
            else:
                logger.info("No games found")
                return None

        if include_metadata:
            games_meta_df = (
                self.game_df[self.game_df["game_id"].isin(game_id)]
                if game_id is not None
                else self.game_df[self.game_df["user_id"].isin(user_id)]
            )
            metadata = {
                "game_id": games_meta_df["game_id"].unique().tolist()
                if len(games_meta_df["game_id"].unique()) > 1
                else games_meta_df["game_id"].unique()[0],
                "user_id": games_meta_df["user_id"].unique().tolist()
                if len(games_meta_df["user_id"].unique()) > 1
                else games_meta_df["user_id"].unique()[0],
                "session_number": games_meta_df["session_number"].unique().tolist()
                if len(games_meta_df["session_number"].unique()) > 1
                else games_meta_df["session_number"].unique()[0],
                "game_in_session": games_meta_df["game_in_session"].unique().tolist()
                if len(games_meta_df["game_in_session"].unique()) > 1
                else games_meta_df["game_in_session"].unique()[0],
                "total_games_played": games_meta_df["total_games_played"]
                .unique()
                .tolist()
                if len(games_meta_df["total_games_played"].unique()) > 1
                else games_meta_df["total_games_played"].unique()[0],
                "game_duration": games_meta_df["game_duration"].unique().tolist()
                if len(games_meta_df["game_duration"].unique()) > 1
                else games_meta_df["game_duration"].unique()[0],
                "win": games_meta_df["win"].unique().tolist()
                if len(games_meta_df["win"].unique()) > 1
                else games_meta_df["win"].unique()[0],
                "level": games_meta_df["level"].unique().tolist()
                if len(games_meta_df["level"].unique()) > 1
                else games_meta_df["level"].unique()[0],
            }
        else:
            metadata = {}

        logger.info(
            f"Found {n_games} games for {'user' if user_id is not None else 'game'} {game_id if game_id is not None else user_id}"
        )
        logger.debug(
            f"Time taken to filter gamestate data: {time.time() - time_start} seconds"
        )

        return filtered_df, metadata

    def get_trajectory(
        self,
        game_id: int | list[int] = None,
        user_id: int | list[int] = None,
        get_timevalues: bool = False,
        include_metadata: bool = True,
    ) -> Trajectory:
        """
        Get Pacman trajectory data from the dataframes, without any metadata.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            get_all_games: Boolean indicating whether to get all games.
            include_metadata: Boolean indicating whether to include metadata in the trajectory.
        Returns:
            trajectory: `Trajectory` object containing Pacman trajectory data (x,y) coordinates and metadata.

        Example:
        ```python
        # Get trajectory for a specific game
        trajectory = data_reader.get_trajectory(game_id=123)
        
        # Get trajectory for a specific user
        trajectory = data_reader.get_trajectory(user_id=456)
        
        # Get trajectory with time values
        trajectory = data_reader.get_trajectory(game_id=123, get_timevalues=True)
        
        # Get trajectory without metadata
        trajectory = data_reader.get_trajectory(game_id=123, include_metadata=False)
        
        # Access trajectory data
        coordinates = trajectory.coordinates  # numpy array of (x,y) coordinates
        timevalues = trajectory.timevalues  # numpy array of time values (if get_timevalues=True)
        metadata = trajectory.metadata  # dictionary of metadata (if include_metadata=True)
        ```
        """
        time_start = time.time()
        logger.debug(f"Getting trajectory for game {game_id} and user {user_id}...")
        if game_id is None and user_id is None:
            raise ValueError("Either game_id or user_id must be provided")

        filtered_df, metadata = self._filter_gamestate_data(
            game_id=game_id, user_id=user_id, include_metadata=include_metadata
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
        game_id: int | list[int] | None = None,
        user_id: int | list[int] | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
        get_timevalues: bool = False,
    ) -> Trajectory:
        """
        Get a partial trajectory from the dataframes.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            start_timestep: Start timestep of the trajectory.
            end_timestep: End timestep of the trajectory.
        Returns:
            trajectory: `Trajectory` object containing Pacman trajectory data (x,y) coordinates.
        """
        trajectory = self.get_trajectory(
            game_id=game_id, user_id=user_id, get_timevalues=get_timevalues
        )

        partial_trajectory = trajectory.get_segment(start_timestep, end_timestep)

        return partial_trajectory

    def get_trajectory_dataframe(
        self,
        series_type=["position"],
        include_game_state_vars=False,
        include_timesteps=True,
        include_game_id=True,
        game_id: int | list[int] = None,
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
            game_id: List of game ids to filter the data by.
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
        elif game_id is not None:
            dataframe, _ = self._filter_gamestate_data(game_id=game_id)
        else:
            dataframe = self.gamestate_df

        features = []

        if include_game_id:
            features.extend(["game_id"])

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
            if col != "game_id":  # Skip game_id conversion
                processed_df[col] = processed_df[col].astype(float)

        return processed_df
