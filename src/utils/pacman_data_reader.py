import pandas as pd
import os
import numpy as np


class PacmanDataReader:
    _instance = None
    BANNED_USERS = [42]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PacmanDataReader, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, data_folder: str, read_games_only: bool = True, verbose: bool = False
    ):
        # Initialize basic attributes if not already initialized
        if not hasattr(self, "initialized"):
            self.data_folder = data_folder
            self.verbose = verbose
            self.initialized = True
            self.read_games_only = read_games_only
            self._read_data(read_games_only)  # Initial load with BANNED_USERS = [42]
        # If already initialized but requesting more data, load it
        elif not self.read_games_only and read_games_only:
            if self.verbose:
                print(
                    "Warning: Instance already loaded with full data. Ignoring read_games_only=True."
                )
        elif self.read_games_only and not read_games_only:
            if self.verbose:
                print("Loading additional data as requested...")
            self._read_data(read_games_only)
            self.read_games_only = read_games_only

    def _read_data(self, read_games_only: bool):
        """
        Initialize the dataframes.
        Reads the data from the data folder and filters out banned users.
        """
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

        ## Filter banned users
        self.banned_game_ids = self.game_df.loc[
            self.game_df["user_id"].isin(self.BANNED_USERS), "game_id"
        ]
        self.game_df = self.game_df[~self.game_df["game_id"].isin(self.banned_game_ids)]

        self.gamestate_df = self.gamestate_df[
            ~self.gamestate_df["game_id"].isin(self.banned_game_ids)
        ]

        if not read_games_only:
            self.user_df = pd.read_csv(os.path.join(self.data_folder, "user.csv"))
            self.ip_df = pd.read_csv(os.path.join(self.data_folder, "userip.csv"))
            self.redcap_df = pd.read_csv(
                os.path.join(self.data_folder, "redcapdata.csv")
            )
            self.psychometrics_df = pd.read_csv(
                os.path.join(
                    self.data_folder, "psych\AiPerCogPacman_DATA_2025-03-03_0927.csv"
                )
            )

    def filter_gamestate_data(
        self, game_id: int | list[int] = None, user_id: int | list[int] = None
    ) -> pd.DataFrame:
        """
        Filter gamestate data from the dataframes. Includes trajectories and game state variables.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
        Returns:
            filtered_df: DataFrame containing the filtered gamestate data.
        """

        if game_id is None and user_id is None:
            raise ValueError("Either game_id or user_id must be provided")

        if game_id is not None:
            game_id = [game_id] if isinstance(game_id, int) else game_id
            filtered_df = self.gamestate_df.copy()
            filtered_df = filtered_df[filtered_df["game_id"].isin(game_id)]
            if len(filtered_df) == 0:
                print(f"No games found for game {game_id}")
                return None

        elif user_id is not None:
            user_id = [user_id] if isinstance(user_id, int) else user_id
            user_games_list = self.game_df[self.game_df["user_id"].isin(user_id)][
                "game_id"
            ].unique()

            if len(user_games_list) > 0:
                filtered_df = self.gamestate_df.copy()
                filtered_df = filtered_df[filtered_df["game_id"].isin(user_games_list)]
                print(
                    f"Found {len(filtered_df['game_id'].unique())} games for user {user_id}"
                )
            else:
                print(f"No games found for user {user_id}")
                return None

        if self.verbose and (isinstance(game_id, list) or user_id is not None):
            print(
                f"Found {len(filtered_df['game_id'].unique())} games for {'user' if user_id is not None else 'game'} {game_id if game_id is not None else user_id}"
            )

        return filtered_df

    def get_trajectory_array(
        self,
        game_id: int | list[int] = None,
        user_id: int | list[int] = None,
        get_timevalues: bool = False,
        get_all_games: bool = False,
    ) -> np.ndarray:
        """
        Get Pacman trajectory data from the dataframes, without any metadata.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            get_all_games: Boolean indicating whether to get all games.
        Returns:
            trajectory_array: Array of Pacman trajectory data (x,y) coordinates.
        """
        if game_id is None and user_id is None and not get_all_games:
            raise ValueError("Either game_id or user_id must be provided")

        if get_all_games:
            filtered_df = self.gamestate_df
        else:
            filtered_df = self.filter_gamestate_data(game_id=game_id, user_id=user_id)

        if filtered_df is None:
            return None

        if get_timevalues:
            return np.array(
                filtered_df[["time_elapsed", "Pacman_X", "Pacman_Y"]].values
            )
        else:
            return np.array(filtered_df[["Pacman_X", "Pacman_Y"]].values)

    def get_partial_trajectory_array(
        self,
        game_id: int | list[int] | None = None,
        user_id: int | list[int] | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
        get_timevalues: bool = False,
    ) -> np.ndarray:
        """
        Get a partial trajectory from the dataframes.
        Args:
            game_id: List of game ids to filter the data by.
            user_id: List of user ids to filter the data by.
            start_timestep: Start timestep of the trajectory.
            end_timestep: End timestep of the trajectory.
        Returns:
            trajectory_array: Array of Pacman trajectory data (x,y) coordinates.
        """
        trajectory_array = self.get_trajectory_array(
            game_id=game_id, user_id=user_id, get_timevalues=get_timevalues
        )

        if end_timestep == -1:
            end_timestep = len(trajectory_array)

        if start_timestep > end_timestep:
            raise ValueError("start_timestep must be less than end_timestep")

        partial_trajectory_array = trajectory_array[start_timestep:end_timestep]

        return partial_trajectory_array

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
            dataframe = self.filter_gamestate_data(user_id=user_id)
        elif game_id is not None:
            dataframe = self.filter_gamestate_data(game_id=game_id)
        else:
            dataframe = self.gamestate_df.copy()

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
