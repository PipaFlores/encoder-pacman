import pandas as pd
import numpy as np
import math
from typing import Callable, NamedTuple
from src.utils import setup_logger
from src.analysis.behavlets_config import BehavletsConfig


logger = setup_logger(__name__)


class Behavlets:
    NAMES = ["Aggression1", "Aggression3", "Aggression4", "Aggression6"]

    @property
    def ENCODING_FUNCTIONS(self) -> dict[str, Callable]:
        "Dictionary mapping behavlets name to their calculation/encoding algorithms"
        return {
            "Aggression1": self._Aggression1,
            "Aggression3": self._Aggression3,
            "Aggression4": self._Aggression4,
            "Aggression6": self._Aggression6,
        }

    def __init__(self, name: str, verbose: bool = False, debug: bool = False, **kwargs):
        """
        Initialize a Behavlets instance for analyzing player behavior in Pacman.

        This class implements various behavlets (behavioral patterns) that can be used to analyze
        player behavior in Pacman. Each behavlet represents a specific type of behavior that can
        be measured and quantified during gameplay.

        Args:
            name (str): The name of the behavlet to initialize. Must be one of the predefined
                       behavlet names in self.NAMES.
            verbose (bool): If True, sets logging level to INFO. Defaults to False.
            debug (bool): If True, sets logging level to DEBUG. Defaults to False.
            **kwargs: Additional parameters specific to each behavlet type (overrides those of `behavlets_config.py`)


        Raises:
            ValueError: If the provided name is not in the list of valid behavlet names.

        Attributes:
            name (str): Short name of the behavlet (e.g., "Aggression1")
            full_name (str): Full descriptive name of the behavlet
            category (str): Category of the behavlet (e.g., "Aggression")
            value (int): Quantitative measure of the behavlet's occurrence
            gamesteps (list): List of tuples containing (start, end) gamesteps for each instance
            timesteps (list): List of tuples containing (start, end) timesteps for each instance
            instances (int): Number of times the behavlet was observed
            value_per_pill (list): Optional attribute for tracking values per powerpill
            died (list): Optional attribute for tracking if player died during observation
        """
        if verbose:
            logger.setLevel("INFO")
            logger.info(f"Initializing {name}")
        if debug:
            logger.setLevel("DEBUG")
            logger.debug(f"Initializing {name}")
        else:
            logger.setLevel("WARNING")

        self.config = BehavletsConfig()
        self.kwargs = {
            **self.config.get_config(name),
            **kwargs,
        }  # Store kwargs to be used in calculate() method

        ###
        ###  COMMON ATTRIBUTES TO ALL BEHAVLET TYPES
        ###
        self.name = name  # Short name of behavlets (<category><number>)
        self.full_name = ""  # Full name of behavlets, as in literature. Attribute is set by specific functions when self.calculate()
        if name not in self.NAMES:
            raise ValueError(f"Unknown behavlet name: {name}")

        self.category = self._get_category(
            self.name
        )  ## Aggression, Caution, Planning, etc..

        self.measurement_type = None
        ## Type of measurement varies of behablet types
        # interval - occurs over an interval of time (e.g., Agression 4 increasing as long as player aproaches ghost)
        # point - is situated in a particular gamestep (e.g., Agression 3, Ghost Kill)

        ## this is the "output" of the original implementation.
        # Usually represents the number of behavlets observed. However, in some other, it represents other quantitative amounts
        # (e.g., Speed - cycles per sector is a value of time)
        self.value = 0

        ## list of tuples with (initial_gamestate, end_gamestate) that define the behavlet.
        # For some behavlet types, multiple can appear in a game, so there will be one tuple per each.
        self.gamesteps = []
        self.original_gamesteps = []  # For reference in case of CONTEXT_LENGTH adjustments
        self.timesteps = []

        self.instances = 0  # if multiple instances. Is len(gamesteps)

        ###
        ### OPTIONAL ATTRIBUTES SHARED OR UNIQUE OF CERTAIN BEHAVLET TYPES
        ###
        self.value_per_pill = []  ## e.g., Aggression4 counts for each pill in the game
        self.died = []  ## Did the player died during the behavlet observation?

    def calculate(self, gamestates: pd.DataFrame, **kwargs):
        """
        Calculates the behavlet using the specific algorithm, according to behavlet name
        (i.e., self.name = Aggression3 will map to self._Aggression3(data))
        Results are stores in the self.values, self.gamesteps, and self.timesteps attributes.
        """
        if not isinstance(gamestates, pd.DataFrame):
            raise TypeError(
                f"type(data) needs to be pd.Dataframe, not {type(gamestates)}"
            )
        # Calculate Behavlets
        self.ENCODING_FUNCTIONS[self.name](gamestates, **{**self.kwargs, **kwargs})
        self.instances = len([x for x in self.gamesteps if x is not None])

        if self.full_name == "":
            raise ValueError(f"full_name not set for behavlet {self.name}")
        if self.measurement_type is None:
            raise ValueError(f"measurement_type not set for behavlet {self.name}")

        return self

    def _get_category(self, name: str) -> str:
        """
        Classify behavlets
        """
        if name.startswith("Aggression"):
            return "Aggression"
        elif name.startswith("Caution"):
            return "Caution"
        elif name.startswith("Thoroughness"):
            return "Thoroughness"
        elif name.startswith("Decisiveness"):
            return "Decisiveness"
        elif name.startswith("Planning"):
            return "Planning"
        elif name.startswith("Resourcing"):
            return "Resourcing"
        elif name.startswith("Speed"):
            return "Speed"
        elif name.startswith("Control_Skill"):
            return "Control_Skill"
        else:
            raise ValueError(f"Unknown behavlet category for name: {name}")

    def _reset_values(self):
        """Reset behavlet values to 0 for fresh calculations"""

        self.value = 0
        self.gamesteps = []
        self.timesteps = []
        self.instances = 0

    def _Aggression1(self, gamestates: pd.DataFrame, **kwargs):
        """
        Hunt close to ghost home

        Counts instances where the player follows the ghosts right up to their house while attacking them.

        Parameters
        ----------
        gamestates : pd.DataFrame
            DataFrame containing the game state information for each timestep.
        BOUNDARY_DISTANCE : float, optional
            The maximum Manhattan distance from the ghost house to consider as "close" (default is 7.5).
        HOUSE_POS : tuple, optional
            The (x, y) position of the ghost house (default is (0, -0.5)).
        X_BOUNDARIES : tuple, optional
            The minimum and maximum x-coordinates of the ghost perimeter area (default is (-6.5, 6.5)).
        Y_BOUNDARIES : tuple, optional
            The minimum and maximum y-coordinates of the ghost perimeter area (default is (-5.5, 4.5)).
        CLOSENESS_DEF : str, optional
            Definition of "closeness", what is considered to be "right up to their house"
            used for the calculation either Distance to house (manhattan distance
            to house center, using Boundary_distance), or House perimeter (boundaries in space)
        CONTEXT_LENGTH : int, optional
            Number of frames to include before and after the sequence is observed

        Returns
        -------
        None
            Updates self.value, self.gamesteps, and self.timesteps with the results of the calculation.

        Notes
        -----
        - self.instance: Number of instances the player hunts close to the ghost home while attacking.
        - self.value: List of integers representin the amount of gamestates where the behavior was observed, for each instance.
        - self.gamesteps: List of (start, end) indices for each detected instance.
        - self.timesteps: List of (start, end) time elapsed for each detected instance.
        """

        self.full_name = "Aggression 1 - Hunt close to ghost home"
        self.measurement_type = "interval"

        BOUNDARY_DISTANCE = kwargs.get("BOUNDARY_DISTANCE", 7.5)
        HOUSE_POS = kwargs.get("HOUSE_POS", (0, -0.5))
        X_BOUNDARIES = kwargs.get("X_BOUNDARIES", (-6.5, 6.5))
        Y_BOUNDARIES = kwargs.get("Y_BOUNDARIES", (-5.5, 4.5))
        CLOSENESS_DEF = kwargs.get("CLOSENESS_DEF", "House perimeter")
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)

        flag = False
        self.value = []  # value for each instance observed.
        for i, state in enumerate(gamestates.itertuples()):
            if state.pacman_attack == 1:
                Pacman_pos = np.array([state.Pacman_X, state.Pacman_Y])
                ghost_Positions = self._get_ghost_pos(state, only_alive=True)

                if CLOSENESS_DEF == "Distance to house":
                    Pacman_house_dist = abs(Pacman_pos[0] - HOUSE_POS[0]) + abs(
                        Pacman_pos[1] - HOUSE_POS[1]
                    )
                    Ghost_house_distances = [
                        abs(HOUSE_POS[0] - ghost_pos[0])
                        + abs(HOUSE_POS[1] - ghost_pos[1])
                        for ghost_pos in ghost_Positions
                    ]
                    Pacman_condition = Pacman_house_dist <= BOUNDARY_DISTANCE
                    Ghost_conditions = any(
                        dist <= BOUNDARY_DISTANCE for dist in Ghost_house_distances
                    )
                elif CLOSENESS_DEF == "House perimeter":
                    logger.debug(f"step {state.Index} ghost positions{ghost_Positions}")
                    Pacman_condition = (
                        X_BOUNDARIES[0] <= Pacman_pos[0] <= X_BOUNDARIES[1]
                    ) and (Y_BOUNDARIES[0] <= Pacman_pos[1] <= Y_BOUNDARIES[1])
                    Ghost_conditions = any(
                        [
                            (X_BOUNDARIES[0] <= ghost_pos[0] <= X_BOUNDARIES[1])
                            and (Y_BOUNDARIES[0] <= ghost_pos[1] <= Y_BOUNDARIES[1])
                            for ghost_pos in ghost_Positions
                        ]
                    )

                if not flag and Pacman_condition and Ghost_conditions:
                    value = 1
                    start_gamestep = state.Index
                    start_timestep = state.time_elapsed
                    flag = True
                elif flag and Pacman_condition and Ghost_conditions:
                    value += 1  # increase value the longer the conditions remain true

                elif flag and (not Pacman_condition or not Ghost_conditions):
                    end_gamestep = state.Index
                    end_timestep = state.time_elapsed

                    if CONTEXT_LENGTH:
                        start_gamestep = max(0, start_gamestep - CONTEXT_LENGTH)
                        end_gamestep = min(
                            gamestates.iloc[-1]["game_state_id"],
                            end_gamestep + CONTEXT_LENGTH,
                        )

                    self.value.append(value)
                    self.gamesteps.append((start_gamestep, end_gamestep))
                    self.timesteps.append((start_timestep, end_timestep))
                    flag = False

            elif state.pacman_attack == 0 and flag:
                end_gamestep = state.Index
                end_timestep = state.time_elapsed

                self.gamesteps.append((start_gamestep, end_gamestep))
                self.timesteps.append((start_timestep, end_timestep))
                flag = False

        return

    def _Aggression2(
        self,
        gamestates: pd.DataFrame,
    ):
        """
        Chase Ghosts or Eat New Cherry

        If cherry appears when player is attacking, count if they abandons ghost chase and gets cherry

        """

        raise NotImplementedError

    def _Aggression3(self, gamestates: pd.DataFrame, **kwargs):
        """
        Aggression3 - Ghost Kills

        Measures the number of times the player successfully eats a ghost during gameplay.
        This behavlet tracks when a ghost's state changes to "eaten" (state 4), indicating
        the player has successfully captured and consumed the ghost.

        The behavlet counts each ghost kill as a separate instance and records the
        gamesteps and timesteps around each kill for visualization and analysis purposes.

        Args:
            gamestates (pd.DataFrame): DataFrame containing game state data
            CONTEXT_LENGTH (int): Number of frames to include before and after each ghost kill
                               for visualization purposes. Defaults to 10.

        Returns:
            None: Updates the behavlet's value, gamesteps, and timesteps attributes
        """
        self.full_name = "Aggression 3 - Ghost kills"
        self.measurement_type = "point"

        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", 10)

        previous_ghost_states = [0, 0, 0, 0]

        for state in gamestates.itertuples():
            new_ghost_states = [
                state.ghost1_state,
                state.ghost2_state,
                state.ghost3_state,
                state.ghost4_state,
            ]
            for i in range(len(previous_ghost_states)):
                if previous_ghost_states[i] != 4 & new_ghost_states[i] == 4:
                    self.value += 1

                    self.gamesteps.append(
                        (
                            max(0, state.Index - CONTEXT_LENGTH),
                            min(
                                gamestates.iloc[-1]["game_state_id"],
                                state.Index + CONTEXT_LENGTH,
                            ),
                        )
                    )
                    self.timesteps.append(state.time_elapsed)
            previous_ghost_states = new_ghost_states

        return

    def _Aggression4(self, gamestates: pd.DataFrame, **kwargs):
        """
        Hunts even after powerpill finishes
        Tracks instances where the player continues hunting ghosts after a powerpill's effects have worn off.

        This behavlet measures aggressive behavior by counting how often the player continues
        pursuing ghosts even after losing the powerpill's protective effects. It tracks each
        powerpill separately and records the gamesteps and timesteps around these instances
        for visualization and analysis.

        Args:
            gamestates (pd.DataFrame): DataFrame containing game state data
            SEARCH_WINDOW (int): Number of frames to include before and after each instance
                               for visualization purposes. Defaults to 10.
            VALUE_THRESHOLD (int): Minimum number of instances required to count as a valid
                                 behavlet occurrence. Defaults to 1.
            GHOST_DISTANCE_THRESHOLD (int): Maximum distance between Pacman and ghosts to
                                          consider as "hunting". Defaults to 7.
            CONTEXT_LENGTH (int), Optional: Number of frames to include before and after the sequence
                                is observed

        Returns:
            None: Updates the behavlet's value, gamesteps, timesteps, and died attributes
        """

        self.full_name = "Aggresssion 4 - Hunt even after powerpill finishes"
        self.measurement_type = "interval"

        SEARCH_WINDOW = kwargs.get("SEARCH_WINDOW", 10)
        VALUE_THRESHOLD = kwargs.get("VALUE_THRESHOLD", 1)
        GHOST_DISTANCE_THRESHOLD = kwargs.get("GHOST_DISTANCE_THRESHOLD", 7)
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)

        logger.debug(f"Searching for {self.name} with Search window {SEARCH_WINDOW}")

        self.value_per_pill = [0] * 4  # Counter per pill
        self.gamesteps = [None] * 4
        self.timesteps = [None] * 4
        self.died = [None] * 4

        flag = False
        pellet_eaten = False

        for i, state in enumerate(gamestates.itertuples()):
            if i == 0:
                continue

            # Check which pill was eaten
            if (
                gamestates.at[state.Index - 1, "pacman_attack"] == 0
                and state.pacman_attack == 1  # Switch to attack mode
            ):
                pacman_pos = [state.Pacman_X, state.Pacman_Y]
                powerpill_eaten_step = state.Index

                # Quadrant based powerpill indexing (Aligned with Readme.md)
                pellet_idx = self._get_quadrant_idx(state=state)

                pellet_eaten = True

            # First gamestep after the pill wears off
            elif (
                gamestates.at[state.Index - 1, "pacman_attack"] == 1
                and state.pacman_attack == 0
                and pellet_eaten
            ):
                logger.debug(
                    f"{self.name} - pellet {pellet_idx} wore off at step {state.Index}"
                )
                flag = True
                starting_gamestep = state.Index
                starting_timestep = state.time_elapsed
                lives_at_wear_off = state.lives
                pacman_pos = [state.Pacman_X, state.Pacman_Y]
                ghost_positions = self._get_ghost_pos(state, only_alive=False)

                prev_distance_to_ghosts = [
                    abs(pacman_pos[0] - ghost_pos[0])
                    + abs(pacman_pos[1] - ghost_pos[1])
                    for ghost_pos in ghost_positions
                ]
            # look gamesteps after pill wears off, within SEARCH_WINDOW, and check if distance to any ghosts diminish
            elif flag:
                pacman_pos = [state.Pacman_X, state.Pacman_Y]
                ghost_positions = self._get_ghost_pos(
                    state, only_alive=False
                )  # Get all ghost to ensure alignment with prev_distances

                distance_to_ghosts = [
                    abs(pacman_pos[0] - ghost_pos[0])
                    + abs(pacman_pos[1] - ghost_pos[1])
                    for ghost_pos in ghost_positions
                ]

                for i, distance in enumerate(distance_to_ghosts):
                    if (
                        (distance < prev_distance_to_ghosts[i])  # closer to ghost
                        and (
                            distance < GHOST_DISTANCE_THRESHOLD
                        )  # withing certain distance (i.e., omit distant ghosts)
                        and (
                            getattr(state, f"ghost{i + 1}_state") not in [0, 4]
                        )  # Filter out dead ghosts
                    ):  # Not at home
                        self.value_per_pill[pellet_idx] += 1

                prev_distance_to_ghosts = distance_to_ghosts

                # Stop looking outside the search window, if player dies (Quite probable), or level end.
                died = state.lives < lives_at_wear_off
                level_end = i == len(gamestates) - 1
                if (
                    state.Index - starting_gamestep == SEARCH_WINDOW
                    or died
                    or level_end
                ):
                    if died:
                        self.died[pellet_idx] = True
                    if self.value_per_pill[pellet_idx] >= VALUE_THRESHOLD:
                        ending_gamestep = state.Index - 1
                        ending_timestep = float(
                            gamestates.at[state.Index - 1, "time_elapsed"]
                        )
                        if CONTEXT_LENGTH:
                            starting_gamestep = max(
                                0, starting_gamestep - CONTEXT_LENGTH
                            )
                            ending_gamestep = min(
                                gamestates.iloc[-1]["game_state_id"],
                                ending_gamestep + CONTEXT_LENGTH,
                            )

                        self.gamesteps[pellet_idx] = (
                            starting_gamestep,
                            ending_gamestep,
                        )
                        self.timesteps[pellet_idx] = (
                            starting_timestep,
                            ending_timestep,
                        )
                    else:
                        self.value_per_pill[pellet_idx] = 0

                    flag = False

            if (
                state.powerPellets == 0 and not flag
            ):  # If no remaining pellets, stop looking.
                break

        self.value = sum([value for value in self.value_per_pill if value != None])

        return

    def _Aggression6(self, gamestates: pd.DataFrame, **kwargs):
        """
        Chase Ghost or Collect Pellet When Hunting

        This behavlet measures whether the player actively chases ghosts or continues collecting pellets
        during power pill states. It tracks behavior for each power pill in the map, similar to Aggression4.

        The behavlet assigns a value to each power pill based on whether the player:
        1. Actively approaches ghosts during the power pill state
        2. Only considers the closest ghost if ONLY_CLOSEST_GHOST is True
        3. Must meet a minimum VALUE_THRESHOLD to be counted as a valid instance

        Parameters:
            VALUE_THRESHOLD (int): Minimum number of ghost approaches required to count as a valid instance
            ONLY_CLOSEST_GHOST (bool): If True, only considers the closest ghost when measuring approach
            CONTEXT_LENGTH (int): Number of frames to include before/after the power pill state

        Returns:
            A Behavlets object with:
            - value: Total number of valid instances across all power pills
            - value_per_pill: List of values for each power pill
            - gamesteps: List of tuples containing start/end frames for each instance
            - timesteps: List of tuples containing start/end times for each instance
        """

        self.full_name = "Aggression 6 - Chase Ghosts or Collect Pellets"
        self.measurement_type = "interval"

        VALUE_THRESHOLD = kwargs.get("VALUE_THRESHOLD", 1)
        ONLY_CLOSEST_GHOST = kwargs.get("ONLY_CLOSEST_GHOST", True)
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)
        NORMALIZE_VALUE = kwargs.get("NORMALIZE_VALUE", False)

        self.value_per_pill = [0] * 4
        self.gamesteps = [None] * 4
        if CONTEXT_LENGTH:
            self.original_gamesteps = [None] * 4

        self.timesteps = [None] * 4

        final_state = len(gamestates) - 1

        for i, state in enumerate(gamestates.itertuples()):
            if i == 0:
                prev_state = state
                ghost_distances = [math.inf] * 4
            elif i > 0:
                if prev_state.pacman_attack == 0 and state.pacman_attack == 1:
                    pellet_idx = self._get_quadrant_idx(state)
                    starting_gamestep = state.Index
                    starting_timestep = state.time_elapsed
                    end_timestep = state.Index
                    pacman_pos = (state.Pacman_X, state.Pacman_Y)
                    ghost_positions = self._get_ghost_pos(state, only_alive=False)
                    ghost_distances = [
                        abs(pacman_pos[0] - ghost_pos[0])
                        + abs(pacman_pos[1] - ghost_pos[1])
                        for ghost_pos in ghost_positions
                    ]

                elif state.pacman_attack == 1 and (i != final_state):
                    pacman_pos = (state.Pacman_X, state.Pacman_Y)
                    ghost_positions = self._get_ghost_pos(state, only_alive=False)
                    ghost_distances = [
                        abs(pacman_pos[0] - ghost_pos[0])
                        + abs(pacman_pos[1] - ghost_pos[1])
                        for ghost_pos in ghost_positions
                    ]

                    if ONLY_CLOSEST_GHOST:
                        for close_idx in np.argsort(ghost_distances):
                            if ghost_distances[close_idx] < prev_ghost_distances[
                                close_idx
                            ] and getattr(state, f"ghost{close_idx + 1}_state") not in [
                                0,
                                4,
                            ]:
                                self.value_per_pill[pellet_idx] += 1
                                break
                            else:  # If closest ghost is dead or at home, continue with the next closest one
                                continue

                    elif not ONLY_CLOSEST_GHOST:
                        for i, distance in enumerate(ghost_distances):
                            if distance < prev_ghost_distances[i] and getattr(
                                state, f"ghost{i + 1}_state"
                            ) not in [0, 4]:
                                self.value_per_pill[pellet_idx] += 1

                elif prev_state.pacman_attack == 1 and (
                    state.pacman_attack == 0 or i == final_state
                ):
                    end_gamestep = state.Index
                    end_timestep = state.time_elapsed
                    if self.value_per_pill[pellet_idx] >= VALUE_THRESHOLD:
                        if CONTEXT_LENGTH:
                            self.original_gamesteps[pellet_idx] = (
                                starting_gamestep,
                                end_gamestep,
                            )  # For normalization
                            starting_gamestep = max(
                                0, starting_gamestep - CONTEXT_LENGTH
                            )
                            end_gamestep = min(
                                gamestates.iloc[-1]["game_state_id"],
                                end_gamestep + CONTEXT_LENGTH,
                            )

                        self.gamesteps[pellet_idx] = (starting_gamestep, end_gamestep)
                        self.timesteps[pellet_idx] = (starting_timestep, end_timestep)
                    else:
                        self.value_per_pill[pellet_idx] = 0

                elif state.powerPellets == 0:
                    break

                prev_state = state
                prev_ghost_distances = ghost_distances

        if NORMALIZE_VALUE:
            for i, pill_value in enumerate(self.value_per_pill):
                if pill_value is not None and pill_value != 0:
                    if CONTEXT_LENGTH:
                        # Use original gamesteps without context length adjustment
                        self.value_per_pill[i] = pill_value / (
                            self.original_gamesteps[i][1]
                            - self.original_gamesteps[i][0]
                        )
                    else:
                        self.value_per_pill[i] = pill_value / (
                            self.gamesteps[i][1] - self.gamesteps[i][0]
                        )
            # Calculate simple average of normalized values
            valid_values = [v for v in self.value_per_pill if v is not None]
            self.value = sum(valid_values) / len(valid_values) if valid_values else 0
        else:
            valid_values = [v for v in self.value_per_pill if v is not None]
            self.value = sum(valid_values)

        return

    ##### Utility Functions

    def _get_ghost_pos(
        self, gamestate: NamedTuple, only_alive: bool
    ) -> list[np.ndarray]:
        """Get positions of all live ghosts (not dead or in house).

        Args:
            gamestate: Named tuple containing game state data

        Returns:
            List of numpy arrays containing (x,y) positions of live ghosts
        """
        if only_alive:
            return [
                np.array(
                    [
                        getattr(gamestate, f"Ghost{i}_X"),
                        getattr(gamestate, f"Ghost{i}_Y"),
                    ]
                )
                for i in range(1, 5)
                if getattr(gamestate, f"ghost{i}_state") not in [0, 4]
            ]
        else:
            return [
                np.array(
                    [
                        getattr(gamestate, f"Ghost{i}_X"),
                        getattr(gamestate, f"Ghost{i}_Y"),
                    ]
                )
                for i in range(1, 5)  # all ghosts
            ]

    def _get_quadrant_idx(self, state: NamedTuple) -> int:
        """Get index number based on quadrant position of Pacman.

        Args:
            state: Named tuple containing game state data

        Returns:
            Integer index (0-3) representing the quadrant:
            0: Top-left
            1: Top-right
            2: Bottom-right
            3: Bottom-left
        """
        pacman_pos = np.array([state.Pacman_X, state.Pacman_Y])

        # Quadrant based powerpill indexing (Aligned with Readme.md)
        if pacman_pos[0] < 0 and pacman_pos[1] > 0:
            return 0  # Top-left
        elif pacman_pos[0] > 0 and pacman_pos[1] > 0:
            return 1  # Top-right
        elif pacman_pos[0] > 0 and pacman_pos[1] < 0:
            return 2  # Bottom-right
        elif pacman_pos[0] < 0 and pacman_pos[1] < 0:
            return 3  # Bottom-left
        else:
            raise ValueError("Pacman position does not fall into any quadrant")
