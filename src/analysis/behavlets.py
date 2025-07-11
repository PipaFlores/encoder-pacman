import pandas as pd
import numpy as np
import math
from typing import Callable, NamedTuple
from src.utils import setup_logger, Astar, generate_squared_walls, load_maze_data
from src.analysis.behavlets_config import BehavletsConfig


## TODO = Expand all behavlets to include simple features such as quadrant idx or geometrical space.

logger = setup_logger(__name__)


class Behavlets:
    NAMES = ["Aggression1", "Aggression3", "Aggression4", "Aggression6",
             "Caution1"]

    @property
    def ENCODING_FUNCTIONS(self) -> dict[str, Callable]:
        "Dictionary mapping behavlets name to their calculation/encoding algorithms"
        return {
            "Aggression1": self._Aggression1,
            "Aggression3": self._Aggression3,
            "Aggression4": self._Aggression4,
            "Aggression6": self._Aggression6,
            "Caution1": self._Caution1,
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
        elif debug:
            logger.setLevel("DEBUG")
            logger.debug(f"Initializing {name}")
        else:
            logger.setLevel("WARNING")

        self.config = BehavletsConfig()
        self.kwargs = {
            **self.config.get_config(name),
            **kwargs,
        }  # Store kwargs to be used in calculate() method
        logger.debug(f"config kwargs: {self.kwargs}")

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
        # continuous - is calculated during the whole game, it can also be calculated for any interval (e.g, Caution 2 Avg. distance to ghosts)

        ## this is the "output" of the original implementation.
        # Usually represents the number of behavlets observed. However, in some other, it represents other quantitative amounts
        # (e.g., Speed - cycles per sector is a value of time)
        self.value = 0

        # All behavlets output at least to the value attribute,
        # but it can be expanded to the other ones (handled by each behavlet algorithm)
        self.output_attributes = ["value"]

        ## list of tuples with (initial_gamestate, end_gamestate) that define the behavlet.
        # For some behavlet types, multiple can appear in a game, so there will be one tuple per each.
        self.gamesteps = []
        self.original_gamesteps = []  # For reference in case of CONTEXT_LENGTH adjustments
        self.timesteps = []

        self.instances = 0  # if multiple instances. Is len(gamesteps)

        ###
        ### OPTIONAL ATTRIBUTES SHARED OR UNIQUE OF CERTAIN BEHAVLET TYPES
        ###
        self.value_per_instance = []  ## e.g, Aggression 1 can have multiple instances and the value is different in each one.
        self.value_per_pill = []  ## e.g., Aggression4 counts for each pill in the game

        self.instant_gamestep = []  ## For point-based behavlets, the gamestep of the instant when the behavlet is observed
        self.instant_timestep = []  ## For point-based behavlets, the timestep of the instant when the behavlet is observed
        self.instant_position = []  ## For point-based behavlets, the position of the instant when the behavlet is observed

        self.died = []  ## Did the player died during the behavlet observation?

    def _reset_values(self):
        """Reset behavlet values to 0 for fresh calculations"""

        self.value = 0
        self.instances = 0
        self.gamesteps = []
        self.timesteps = []

        self.value_per_instance = []
        self.value_per_pill = []
        self.instant_gamestep = []
        self.instant_timestep = []
        self.instant_position = []
        self.died =[]

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
        self._reset_values()
        # Calculate Behavlets
        self.ENCODING_FUNCTIONS[self.name](gamestates, **{**self.kwargs, **kwargs})
        self.instances = len([x for x in self.gamesteps if x is not None])

        if self.full_name == "":
            raise ValueError(f"full_name not set for behavlet {self.name}")
        if self.measurement_type is None:
            raise ValueError(f"measurement_type not set for behavlet {self.name}")

        return self

    def report(self):
        """
        Report results based on self.output_attributes list of attributes"""

        print(f"Results for {self.full_name} ({self.name})")
        for attr in self.output_attributes:
            try:
                value = getattr(self, attr)
            except AttributeError:
                print(f"Attribute '{attr}' not found")
            print(f"{attr} : {value}")

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
        self.output_attributes = [
            "value",
            "instances",
            "value_per_instance",
            "gamesteps",
            "timesteps",
        ]

        CLOSENESS_DEF = kwargs.get("CLOSENESS_DEF")
        BOUNDARY_DISTANCE = kwargs.get("BOUNDARY_DISTANCE")
        HOUSE_POS = kwargs.get("HOUSE_POS")
        X_BOUNDARIES = kwargs.get("X_BOUNDARIES")
        Y_BOUNDARIES = kwargs.get("Y_BOUNDARIES")
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH")

        logger.debug(
            f"Calculating Aggression 1 with BOUNDARY_DISTANCE={BOUNDARY_DISTANCE}, "
            f"HOUSE_POS={HOUSE_POS}, "
            f"X_BOUNDARIES={X_BOUNDARIES}, "
            f"Y_BOUNDARIES={Y_BOUNDARIES}, "
            f"CLOSENESS_DEF={CLOSENESS_DEF}, "
            f"CONTEXT_LENGTH={CONTEXT_LENGTH}"
        )

        flag = False
        self.value_per_instance = []  # value for each instance observed.
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
                        if ghost_pos is not None
                    ]
                    Pacman_condition = Pacman_house_dist <= BOUNDARY_DISTANCE
                    Ghost_conditions = any(
                        dist <= BOUNDARY_DISTANCE for dist in Ghost_house_distances
                    )
                elif CLOSENESS_DEF == "House perimeter":
                    # logger.debug(f"step {state.Index} ghost positions{ghost_Positions}")
                    Pacman_condition = (
                        X_BOUNDARIES[0] <= Pacman_pos[0] <= X_BOUNDARIES[1]
                    ) and (Y_BOUNDARIES[0] <= Pacman_pos[1] <= Y_BOUNDARIES[1])
                    Ghost_conditions = any(
                        [
                            (X_BOUNDARIES[0] <= ghost_pos[0] <= X_BOUNDARIES[1])
                            and (Y_BOUNDARIES[0] <= ghost_pos[1] <= Y_BOUNDARIES[1])
                            for ghost_pos in ghost_Positions
                            if ghost_pos is not None
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
                    instance_gamestep , instance_timestep = self._end_instance(
                        state,
                        gamestates,
                        start_gamestep,
                        start_gamestep,
                        CONTEXT_LENGTH
                    )

                    self.gamesteps.append(instance_gamestep)
                    self.timesteps.append(instance_timestep)
                    self.value_per_instance.append(value)
                    flag = False

            elif state.pacman_attack == 0 and flag:
                instance_gamestep, instance_timestep = self._end_instance(
                    state,
                    gamestates,
                    start_gamestep,
                    start_timestep,
                    CONTEXT_LENGTH
                )
                self.value_per_instance.append(value)
                flag = False

        self.value = sum(self.value_per_instance)

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

        Parameters:
            gamestates (pd.DataFrame): DataFrame containing game state data
            CONTEXT_LENGTH (int): Number of frames to include before and after each ghost kill
                               for visualization purposes. Defaults to 10.

        Returns:
            None: Updates the behavlet's value, gamesteps, and timesteps attributes
        """
        self.full_name = "Aggression 3 - Ghost kills"
        self.measurement_type = "point"
        self.output_attributes = [
            "value",
            "instances",
            "gamesteps",
            "timesteps",
            "instant_gamestep",
            "instant_timestep",
            "instant_position",
        ]

        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", 10)

        logger.debug(f"Calculating Aggression 3 with CONTEXT_LENGTH={CONTEXT_LENGTH}")

        first_state = gamestates.iloc[0]
        previous_ghost_states = [
            first_state.ghost1_state,
            first_state.ghost2_state,
            first_state.ghost3_state,
            first_state.ghost4_state,
        ]

        for state in gamestates.itertuples():
            new_ghost_states = [
                state.ghost1_state,
                state.ghost2_state,
                state.ghost3_state,
                state.ghost4_state,
            ]
            for i in range(len(previous_ghost_states)):
                if previous_ghost_states[i] == 3 and new_ghost_states[i] == 4:
                    self.value += 1

                    self.gamesteps.append(
                        (
                            max(
                                gamestates.iloc[0]["game_state_id"],
                                state.Index - CONTEXT_LENGTH,
                            ),
                            min(
                                gamestates.iloc[-1]["game_state_id"],
                                state.Index + CONTEXT_LENGTH,
                            ),
                        )
                    )
                    self.instant_gamestep.append(state.Index)
                    self.instant_position.append((state.Pacman_X, state.Pacman_Y))
                    self.instant_timestep.append(state.time_elapsed)
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

        Parameters:
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
        self.output_attributes = [
            "value",
            "value_per_pill",
            "instances",
            "gamesteps",
            "timesteps",
            "died",
        ]

        SEARCH_WINDOW = kwargs.get("SEARCH_WINDOW", 10)
        VALUE_THRESHOLD = kwargs.get("VALUE_THRESHOLD", 1)
        GHOST_DISTANCE_THRESHOLD = kwargs.get("GHOST_DISTANCE_THRESHOLD", 5)
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)

        logger.debug(
            f"Calculating Aggression 4 with SEARCH_WINDOW={SEARCH_WINDOW}, "
            f"VALUE_THRESHOLD={VALUE_THRESHOLD}, "
            f"GHOST_DISTANCE_THRESHOLD={GHOST_DISTANCE_THRESHOLD}, "
            f"CONTEXT_LENGTH={CONTEXT_LENGTH}"
        )

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
                ghost_positions = self._get_ghost_pos(state, only_alive=True)

                prev_distance_to_ghosts = self._get_distance_to_ghosts(
                    pacman_pos, ghost_positions
                )

            # look gamesteps after pill wears off, within SEARCH_WINDOW, and check if distance to any ghosts diminish
            elif flag:
                pacman_pos = [state.Pacman_X, state.Pacman_Y]
                ghost_positions = self._get_ghost_pos(state, only_alive=True)

                distance_to_ghosts = self._get_distance_to_ghosts(
                    pacman_pos, ghost_positions
                )

                for i, distance in enumerate(distance_to_ghosts):
                    if (
                        (distance, prev_distance_to_ghosts[i]) != (math.inf, math.inf)
                    ) and (
                        (distance < prev_distance_to_ghosts[i])  # closer to ghost
                        and (
                            distance < GHOST_DISTANCE_THRESHOLD
                        )  # withing certain distance (i.e., omit distant ghosts)
                        and (
                            getattr(state, f"ghost{i + 1}_state") not in [0, 4]
                        )  # Filter out dead ghosts
                    ):  # Not at home
                        self.value_per_pill[pellet_idx] += 1

                prev_distance_to_ghosts = distance_to_ghosts.copy()

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
                    else:
                        self.died[pellet_idx] = False
                    if self.value_per_pill[pellet_idx] >= VALUE_THRESHOLD:
                        ending_gamestep = state.Index - 1
                        ending_timestep = float(
                            gamestates.at[state.Index - 1, "time_elapsed"]
                        )
                        if CONTEXT_LENGTH:
                            starting_gamestep = max(
                                gamestates.iloc[0]["game_state_id"],
                                starting_gamestep - CONTEXT_LENGTH,
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

        NOTE:
            Unnacounted situation: If player eats multiple powerpills and extend the effects, the
            instance is counted as one and assigned to the idx of the first pellet eaten.
        """

        self.full_name = "Aggression 6 - Chase Ghosts or Collect Pellets"
        self.measurement_type = "interval"
        self.output_attributes = [
            "value",
            "value_per_pill",
            "instances",
            "gamesteps",
            "timesteps",
        ]

        VALUE_THRESHOLD = kwargs.get("VALUE_THRESHOLD", 1)
        GHOST_DISTANCE_THRESHOLD = kwargs.get("GHOST_DISTANCE_THRESHOLD", 5)
        ONLY_CLOSEST_GHOST = kwargs.get("ONLY_CLOSEST_GHOST", False)
        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)
        NORMALIZE_VALUE = kwargs.get("NORMALIZE_VALUE", True)

        logger.debug(
            f"Calculating Aggression 6 with VALUE_THRESHOLD={VALUE_THRESHOLD}, "
            f"ONLY_CLOSEST_GHOST={ONLY_CLOSEST_GHOST}, "
            f"CONTEXT_LENGTH={CONTEXT_LENGTH}, "
            f"NORMALIZE_VALUE={NORMALIZE_VALUE}, "
            f"GHOST_DISTANCE_THRESHOLD={GHOST_DISTANCE_THRESHOLD}"
        )

        self.value_per_pill = [0] * 4
        self.gamesteps = [None] * 4
        if CONTEXT_LENGTH:
            self.original_gamesteps = [None] * 4
            self.output_attributes.append("original_gamesteps")

        self.timesteps = [None] * 4

        final_state = len(gamestates) - 1

        for i, state in enumerate(gamestates.itertuples()):
            if i == 0:
                prev_state = state
                ghost_distances = [math.inf] * 4
            elif i > 0:
                if (
                    prev_state.pacman_attack == 0 and state.pacman_attack == 1
                ):  # When powerpill is eaten
                    pellet_idx = self._get_quadrant_idx(state)
                    starting_gamestep = state.Index
                    starting_timestep = state.time_elapsed
                    end_timestep = state.Index
                    pacman_pos = (state.Pacman_X, state.Pacman_Y)
                    ghost_positions = self._get_ghost_pos(state, only_alive=True)
                    ghost_distances = self._get_distance_to_ghosts(
                        pacman_pos, ghost_positions
                    )

                elif state.pacman_attack == 1 and (
                    i != final_state
                ):  # While powerpill is active
                    pacman_pos = (state.Pacman_X, state.Pacman_Y)
                    ghost_positions = self._get_ghost_pos(state, only_alive=True)
                    ghost_distances = self._get_distance_to_ghosts(
                        pacman_pos, ghost_positions
                    )

                    if not ONLY_CLOSEST_GHOST:  ## Default behavior
                        for ghost_idx, distance in enumerate(ghost_distances):
                            if (
                                distance < prev_ghost_distances[ghost_idx]
                                and distance != math.inf
                                and prev_ghost_distances[ghost_idx] != math.inf
                                and distance < GHOST_DISTANCE_THRESHOLD
                            ):
                                self.value_per_pill[pellet_idx] += 1

                    elif ONLY_CLOSEST_GHOST:
                        closest_ghost_idx = np.argmin(ghost_distances)
                        if (
                            (
                                ghost_distances[closest_ghost_idx]
                                < prev_ghost_distances[closest_ghost_idx]
                            )
                            and (
                                ghost_distances[closest_ghost_idx]
                                < GHOST_DISTANCE_THRESHOLD
                            )
                            and (prev_ghost_distances[closest_ghost_idx] != math.inf)
                        ):
                            self.value_per_pill[pellet_idx] += 1

                elif prev_state.pacman_attack == 1 and (
                    state.pacman_attack == 0 or i == final_state
                ):  # When powerpill wears off, or at the end of the level
                    end_gamestep = state.Index
                    end_timestep = state.time_elapsed
                    if self.value_per_pill[pellet_idx] >= VALUE_THRESHOLD:
                        if CONTEXT_LENGTH:
                            self.original_gamesteps[pellet_idx] = (
                                starting_gamestep,
                                end_gamestep,
                            )  # For normalization
                            starting_gamestep = max(
                                gamestates.iloc[0]["game_state_id"],
                                starting_gamestep - CONTEXT_LENGTH,
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
                prev_ghost_distances = ghost_distances.copy()

        if NORMALIZE_VALUE:
            for pill_idx, pill_value in enumerate(self.value_per_pill):
                if pill_value is not None and pill_value != 0:
                    if CONTEXT_LENGTH:
                        # Use original gamesteps without context length adjustment
                        self.value_per_pill[pill_idx] = pill_value / (
                            self.original_gamesteps[pill_idx][1]
                            - self.original_gamesteps[pill_idx][0]
                        )
                    else:
                        self.value_per_pill[pill_idx] = pill_value / (
                            self.gamesteps[pill_idx][1] - self.gamesteps[pill_idx][0]
                        )
            # Calculate simple average of normalized values
            valid_values = [v for v in self.value_per_pill if v not in [None, 0]]
            self.value = sum(valid_values) / len(valid_values) if valid_values else 0
        else:
            valid_values = [v for v in self.value_per_pill if v not in [None, 0]]
            self.value = sum(valid_values)

        return
    
    def _Caution1(self, gamestates: pd.DataFrame, **kwargs):
        """
        Times trapped by Ghosts
        
        Pacman trapped in corridor by ghosts, and possibly losing a life.

        Parameters:
            gamestates (pd.DataFrame): DataFrame containing game state data
            CONTEXT_LENGTH (int): Number of frames to include before and after each instance
                               for visualization purposes. Defaults to 10.
            SEARCH_WINDOW (int): Number of frames to look ahead from the moment Pacman is trapped, to check if Pacman loses a life.
            GHOST_DISTANCE_THRESHOLD (int): Distance threshold to consider a ghost as a threat.

            
        Returns:
        """
        self.full_name = "Caution 1 - Times trapped by Ghosts"
        self.measurement_type = "interval"
        self.output_attributes = [
            "value", # Times trapped
            "value_per_instance", # Number of gamesteps trapped.
            "instances",
            "gamesteps",
            "timesteps",
            "died"
        ]

        CONTEXT_LENGTH = kwargs.get("CONTEXT_LENGTH", None)
        # SEARCH_WINDOW = kwargs.get("SEARCH_WINDOW", 10)
        GHOST_DISTANCE_THRESHOLD = kwargs.get("GHOST_DISTANCE_THRESHOLD", 5)
        OPPOSITE_POSITIONS = {
            0 : (12.5,-9.5),
            1: (-12.5,-9.5),
            2: (-12.5,8.5),
            3: (12.5, 8.5)
            }
        logger.debug(
            f"Calculating Caution 1 with CONTEXT_LENGTH={CONTEXT_LENGTH}, "
            # f"SEARCH_WINDOW={SEARCH_WINDOW}, "
            f"GHOST_DISTANCE_THRESHOLD={GHOST_DISTANCE_THRESHOLD}"
        )

        final_state = len(gamestates) - 1
        self.died = []
        self.value_per_instance = []
        died = False
        flag = False

        wall_grid = Astar.generate_squared_walls(load_maze_data()[0])

        for i, state in enumerate(gamestates.itertuples()):
            # logger.debug(f"checking state {i}")
            # if on final state, return. If behavlet flag active, terminate and append endstep.
            if i == final_state:
                if not flag:
                    return
                elif flag:
                    instance_gamestep, instance_timestep = self._end_instance(state, 
                                                                              gamestates, 
                                                                              start_gamestep=start_gamestep, 
                                                                              start_timestep=start_timestep,
                                                                              CONTEXT_LENGTH=CONTEXT_LENGTH)
                    if state.lives > 1:
                        died = False
                    elif state.lives == 1 and state.pellets >= 2:
                        died = True

                    self.gamesteps.append(instance_gamestep)
                    self.timesteps.append(instance_timestep)
                    self.value_per_instance.append(value_per_instance)
                    self.died.append(died)
                    logger.debug(f"trapped and ended game (died) on steps {instance_gamestep}")

            
            # Trapped but took powerpill
            elif flag and state.pacman_attack == 1:
                flag = False
                instance_gamestep, instance_timestep = self._end_instance(
                    state,
                    gamestates,
                    start_gamestep,
                    start_timestep,
                    CONTEXT_LENGTH
                )
                self.gamesteps.append(instance_gamestep)
                self.timesteps.append(instance_timestep)
                self.value_per_instance.append(value_per_instance)
                self.died.append(False)
                logger.debug(f"trapped but took powerpill. steps {instance_gamestep}")

            # Main logic
            elif state.pacman_attack == 0: 
                # Get relevant positions and transform to grid for Astar algorithm
                pacman_position = Astar.transform_to_grid((state.Pacman_X, state.Pacman_Y))
                ghost_positions = self._get_ghost_pos(state, only_alive=True)
                ghost_positions = [Astar.transform_to_grid(ghost_pos) for ghost_pos in ghost_positions if ghost_pos is not None]
                ghost_distances = self._get_distance_to_ghosts(pacman_pos=pacman_position, ghost_positions=ghost_positions)

                # Filter out ghosts based on distance threshold
                ghost_positions = [position for idx, position in enumerate(ghost_positions) if ghost_distances[idx] <= GHOST_DISTANCE_THRESHOLD]
                ghost_distances = [distance for distance in ghost_distances if distance <= GHOST_DISTANCE_THRESHOLD]
                quadrant = self._get_quadrant_idx(state)
                opposite_position = OPPOSITE_POSITIONS[quadrant]

                ghosts_in_valid_distance = len(ghost_distances) >= 2

                if not ghosts_in_valid_distance: # Avoid calculating Astar when not necessary, close instance if flagged.
                    if flag:
                       logger.debug(f"conditions in flagged state {state.Index}: ghosts in distance:{ghosts_in_valid_distance}, blocked :{blocked_path}")
                       flag = False
                       instance_gamestep, instance_timestep = self._end_instance(
                           state,
                           gamestates,
                           start_gamestep,
                           start_timestep,
                           CONTEXT_LENGTH
                       )
                       self.gamesteps.append(instance_gamestep)
                       self.timesteps.append(instance_timestep)
                       self.value_per_instance.append(value_per_instance)
                       self.died.append(False)
                       logger.debug(f"no longer trapped, instance finished at {instance_gamestep[1]}")

                    continue
                
                # logger.debug(f"Calculating astar from {grid_pac_pos} to {opposite_position}, blocked by {grid_ghost_positions}")
                _, astar_to_opposite = Astar.calculate_path_and_distance(start=pacman_position,
                                                                         goal=opposite_position,
                                                                         grid= wall_grid,
                                                                         blocked_positions=ghost_positions)
                
                blocked_path = astar_to_opposite == math.inf

                if flag:
                    logger.debug(f"conditions in flagged state {state.Index}: ghosts in distance:{ghosts_in_valid_distance}, blocked :{blocked_path}")
            
                # If two or more ghosts are in valid distance and blocked paths,
                    # create flag.
                    # get instance starting step
                    # increase value
                if not flag and ghosts_in_valid_distance and blocked_path:
                    flag = True
                    start_gamestep = state.Index
                    start_timestep = state.time_elapsed
                    self.value += 1
                    value_per_instance = 1
                    logger.debug(f"trapped at gamestep {start_gamestep}")

                    if flag and gamestates.iloc[i + 1]["lives"] < state.lives:
                        # If dead right after trapped, use gamestep before to avoid an len() == 1 interval 
                        start_gamestep -= 1
                        start_timestep = gamestates.loc[state.Index - 1, "time_elapsed"]
                        flag = False
                        instance_gamestep, instance_timestep = self._end_instance(
                            state,
                            gamestates,
                            start_gamestep,
                            start_timestep,
                            CONTEXT_LENGTH
                        )
                        self.gamesteps.append(instance_gamestep)
                        self.timesteps.append(instance_timestep)
                        self.died.append(True)
                        self.value_per_instance.append(value_per_instance)
                        logger.debug(f"Died right after trapped, ending instance with len(interval) = 2. gamesteps {instance_gamestep}")

                # Increase value for each state trapped.
                elif flag and ghosts_in_valid_distance and blocked_path:
                    value_per_instance += 1
                    logger.debug(f"still trapped at gamestep {state.Index}, increasing instance value")
                    # If dies, end instance
                    if flag and gamestates.iloc[i + 1]["lives"] < state.lives:
                        flag = False
                        instance_gamestep, instance_timestep = self._end_instance(
                            state,
                            gamestates,
                            start_gamestep,
                            start_timestep,
                            CONTEXT_LENGTH
                        )
                        self.gamesteps.append(instance_gamestep)
                        self.timesteps.append(instance_timestep)
                        self.died.append(True)
                        self.value_per_instance.append(value_per_instance)
                        logger.debug(f"Died after trapped, ending instance. gamesteps {instance_gamestep}")
                
                # If not trapped anymore, end instance
                elif flag and not (ghosts_in_valid_distance and blocked_path):
                    flag = False
                    instance_gamestep, instance_timestep = self._end_instance(
                        state,
                        gamestates,
                        start_gamestep,
                        start_timestep,
                        CONTEXT_LENGTH
                    )
                    self.gamesteps.append(instance_gamestep)
                    self.timesteps.append(instance_timestep)
                    self.value_per_instance.append(value_per_instance)
                    self.died.append(False)
                    logger.debug(f"no longer trapped, instance finished at {instance_gamestep[1]}")

        self._Caution1_post_process() ## Post-processing after calculations

    def _Caution1_post_process(self):
        """
        Post-processes the results of the Caution 1 (Times Trapped by Ghosts) behavlet.

        This method merges consecutive or closely occurring trapped instances that are separated by a small number of gamesteps,
        treating them as a single, continuous trapping event. Such fragmentation can occur if the trapped condition is briefly
        interrupted for only a few frames (e.g., due to minor state changes), but the overall context is still a single trapping.

        The merging is controlled by a configurable MERGE_THRESHOLD (default: 20 gamesteps), which determines the maximum allowed
        gap between the end of one instance and the start of the next for them to be merged. This threshold is based on empirical
        analysis of gameplay and is intended to robustly identify contiguous trapping episodes.
        """
         # Merge Caution1 instances that are close in time (gamesteps)
        if not self.gamesteps or len(self.gamesteps) <= 1:
            return  # Nothing to merge

        MERGE_THRESHOLD = self.kwargs.get("MERGE_THRESHOLD", 20)  # Gamesteps or frames between instances to consider merging
        
        logger.debug(f"Post-processing of results, merging proximal instances (MERGE_THRESHOLD: {MERGE_THRESHOLD})")

        merged_gamesteps = []
        merged_timesteps = []
        merged_value_per_instance = []
        merged_died = []

        # Prepare for merging
        current_start_gamestep = self.gamesteps[0][0]
        current_end_gamestep = self.gamesteps[0][1] 
        current_start_timestep = self.timesteps[0][0]
        current_end_timestep = self.timesteps[0][1]
        current_value_per_instance = self.value_per_instance[0]
        current_died = self.died[0]

        for i in range(1, len(self.gamesteps)):
            prev_end = current_end_gamestep
            this_start = self.gamesteps[i][0]
            this_end = self.gamesteps[i][1] 
            this_start_timestep = self.timesteps[i][0]
            this_end_timestep = self.timesteps[i][1]
            this_value_per_instance = self.value_per_instance[i]
            this_died = self.died[i]

            # If the gap between previous end and this start is less than threshold, merge
            if this_start - prev_end <= MERGE_THRESHOLD:
                # Extend current instance
                current_end_gamestep = this_end
                current_end_timestep = this_end_timestep
                current_value_per_instance += this_value_per_instance
                current_died = this_died  # Use the last instance's died status
                logger.debug(f"merged instances: start={current_start_gamestep}, end={current_end_gamestep}, value={current_value_per_instance}, died={current_died}")
            else:
                # Save current merged instance
                merged_gamesteps.append((current_start_gamestep, current_end_gamestep))
                merged_timesteps.append((current_start_timestep, current_end_timestep))
                merged_value_per_instance.append(current_value_per_instance)
                merged_died.append(current_died)
                # Start new instance
                current_start_gamestep = this_start
                current_end_gamestep = this_end
                current_start_timestep = this_start_timestep
                current_end_timestep = this_end_timestep
                current_value = this_value_per_instance
                current_died = this_died

        # Save the last merged instance
        merged_gamesteps.append((current_start_gamestep, current_end_gamestep))
        merged_timesteps.append((current_start_timestep, current_end_timestep))
        merged_value_per_instance.append(current_value_per_instance)
        merged_died.append(current_died)

        # Replace with merged results
        self.gamesteps = merged_gamesteps
        self.timesteps = merged_timesteps
        self.value_per_instance = merged_value_per_instance
        self.died = merged_died
        self.instances = len(self.gamesteps)
        self.value = self.instances


    def _Caution2a(self, gamestates: pd.DataFrame, **kwargs):
        """
        Average distance to ghosts - not on powerpill

        Average distance the player keeps from ghosts, when not on a powerpill

        """
        self.full_name = "Caution 2a - Average distance to ghosts - not on powerpill"
        self.measurement_type = "interval"
        self.output_attributes = [
            "value"
        ]


        final_state = len(gamestates) - 1
        self.died = []
        self.value_per_instance = []
        died = False
        flag = False

        return
    ##### Utility Functions

    def _get_ghost_pos(
        self, gamestate: NamedTuple, only_alive: bool
    ) -> list[np.ndarray | None]:
        """Get fixed-length list of positions of ghosts based on specified criteria.

        Args:
            gamestate: Named tuple containing game state data
            only_alive: If True, only return positions of live ghosts (not dead or in house), checked by both by ghost state and geometric position

        Returns:
            List of numpy arrays containing (x,y) positions of ghosts, with None for excluded ghosts
        """
        positions = [None] * 4
        HOUSE_X = [-3.5, 3.5]
        HOUSE_Y = [-2.5, 1.5]

        if only_alive:
            for i in range(1, 5):
                if (
                    getattr(gamestate, f"ghost{i}_state") not in [0, 4]
                    and not (  # and not in house (i.e., if ghost is in house, set position to None)
                        HOUSE_X[0] < getattr(gamestate, f"Ghost{i}_X") < HOUSE_X[1]
                        and HOUSE_Y[0] < getattr(gamestate, f"Ghost{i}_Y") < HOUSE_Y[1]
                    )
                ):
                    positions[i - 1] = np.array(
                        [
                            getattr(gamestate, f"Ghost{i}_X"),
                            getattr(gamestate, f"Ghost{i}_Y"),
                        ]
                    )
                else:
                    positions[i - 1] = None

        else:
            positions = [
                np.array(
                    [
                        getattr(gamestate, f"Ghost{i}_X"),
                        getattr(gamestate, f"Ghost{i}_Y"),
                    ]
                )
                for i in range(1, 5)  # all ghosts
            ]

        return positions

    def _get_distance_to_ghosts(
        self, pacman_pos: np.ndarray, ghost_positions: list[np.ndarray | None]
    ) -> list[float]:
        """Get Manhattan distance to ghosts from pacman position.
        inputs a fixed-length list of ghost positions and returns a list of distances to ghosts.
        If ghost position is None, the distance is set to math.inf.
        This is used to avoid calculating distance to ghosts that are not alive or in house.
        """
        distance_to_ghosts = [math.inf] * 4
        for ghost_idx, ghost_pos in enumerate(ghost_positions):
            if ghost_pos is None:
                distance_to_ghosts[ghost_idx] = math.inf
            else:
                distance_to_ghosts[ghost_idx] = abs(pacman_pos[0] - ghost_pos[0]) + abs(
                    pacman_pos[1] - ghost_pos[1]
                )
        return distance_to_ghosts

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
        elif pacman_pos[0] >= 0 and pacman_pos[1] >= 0:
            return 1  # Top-right
        elif pacman_pos[0] > 0 and pacman_pos[1] < 0:
            return 2  # Bottom-right
        elif pacman_pos[0] <= 0 and pacman_pos[1] <= 0:
            return 3  # Bottom-left
        else:
            raise ValueError("Pacman position does not fall into any quadrant")

    def _end_instance(self, state, gamestates ,start_gamestep, start_timestep, CONTEXT_LENGTH = None):
        """
        Sets the instance's start and end values for gamestep and timestep.
        
        This helper method calculates the final gamestep and timestep boundaries for a behavlet
        instance, optionally applying context length adjustments to include additional frames
        before and after the observed behavior.
        
        Parameters:
            state: Named tuple containing the current game state data
            gamestates (pd.DataFrame): DataFrame containing all game state data
            start_gamestep (int): The starting gamestep of the instance
            start_timestep (float): The starting timestep of the instance
            CONTEXT_LENGTH (int, optional): Number of frames to include before/after the instance
                                          for visualization purposes. Defaults to None.
        
        Returns:
            tuple: A tuple containing:
                - instance_gamesteps (tuple): (start_gamestep, end_gamestep) with optional context
                - instance_timesteps (tuple): (start_timestep, end_timestep), of the original behavior (no context modification ).
        """
        
        end_gamestep = state.Index
        end_timestep = state.time_elapsed

        if CONTEXT_LENGTH:
                start_gamestep = max(
                    gamestates.iloc[0]["game_state_id"],
                    start_gamestep - CONTEXT_LENGTH,
                )
                end_gamestep = min(
                    gamestates.iloc[-1]["game_state_id"],
                    end_gamestep + CONTEXT_LENGTH,
                )
        instance_gamesteps = (start_gamestep, end_gamestep)
        instance_timesteps = (start_timestep, end_timestep)

        return instance_gamesteps, instance_timesteps