import pandas as pd
import numpy as np
from typing import Callable


class Behavlets:
    NAMES = ["Aggression1", "Aggression3"]

    @property
    def ENCODING_FUNCTIONS(self) -> dict[str, Callable]:
        "Dictionary mapping behavlets name to their calculation/encoding algorithms"
        return {
            "Aggression1": self._Aggression1,
            "Aggression3": self._Aggression3,
        }

    def __init__(self, name: str, window_length=10):
        self.name = name  # Short name of behavlets (<category><number>)
        self.full_name = ""  # Full name of behavlets, as in literature. Attribute is set by specific functions when self.calculate()
        if name not in self.NAMES:
            raise ValueError(f"Unknown behavlet name: {name}")

        self.category = self._get_category(self.name)

        # How many gamestates before and after the behavlet definition are extracted as context (for visualization or trajectory/environment analysis)
        # e.g., for Aggresssion 3 - GhostKills, n# of gamesteps before and after the kill.
        self.WINDOW_LENGTH = window_length

        ## this is the "output" of the original implementation.
        # Usually represents the number of behavlets observed. However, in some other, it represents other quantitative amounts
        # (e.g., Speed - cycles per sector is a value of time)
        self.value = 0

        ## list of tuples with (initial_gamestate, end_gamestate) that define the behavlet.
        # For some behavlet types, multiple can appear in a game, so there will be one tuple per each.
        self.gamesteps = []
        self.timesteps = []

        self.instances = 0  # if multiple instances. Is len(gamesteps)

    def calculate(self, gamestates: pd.DataFrame):
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
        self.ENCODING_FUNCTIONS[self.name](gamestates)
        self.instances = len(self.gamesteps)

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

    def _Aggression1(
        self,
        gamestates: pd.DataFrame,
        BOUNDARY_DISTANCE: float = 7.5,
        HOUSE_POS: tuple = (0, -0.5),
        X_BOUNDARIES: tuple = (-6.5, 6.5),
        Y_BOUNDARIES: tuple = (-5.5, 3.5),
        CLOSENESS_DEF: str = "Distance to house",
    ):
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
            The minimum and maximum y-coordinates of the ghost perimeter area (default is (-5.5, 3.5)).
        CLOSENESS_DEF : str, optional
            Definition of "closeness", what is considered to be "right up to their house"
            used for the calculation either Distance to house (manhattan distance
            to house center, using Boundary_distance), or House perimeter (boundaries in space)

        Returns
        -------
        None
            Updates self.value, self.gamesteps, and self.timesteps with the results of the calculation.

        Notes
        -----
        - self.value: Number of times the player hunts close to the ghost home while attacking.
        - self.gamesteps: List of (start, end) indices for each detected instance.
        - self.timesteps: List of (start, end) time elapsed for each detected instance.
        """

        self.full_name = "Aggression 1 - Hunt close to ghost home"

        flag = False
        for i, gamestate in enumerate(gamestates.itertuples()):
            if gamestate.pacman_attack == 1:
                Pacman_pos = np.array([gamestate.Pacman_X, gamestate.Pacman_Y])
                Ghost_Positions = [
                    np.array(
                        [
                            getattr(gamestate, f"Ghost{i}_X"),
                            getattr(gamestate, f"Ghost{i}_Y"),
                        ]
                    )
                    for i in range(1, 5)
                    if getattr(gamestate, f"ghost{i}_state") not in [0, 4]
                ]  # omit dead ghosts

                if CLOSENESS_DEF == "Distance to house":
                    Pacman_house_dist = abs(Pacman_pos[0] - HOUSE_POS[0]) + abs(
                        Pacman_pos[1] - HOUSE_POS[1]
                    )
                    Ghost_house_distances = [
                        abs(HOUSE_POS[0] - ghost_pos[0])
                        + abs(HOUSE_POS[1] - ghost_pos[1])
                        for ghost_pos in Ghost_Positions
                    ]
                    Pacman_condition = Pacman_house_dist <= BOUNDARY_DISTANCE
                    Ghost_conditions = any(
                        dist <= BOUNDARY_DISTANCE for dist in Ghost_house_distances
                    )
                elif CLOSENESS_DEF == "House perimeter":
                    Pacman_condition = (
                        X_BOUNDARIES[0] <= Pacman_pos[0] <= X_BOUNDARIES[1]
                    ) and (Y_BOUNDARIES[0] <= Pacman_pos[1] <= Y_BOUNDARIES[1])
                    Ghost_conditions = [
                        (X_BOUNDARIES[0] <= ghost_pos[0] <= X_BOUNDARIES[1])
                        and (Y_BOUNDARIES[0] <= ghost_pos[0] <= Y_BOUNDARIES[1])
                        for ghost_pos in Ghost_Positions
                    ]

                if not flag and Pacman_condition and Ghost_conditions:
                    self.value += 1
                    start_gamestep = gamestate.Index
                    start_timestep = gamestate.time_elapsed
                    flag = True

                elif flag and (not Pacman_condition or not Ghost_conditions):
                    end_gamestep = gamestate.Index
                    end_timestep = gamestate.time_elapsed

                    self.gamesteps.append((start_gamestep, end_gamestep))
                    self.timesteps.append((start_timestep, end_timestep))
                    flag = False

            elif gamestate.pacman_attack == 0 and flag:
                end_gamestep = gamestate.Index
                end_timestep = gamestate.time_elapsed

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

    def _Aggression3(self, gamestates: pd.DataFrame):
        """
        Ghost Kills

        The amount of times that the player eats a ghost.
        """
        self.full_name = "Aggression 3 - Ghost kills"

        previous_ghost_states = [0, 0, 0, 0]

        for gamestate in gamestates.itertuples():
            new_ghost_states = [
                gamestate.ghost1_state,
                gamestate.ghost2_state,
                gamestate.ghost3_state,
                gamestate.ghost4_state,
            ]
            for i, state in enumerate(previous_ghost_states):
                if previous_ghost_states[i] != 4 & new_ghost_states[i] == 4:
                    self.value += 1
                    self.gamesteps.append(
                        (
                            gamestate.Index - self.WINDOW_LENGTH,
                            gamestate.Index + self.WINDOW_LENGTH,
                        )
                    )
                    self.timesteps.append(gamestate.time_elapsed)
            previous_ghost_states = new_ghost_states

        return
