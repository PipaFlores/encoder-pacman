import pandas as pd
from typing import Callable


class Behavlets:
    NAMES = ["Aggression1", "Aggression3"]

    @property
    def ENCODING_FUNCTIONS(self) -> dict[str, Callable]:
        "Dictionary mapping behavlets name to their calculation/encoding algorithms"
        return {"Aggression1": self._Aggression1, "Aggression3": self._Aggression3}

    def __init__(self, name: str, window_length=10):
        self.WINDOW_LENGTH = window_length

        self.name = name  # Short name of behavlets (<category><number>)
        self.full_name = ""  # Full name of behavlets, as in literature. Attribute is set by specific functions when self.calculate()
        if name not in self.NAMES:
            raise ValueError(f"Unknown behavlet name: {name}")

        self.category = self._get_category(self.name)

        ## this is the "output" of the original implementation.
        # Usually represents the number of behavlets observed. However, in some other, it represents other quantitative amounts
        # (e.g., Speed - cycles per sector is a value of time)
        self.value = 0

        ## list of tuples with (initial_gamestate, end_gamestate) that define the behavlet.
        # For some behavlet types, multiple can appear in a game, so there will be one tuple per each.
        self.gamesteps = []
        self.timesteps = []

        self.instances = 0  # if multiple instances. Is len(gamesteps)

        # How many gamestates before and after the behavlet definition are extracted as context (for visualization or trajectory/environment analysis)
        # e.g., for Aggresssion 3 - GhostKills, n# of gamesteps before and after the kill.

    def calculate(self, gamestates: pd.DataFrame):
        """
        Calculates the behavlet using the specific algorithm, according to behavlet name
        """
        if not isinstance(gamestates, pd.DataFrame):
            raise TypeError(
                f"type(data) needs to be pd.Dataframe, not {type(gamestates)}"
            )
        # Calculate Behavlets
        self.ENCODING_FUNCTIONS[self.name](gamestates)
        self.instances = len(self.gamesteps)

        return self

    def _Aggression1(self, data):
        raise NotImplementedError

    def _Aggression3(self, data: pd.DataFrame):
        """
        Ghost Kills

        The amount of times that the player eats a ghost.
        """
        self.full_name = "Aggression 3 - Ghost Kills"

        previous_ghost_states = [0, 0, 0, 0]

        for gamestate in data.itertuples():
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

    def _get_category(self, name: str) -> str:
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
