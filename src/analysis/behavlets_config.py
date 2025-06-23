from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass
class Aggression1Config:
    """Hunt close to ghost house
    Counts instances where the player follows the ghosts right up to their house while attacking them.
    """
    HOUSE_POS: Tuple[float, float] = (0, -0.5)
    # CLOSENESS_DEF can be "House perimeter" or "Distance to house" and it
    # determines how the closeness to ghost house is defined, either by
    # a distance threshold or by a bounded region of space.
    CLOSENESS_DEF: str = "House perimeter"

    # If CLOSENESS_DEF == Distance to house,
    BOUNDARY_DISTANCE: float = 7.5

    # If CLOSENESS_DEF == "House Perimeter"
    # Defined as the grid region where close to the ghost house where there are no initial pellets.
    # (no need to traverse for level completion)
    X_BOUNDARIES: Tuple[float, float] = (-6.5, 6.5)
    Y_BOUNDARIES: Tuple[float, float] = (-5.5, 4.5)
    CONTEXT_LENGTH: Optional[int] = None

@dataclass
class Aggression3Config:
    """
    Ghost Kills
    """

    CONTEXT_LENGTH: int = 10


@dataclass
class Aggression4Config:
    """
    Hunter even after powerpill finishes
    """

    SEARCH_WINDOW: int = 10  # How many gamestates to check after powerpill ends.
    VALUE_THRESHOLD: int = 1
    GHOST_DISTANCE_THRESHOLD: int = 7
    CONTEXT_LENGTH: Optional[int] = None


@dataclass
class Aggression6Config:
    """
    Chase Ghosts or Collect Dots
    """

    VALUE_THRESHOLD: int = 1
    ONLY_CLOSEST_GHOST: bool = (
        True  # Is the behavlet considering all ghost or only the nearest one.
    )
    CONTEXT_LENGTH: Optional[int] = None
    NORMALIZE_VALUE: bool = (
        True  # Normalize the value according to overall duration of the hunt state
    )


class BehavletsConfig:
    """Configuration class for all behavlets parameters"""

    def __init__(self):
        self.aggression1 = Aggression1Config()
        self.aggression3 = Aggression3Config()
        self.aggression4 = Aggression4Config()
        self.aggression6 = Aggression6Config()

    def get_config(self, behavlet_name: str) -> Dict[str, Any]:
        """Get configuration for a specific behavlet"""
        config_map = {
            "Aggression1": self.aggression1.__dict__,
            "Aggression3": self.aggression3.__dict__,
            "Aggression4": self.aggression4.__dict__,
            "Aggression6": self.aggression6.__dict__,
        }
        return config_map.get(behavlet_name, {})
