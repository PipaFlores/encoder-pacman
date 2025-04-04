"""
Default configuration values used across the project.
"""

from typing import Tuple
from threading import Lock


class SingletonMeta(type):
    """Thread-safe implementation of the Singleton pattern."""

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    """Main configuration class containing all default values."""

    def __init__(self):
        # Visualization settings
        self.figsize: Tuple[int, int] = (6, 6)

        # Distance measure settings (commented out for future use)
        # self.distance_measure: str = 'euclidean'


# Get the singleton instance
config = Config()

# Usage example:
# from config.defaults import config
# print(config.figsize)
