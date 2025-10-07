try:
    from .datamodule import PacmanDataset, UCR_Dataset
except ImportError:
    PacmanDataset = None
    
from .trajectory import Trajectory
from .pacman_data_reader import PacmanDataReader

__all__ = ["PacmanDataset", "Trajectory", "PacmanDataReader", "UCR_Dataset"]
