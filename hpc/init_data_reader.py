import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datahandlers import PacmanDataReader


reader = PacmanDataReader(
    data_folder=os.path.join("..", "data"),
    force_preprocess=True,
    verbose = True
)