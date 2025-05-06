from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple, List
import numpy as np
from pathlib import Path
import json
from src.utils import setup_logger

# Initialize module-level logger
logger = setup_logger(__name__)


@dataclass
class Trajectory:
    """
    A class representing a game trajectory with both spatial data and metadata.

    Attributes:
        coordinates: Array of shape (num_timesteps, 2) containing x,y coordinates
        timevalues: Optional array of shape (num_timesteps,) containing time values
        metadata: Optional dictionary containing additional game information, currently includes
            - game_id: The ID of the game
            - user_id: The ID of the user
            - session_number: The number of the session
            - game_in_session: The number of the game in the session
            - total_games_played: The total number of games played
            - game_duration: The duration of the game
            - win: Whether the game was won
            - level: The level of the game

    """

    coordinates: np.ndarray
    timevalues: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Ensure coordinates are numpy array with correct shape
        self.coordinates = np.array(self.coordinates)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError(
                "Coordinates must be a 2D array with shape (n_timesteps, 2)"
            )

        if self.timevalues is not None:
            self.timevalues = np.array(self.timevalues)
            if len(self.timevalues) != len(self.coordinates):
                raise ValueError("Timevalues and coordinates must have the same length")

        # Initialize empty metadata dict if None
        if self.metadata is None:
            self.metadata = {}

    def __getitem__(
        self, key: Union[int, slice, Tuple[slice, int], Tuple[slice, slice]]
    ) -> Union[Tuple[float, float], "Trajectory", np.ndarray]:
        """
        Support various indexing operations on the trajectory.

        Args:
            key: Can be:
                - Integer index (returns single point)
                - Slice (returns sub-trajectory)
                - Tuple of (slice/int, int) for accessing specific coordinate arrays

        Returns:
            - Single point as tuple
            - New Trajectory object
            - Numpy array for coordinate selections
        """
        if isinstance(key, tuple):
            # Handle multi-dimensional indexing (e.g., trajectory[:, 0])
            return self.coordinates[key]
        elif isinstance(key, int):
            # Return single timestep as (x, y) tuple
            return tuple(self.coordinates[key])
        elif isinstance(key, slice):
            # Return new Trajectory object with sliced coordinates
            return Trajectory(coordinates=self.coordinates[key], metadata=self.metadata)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self) -> int:
        """Return number of timesteps"""
        return len(self.coordinates)

    def __array__(
        self, dtype: Optional[np.dtype] = None, copy: bool = True
    ) -> np.ndarray:
        """Allow numpy array conversion for backward compatibility"""
        return self.coordinates

    @property
    def x(self) -> np.ndarray:
        """Get x coordinates"""
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Get y coordinates"""
        return self.coordinates[:, 1]

    def get_segment(self, start_step: int = 0, end_step: int = -1) -> "Trajectory":
        """Get a segment of the trajectory"""
        if end_step >= len(self.coordinates):
            logger.warning(
                f"Trajectory for game {self.metadata['game_id'] if self.metadata else 'NA'} ends before the inputed end step ({len(self.coordinates)} < {end_step}). Returning segment until last observed position instead."
            )
            end_step = len(self.coordinates)
        if end_step == -1:
            end_step = len(self.coordinates)

        if start_step > end_step:
            raise ValueError("start_step must be less than end_step")

        return Trajectory(
            coordinates=self.coordinates[start_step:end_step],
            timevalues=self.timevalues[start_step:end_step]
            if self.timevalues is not None
            else None,
            metadata=self.metadata,
        )

    def reduce_granularity(self, n: int):
        """
        Reduces the granularity (resolution) of a trajectory by just extracting datapoints every n steps in the time-series.

        Args:
            n: Number of steps in the original time-series used to extract the reduced trajectory
        """

        return self[range(0)]

    @classmethod
    def save_trajectories(cls, trajectories: List["Trajectory"], filepath: str):
        """
        Save a list of trajectories to disk.

        Args:
            trajectories: List of Trajectory objects
            filepath: Path to save the data (without extension)
        """
        # Convert filepath to Path object
        filepath = Path(filepath)

        # Find the maximum length among all trajectories
        max_length = max(len(t.coordinates) for t in trajectories)
        n_trajectories = len(trajectories)

        # Create a padded array filled with zeros
        coords_array = np.zeros((n_trajectories, max_length, 2))

        # Fill the array with actual coordinates
        for i, traj in enumerate(trajectories):
            coords_array[i, : len(traj.coordinates)] = traj.coordinates

        # Save coordinates using numpy's compressed format
        np.savez_compressed(
            filepath.with_suffix(".npz"),
            trajectories=coords_array.reshape(n_trajectories, max_length * 2),
            shape=np.array([n_trajectories, max_length, 2]),
        )

        # Save metadata separately as JSON, converting numpy types to native Python types
        metadata_list = []
        for i, t in enumerate(trajectories):
            metadata = t.metadata if t.metadata is not None else {}
            # Convert numpy types to native Python types in metadata
            converted_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                converted_metadata[key] = value

            metadata_list.append({"index": i, "metadata": converted_metadata})

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

    @classmethod
    def load_trajectories(cls, filepath: str) -> List["Trajectory"]:
        """
        Load trajectories from disk.

        Args:
            filepath: Path to the saved data (without extension)

        Returns:
            List of Trajectory objects
        """
        filepath = Path(filepath)

        # Load coordinate data
        with np.load(filepath.with_suffix(".npz")) as data:
            shape = data["shape"]
            n_trajectories, n_timesteps, n_coords = shape
            coords_array = data["trajectories"].reshape(
                n_trajectories, n_timesteps, n_coords
            )

        # Load metadata if it exists
        metadata_list = [{}] * n_trajectories  # Default empty metadata
        if filepath.with_suffix(".json").exists():
            with open(filepath.with_suffix(".json"), "r") as f:
                metadata_data = json.load(f)
                metadata_list = [None] * n_trajectories
                for item in metadata_data:
                    metadata_list[item["index"]] = item["metadata"]

        # Reconstruct trajectory objects
        trajectories = [
            cls(coordinates=coords_array[i], metadata=metadata_list[i])
            for i in range(n_trajectories)
        ]

        return trajectories
