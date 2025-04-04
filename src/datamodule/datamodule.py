import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
from typing import Optional, List
import sys

sys.path.append("..")
from src.utils import PacmanDataReader


class TrajectoryDataset(Dataset):
    def __init__(
        self, trajectories: torch.Tensor, masks: torch.Tensor, game_ids: torch.Tensor
    ):
        """
        Args:
            trajectories: tensor of shape (n_trajectories, sequence_length, features)
            masks: tensor of shape (n_trajectories, sequence_length) indicating valid timesteps
            game_ids: tensor of shape (n_trajectories) containing the game ids
        """
        self.trajectories = trajectories
        self.masks = masks
        self.game_ids = game_ids

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return {
            "trajectory": self.trajectories[idx],
            "mask": self.masks[idx],
            "game_id": self.game_ids[idx],
        }


class TrajectoryDataModule(pl.LightningDataModule):
    """
    TrajectoryDataModule is a PyTorch Lightning DataModule for handling trajectory data.

    Args:
        data_folder (str): Path to the CSV file containing the raw game data.
        batch_size (int): Number of samples per batch to load. Default is 32.
        train_val_test_split (tuple): Proportions for splitting the data into training, validation, and test sets. Default is (0.7, 0.15, 0.15).
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.
        max_sequence_length (Optional[int]): Maximum sequence length for padding/truncating sequences. If None, use the longest sequence. Default is None.
        series_type (List[str]): List of series types to include in the preprocessing (e.g., ['position', 'movements', 'input']). Default is ['position'].
        include_game_state_vars (bool): Whether to include game state variables (score, powerPellets) in the features. Default is False.
        include_timesteps (bool): Whether to include time elapsed in the features. Default is True.

    Methods:
        setup(stage: str = None): Reads and converts the processed data to tensors and splits it into training, validation, and test sets.
        train_dataloader(): Returns the DataLoader for the training set.
        val_dataloader(): Returns the DataLoader for the validation set.
        test_dataloader(): Returns the DataLoader for the test set.

    ```python
    Example usage:
        data_module = TrajectoryDataModule(
            data_folder='path/to/your/data.csv',
            batch_size=32,
            max_sequence_length=100,
            series_type=['position', 'movements'],
            include_game_state_vars=True,
            include_timesteps=True
        )
        data_module.setup()
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        print(batch['trajectory'].shape)
        print(batch['mask'].shape)
        print(batch['game_id'].shape)
    ```
    """

    def __init__(
        self,
        data_folder: str,
        batch_size: int = 32,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 4,
        max_sequence_length: Optional[int] = None,
        series_type: List[str] = ["position"],
        include_game_state_vars: bool = False,
        include_timesteps: bool = True,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers

        # Data processing parameters
        self.max_sequence_length = max_sequence_length
        self.series_type = series_type
        self.include_game_state_vars = include_game_state_vars
        self.include_timesteps = include_timesteps

    def setup(self, stage: str = None):
        # Read data and Convert processed data to tensor

        datareader = PacmanDataReader(self.data_folder, read_games_only=True)

        trajectories_df = datareader.get_trajectory_dataframe(
            series_type=self.series_type,
            include_game_state_vars=self.include_game_state_vars,
            include_timesteps=self.include_timesteps,
        )

        trajectories, masks, game_ids = self._create_game_trajectory_tensor(
            trajectories_df, max_sequence_length=self.max_sequence_length
        )

        # Create dataset with trajectories, masks, and game_ids
        dataset = TrajectoryDataset(trajectories, masks, game_ids)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Store feature dimension for model initialization
        self.feature_dim = trajectories.shape[-1]
        self.seq_length = trajectories.shape[1]

    @property
    def feature_dimension(self):
        """Return the number of features in the data"""
        return self.feature_dim

    @property
    def sequence_length(self):
        """Return the sequence length of the trajectories"""
        return self.seq_length

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _create_game_trajectory_tensor(
        self, trajectories_df: pd.DataFrame, max_sequence_length: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates a tensor of shape (num_games, sequence_length, num_features) from a dataframe of trajectories.
        The dataframe must have a 'game_id' column.

        Args:
            trajectories_df: DataFrame containing all games' preprocessed data with game_id column
            max_sequence_length: Optional, pad/truncate all sequences to this length

        Returns:
            tensor: A tensor of shape (num_games, max_sequence_length, num_features)
            mask: A mask tensor of shape (num_games, max_sequence_length) indicating valid timesteps
            game_ids: A list of unique game IDs
        """
        # If max_sequence_length isn't provided, use the longest game
        if max_sequence_length is None:
            max_sequence_length = trajectories_df.groupby("game_id").size().max()

        game_ids = trajectories_df["game_id"].unique()
        # Determine the number of features from the DataFrame
        num_features = (
            trajectories_df.shape[1] - 1
        )  # Subtract 1 for the 'game_id' column
        num_games = len(game_ids)

        # Initialize tensor with zeros
        # Shape: (num_games, max_sequence_length, num_features)
        tensor = torch.zeros((num_games, max_sequence_length, num_features))

        # Create mask to track actual sequence lengths
        mask = torch.zeros((num_games, max_sequence_length), dtype=torch.bool)

        # Create dictionary to store game_id to index mapping
        game_to_idx = {game_id: idx for idx, game_id in enumerate(game_ids)}

        # Convert game_ids to a list and create ordered tensor of game IDs
        game_ids_list = list(game_ids)  # Convert from numpy array to list
        ordered_game_ids = torch.tensor(
            [game_ids_list[i] for i in range(len(game_ids_list))]
        )

        for game_id in game_ids:
            # Get game data excluding 'game_id' column
            game_data = (
                trajectories_df[trajectories_df["game_id"] == game_id]
                .iloc[:, 1:]
                .values
            )
            seq_len = min(len(game_data), max_sequence_length)
            game_idx = game_to_idx[game_id]

            # Fill tensor and mask
            tensor[game_idx, :seq_len, :] = torch.FloatTensor(game_data[:seq_len])
            mask[game_idx, :seq_len] = 1

        # The resulting tensor will have:
        # - First dimension: different games
        # - Second dimension: log steps in the game
        # - Third dimension: features (time,X, Y, score, powerPellets)

        return tensor, mask, ordered_game_ids
