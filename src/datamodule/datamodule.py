import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
import pandas as pd
from typing import Optional, List
import sys
import os
sys.path.append('..')
from src.utils.utils import preprocess_game_data, create_game_trajectory_tensor

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            trajectories: tensor of shape (n_trajectories, sequence_length, features)
            masks: tensor of shape (n_trajectories, sequence_length) indicating valid timesteps
        """
        self.trajectories = trajectories
        self.masks = masks
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'trajectory': self.trajectories[idx],
            'mask': self.masks[idx]
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
        prepare_data(): Reads and preprocesses the raw data.
        setup(stage: str = None): Converts the processed data to tensors and splits it into training, validation, and test sets.
        train_dataloader(): Returns the DataLoader for the training set.
        val_dataloader(): Returns the DataLoader for the validation set.
        test_dataloader(): Returns the DataLoader for the test set.

    Example usage:
        data_module = TrajectoryDataModule(
            data_folder='path/to/your/data.csv',
            batch_size=32,
            max_sequence_length=100,
            series_type=['position', 'movements'],
            include_game_state_vars=True,
            include_timesteps=True
        )
        data_module.prepare_data()
        data_module.setup()
        train_loader = data_module.train_dataloader()
    """
    def __init__(self, 
                 data_folder: str,
                 batch_size: int = 32,
                 train_val_test_split: tuple = (0.7, 0.15, 0.15),
                 num_workers: int = 4,
                 max_sequence_length: Optional[int] = None,
                 series_type: List[str] = ['position'],
                 include_game_state_vars: bool = False,
                 include_timesteps: bool = True):
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

    def prepare_data(self):
        
        BANNED_USERS = [42] # Myself

        # Read raw dat
        raw_df = pd.read_csv(os.path.join(self.data_folder, 'gamestate.csv'), converters={'user_id': lambda x: int(x),
                                                                                      'Pacman_X': lambda x: round(float(x), 2),
                                                                                      'Pacman_Y': lambda x: round(float(x), 2)
                                                                                      })
        game_df = pd.read_csv(os.path.join(self.data_folder, 'game.csv'), converters={'date_played': lambda x: pd.to_datetime(x)})
        banned_game_ids = game_df.loc[game_df['user_id'].isin(BANNED_USERS), 'game_id']

        raw_df = raw_df[~raw_df['game_id'].isin(banned_game_ids)]
        
        # Preprocess the data
        self.processed_df = preprocess_game_data(
            raw_df,
            series_type=self.series_type,
            include_game_state_vars=self.include_game_state_vars,
            include_timesteps=self.include_timesteps
        )

    def setup(self, stage: str = None):
        # Convert processed data to tensor
        trajectories, masks, game_ids = create_game_trajectory_tensor(
            self.processed_df,
            max_sequence_length=self.max_sequence_length
        )
        
        # Create dataset with both trajectories and masks
        dataset = TrajectoryDataset(trajectories, masks)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])
        
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
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
