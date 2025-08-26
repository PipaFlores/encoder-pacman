import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
import numpy as np
from typing import Optional, List
import sys

sys.path.append("..")
from .pacman_data_reader import PacmanDataReader

class PacmanDataset(Dataset):
    def __init__(
        self, 
        gamestates: torch.Tensor | np.ndarray, 
        padding_value=-999,
        # game_ids: torch.Tensor,
        # masks: torch.Tensor| None = None, 
    ):
        """
        Args:
            trajectories: tensor of shape (n_trajectories, sequence_length, features)
            masks: tensor of shape (n_trajectories, sequence_length) indicating valid timesteps
            game_ids: tensor of shape (n_trajectories) containing the game ids
        """
        if isinstance(gamestates, torch.Tensor):
            self.gamestates = gamestates
        else:
            self.gamestates = torch.Tensor(gamestates)

        self.masks = (self.gamestates != padding_value).float()
        self.masks = self.masks.any(dim=-1).float()
        # self.masks = masks
        # self.game_ids = game_ids

    def __len__(self):
        return len(self.gamestates)

    def __getitem__(self, idx):
        return {
            "data": self.gamestates[idx],
            "mask": self.masks[idx],
            # "level_id": self.game_ids[idx],
        }
    

    def _create_game_trajectory_tensor(
        self, trajectories_df: pd.DataFrame, max_sequence_length: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates a tensor of shape (num_games, sequence_length, num_features) from a dataframe of trajectories.
        The dataframe must have a 'level_id' column.

        Args:
            trajectories_df: DataFrame containing all games' preprocessed data with level_id column
            max_sequence_length: Optional, pad/truncate all sequences to this length

        Returns:
            tensor: A tensor of shape (num_games, max_sequence_length, num_features)
            mask: A mask tensor of shape (num_games, max_sequence_length) indicating valid timesteps
            game_ids: A list of unique game IDs
        """
        # If max_sequence_length isn't provided, use the longest game
        if max_sequence_length is None:
            max_sequence_length = trajectories_df.groupby("level_id").size().max()

        game_ids = trajectories_df["level_id"].unique()
        # Determine the number of features from the DataFrame
        num_features = (
            trajectories_df.shape[1] - 1
        )  # Subtract 1 for the 'level_id' column
        num_games = len(game_ids)

        # Initialize tensor with zeros
        # Shape: (num_games, max_sequence_length, num_features)
        tensor = torch.zeros((num_games, max_sequence_length, num_features))

        # Create mask to track actual sequence lengths
        mask = torch.zeros((num_games, max_sequence_length), dtype=torch.bool)

        # Create dictionary to store level_id to index mapping
        game_to_idx = {level_id: idx for idx, level_id in enumerate(game_ids)}

        # Convert game_ids to a list and create ordered tensor of game IDs
        game_ids_list = list(game_ids)  # Convert from numpy array to list
        ordered_game_ids = torch.tensor(
            [game_ids_list[i] for i in range(len(game_ids_list))]
        )

        for level_id in game_ids:
            # Get game data excluding 'level_id' column
            game_data = (
                trajectories_df[trajectories_df["level_id"] == level_id]
                .iloc[:, 1:]
                .values
            )
            seq_len = min(len(game_data), max_sequence_length)
            game_idx = game_to_idx[level_id]

            # Fill tensor and mask
            tensor[game_idx, :seq_len, :] = torch.FloatTensor(game_data[:seq_len])
            mask[game_idx, :seq_len] = 1

        # The resulting tensor will have:
        # - First dimension: different games
        # - Second dimension: log steps in the game
        # - Third dimension: features (time,X, Y, score, powerPellets)

        return tensor, mask, ordered_game_ids
