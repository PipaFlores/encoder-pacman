
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
import numpy as np
import sys

sys.path.append("..")

class PacmanDataset(Dataset):
    ## TODO : this can be abstaracted as a "PaddedDataSet" or DataTensor. there is nothing specific to pacman
    def __init__(
        self, 
        gamestates: torch.Tensor | np.ndarray, 
        padding_value=-999,
    ):
        """
        PacmanDataset handles padded game state sequences for each trajectory.

        Note:
            Any required normalization of the data should be performed before creating this dataset,
            for example using a method in PacmanDataReader.

        Args:
            gamestates (torch.Tensor or np.ndarray): Array or tensor of shape (n_trajectories, sequence_length, features)
                containing the padded game state sequences per trajectory.
            padding_value (float, optional): Value used for padding invalid timesteps. Default: -999.

        Attributes:
            gamestates (torch.Tensor): Tensor of shape (n_trajectories, sequence_length, features), containing the data.
            padding_mask (torch.Tensor): Tensor of shape (n_trajectories, sequence_length), where 1 indicates valid (not padding), 0 otherwise.
            obs_mask (torch.Tensor): Tensor of shape (n_trajectories, sequence_length, features), 1 for finite values, 0 otherwise.
        """
        if isinstance(gamestates, torch.Tensor):
            self.gamestates = gamestates.copy()
        else:
            self.gamestates = torch.Tensor(gamestates.copy())

        self.padding_mask = (self.gamestates != padding_value).float()
        self.padding_mask = self.padding_mask.any(dim=-1).float()
        self.obs_mask = torch.isfinite(self.gamestates).float()

        if torch.isinf(self.gamestates).any():
            self.gamestates = replace_inf(self.gamestates)

    def __len__(self):
        return len(self.gamestates)

    def __getitem__(self, idx):
        return {
            "data": self.gamestates[idx],
            "padding_mask": self.padding_mask[idx],
            "obs_mask": self.obs_mask[idx]
        }
    



class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(self, 
                 gamestates : np.ndarray, 
                 mean_mask_length: int = 3, 
                 masking_ratio: float = 0.15,
                 mode: str = 'separate', 
                 distribution: str = 'geometric', 
                 exclude_feats: list[int] | None = None,
                 padding_value: int = -999):
        """
        A PyTorch Dataset that dynamically generates a random missingness (noise) mask for each sample at retrieval time,
        suitable for self-supervised masked imputation pretraining or evaluation.

        Args:
            gamestates (np.ndarray or torch.Tensor): Array of shape (num_samples, seq_length, features) containing the padded
                multivariate time-series data.
            mean_mask_length (int, optional): The mean length (in timesteps) of contiguous masked segments.
            masking_ratio (float, optional): Fraction of observed values to be masked (between 0 and 1).
            mode (str, optional): 'separate' (masks each variable independently) or 'block' (masks all features simultaneously per segment).
            distribution (str, optional): Distribution to sample mask lengths. Typically "geometric".
            exclude_feats (list[int] or None): Indices of features to exclude from masking.
            padding_value (float, optional): Value denoting padded (invalid) timesteps.

        Each time an item is sampled, a fresh noise_mask is generated, with 0 indicating masked/missing elements to be imputed, and 1 indicating unmasked elements.

        Attributes:
            gamestates: The tensor of input time series, with padding replaced and any infs handled.
            padding_mask: Binary tensor of valid timesteps per sample.
            observation_mask: Binary tensor of valid (finite) observed values per sample.
            masking_ratio: Target proportion of observed (non-pad) values to mask for imputation.
            mean_mask_length: Average length of consecutive masks (in timesteps).
            mode: How masking is applied ('separate' or 'block').
            distribution: The distribution from which mask lengths are sampled.
            exclude_feats: List of features (by index) not to be masked.
        """
        super(ImputationDataset, self).__init__()
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

        if isinstance(gamestates, torch.Tensor):
            self.gamestates = gamestates.copy()
        else:
            self.gamestates = torch.Tensor(gamestates.copy())

        self.padding_mask = (self.gamestates != padding_value).float()
        self.padding_mask = self.padding_mask.any(dim=-1).float()
        self.obs_mask = torch.isfinite(self.gamestates).float()


        if torch.isinf(self.gamestates).any():
            self.gamestates = replace_inf(self.gamestates)
    


    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape.
        The noise mask is recalculated each time a sample is accessed (data augmentation.)
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            noise_mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
        """

        X = self.gamestates[ind]  # (seq_length, feat_dim) array
        valid_steps = self.padding_mask[ind].bool()
        valid_len = int(valid_steps.sum().item())

        noise_mask = np.ones(X.shape, dtype=bool)
        if valid_len > 0:
            mask_core = create_noise_mask(
                X = X[:valid_len].cpu().numpy(),
                masking_ratio=self.masking_ratio,
                lm=self.mean_mask_length,
                mode=self.mode,
                distribution=self.distribution,
                exclude_feats=self.exclude_feats
            )
            noise_mask[:valid_len] = mask_core

        # noise_mask = create_noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
        #                   self.exclude_feats)  # (seq_length, feat_dim) boolean array

        return {
            "data": X,
            "noise_mask": torch.from_numpy(noise_mask).float(),
            "padding_mask": self.padding_mask[ind],
            "obs_mask": self.obs_mask[ind]
        }

    def update(self):
        """
        Makes training progressively harder by masking more of the input.
        Only if called during training (e.g., at the end of epochs)
        
        """
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.gamestates)

def replace_inf(array: np.ndarray):
    """
    Replace infinite values (np.inf or -np.inf) in a multi-dimensional array with the maximum finite value of 
    their corresponding feature (last dimension/column).

    This function:
    - Flattens the array across the first dimensions, preserving the last dimension as features.
    - Computes the maximum finite value for each feature, ignoring any infinite entries.
    - For each feature, all infinite values are replaced with that feature's maximum finite value.
      If an entire feature consists only of infinite values, those entries are set to 0.

    Args:
        array (np.ndarray): Input array of shape (..., feature_dim) containing numerical values. Infinite values 
            will be replaced.

    Returns:
        np.ndarray: Array of the same shape as the input, with infinite values replaced as described above.
    """
    flat = array.view(-1, array.shape[-1])
    # Mask out inf values for max computation
    finite_mask = torch.isfinite(flat)
    # For each feature, get max of finite values
    max_per_feature = torch.where(
        finite_mask.any(dim=0),
        torch.where(
            finite_mask, flat, float('-inf')
        ).max(dim=0).values,
        torch.zeros(flat.shape[1], device=flat.device)
    )
    # Now, replace inf/-inf in self.gamestates with max_per_feature
    inf_mask = torch.isinf(array)
    for feat in range(array.shape[-1]):
        array[..., feat][inf_mask[..., feat]] = max_per_feature[feat]
        # self.gamestates[..., feat][inf_mask[..., feat]] = 1e6 ## Unstable
    
    return array

def create_noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        elif mode == 'concurrent':  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        elif mode == 'concurrent':
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


class UCR_Dataset(torch.utils.data.Dataset):

    def __init__(self, ucr_dataset):
        """
        Utility class to train PyTorch models.
        Input a UCR classification dataset obtained with aeon.datasets.load_classification()
        The class will transpose the data from [N, channels, seq_length] -> [N, seq_length, channels]
        and store the data in batch["data"] and labels in batch["labels]
        """
        self.time_series = torch.Tensor(ucr_dataset[0]).transpose(1,2)
        self.labels = ucr_dataset[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            "data": self.time_series[idx],
            "labels": self.labels[idx]
        }
