try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt

from src.datahandlers import PacmanDataset, ImputationDataset
from src.models import TSTransformerEncoder, AELSTM


class UtilVisualizer():

    def __init__(self):
        return

    def plot_reconstruction_comparison(
            self,
            sample: np.ndarray,
            sample_prediction: np.ndarray,
            feature_min: np.ndarray,
            feature_max: np.ndarray,
            features_columns: list[str],
            sample_id: int | None = None,
            figsize: tuple[int, int] | None = None,
        ):
        """
        Plot true vs reconstructed time series for one sample.

        Parameters
        ----------
        sample : np.ndarray
            True data sequence of shape (timesteps, num_features).
        sample_prediction : np.ndarray
            Reconstructed or predicted data for the same sequence.
        feature_min : np.ndarray
            Minimum value for each feature for plot y-axis limits.
        feature_max : np.ndarray
            Maximum value for each feature for plot y-axis limits.
        features_columns : list of str
            List of column/feature names.
        sample_id : int, optional
            If provided, adds sample_id to the plot title.
        figsize : tuple, optional
            Figure size for the matplotlib plot.
        """
        num_features = sample.shape[-1]
        timesteps = np.arange(sample.shape[0])
        if figsize is None:
            figsize = (15, num_features * 2)

        plt.figure(figsize=figsize)
        for feat_idx in range(num_features):
            plt.subplot(num_features, 1, feat_idx + 1)
            plt.plot(timesteps, sample[:, feat_idx], label="True", color="blue")
            plt.plot(
                timesteps,
                sample_prediction[:, feat_idx],
                label="Predicted",
                color="orange",
                linestyle="--",
            )
            plt.ylabel(f"{features_columns[feat_idx]}")
            plt.ylim(feature_min[feat_idx], feature_max[feat_idx] + 0.1)
            plt.legend(loc="best")
            plt.xlabel("Timestep")
        plt.tight_layout()
        title = "Reconstruction Comparison"
        if sample_id is not None:
            title += f" (sample_id: {sample_id})"
        plt.suptitle(title)
        plt.show()