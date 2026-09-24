import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
from src.visualization.game_visualizer import GameVisualizer
from src.datahandlers.pacman_data_reader import PacmanDataReader

# Constants
TEST_DATA_FOLDER = "data"
# Use paths relative to the src/tests directory
BASE_DIR = Path(__file__).parent  # src/tests directory
BASELINE_DIR = BASE_DIR / "visualization/baseline_images"
RESULT_DIR = BASE_DIR / "visualization/result_images"
DIFF_DIR = BASE_DIR / "visualization/diff_images"
TEST_LEVEL_ID = 602  # Example level ID for testing
# Pinned explicitly (not taken from level_df order) so the baselines only change when the
# rendering does. 606-612 are filtered out by the reader, so the aggregate range starts at 613.
AGGREGATE_LEVEL_IDS = list(range(613, 663))
HEATMAP_LEVEL_IDS = [600, 601, 602]


@pytest.fixture(scope="module")
def visualizer():
    """Create a GameVisualizer instance for testing"""
    return GameVisualizer(data_folder=TEST_DATA_FOLDER)


@pytest.fixture(scope="module")
def reader():
    """Create a PacmanDataReader instance for testing"""
    return PacmanDataReader(data_folder=TEST_DATA_FOLDER)


@pytest.fixture(scope="module")
def setup_dirs():
    """Set up directories for test images"""
    # Create directories if they don't exist
    for directory in [BASELINE_DIR, RESULT_DIR, DIFF_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Return without cleaning up, so images persist after tests
    return


# Block size and tolerance for compare_images. Both images are averaged over BLOCK x BLOCK
# pixel blocks before comparing, so anti-aliasing and one-pixel text/line shifts (e.g. from a
# matplotlib upgrade) mostly average out, while changes to what is plotted do not. TOL is the
# mean absolute difference (0-255 grey levels) between the block-averaged images. Measured
# when it was set: matplotlib 3.10 -> 3.11 drift scored 0.13-0.59 on all plots but
# multiple_trajectories (2.46); real changes (a different game, the aggregated-heatmap state
# leak) scored 1.14-9.8.
BLOCK = 8
TOL = 0.8


def _block_mean(array, block):
    h, w = (array.shape[0] // block) * block, (array.shape[1] // block) * block
    return array[:h, :w].reshape(h // block, block, w // block, block, -1).mean(axis=(1, 3))


def compare_images(baseline_path, result_path, diff_path, tol=TOL, block=BLOCK):
    """
    Compare a result image against its baseline, saving a side-by-side comparison on failure.

    Args:
        baseline_path: Path to baseline image
        result_path: Path to result image
        diff_path: Path to save comparison image
        tol: Maximum mean absolute difference between the block-averaged images
        block: Size of the pixel blocks averaged before comparing

    Returns:
        bool: True if images match within tolerance or baseline doesn't exist, False otherwise
    """
    from PIL import Image

    # Check if baseline exists
    if not os.path.exists(baseline_path):
        # If running for the first time, save result as baseline
        if os.path.exists(result_path):
            shutil.copy(result_path, baseline_path)
            print(f"Created new baseline image: {baseline_path}")
        return True

    # Load images
    baseline = Image.open(baseline_path).convert("RGB")
    result = Image.open(result_path).convert("RGB")

    # Check dimensions
    if baseline.size != result.size:
        print(f"Image size mismatch: {baseline.size} vs {result.size}")
        return False

    # Signed arithmetic: subtracting the uint8 arrays directly wraps around (0 - 1 -> 255).
    baseline_array = np.asarray(baseline, dtype=np.float64)
    result_array = np.asarray(result, dtype=np.float64)
    mean_diff = np.abs(
        _block_mean(baseline_array, block) - _block_mean(result_array, block)
    ).mean()

    # Return True if differences are within tolerance
    if mean_diff <= tol:
        return True
    else:
        # Create side-by-side comparison for manual inspection
        width = baseline.width + result.width
        height = (
            max(baseline.height, result.height) + 30
        )  # Add extra height for headers
        comparison = Image.new("RGB", (width, height), "white")

        # Add headers
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw headers
        draw.text((baseline.width // 2 - 40, 5), "Baseline", fill="black", font=font)
        draw.text(
            (baseline.width + result.width // 2 - 30, 5),
            "Result",
            fill="black",
            font=font,
        )

        # Paste images side by side (with offset for header)
        comparison.paste(baseline, (0, 30))
        comparison.paste(result, (baseline.width, 30))

        # Save the comparison image
        os.makedirs(os.path.dirname(diff_path), exist_ok=True)
        comparison.save(diff_path)
        print(f"Saved side-by-side comparison to: {diff_path}")

        print(f"Images differ by {mean_diff:.3f} (tolerance: {tol})")
        return False


class TestGameVisualizer:
    def save_and_compare(self, fig, test_name, setup_dirs):
        """Save figure and compare with baseline"""
        result_path = RESULT_DIR / f"{test_name}.png"
        baseline_path = BASELINE_DIR / f"{test_name}.png"
        diff_path = DIFF_DIR / f"{test_name}_diff.png"

        # Ensure directories exist
        result_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the result image
        fig.savefig(result_path)
        plt.close(fig)

        # Compare with baseline or create baseline if it doesn't exist
        return compare_images(str(baseline_path), str(result_path), str(diff_path))

    def test_plot_heatmap(self, visualizer, setup_dirs):
        """Test heatmap visualization"""
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_heatmap(level_id=TEST_LEVEL_ID, ax=ax)

        assert self.save_and_compare(fig, "heatmap", setup_dirs)

    def test_plot_trajectory_line(self, visualizer, setup_dirs):
        """Test trajectory line visualization"""
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_trajectory_line(level_id=TEST_LEVEL_ID, ax=ax)

        assert self.save_and_compare(fig, "trajectory_line", setup_dirs)

    def test_plot_velocity_grid(self, visualizer, setup_dirs):
        """Test velocity grid visualization"""
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_velocity_grid(level_id=TEST_LEVEL_ID, ax=ax)

        assert self.save_and_compare(fig, "velocity_grid", setup_dirs)

    def test_plot_aggregated_velocity_grid(self, visualizer, setup_dirs, reader):
        """Test aggregated velocity grid visualization"""

        fig, ax = plt.subplots(figsize=(6, 6))
        trajectories = [
            reader.get_trajectory(level_id=level_id) for level_id in AGGREGATE_LEVEL_IDS
        ]
        visualizer.plot_velocity_grid(trajectory=trajectories, ax=ax)

        assert self.save_and_compare(fig, "aggregated_velocity_grid", setup_dirs)

    def test_plot_count_grid(self, visualizer, setup_dirs):
        """Test count grid visualization"""
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_count_grid(level_id=TEST_LEVEL_ID, ax=ax)

        assert self.save_and_compare(fig, "count_grid", setup_dirs)

    def test_plot_trajectory_scatter(self, visualizer, setup_dirs):
        """Test trajectory scatter visualization"""
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_trajectory_scatter(level_id=TEST_LEVEL_ID, ax=ax)

        assert self.save_and_compare(fig, "trajectory_scatter", setup_dirs)

    def test_plot_multiple_trajectories(self, visualizer, reader, setup_dirs):
        """Test multiple trajectories visualization"""
        # Get a few level IDs for testing
        level_ids = reader.level_df["level_id"].loc[600:603].tolist()

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        visualizer.plot_multiple_trajectories(
            level_ids=range(600, 604), plot_type="line", axs=axs, show_maze=True
        )

        assert self.save_and_compare(fig, "multiple_trajectories", setup_dirs)

    def test_custom_trajectory_input(self, visualizer, reader, setup_dirs):
        """Test visualization with custom trajectory input"""
        # Get trajectory data
        trajectory = reader.get_trajectory(level_id=TEST_LEVEL_ID)

        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_heatmap(trajectory=trajectory, ax=ax)

        assert self.save_and_compare(fig, "custom_trajectory", setup_dirs)

    def test_aggregated_heatmap(self, visualizer, reader, setup_dirs):
        """Test aggregated heatmap from multiple trajectories"""
        # Get multiple trajectories
        trajectories = [reader.get_trajectory(level_id=lid) for lid in HEATMAP_LEVEL_IDS]

        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer.plot_heatmap(trajectory=trajectories, ax=ax)

        assert self.save_and_compare(fig, "aggregated_heatmap", setup_dirs)
