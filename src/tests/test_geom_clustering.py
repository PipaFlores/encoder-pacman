import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
from src.analysis.geom_clustering import GeomClustering
from src.datahandlers.pacman_data_reader import PacmanDataReader

# Constants
TEST_DATA_FOLDER = "data"
# Use paths relative to the src/tests directory
BASE_DIR = Path(__file__).parent  # src/tests directory
BASELINE_DIR = BASE_DIR / "analysis/geom_clustering/baseline_images"
RESULT_DIR = BASE_DIR / "analysis/geom_clustering/result_images"
DIFF_DIR = BASE_DIR / "analysis/geom_clustering/diff_images"
# level_ids for baseline comparisons. These ids form two different clusters when analyzed in the 50 first timesteps (with euclidean HDBSCAN fit , min_cluster_size=20)
CLUSTER_1 = [
    1290,
    1281,
    1269,
    1262,
    1259,
    1256,
    1254,
    1227,
    1163,
    1157,
    1122,
    1092,
    1038,
    1034,
    1031,
    1029,
    1024,
    1016,
    997,
    996,
    931,
    930,
    925,
    924,
    917,
    895,
    871,
    854,
    839,
    827,
    826,
    825,
    823,
    814,
    797,
    791,
    778,
    720,
    695,
    694,
    693,
    604,
    603,
    592,
    580,
    569,
    568,
    565,
    551,
    535,
    505,
    502,
    501,
    494,
    490,
    489,
    479,
    421,
    420,
    404,
]
CLUSTER_2 = [
    1270,
    1268,
    1266,
    1265,
    1264,
    1263,
    1261,
    1258,
    1257,
    1251,
    1250,
    1249,
    1248,
    1167,
    1166,
    1161,
    1159,
    1154,
    1151,
    1149,
    1067,
    1064,
    1063,
    1040,
    935,
    829,
    815,
    745,
    737,
    733,
    729,
    708,
    707,
    706,
    678,
    667,
    657,
    649,
    646,
    615,
    613,
    593,
    574,
    563,
    561,
    557,
    555,
    552,
    550,
    548,
    531,
    465,
    422,
    411,
    409,
    405,
]
TEST_LEVEL_IDS = CLUSTER_1 + CLUSTER_2


@pytest.fixture(scope="module")
def reader():
    """Create a PacmanDataReader instance for testing"""
    return PacmanDataReader(data_folder=TEST_DATA_FOLDER)


@pytest.fixture(scope="module")
def test_trajectories(reader):
    """Create test trajectories for clustering, all equal length first 50 steps trajectories"""
    return [
        reader.get_partial_trajectory(level_id=lid, end_timestep=50)
        for lid in reader.level_df["level_id"].to_list()
    ]


@pytest.fixture(scope="module")
def clustering():
    """Create a GeomClustering instance for testing"""
    return GeomClustering(similarity_measure="euclidean", cluster_method="HDBSCAN")


@pytest.fixture(scope="module")
def fitted_clustering(clustering, test_trajectories):
    labels = clustering.fit(test_trajectories, min_cluster_size=20)
    return clustering, labels


@pytest.fixture(scope="module")
def setup_dirs():
    """Set up directories for test images"""
    # Create directories if they don't exist
    for directory in [BASELINE_DIR, RESULT_DIR, DIFF_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Return without cleaning up, so images persist after tests
    return


def compare_images(baseline_path, result_path, diff_path, tol=10):
    """
    Compare two images by displaying them side by side for manual inspection.

    Args:
        baseline_path: Path to baseline image
        result_path: Path to result image
        diff_path: Path to save comparison image
        tol: Tolerance for pixel differences

    Returns:
        bool: True if images match within tolerance or baseline doesn't exist, False otherwise
    """
    from PIL import Image
    import numpy as np

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

    # Convert images to numpy arrays for comparison
    baseline_array = np.array(baseline)
    result_array = np.array(result)

    # Calculate difference
    diff_array = np.abs(baseline_array - result_array)
    max_diff = np.max(diff_array)

    # Return True if differences are within tolerance
    if max_diff <= tol:
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

        print(f"Images differ by {max_diff} (tolerance: {tol})")
        return False


class TestGeomClustering:
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

    def test_affinity_matrix_calculation(self, clustering, test_trajectories):
        """Test affinity matrix calculation"""
        affinity_matrix = clustering.calculate_affinity_matrix(test_trajectories)

        # Check that the affinity matrix has the correct shape and properties
        assert clustering.similarity_measures.measure_type == "euclidean"
        assert affinity_matrix.shape == (len(test_trajectories), len(test_trajectories))
        assert np.all(np.diag(affinity_matrix) == 0)  # Diagonal should be zeros
        assert np.allclose(affinity_matrix, affinity_matrix.T)  # Should be symmetric

    def test_clustering_fit(self, fitted_clustering, test_trajectories):
        """Test clustering fit method"""
        clustering, labels = fitted_clustering

        # Check that labels has the correct length
        assert len(labels) == len(test_trajectories)
        # Check that benchmark clusters are foundh
        cluster_1 = clustering.get_cluster_elements(cluster_id=1, type="level_id")
        cluster_2 = clustering.get_cluster_elements(cluster_id=2, type="level_id")
        cluster_1 = [int(x) for x in cluster_1]
        cluster_2 = [int(x) for x in cluster_2]
        assert cluster_1 == CLUSTER_1
        assert cluster_2 == CLUSTER_2

    def test_plot_affinity_matrix(self, fitted_clustering, setup_dirs):
        """Test affinity matrix visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_affinity_matrix(ax=ax)

        assert self.save_and_compare(fig, "affinity_matrix", setup_dirs)

    def test_plot_distance_matrix_histogram(self, fitted_clustering, setup_dirs):
        """Test distance matrix histogram visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_distance_matrix_histogram(ax=ax)

        assert self.save_and_compare(fig, "distance_histogram", setup_dirs)

    def test_plot_non_repetitive_distances_values_barchart(
        self, fitted_clustering, setup_dirs
    ):
        """Test non-repetitive distances bar chart visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_non_repetitive_distances_values_barchart(ax=ax)

        assert self.save_and_compare(fig, "distances_barchart", setup_dirs)

    def test_plot_average_column_value(self, fitted_clustering, setup_dirs):
        """Test average column value visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_average_column_value(ax=ax)

        assert self.save_and_compare(fig, "average_column_value", setup_dirs)

    def test_plot_trajectories_embedding(self, fitted_clustering, setup_dirs):
        """Test trajectories embedding visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_trajectories_embedding(ax=ax)

        assert self.save_and_compare(fig, "trajectories_embedding", setup_dirs)

    def test_plot_clustering_centroids(self, fitted_clustering, setup_dirs):
        """Test clustering centroids visualization"""
        clustering, _ = fitted_clustering

        fig, ax = plt.subplots(figsize=(6, 6))
        clustering.plot_clustering_centroids(ax=ax)

        assert self.save_and_compare(fig, "clustering_centroids", setup_dirs)

    def test_plot_affinity_matrix_overview(self, fitted_clustering, setup_dirs):
        """Test affinity matrix overview visualization"""
        clustering, _ = fitted_clustering

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        clustering.plot_affinity_matrix_overview(axs=axs)
        fig.suptitle(
            f"Affinity Matrix Overview - {clustering.similarity_measures.measure_type.capitalize()} measure"
        )
        fig.tight_layout()
        # plt.show()

        assert self.save_and_compare(fig, "affinity_matrix_overview", setup_dirs)

    def test_plot_cluster_overview(self, fitted_clustering, setup_dirs):
        """Test cluster overview visualization"""
        clustering, _ = fitted_clustering

        # This method creates its own figure internally
        clustering.plot_cluster_overview(cluster_id=1)
        fig = plt.gcf()

        assert self.save_and_compare(fig, f"cluster_overview_{1}", setup_dirs)
