import pytest
import numpy as np
from src.datahandlers.pacman_data_reader import PacmanDataReader
from src.datahandlers.trajectory import Trajectory


# Create a standard DataReader instance for all tests
@pytest.fixture(scope="module")
def reader():
    """Create a standard DataReader instance for all tests"""
    return PacmanDataReader(data_folder="data")


class TestPacmanDataReader:
    def test_singleton_pattern(self, reader):
        """Test that PacmanDataReader implements singleton pattern correctly"""
        reader1 = PacmanDataReader(data_folder="data")
        reader2 = PacmanDataReader(data_folder="data/pilot")
        assert reader1 is reader2
        assert reader1.data_folder == "data"  # Should keep first initialization
        assert reader is reader1  # Our fixture should be the same instance

    def test_initialization(self, reader):
        """Test basic initialization and data loading"""
        assert reader.level_df is not None
        assert reader.gamestate_df is not None
        assert len(reader.level_df) > 0
        assert len(reader.gamestate_df) > 0

        # Verify data relationships
        assert set(reader.gamestate_df["level_id"]).issubset(
            set(reader.level_df["level_id"])
        )

    def test_banned_users_filtering(self, reader):
        """Test that banned users are properly filtered out"""
        assert 42 not in reader.level_df["user_id"].values
        assert 419 not in reader.gamestate_df["level_id"].values

    def test_get_trajectory(self, reader):
        """Test trajectory extraction functionality"""
        # Get a valid level_id from the data
        test_level_id = reader.level_df["level_id"].iloc[0]

        # Test getting trajectory by level_id
        trajectory = reader.get_trajectory(level_id=test_level_id)
        assert isinstance(trajectory, Trajectory)
        assert trajectory.coordinates.shape[1] == 2  # x,y coordinates

        # Test getting trajectory with time values
        trajectory = reader.get_trajectory(level_id=test_level_id, get_timevalues=True)
        assert trajectory.timevalues is not None
        assert len(trajectory.timevalues) == len(trajectory.coordinates)

        # Test getting trajectory with metadata
        trajectory = reader.get_trajectory(level_id=test_level_id)
        assert trajectory.metadata is not None
        assert list(trajectory.metadata.keys()) == [
            "level_id",
            "game_id",
            "user_id",
            "session_number",
            "level_in_session",
            "total_levels_played",
            "duration",
            "win",
            "level",
        ]

    def test_get_partial_trajectory(self, reader):
        """Test partial trajectory extraction"""
        # Get a valid level_id from the data
        test_level_id = reader.level_df["level_id"].iloc[0]

        # Get full trajectory
        full_trajectory = reader.get_trajectory(level_id=test_level_id)

        # Get partial trajectory
        partial_trajectory = reader.get_partial_trajectory(
            level_id=test_level_id, start_timestep=0, end_timestep=2
        )

        assert len(partial_trajectory) <= len(full_trajectory)
        assert np.array_equal(
            partial_trajectory.coordinates, full_trajectory.coordinates[:2]
        )

        assert full_trajectory.metadata == partial_trajectory.metadata

    def test_get_trajectory_dataframe(self, reader):
        """Test trajectory dataframe generation"""
        # Get a valid level_id from the data
        test_level_id = reader.level_df["level_id"].iloc[0]

        # Test with different series types
        df = reader.get_trajectory_dataframe(
            level_id=test_level_id,
            series_type=["position", "movement"],
            include_game_state_vars=True,
            include_timesteps=True,
        )

        assert "Pacman_X" in df.columns
        assert "Pacman_Y" in df.columns
        assert "movement_dx" in df.columns
        assert "movement_dy" in df.columns
        assert "score" in df.columns
        assert "time_elapsed" in df.columns

    def test_psychometric_processing(self, reader):
        """Test psychometric data processing"""
        # Create a new reader with read_games_only=False to get psychometric data
        psych_reader = PacmanDataReader(data_folder="data", read_games_only=False)

        # Check BISBAS data
        assert psych_reader.bisbas_df is not None
        assert "BIS" in psych_reader.bisbas_df.columns
        assert "REW" in psych_reader.bisbas_df.columns
        assert "DRIVE" in psych_reader.bisbas_df.columns
        assert "FUN" in psych_reader.bisbas_df.columns

        # Check Flow measures
        assert psych_reader.game_flow_df is not None
        assert psych_reader.game_flow_df.columns.to_list() == [
            "user_id",
            "FLOW",
            "total_games_played",
            "max_score",
            "log(max_score)",
            "inv(max_score)",
            "log(inv(max_score))",
            "log(total_games_played)",
            "flow_z_score",
            "cum_score",
            "log(cum_score)",
            "score_deviation",
        ]

        # Verify it's the same instance
        assert reader is psych_reader

    def test_error_handling(self, reader):
        """Test error handling for invalid inputs"""
        # Test invalid level_id
        with pytest.raises(ValueError):
            reader.get_trajectory(level_id=None)

        # Test invalid trajectory segment
        test_level_id = reader.level_df["level_id"].iloc[0]
        with pytest.raises(IndexError):
            reader.get_partial_trajectory(
                level_id=test_level_id, start_timestep=-10, end_timestep=200
            )
