import pytest
from src.datahandlers import PacmanDataReader
from src.analysis import BehavletsEncoding, Behavlets


@pytest.fixture(scope="module")
def reader():
    return PacmanDataReader(data_folder="data")


class TestBehavlets:
    def test_init(self):
        Beh_encodings = BehavletsEncoding(data_folder="data")

        assert Beh_encodings.reader != None
        assert len(Beh_encodings.reader.gamestate_df) > 0
        assert list(Beh_encodings.behavlets.keys()) == Behavlets.NAMES

        # Test individual Behavlets initialization
        for behavlet_name in Behavlets.NAMES:
            beh = Behavlets(name=behavlet_name)
            assert beh.name == behavlet_name
            assert beh.category in ["Aggression"]
            assert beh.value == 0
            assert beh.instances == 0
            assert beh.gamesteps == []
            assert beh.timesteps == []

        # Test invalid behavlet name raises error
        with pytest.raises(ValueError, match="Unknown behavlet name"):
            Behavlets(name="InvalidBehavlet")

    def test_calculation(self):
        Beh_encodings = BehavletsEncoding(data_folder="data")

        for level_id in range(400, 800):
            try:
                Beh_encodings.calculate_behavlets(level_id=level_id)
            except ValueError:
                continue

        assert Beh_encodings.summary_results.shape[0] >= 0
        assert Beh_encodings.instance_details.shape[0] >= 0

        ## TODO : Include tests for specific Behavlets

        Common_attributes = [
            "level_id",
            "instance_idx",
            "behavlet_name",
            "start_gamestep",
            "end_gamestep",
        ]
        # Aggression 1 - Hunt close to ghost home
        assert (
            Beh_encodings.summary_results[
                Beh_encodings.summary_results["Aggression1_value"] > 0
            ].shape[0]
            >= 0
        )
        assert (
            Beh_encodings.instance_details[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression1"
            ].shape[0]
            >= 0
        )

        Aggression1_instance_subset = Beh_encodings.instance_details.loc[
            Beh_encodings.instance_details["behavlet_name"] == "Aggression1",
            Common_attributes + ["value_per_instance"],
        ]

        empty_rows = Aggression1_instance_subset[
            Aggression1_instance_subset.isna().any(axis=1)
        ]
        assert empty_rows.empty, (
            f"Found {len(empty_rows)} rows with NaN values:\n{empty_rows}"
        )

        # Aggression 3 - Ghost kills

        assert (
            Beh_encodings.summary_results[
                Beh_encodings.summary_results["Aggression3_value"] > 0
            ].shape[0]
            >= 0
        )
        assert (
            Beh_encodings.instance_details[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression3"
            ].shape[0]
            >= 0
        )

        Aggression_3_instance_subset = Beh_encodings.instance_details.loc[
            Beh_encodings.instance_details["behavlet_name"] == "Aggression3",
            Common_attributes + ["instant_position", "instant_gamestep"],
        ]

        empty_rows = Aggression_3_instance_subset[
            Aggression_3_instance_subset.isna().any(axis=1)
        ]
        assert empty_rows.empty, (
            f"Found {len(empty_rows)} rows with NaN values:\n{empty_rows}"
        )

        # Aggression 4 - Hunt even after powerpill finishes

        assert (
            Beh_encodings.summary_results[
                Beh_encodings.summary_results["Aggression4_value"] > 0
            ].shape[0]
            >= 0
        )
        assert (
            Beh_encodings.instance_details[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression4"
            ].shape[0]
            >= 0
        )

        Aggression4_instance_subset = Beh_encodings.instance_details.loc[
            Beh_encodings.instance_details["behavlet_name"] == "Aggression4",
            Common_attributes + ["value_per_pill", "died"],
        ]

        empty_rows = Aggression4_instance_subset[
            Aggression4_instance_subset.isna().any(axis=1)
        ]
        assert empty_rows.empty, (
            f"Found {len(empty_rows)} rows with NaN values:\n{empty_rows}"
        )

        # Aggression 6 - Chase Ghosts or Collect Pellets

        assert (
            Beh_encodings.summary_results[
                Beh_encodings.summary_results["Aggression6_value"] > 0
            ].shape[0]
            >= 0
        )
        assert (
            Beh_encodings.instance_details[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression6"
            ].shape[0]
            >= 0
        )

        if Beh_encodings.behavlets["Aggression6"].kwargs["CONTEXT_LENGTH"] is not None:
            Aggression6_instance_subset = Beh_encodings.instance_details.loc[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression6",
                Common_attributes + ["value_per_pill", "original_gamesteps"],
            ]
        else:
            Aggression6_instance_subset = Beh_encodings.instance_details.loc[
                Beh_encodings.instance_details["behavlet_name"] == "Aggression6",
                Common_attributes + ["value_per_pill"],
            ]

        empty_rows = Aggression6_instance_subset[
            Aggression6_instance_subset.isna().any(axis=1)
        ]
        assert empty_rows.empty, (
            f"Found {len(empty_rows)} rows with NaN values:\n{empty_rows}"
        )

    def test_Aggression1(self, reader):
        Beh = Behavlets(name="Aggression1")
        gamestates = reader._filter_gamestate_data(level_id=603)[0]

        results = Beh.calculate(gamestates=gamestates)

        assert results == Beh
        assert results.value == 21
        assert results.value_per_instance == [21]
        assert results.instances == 1
        assert results.gamesteps == [(461349, 461370)]

        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 1 - Hunt close to ghost home"
        assert results.measurement_type == "interval"
        assert isinstance(results.value, int)
        assert isinstance(results.value_per_instance, list)
        assert len(results.gamesteps) == results.instances
        assert len(results.timesteps) == results.instances

        # Test with different parameters
        Beh_custom = Behavlets(name="Aggression1", CONTEXT_LENGTH=15)
        results_custom = Beh_custom.calculate(gamestates=gamestates)

        assert results_custom.instances >= 1
        assert results_custom.gamesteps == [(461334, 461385)]

        # Test with different closeness definition
        Beh_distance = Behavlets(name="Aggression1", CLOSENESS_DEF="Distance to house")
        results_distance = Beh_distance.calculate(gamestates=gamestates)

        assert results_distance.instances >= 1
        assert results_distance.value == 25
        assert results_distance.value_per_instance == [25]
        assert results_distance.gamesteps == [(461349, 461374)]

    def test_Aggression3(self, reader):
        Beh = Behavlets(name="Aggression3", CONTEXT_LENGTH=20)
        gamestates = reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(gamestates=gamestates)

        assert results == Beh
        assert results.value == 2
        assert results.gamesteps.__len__() == 2
        assert results.gamesteps == [(457810, 457850), (458549, 458589)]

        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 3 - Ghost kills"
        assert results.measurement_type == "point"
        assert (
            results.instances == results.value
        )  # For ghost kills, instances should equal value
        assert len(results.gamesteps) == results.instances

        # Test with default CONTEXT_LENGTH
        Beh_default = Behavlets(name="Aggression3")  # Default CONTEXT_LENGTH=10
        results_default = Beh_default.calculate(gamestates=gamestates)

        assert results_default.value == results.value  # Same number of kills
        assert results_default.gamesteps != results.gamesteps

        # Test invalid input
        with pytest.raises(TypeError, match="type\\(data\\) needs to be pd.Dataframe"):
            Beh.calculate("invalid_input")

    def test_Aggression4(self, reader):
        Beh = Behavlets(name="Aggression4")
        gamestates = reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(
            gamestates=gamestates,
            SEARCH_WINDOW=10,
            VALUE_THRESHOLD=1,
            GHOST_DISTANCE_THRESHOLD=7,
        )

        assert results == Beh
        assert results.instances == 1
        assert results.gamesteps.__len__() == 4
        assert results.died[3] == True
        assert results.gamesteps[3] == (457881, 457887)

        # Test additional attributes are set correctly
        assert results.full_name == "Aggresssion 4 - Hunt even after powerpill finishes"
        assert results.measurement_type == "interval"
        assert len(results.value_per_pill) == 4
        assert len(results.died) == 4
        assert len(results.gamesteps) == 4
        assert len(results.timesteps) == 4
        assert results.value == sum(
            [v for v in results.value_per_pill if v is not None]
        )

        # Test with different parameters
        results_custom = Beh.calculate(
            gamestates=gamestates,
            SEARCH_WINDOW=30,  # Larger search window
            VALUE_THRESHOLD=3,  # Higher threshold
            GHOST_DISTANCE_THRESHOLD=15,  # Larger distance threshold
        )

        assert results_custom.instances == 2
        assert results_custom.value == 31
        assert results_custom.value_per_pill == [0, 24, 0, 7]

        # Test with CONTEXT_LENGTH
        Beh_context = Behavlets(name="Aggression4", CONTEXT_LENGTH=15)
        results_context = Beh_context.calculate(gamestates=gamestates)

        assert results_context.instances == 1
        assert results_context.gamesteps[3] == (457866, 457902)

    def test_Aggression6(self, reader):
        Beh = Behavlets(name="Aggression6", NORMALIZE_VALUE=True)
        gamestates = reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(
            gamestates=gamestates,
        )

        assert Beh == results
        assert Beh.value == 0.19021739130434784
        assert Beh.value_per_pill == [0, 0.13043478260869565, 0, 0.25]
        assert Beh.gamesteps == [None, (458403, 458633), None, (457761, 457881)]

        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 6 - Chase Ghosts or Collect Pellets"
        assert results.measurement_type == "interval"
        assert len(results.value_per_pill) == 4
        assert len(results.gamesteps) == 4
        assert len(results.timesteps) == 4
        assert isinstance(results.value, float)  # Should be float when normalized

        # Test without normalization
        Beh_no_norm = Behavlets(name="Aggression6", NORMALIZE_VALUE=False)
        results_no_norm = Beh_no_norm.calculate(gamestates=gamestates)

        assert results_no_norm.value == sum(results_no_norm.value_per_pill)
        assert results_no_norm.value == 60

        # Test with ONLY_CLOSEST_GHOST=True
        Beh_all_ghosts = Behavlets(
            name="Aggression6", ONLY_CLOSEST_GHOST=True, NORMALIZE_VALUE=False
        )
        results_all_ghosts = Beh_all_ghosts.calculate(gamestates=gamestates)

        assert results_all_ghosts.instances == 2
        assert results_all_ghosts.value == 60

        # Test with CONTEXT_LENGTH and normalization interaction
        Beh_context_norm = Behavlets(
            name="Aggression6", NORMALIZE_VALUE=True, CONTEXT_LENGTH=25
        )
        results_context_norm = Beh_context_norm.calculate(gamestates=gamestates)

        assert results_context_norm.instances == 2
        assert results_context_norm.value == results.value
        assert results_context_norm.original_gamesteps == results.gamesteps
