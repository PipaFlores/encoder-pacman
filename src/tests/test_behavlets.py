import pytest
import pandas as pd
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
        results = Beh_encodings.calculate_behavlets(
            level_id=600, behavlet_type="Aggression3"
        )

        ### TO COMPLETE FOR HIGHER-LEVEL CALCULATIONS WHEN MOST BEHAVLETS ARE IMPLEMENTED.

        assert True

    def test_Aggression1(self, reader):
        Beh = Behavlets(name="Aggression1")
        gamestates=reader._filter_gamestate_data(level_id=603)[0]

        results = Beh.calculate(
            gamestates=gamestates
        )

        assert results == Beh
        assert results.value == 25
        assert results.value_per_instance == [25]
        assert results.instances == 1
        assert results.gamesteps == [(461345, 461370)]
        
        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 1 - Hunt close to ghost home"
        assert results.measurement_type == "interval"
        assert isinstance(results.value, int)
        assert isinstance(results.value_per_instance, list)
        assert len(results.gamesteps) == results.instances
        assert len(results.timesteps) == results.instances
        
        # Test with different parameters
        Beh_custom = Behavlets(name="Aggression1", CONTEXT_LENGTH=15)
        results_custom = Beh_custom.calculate(
            gamestates=gamestates
        )
        
        
        assert results_custom.instances >= 1
        assert results_custom.gamesteps == [(461330, 461385)]
        
        # Test with different closeness definition
        Beh_distance = Behavlets(name="Aggression1", CLOSENESS_DEF="Distance to house")
        results_distance = Beh_distance.calculate(
            gamestates=gamestates
        )
        
        
        assert results_distance.instances >= 1
        assert results_distance.value == 31
        assert results_distance.value_per_instance == [31]
        assert results_distance.gamesteps == [(461343, 461374)]

    def test_Aggression3(self, reader):
        Beh = Behavlets(name="Aggression3", CONTEXT_LENGTH=20)
        gamestates = reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(
            gamestates=gamestates
        )

        assert results == Beh
        assert results.value == 2  
        assert results.gamesteps.__len__() == 2  
        assert results.gamesteps == [(457810, 457850), (458549, 458589)]  
        
        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 3 - Ghost kills"
        assert results.measurement_type == "point"
        assert results.instances == results.value  # For ghost kills, instances should equal value
        assert len(results.gamesteps) == results.instances
        
        # Test with default CONTEXT_LENGTH
        Beh_default = Behavlets(name="Aggression3")  # Default CONTEXT_LENGTH=10
        results_default = Beh_default.calculate(
            gamestates=gamestates
        )
        
        assert results_default.value == results.value  # Same number of kills
        assert results_default.gamesteps != results.gamesteps
        
        # Test invalid input
        with pytest.raises(TypeError, match="type\\(data\\) needs to be pd.Dataframe"):
            Beh.calculate("invalid_input")

    def test_Aggression4(self, reader):
        Beh = Behavlets(name="Aggression4")
        gamestates=reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(gamestates=gamestates,
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
        assert results.value == sum([v for v in results.value_per_pill if v is not None])
        
        # Test with different parameters
        results_custom = Beh.calculate(
            gamestates=gamestates,
            SEARCH_WINDOW=30,  # Larger search window
            VALUE_THRESHOLD=3,  # Higher threshold
            GHOST_DISTANCE_THRESHOLD=15,  # Larger distance threshold
        )
        
        
        assert results_custom.instances == 2
        assert results_custom.value == 31
        assert results_custom.value_per_pill == [0,24,0,7]
        
        # Test with CONTEXT_LENGTH
        Beh_context = Behavlets(name="Aggression4", CONTEXT_LENGTH=15)
        results_context = Beh_context.calculate(
            gamestates=gamestates
        )
        
        
        assert results_context.instances == 1
        assert results_context.gamesteps[3] == (457866, 457902)

    def test_Aggression6(self, reader):
        Beh = Behavlets(name="Aggression6", NORMALIZE_VALUE = True)
        gamestates = reader._filter_gamestate_data(level_id=600)[0]

        results = Beh.calculate(
            gamestates=gamestates,
            SEARCH_WINDOW=10,
            VALUE_THRESHOLD=1,
            GHOST_DISTANCE_THRESHOLD=7,
        ) 

        assert Beh == results
        assert Beh.value == 0.6519927536231884 
        assert Beh.value_per_pill == [0, 0.8913043478260869, 0.825, 0.8916666666666667]  
        assert Beh.gamesteps == [None, (458403, 458633), (458811, 458931), (457761, 457881)] 
        
        # Test additional attributes are set correctly
        assert results.full_name == "Aggression 6 - Chase Ghosts or Collect Pellets"
        assert results.measurement_type == "interval"
        assert len(results.value_per_pill) == 4
        assert len(results.gamesteps) == 4
        assert len(results.timesteps) == 4
        assert isinstance(results.value, float)  # Should be float when normalized
        
        # Test without normalization
        Beh_no_norm = Behavlets(name="Aggression6", NORMALIZE_VALUE=False)
        results_no_norm = Beh_no_norm.calculate(
            gamestates=gamestates
        )
        
        assert results_no_norm.value == sum(results_no_norm.value_per_pill)
        assert results_no_norm.value == 411
        
        
        # Test with ONLY_CLOSEST_GHOST=False
        Beh_all_ghosts = Behavlets(name="Aggression6", ONLY_CLOSEST_GHOST=False)
        results_all_ghosts = Beh_all_ghosts.calculate(
            gamestates=gamestates
        )
        
        
        assert results_all_ghosts.instances == 3
        assert results_all_ghosts.value == 953
        
        # Test with CONTEXT_LENGTH and normalization interaction
        Beh_context_norm = Behavlets(name="Aggression6", NORMALIZE_VALUE=True, CONTEXT_LENGTH=25)
        results_context_norm = Beh_context_norm.calculate(
            gamestates=gamestates
        )
        
        assert results_context_norm.instances == 3
        assert results_context_norm.value == results.value
        assert results_context_norm.original_gamesteps == results.gamesteps
