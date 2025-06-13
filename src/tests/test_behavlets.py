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

    def test_calculation(self):
        # TODO Complete when behavlets are done.

        Beh_encodings = BehavletsEncoding(data_folder="data")
        results = Beh_encodings.calculate_behavlets(
            level_id=600, behavlet_type="Aggression3"
        )

        assert True

    def test_Aggression1(self, reader):
        Beh = Behavlets(name="Aggression1")

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=603)[0]
        )

        assert results == Beh
        assert sum(results.value) == 25
        assert results.instances == 1
        assert results.gamesteps == [(461345, 461370)]

    def test_Aggression3(self, reader):
        Beh = Behavlets(name="Aggression3", CONTEXT_LENGTH=20)

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=600)[0]
        )

        assert results == Beh
        assert results.value == 2
        assert results.gamesteps.__len__() == 2
        assert results.gamesteps == [(457810, 457850), (458549, 458589)]

    def test_Aggression4(self, reader):
        Beh = Behavlets(name="Aggression4")

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=600)[0],
            SEARCH_WINDOW=10,
            VALUE_THRESHOLD=1,
            GHOST_DISTANCE_THRESHOLD=7,
        )

        assert results == Beh
        assert results.instances == 1
        assert results.gamesteps.__len__() == 4
        assert results.died[3] == True
        assert results.gamesteps[3] == (457881, 457887)


    def test_Aggression6(self, reader):
        Beh = Behavlets(name="Aggression6", NORMALIZE_VALUE = True)

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=600)[0],
            SEARCH_WINDOW=10,
            VALUE_THRESHOLD=1,
            GHOST_DISTANCE_THRESHOLD=7,
        ) 

        assert Beh == results
        assert Beh.value == 0.6519927536231884
        assert Beh.value_per_pill == [0, 0.8913043478260869, 0.825, 0.8916666666666667]
        assert Beh.gamesteps == [None, (458403, 458633), (458811, 458931), (457761, 457881)]
