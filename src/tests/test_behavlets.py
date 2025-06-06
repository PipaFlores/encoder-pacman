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
        # Complete when behavlets are done.

        Beh_encodings = BehavletsEncoding(data_folder="data")
        results = Beh_encodings.calculate_behavlets(
            level_id=600, behavlet_type="Aggression3"
        )

        assert True

    def test_Aggression3(self, reader):
        Beh = Behavlets(name="Aggression3", window_length=20)

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=600)[0]
        )

        assert results == Beh
        assert results.value == 2
        assert results.gamesteps.__len__() == 2
        assert results.gamesteps == [(457810, 457850), (458549, 458589)]

    def test_Aggression1(self, reader):
        Beh = Behavlets(name="Aggression1")

        results = Beh.calculate(
            gamestates=reader._filter_gamestate_data(level_id=603)[0]
        )

        assert results == Beh
        assert results.value == 1
        assert results.gamesteps == [(461343, 461374)]
