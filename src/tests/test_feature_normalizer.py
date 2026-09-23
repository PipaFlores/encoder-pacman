import numpy as np
import pandas as pd
import pytest

from src.datahandlers.feature_normalizer import FeatureNormalizer

PELLETS = 8


def _pellet_column(eaten_at=(3, 7, 9), n_rows=12, n_pellets=PELLETS):
    """
    Builds a pellet-state column the way PacmanDataReader does: a new array is
    created only when a pellet is eaten, every other row reuses the *same*
    object as the row before it.
    """
    vectors, current = [], np.ones(n_pellets)
    for i in range(n_rows):
        if i in eaten_at:
            current = current.copy()
            current[i % n_pellets] = 0.0
        vectors.append(current)
    return pd.Series(vectors, dtype=object)


@pytest.fixture
def df():
    column = _pellet_column()
    return pd.DataFrame(
        {
            "available_pellets_states": column,
            "pellets": np.arange(len(column), 0, -1),
            "available_pellets": pd.Series(
                [np.zeros((3, 2))] * len(column), dtype=object
            ),
        }
    )


class TestPelletStateNormalization:
    def test_pooled_statistics(self, df):
        """The whole vector is standardized against one pooled mean/std."""
        normalized = FeatureNormalizer().normalize(df)
        stacked = np.stack(normalized["available_pellets_states"].to_numpy())

        assert stacked.mean() == pytest.approx(0.0, abs=1e-5)
        assert stacked.std(ddof=1) == pytest.approx(1.0, abs=1e-5)
        # Binary input stays binary-valued, just rescaled.
        assert len(np.unique(stacked)) == 2

    def test_preserves_eaten_ordering(self, df):
        """Eaten pellets stay below remaining ones after normalization."""
        raw = np.stack(df["available_pellets_states"].to_numpy())
        normalized = np.stack(
            FeatureNormalizer()
            .normalize(df)["available_pellets_states"]
            .to_numpy()
        )
        assert normalized[raw == 0].max() < normalized[raw == 1].min()

    def test_preserves_object_sharing(self, df):
        """
        Rows sharing an input array must share an output array, otherwise the
        full gamestate frame would materialize one vector per row.
        """
        normalized = FeatureNormalizer().normalize(df)
        ids_in = [id(v) for v in df["available_pellets_states"]]
        ids_out = [id(v) for v in normalized["available_pellets_states"]]

        assert len(set(ids_out)) == len(set(ids_in))
        assert [ids_in.index(i) for i in ids_in] == [ids_out.index(i) for i in ids_out]

    def test_leaves_input_and_other_object_columns_untouched(self, df):
        raw = np.stack(df["available_pellets_states"].to_numpy()).copy()
        normalized = FeatureNormalizer().normalize(df)

        assert np.array_equal(
            np.stack(df["available_pellets_states"].to_numpy()), raw
        )
        # 'available_pellets' holds positions and has no strategy registered.
        assert all(
            a is b
            for a, b in zip(df["available_pellets"], normalized["available_pellets"])
        )

    def test_constant_column_collapses_to_zero(self):
        """Matches how `standardize` handles a zero-variance column."""
        constant = pd.DataFrame(
            {"available_pellets_states": pd.Series([np.ones(PELLETS)] * 5, dtype=object)}
        )
        normalized = FeatureNormalizer().normalize(constant)
        stacked = np.stack(normalized["available_pellets_states"].to_numpy())

        assert np.all(stacked == 0.0)

    def test_missing_cells_are_passed_through(self):
        column = pd.Series([np.ones(PELLETS), None, np.nan, np.zeros(PELLETS)], dtype=object)
        normalized = FeatureNormalizer().normalize(
            pd.DataFrame({"available_pellets_states": column})
        )["available_pellets_states"]

        assert normalized[1] is None
        assert np.isnan(normalized[2])
        assert isinstance(normalized[0], np.ndarray)
        assert normalized[0][0] > normalized[3][0]

    def test_strict_mode_accepts_the_column(self, df):
        """The column now has a registered strategy, so strict mode is happy."""
        FeatureNormalizer(strict=True).normalize(
            df[["available_pellets_states", "pellets"]]
        )
