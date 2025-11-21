import logging
from functools import partial
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalizes Pac-Man gameplay features column by column.

    The class keeps a registry of column → normalization strategy pairs. Calling
    `normalize` returns a new `DataFrame` with only the columns that have a
    registered strategy transformed. The original `DataFrame` is left untouched.

    Example
    -------
    ```python
    import pandas as pd
    from src.datahandlers.feature_normalizer import FeatureNormalizer

    df = pd.DataFrame(
        {
            "score": [0, 1200, 3000],
            "Pacman_X": [-13.5, 0.0, 12.5],
            "pacman_attack": [0, 1, 0],
        }
    )

    normalizer = FeatureNormalizer()
    normalized_df = normalizer.normalize(df)
    ```
    """

    EPS = np.finfo(np.float32).eps
    POSITION_BOUNDS = {
        "x": (-13.5, 13.5),
        "y": (-16.5, 13.5),
    }

    def __init__(
        self,
        column_strategies: Mapping[str, Callable[[pd.Series], pd.Series]] | None = None,
        copy: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Args:
            column_strategies: Optional mapping of column names to normalization callables.
                If omitted, defaults tailored to Pac-Man are installed.
            copy: When True (default), the input DataFrame is copied before modifications.
            strict: When True, encountering an unknown column raises a KeyError.
        """
        self.copy = copy
        self.strict = strict
        if column_strategies is None:
            self.column_strategies: Dict[str, Callable[[pd.Series], pd.Series]] = (
                self._default_strategies()
            )
        else:
            self.column_strategies = dict(column_strategies)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the configured normalization strategies to matching columns.

        Args:
            df: Input dataframe.

        Returns:
            DataFrame: A (possibly) new dataframe with normalized columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas.DataFrame")

        result = df.copy(deep=True) if self.copy else df

        unknown_columns = set(df.columns) - set(self.column_strategies)
        if unknown_columns and self.strict:
            raise KeyError(
                f"Columns {sorted(unknown_columns)} do not have registered normalization strategies."
            )
        elif unknown_columns:
            logger.debug(
                "FeatureNormalizer: no strategy for columns: %s",
                sorted(unknown_columns),
            )

        for column, strategy in self.column_strategies.items():
            if column not in result.columns:
                continue
            result[column] = strategy(result[column])

        return result

    def register_strategy(
        self, column: str, strategy: Callable[[pd.Series], pd.Series]
    ) -> None:
        """Registers or overrides the strategy for a column."""
        self.column_strategies[column] = strategy

    def extend_strategies(
        self, mapping: Mapping[str, Callable[[pd.Series], pd.Series]]
    ) -> None:
        """Bulk registration helper."""
        for column, strategy in mapping.items():
            self.register_strategy(column, strategy)

    # --------------------------------------------------------------------- #
    # Normalization helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def standardize(
        series: pd.Series,
        mean: float | None = None,
        std: float | None = None,
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        mean = float(values.mean()) if mean is None else mean
        std = float(values.std()) if std is None else std
        std = std if std > FeatureNormalizer.EPS else 1.0
        return ((values - mean) / std).fillna(0.0)

    @staticmethod
    def minmax(
        series: pd.Series,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        min_val = float(values.min()) if min_val is None else min_val
        max_val = float(values.max()) if max_val is None else max_val
        span = max(max_val - min_val, FeatureNormalizer.EPS)
        normalized = (values - min_val) / span
        return normalized.fillna(0.0)

    def normalize_counter(self, series: pd.Series) -> pd.Series:
        return self.minmax(series)

    def normalize_binary_flag(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return values.clip(lower=0.0, upper=1.0)

    def normalize_multistate_flag(
        self, series: pd.Series, max_state: int = 3
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        values = values.clip(lower=0.0, upper=max_state)
        return values / max(max_state, 1)

    def normalize_position(
        self,
        series: pd.Series,
        axis: str,
        bounds: tuple[float, float] | None = None,
    ) -> pd.Series:
        if bounds is None:
            if axis not in self.POSITION_BOUNDS:
                raise ValueError(f"Unsupported axis '{axis}'. Expected one of x/y.")
            bounds = self.POSITION_BOUNDS[axis]
        return self.minmax(series, min_val=bounds[0], max_val=bounds[1])

    def normalize_distance(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")

        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]

        if finite_values.empty:
            return values  # nothing finite to normalize

        normalized = self.minmax(finite_values)

        result = pd.Series(index=series.index, dtype=float)
        result[finite_mask] = normalized
        result[~finite_mask] = values[~finite_mask]  # keep ±inf for masks

        return result.fillna(0.0)

    # --------------------------------------------------------------------- #
    # Defaults
    # --------------------------------------------------------------------- #
    def _default_strategies(self) -> Dict[str, Callable[[pd.Series], pd.Series]]:
        strategies: Dict[str, Callable[[pd.Series], pd.Series]] = {}

        counter_columns = ("score", "lives", "pellets")
        binary_columns = [
            "pacman_attack",
            "powerpelletstate_1",
            "powerpelletstate_2",
            "powerpelletstate_3",
            "powerpelletstate_4",
            "fruitState_1",
            "fruitState_2",
        ]
        multistate_columns = [
            "ghost1_state",
            "ghost2_state",
            "ghost3_state",
            "ghost4_state",
        ]
        position_x_columns = [
            "Pacman_X",
            "Ghost1_X",
            "Ghost2_X",
            "Ghost3_X",
            "Ghost4_X",
        ]
        position_y_columns = [
            "Pacman_Y",
            "Ghost1_Y",
            "Ghost2_Y",
            "Ghost3_Y",
            "Ghost4_Y",
        ]
        distance_columns = [
            "Ghost1_distance",
            "Ghost2_distance",
            "Ghost3_distance",
            "Ghost4_distance",
        ]

        for column in counter_columns:
            strategies[column] = self.normalize_counter
        for column in binary_columns:
            strategies[column] = self.normalize_binary_flag
        for column in multistate_columns:
            strategies[column] = partial(self.normalize_multistate_flag, max_state=3)
        for column in position_x_columns:
            strategies[column] = partial(self.normalize_position, axis="x")
        for column in position_y_columns:
            strategies[column] = partial(self.normalize_position, axis="y")
        for column in distance_columns:
            strategies[column] = self.normalize_distance

        return strategies


def normalize_gameplay_dataframe(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    normalizer: FeatureNormalizer | None = None,
) -> pd.DataFrame:
    """
    Convenience helper to normalize a selection of gameplay columns.

    Args:
        df: Input dataframe containing Pac-Man gameplay features.
        columns: Optional iterable to limit the normalization to a subset of columns.
        normalizer: Optional custom `FeatureNormalizer`. A default one is created otherwise.

    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            logger.debug(
                "normalize_gameplay_dataframe: requested columns not in dataframe: %s",
                sorted(missing),
            )
        df = df[list(columns)].copy()

    normalizer = normalizer or FeatureNormalizer()
    return normalizer.normalize(df)

