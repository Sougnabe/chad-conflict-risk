"""Common interface of the feature groups."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

COUNT_COLUMNS = ("n_fh", "n_broad", "n_all_events", "fatalities_fh")


@dataclass
class FeatureContext:
    """Everything a feature group may need, computed once by the builder.

    panel       department x month rows, including future months (unknown counts)
    counts      wide tables (month x department) of monthly counts; NaN after data_end
    neighbours  departments sharing a border, for each department
    same_province other departments of the same province, for each department
    data_end    last month with conflict data
    """

    panel: pd.DataFrame
    counts: dict[str, pd.DataFrame]
    neighbours: dict[str, list[str]]
    same_province: dict[str, list[str]]
    data_end: pd.Period


class FeatureGroup(ABC):
    """A group of features computed from the context.

    `compute` returns one row per (ym, dept_id), with the index named ["ym", "dept_id"] and
    one column per feature. Every feature must only use information available before the
    month it is attached to (no look-ahead).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short description used in logs."""

    @abstractmethod
    def compute(self, context: FeatureContext) -> pd.DataFrame:
        """Feature values indexed by (ym, dept_id)."""

    @staticmethod
    def _long(wide: pd.DataFrame, name: str) -> pd.Series:
        """Month x department table -> series indexed by (ym, dept_id)."""
        series = wide.stack(future_stack=True).rename(name)
        series.index.names = ["ym", "dept_id"]
        return series
