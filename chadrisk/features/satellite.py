"""Satellite indicators, attached to the month after they were observed."""

import pandas as pd

from chadrisk.features.base import FeatureContext, FeatureGroup


class SatelliteFeatures(FeatureGroup):
    """south_* and north_* indicators from the satellite table, lagged by `lag` months."""

    PREFIXES = ("south_", "north_")

    def __init__(self, table: pd.DataFrame, lag: int = 1):
        if lag < 1:
            raise ValueError("lag must be at least 1 month (no look-ahead)")
        self.table = table
        self.lag = lag

    @property
    def name(self) -> str:
        return f"satellite indicators of month t-{self.lag}"

    @property
    def columns(self) -> list[str]:
        return [c for c in self.table.columns if c.startswith(self.PREFIXES)]

    def compute(self, context: FeatureContext) -> pd.DataFrame:
        sat = self.table[["dept_id", "ym", *self.columns]].copy()
        sat["ym"] = pd.PeriodIndex(sat["ym"], freq="M") + self.lag  # information of t-lag at t
        return sat.set_index(["ym", "dept_id"])
