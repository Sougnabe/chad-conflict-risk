"""Seasonality, encoded as a point on a circle so that December and January are close."""

import numpy as np
import pandas as pd

from chadrisk.features.base import FeatureContext, FeatureGroup


class SeasonalFeatures(FeatureGroup):
    """Month of the year as sine and cosine (known in advance: no leakage)."""

    @property
    def name(self) -> str:
        return "seasonality"

    def compute(self, context: FeatureContext) -> pd.DataFrame:
        index = pd.MultiIndex.from_frame(context.panel[["ym", "dept_id"]])
        month = context.panel["month"].to_numpy()
        return pd.DataFrame({"h_month_sin": np.sin(2 * np.pi * month / 12),
                             "h_month_cos": np.cos(2 * np.pi * month / 12)}, index=index)
