"""
Step 7 — Build the feature table (no look-ahead).

Feature groups:
    h_      conflict history up to month t-1 (research model)
    o_      conflict history up to month t-13 (operational model: ACLED 12-month delay)
    h_month seasonality
    south_ / north_  satellite indicators of month t-1 (if the satellite table exists)

Usage: python scripts/build_features.py
"""

import _bootstrap  # noqa: F401
import geopandas as gpd
import pandas as pd

from chadrisk.config import ACLED_DELAY_MONTHS, PATHS
from chadrisk.data.geography import Geography
from chadrisk.features import (ConflictHistoryFeatures, FeatureTableBuilder,
                               SatelliteFeatures, SeasonalFeatures)

panel = pd.read_csv(PATHS.panel)
south = gpd.read_file(PATHS.south_departments)
data_end = pd.Period(panel["ym"].max(), freq="M")

groups = [ConflictHistoryFeatures(shift=1, prefix="h_", short_windows=True),
          ConflictHistoryFeatures(shift=1 + ACLED_DELAY_MONTHS, prefix="o_",
                                  short_windows=False),
          SeasonalFeatures()]
horizon_end = data_end + 1
if PATHS.satellite.exists():
    satellite = pd.read_csv(PATHS.satellite)
    groups.append(SatelliteFeatures(satellite, lag=1))
    # forecasts are possible up to the month after the last satellite month, as long as the
    # delayed conflict history (t-13) is still inside the conflict data
    horizon_end = min(pd.Period(satellite["ym"].max(), freq="M") + 1,
                      data_end + 1 + ACLED_DELAY_MONTHS)
else:
    print("No satellite table yet: research model A only.")

builder = FeatureTableBuilder(groups, Geography.neighbours(south),
                              Geography.same_province(south))
features = builder.build(panel, horizon_end)
features.to_csv(PATHS.features, index=False)
print(f"features.csv: {len(features)} rows | conflict data end: {data_end} | "
      f"forecast rows up to: {horizon_end}")
