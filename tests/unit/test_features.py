"""Unit tests of the feature groups, on a small synthetic panel."""

import numpy as np
import pandas as pd
import pytest

from chadrisk.features import (ConflictHistoryFeatures, FeatureTableBuilder, SatelliteFeatures,
                               SeasonalFeatures)
from chadrisk.features.history import months_since_positive


def synthetic_panel(n_months=30):
    """Two departments; department A has a conflict in months 3 and 20."""
    months = pd.period_range("2020-01", periods=n_months, freq="M")
    rows = []
    for dept in ("A", "B"):
        for i, m in enumerate(months):
            n = 1 if (dept == "A" and i in (3, 20)) else 0
            rows.append({"dept_id": dept, "department": dept, "province": "P", "ym": str(m),
                         "n_fh": n, "n_broad": n, "fatalities_fh": 2 * n, "n_all_events": n,
                         "y_fh": n, "y_broad": n, "year": m.year, "month": m.month})
    return pd.DataFrame(rows)


def build(groups, n_months=30, horizon=2):
    panel = synthetic_panel(n_months)
    end = pd.Period(panel["ym"].max(), freq="M") + horizon
    builder = FeatureTableBuilder(groups, neighbours={"A": ["B"], "B": ["A"]},
                                  same_province={"A": ["B"], "B": ["A"]})
    return builder.build(panel, end, log=lambda *_: None)


@pytest.mark.parametrize("values, expected", [
    ([0, 2, 0, 0, 1, 0], [36, 0, 1, 2, 0, 1]),
    ([0, 0, 0], [36, 36, 36]),
    ([np.nan, 1, np.nan], [36, 0, 1]),  # missing values count as zero
])
def test_months_since_positive(values, expected):
    assert months_since_positive(values).tolist() == expected


def test_months_since_positive_is_capped():
    assert months_since_positive([1] + [0] * 50)[-1] == 36


def test_research_history_uses_previous_month_only():
    table = build([ConflictHistoryFeatures(shift=1, prefix="h_", short_windows=True)])
    a = table[table["dept_id"] == "A"].reset_index(drop=True)
    assert a.loc[4, "h_fh_1m"] == 1      # conflict in month 3 seen in month 4
    assert a.loc[3, "h_fh_1m"] == 0      # ...but not in month 3 itself (no look-ahead)
    assert a.loc[5, "h_months_since_fh"] == 1


def test_operational_history_is_delayed_13_months():
    table = build([ConflictHistoryFeatures(shift=13, prefix="o_", short_windows=False)])
    a = table[table["dept_id"] == "A"].reset_index(drop=True)
    assert a.loc[15, "o_fh_12m"] == 0    # month 3 + 12: not yet known
    assert a.loc[16, "o_fh_12m"] == 1    # month 3 + 13: known


def test_neighbour_features_sum_the_neighbours():
    table = build([ConflictHistoryFeatures(shift=1, prefix="h_", short_windows=True)])
    b = table[table["dept_id"] == "B"].reset_index(drop=True)
    assert b.loc[4, "h_nb_fh_3m"] == 1   # A's conflict is B's neighbour conflict
    assert b.loc[4, "h_fh_1m"] == 0


def test_history_is_empty_after_the_conflict_data():
    table = build([ConflictHistoryFeatures(shift=1, prefix="h_", short_windows=True)],
                  horizon=3)
    future = table[table["n_fh"].isna()].sort_values("ym")
    first_future = future["ym"].min()
    assert future.loc[future["ym"] == first_future, "h_fh_1m"].notna().all()
    assert future.loc[future["ym"] > first_future, "h_fh_1m"].isna().all()


def test_future_rows_have_unknown_targets():
    table = build([SeasonalFeatures()], horizon=3)
    assert len(table) == 2 * 33
    assert table.loc[table["n_fh"].isna(), "y_fh"].isna().all()


def test_seasonal_features_are_on_the_unit_circle():
    table = build([SeasonalFeatures()])
    radius = table["h_month_sin"] ** 2 + table["h_month_cos"] ** 2
    assert np.allclose(radius, 1)


def test_satellite_features_are_lagged():
    sat = pd.DataFrame({"dept_id": ["A", "A"], "ym": ["2020-01", "2020-02"],
                        "south_rain_mm": [10.0, 20.0], "other": [0, 0]})
    table = build([SatelliteFeatures(sat, lag=1)])
    a = table[table["dept_id"] == "A"].set_index("ym")
    assert a.loc["2020-02", "south_rain_mm"] == 10.0   # January value used in February
    assert a.loc["2020-03", "south_rain_mm"] == 20.0
    assert "other" not in table.columns                 # only south_/north_ columns


@pytest.mark.parametrize("factory", [
    lambda: ConflictHistoryFeatures(shift=0, prefix="h_", short_windows=False),
    lambda: SatelliteFeatures(pd.DataFrame(), lag=0),
])
def test_zero_lag_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()
