"""
Build the department x month feature table.

No look-ahead rule: to predict month t, only information available at the end of
month t-1 is used (every variable is lagged by at least one month).

Feature groups (prefixes):
    h_      conflict history up to month t-1                    -> research models A, B, C
    o_      conflict history up to month t-13                   -> operational models
            (ACLED Research-tier data are released with a 12-month delay, so in real use the
            most recent conflict information available at the start of month t is t-13)
    south_  satellite conditions in the south (month t-1)
    north_  satellite conditions in the northern departure zones (month t-1)
The south_ / north_ columns come from data/processed/satellite_features.csv (gee_extract.py)
and are added automatically when that file exists.

Rows are added after the end of the conflict data (targets unknown):
    - research forecast month: the month after the last ACLED month;
    - operational months: up to the month after the last satellite month. Only o_, south_,
      north_ and seasonal features are defined there; h_ features are left empty because the
      recent conflict history they need is not available.

Usage: python scripts/build_features.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LAG = 12  # ACLED publication delay (months)

panel = pd.read_csv(PROC / "panel_dept_month.csv")
panel["ym"] = pd.PeriodIndex(panel["ym"], freq="M")
south = gpd.read_file(PROC / "south_departments.geojson")
sat_path = PROC / "satellite_features.csv"
sat = pd.read_csv(sat_path) if sat_path.exists() else None

DATA_END = panel["ym"].max()  # last month with conflict data
OPS_END = (pd.Period(sat["ym"].max(), freq="M") + 1) if sat is not None else DATA_END + 1
OPS_END = min(OPS_END, DATA_END + 1 + LAG)  # o_ features need conflicts up to t-13

# Add the months after the end of the conflict data (unknown targets)
future_months = pd.period_range(DATA_END + 1, OPS_END, freq="M")
future = pd.concat([south[["dept_id", "department", "province"]].assign(ym=m)
                    for m in future_months], ignore_index=True)
panel = pd.concat([panel, future], ignore_index=True)
panel["year"] = panel["ym"].dt.year
panel["month"] = panel["ym"].dt.month
panel = panel.sort_values(["dept_id", "ym"]).reset_index(drop=True)

# --- Neighbourhood (southern departments sharing a border) ---
geo = south.set_index("dept_id")["geometry"]
neighbors = {
    d: [o for o in geo.index if o != d and geo[d].touches(geo[o])] for d in geo.index
}
prov_of = south.set_index("dept_id")["province"]
same_prov = {d: list(prov_of.index[(prov_of == prov_of[d]) & (prov_of.index != d)])
             for d in prov_of.index}

# --- Monthly conflict counts (0 when no event; unknown after DATA_END) ---
counts = ["n_fh", "n_broad", "n_all_events", "fatalities_fh"]
wide = {}
for c in counts:
    w = panel.pivot(index="ym", columns="dept_id", values=c)
    w.loc[w.index <= DATA_END] = w.loc[w.index <= DATA_END].fillna(0)
    wide[c] = w


def roll(df, k):
    return df.rolling(k, min_periods=1).sum()


def to_long(df, name):
    return df.stack(future_stack=True).rename(name)


def months_since(series_wide):
    """Months since the last conflict in the lagged series (capped at 36)."""
    out = pd.DataFrame(index=series_wide.index, columns=series_wide.columns, dtype=float)
    for d in series_wide.columns:
        last, vals = None, []
        for i, v in enumerate(series_wide[d].fillna(0).to_numpy()):
            if v > 0:
                last = i
            vals.append(36.0 if last is None else min(i - last, 36))
        out[d] = vals
    return out


def history_features(shift, prefix, short_windows):
    """Conflict-history features using counts up to month t-shift."""
    past = {c: w.shift(shift) for c, w in wide.items()}
    fh = past["n_fh"]
    feats = [
        to_long(roll(fh, 12), f"{prefix}fh_12m"),
        to_long(fh.cumsum(), f"{prefix}fh_cum"),
        to_long(roll(past["n_broad"], 12), f"{prefix}broad_12m"),
        to_long(roll(past["n_all_events"], 12), f"{prefix}allev_12m"),
        to_long(roll(past["fatalities_fh"], 12), f"{prefix}fatal_12m"),
        to_long(pd.DataFrame({d: roll(fh, 12)[neighbors[d]].sum(axis=1) for d in fh.columns}),
                f"{prefix}nb_fh_12m"),
        to_long(pd.DataFrame({d: roll(fh, 12)[same_prov[d]].sum(axis=1) for d in fh.columns}),
                f"{prefix}prov_fh_12m"),
        to_long(months_since(fh), f"{prefix}months_since_fh"),
    ]
    if short_windows:
        feats += [
            to_long(fh, f"{prefix}fh_1m"),
            to_long(roll(fh, 3), f"{prefix}fh_3m"),
            to_long(pd.DataFrame({d: roll(fh, 3)[neighbors[d]].sum(axis=1)
                                  for d in fh.columns}), f"{prefix}nb_fh_3m"),
        ]
    out = pd.concat(feats, axis=1)
    out.index.names = ["ym", "dept_id"]
    # Only defined while the lagged window lies inside the conflict data
    out.loc[out.index.get_level_values("ym") > DATA_END + shift] = np.nan
    return out.reset_index()


research = history_features(1, "h_", short_windows=True)      # conflicts up to t-1
operational = history_features(1 + LAG, "o_", short_windows=False)  # conflicts up to t-13
X = panel.merge(research, on=["ym", "dept_id"], how="left")
X = X.merge(operational, on=["ym", "dept_id"], how="left")

# Seasonality (known in advance, no leakage)
X["h_month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["h_month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

# --- Satellite variables (if available), lagged by one month ---
if sat is not None:
    sat["ym"] = pd.PeriodIndex(sat["ym"], freq="M") + 1  # information from t-1 used at t
    sat_cols = [c for c in sat.columns if c.startswith(("south_", "north_"))]
    X = X.merge(sat[["dept_id", "ym"] + sat_cols], on=["dept_id", "ym"], how="left")
    print(f"Satellite variables added: {len(sat_cols)}")
else:
    print("No satellite variables yet (satellite_features.csv missing): model A only.")

X["ym"] = X["ym"].astype(str)
X.to_csv(PROC / "features.csv", index=False)
cols = [c for c in X.columns if c.startswith(("h_", "o_", "south_", "north_"))]
print(f"features.csv: {len(X)} rows, {len(cols)} features")
print(f"Conflict data end: {DATA_END} | research forecast month: {DATA_END + 1} | "
      f"operational months up to: {OPS_END}")
