"""
Build the department x month feature table.

No look-ahead rule: to predict month t, only information available at the end of
month t-1 is used (every variable is lagged by at least one month).

Feature groups (prefixes):
    h_  conflict history (ACLED)                    -> model A
    south_  satellite conditions in the south          -> model B = A + south_
    north_  satellite conditions in the north (origin) -> model C = B + north_
The south_ / north_ columns come from data/processed/satellite_features.csv (gee_extract.py)
and are added automatically when that file exists.

An extra row is created for the month after the end of the data (forecast).

Usage: python scripts/build_features.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

panel = pd.read_csv(PROC / "panel_dept_month.csv")
panel["ym"] = pd.PeriodIndex(panel["ym"], freq="M")
south = gpd.read_file(PROC / "south_departments.geojson")

# Add the month to forecast (unknown targets)
next_month = panel["ym"].max() + 1
future = south[["dept_id", "department", "province"]].assign(ym=next_month)
panel = pd.concat([panel, future], ignore_index=True)
panel["year"] = panel["ym"].dt.year
panel["month"] = panel["ym"].dt.month
panel = panel.sort_values(["dept_id", "ym"]).reset_index(drop=True)

# --- Neighbourhood (southern departments sharing a border) ---
geo = south.set_index("dept_id")["geometry"]
neighbors = {
    d: [o for o in geo.index if o != d and geo[d].touches(geo[o])] for d in geo.index
}

# --- Conflict history (lagged by one month) ---
counts = ["n_fh", "n_broad", "n_all_events", "fatalities_fh"]
wide = {c: panel.pivot(index="ym", columns="dept_id", values=c).fillna(0) for c in counts}
past = {c: w.shift(1) for c, w in wide.items()}  # value of month t-1 on row t


def roll(df, k):
    return df.rolling(k, min_periods=1).sum()


def to_long(df, name):
    return df.stack().rename(name)


fh = past["n_fh"]
feats = [
    to_long(fh, "h_fh_1m"),
    to_long(roll(fh, 3), "h_fh_3m"),
    to_long(roll(fh, 12), "h_fh_12m"),
    to_long(fh.cumsum(), "h_fh_cum"),
    to_long(roll(past["n_broad"], 12), "h_broad_12m"),
    to_long(roll(past["n_all_events"], 12), "h_allev_12m"),
    to_long(roll(past["fatalities_fh"], 12), "h_fatal_12m"),
]

# Neighbours and province (excluding the department itself)
nb3 = pd.DataFrame({d: roll(fh, 3)[neighbors[d]].sum(axis=1) for d in fh.columns})
nb12 = pd.DataFrame({d: roll(fh, 12)[neighbors[d]].sum(axis=1) for d in fh.columns})
prov_of = south.set_index("dept_id")["province"]
prov12 = pd.DataFrame({
    d: roll(fh, 12)[prov_of.index[(prov_of == prov_of[d]) & (prov_of.index != d)]].sum(axis=1)
    for d in fh.columns
})
feats += [to_long(nb3, "h_nb_fh_3m"), to_long(nb12, "h_nb_fh_12m"),
          to_long(prov12, "h_prov_fh_12m")]

# Months since the last conflict (capped at 36)
since = pd.DataFrame(index=fh.index, columns=fh.columns, dtype=float)
for d in fh.columns:
    last, vals = None, []
    for i, v in enumerate(fh[d].fillna(0).to_numpy()):
        if v > 0:
            last = i
        vals.append(36.0 if last is None else min(i - last, 36))
    since[d] = vals
feats.append(to_long(since, "h_months_since_fh"))

hist = pd.concat(feats, axis=1)
hist.index.names = ["ym", "dept_id"]
hist = hist.reset_index()
X = panel.merge(hist, on=["ym", "dept_id"], how="left")

# Seasonality (known in advance, no leakage)
X["h_month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["h_month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

# --- Satellite variables (if available), lagged by one month ---
sat_path = PROC / "satellite_features.csv"
if sat_path.exists():
    sat = pd.read_csv(sat_path)
    sat["ym"] = pd.PeriodIndex(sat["ym"], freq="M") + 1  # information from t-1 used at t
    sat_cols = [c for c in sat.columns if c.startswith(("south_", "north_"))]
    X = X.merge(sat[["dept_id", "ym"] + sat_cols], on=["dept_id", "ym"], how="left")
    print(f"Satellite variables added: {len(sat_cols)}")
else:
    print("No satellite variables yet (satellite_features.csv missing): model A only.")

X["ym"] = X["ym"].astype(str)
X.to_csv(PROC / "features.csv", index=False)
cols = [c for c in X.columns if c.startswith(("h_", "south_", "north_"))]
print(f"features.csv: {len(X)} rows, {len(cols)} features")
print(f"Forecast month: {next_month}")
