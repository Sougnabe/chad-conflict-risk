"""
Build the department x month panel for southern Chad (the model's unit of analysis).

- Boundaries: geoBoundaries TCD ADM1/ADM2 (2019, 70 departments)
- Events: data/processed/acled_chad_labeled.csv (output of build_target.py)
- Each event is assigned to a department by spatial join (lat/lon), because
  ACLED admin2 names and geoBoundaries names do not match.

Outputs:
    data/processed/south_departments.geojson
    data/processed/panel_dept_month.csv

Usage: python scripts/build_panel.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"

SOUTH_ADM1 = [
    "Logone Occidental", "Logone Oriental", "Mandoul", "Moyen-Chari",
    "Tandjilé", "Mayo-Kebbi Est", "Mayo-Kebbi Ouest",
]
# ACLED coverage of Chad is poor before 2019; data available up to September 2025.
# 2018 is only a burn-in year for the history features (12-month lags);
# modelling and the app start in 2019.
START, END = "2018-01", "2025-09"

adm1 = gpd.read_file(EXT / "tcd_adm1.geojson")[["shapeName", "geometry"]]
adm2 = gpd.read_file(EXT / "tcd_adm2.geojson")[["shapeName", "shapeID", "geometry"]]
adm1 = adm1.rename(columns={"shapeName": "province"})
adm2 = adm2.rename(columns={"shapeName": "department", "shapeID": "dept_id"})

# Province of each department (via an interior point)
pts = adm2.copy()
pts["geometry"] = adm2.representative_point()
adm2["province"] = gpd.sjoin(pts, adm1, predicate="within")["province"]
south = adm2[adm2["province"].isin(SOUTH_ADM1)].reset_index(drop=True)
assert south["province"].nunique() == len(SOUTH_ADM1), "Missing southern province"
south.to_file(PROC / "south_departments.geojson", driver="GeoJSON")
print(f"{len(south)} departments in the {len(SOUTH_ADM1)} southern provinces")

# Spatial assignment of ACLED events
ev = pd.read_csv(PROC / "acled_chad_labeled.csv", parse_dates=["event_date"])
ev = gpd.GeoDataFrame(
    ev, geometry=gpd.points_from_xy(ev["longitude"], ev["latitude"]), crs="EPSG:4326"
)
ev = gpd.sjoin(ev, adm2[["dept_id", "geometry"]], predicate="within", how="left")
print(f"Events outside all department polygons: {ev['dept_id'].isna().sum()}")
ev["ym"] = ev["event_date"].dt.to_period("M")
ev["fatalities_fh"] = ev["fatalities"].where(ev["target_fh"] == 1, 0)

# Full department x month panel (including months without events)
months = pd.period_range(START, END, freq="M")
panel = pd.MultiIndex.from_product(
    [south["dept_id"], months], names=["dept_id", "ym"]
).to_frame(index=False)
panel = panel.merge(south[["dept_id", "department", "province"]], on="dept_id")

agg = (
    ev[ev["dept_id"].isin(south["dept_id"])]
    .groupby(["dept_id", "ym"])
    .agg(
        n_fh=("target_fh", "sum"),
        n_broad=("target_broad", "sum"),
        fatalities_fh=("fatalities_fh", "sum"),
        n_all_events=("event_id_cnty", "size"),
    )
    .reset_index()
)
panel = panel.merge(agg, on=["dept_id", "ym"], how="left").fillna(
    {"n_fh": 0, "n_broad": 0, "fatalities_fh": 0, "n_all_events": 0}
)
panel["y_fh"] = (panel["n_fh"] > 0).astype(int)
panel["y_broad"] = (panel["n_broad"] > 0).astype(int)
panel["year"] = panel["ym"].dt.year
panel["month"] = panel["ym"].dt.month
panel["ym"] = panel["ym"].astype(str)
panel.to_csv(PROC / "panel_dept_month.csv", index=False)

study = panel[panel["year"] >= 2019]
print(f"\nPanel: {len(panel)} rows ({len(south)} departments x {len(months)} months)")
print(f"Department-months (2019+) with at least one FH conflict: {study['y_fh'].sum()} "
      f"({study['y_fh'].mean():.1%})")
print(f"With the broad target (FH + AMB): {study['y_broad'].sum()} ({study['y_broad'].mean():.1%})")
print("\nFH conflicts by department (top 10):")
print(study.groupby(["province", "department"])["n_fh"].sum()
      .sort_values(ascending=False).head(10).to_string())
