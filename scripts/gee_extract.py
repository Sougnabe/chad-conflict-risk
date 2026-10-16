"""
Extract monthly satellite indicators per southern department (Google Earth Engine).

Prerequisites:
    1. Earth Engine account + registered Google Cloud project (non-commercial use).
    2. In .env: EE_PROJECT=your-cloud-project-id
    3. Once: earthengine authenticate

Output variables (one row per department x month, 2018-01 -> 2025-09):
    South (per department)
        south_ndvi_crop        mean NDVI over cropland (MODIS MOD13Q1)
        south_ndvi_crop_anom   deviation from the 2003-2017 mean of the same calendar month
        south_ndvi_crop_delta  change from the previous month (drying / harvest signal)
        south_rain_mm          monthly rainfall (CHIRPS)
        south_rain_anom        deviation from the 1991-2017 mean of the same calendar month
        south_cropland_frac    cropland share (ESA WorldCover 2021, static)
        south_water_occ        mean surface-water occurrence (JRC GSW, static)
        south_pop_density      population density (WorldPop 2020, static)
    North (herd departure zones: Batha, Guera, Salamat, Ouaddai; same value for all departments)
        north_ndvi_anom        mean NDVI anomaly
        north_rain_anom        monthly rainfall anomaly

The one-month lag (no look-ahead) is applied afterwards in build_features.py.

Usage: python scripts/gee_extract.py
"""

import os
import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
OUT = PROC / "satellite_features.csv"

load_dotenv(ROOT / ".env")
project = os.getenv("EE_PROJECT")
if not project:
    sys.exit("EE_PROJECT missing from .env (ID of the Google Cloud project linked to Earth Engine).")
ee.Initialize(project=project)

NORTH_ADM1 = ["Batha", "Guéra", "Salamat", "Ouaddaï"]
MONTHS = pd.period_range("2017-12", "2025-09", freq="M")
SCALE_NDVI = 250
SCALE_RAIN = 5566


def to_fc(gdf, id_col):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(0.005)
    feats = [ee.Feature(ee.Geometry(g.__geo_interface__), {id_col: i})
             for i, g in zip(gdf[id_col], gdf.geometry)]
    return ee.FeatureCollection(feats)


south = gpd.read_file(PROC / "south_departments.geojson")
depts = to_fc(south, "dept_id")
adm1 = gpd.read_file(EXT / "tcd_adm1.geojson")
assert adm1["shapeName"].isin(NORTH_ADM1).sum() == 4, "Northern province not found"
north = adm1[adm1["shapeName"].isin(NORTH_ADM1)].dissolve()
north_geom = ee.Geometry(north.geometry.iloc[0].simplify(0.01).__geo_interface__)

# --- Collections ---
ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
rain = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select("precipitation")
cropland = ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)


def monthly_ndvi(y, m):
    start = ee.Date.fromYMD(y, m, 1)
    return ndvi.filterDate(start, start.advance(1, "month")).mean().multiply(0.0001)


def monthly_rain(y, m):
    start = ee.Date.fromYMD(y, m, 1)
    return rain.filterDate(start, start.advance(1, "month")).sum()


def clim(collection_fn, years, m):
    return ee.ImageCollection([collection_fn(y, m) for y in years]).mean()


ndvi_clim = {m: clim(monthly_ndvi, range(2003, 2018), m) for m in range(1, 13)}
rain_clim = {m: clim(monthly_rain, range(1991, 2018), m) for m in range(1, 13)}

# --- Static variables ---
static_img = ee.Image.cat([
    cropland.rename("south_cropland_frac"),
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
      .divide(100).rename("south_water_occ"),
    ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "TCD"))
      .filter(ee.Filter.eq("year", 2020)).first()
      .divide(0.01).rename("south_pop_density"),  # people per 100 m pixel / 0.01 km2
])
static = static_img.reduceRegions(depts, ee.Reducer.mean(), scale=250).getInfo()
static_df = pd.DataFrame([f["properties"] for f in static["features"]])
print("Static variables extracted.")

# --- Monthly variables ---
rows = []
for p in MONTHS:
    y, m = p.year, p.month
    nd = monthly_ndvi(y, m)
    nd_crop = nd.updateMask(cropland)
    rn = monthly_rain(y, m)
    img = ee.Image.cat([
        nd_crop.rename("south_ndvi_crop"),
        nd_crop.subtract(ndvi_clim[m]).rename("south_ndvi_crop_anom"),
    ])
    r_ndvi = img.reduceRegions(depts, ee.Reducer.mean(), scale=SCALE_NDVI).getInfo()
    r_rain = ee.Image.cat([rn.rename("south_rain_mm"), rn.subtract(rain_clim[m]).rename("south_rain_anom")]) \
        .reduceRegions(depts, ee.Reducer.mean(), scale=SCALE_RAIN).getInfo()
    n_vals = ee.Image.cat([
        nd.subtract(ndvi_clim[m]).rename("north_ndvi_anom"),
        rn.subtract(rain_clim[m]).rename("north_rain_anom"),
    ]).reduceRegion(ee.Reducer.mean(), north_geom, scale=1000, maxPixels=1e10).getInfo()
    for a, b in zip(r_ndvi["features"], r_rain["features"]):
        rows.append({"ym": str(p), **a["properties"],
                     **{k: v for k, v in b["properties"].items() if k != "dept_id"},
                     **n_vals})
    print(f"{p} done")

sat = pd.DataFrame(rows).merge(static_df, on="dept_id")
sat = sat.sort_values(["dept_id", "ym"])
sat["south_ndvi_crop_delta"] = sat.groupby("dept_id")["south_ndvi_crop"].diff()
sat = sat[sat["ym"] >= "2018-01"]
sat.to_csv(OUT, index=False)
print(f"\n{OUT}: {len(sat)} rows, columns: {[c for c in sat.columns if c.startswith(("south_", "north_"))]}")
