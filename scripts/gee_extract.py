"""
Step 6 — Extract (or update) the monthly satellite indicators with Google Earth Engine.

Only the months missing from data/processed/satellite_features.csv are extracted, up to the
last month complete in both CHIRPS and MODIS (detected automatically, or --end YYYY-MM).

Requires EE_PROJECT in .env and a one-time `earthengine authenticate`.

Usage: python scripts/gee_extract.py [--end YYYY-MM]
"""

import argparse
import os

import _bootstrap  # noqa: F401
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from chadrisk.config import NORTH_PROVINCES, PATHS
from chadrisk.data.geography import Geography
from chadrisk.data.satellite import SatelliteExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--end", default=None, help="last month to extract (YYYY-MM)")
args = parser.parse_args()

load_dotenv(PATHS.root / ".env")
geography = Geography.from_files(PATHS.external / "tcd_adm1.geojson",
                                 PATHS.external / "tcd_adm2.geojson")
north = geography.provinces[geography.provinces["province"].isin(NORTH_PROVINCES)]
if len(north) != len(NORTH_PROVINCES):
    raise SystemExit("northern province not found in the boundaries")

extractor = SatelliteExtractor(os.getenv("EE_PROJECT"),
                               gpd.read_file(PATHS.south_departments),
                               north.dissolve().geometry.iloc[0])
end = pd.Period(args.end, freq="M") if args.end else extractor.last_complete_month()
print(f"Last month to extract: {end}")

existing = pd.read_csv(PATHS.satellite) if PATHS.satellite.exists() else None
table = extractor.update(existing, end)
if table is existing:
    print("Nothing to extract: the satellite table is already up to date.")
else:
    table.to_csv(PATHS.satellite, index=False)
    print(f"{PATHS.satellite}: {len(table)} rows, up to {table['ym'].max()}")
