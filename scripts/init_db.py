"""
Step 5 — Create / refresh the dashboard database (data/app.db).

Usage: python scripts/init_db.py
"""

import _bootstrap  # noqa: F401
import geopandas as gpd
import pandas as pd

from chadrisk.config import PATHS, STUDY_START_YEAR
from chadrisk.storage import RiskRepository

repository = RiskRepository(PATHS.database)
repository.create_schema()
repository.replace_departments(gpd.read_file(PATHS.south_departments))
panel = pd.read_csv(PATHS.panel)
repository.replace_panel(panel[panel["year"] >= STUDY_START_YEAR])
for table, n in repository.counts().items():
    print(f"{table:12s}: {n} rows")
