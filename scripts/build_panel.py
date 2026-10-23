"""
Step 4 — Build the department x month panel of southern Chad (2017-01 -> 2025-09).

2017-2018 are burn-in years for the lagged history features; modelling starts in 2019.

Usage: python scripts/build_panel.py
"""

import _bootstrap  # noqa: F401
import pandas as pd

from chadrisk.config import CONFLICT_DATA_END, PANEL_START, PATHS
from chadrisk.data.geography import Geography
from chadrisk.data.panel import PanelBuilder

geography = Geography.from_files(PATHS.external / "tcd_adm1.geojson",
                                 PATHS.external / "tcd_adm2.geojson")
south = geography.region()
south.to_file(PATHS.south_departments, driver="GeoJSON")

events = geography.assign_events(pd.read_csv(PATHS.labelled_events))
print(f"{len(south)} southern departments | events outside every department: "
      f"{events['dept_id'].isna().sum()}")

panel = PanelBuilder(south, PANEL_START, CONFLICT_DATA_END).build(events)
panel.to_csv(PATHS.panel, index=False)
study = panel[panel["year"] >= 2019]
print(f"Panel: {len(panel)} rows | department-months (2019+) with a conflict: "
      f"{study['y_fh'].sum()} ({study['y_fh'].mean():.1%})")
