"""
Monthly update of the operational outlook, in one command.

Steps:
    1. Append the latest complete satellite month(s) from Google Earth Engine.
    2. Rebuild the feature table (no look-ahead).
    3. Retrain the operational models and write the new outlook to the dashboard database.

Conflict data (ACLED) are not refreshed here: new events must first be classified with the
codebook (docs/labeling_codebook.md) before they can be used. Refreshing them is a separate,
periodic step (download_acled.py, manual labelling, build_target.py, build_panel.py, init_db.py).

Usage: python scripts/update_outlook.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/gee_extract.py"],
    ["scripts/build_features.py"],
    ["scripts/train_models.py", "--mode", "operational", "--unit", "quarter",
     "--fixed-model", "logreg_OB"],
    ["scripts/train_models.py", "--mode", "operational", "--unit", "month"],
]

for step in STEPS:
    print(f"\n>>> python {' '.join(step)}")
    result = subprocess.run([sys.executable, *step], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Step failed: {' '.join(step)}")

meta = json.loads((ROOT / "reports" / "outlook_operational_quarter.json").read_text("utf8"))
print(f"\nOutlook updated for {meta['forecast_start']} to {meta['forecast_end']} "
      f"(satellite data up to {meta['satellite_data_end']}). "
      f"Next update expected around {meta['next_update_expected']}.")
