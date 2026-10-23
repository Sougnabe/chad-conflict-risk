"""
Performance benchmark of the pipeline and the dashboard.

Measures, for each stage, the wall-clock time and the peak resident memory of the process
(RSS, sampled every 20 ms), and for the dashboard the server-side rendering time of a cold start, a warm
rerun and a language switch. Results: reports/performance/benchmark.json and benchmark.md.

Usage: python scripts/benchmark.py
"""

import json
import platform
import threading
import time

import _bootstrap  # noqa: F401
import geopandas as gpd
import pandas as pd
import psutil

from chadrisk.config import ACLED_DELAY_MONTHS, CONFLICT_DATA_END, PANEL_START, PATHS
from chadrisk.data.geography import Geography
from chadrisk.data.labels import LabelledEvents
from chadrisk.data.panel import PanelBuilder
from chadrisk.experiment import Experiment, ExperimentConfig
from chadrisk.features import (ConflictHistoryFeatures, FeatureTableBuilder,
                               SatelliteFeatures, SeasonalFeatures)

OUT = PATHS.reports / "performance"
results = []


class PeakMemory:
    """Samples the resident memory of this process in a background thread."""

    def __init__(self, interval=0.02):
        self.process, self.interval, self.peak = psutil.Process(), interval, 0
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()


def measure(name, function, repeat=1):
    """Runs `function` `repeat` times; records the median time and the peak RSS."""
    times, value = [], None
    with PeakMemory() as memory:
        for _ in range(repeat):
            start = time.perf_counter()
            value = function()
            times.append(time.perf_counter() - start)
    times.sort()
    results.append({"stage": name, "seconds": round(times[len(times) // 2], 3),
                    "peak_memory_mb": round(memory.peak / 1e6, 1), "runs": repeat})
    print(f"{name:45s} {results[-1]['seconds']:8.3f} s  {results[-1]['peak_memory_mb']:7.1f} MB")
    return value


# ---------------- pipeline ----------------
events = measure("Labels and target (1,593 events)",
                 lambda: LabelledEvents.from_files(PATHS.acled_raw, PATHS.manual_labels).build(),
                 repeat=3)
geography = Geography.from_files(PATHS.external / "tcd_adm1.geojson",
                                 PATHS.external / "tcd_adm2.geojson")
south = geography.region()
panel = measure("Spatial join + panel (26 depts x 105 months)",
                lambda: PanelBuilder(south, PANEL_START, CONFLICT_DATA_END).build(
                    geography.assign_events(events)), repeat=3)
satellite = pd.read_csv(PATHS.satellite)
south_file = gpd.read_file(PATHS.south_departments)


def features():
    groups = [ConflictHistoryFeatures(1, "h_", True),
              ConflictHistoryFeatures(1 + ACLED_DELAY_MONTHS, "o_", False),
              SeasonalFeatures(), SatelliteFeatures(satellite)]
    builder = FeatureTableBuilder(groups, Geography.neighbours(south_file),
                                  Geography.same_province(south_file))
    horizon = pd.Period(satellite["ym"].max(), freq="M") + 1
    return builder.build(panel, horizon, log=lambda *_: None)


table = measure("Feature table (31 features, 3,068 rows)", features, repeat=3)
for mode, unit in [("operational", "quarter"), ("research", "month")]:
    config = ExperimentConfig(mode, unit, "logreg_OB" if mode == "operational" else None)
    measure(f"Experiment {mode}/{unit} (selection + evaluation)",
            lambda c=config: Experiment(c, table).run())

# ---------------- dashboard (server side) ----------------
from streamlit.testing.v1 import AppTest  # noqa: E402

app_file = str(PATHS.root / "app" / "app.py")
at = measure("Dashboard: cold start (first render)",
             lambda: AppTest.from_file(app_file, default_timeout=180).run())
measure("Dashboard: warm rerun (cached data)", lambda: at.run(), repeat=5)
measure("Dashboard: switch to French", lambda: at.radio(key="language").set_value("Français").run())

# ---------------- payload ----------------
geometry_kb = sum(len(row[0]) for row in
                  __import__("sqlite3").connect(PATHS.database)
                  .execute("SELECT geometry FROM departments")) / 1e3
environment = {
    "os": platform.platform(), "python": platform.python_version(),
    "processor": platform.processor(),
    "cpu_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
    "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    "database_kb": round(PATHS.database.stat().st_size / 1e3, 1),
    "map_geometry_kb": round(geometry_kb, 1),
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "benchmark.json").write_text(json.dumps({"environment": environment,
                                                "stages": results}, indent=2), "utf8")
lines = ["| Stage | Time (s) | Peak process memory, RSS (MB) | Runs |", "|---|---|---|---|"]
lines += [f"| {r['stage']} | {r['seconds']} | {r['peak_memory_mb']} | {r['runs']} |"
          for r in results]
(OUT / "benchmark.md").write_text("\n".join(lines) + "\n\n" + "\n".join(
    f"- {k}: {v}" for k, v in environment.items()) + "\n", "utf8")
print(f"\nEnvironment: {environment}")
