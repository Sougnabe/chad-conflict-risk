"""
Create / refresh the application's SQLite database (data/app.db).

Tables:
    departments : southern departments (id, name, province, GeoJSON geometry)
    panel       : department x month panel (observed conflicts, targets)
    features    : explanatory variables (reserved for satellite indicators)
    predictions : model risk estimates (written by train_models.py)

Raw ACLED notes are not stored (ACLED licence): aggregates only.

Usage: python scripts/init_db.py
"""

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DB = ROOT / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS departments (
    dept_id     TEXT PRIMARY KEY,
    department  TEXT NOT NULL,
    province    TEXT NOT NULL,
    geometry    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS panel (
    dept_id       TEXT NOT NULL REFERENCES departments(dept_id),
    ym            TEXT NOT NULL,
    year          INTEGER,
    month         INTEGER,
    n_fh          INTEGER,
    n_broad       INTEGER,
    fatalities_fh INTEGER,
    n_all_events  INTEGER,
    y_fh          INTEGER,
    y_broad       INTEGER,
    PRIMARY KEY (dept_id, ym)
);
CREATE TABLE IF NOT EXISTS features (
    dept_id TEXT NOT NULL REFERENCES departments(dept_id),
    ym      TEXT NOT NULL,
    name    TEXT NOT NULL,
    value   REAL,
    PRIMARY KEY (dept_id, ym, name)
);
CREATE TABLE IF NOT EXISTS predictions (
    dept_id     TEXT NOT NULL REFERENCES departments(dept_id),
    ym          TEXT NOT NULL,
    model       TEXT NOT NULL,
    risk        REAL NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dept_id, ym, model)
);
"""

south = gpd.read_file(PROC / "south_departments.geojson")
panel = pd.read_csv(PROC / "panel_dept_month.csv")
panel = panel[panel["year"] >= 2019]  # 2018 = burn-in year for history features

with sqlite3.connect(DB) as con:
    con.executescript(SCHEMA)
    con.execute("DELETE FROM panel")
    con.execute("DELETE FROM departments")
    con.executemany(
        "INSERT INTO departments VALUES (?, ?, ?, ?)",
        [(r.dept_id, r.department, r.province, gpd.GeoSeries([r.geometry]).to_json())
         for r in south.itertuples()],
    )
    cols = ["dept_id", "ym", "year", "month", "n_fh", "n_broad",
            "fatalities_fh", "n_all_events", "y_fh", "y_broad"]
    con.executemany(
        f"INSERT INTO panel ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        panel[cols].astype({c: int for c in cols[2:]}).itertuples(index=False),
    )
    for t in ["departments", "panel", "features", "predictions"]:
        print(f"{t:12s}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
print(f"Database: {DB}")
