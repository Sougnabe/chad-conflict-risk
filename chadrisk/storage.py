"""SQLite repository: the single access point to the dashboard database.

Only aggregated, department-level data are stored (no ACLED event notes, per its licence).
This module depends only on the standard library and pandas, so the dashboard stays light.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

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

PANEL_COLUMNS = ["dept_id", "ym", "year", "month", "n_fh", "n_broad", "fatalities_fh",
                 "n_all_events", "y_fh", "y_broad"]


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when the dashboard database has not been created yet."""


class RiskRepository:
    """Reads and writes departments, panel, predictions and explanations."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self, must_exist: bool = False):
        if must_exist and not self.path.exists():
            raise DatabaseNotFoundError(
                f"{self.path} not found: run scripts/init_db.py and scripts/train_models.py")
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def create_schema(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA)

    # ---------- writing ----------
    def replace_departments(self, departments, map_tolerance: float = 0.002) -> None:
        """`departments`: GeoDataFrame with dept_id, department, province, geometry.

        Outlines are simplified for display (tolerance in degrees, about 200 m at 0.002):
        invisible at the dashboard's zoom level, but it divides the map payload sent to the
        browser. The full outlines remain in data/processed/south_departments.geojson.
        """
        import geopandas as gpd  # only needed when writing
        shapes = departments.geometry.simplify(map_tolerance, preserve_topology=True) \
            if map_tolerance else departments.geometry
        rows = [(r.dept_id, r.department, r.province, gpd.GeoSeries([shape]).to_json())
                for r, shape in zip(departments.itertuples(), shapes)]
        with self._connect() as con:
            con.execute("DELETE FROM departments")
            con.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", rows)

    def replace_panel(self, panel: pd.DataFrame) -> None:
        values = panel[PANEL_COLUMNS].astype({c: int for c in PANEL_COLUMNS[2:]})
        with self._connect() as con:
            con.execute("DELETE FROM panel")
            con.executemany(
                f"INSERT INTO panel ({', '.join(PANEL_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(PANEL_COLUMNS))})",
                values.itertuples(index=False))

    def replace_predictions(self, tag: str, predictions: pd.DataFrame) -> None:
        """Replaces the predictions of one experiment (`tag`, e.g. operational_quarter)."""
        with self._connect() as con:
            con.execute("DELETE FROM predictions WHERE model LIKE ?", (f"%{tag}%",))
            con.executemany(
                "INSERT INTO predictions (dept_id, ym, model, risk) VALUES (?, ?, ?, ?)",
                predictions[["dept_id", "ym", "model", "risk"]].itertuples(index=False))

    def replace_explanations(self, tag: str, rows: list[tuple]) -> None:
        """Per-department explanation values: (dept_id, ym, 'contrib|tag|feature', value)."""
        with self._connect() as con:
            con.execute("DELETE FROM features WHERE name LIKE ?", (f"%|{tag}|%",))
            con.executemany("INSERT INTO features (dept_id, ym, name, value) "
                            "VALUES (?, ?, ?, ?)", rows)

    # ---------- reading ----------
    def _read(self, query: str, params=()) -> pd.DataFrame:
        with self._connect(must_exist=True) as con:
            return pd.read_sql(query, con, params=params)

    def departments(self) -> pd.DataFrame:
        return self._read("SELECT * FROM departments")

    def panel(self) -> pd.DataFrame:
        return self._read("SELECT * FROM panel")

    def predictions(self) -> pd.DataFrame:
        return self._read("SELECT * FROM predictions")

    def explanations(self, tag: str) -> pd.DataFrame:
        return self._read("SELECT * FROM features WHERE name LIKE ?", (f"%|{tag}|%",))

    def counts(self) -> dict[str, int]:
        with self._connect(must_exist=True) as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("departments", "panel", "features", "predictions")}
