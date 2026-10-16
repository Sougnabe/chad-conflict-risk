"""
Automated tests for the data pipeline, the model inputs and the dashboard.

They check the properties the results depend on:
    - the labels and the panel are internally consistent;
    - no feature uses information from the month being predicted (no look-ahead);
    - outcome columns never enter a feature set (no leakage);
    - the database and the predictions are valid;
    - the dashboard runs without errors.

Run after the pipeline (README, steps 2-8):  python -m pytest -v
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DB = ROOT / "data" / "app.db"
OUTCOMES = {"n_fh", "n_broad", "n_all_events", "fatalities_fh", "y_fh", "y_broad"}


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PROC / "panel_dept_month.csv")


@pytest.fixture(scope="module")
def features():
    return pd.read_csv(PROC / "features.csv").sort_values(["dept_id", "ym"])


# ---------- Labels ----------
def test_labels_are_unique_and_valid():
    labels = pd.read_csv(PROC / "manual_labels.csv")
    assert labels["event_id_cnty"].is_unique
    assert set(labels["label"]) <= {"FH", "AMB", "REL", "NO"}


def test_labelled_dataset_matches_codebook_counts():
    ev = pd.read_csv(PROC / "acled_chad_labeled.csv")
    assert len(ev) == 1593
    assert ev["target_fh"].sum() == 154
    assert set(ev["target_broad"][ev["target_fh"] == 1]) == {1}  # FH is a subset of broad


# ---------- Panel ----------
def test_panel_is_complete_department_by_month(panel):
    assert panel["dept_id"].nunique() == 26
    assert not panel.duplicated(["dept_id", "ym"]).any()
    assert len(panel) == 26 * panel["ym"].nunique()


def test_panel_targets_follow_counts(panel):
    assert (panel["y_fh"] == (panel["n_fh"] > 0)).all()
    assert (panel["y_broad"] == (panel["n_broad"] > 0)).all()
    assert (panel["n_broad"] >= panel["n_fh"]).all()


def test_panel_keeps_all_southern_conflicts(panel):
    ev = pd.read_csv(PROC / "acled_chad_labeled.csv", parse_dates=["event_date"])
    # Panel period: 2018-01 to 2025-09 (an event on 2025-10-02 is outside it by design)
    in_period = ev["event_date"].between("2018-01-01", "2025-09-30")
    south_fh = ev[(ev["target_fh"] == 1) & ev["is_south"] & in_period]
    assert panel["n_fh"].sum() == len(south_fh)


# ---------- No look-ahead ----------
def test_history_features_use_only_past_months(features):
    for _, g in features.groupby("dept_id"):
        n = g["n_fh"].fillna(0).to_numpy()
        h1 = g["h_fh_1m"].to_numpy()
        h12 = g["h_fh_12m"].to_numpy()
        for t in range(13, len(g)):
            assert h1[t] == n[t - 1]
            assert h12[t] == n[t - 12:t].sum()


def test_satellite_features_are_lagged_one_month(features):
    sat = pd.read_csv(PROC / "satellite_features.csv")
    sat["ym_next"] = (pd.PeriodIndex(sat["ym"], freq="M") + 1).astype(str)
    merged = features.merge(sat, left_on=["dept_id", "ym"], right_on=["dept_id", "ym_next"],
                            suffixes=("", "_raw"))
    assert len(merged) > 0
    for col in ["south_ndvi_crop_anom", "south_rain_mm", "north_rain_anom"]:
        assert np.allclose(merged[col], merged[f"{col}_raw"])


def test_no_outcome_column_is_a_feature(features):
    feature_cols = [c for c in features.columns if c.startswith(("h_", "south_", "north_"))]
    assert not OUTCOMES & set(feature_cols)


def test_forecast_row_has_no_target(features):
    last = features["ym"].max()
    future = features[features["ym"] == last]
    assert len(future) == 26
    assert future["y_fh"].isna().all()
    assert future["h_fh_1m"].notna().all()


# ---------- Database and predictions ----------
def test_database_tables_and_predictions():
    with sqlite3.connect(DB) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"departments", "panel", "features", "predictions"} <= tables
        preds = pd.read_sql("SELECT * FROM predictions", con)
    assert preds["risk"].between(0, 1).all()
    for unit in ("month", "quarter"):
        sub = preds[preds["model"].str.contains(unit)]
        assert not sub.empty, f"no {unit} predictions"
        assert (sub[sub["model"].str.endswith("_forecast")]["dept_id"].nunique()) == 26


def test_metrics_reports_exist():
    for unit in ("month", "quarter"):
        m = pd.read_csv(ROOT / "reports" / f"metrics_{unit}.csv")
        assert "rule_history_12m" in set(m["model"])
        assert m["pr_auc"].between(0, 1).all()


# ---------- Dashboard ----------
def test_dashboard_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Observed conflicts", "Predicted risk",
                                          "About the model"]
