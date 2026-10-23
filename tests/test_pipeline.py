"""
Automated tests for the data pipeline, the model inputs and the dashboard.

They check the properties the results depend on:
    - the labels and the panel are internally consistent;
    - no feature uses information from the month being predicted (no look-ahead);
    - outcome columns never enter a feature set (no leakage);
    - the database and the predictions are valid;
    - the dashboard runs without errors, in English and in French.

Run after the pipeline (README, steps 2-8):  python -m pytest -v
"""

import json
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
    # Panel period: 2017-01 to 2025-09 (an event on 2025-10-02 is outside it by design)
    in_period = ev["event_date"].between(panel["ym"].min() + "-01", "2025-09-30")
    south_fh = ev[(ev["target_fh"] == 1) & ev["is_south"] & in_period]
    assert panel["n_fh"].sum() == len(south_fh)


# ---------- No look-ahead ----------
def test_history_features_use_only_past_months(features):
    for _, g in features.groupby("dept_id"):
        n = g["n_fh"].fillna(0).to_numpy()
        h1 = g["h_fh_1m"].to_numpy()
        h12 = g["h_fh_12m"].to_numpy()
        for t in range(13, len(g)):
            if np.isnan(h1[t]):  # after the conflict data: empty by design (tested below)
                continue
            assert h1[t] == n[t - 1]
            assert h12[t] == n[t - 12:t].sum()


def test_operational_features_use_only_conflicts_13_months_old(features):
    for _, g in features.groupby("dept_id"):
        n = g["n_fh"].to_numpy()
        o12 = g["o_fh_12m"].to_numpy()
        known = ~np.isnan(n)
        for t in range(25, len(g)):
            window = n[t - 24:t - 12]  # months t-24 .. t-13
            if known[t - 24:t - 12].all():
                assert o12[t] == window.sum()


def test_research_history_is_empty_without_recent_conflict_data(features):
    data_end = features.loc[features["n_fh"].notna(), "ym"].max()
    after = features[features["ym"] > data_end]
    research_month = (pd.Period(data_end, freq="M") + 1).strftime("%Y-%m")
    assert after.loc[after["ym"] > research_month, "h_fh_12m"].isna().all()
    assert after["o_fh_12m"].notna().all()


def test_satellite_features_are_lagged_one_month(features):
    sat = pd.read_csv(PROC / "satellite_features.csv")
    sat["ym_next"] = (pd.PeriodIndex(sat["ym"], freq="M") + 1).astype(str)
    merged = features.merge(sat, left_on=["dept_id", "ym"], right_on=["dept_id", "ym_next"],
                            suffixes=("", "_raw"))
    assert len(merged) > 0
    for col in ["south_ndvi_crop_anom", "south_rain_mm", "north_rain_anom"]:
        assert np.allclose(merged[col], merged[f"{col}_raw"])


def test_no_outcome_column_is_a_feature(features):
    feature_cols = [c for c in features.columns
                    if c.startswith(("h_", "o_", "south_", "north_"))]
    assert not OUTCOMES & set(feature_cols)


def test_forecast_rows_have_no_target(features):
    data_end = features.loc[features["n_fh"].notna(), "ym"].max()
    future = features[features["ym"] > data_end]
    assert len(future) % 26 == 0
    assert future["y_fh"].isna().all()
    assert features[features["ym"] == features["ym"].max()]["south_rain_mm"].notna().all()


# ---------- Database and predictions ----------
def test_database_tables_and_predictions():
    with sqlite3.connect(DB) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"departments", "panel", "features", "predictions"} <= tables
        preds = pd.read_sql("SELECT * FROM predictions", con)
    assert preds["risk"].between(0, 1).all()
    for tag in ("research_month", "research_quarter", "operational_month",
                "operational_quarter"):
        sub = preds[preds["model"].str.contains(tag)]
        assert not sub.empty, f"no {tag} predictions"
        fc = sub[sub["model"].str.endswith("_forecast")]
        assert fc["dept_id"].nunique() == 26 and fc["ym"].nunique() == 1
    # The operational outlook is issued after the end of the conflict data
    ops_fc = preds[preds["model"].str.contains("operational_quarter_forecast")]
    assert ops_fc["ym"].iloc[0] > "2025-10"


def test_metrics_reports_exist():
    for tag in ("research_month", "research_quarter", "operational_month",
                "operational_quarter"):
        m = pd.read_csv(ROOT / "reports" / f"metrics_{tag}.csv")
        assert "rule_history_12m" in set(m["model"])
        assert m["pr_auc"].between(0, 1).all()


def test_priority_levels_and_outlook_metadata():
    tiers = pd.read_csv(ROOT / "reports" / "tiers_operational_quarter.csv")
    pri = tiers[tiers["tier"] == "Priority"].iloc[0]
    low = tiers[tiers["tier"] == "Lower"].iloc[0]
    assert pri["periods"] + low["periods"] == 26 * tiers["periods"].iloc[0] / 5
    meta = json.loads((ROOT / "reports" / "outlook_operational_quarter.json").read_text("utf8"))
    assert meta["forecast_start"] > meta["satellite_data_end"] > meta["conflict_data_end"]


# ---------- Dashboard ----------
def test_dashboard_runs_in_english_and_french():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Current outlook", "Replay 2024–2025", "Findings",
                                          "Observed conflicts", "About the models"]
    assert any("Research question" in m.value for m in at.markdown)
    at.sidebar.radio[0].set_value("Français").run()
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "Perspectives actuelles"
