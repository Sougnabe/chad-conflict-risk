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

ROOT = Path(__file__).resolve().parents[2]
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


# ---------- Dashboard: functional tests with different inputs ----------
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app" / "app.py")
PAGES_EN = ["Current outlook", "Replay 2024–2025", "Findings", "Observed conflicts",
            "About the models"]


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=180).run()


@pytest.mark.parametrize("language", ["English", "Français"])
def test_every_page_renders_in_both_languages(app, language):
    app.radio(key="language").set_value(language).run()
    for page in range(len(PAGES_EN)):
        app.radio(key="page").set_value(page).run()   # fresh element after each rerun
        assert not app.exception, f"{language} / page {page}"
    assert (app.radio(key="page").options == PAGES_EN) == (language == "English")


def test_page_survives_a_change_of_language(app):
    app.radio(key="page").set_value(2).run()
    app.radio(key="language").set_value("Français").run()
    assert not app.exception
    assert app.radio(key="page").value == 2


def test_findings_page_reads_the_results(app):
    app.radio(key="page").set_value(PAGES_EN.index("Findings")).run()
    text = " ".join(m.value for m in app.markdown)
    assert "Research question" in text and "0.214" in text


def test_every_department_has_an_explanation(app):
    select = app.selectbox[0]
    for department in select.options:
        select.set_value(department).run()
        assert not app.exception
        assert len(app.dataframe[1].value) >= 3, department


def test_every_replay_quarter(app):
    app.radio(key="page").set_value(1).run()
    with sqlite3.connect(DB) as con:
        quarters = sorted(r[0] for r in con.execute(
            "SELECT DISTINCT ym FROM predictions WHERE model LIKE '%operational_quarter_test'"))
    assert len(quarters) == 7
    for quarter in quarters:
        app.select_slider[0].set_value(quarter).run()
        assert not app.exception
        flagged = int(app.metric[1].value)
        assert 0 <= flagged <= int(app.metric[0].value)


def test_observed_page_edge_inputs(app):
    app.radio(key="page").set_value(3).run()
    app.multiselect[0].set_value([]).run()                  # no province selected
    assert not app.exception
    assert app.metric[0].value == "0"
    app.multiselect[0].set_value(["Mandoul"]).run()         # one province
    app.select_slider[0].set_value(("2024-01", "2024-01")).run()  # a single month
    assert not app.exception
    definition = [r for r in app.radio if r.key not in ("page", "language")][-1]
    definition.set_value(definition.options[1]).run()       # broad definition
    assert not app.exception


def test_missing_database_shows_an_error_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("CHADRISK_DB", str(tmp_path / "absent.db"))
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception
    assert app.error and "not found" in app.error[0].value
