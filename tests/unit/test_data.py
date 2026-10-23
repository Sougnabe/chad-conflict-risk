"""Unit tests of labels, panel, geography, storage, experiment configuration and translations."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from chadrisk.data.geography import Geography
from chadrisk.data.labels import LabelledEvents
from chadrisk.data.panel import PanelBuilder
from chadrisk.experiment import ExperimentConfig
from chadrisk.storage import DatabaseNotFoundError, RiskRepository
from i18n import FINDINGS, TXT, Translator


# ---------- labels ----------
def events():
    return pd.DataFrame({
        "event_id_cnty": ["E1", "E2", "E3"], "event_date": ["2024-01-05"] * 3,
        "admin1": ["Mandoul", "Lac", "Tandjilé"], "fatalities": [2, 0, "1"],
        "notes": ["Farmers and herders clashed", "Boko Haram attack", "cattle in a field"]})


def test_labels_build_targets():
    labels = pd.DataFrame({"event_id_cnty": ["E1", "E3"], "label": ["FH", "AMB"]})
    df = LabelledEvents(events(), labels).build().set_index("event_id_cnty")
    assert df.loc["E2", "label"] == "NO"                          # unlabelled -> NO
    assert df["target_fh"].tolist() == [1, 0, 0]
    assert df["target_broad"].tolist() == [1, 0, 1]
    assert df.loc["E3", "is_south"]                               # accent removed: Tandjile
    precision, recall = LabelledEvents.keyword_agreement(df)
    assert (precision, recall) == (0.5, 1.0)


@pytest.mark.parametrize("labels, message", [
    (pd.DataFrame({"event_id_cnty": ["E1", "E1"], "label": ["FH", "NO"]}), "duplicate"),
    (pd.DataFrame({"event_id_cnty": ["E1"], "label": ["MAYBE"]}), "invalid labels"),
    (pd.DataFrame({"id": ["E1"], "label": ["FH"]}), "columns"),
])
def test_invalid_labels_are_rejected(labels, message):
    with pytest.raises(ValueError, match=message):
        LabelledEvents(events(), labels)


# ---------- panel ----------
def test_panel_is_complete_with_zero_counts():
    departments = pd.DataFrame({"dept_id": ["A", "B"], "department": ["A", "B"],
                                "province": ["P", "P"]})
    ev = pd.DataFrame({"dept_id": ["A", "A", "Z"], "event_date": ["2024-02-03", "2024-02-20",
                                                                  "2024-02-01"],
                       "target_fh": [1, 0, 1], "target_broad": [1, 1, 1],
                       "fatalities": [3, 1, 9], "event_id_cnty": ["1", "2", "3"]})
    panel = PanelBuilder(departments, "2024-01", "2024-03").build(ev).set_index(
        ["dept_id", "ym"])
    assert len(panel) == 6                                   # 2 departments x 3 months
    assert panel.loc[("A", "2024-02"), "n_fh"] == 1
    assert panel.loc[("A", "2024-02"), "n_broad"] == 2
    assert panel.loc[("A", "2024-02"), "fatalities_fh"] == 3  # only FH deaths
    assert panel.loc[("B", "2024-02"), "n_all_events"] == 0   # event in Z ignored


def test_panel_with_inverted_dates_is_rejected():
    with pytest.raises(ValueError):
        PanelBuilder(pd.DataFrame(columns=["dept_id", "department", "province"]),
                     "2024-05", "2024-01")


# ---------- geography ----------
@pytest.fixture
def grid():
    """Three departments in a row: A | B | C (A and C do not touch)."""
    provinces = gpd.GeoDataFrame({"shapeName": ["P"]}, geometry=[box(0, 0, 3, 1)], crs="EPSG:4326")
    departments = gpd.GeoDataFrame({"shapeName": ["A", "B", "C"], "shapeID": ["a", "b", "c"]},
                                   geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1),
                                             box(2, 0, 3, 1)], crs="EPSG:4326")
    return Geography(provinces, departments)


def test_neighbours_graph(grid):
    graph = Geography.neighbours(grid.region(["P"]))
    assert graph == {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def test_events_are_assigned_by_coordinates(grid):
    ev = pd.DataFrame({"longitude": [0.5, 2.5, 9.0], "latitude": [0.5, 0.5, 0.5]})
    assert grid.assign_events(ev)["dept_id"].tolist()[:2] == ["a", "c"]
    assert pd.isna(grid.assign_events(ev)["dept_id"].iloc[2])   # outside every department


def test_unknown_province_is_rejected(grid):
    with pytest.raises(ValueError, match="not found"):
        grid.region(["Atlantis"])


# ---------- storage ----------
def test_repository_round_trip(tmp_path, grid):
    repo = RiskRepository(tmp_path / "test.db")
    repo.create_schema()
    repo.replace_departments(grid.departments.assign(geometry=[Point(0, 0)] * 3))
    preds = pd.DataFrame({"dept_id": ["a", "b"], "ym": ["2026-10"] * 2,
                          "model": ["m_operational_quarter_forecast"] * 2, "risk": [0.2, 0.7]})
    repo.replace_predictions("operational_quarter", preds)
    repo.replace_predictions("research_month", preds.assign(model="m_research_month_test"))
    assert repo.counts()["departments"] == 3
    assert len(repo.predictions()) == 4
    repo.replace_predictions("operational_quarter", preds.iloc[:1])   # only that tag replaced
    assert len(repo.predictions()) == 3


def test_missing_database_raises_a_clear_error(tmp_path):
    with pytest.raises(DatabaseNotFoundError):
        RiskRepository(tmp_path / "absent.db").predictions()


# ---------- experiment configuration ----------
@pytest.mark.parametrize("kwargs", [{"mode": "live"}, {"unit": "week"}])
def test_invalid_experiment_configuration(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_experiment_tag():
    assert ExperimentConfig("operational", "quarter").tag == "operational_quarter"


# ---------- translations ----------
def test_every_text_exists_in_both_languages():
    assert all(len(v) == 2 and all(v) for v in TXT.values())
    assert set(FINDINGS) == {"en", "fr"}


def test_translator():
    fr = Translator("fr")
    assert fr.quarter("2026-10") == "oct.–déc. 2026"
    assert fr.date("2026-11-20") == "20 novembre 2026"
    assert Translator("en").month("2025-09", long=True) == "September 2025"
    with pytest.raises(ValueError):
        Translator("de")
