"""Project paths and constants, defined once and shared by every module."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """File locations of the project. Tests can pass another root to work in isolation."""

    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def database(self) -> Path:
        return self.root / "data" / "app.db"

    @property
    def acled_raw(self) -> Path:
        return self.raw / "acled_chad_2015_2025.csv"

    @property
    def manual_labels(self) -> Path:
        return self.processed / "manual_labels.csv"

    @property
    def labelled_events(self) -> Path:
        return self.processed / "acled_chad_labeled.csv"

    @property
    def validation_sample(self) -> Path:
        return self.processed / "validation_sample.csv"

    @property
    def south_departments(self) -> Path:
        return self.processed / "south_departments.geojson"

    @property
    def panel(self) -> Path:
        return self.processed / "panel_dept_month.csv"

    @property
    def satellite(self) -> Path:
        return self.processed / "satellite_features.csv"

    @property
    def features(self) -> Path:
        return self.processed / "features.csv"


PATHS = Paths()

# Provinces, as spelled in geoBoundaries (boundaries) and in ACLED (event records)
SOUTH_PROVINCES = ("Logone Occidental", "Logone Oriental", "Mandoul", "Moyen-Chari",
                   "Tandjilé", "Mayo-Kebbi Est", "Mayo-Kebbi Ouest")
SOUTH_PROVINCES_ACLED = ("Logone Occidental", "Logone Oriental", "Mandoul", "Moyen-Chari",
                         "Tandjile", "Mayo Kebbi Est", "Mayo Kebbi Ouest")
NORTH_PROVINCES = ("Batha", "Guéra", "Salamat", "Ouaddaï")  # herd departure zones

# Time frame
PANEL_START = "2017-01"   # 2017-2018: burn-in for lagged history features only
CONFLICT_DATA_END = "2025-09"  # ACLED Research tier: 12-month publication delay
STUDY_START_YEAR = 2019
VALIDATION_YEARS = (2021, 2022, 2023)  # rolling-origin model selection
TEST_START_YEAR = 2024
ACLED_DELAY_MONTHS = 12

# Columns describing the outcome of the predicted period: never allowed as features
OUTCOME_COLUMNS = frozenset({"n_fh", "n_broad", "n_all_events", "fatalities_fh", "y_fh",
                             "y_broad"})

# Keyword filter used only to compare with the manual classification
KEYWORDS = (r"herder|herdsm|pastoral|cattle|livestock|transhuman|nomad|grazing|"
            r"farmer|crop|field|farm|camel|cow|oxen|herd")

SEED = 42
N_DEPARTMENTS = 26
