"""Validated event labels and the farmer-herder target variable.

Labels (see docs/labeling_codebook.md):
    FH  direct violence between farmers and herders (model target)
    AMB communal / pastoral violence without explicit farmer-herder framing
    REL related but non-violent event (protest, arrests, reconciliation)
    NO  everything else (default for unlabelled events)
"""

from pathlib import Path

import pandas as pd

from chadrisk.config import KEYWORDS, SOUTH_PROVINCES_ACLED

VALID_LABELS = ("FH", "AMB", "REL", "NO")


class LabelledEvents:
    """Joins raw ACLED events with the manual labels and derives the targets."""

    def __init__(self, events: pd.DataFrame, labels: pd.DataFrame):
        self._validate_labels(labels)
        self._events = events
        self._labels = labels

    @classmethod
    def from_files(cls, events_path: Path, labels_path: Path) -> "LabelledEvents":
        return cls(pd.read_csv(events_path), pd.read_csv(labels_path))

    @staticmethod
    def _validate_labels(labels: pd.DataFrame) -> None:
        if not {"event_id_cnty", "label"} <= set(labels.columns):
            raise ValueError("labels need the columns event_id_cnty and label")
        if not labels["event_id_cnty"].is_unique:
            raise ValueError("duplicate event IDs in the labels")
        invalid = set(labels["label"]) - set(VALID_LABELS)
        if invalid:
            raise ValueError(f"invalid labels: {sorted(invalid)}")

    def build(self) -> pd.DataFrame:
        """Events with label, region flags and targets (target_fh, target_broad)."""
        df = self._events.merge(self._labels, on="event_id_cnty", how="left")
        df["label"] = df["label"].fillna("NO")
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["month"] = df["event_date"].dt.month
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
        df["admin1_norm"] = (df["admin1"].str.normalize("NFKD").str.encode("ascii", "ignore")
                             .str.decode("ascii"))
        df["is_south"] = df["admin1_norm"].isin(SOUTH_PROVINCES_ACLED)
        df["keyword_hit"] = df["notes"].str.contains(KEYWORDS, case=False, na=False)
        df["target_fh"] = (df["label"] == "FH").astype(int)
        df["target_broad"] = df["label"].isin(["FH", "AMB"]).astype(int)  # sensitivity target
        return df

    @staticmethod
    def keyword_agreement(df: pd.DataFrame) -> tuple[float, float]:
        """Precision and recall of the keyword filter against the manual labels."""
        hits = df["keyword_hit"] & (df["target_fh"] == 1)
        return hits.sum() / df["keyword_hit"].sum(), hits.sum() / df["target_fh"].sum()

    @staticmethod
    def validation_sample(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
        """Stratified sample for blind human validation (labels hidden)."""
        parts = [
            df[df["label"] == "FH"].sample(30, random_state=seed),
            df[df["label"] == "AMB"].sample(min(15, (df["label"] == "AMB").sum()),
                                            random_state=seed),
            df[(df["label"] == "NO") & df["keyword_hit"]].sample(15, random_state=seed),
        ]
        sample = pd.concat(parts).sample(frac=1, random_state=seed)
        return sample[["event_id_cnty", "event_date", "admin1", "notes"]].assign(
            author_label="")
