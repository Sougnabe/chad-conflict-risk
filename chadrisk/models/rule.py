"""Baseline rule: rank departments by their recent conflict count."""

import pandas as pd

from chadrisk.models.base import RiskModel


class HistoryRuleModel(RiskModel):
    """Risk = conflicts in the last 12 known months (+ a small weight on the full history to
    break ties). No training: it is the reference every model must beat."""

    family = "rule"
    probabilistic = False

    def __init__(self, recent: str = "h_fh_12m", cumulative: str = "h_fh_cum"):
        super().__init__()
        self.recent = recent
        self.cumulative = cumulative

    @classmethod
    def candidates(cls, y: pd.Series) -> list["HistoryRuleModel"]:
        return [cls()]

    @property
    def hyperparameters(self) -> str:
        return "-"

    def params(self) -> dict:
        return {"recent": self.recent, "cumulative": self.cumulative}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HistoryRuleModel":
        self.features = [self.recent, self.cumulative]
        return self

    def _fit(self, X, y):  # pragma: no cover - the rule has no parameters to learn
        pass

    def _score(self, X):
        return (X[self.recent] + 0.01 * X[self.cumulative]).to_numpy()
