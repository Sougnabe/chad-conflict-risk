"""Regularised logistic regression: stable and interpretable with few positive cases."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chadrisk.models.base import RiskModel


class LogisticRiskModel(RiskModel):
    """Standardised inputs + L2-regularised, class-weighted logistic regression."""

    family = "logreg"
    GRID = (0.01, 0.1, 1.0)  # inverse regularisation strength

    def __init__(self, C: float = 0.1):
        super().__init__()
        self.C = C
        self._pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000))

    @classmethod
    def candidates(cls, y: pd.Series) -> list["LogisticRiskModel"]:
        return [cls(C=c) for c in cls.GRID]

    @property
    def hyperparameters(self) -> str:
        return f"C={self.C:g}"

    def params(self) -> dict:
        return {"C": self.C}

    def _fit(self, X, y):
        self._pipeline.fit(X, y)

    def _score(self, X):
        return self._pipeline.predict_proba(X)[:, 1]

    def coefficients(self) -> pd.Series:
        """Standardised coefficients, largest effects first (positive = higher risk)."""
        coef = self._pipeline.named_steps["logisticregression"].coef_[0]
        return pd.Series(coef, index=self.features).sort_values(key=abs, ascending=False)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribution of each feature to the log-odds of each row:
        standardised coefficient x standardised value."""
        z = self._pipeline.named_steps["standardscaler"].transform(X[self.features])
        coef = self._pipeline.named_steps["logisticregression"].coef_[0]
        return pd.DataFrame(z * coef, columns=self.features, index=X.index)
