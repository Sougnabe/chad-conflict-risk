"""Gradient-boosted trees (XGBoost): tests whether non-linear interactions help."""

import pandas as pd
from xgboost import XGBClassifier

from chadrisk.config import SEED
from chadrisk.models.base import RiskModel


class XGBoostRiskModel(RiskModel):
    """Shallow boosted trees, re-weighted for the rare positive class."""

    family = "xgboost"

    def __init__(self, max_depth: int = 2, n_estimators: int = 100, min_child_weight: int = 1,
                 scale_pos_weight: float = 1.0):
        super().__init__()
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self._model = XGBClassifier(
            max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.05,
            min_child_weight=min_child_weight, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=SEED,
            n_jobs=4)

    @classmethod
    def candidates(cls, y: pd.Series) -> list["XGBoostRiskModel"]:
        spw = (len(y) - y.sum()) / max(y.sum(), 1)  # negatives per positive in this data
        return [cls(max_depth=d, n_estimators=n, min_child_weight=m, scale_pos_weight=spw)
                for d in (2, 3) for n in (100, 300) for m in (1, 5)]

    @property
    def hyperparameters(self) -> str:
        return f"depth={self.max_depth},n={self.n_estimators},mcw={self.min_child_weight}"

    def params(self) -> dict:
        return {"max_depth": self.max_depth, "n_estimators": self.n_estimators,
                "min_child_weight": self.min_child_weight,
                "scale_pos_weight": self.scale_pos_weight}

    def _fit(self, X, y):
        self._model.fit(X, y)

    def _score(self, X):
        return self._model.predict_proba(X)[:, 1]

    @property
    def booster(self) -> XGBClassifier:
        """Underlying fitted model (used for SHAP explanations)."""
        return self._model
