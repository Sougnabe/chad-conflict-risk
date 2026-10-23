"""Abstract interface of a conflict risk model."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RiskModel(ABC):
    """A model that scores the risk of at least one conflict in a department-period.

    Subclasses define a family name, their hyperparameter candidates, how to fit, and how
    to score. Callers (selection, evaluation, prediction) only use this interface.
    """

    family: str = "base"
    probabilistic: bool = True  # False when scores are rankings, not probabilities

    def __init__(self):
        self.features: list[str] | None = None

    @classmethod
    @abstractmethod
    def candidates(cls, y: pd.Series) -> list["RiskModel"]:
        """Unfitted models to compare during selection (may depend on class balance)."""

    @property
    @abstractmethod
    def hyperparameters(self) -> str:
        """Readable description of the hyperparameters, e.g. 'C=0.1'."""

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Family-specific training."""

    @abstractmethod
    def _score(self, X: pd.DataFrame) -> np.ndarray:
        """Family-specific scoring."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RiskModel":
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        if y.nunique() < 2:
            raise ValueError("training data must contain both classes")
        self.features = list(X.columns)
        self._fit(X, y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        if self.features is None:
            raise RuntimeError(f"{self} must be fitted before predicting")
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")
        return self._score(X[self.features])

    def clone(self) -> "RiskModel":
        """Unfitted copy with the same hyperparameters."""
        return type(self)(**self.params())

    @abstractmethod
    def params(self) -> dict:
        """Constructor arguments (used by clone)."""

    def __repr__(self) -> str:
        return f"{self.family}({self.hyperparameters})"
