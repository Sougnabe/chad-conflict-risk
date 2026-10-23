"""Rolling-origin (expanding window) model selection for time-ordered data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from chadrisk.models.base import RiskModel


@dataclass
class SelectionResult:
    model: RiskModel          # chosen hyperparameters, refitted on all selection data
    validation_score: float   # mean PR-AUC over the validation folds


class RollingOriginSelector:
    """Chooses hyperparameters without ever training on the future.

    For each validation year v: train on all years before v, validate on v. The candidate
    with the best mean PR-AUC over the folds is refitted on every selection year.
    Folds without a positive case (in training or validation) are skipped.
    """

    def __init__(self, validation_years, year_column: str = "year"):
        self.validation_years = list(validation_years)
        self.year_column = year_column

    def folds(self, data: pd.DataFrame, target: str):
        for year in self.validation_years:
            train = data[data[self.year_column] < year]
            valid = data[data[self.year_column] == year]
            if train[target].sum() > 0 and valid[target].sum() > 0:
                yield train, valid

    def select(self, family: type[RiskModel], data: pd.DataFrame, features: list[str],
               target: str) -> SelectionResult:
        folds = list(self.folds(data, target))
        if not folds:
            raise ValueError("no usable validation fold (no positive case)")
        n_candidates = len(family.candidates(data[target]))
        best_index, best_score = None, -np.inf
        for i in range(n_candidates):
            scores = []
            for train, valid in folds:
                # candidates may depend on the class balance of the fold's training data
                model = family.candidates(train[target])[i].fit(train[features], train[target])
                scores.append(average_precision_score(valid[target],
                                                      model.predict_risk(valid[features])))
            if np.mean(scores) > best_score:
                best_index, best_score = i, float(np.mean(scores))
        final = family.candidates(data[target])[best_index].fit(data[features], data[target])
        return SelectionResult(model=final, validation_score=best_score)
