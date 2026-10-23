"""Rolling-origin (expanding window) model selection for time-ordered data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

from chadrisk.models.base import RiskModel


def _score_candidate(family, index, train, valid, features, target) -> float:
    """Validation PR-AUC of candidate `index`, trained on one fold.

    Candidates may depend on the class balance of the fold's training data (e.g. XGBoost's
    positive-class weight), so they are rebuilt from that data.
    """
    model = family.candidates(train[target])[index].fit(train[features], train[target])
    return average_precision_score(valid[target], model.predict_risk(valid[features]))


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

    def __init__(self, validation_years, year_column: str = "year", n_jobs: int = -1):
        self.validation_years = list(validation_years)
        self.year_column = year_column
        self.n_jobs = n_jobs  # -1: all processor cores

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
        # Every (candidate, fold) pair is independent: they are scored in parallel processes.
        jobs = [(i, k) for i in range(n_candidates) for k in range(len(folds))]
        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_score_candidate)(family, i, folds[k][0], folds[k][1], features, target)
            for i, k in jobs)
        mean = np.array(scores).reshape(n_candidates, len(folds)).mean(axis=1)
        best_index = int(np.argmax(mean))  # first best candidate on ties
        final = family.candidates(data[target])[best_index].fit(data[features], data[target])
        return SelectionResult(model=final, validation_score=float(mean[best_index]))
