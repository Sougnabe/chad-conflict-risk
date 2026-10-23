"""Unit tests of metrics, bootstrap, rolling-origin selection and priority levels."""

import numpy as np
import pandas as pd
import pytest

from chadrisk.evaluation import (Evaluator, MonthBlockBootstrap, PriorityTiers,
                                 RollingOriginSelector, top_fraction_hits)
from chadrisk.models import LogisticRiskModel


def test_top_fraction_hits():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.arange(10)[::-1].astype(float)   # row 0 ranked first
    recall, precision = top_fraction_hits(y, scores, fraction=0.1)
    assert (recall, precision) == (0.5, 1.0)


def test_perfect_and_random_rankings():
    y = np.array([0] * 90 + [1] * 10)
    months = np.repeat(np.arange(10), 10)
    evaluator = Evaluator(MonthBlockBootstrap(n_resamples=200))
    assert evaluator.evaluate(y, y.astype(float), months)["pr_auc"] == 1.0
    random = evaluator.evaluate(y, np.random.default_rng(1).random(100), months)
    assert random["pr_auc"] < 0.5
    assert random["prevalence"] == 0.1


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.15).astype(int)
    scores = y + rng.normal(0, 0.8, 300)
    months = np.repeat(np.arange(30), 10)
    result = Evaluator(MonthBlockBootstrap(n_resamples=300, seed=5)).evaluate(
        y, scores, months, probabilistic=False)  # scores are not probabilities
    again = Evaluator(MonthBlockBootstrap(n_resamples=300, seed=5)).evaluate(
        y, scores, months, probabilistic=False)
    assert result["pr_auc_lo"] == again["pr_auc_lo"] and result["pr_auc_hi"] == again["pr_auc_hi"]
    assert result["pr_auc_lo"] <= result["pr_auc"] <= result["pr_auc_hi"]


def test_evaluation_without_positive_case_is_an_error():
    with pytest.raises(ValueError):
        Evaluator().evaluate([0, 0, 0], [0.1, 0.2, 0.3], [1, 1, 1])


@pytest.fixture
def yearly_data():
    rng = np.random.default_rng(3)
    rows = []
    for year in range(2019, 2024):
        x = rng.normal(0, 1, 60)
        y = (x + rng.normal(0, 0.5, 60) > 1.2).astype(int)
        rows.append(pd.DataFrame({"year": year, "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_rolling_origin_folds_never_train_on_the_future(yearly_data):
    selector = RollingOriginSelector([2021, 2022, 2023])
    for train, valid in selector.folds(yearly_data, "y"):
        assert train["year"].max() < valid["year"].min()
        assert valid["year"].nunique() == 1


def test_selection_returns_a_fitted_model(yearly_data):
    result = RollingOriginSelector([2021, 2022, 2023]).select(
        LogisticRiskModel, yearly_data, ["x"], "y")
    assert 0 < result.validation_score <= 1
    assert result.model.predict_risk(yearly_data[["x"]]).shape == (len(yearly_data),)


def test_selection_without_usable_fold_is_an_error(yearly_data):
    data = yearly_data.assign(y=0)
    with pytest.raises(ValueError, match="no usable validation fold"):
        RollingOriginSelector([2021]).select(LogisticRiskModel, data, ["x"], "y")


def test_priority_tiers():
    periods = pd.Series(np.repeat(["2024-01", "2024-04"], 26))
    scores = np.tile(np.arange(26)[::-1], 2).astype(float)       # dept 0 ranked first
    y = pd.Series(np.tile([1] * 3 + [0] * 23, 2))                # top 3 had conflicts
    tiers = PriorityTiers().compute(periods, y, scores).set_index("tier")
    assert tiers.loc["Priority", "periods"] == 20
    assert tiers.loc["Priority", "with_conflict"] == 6
    assert tiers.loc["Lower", "conflict_rate"] == 0
