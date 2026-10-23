"""Unit tests of the risk models: one interface, three implementations."""

import numpy as np
import pandas as pd
import pytest

from chadrisk.models import (FAMILIES, HistoryRuleModel, LogisticRiskModel, RiskModel,
                             XGBoostRiskModel)


@pytest.fixture
def separable():
    """Risk rises with x1; x2 is noise."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": np.r_[rng.normal(0, 1, 80), rng.normal(3, 1, 20)],
                      "x2": rng.normal(0, 1, 100)})
    y = pd.Series([0] * 80 + [1] * 20)
    return X, y


@pytest.mark.parametrize("family", list(FAMILIES.values()))
def test_every_family_follows_the_interface(family, separable):
    X, y = separable
    for model in family.candidates(y):
        assert isinstance(model, RiskModel)
    model = family.candidates(y)[0].fit(X, y)
    scores = model.predict_risk(X)
    assert scores.shape == (100,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[y == 1].mean() > scores[y == 0].mean()  # learns the signal


@pytest.mark.parametrize("family", list(FAMILIES.values()))
def test_clone_keeps_hyperparameters_but_not_training(family, separable):
    X, y = separable
    model = family.candidates(y)[-1].fit(X, y)
    copy = model.clone()
    assert copy.hyperparameters == model.hyperparameters
    with pytest.raises(RuntimeError):
        copy.predict_risk(X)


def test_predict_before_fit_is_an_error(separable):
    with pytest.raises(RuntimeError):
        LogisticRiskModel().predict_risk(separable[0])


def test_missing_feature_is_an_error(separable):
    X, y = separable
    model = LogisticRiskModel().fit(X, y)
    with pytest.raises(ValueError, match="missing features"):
        model.predict_risk(X[["x1"]])


def test_single_class_training_data_is_an_error(separable):
    X, _ = separable
    with pytest.raises(ValueError, match="both classes"):
        LogisticRiskModel().fit(X, pd.Series([0] * 100))


def test_mismatched_lengths_are_an_error(separable):
    X, y = separable
    with pytest.raises(ValueError):
        XGBoostRiskModel().fit(X, y.iloc[:50])


def test_logistic_coefficients_and_contributions(separable):
    X, y = separable
    model = LogisticRiskModel(C=1.0).fit(X, y)
    coefs = model.coefficients()
    assert coefs.index[0] == "x1" and coefs["x1"] > 0   # strongest, positive effect
    contrib = model.contributions(X)
    assert contrib.shape == X.shape


def test_xgboost_weights_the_rare_class(separable):
    _, y = separable
    assert XGBoostRiskModel.candidates(y)[0].scale_pos_weight == pytest.approx(4.0)


def test_rule_ranks_by_recent_conflicts():
    X = pd.DataFrame({"h_fh_12m": [0, 3, 1], "h_fh_cum": [9, 0, 0]})
    scores = HistoryRuleModel().fit(X).predict_risk(X)
    assert list(np.argsort(-scores)) == [1, 2, 0]   # recent conflicts first, history breaks ties
    assert HistoryRuleModel.probabilistic is False
