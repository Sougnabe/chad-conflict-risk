"""Risk models sharing one interface (RiskModel)."""

from chadrisk.models.base import RiskModel
from chadrisk.models.boosting import XGBoostRiskModel
from chadrisk.models.logistic import LogisticRiskModel
from chadrisk.models.rule import HistoryRuleModel

# Model families compared in every experiment (the baseline rule is evaluated separately)
FAMILIES = {"logreg": LogisticRiskModel, "xgboost": XGBoostRiskModel}

__all__ = ["RiskModel", "LogisticRiskModel", "XGBoostRiskModel", "HistoryRuleModel",
           "FAMILIES"]
