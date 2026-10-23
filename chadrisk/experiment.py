"""One training and evaluation run: a mode (research / operational) and a unit (month / quarter).

research     conflict history up to month t-1; answers the research question
             (A = history, B = A + southern satellite, C = B + northern satellite)
operational  only information really available when a forecast is issued: conflict history up
             to month t-13 (ACLED 12-month delay) and satellite data up to t-1
             (OA = delayed history + season, OB = OA + satellite); produces the outlook

Temporal protocol: model selection on 2019-2023 (rolling origin), single evaluation on
2024 - 2025-09, never used for tuning.
"""

from dataclasses import dataclass, field

import pandas as pd

from chadrisk.config import (N_DEPARTMENTS, OUTCOME_COLUMNS, STUDY_START_YEAR,
                             TEST_START_YEAR, VALIDATION_YEARS)
from chadrisk.evaluation import Evaluator, PriorityTiers, RollingOriginSelector
from chadrisk.models import FAMILIES, HistoryRuleModel, LogisticRiskModel, RiskModel

MODES = ("research", "operational")
UNITS = ("month", "quarter")
SEASON = ["h_month_sin", "h_month_cos"]


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "research"
    unit: str = "quarter"
    fixed_model: str | None = None  # e.g. "logreg_OB": used for predictions by design

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")

    @property
    def tag(self) -> str:
        return f"{self.mode}_{self.unit}"


@dataclass
class FittedModel:
    model: RiskModel
    features: list[str]
    test_scores: object


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    test: pd.DataFrame
    metrics: pd.DataFrame
    fitted: dict[str, FittedModel]
    rule_scores: object
    auto_selected: str
    used: str
    tiers: pd.DataFrame
    sensitivity: dict
    forecast: pd.DataFrame
    final_model: RiskModel
    coefficients: pd.Series | None = None
    explanations: list = field(default_factory=list)


class Experiment:
    """Prepares the data for one configuration, selects, evaluates and predicts."""

    def __init__(self, config: ExperimentConfig, features: pd.DataFrame,
                 evaluator: Evaluator | None = None,
                 selector: RollingOriginSelector | None = None):
        self.config = config
        self.evaluator = evaluator or Evaluator()
        self.selector = selector or RollingOriginSelector(VALIDATION_YEARS)
        self.data, self.forecast_rows = self._prepare(features)
        self.selection_data = self.data[self.data["year"] < TEST_START_YEAR]
        self.test = self.data[self.data["year"] >= TEST_START_YEAR]
        self.feature_sets = self._feature_sets()

    # ---------- data preparation ----------
    def _prepare(self, features: pd.DataFrame):
        df = features.copy()
        if self.config.unit == "quarter":
            # target = at least one conflict in months t, t+1, t+2; one row per quarter start
            df = df.sort_values(["dept_id", "ym"]).reset_index(drop=True)
            for y in ["y_fh", "y_broad"]:
                g = df.groupby("dept_id")[y]
                df[y] = pd.concat([df[y], g.shift(-1), g.shift(-2)], axis=1).max(
                    axis=1, skipna=False)
            df = df[df["month"].isin([1, 4, 7, 10])]
        df = df[df["year"] >= STUDY_START_YEAR].reset_index(drop=True)
        forecast_rows = df[df["y_fh"].isna()].copy()
        data = df[df["y_fh"].notna()].copy()
        data[["y_fh", "y_broad"]] = data[["y_fh", "y_broad"]].astype(int)
        return data, forecast_rows

    def _feature_sets(self) -> dict[str, list[str]]:
        cols = self.data.columns
        h = [c for c in cols if c.startswith("h_")]
        o = [c for c in cols if c.startswith("o_")] + SEASON
        south = [c for c in cols if c.startswith("south_")]
        north = [c for c in cols if c.startswith("north_")]
        if self.config.mode == "research":
            sets = {"A": h}
            if south:
                sets["B"] = h + south
            if north:
                sets["C"] = h + south + north
        else:
            sets = {"OA": o}
            if south or north:
                sets["OB"] = o + south + north
        for name, columns in sets.items():  # leakage guard
            leaked = OUTCOME_COLUMNS & set(columns)
            if leaked:
                raise ValueError(f"outcome columns used as features in set {name}: {leaked}")
        return sets

    def baseline(self) -> HistoryRuleModel:
        prefix = "h_" if self.config.mode == "research" else "o_"
        return HistoryRuleModel(recent=f"{prefix}fh_12m", cumulative=f"{prefix}fh_cum")

    # ---------- run ----------
    def run(self) -> ExperimentResult:
        months = self.test["ym"].to_numpy()
        y_test = self.test["y_fh"]
        rows, fitted = [], {}

        rule = self.baseline().fit(self.test)
        rule_scores = rule.predict_risk(self.test)
        rows.append({"model": "rule_history_12m", "features": "-", "hyperparams": "-",
                     "val_pr_auc": float("nan"),
                     **self.evaluator.evaluate(y_test, rule_scores, months,
                                               probabilistic=False)})

        for set_name, columns in self.feature_sets.items():
            for family_name, family in FAMILIES.items():
                result = self.selector.select(family, self.selection_data, columns, "y_fh")
                scores = result.model.predict_risk(self.test)
                name = f"{family_name}_{set_name}"
                fitted[name] = FittedModel(result.model, columns, scores)
                rows.append({"model": name, "features": set_name,
                             "hyperparams": result.model.hyperparameters,
                             "val_pr_auc": result.validation_score,
                             **self.evaluator.evaluate(y_test, scores, months)})
        metrics = pd.DataFrame(rows)

        candidates = metrics[metrics["model"] != "rule_history_12m"]
        auto = candidates.loc[candidates["val_pr_auc"].idxmax(), "model"]
        used = self.config.fixed_model or auto
        if used not in fitted:
            raise ValueError(f"unknown model {used}; available: {sorted(fitted)}")
        chosen = fitted[used]

        tiers = PriorityTiers().compute(self.test["ym"], y_test, chosen.test_scores)
        sensitivity = self._sensitivity(type(chosen.model), chosen.features, months)
        final = chosen.model.clone().fit(self.data[chosen.features], self.data["y_fh"])
        forecast = self._forecast(final, chosen.features)

        result = ExperimentResult(
            config=self.config, test=self.test, metrics=metrics, fitted=fitted,
            rule_scores=rule_scores, auto_selected=auto, used=used, tiers=tiers,
            sensitivity=sensitivity, forecast=forecast, final_model=final)
        if isinstance(chosen.model, LogisticRiskModel):
            result.coefficients = chosen.model.coefficients()
            result.explanations = self._explanations(final, forecast)
        return result

    def _sensitivity(self, family, columns, months) -> dict:
        """Same model family and features, trained and tested on the broad target."""
        model = self.selector.select(family, self.selection_data, columns, "y_broad").model
        return self.evaluator.evaluate(self.test["y_broad"], model.predict_risk(self.test),
                                       months)

    def _forecast(self, model: RiskModel, columns: list[str]) -> pd.DataFrame:
        """Latest period whose features are all available."""
        rows = self.forecast_rows[self.forecast_rows[columns].notna().all(axis=1)]
        rows = rows[rows["ym"] == rows["ym"].max()].copy()
        if rows["dept_id"].nunique() != N_DEPARTMENTS:
            raise ValueError("incomplete forecast rows")
        rows["risk"] = model.predict_risk(rows)
        return rows

    def _explanations(self, model: LogisticRiskModel, forecast: pd.DataFrame) -> list[tuple]:
        """Contribution and value of each feature, per department, for the dashboard."""
        contrib = model.contributions(forecast)
        tag, out = self.config.tag, []
        for i, r in enumerate(forecast.itertuples()):
            for col in model.features:
                out.append((r.dept_id, r.ym, f"contrib|{tag}|{col}",
                            float(contrib.iloc[i][col])))
                out.append((r.dept_id, r.ym, f"value|{tag}|{col}", float(getattr(r, col))))
        return out
