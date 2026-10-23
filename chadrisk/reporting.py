"""Writes the outputs of an experiment: tables, figures, metadata and database rows."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from chadrisk.config import Paths  # noqa: E402
from chadrisk.experiment import ExperimentResult  # noqa: E402
from chadrisk.storage import RiskRepository  # noqa: E402


class ExperimentReporter:
    """Saves everything the report and the dashboard need from one experiment."""

    def __init__(self, paths: Paths, repository: RiskRepository):
        self.paths = paths
        self.repository = repository
        paths.figures.mkdir(parents=True, exist_ok=True)

    def save(self, result: ExperimentResult) -> None:
        tag = result.config.tag
        reports = self.paths.reports
        result.metrics.to_csv(reports / f"metrics_{tag}.csv", index=False)
        result.tiers.to_csv(reports / f"tiers_{tag}.csv", index=False)
        self._json(reports / f"selection_{tag}.json", {
            "auto_selected": result.auto_selected, "used_for_predictions": result.used,
            "fixed_by_design": result.config.fixed_model is not None})
        self._json(reports / f"sensitivity_broad_{tag}.json",
                   {"model": result.used, **{k: float(v) for k, v in result.sensitivity.items()}})
        if result.coefficients is not None:
            result.coefficients.to_csv(reports / f"coefficients_{tag}.csv",
                                       header=["std_coefficient"])
        self._json(reports / f"outlook_{tag}.json", self.outlook_metadata(result))
        self._precision_recall_figure(result)
        self._shap_figure(result)
        self._predictions(result)

    @staticmethod
    def _json(path, content):
        path.write_text(json.dumps(content, indent=2), encoding="utf8")

    @staticmethod
    def outlook_metadata(result: ExperimentResult) -> dict:
        """Dates of the forecast and of the data it uses (shown by the dashboard)."""
        config = result.config
        span = 2 if config.unit == "quarter" else 0
        start = pd.Period(result.forecast["ym"].iloc[0], freq="M")
        operational = config.mode == "operational"
        return {
            "mode": config.mode, "unit": config.unit, "model": result.used,
            "issued_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "forecast_start": str(start), "forecast_end": str(start + span),
            "conflict_data_end": str(pd.Period(result.test["ym"].max(), freq="M") + span),
            "satellite_data_end": str(start - 1) if operational else None,
            # a satellite month is usually published about three weeks after it ends
            "next_update_expected": f"{start + 1}-20" if operational else None,
        }

    def _precision_recall_figure(self, result: ExperimentResult) -> None:
        y = result.test["y_fh"]
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, fitted in result.fitted.items():
            pr, rc, _ = precision_recall_curve(y, fitted.test_scores)
            ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, fitted.test_scores):.2f})")
        pr, rc, _ = precision_recall_curve(y, result.rule_scores)
        ax.plot(rc, pr, "k--",
                label=f"rule (AP={average_precision_score(y, result.rule_scores):.2f})")
        ax.axhline(y.mean(), color="grey", lw=0.8, label="random")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-recall curves ({result.config.mode}, {result.config.unit}), "
                     "test period 2024 - 2025-09")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.paths.figures / f"pr_curves_{result.config.tag}.png", dpi=150)
        plt.close(fig)

    def _shap_figure(self, result: ExperimentResult) -> None:
        """SHAP values of the XGBoost model trained on the same feature set as the used one."""
        import shap  # heavy import, only needed here
        set_name = result.used.split("_", 1)[1]
        fitted = result.fitted[f"xgboost_{set_name}"]
        X = result.test[fitted.features]
        values = shap.TreeExplainer(fitted.model.booster).shap_values(X)
        shap.summary_plot(values, X, show=False, max_display=15)
        plt.title(f"SHAP values, xgboost_{set_name} ({result.config.mode}, "
                  f"{result.config.unit}), test period")
        plt.tight_layout()
        plt.savefig(self.paths.figures / f"shap_summary_{result.config.tag}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        importance = pd.Series(np.abs(values).mean(axis=0), index=fitted.features)
        importance.sort_values(ascending=False).to_csv(
            self.paths.reports / f"shap_importance_{result.config.tag}.csv",
            header=["mean_abs_shap"])

    def _predictions(self, result: ExperimentResult) -> None:
        tag, used = result.config.tag, result.used
        chosen = result.fitted[used]
        predictions = pd.concat([
            result.test[["dept_id", "ym"]].assign(model=f"{used}_{tag}_test",
                                                  risk=chosen.test_scores),
            result.forecast[["dept_id", "ym", "risk"]].assign(model=f"{used}_{tag}_forecast"),
        ])
        self.repository.replace_predictions(tag, predictions)
        if result.explanations:
            self.repository.replace_explanations(tag, result.explanations)
