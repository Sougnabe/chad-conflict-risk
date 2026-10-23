"""Metrics for rare-event risk rankings, with month-block bootstrap confidence intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from chadrisk.config import SEED


def top_fraction_hits(y, scores, fraction: float = 0.10) -> tuple[float, float]:
    """Recall and precision within the `fraction` of rows ranked highest by `scores`."""
    y, scores = np.asarray(y), np.asarray(scores)
    k = max(1, int(round(fraction * len(y))))
    top = np.argsort(-scores, kind="stable")[:k]
    hits = y[top].sum()
    return hits / max(y.sum(), 1), hits / k


class MonthBlockBootstrap:
    """Confidence interval of a metric by resampling whole months.

    Departments observed in the same month are correlated (shared weather, season, political
    context), so months, not rows, are resampled with replacement.
    """

    def __init__(self, n_resamples: int = 1000, seed: int = SEED, level: float = 0.95):
        self.n_resamples = n_resamples
        self.seed = seed
        self.level = level

    def interval(self, metric, y, scores, months) -> tuple[float, float]:
        y, scores, months = np.asarray(y), np.asarray(scores), np.asarray(months)
        rng = np.random.default_rng(self.seed)
        unique = np.unique(months)
        rows_of = {m: np.where(months == m)[0] for m in unique}
        values = []
        for _ in range(self.n_resamples):
            idx = np.concatenate([rows_of[m] for m in rng.choice(unique, len(unique))])
            if y[idx].sum() > 0:  # the metric needs at least one positive case
                values.append(metric(y[idx], scores[idx]))
        tail = (1 - self.level) / 2 * 100
        lo, hi = np.percentile(values, [tail, 100 - tail])
        return float(lo), float(hi)


class Evaluator:
    """Computes the evaluation table of one model on one set of predictions."""

    def __init__(self, bootstrap: MonthBlockBootstrap | None = None, top_fraction: float = 0.10):
        self.bootstrap = bootstrap or MonthBlockBootstrap()
        self.top_fraction = top_fraction

    def evaluate(self, y, scores, months, probabilistic: bool = True) -> dict:
        y, scores = np.asarray(y), np.asarray(scores)
        if y.sum() == 0:
            raise ValueError("cannot evaluate a ranking without any positive case")
        recall, precision = top_fraction_hits(y, scores, self.top_fraction)
        lo, hi = self.bootstrap.interval(average_precision_score, y, scores, months)
        return {
            "pr_auc": average_precision_score(y, scores),
            "roc_auc": roc_auc_score(y, scores),
            "recall_top10": recall,
            "precision_top10": precision,
            "brier": brier_score_loss(y, scores) if probabilistic else np.nan,
            "prevalence": y.mean(),
            "pr_auc_lo": lo,
            "pr_auc_hi": hi,
        }
