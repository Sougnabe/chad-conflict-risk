"""Conflict-history features with a configurable information delay."""

import numpy as np
import pandas as pd

from chadrisk.features.base import FeatureContext, FeatureGroup


def rolling_sum(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    return wide.rolling(window, min_periods=1).sum()


def months_since_positive(values: np.ndarray, cap: int = 36) -> np.ndarray:
    """Months elapsed since the last positive value, capped (vectorised, O(n)).

    Missing values count as zero. Before any positive value the result is `cap`.
    Example: [0, 2, 0, 0, 1, 0] -> [cap, 0, 1, 2, 0, 1].
    """
    values = np.nan_to_num(np.asarray(values, dtype=float))
    idx = np.arange(len(values), dtype=float)
    last = pd.Series(np.where(values > 0, idx, np.nan)).ffill().to_numpy()
    since = idx - last
    return np.where(np.isnan(since), cap, np.minimum(since, cap))


class ConflictHistoryFeatures(FeatureGroup):
    """Conflict history known `shift` months before the predicted month.

    shift=1  (prefix h_): research model, conflicts up to the previous month.
    shift=13 (prefix o_): operational model; ACLED publishes with a 12-month delay, so at the
             start of month t the latest known conflict month is t-13.
    """

    def __init__(self, shift: int, prefix: str, short_windows: bool):
        if shift < 1:
            raise ValueError("shift must be at least 1 month (no look-ahead)")
        self.shift = shift
        self.prefix = prefix
        self.short_windows = short_windows

    @property
    def name(self) -> str:
        return f"conflict history up to t-{self.shift} ({self.prefix}*)"

    def compute(self, context: FeatureContext) -> pd.DataFrame:
        past = {c: w.shift(self.shift) for c, w in context.counts.items()}
        fh, p = past["n_fh"], self.prefix
        fh12 = rolling_sum(fh, 12)
        since = pd.DataFrame({d: months_since_positive(fh[d].to_numpy()) for d in fh.columns},
                             index=fh.index)
        columns = [
            self._long(fh12, f"{p}fh_12m"),
            self._long(fh.cumsum(), f"{p}fh_cum"),
            self._long(rolling_sum(past["n_broad"], 12), f"{p}broad_12m"),
            self._long(rolling_sum(past["n_all_events"], 12), f"{p}allev_12m"),
            self._long(rolling_sum(past["fatalities_fh"], 12), f"{p}fatal_12m"),
            self._long(self._sum_over(fh12, context.neighbours), f"{p}nb_fh_12m"),
            self._long(self._sum_over(fh12, context.same_province), f"{p}prov_fh_12m"),
            self._long(since, f"{p}months_since_fh"),
        ]
        if self.short_windows:
            columns += [
                self._long(fh, f"{p}fh_1m"),
                self._long(rolling_sum(fh, 3), f"{p}fh_3m"),
                self._long(self._sum_over(rolling_sum(fh, 3), context.neighbours),
                           f"{p}nb_fh_3m"),
            ]
        out = pd.concat(columns, axis=1)
        # Only defined while the lagged window lies inside the conflict data
        out.loc[out.index.get_level_values("ym") > context.data_end + self.shift] = np.nan
        return out

    @staticmethod
    def _sum_over(wide: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
        """For each department, the sum over its group (neighbours or same province)."""
        return pd.DataFrame({d: wide[groups[d]].sum(axis=1) for d in wide.columns})
