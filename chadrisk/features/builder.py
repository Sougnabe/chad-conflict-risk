"""Assembles the feature groups into one department x month table."""

import pandas as pd

from chadrisk.features.base import COUNT_COLUMNS, FeatureContext, FeatureGroup


class FeatureTableBuilder:
    """Builds the feature table from a panel and a list of feature groups.

    Rows are added after the end of the conflict data, up to `horizon_end`, so that forecasts
    can be made; their counts and targets stay unknown (NaN).
    """

    def __init__(self, groups: list[FeatureGroup], neighbours: dict[str, list[str]],
                 same_province: dict[str, list[str]]):
        self.groups = groups
        self.neighbours = neighbours
        self.same_province = same_province

    def build(self, panel: pd.DataFrame, horizon_end: pd.Period, log=print) -> pd.DataFrame:
        panel = panel.copy()
        panel["ym"] = pd.PeriodIndex(panel["ym"], freq="M")
        data_end = panel["ym"].max()
        panel = self._add_future_rows(panel, data_end, horizon_end)
        context = FeatureContext(panel=panel, counts=self._wide_counts(panel, data_end),
                                 neighbours=self.neighbours, same_province=self.same_province,
                                 data_end=data_end)
        table = panel
        for group in self.groups:
            values = group.compute(context).reset_index()
            table = table.merge(values, on=["ym", "dept_id"], how="left")
            log(f"  + {group.name}: {values.shape[1] - 2} features")
        table["ym"] = table["ym"].astype(str)
        return table

    @staticmethod
    def _add_future_rows(panel, data_end, horizon_end):
        departments = panel[["dept_id", "department", "province"]].drop_duplicates()
        future = [departments.assign(ym=m)
                  for m in pd.period_range(data_end + 1, horizon_end, freq="M")]
        panel = pd.concat([panel, *future], ignore_index=True)
        panel["year"] = panel["ym"].dt.year
        panel["month"] = panel["ym"].dt.month
        return panel.sort_values(["dept_id", "ym"]).reset_index(drop=True)

    @staticmethod
    def _wide_counts(panel, data_end):
        """Month x department count tables: zero when no event, unknown after data_end."""
        counts = {}
        for c in COUNT_COLUMNS:
            wide = panel.pivot(index="ym", columns="dept_id", values=c)
            wide.loc[wide.index <= data_end] = wide.loc[wide.index <= data_end].fillna(0)
            counts[c] = wide
        return counts
