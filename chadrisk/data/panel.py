"""Department x month panel: the unit of analysis of the models."""

import pandas as pd

COUNT_COLUMNS = ("n_fh", "n_broad", "fatalities_fh", "n_all_events")


class PanelBuilder:
    """Aggregates labelled events into a complete department x month table.

    Every department appears in every month, with zero counts when nothing was recorded.
    """

    def __init__(self, departments: pd.DataFrame, start: str, end: str):
        if pd.Period(start, freq="M") > pd.Period(end, freq="M"):
            raise ValueError("start must not be after end")
        self.departments = departments[["dept_id", "department", "province"]]
        self.months = pd.period_range(start, end, freq="M")

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """`events` must contain dept_id (see Geography.assign_events) and the targets."""
        ev = events[events["dept_id"].isin(self.departments["dept_id"])].copy()
        ev["ym"] = pd.to_datetime(ev["event_date"]).dt.to_period("M")
        ev["fatalities_fh"] = ev["fatalities"].where(ev["target_fh"] == 1, 0)
        counts = (ev.groupby(["dept_id", "ym"])
                  .agg(n_fh=("target_fh", "sum"), n_broad=("target_broad", "sum"),
                       fatalities_fh=("fatalities_fh", "sum"),
                       n_all_events=("event_id_cnty", "size"))
                  .reset_index())

        grid = pd.MultiIndex.from_product([self.departments["dept_id"], self.months],
                                          names=["dept_id", "ym"]).to_frame(index=False)
        panel = grid.merge(self.departments, on="dept_id")
        panel = panel.merge(counts, on=["dept_id", "ym"], how="left")
        panel = panel.fillna({c: 0 for c in COUNT_COLUMNS})
        panel["y_fh"] = (panel["n_fh"] > 0).astype(int)
        panel["y_broad"] = (panel["n_broad"] > 0).astype(int)
        panel["year"] = panel["ym"].dt.year
        panel["month"] = panel["ym"].dt.month
        panel["ym"] = panel["ym"].astype(str)
        return panel
