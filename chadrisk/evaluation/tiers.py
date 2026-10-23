"""Priority levels: what being ranked high has meant in the test period."""

import pandas as pd


class PriorityTiers:
    """Conflict rate by rank band, with departments ranked within each period.

    The dashboard shows two levels (Priority = ranks 1-10, Lower = the rest); the finer
    split (1-5, 6-10) is kept to justify that choice.
    """

    BANDS = (("Top 5", 1, 5), ("Ranks 6-10", 6, 10), ("Priority", 1, 10), ("Lower", 11, 26))

    def __init__(self, bands=BANDS):
        self.bands = bands

    def compute(self, periods: pd.Series, outcomes: pd.Series, scores) -> pd.DataFrame:
        table = pd.DataFrame({"period": periods.to_numpy(), "y": outcomes.to_numpy(),
                              "score": scores})
        table["rank"] = table.groupby("period")["score"].rank(ascending=False, method="first")
        rows = []
        for label, lo, hi in self.bands:
            band = table[table["rank"].between(lo, hi)]
            rows.append({"tier": label, "ranks": f"{lo}-{hi}", "periods": int(len(band)),
                         "with_conflict": int(band["y"].sum()),
                         "conflict_rate": float(band["y"].mean())})
        result = pd.DataFrame(rows)
        result["overall_rate"] = float(table["y"].mean())
        return result
