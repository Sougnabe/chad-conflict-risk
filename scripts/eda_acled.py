"""
Exploratory analysis of ACLED data for Chad (2015-2025), using the validated labels.
Goal: document the concentration, trend and seasonality of farmer-herder conflict
in the south (figures for the report).

Usage: python scripts/build_target.py, then python scripts/eda_acled.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "acled_chad_labeled.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["event_date"])
fh = df[df["target_fh"] == 1]

print("Events per year (all ACLED events, Chad):")
print(df.groupby("year").size().to_string())
print("\nEvent types:")
print(df["event_type"].value_counts().to_string())
print("\nGeographic precision (1-2 = precise, 3 = province level):")
print(df["geo_precision"].value_counts().sort_index().to_string())

by_prov = (
    fh.groupby("admin1_norm")
    .agg(events=("event_id_cnty", "size"), fatalities=("fatalities", "sum"),
         south=("is_south", "first"))
    .sort_values("events", ascending=False)
)
season = fh[fh["is_south"]].groupby("month").size().reindex(range(1, 13), fill_value=0)
yearly = (fh.groupby(["year", "is_south"]).size().unstack(fill_value=0)
          .rename(columns={True: "South", False: "Rest of Chad"}))

# Figure 1: events by province
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0392b" if s else "#95a5a6" for s in by_prov["south"]]
ax.barh(by_prov.index[::-1], by_prov["events"][::-1], color=colors[::-1])
ax.set_xlabel("Number of events")
ax.set_title("Farmer-herder conflict events by province, Chad 2015-2025\n"
             "(red = southern provinces; source: ACLED, manually validated)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fh_by_province.png", dpi=150)
plt.close()

# Figure 2: seasonality in the south
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax.bar(labels, season.values, color="#c0392b")
ax.set_ylabel("Number of events")
ax.set_title("Seasonality of farmer-herder conflict in southern Chad (2015-2025)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fh_south_seasonality.png", dpi=150)
plt.close()

# Figure 3: yearly trend
ax = yearly.plot(kind="bar", stacked=True, color=["#95a5a6", "#c0392b"], figsize=(8, 4))
ax.set_ylabel("Number of events")
ax.set_xlabel("")
ax.set_title("Farmer-herder conflict events per year, Chad")
plt.tight_layout()
plt.savefig(FIG_DIR / "fh_yearly_trend.png", dpi=150)
plt.close()

# Figure 4: map of events
fig, ax = plt.subplots(figsize=(6, 7))
other = df[df["target_fh"] == 0]
ax.scatter(other["longitude"], other["latitude"], s=4, c="#bdc3c7", label="Other events")
ax.scatter(fh["longitude"], fh["latitude"], s=10, c="#c0392b", label="Farmer-herder")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("ACLED events in Chad, 2015-2025")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "map_events.png", dpi=150)
plt.close()

print(f"\nFigures saved to {FIG_DIR}")
