"""
Build the target variable (farmer-herder conflicts) from the manual classification
of ACLED events, and compare it with keyword filtering.

Labels (data/processed/manual_labels.csv), see docs/labeling_codebook.md:
    FH  = direct violence between farmers and herders (model target)
    AMB = communal / pastoral violence without explicit farmer-herder framing
    REL = related but non-violent event (protest, arrests, reconciliation)
    NO  = everything else (default)

Usage: python scripts/build_target.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "acled_chad_2015_2025.csv"
LABELS = ROOT / "data" / "processed" / "manual_labels.csv"
OUT = ROOT / "data" / "processed" / "acled_chad_labeled.csv"
VALIDATION = ROOT / "data" / "processed" / "validation_sample.csv"

SOUTH = [
    "Logone Occidental", "Logone Oriental", "Mandoul", "Moyen-Chari",
    "Tandjile", "Mayo Kebbi Est", "Mayo Kebbi Ouest",
]
KEYWORDS = (
    r"herder|herdsm|pastoral|cattle|livestock|transhuman|nomad|grazing|"
    r"farmer|crop|field|farm|camel|cow|oxen|herd"
)

df = pd.read_csv(RAW)
labels = pd.read_csv(LABELS)
assert labels["event_id_cnty"].is_unique, "Duplicate IDs in manual_labels.csv"

df = df.merge(labels, on="event_id_cnty", how="left")
df["label"] = df["label"].fillna("NO")
df["event_date"] = pd.to_datetime(df["event_date"])
df["month"] = df["event_date"].dt.month
df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
df["admin1_norm"] = (
    df["admin1"].str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
)
df["is_south"] = df["admin1_norm"].isin(SOUTH)
df["keyword_hit"] = df["notes"].str.contains(KEYWORDS, case=False, na=False)
df["target_fh"] = (df["label"] == "FH").astype(int)
# Broad target for the sensitivity analysis
df["target_broad"] = df["label"].isin(["FH", "AMB"]).astype(int)

df.to_csv(OUT, index=False)

print("=" * 70)
print("LABEL DISTRIBUTION")
print(df["label"].value_counts().to_string())

fh = df[df["target_fh"] == 1]
print("\n" + "=" * 70)
print(f"CONFIRMED FARMER-HERDER CONFLICTS: {len(fh)}")
print(f"  of which south: {fh['is_south'].sum()} ({fh['is_south'].mean():.0%})")
print(f"  total fatalities: {fh['fatalities'].sum():.0f} | "
      f"south: {fh.loc[fh['is_south'], 'fatalities'].sum():.0f}")

print("\nBy province:")
print(fh.groupby("admin1_norm").agg(events=("event_id_cnty", "size"),
                                     fatalities=("fatalities", "sum"))
      .sort_values("events", ascending=False).to_string())

print("\nBy year (south / rest):")
print(fh.groupby(["year", "is_south"]).size().unstack(fill_value=0)
      .rename(columns={True: "south", False: "rest"}).to_string())

print("\nSeasonality in the south (by month):")
print(fh[fh["is_south"]].groupby("month").size().reindex(range(1, 13), fill_value=0).to_string())

print("\nMost affected southern departments (admin2):")
print(fh[fh["is_south"]]["admin2"].value_counts().head(15).to_string())

print("\n" + "=" * 70)
print("COMPARISON WITH KEYWORD FILTERING")
tp = ((df["keyword_hit"]) & (df["target_fh"] == 1)).sum()
precision = tp / df["keyword_hit"].sum()
recall = tp / df["target_fh"].sum()
print(f"Keyword precision: {precision:.0%} | Recall: {recall:.0%}")

# Stratified sample for blind manual validation by the author.
# Never overwrite a sample that has already been annotated.
if VALIDATION.exists():
    print(f"\n{VALIDATION.name} already exists: not regenerated (annotations preserved).")
else:
    parts = [
        df[df["label"] == "FH"].sample(30, random_state=1),
        df[df["label"] == "AMB"].sample(min(15, (df["label"] == "AMB").sum()), random_state=1),
        df[(df["label"] == "NO") & df["keyword_hit"]].sample(15, random_state=1),
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=1)
    sample = sample[["event_id_cnty", "event_date", "admin1", "notes"]].assign(author_label="")
    sample.to_csv(VALIDATION, index=False)
    print(f"\nValidation sample (60 events, labels hidden): {VALIDATION}")
