"""
Step 2 — Apply the validated labels and build the target variable.

Usage: python scripts/build_target.py
"""

import _bootstrap  # noqa: F401

from chadrisk.config import PATHS
from chadrisk.data.labels import LabelledEvents

events = LabelledEvents.from_files(PATHS.acled_raw, PATHS.manual_labels).build()
events.to_csv(PATHS.labelled_events, index=False)

fh = events[events["target_fh"] == 1]
print("Label distribution:")
print(events["label"].value_counts().to_string())
print(f"\nConfirmed farmer-herder conflicts: {len(fh)} | south: {fh['is_south'].sum()} "
      f"({fh['is_south'].mean():.0%}) | fatalities: {fh['fatalities'].sum():.0f}")
print("\nBy province:")
print(fh.groupby("admin1_norm").agg(events=("event_id_cnty", "size"),
                                     fatalities=("fatalities", "sum"))
      .sort_values("events", ascending=False).to_string())

precision, recall = LabelledEvents.keyword_agreement(events)
print(f"\nKeyword filter vs manual labels: precision {precision:.0%}, recall {recall:.0%}")

# Blind validation sample: never overwrite one that has already been annotated
if PATHS.validation_sample.exists():
    print(f"\n{PATHS.validation_sample.name} already exists (annotations preserved).")
else:
    LabelledEvents.validation_sample(events).to_csv(PATHS.validation_sample, index=False)
    print(f"\nValidation sample written: {PATHS.validation_sample}")
