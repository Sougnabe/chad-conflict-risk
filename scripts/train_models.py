"""
Step 8 — Train, evaluate and predict for one configuration.

Usage:
    python scripts/train_models.py --mode research --unit quarter
    python scripts/train_models.py --mode research --unit month
    python scripts/train_models.py --mode operational --unit quarter --fixed-model logreg_OB
    python scripts/train_models.py --mode operational --unit month

See chadrisk/experiment.py for the protocol, chadrisk/reporting.py for the outputs.
"""

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from chadrisk.config import PATHS
from chadrisk.experiment import MODES, UNITS, Experiment, ExperimentConfig
from chadrisk.reporting import ExperimentReporter
from chadrisk.storage import RiskRepository

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=MODES, default="research")
parser.add_argument("--unit", choices=UNITS, default="quarter")
parser.add_argument("--fixed-model", default=None,
                    help="model used for the predictions instead of the automatic selection "
                         "(design choice, documented in the model card)")
args = parser.parse_args()

config = ExperimentConfig(mode=args.mode, unit=args.unit, fixed_model=args.fixed_model)
experiment = Experiment(config, pd.read_csv(PATHS.features))
print(f"{config.tag}: selection 2019-2023 = {len(experiment.selection_data)} rows "
      f"({experiment.selection_data['y_fh'].sum()} positives) | test 2024-2025 = "
      f"{len(experiment.test)} rows ({experiment.test['y_fh'].sum()} positives)")

result = experiment.run()
ExperimentReporter(PATHS, RiskRepository(PATHS.database)).save(result)

pd.set_option("display.width", 200)
print(result.metrics[["model", "val_pr_auc", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc",
                      "recall_top10"]].round(3).to_string(index=False))
print(f"Automatic selection: {result.auto_selected} | used for predictions: {result.used}")
print(result.tiers[["tier", "ranks", "conflict_rate"]].round(3).to_string(index=False))
print(f"Forecast saved for {result.forecast['ym'].iloc[0]}")
