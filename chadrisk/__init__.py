"""
chadrisk — estimating farmer-herder conflict risk in southern Chad.

Package layout:
    config       paths and project-wide constants
    data         data access and preparation (ACLED, labels, geography, panel, satellite)
    features     feature engineering (one class per feature group)
    models       risk models sharing one interface (logistic, XGBoost, baseline rule)
    evaluation   metrics with bootstrap intervals, rolling-origin selection, priority levels
    experiment   one training/evaluation run (research or operational, month or quarter)
    reporting    report files and figures
    storage      SQLite repository used by the dashboard

Sub-packages are imported explicitly (e.g. ``from chadrisk.storage import RiskRepository``) so
that the dashboard only loads lightweight modules.
"""

__version__ = "1.0.0"
