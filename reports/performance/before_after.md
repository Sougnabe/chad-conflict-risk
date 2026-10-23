Same benchmark (scripts/benchmark.py), same machine, before = commit f201fc6 (after the
refactor, before the performance work), after = current version.

| Stage | Before (s) | After (s) | Speed-up | Before RSS (MB) | After RSS (MB) |
|---|---|---|---|---|---|
| Labels and target (1,593 events) | 0.09 | 0.09 | x1.0 | 194.8 | 193.3 |
| Spatial join + panel (26 depts x 105 months) | 0.03 | 0.032 | x0.9 | 206.7 | 205.9 |
| Feature table (31 features, 3,068 rows) | 0.356 | 0.36 | x1.0 | 210.2 | 209.7 |
| Experiment operational/quarter (selection + evaluation) | 25.532 | 19.542 | x1.3 | 216.3 | 212.7 |
| Experiment research/month (selection + evaluation) | 82.48 | 17.583 | x4.7 | 219.5 | 214.4 |
| Dashboard: cold start (first render) | 7.771 | 1.544 | x5.0 | 293.4 | 246.6 |
| Dashboard: warm rerun (cached data) | 1.199 | 0.129 | x9.3 | 305.4 | 250.0 |
| Dashboard: switch to French | 1.396 | 0.17 | x8.2 | 307.7 | 252.9 |

Changes between the two versions: parallel rolling-origin selection (joblib) with
single-threaded deterministic XGBoost; simplified map outlines (140 KB -> 51 KB);
outlines parsed once and cached; only the selected dashboard page is rendered.
