| Stage | Time (s) | Peak process memory, RSS (MB) | Runs |
|---|---|---|---|
| Labels and target (1,593 events) | 0.09 | 193.3 | 3 |
| Spatial join + panel (26 depts x 105 months) | 0.032 | 205.9 | 3 |
| Feature table (31 features, 3,068 rows) | 0.36 | 209.7 | 3 |
| Experiment operational/quarter (selection + evaluation) | 19.542 | 212.7 | 1 |
| Experiment research/month (selection + evaluation) | 17.583 | 214.4 | 1 |
| Dashboard: cold start (first render) | 1.544 | 246.6 | 1 |
| Dashboard: warm rerun (cached data) | 0.129 | 250.0 | 5 |
| Dashboard: switch to French | 0.17 | 252.9 | 1 |

- os: Windows-11-10.0.26300-SP0
- python: 3.13.2
- processor: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- cpu_cores: 4
- logical_cpus: 8
- ram_gb: 17.0
- database_kb: 1060.9
- map_geometry_kb: 50.7
