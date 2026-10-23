| Stage | Time (s) | Peak memory (MB) | Runs |
|---|---|---|---|
| Labels and target (1,593 events) | 0.292 | 2.2 | 3 |
| Spatial join + panel (26 depts x 105 months) | 0.35 | 2.9 | 3 |
| Feature table (31 features, 3,068 rows) | 3.674 | 4.4 | 3 |
| Experiment operational/quarter (selection + evaluation) | 131.164 | 3.2 | 1 |
| Experiment research/month (selection + evaluation) | 157.226 | 4.2 | 1 |
| Dashboard: cold start (first render) | 43.557 | 55.4 | 1 |
| Dashboard: warm rerun (cached data) | 3.838 | 6.8 | 5 |
| Dashboard: switch to French | 5.289 | 8.3 | 1 |

- os: Windows-11-10.0.26300-SP0
- python: 3.13.2
- processor: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- cpu_cores: 4
- logical_cpus: 8
- ram_gb: 17.0
- database_kb: 876.5
- map_geometry_kb: 0.0
