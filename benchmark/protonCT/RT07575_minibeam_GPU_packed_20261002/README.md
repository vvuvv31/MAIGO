# RT07575 proton minibeam rerun, 2026-10-02

Same 236,086,432 incident primaries (0.1% of the optimized 4 Gy plan), 487 active
spots, 119 source shards and seeds as `../RT07575_minibeam_GPU_0p1pct`.
The frozen current executable uses FP32 dose scoring; FP64 is disabled.
Electron grouping remains 32. The historical YAML files differ only by the
additional `material_electron_packed_directory` key.

The extended RT07575 response index contains 82 tables. `electron_packed/`
references 63 already validated default proton artifact tables and stores the
other 19 tables generated for this run. `prepare.json`, `artifact_check.json`
and `input_check.json` record preparation, raw-reference checks and input pins.
Preparation time therefore does not represent rebuilding all 82 tables.

From this directory:

```bash
python3 run_gpu.py --group 119
python3 progress.py
# After the runner and its TOPAS-reference analysis have finished:
python3 compare_rerun.py
```

`run_gpu.py` launches all 119 YAML configurations in one `--plan-manifest`
process. It validates every output and aggregates raw dose using only the
original incident-primary normalization. Successful group records allow the
Python runner to skip an already accepted full group; the C++ manifest itself
does not provide partial-group resume. Do not launch a second runner while the
first is active. Preserve this directory; use a fresh directory for new trials.

`group000.log` and `group000_accepted.json` hold process timing and quality
records. Monte Carlo timing includes setup, transport, per-batch outputs and
exit; it excludes artifact generation and postprocessing. `completion.json`
includes runner postprocessing and is intentionally a different timing metric.
`gpu_telemetry.csv` samples device utilization, memory, temperature and clocks
every five seconds. No OS page-cache eviction was performed.

`REPORT_zh.md` and `results/rerun_comparison.json` compare this run against the
historical GPU run and the same fixed TOPAS weighted-DIJ reference. No new TOPAS
transport, dose fitting, registration or source reweighting is performed.
Research physics limitations and gamma interpolation limitations remain.
