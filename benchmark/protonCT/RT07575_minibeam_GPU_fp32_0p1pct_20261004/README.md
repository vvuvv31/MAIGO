# RT07575 current FP32: 0.1% full plan

236,086,432 primary; 487 spots; 119 batches in one process; group32,
chunk262144, lanes16384. Same source files and seeds as the historical0.1% plan.
The frozen current executable is in `inputs/carbon_mc`; identity in `build.json`.
No elastic-batch experimental transport and no legacy electron-bank fallback.

- `campaign_status.json`: current transport status and estimated remaining time.
- `gpu.log`, `runner.log`, `batch_checks.json`: progress and per-batch audits.
- `results/dose_4Gy.mhd` and `.raw`: summed patient-coordinate dose, scaled only
  by the prescribed plan primary count / simulated primary count.
- `results/timing.json`: process, resident and electron-stage times.
- `results/gamma.json`: comparison against the existing TOPAS weighted DIJ.
- `REPORT_zh.md`: final report, generated after transport and postprocessing.

`run_current.py` validates each completed batch and stops on failure.
`finish_when_ready.py` runs the postprocessor only after all119 batches complete.
No new TOPAS simulation is launched; historical reference statistics are retained.
