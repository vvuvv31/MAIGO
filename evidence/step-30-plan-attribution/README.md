# Step 30 plan-attribution evidence (frozen)

Question: is the RT06423 +3-6% primary-origin bias a TPS source /
aggregation artifact or nuclear final-state physics?

- `manifest.json`: SHA256 + bytes + absolute paths for executables,
  configs, spots/beam-model/CT, TOPAS params, v2.1 package/bundle,
  dose artifacts, analysis script, key numbers; plus Slurm job IDs,
  tasks, resources, logs. Large raw doses are NOT duplicated here;
  they stay at their recorded absolute paths with SHAs.
- `key_numbers.json`: reproduced by `tools/analyze_plan_attribution.py`
  (all gates pass at freeze time).

Headline results:
- Source truth (9 spots, vacuum planes): counts exact, moments
  consistent (global chi2/dof 0.61); low_center 5x rerun all-pass.
- 5 energy groups: totals T/G 0.982-0.987; GPU superposition exact
  (1.000000); air T/G 0.31 -> 0.12 (GPU 3x -> 8x with energy).
- Uniform air slab: EM 1.0003, nuclear GPU +21% (halo 7.4 vs 4.2 mm),
  identical interaction rates (3.20% vs 3.24%).
- Sandwich: tisA 1.008 / air 0.22 / tisB 1.13, totals 0.991.

Interpretation: charged-secondary angular spectra (C12 primary package
is FTFP_INCLXX; TOPAS reference final state is BIC). Package frozen;
no transport-code change. Step 30 stays IN_PROGRESS.
