# Proton RT07575 head minibeam upstream throughput screens, 2026-10-10

No new throughput optimization was retained. Three isolated, unchanged-physics candidates were tested on RTX A6000 (`v@172.17.152.19`) using the completed 0.5 mm patient-cap research configuration. Keep the prior 0.5 mm result (236,086,432 histories, TOPAS local3%/0.2 mm/search0.02 mm = 98.941577%, warm 304,240/s) as the high-stat dose evidence. These screens are matched GPU implementation/performance checks, not fresh TOPAS dose acceptance.

## Measured results

Each process runs two original whole-plan batches of 9,836,935 histories using one context/packages. The first batch warms the context; the second is timed. Each batch covers all 487 active spots with original optics, weights, CT, placement, collimator and per-spot seeds. Each candidate repeats exactly the same two source CSVs; repeated seeds are not independent dose samples.

| Experiment | Warm control /s | Warm candidate /s | Change | Decision |
|---|---:|---:|---:|---|
| Chunk 262144→524288 | 312,089 | 309,383 | -0.867% | Not retained |
| Chunk 262144→1048576 | 312,089 | 312,634 | +0.175% | Not retained |
| Per-history air integer audit aggregation | 299,776 | 303,499 | +1.242% | Not retained |
| Air ion rate reuse until existing threshold changes | 304,870 | 304,292 | -0.190% | Not retained |

Chunk sequence A/B/C/C/B/A (262144/524288/1048576) shows no useful total benefit. The 262144 control drifts from 323456/s initially to 301494/s finally; small pooled differences cannot be interpreted as precise gains. Larger chunks reduce patient-primary timing while upstream timing rises; no hardware occupancy/stall measurement was made. Keep 262144.

Both code candidates use A/B/B/A timing, totaling 39,347,740 warm timed histories per experiment across both arms. Air aggregation replaces per-step global integer audit additions with per-history totals, reducing write frequency without changing counters. Its +1.242% pooled total change is small, and upstream improvement is only about 0.44%; two pair ratios are approximately 1.005 and 1.020. This does not establish a useful stable gain; do not retain the extra state. Air-rate reuse recomputes the identical FP32 table value only when the original threshold changes. Its pooled -0.190% and mixed pairs give no resolved benefit; do not retain it. Do not infer an atomic, memory or register bottleneck solely from static source or lookup/write counts.

GPU clocks were unlocked; runs were sequential but each candidate includes reversed order. No concurrent GPU jobs. Candidate builds happened locally, not on the benchmark machine.

## Physics and result checks

Patient CT max step 0.5 mm; Cu 0.05 mm; slit-air 0.2 mm; native clock; all primary/secondary/Cu primary/fragment energy fluctuations ON; explicit electrons OFF in the existing isolated research exception. Continuous26 packages, Urban, nuclear reactions, energy sinks and FP32 dose atomics remain unchanged. Source optics remain unrestricted by the optimization; no assumption of axial or zero-emittance source is added.

All 28 saved batches across three experiments pass quality/energy/native-EM/coverage/completion; zero overflow and all Cu child tasks completed. Air clock, Cu cascade, Cu child, unified EM and discrete nuclear counts exactly match each arm's baseline. Live ion species terminal-reason counts and ledger layouts match; FP32 species energy-ledger differences are below relative L1 1e-5. Maximum batch dose relative L1 differences are 2.2783e-7 (chunks), 1.4852e-7 (aggregation), and 1.4898e-7 (rate reuse). Full-plan normalization remains the original campaign scale; no repeat-seed dose pooling or batch multiplication.

The initial ledger-check script mistakenly required whole `schneider_diagnostics` equality, including FP32 energy totals. It stopped at ordinary atomic summation differences. The corrected check explicitly separates integer work counters from floating energy ledgers; all discrete counters match and ledger/dose checks pass. JSON integer serialization of an energy value is not evidence that it is an integer counter.

## Repository state and artifacts

All runtime changes were confined to the private `out/proton_minibeam_opt_20261010/source` build. Its original beamline file and original static library were restored and verified byte-for-byte. Production source/defaults remain untouched. Candidate source snapshots/binaries are evidence under `out/`; no inactive switches were added to the release runtime.

- `out/proton_minibeam_upstream_20261010/`: chunk `summary.json`, `ledger_comparison.json`, `provenance.json`, candidate source snapshots and build logs.
- `out/proton_minibeam_air_aggregate_20261010/`: ABBA `summary.json`, `ledger_comparison.json`, per-arm reports/raw doses/audits.
- `out/proton_minibeam_air_rate_reuse_20261010/`: same evidence for rate reuse.

Raw remote runs mirror these directory names under `/home/v/MAIGO_clean_a6000_20261010/`; all completed. Benchmark GPU is idle. Future work should materially reduce upstream transport cost, or measure hot kernels with available hardware tools, rather than repeat these unchanged chunk/count/rate-reuse trials. Changing copper step or source-response reuse would need a separate physics/statistical validation.
