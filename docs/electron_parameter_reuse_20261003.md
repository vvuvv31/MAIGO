# FP32 electron same-step parameter reuse

RT07575, RTX 2080 Ti, group32, primary chunk262144, electron lanes16384,
FP32 scoring throughout. This experiment starts from the frozen full-FP32
baseline, not the earlier double-precision transport.

The charged step already computes material parameters at the pre-step energy.
The loss sampler now accepts this result instead of repeating the density-node
parameter/rate searches and interpolation. All terminal conditions, hard-rate
majorant checks, physics sampling and random-number consumption are retained.
There is no cross-step cache and no step-length/energy-loss-cap change.

Generated implementation: `scripts/generate_electron_fp32.py` owns the prepared
loss helper in `include/carbon/electron_fp32.hpp`. The reference equations remain
available. `CARBON_ELECTRON_REUSE_PARAMETERS` selects the candidate in the FP32
worker; `CARBON_ELECTRON_STEP_DIAGNOSTICS` is OFF for all throughput runs.

## Matched ABBA result

Each process runs the same two 1,983,920-primary shards. ABBA order is frozen
FP32 reference A, candidate A, candidate B, reference B. Repeated timing runs
reuse the same seeds: there are 3,967,840 distinct primaries per arm.

| Measure | FP32 reference | Parameter reuse |
|---|---:|---:|
| Resident second batch, mean |22.897071 s|21.149436 s|
| Resident throughput |86,645 primary/s|93,805 primary/s|
| Electron stage, mean per batch |12.854692 s|11.020329 s|
| Direct worker registers |242|241|
| Direct worker static stack |1280 B|1280 B|

Resident throughput increases **8.263%**, with both pairs positive
(6.993% and 9.527%). Electron time decreases **14.270%**. The static instruction
count falls 14,784→13,864; static stack/local instructions barely change.
These resource counts do not establish dynamic spill reductions.

All eight batches pass research-quality, primary/EM, queue-completion and
energy checks, with zero overflow/coverage gaps/invalid states. All integer
electron/joint/photon work counts match the reference. Maximum relative energy
residual is 4.942e-8. BODY dose relative L1≤5.956e-8, maximum pooled peak/valley
change≤6.027e-7%, and maximum local change at the 1%-BODY-maximum threshold
≤0.0001122%. This checks implementation equivalence; it is not new TOPAS
validation or resolution of the existing FP32 Urban threshold issue.

The isolated GPU component check covers 82 native nodes plus 57 intermediate
densities, 1,138,688 loss samples including native termination, zero path,
range termination and ordinary small steps. Every loss field and the final
RNG state are identical.

## Evidence

- `out/electron_parameter_reuse_20261003/summary.json`
- `out/electron_parameter_reuse_20261003/screen/report.json`
- `out/electron_parameter_reuse_20261003/check_loss.json`
- `out/electron_parameter_reuse_20261003/source/fingerprint.json`
- `out/electron_parameter_reuse_20261003/candidate_resource.json`
- `out/electron_diagnostics_20261003/histogram_report.json`

Candidate binary SHA256:
`b3039e0d24a725d54ca762d0634f7f0fd31e046277c08dccd7f5281391e7b4b9`.
The immutable original full-FP32 baseline remains under
`out/electron_fp32_20261003/full_carbon_mc`.

Default lung CT/170 MeV regression also passes 2×500,000 primaries, with
exact electron work versus the frozen baseline, cache reuse, completion
4,221,147 / 4,234,724, zero overflow, and energy residual≤3.646e-7.
The build preset now enables parameter reuse and explicitly disables diagnostic
instrumentation. `build/carbon_mc` is the accepted candidate binary above.
