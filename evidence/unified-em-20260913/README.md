# Unified all-ion EM integration, 2026-09-13

Research candidate; no production promotion. GPU runs use the local RTX 2080 Ti.
TOPAS reference doses are the unchanged `benchmark/benchmark20260913` references.
CPU extraction/probe runs use `v@10.10.10.216`, one thread per sequential export.

## Initial 77-density-point package

Package SHA256: `672c5017b32cda0efd5a1a83635ca81f1caaa6074a09869691c387ca40b1f933`.
This initial grid is superseded by density refinement. These results are preserved
as an implementation baseline, not as evidence for a different package hash.

- Native lookup: 1,386 material/ion records, 308,000 queries, pass.
- Actual Geant4 finite-step mean: 41,580 queries each on host and local GPU,
  zero failures, maximum relative error 0.00849652%.
- Native fluctuation distribution: 432 material/ion/energy combinations, zero
  failures for first and second raw moments. Geant4 uses 50k independent samples
  per combination; the implementation uses 100k. The family-wide statistical
  guard is seven combined standard errors plus 0.1% interpolation allowance.
- Delta rejection sampling: 162 acceptance/first/second-moment checks against
  independent quadrature, zero failures.
- Seven 50k full transport runs: accepted research quality, zero queue overflow,
  zero unified-EM query/sampling failures. Refer to `transport_summary.csv`.
- Density diagnostic: withholding a middle density node exposed up to 4.92%
  mean-loss and 3.68% range errors in the sparse grid. Near-threshold relative
  rate error is ill-conditioned when the reference is zero; use an absolute
  rate allowance in addition to relative error. This diagnostic triggered
  denser material extraction rather than production promotion.

IDD numbers compare GPU 50k to TOPAS 200k after normalization per primary.
They do not establish a same-statistics Gamma or full-curve acceptance gate.
The b3/b4 benchmark material densities are explicit nodes, so those successful
runs do not validate interpolation across arbitrary patient densities.

Full inputs, output doses, quality JSON and logs are in
`scratch/unified_joint_em_20260913/transport`. The accompanying scripts retain
quality/overflow guards and normalize the two particle counts independently.

## Refined 175-density-point package

Final research package SHA256:
`8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`.
Size: 1,386,682,652 bytes; 3,150 material/ion records. Detailed checks are in `dense/`.

- Native lookup: 700,000 queries, pass.
- CPU and GPU finite-step mean: 94,500 each, zero failures, maximum relative
  difference 0.0130385%.
- Fluctuation: 432 combinations against native samples at exactly matched material
  densities, pass. Delta: 162 checks, pass.
- Density leave-one-out: 15,750 queries; maximum mean-loss difference 0.787%,
  range difference 0.669%. This is a diagnostic, **not** a passed production gate.
  The maximum occurs in section 0 near the production-cut floor transition.
  The rate onset also remains unresolved by this interpolation diagnostic.

Final-package transport: all seven 50k cases passed with zero overflow and zero
unified EM failures. Final IDD comparisons, plots, exact configs and quality
reports are in `dense/`. These use the unchanged 200k TOPAS references,
normalized per primary; they are not a patient Gamma promotion gate.

## Remaining production gates

Independent density/interpolation acceptance, all-ion 50k closure in additional
materials, patient one-shard BODY-only Gamma with search step DTA/10, and a full
zero-overflow patient comparison remain required. Defaults are unchanged.
