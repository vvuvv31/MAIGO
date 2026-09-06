# Longitudinal candidate review — 2026-09-06

Status: numerical safety checks passed; independent physics validation BLOCKED.

The implementation is diagnostic-only (smoke, scale=1); every enabled run fails
physical acceptance explicitly. The original CSV and metadata remain unchanged,
and the validated v2.1 stack is not upgraded.

Implemented: pinned provenance, bounded lookup, homogeneous reference-density
scope, exact voxel-face path shares, domain counters, and real CUDA host/device tests.
An unsupported interface retains remaining energy at source; this is a limitation,
not a physical interface solution. No patient-specific scale is allowed.

## Evidence and limitations

See validation.json for source/build hashes, 32 artifact records and seven small GPU
runs. Artifacts live under /mnt/sda/wuwei/schneider_longitudinal_review_20260906_tuvUBr.
The final rebuild differs from the smoke binary by a source comment only; both hashes
are recorded, not conflated.

- CUDA test passed on the final build; Python 8/8 and manifest 16/16 passed.
- 120 MeV/u controlled candidate on/off raw files are byte-identical.
- 200 MeV/u controlled on/off both fail the writer at bin 1: diff 0.008273 MeV.
  Baseline quality accepted=true does NOT imply a successful complete output.
- Initial controls also share a physical residual failure (~1.09346e-4 > 1e-4).
- No thresholds were loosened. Full legacy CTest was not run.
- No new TOPAS, patient Gamma, commit or push.

Historical three-case scales 0.63/0.75/0.90 are calibration evidence, not independent
validation. RT06423 G30 decreased about 0.028 percentage points.
Before patient use: resolve the shared writer failure, then independently validate
energy coverage, density/interface response and longitudinal/transverse correlation.
