# Uniform three-density GPU diagnostic — 2026-09-06

Status: PARTIAL / strict aggregate gate FAILED. Do not promote to patient physics.

## Implementation

New explicit key: ct_longitudinal_homogeneous_density_diagnostic: true (default false).
Only smoke, scale=1, 175 MeV/u single beam, nuclear off and existing candidate file.
Entire CCTG is checked at launch: every voxel section=0 and bit-identical density,
one of the three frozen HU=-1000/-975/-951 densities. A single changed density or
section is rejected. No heterogeneous or patient-source route was enabled.
The scaled range is used only in that prevalidated uniform grid. Exact path
segment scoring, bounded energy lookup, and candidate acceptance rejection remain.
Ledger records the selected runtime kernel, explicit flag and kernel density.

## Actual GPU results (20k per case, same seed)

| HU | Original baseline 2–20mm error | Diagnostic ON error | Max 5mm error |
|---|---:|---:|---:|
| -1000 | +3.944% | +0.547% | 0.590% |
| -975 | +1.284% | +0.324% | 0.458% |
| -951 | +0.563% | +0.134% | 0.434% |

All window / 5mm dose-error gates pass; these are actual GPU outputs this turn,
not the preceding offline predictions. Compared to the offline model, maximum
5mm relative differences are 0.0212%, 0.0159%, 0.0173%.
OFF paths reproduce previous candidate raw files byte-for-byte at every density;
reference-density ON also reproduces its old raw exactly.
All candidate quality reports remain accepted=false with the mandatory
unvalidated_longitudinal_candidate marker, not accepted physics.

## Failed gate — retained, not waived

HU=-951 ON has 521 out-of-domain queries and 298.674904 MeV retained locally.
This is 0.000711% of nominal beam energy, 0.00634% of scored voxel energy.
Range/domain clamps were NOT introduced. Despite small dose impact, the
prespecified zero-domain-query gate fails, so report.json status stays FAILED.
Current counters do not localize energy/depth; do not infer a precise cutoff
location from dose curves. Next diagnostic should locate these queries and
establish required energy coverage before extending data, not loosen the gate.

No kernel/scorer invalid marches, no overflow; strict voxel writer completed.
HU=-951 voxel/in-grid=1.00000000001; physical residual=1.14047e-5.
This is a numerical result, not sufficient proof of electron physics.

## Tests / evidence

Local CUDA CTest test_schneider_longitudinal passed (1.96s), including homogeneous
probe/rejection checks. Python diagnostic tests 11/11, v2.1 verifier 16/16 passed.
Real launch tests with one mixed-density voxel and one changed section both
refused as heterogeneous grid, before transport.
Six sequential local GPU runs; no new TOPAS jobs or patient Gamma.

All inputs/configs/logs/raw:
 /mnt/sda/wuwei/longitudinal_uniform_gpu_20260906
report.json retains the failure. validation.json here adds offline comparisons,
selected source hashes and worktree status without changing original outputs.
Reproduce to a NEW directory using tools/run_longitudinal_uniform_gpu.py with
the prior reference/density roots and current binary. No commit or push.

Next: localize the 521 domain queries without affecting dose; then decide on
independent energy-coverage data. Fixed-birth/joint/interface gates remain
incomplete. Do not re-enable heterogeneous source-density-only scaling.
