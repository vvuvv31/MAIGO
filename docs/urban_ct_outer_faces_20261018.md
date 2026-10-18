# CT outer-face ownership repair (2026-10-18)

## Failure and cause

The first G4 twelve-case run rejected 20022516 Lung 1 carbon minibeam in its
actual 180-degree field with 1,226 secondary Urban preparation failures
(`unified-em-failure reason=100`). The actual 0-degree field completed. The
90/270 file labels in this fixture are driver identifiers, not the beam angles.
The full-plan spot allocation, source and collimator came from the optimized
TOPAS replay campaign, including the 0/180-degree geometry.

`urban_voxel_face()` reconstructed the outer faces from the FP32 rounded grid
centre, while the CT-box containment test used its explicit lower/upper bounds.
For the failing field's X axis, lower bound `float(-170.3395)` with 960 bins of
0.5 mm produced a reconstructed lower face 1.52587890625e-5 mm inside the box.
A particle exactly on the lower CT bound and travelling inward passed box
containment but failed the first voxel's face test. The correcting loop moved
its index to -1, leaving a physically valid particle without an Urban cell.
The failure log's H1/H2 energies are covered by the loaded Urban packages;
this is a numerical geometry inconsistency, not a missing-data fallback case.

## Production change

`src/detail/sycl_device_math.inc`: `urban_voxel_face(0)` returns the CT lower
bound and `urban_voxel_face(count)` returns its upper bound. All internal
faces retain the original centre-anchored expression, including symmetric
reference-grid face ties. No epsilon nudge, geometry translation, changed
step cap, physics switch, fitted normalization or audit suppression is used.
All primary fluctuations, unrestricted stopping, nuclear physics and queue
checks remain unchanged.

## Verification

- Host check: 100,136 actual/synthetic grids, 1,201,632
  boundary samples; all pass. It reproduces the old failure and confirms
  51,250,196 internal faces are bit-identical.
- Native G4 `sm_120` AOT rebuild, `CUDA_DISABLE_PTX_JIT=1` throughout. Source
  SHA inventory confirms only `src/detail/sycl_device_math.inc` differs from
  the failing binary's source snapshot.
- All 12 CT case/modality combinations complete three resident batches:
  45 field-batches pass runtime quality, energy, coverage and
  completion checks with zero queue overflow; finite nonnegative dose checked.
- Formerly failing Lung 1 carbon minibeam completes
  7,499,935 histories over three batches; no EM failure.
- The 11 formerly successful cases use exactly the same source CSVs and seeds.
  40 available old/new field-batch doses are compared; maximum
  relative dose L1 difference is 1.87474457e-07.
  The compared nuclear/elastic interaction counts, terminal-reason counts and
  overflow counters are identical. Maximum physical relative energy residual
  across all 12 cases is 4.9278e-6.

This verifies the geometry repair and matched GPU implementation behaviour;
it does not constitute a new full-plan TOPAS dose-accuracy acceptance.

## Evidence

`tmp/urban_ct_outer_face_20261018/`: `HOST_GEOMETRY_CHECK.json`,
`source_sha256.json`, `build_commands.json`, `BUILD_COMPLETE.json`,
`COMPLETE.json`, `RESULTS.json`, `MATCHED_COMPARISON.json`, and
`audit_records.tar.gz` with field-level quality/energy/completion records.
The previous rejected run remains in `tmp/g4_twelve_modes_20261018/`.
