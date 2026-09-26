# Global Urban v2 benchmark path (2026-09-23)

`multiple_scattering_model: urban_v2` selects the shared Geant4 11.3.2
Urban implementation for primary and all transported charged secondary ions
on the CUDA SYCL path. The old copper-only `urban` selector is deprecated;
its existing shared math helpers are still used by v2.

This is a research implementation. The current reference package supports
Water_75eV at 1 g/cm3 and the four exact Schneider material/density couples
used by benchmark20260921 b1–b4 (sections 0, 1, 8, 20). It is not a general
25-section/continuous-density patient package. Unknown materials, densities,
ions, energies or malformed tables fail the run instead of switching models.

The 52-ion union includes the ordinary transported charged species and all
elastic-recoil isotopes. Existing nuclear scope/generation policies remain
in effect. Delta electrons are deposited locally by the unified EM model;
this change does not add explicit electron transport.

## Configuration

```yaml
multiple_scattering_model: urban_v2
urban_mcs_package_file: /absolute/path/urban_mcs_candidate.bin
urban_mcs_package_sha256: 493015506d0168efc16afae6b81441010d7937e899d3e88196c45a5b8c511eb9
# Required for homogeneous water: TOPAS physical water-mother half width.
urban_water_half_width_mm: 2000
```

Requires unscaled MCS, `g4_material_joint_v1`, secondary unified EM,
transport cutoffs >= 0.05 MeV, and (for CT) Schneider materials with exact
secondary voxel faces. The package uses a 0.05 mm production cut. Water must
use 75 eV excitation energy; no stopping-power or dose calibration is applied.

The physical water mother surrounds a scored replica ROI. Outside the ROI,
water transport and possible reentry continue without fictitious voxel faces.
Geometry clips the Urban geometrical path; energy loss and nuclear clocks use
the corresponding true path. Nuclear competition can shorten that true path
before one angular/displacement sample is committed. Continuous secondary
loss is assigned to the source voxel, with direction-owned exact faces.
Track limiter state persists across secondary scheduler slices.

## Reference extraction and verification

`extensions/tools/urban/extract_active_urban.cc` binds the actual Geant4 Urban
and energy-loss processes. It exports native loss/range/inverse spline
intervals, active charge/density factors, and transport mean free paths.
All inverse/range/DEDX queries for one step hold the active factor fixed,
matching `G4VEnergyLossProcess`; heavy-ion active ranges cannot be treated as
a single monotonic global R(E) curve.

The native equivalent-electron 10 MeV mean-free-path branch discontinuity is
preserved with adjacent FP32 energy knots. It is not smoothed away.
Campaign and packing scripts live beside the extractor. Provenance, SHA pins
and independently sampled validation records are in the package manifest.

Reference work ran on cluster cpu188 in allocation 2022978, up to 56 CPU
workers and below 256 GB. The production FP32 lookup implementation passed
1,064,960 held-out Geant4 steps, with worst relative error 6.80e-6 (0.00068%).
This validates lookup arithmetic, not complete Monte Carlo dose agreement.

`urban_global_contract PACKAGE SHA256` exercises every material/ion record,
low and high energies, nuclear truncation, and navigation/face ownership.
Benchmark dose acceptance additionally requires EM audits, energy accounting,
voxel-to-ledger closure, finite nonnegative dose, and zero queue overflow.

The pre-existing `urban_v2_helpers` subdivision test fails at 0.05/0.1 MeV
with the old C12 water table, identically at baseline commit 0f6747. This has
not been relabelled as a pass or repaired by changing the legacy table.

GPU-only results and reused TOPAS references are under
`/mnt/sdb/wuwei/MAIGO_pristine/benchmark/benchmark20260923Urban`.
See its RUN_STATUS.json and comparison artifacts for run acceptance/results.

The all-ion threshold regression also exercises the representable values
immediately around `0.05 * range`. The float branch selector can enter the
range-law branch when the subsequent double ratio is just below 0.05.
The analytic delta is valid for all 0 < t/range < 1; a redundant double
0.05-domain guard incorrectly rejected rare oxygen recoil steps. Removing
that inconsistent guard preserves the selected analytic law. The extended
28,088-contract suite passes (401 failures in the initial edge-only suite
before this correction).

## Ionization proposal clock (2026-09-24)

The global `urban_v2` path now uses the existing native ionization proposal
rate and exponential optical-depth clock to limit true transport steps.
This applies to primaries and secondary ions using unified EM. The clock
survives voxel/material boundaries and secondary scheduler slices; only the
accepted true path is debited after geometry and nuclear competition. A
proposal restarts the clock even when it would be a null/rejected collision.
The existing aggregate energy-loss distribution remains in use: this is
not explicit delta-electron transport or a complete restoration of the
energy-jump/ionization-clock correlation.

This prevents the user's 1 mm / 0.5 mm maximum step from replacing the
shorter ionIoni segmentation used by the reference. No MCS, stopping-power,
safety, material, or dose normalization scale was adjusted. The dedicated
minibeam copper/water Urban path is unchanged.

Validation: full-physics C12 250 MeV/u water and soft-tissue/lung CT,
4 paired seeds x 200,000 histories, at 1, 0.5 and 0.0625 mm (24 runs).
CT lung-side dose at 240--280 mm changes by -0.992% / -1.128% relative
to the finest step, versus baseline -2.788% / -2.969% and TOPAS
-0.744% / -0.795%. Water center point estimates remain within 0.10%.
All quality, unified-EM, dose, energy-accounting and queue checks passed.
These are region-level results; low-dose tails and other materials/energies
have not established universal 1% convergence. See the complete uncertainty,
integration and runtime records under
`/mnt/sdb/wuwei/MAIGO_pristine/benchmark/benchmark20260924StepFix`.

## Proton primary source (2026-09-26)

The non-minibeam water entry is `config/proton_water_fullphysics_urban_v2.yaml`.
It uses the existing Z/A-based primary Urban routing and a pinned local copy
of the 52-species reference bank, which includes proton. The global selector
also applies to charged secondaries and elastic recoils. See
[proton configuration, geometry and validation scope](proton_urban_v2.md).
No new proton transport/dose acceptance is implied by this configuration.
