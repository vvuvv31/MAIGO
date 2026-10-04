# Eight-preset cleanup and common minibeam electron transport

The pre-cleanup snapshot is commit `9ae8b88` on `origin/master`. This cleanup
retains proton/carbon × CT/water × broadbeam/minibeam. Historical TOPAS references,
native raw histories and experiment reports remain evidence for their original
versions; their old commands are not additional supported presets.

## Runtime changes

- Removed active content-digest implementations, checks, configuration fields,
  digest manifests and extractor dependencies. Package readers still reject
  schema/layout errors, invalid values, missing coverage and energy/queue failures.
- Removed water/material spatial trajectory replay, FE joint replay and proton
  tail routes, old explicit-lepton tables, water-only C12/light-ion scattering
  overrides, absorbing collimator geometry, old Copper FE/Urban scattering,
  artificial Be6 compatibility sinks and ignored legacy EM configuration aliases.
- All four minibeams use global all-ion Urban plus the independent FP32
  electron/photon worker. Projectile birth distributions, nuclear interactions
  and Copper physics retain their own data. No secondary-ion electron birth
  extension is claimed.
- Water electrons use physical homogeneous Water_75eV geometry, separate from the
  scoring ROI. The ion path remains native water. One water node was added to the
  shared operators and each projectile birth bank; the original 82 Schneider
  nodes were retained. No water joint acceleration is used.
- Independent birth tasks now retain only parent energy, material-node identity
  and the exact CDF random bits. The task record is 104 bytes instead of 128;
  trajectory references and 31 unreachable replay/compression headers were removed.
  Sampling draws and weights are preserved. This is not a measured throughput claim.
- In-process batches retain immutable package/geometry caches. File dependencies
  use path, size and modification time; this is invalidation metadata rather than
  cryptographic integrity verification.

## Data and regeneration

The active inventories are `docs/package_manifest.json`,
`docs/electron_packages_manifest.json` and `extensions/package_manifest.json`.
The current package schemas are MAIBIR02, MAIJNB02, MAIECB02 and MAIPHB02.
Large packages remain local/ignored; cloning Git does not install them.

Shared proton/carbon native birth extraction reproduces the previous water raw
records and compiled bank bytes exactly. Reassembling both complete 83-node banks
is also byte-exact. The carbon Schneider-node extraction fixture passes native
coverage and energy checks. The current joint compiler passes an eight-history
complete genealogy fixture. Native AlongStep and CINEL event-writer sources compile.
The delta-moment generator reproduces the installed table byte-exact with its
32/64-point quadrature checks, without digest checks.

## Validation status

All eight presets pass two-batch regression against their matching preserved
controls. Every checked transport counter, Unified EM audit and electron audit
is identical. Quality, energy, material/energy coverage, completion and overflow
checks pass. Relative dose L1 is at most 1.45e-7; the largest physical relative
energy residual is 5.42e-6. These differences include FP32 accumulation order.
The machine-readable summary is
[repository_cleanup_regression_20261005.json](repository_cleanup_regression_20261005.json).
Raw commands, logs and outputs are retained in
`out/repository_cleanup_20261004/final_preset_regression/`.

Broadbeam tracking configuration/roundtrip/guard checks and full-plan native batch
allocation, tiny/zero-spot, seed, overflow and cache tests also pass.

Ten host tests pass: independent births, joint handoff, electron birth geometry,
photon optical depth/outcomes, charged steps, broadbeam regions/distances, body
mask and input-cache dependencies. All three generated-electron header checks pass.
The active 218-file presence/size inventory and EM, elastic/recoil, Schneider and
electron package checks pass. No dose precision fallback was used:
`CARBON_DOSE_FP32=ON`, `CARBON_DOSE_FP64=OFF`.

Smoke/regression checks are not new TOPAS gamma acceptance or throughput results.
Carbon minibeam dose acceptance is pending. Water dose/performance validation is
deferred. Subsequent CT optimization requires local gamma >99% at 1%/1mm for
both broadbeams (search .1mm) and 3%/.2mm for proton minibeam (search .02mm).
Carbon minibeam must pass energy, coverage, completion and overflow audits.

## Larger-statistics correction

A subsequent RT07575 proton broadbeam screen (seed1001901102,1,253,150 histories)
exposed13 secondary elastic-draw errors that the small regression did not exercise.
Removing a retired water phase-space branch had left a dangling `else` before the
post-EM zero-rate guard. The guard is restored as an independent statement,
matching the pre-cleanup source. No audit or physical branch was disabled.
The identical seed now completes three batches (3,759,450 histories), with zero
elastic query errors, all quality/completion/coverage/overflow checks passing,
and maximum physical relative energy residual3.92e-7. Evidence:
`out/ct_throughput_20261005/runs/guard_fixed_proton_broadbeam`.
The eight-case small regression above remains a limited regression result, not
proof that every rare branch was covered.
