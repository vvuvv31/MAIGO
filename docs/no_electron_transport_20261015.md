# Explicit electron/photon transport removal — 2026-10-15

The runtime now transports primary protons/C12 and charged nuclear fragments,
with local ionization-energy deposition in water and CT. Explicit electron and
electromagnetic photon transport is removed for all eight release presets and
the generic pipeline.

## Preserved physics

- EMCNT001 continuous26 ion EM and the delta-moment sidecar, including existing
  calibrated rates and conditional terminal approximations.
- Primary energy-loss fluctuations and the existing secondary fluctuation policy.
- Aggregate delta loss and local scoring, including the same FP32 dose atomics.
- The Urban ionization proposal/segmentation clock. Removing the electron worker
  does not remove this step limiter or change its native rate.
- Nuclear elastic/inelastic competition, charged fragments, Urban/FE scattering,
  geometry, sources, spot seeds, normalization, and completion/energy/coverage audits.
- Copper hard-delta sampling and its existing beamline removed-energy convention;
  nuclear neutral energy remains an explicit untracked sink.

This is a maintenance refactor against the previously electron-disabled runtime.
It does not establish that electron/photon redistribution is negligible in every
low-density interface or minibeam valley. Historical dose comparisons retain their
original model and statistical scope.

## Removed interfaces and assets

Removed electron/photon package readers and device uploads, birth/grouping tasks,
charged/photon workers, spatial joint response, roulette/order queues, electron
energy ledgers, timing/output fields and electron-only kernel specializations.
The common CT geometry cache remains and is named for CT geometry.

The following configuration families are no longer accepted, even with false or
empty values: `enable_electron_transport`, `material_electron_*`,
`material_photon_response_index_file`, and `broadbeam_electron_*`. Remove these
keys from external YAMLs and corresponding fields from private C++ plan drivers.
Current release YAMLs and the dependency inventory have been updated.

Electron/photon extraction extensions, packaging tools and package-specific
checks are removed. Ion EM extraction remains supported. No runtime asset under
`data/em/electrons/` is required. Old release archives/Drive caches may still
contain that directory; omit it for new installations. Existing published
archives and historical reports have not been rewritten.

On the development machine, the former assets were moved to the ignored
`tmp/remove_electron_20261015/retired_electron_assets/` to preserve historical
reproduction without keeping them in the active runtime data tree.

## Validation

Validation artifacts are under `tmp/remove_electron_20261015/`. Native CUDA
compilation explicitly targets RTX 2080 Ti (`-DCARBON_CUDA_ARCH=sm_75`); all
executions set `CUDA_DISABLE_PTX_JIT=1`. Matched checks use the frozen pre-removal
binary, identical seeds and two batches per case. This is implementation
regression, not a new TOPAS acceptance or throughput measurement.

Completed CUDA checks:

- All five pipeline families compile as native `sm_75` code.
- Eight presets × two batches × 20,000 histories, in each arm: all accepted,
  zero overflow, matching source allocations, nuclear/elastic counters, Urban
  step audits and species terminal counts. Maximum dose relative L1 difference
  is **2.9525245909e-7** (0.00002953%).
- Generic proton broadbeam and generic carbon water minibeam: two matched
  20,000-history batches each; all accepted, maximum relative L1 **4.6234e-8**.
- 106 retired-interface checks (including the 16 removed electron settings),
  12 package contract checks, identity/mass/TPS/geometry checks, 104 installed
  asset checks and extraction-kit verification pass.
- Arc-specific device selection/capture code is retained; no new Arc device
  regression was run for this removal. The CUDA check is not an Arc validation.

No large paused campaign was resumed, no new TOPAS gamma was calculated, and no
throughput improvement is claimed from these matched implementation checks.
