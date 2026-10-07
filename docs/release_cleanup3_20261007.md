# Release compatibility cleanup, third pass (2026-10-07)

The three remaining audited interfaces are removed without changing the eight
presets' numerical inputs, source geometry, model laws or scheduling.

- Primary nuclear channel sidecars require explicit boolean `has_support`.
  Missing fields are no longer inferred as `true`; nonboolean values fail.
  Support, energy domains and zero rates for unsupported channels still undergo
  the existing rate/package consistency checks. The carbon sidecar's 13 channels
  now explicitly declare `true`. Removing these new keys yields the original
  JSON exactly; all original values are unchanged. Proton's explicit unsupported
  channel remains unchanged.
- Neutron loading accepts only the current native-collisions.v2 manifest and
  always loads and validates its independent macroscopic grid. The v1 manifest
  reader and unused audit/runtime mode argument are removed from the API and all
  three callers. The current package/backend remain explicitly experimental;
  `runtime_validated: false` and its corresponding header flag are required.
  No dose-acceptance status is upgraded. The v2 manifest still uses the current
  NEUTV001/version-1 collision binary layout; that layout is not v1 manifest
  compatibility. Both neutron binaries are unchanged.
- The unreachable TPS isocenter vector/X/Y parser is removed. Only the supported
  `tps_isocenter_z_mm` input is parsed. Internal source geometry remains unchanged.

Ship the updated `data/schneider/cinel03_c12_targets_v2_1.channels.json` with the
release. Old sidecars without `has_support`, neutron v1 manifests, and the removed
TPS keys are rejected; no aliases remain for these interfaces. README, repository
instructions and dependency/copy inventories describe the current contracts.

The frozen control is `/tmp/maigo_release3_before_20261007/`: original binary,
source/config/scripts and saved original carbon sidecar. Its data/inputs reference
unchanged assets and the migrated sidecar, which the control interprets identically
because its former missing-field default was `true`.

The complete eight-case regression passed. Validation evidence follows.
The regression checks implementation consistency, not new TOPAS gamma acceptance,
full-plan validation or throughput. Carbon minibeam dose accuracy remains deferred.


Validation:

- SYCL/CUDA release build passed; existing compiler warnings remain.
- All eight matched GPU presets passed, two batches × 10,000 incident primaries
  per arm per case. Primary source CSVs are byte-identical. Discrete EM,
  electron/photon, Urban, cascade, elastic and nuclear audits match exactly.
  Live ion ledgers pass existing energy tolerances and exact terminal counts.
  Quality, energy, coverage and completion checks pass with zero queue overflow
  and finite, nonnegative, nonempty dose. Maximum relative dose L1: 2.65987250529e-07.
- All 41 retired-interface/missing-identity checks passed on the final binary.
- All eight package contract checks passed: explicit carbon/proton support,
  current neutron v2 payload, v1 rejection, missing macroscopic rejection,
  forged validation rejection, missing support rejection, nonboolean support
  rejection and support/domain mismatch rejection.
- Existing 12 strict identity/mass/TPS/CT-geometry check groups passed.
- Source/header runtime diff: 48 removed lines, 18 added lines, net 30 fewer
  runtime lines. New negative-input verification scripts are separate.

Candidate SHA256: `416445ee023deff5c8bf484a02f3c379b22c448b4c58192801563c2be2f024ce`.
Control SHA256: `750c44251a04c3363ac289bf9de152a01a8705c3ea8263514c60c51f957b2f89`.

Evidence:

- `validation_release3_20261007/summary.json` and per-arm logs/results.
- `release3_regression_20261007.log` and `release3_build_20261007.log`.
- `interface_release3_20261007.json`.
- `package_contracts_release3_20261007.json` and `release3_inputs_20261007.json`.
- `release3_changes_20261007.patch` and `release3_changes_20261007.json`.
