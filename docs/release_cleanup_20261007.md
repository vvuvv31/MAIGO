# Release compatibility cleanup (2026-10-07)

The live runtime removes the audited old-route compatibility code and inactive
interfaces while preserving the eight presets' source and physics behavior.
Source/header/config diff: 2218 removed lines, 301 added lines,
net 1917 fewer lines (including whitespace cleanup and identifier renames).

Removed:
- Unused Schneider CSV resampling/interpolation and GPU-test interfaces;
  obsolete partial-water and neutral CSV reader APIs.
- Unused HU conversion tables, built-in path search, and ignored DICOM arguments
  on native CCTG loading and input-cache keys. Native Schneider composition
  parsing remains required for water nuclear-rate construction.
- Fixed-off LET/validation scoring and primary survival/first-interaction buffers,
  helper functions, result fields and obsolete build-mode definitions.
- Retired CINEL02 zero-only diagnostic, replay and outcome arrays/JSON layouts,
  and 17 unused device helpers. Live ion species energy and terminal audits remain.
- The duplicated phase-space `copper_touched` field; initial-ray and actual-touch
  fields retain their existing meanings.
- Missing-projectile C12 inference, the old C12-specific oracle-status exception,
  and redundant old-key rejection after the strict preset allowlist.
- Missing-dimension CT face-clamp defaults. Zero dimensions now yield an invalid
  result; required outside-grid face geometry remains unchanged.
- Acceptance of absent grid-split accounting for nonzero deposited energy.

No compatibility aliases for removed keys or renamed live audit fields are kept.
The active fluctuation law is retained byte-for-byte, with its selector renamed.
Current source aliases used by the presets, current nuclear event replay,
fragment species, copper/air/water physics and calibration approximations remain.

## Configuration and output migrations

| Before | Release |
|---|---|
| `cinel02_max_secondary_inelastic_generations` | `max_secondary_inelastic_generations` |
| `legacy_clipped_gaussian` | `clipped_gaussian` (same model) |
| `scorer_mode`, `enable_fragment_species_scoring` | Removed; unsupported keys |
| `cinel02_species_transport_ledger_*` | `ion_species_transport_ledger_*` |
| `cinel02_species_terminal_reason_*` | `ion_species_terminal_reason_*` |
| Other retired `cinel02_*` zero outputs | Removed |

The eight supplied YAMLs are migrated. Ship the updated inputs with the release:
- `data/schneider/schneider_inelastic_rates_v2_1.metadata.json` now explicitly
  declares projectile `z=6,a=12`; all other JSON values are unchanged.
- `data/carbon_urban/active_copper005.csv` uses `VALID_ACTIVE_STEP_CONTEXT`;
  every other header/data byte is unchanged. The old status is rejected.

Source/input/dependency inventories are refreshed. The previous binary and
original preset files were frozen under `/tmp/maigo_release_before_20261007/`.
Original copies of the two migrated input files are retained there for checking;
the GPU control accepts the migrated identity metadata without numeric changes.

## Validation

- Release SYCL/CUDA build passed (`BUILD_JOBS=6 bash scripts/build.sh`).
- Eight matched GPU cases passed, two batches of 10,000 primaries per arm per case.
  Primary batch/source CSVs are byte-identical. All integer EM, charged/photon,
  completion, Urban, cascade, elastic and nuclear audits match. Current ion energy
  ledgers match within the existing comparison tolerance; terminal counts match
  exactly. Quality/energy/coverage/completion checks pass with zero queue overflows
  and finite nonnegative nonempty dose. Maximum relative dose L1: 3.00661672673e-07.
- All 28 retired interface rejection checks passed.
- Seven strict metadata/oracle/grid checks passed, including unchanged numerical
  copper reference arrays and rejection of the old inputs.
- Current strict26 proton/carbon/shared electron package checks passed.

Candidate SHA256: `1feeb347bac76c47966d7c9f12e4bc67fe31959e0e6e8c66594d152bf9b79c63`.
Control SHA256: `27bb1d613433c47effd40bcd86add8c0956b8faee86f67c873234dbf52adbfdb`.

The first GPU regression stopped at the carbon CT broadbeam because its YAML still
contained the removed false-valued scorer field. That YAML was corrected and the
complete eight-case suite rerun in a fresh directory. The stopped run is retained
as `validation_release_20261007/` and excluded from acceptance.

Evidence:
- `validation_release_final_20261007/summary.json` and per-arm logs/quality/energy files.
- `interface_release_20261007.json`.
- `release_inputs_20261007.json`.
- `package_release_20261007.json`.
- `release_build_20261007.log`.

These checks establish matched implementation consistency, not new TOPAS gamma,
full-plan dose acceptance or throughput. Carbon minibeam dose accuracy remains
deferred; current continuous26 calibration approximations are unchanged.
