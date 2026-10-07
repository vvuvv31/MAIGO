# Release compatibility cleanup, second pass (2026-10-07)

This pass replaces the remaining audited input compatibility branches with
explicit production contracts. It preserves the eight scenes, source geometry,
normalization, per-spot seeds, fluctuation laws and calibrated physics.

## Removed paths

- `primary_particle` names, implicit C12 source identity, total-MeV YAML energy
  alias, percent-spread YAML alias, configurable zero-mass sentinel and generic
  primary-species construction. Internal charged-fragment species remain.
- C12 exceptions that skipped package mass validation, including the primary
  CINEL03 event loop. All primary masses now undergo validation.
- Copper Urban CSV oracle reader, metadata API, configuration key, reference CSV,
  GPU arrays and interpolation route. Current packaged loss/range/inverse/MFP
  contexts already superseded these arrays before every production call.
- Unused analytic copper/water Urban cross-section and MFP implementations that
  belonged to the nonpackaged route. Current Urban scattering/limiting remains.
- TPS column aliases and the secondary per-nucleon spot-energy representation.
  Central generated spots convert to total MeV at construction. Beam optics
  accepts `energy_MeV` only. Duplicate normalized CSV columns are rejected.
- Duplicate accounting closure API/JSON fields. One physical ledger remains;
  absolute and relative closure both include the explicit other-primary terminal
  energy sink. Physical audit tolerances and all overflow/coverage gates remain.
- Impossible noncontinuous-format conditions in the EMCNT001-only reader;
  regression-script fallback to retired CINEL02 ledger names; obsolete comments
  and constant old82/fallback diagnostic labels.

## Explicit mass conventions

The current reference families do not all use the same C12 mass. Unifying them
would change calibrated physics. The production code declares both conventions
and validates the expected convention rather than accepting unchecked values:

| Model | C12 mass (MeV) |
|---|---:|
| Primary kinematics / primary CINEL03 | 12 × 931.49410242 = 11177.92922904 (FP32 events: 11177.9296875) |
| Urban / continuous26 EM / elastic reference | 11174.86338798439 (FP32 banks: 11174.86328125) |

The existing 2e-6 relative mass tolerance allows reference precision differences
but rejects interchange of these two conventions. FE retains its calibrated
FP32 nucleon mass; copper retains its FP32 A-times-nucleon product. No zero-mass
input, explicit-mass override or C12 validation bypass remains.

## Migration

| Interface | Current release |
|---|---|
| YAML `primary_particle` / omitted identity | Required `primary_atomic_number`, `primary_mass_number` |
| YAML `initial_energy_MeV` | `initial_energy_MeVu`; all proton values are unchanged because A=1 |
| TPS `energy`, `energy_MeVu`, `x`, `y`, `mu_weight` | `energy_MeV` (total ion KE), `x_mm`, `y_mm`, `weight` |
| Beam model `energy` | `energy_MeV` |
| `minibeam_copper_loss_range_file` | Removed; current Urban packages supply the context |
| Ledger `energy_balance_error` duplicate | `physical_energy_balance_error` |
| Quality `absolute_accounting_energy_residual_MeV`, `accounting_relative_energy_residual` | Existing `absolute_physical_energy_residual_MeV`, `physical_relative_energy_residual` |

All eight YAMLs are migrated. The carbon water minibeam spot CSV header is
migrated; its numerical rows are unchanged. The obsolete
`data/carbon_urban/active_copper005.csv` is removed from release inputs. Dependency
and copy inventories are refreshed. Earlier documents describe their historical
validation snapshots; this document describes the current release contract.

The frozen control is `/tmp/maigo_release2_before_20261007/`. It has the previous
binary, original source/config/inputs, shared unchanged physics packages and a
saved copy of the retired copper CSV. Its one CSV path points to that saved copy.

## Validation

The initial run in `validation_release2_20261007/` stopped on the first preset:
a secondary FE caller still supplied the retired zero-mass override after the
API changed to per-nucleon mass. The source CSVs were byte-identical, but EM
integer audits differed. That caller now supplies the explicit original FP32
nucleon mass. The failed run is excluded from acceptance; the complete suite
passed in a fresh directory. Final validation results follow below.
These checks are implementation regressions, not new TOPAS dose acceptance,
full-plan validation or a throughput measurement. Carbon minibeam dose accuracy
remains deferred; continuous26 calibration approximations are unchanged.


Final accepted evidence:

- SYCL/CUDA release build passed. Existing compiler warnings remain; this is not
  a warning-free build claim. Log: `release2_build_20261007.log`.
- All eight matched GPU cases passed, two batches of 10,000 primaries per arm
  per case. Source batch CSVs are byte-identical, discrete EM/electron/Urban/
  elastic/cascade/nuclear audits match exactly, and live ion energy/terminal
  ledgers satisfy the existing comparison checks. Quality, energy, coverage and
  completion checks pass with zero queue overflows. All dose arrays are finite,
  nonnegative and nonempty. Maximum relative dose L1: 2.92725282525e-07.
- All 38 retired-interface and missing-identity rejection checks passed.
- All 12 strict projectile/mass/source/geometry check groups passed.
- Shared/proton/carbon strict26 electron package checks passed.
- Source/header/config/script diff: 1043 removed lines, 207 added lines, net 836
  fewer lines. Input CSV migration and retired reference-data removal are
  additional to these counts.

Candidate SHA256: `750c44251a04c3363ac289bf9de152a01a8705c3ea8263514c60c51f957b2f89`.
Control SHA256: `1feeb347bac76c47966d7c9f12e4bc67fe31959e0e6e8c66594d152bf9b79c63`.

Evidence files:

- `validation_release2_final_20261007/summary.json` and its per-arm results.
- `release2_final_regression_20261007.log`.
- `interface_release2_20261007.json`.
- `release2_inputs_20261007.json`.
- `release2_packages_20261007.json`.
- `release2_changes_20261007.patch` and `release2_changes_20261007.json`.
