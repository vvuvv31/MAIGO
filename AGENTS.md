# MAIGO Agent Instructions

## Case identity and validation

- RT07575 is a **head (cranial) case**, not a lung case. Its CT, spot plan,
  collimator geometry, normalization and TOPAS full-plan reference must be kept
  together when validating a full plan.
- The lung CT minibeam presets are a separate benchmark. Do not label their
  dose, gamma or throughput results as RT07575 head full-plan validation.

## Default ion EM representation

- All eight presets use `data/em/unified_em_v1.bin` (EMCNT001), with exactly
  26 compositions (25 Schneider + Water_75eV), 468 records and continuous
  density response. No 175-state host expansion or GPU density-pair lookup.
- The production GPU contains only the continuous26 path and rejects old
  density-state interfaces at startup. Host diagnostics and frozen reference
  binaries may explicitly read archived packages; no implicit fallback.
- The cleaned generation source `data/em/ion_em_26_composition_bases.npz`
  has exactly26 directory entries and26 density-domain bounds. Small independently
  calculated cut/density-effect scalar grids and range/rate polynomial
  coefficients are not complete material-density EM response states.
- Historical sources are separated in `data/archive/ion_em_native_20261005/`,
  `data/archive/ion_em_density_residual_20261005/` and
  `data/archive/ion_em_analytic_expanded_20261005/`.
- Broadbeam absolute TOPAS gates pass. The0.1% head minibeam comparison must
  be reported independently; default promotion is not universal dose acceptance.
  Carbon-minibeam accuracy remains deferred; its runtime audits are required.

## Ion fluctuation defaults

- All eight presets and TransportConfig defaults enable primary and secondary
  ion energy-loss fluctuations. All four minibeams enable copper primary and
  charged-fragment fluctuations. Historical carbon CT broadbeam acceptance
  with fluctuations OFF does not validate the current ON configuration.

## Default electron collision representation

- All eight presets select the installed26 composition packages in
  data/electrons/{shared,proton,carbon}/. Shared MAIEC026/MAIPH026 indices
  each contain26 entries; no runtime density-node lookup or implicit83 fallback.
- Birth uses MAIBCA26 actual-parent/cut shell sampling. Projectile-specific
  MAIEPR26 parameters are44 bytes each; shared aliases are hardlinks to the
  species files. Shared charged hard outcomes use Penelope/Moller, complete
  atomic cascades and elemental SB/2BS. Photon terminals use MAIPHF01
  calibrated composition laws with uint64 CDF and FP32 records.
- Continuous scalar fits handle stopping/range/cuts. Actual-cut hard rates
  mass-scale; Penelope laws and photon conditional terminals retain calibration
  density/cut approximations. Native continuous-density component checks do not
  imply those refinements are already integrated in this default.
- Existing25-composition spatial joint remains active; Water_75eV uses
  independent steps. Electron section25 maps to ion EM water section-1 when
  preparing birth energy/cut support. All GPU scoring/geometry/loss stays FP32.
- Old density-node banks are explicit offline references in
  data/archive/electron_density_nodes_20261006/. Native collision sources
  remain in data/archive/electron_collision_native_20261005/. No implicit
  default fallback or local-deposition substitution for missing coverage.
- CT dose gates pass under the user-authorized minibeam relative1pp criterion;
  absolute head TOPAS gamma98.9001% remains below99%. Carbon-minibeam dose
  accuracy is deferred. Migration smoke does not establish water dose acceptance
  or a throughput gain. See docs/electron_complete26_20261006.{md,json}.

## GPU performance work

- User-authorized complete electron26 replacement (2026-10-06): build one
  shared 25-Schneider+Water_75eV electron/photon response and separate
  proton/carbon parent-dependent parameters; retire all83 runtime density
  nodes only after the replacement runs independently. No silent83 fallback.
  For minibeam, use local3%/0.2mm/search0.02mm and require the TOPAS gamma
  pass percentage to drop by no more than1 percentage point from the matched
  old-path baseline at identical histories/geometry/normalization. Report
  absolute TOPAS gamma independently; the former absolute>99% minibeam
  promotion gate is superseded by this user instruction. Broadbeam remains
  local1%/1mm/search0.1mm>99%. Carbon-minibeam dose accuracy remains deferred;
  its runtime/energy/coverage/completion/overflow audits are still mandatory.

- Before proposing or implementing a performance experiment, search
  `failed.md`. Do not repeat a listed failed route unless its documented
  retry condition is satisfied by new evidence. Append every newly rejected
  candidate to `failed.md` before ending the task.
- Always use FP32 dose scoring/atomics for GPU performance development:
  `CARBON_DOSE_FP32=ON` and `CARBON_DOSE_FP64=OFF`.
- Do not configure, build, run, propose, or use an FP64 dose scorer as a
  performance or validation fallback unless the user explicitly overrides this
  rule in the current request. FP64 scoring is too slow for this project.
- For scheduling optimizations, FP32 atomic accumulation-order differences are
  expected and are not, by themselves, a reason to reject a candidate. Continue
  to require the transport/EM audits, quality checks, energy accounting, and
  queue-overflow checks to pass.

## Water physics reference

- The project water reference is the TOPAS built-in `Water_75eV`
  (`MeanExcitationEnergy = 75 eV`). The unified EM extraction
  (`extensions/tools/unified_em/prepare.py`), `data/stopping_power_water_geant4_11_3_2.csv`
  and `data/ion_stopping_power_water_geant4_11_3_2.csv` are all extracted for
  `Water_75eV`.
- Do NOT use Geant4 `G4_WATER` (I = 78 eV) as the water material in GPU/TOPAS
  comparisons. The Bethe `ln(I)` term makes `Water_75eV` stopping ~0.43% higher
  than `G4_WATER`, shortening the C12 CSDA range by ~0.45% (~0.6 mm at 250 MeV/u,
  124 mm range).
- `minibeam_water_primary_stopping_power_scale = 0.9958` is exactly that
  `S(78 eV)/S(75 eV)` ratio, not a free calibration. Fix the material reference
  instead of retuning the scale; keep the scale at 1.0 once both engines use
  `Water_75eV`.
- The water material probe `data/water_unified/g4_water_material.json` reports
  `G4_WATER` (78 eV) for composition/radiation-length purposes only; it must not
  be treated as the EM stopping material.

## Independent26 electron dose-first experiment (2026-10-06)

- The user explicitly authorized running a complete26-composition candidate
  before finishing continuous-density hard-rate/terminal refinements. Candidate
  runtime must never read the old82/83-node banks; offline generation/reference
  controls may read them explicitly. Default promotion follows the completed CT dose gates and the user
  authorization to switch; smoke checks runtime integration only.
- Candidate package: `out/electron_complete26_20261006/gpu_package_v1/`, using
  MAIEC026/MAIPH026 directories with exactly26 compositions, zero runtime
  density nodes, and separate44-byte proton/carbon MAIEPR26 birth parameters.
  Charged outcomes use Penelope/Moller, atomic cascades and SB/2BS. Preserve
  spatial joint spreading. The whole shared candidate is about1.75GB, not23MB.
- First-dose approximations are explicit: actual-cut hard-rate mass scaling,
  calibrated Penelope oscillator laws and photon conditional terminals. Scalar
  stopping/range/cuts use continuous density fits. Do not claim these unfinished
  physical refinements are qualified merely because runtime audits pass.
- Rebuilt GPU v2 passes all four CT smoke/audit screens. File-open tracing
  verifies zero old-node bank reads. A two-ulp scalar logdomain endpoint guard
  fixes CPU/GPU log rounding; exact physical density bounds remain enforced.
- Formal RT07575 head0.1% comparison uses236,086,480 primaries and the matched
  original83 control. Minibeam TOPAS local3%/0.2mm may decrease by at most1
  percentage point (search0.02mm); broadbeam remains local1%/1mm>99%
  (search0.1mm). Carbon minibeam dose acceptance remains deferred. If gamma
  changes are within the authorized allowance, the user does not require
  additional component refinement for this dose-first trial.

- Completed dose-first results: all119 head batches /236,086,480 primaries
  pass runtime/energy/EM/coverage/completion and zero-overflow audits. Original83
  TOPAS local3%/0.2mm98.90331377%, independent26 98.90008030%, decrease
  0.00323347pp; matched GPU gamma100%. Both remain below absolute99%,
  which is separate from the user's relative replacement gate. Proton/carbon
  broadbeam local1%/1mm99.86565492%/99.79050826% pass. Under the user's
  dose-first instruction, more physical component refinement is not required
  by these gamma results. Production preset defaults were switched on2026-10-06 after these dose results; see the migration smoke report.
  Evidence: docs/electron_complete26_20261006.{md,json} and
  out/electron_complete26_20261006/dose_summary_v2.json. A v4 diagnostic-only
  rebuild fixes source-byte double counting and passes a fresh head closeout;
  formal dose results use the frozen v2 transport binary.
