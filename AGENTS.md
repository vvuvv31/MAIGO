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

## GPU performance work

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
