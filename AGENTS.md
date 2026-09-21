# MAIGO Agent Instructions

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
