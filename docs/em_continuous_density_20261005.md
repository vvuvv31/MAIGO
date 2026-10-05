# Continuous-density ion EM runtime — 2026-10-05

Status: installed as the eight-preset default. Broadbeam comparison passes; the final RT07575 head 0.1% campaign is complete and remains below the TOPAS >99% gate. Universal TOPAS accuracy acceptance is not claimed.

The installed model removes the 175-state ion EM directory and dense host expansion. Its `EMCNT001` package contains 26 compositions (Water_75eV plus 25 Schneider compositions), 18 ion species and 468 records. Material selection is a direct composition index. Density is the actual voxel density; there is no low/high EM material selection. Density responses are not stored as separate transport states. Historical packages remain readable only for explicit regression comparisons.

## Density and energy treatment

HU-to-composition and HU-to-density still follow the existing TOPAS Schneider mapping. Material composition remains discrete at Schneider section boundaries. Density is continuous within each section; this does not linearly mix different compositions.

Restricted stopping applies the analytical production-cut and density-effect correction to the one composition reference spline. The Bragg cut floor, reference-alpha normalization, effective charge and low-energy ion correction remain explicit. Ion fluctuations use the actual production-cut energy and density-scaled variance.

A lightweight 257-sample scalar table per composition interpolates independently calculated cut energy, electron-density coefficient and Sternheimer parameters in log density. This is not 257 complete EM response states: the entire scalar payload is 213,824 bytes. It contains no target stopping, range, delta moment or response-residual tables. The cut converter is Geant4's range-to-energy conversion; it is not described as a closed-form conversion.

Restricted-range integrals are represented by degree-23 Chebyshev polynomials in log density, evaluated at the reference energy knots. Their training data comes from the analytical 26-base model, never the 175 target EM responses. Range inversion uses the reference inverse spline as an initial estimate and Newton iteration on the corrected integral, retaining the original near-endpoint range branch. This is a fitted integral approximation and requires separate dose validation.

The hard-delta rate retains a continuous correction for the Geant4 lambda spline on normalized log energy. Direct Born-rate substitution was rejected at the component stage because it changes the threshold spline. First and second delta moments are evaluated from the conditional native spectrum; finite-size and magnetic/spin veto corrections use log Gauss-Legendre quadrature, except spin-zero small-form-factor cases (F×Tmax≤0.001), where a second-order expansion on the native active interval has a uniform remainder bound of 4.01×10⁻⁹. Unsupported threshold support remains zero. Production scoring and GPU arithmetic remain FP32.

Electron birth/charged/photon packages, nuclear event banks, ion/Urban scattering, dose scoring and step defaults are unchanged: broadbeam 0.5 mm, minibeam 0.1 mm. Carbon-minibeam dose acceptance remains separate.

## Storage and reproduction

Installed EM: 225,707,600 bytes. Delta base moments: 30,750,944 bytes. Total: 256,458,544 bytes, versus 1,593,660,068 bytes for the archived expanded analytical interface. This is an 83.908% reduction in the pair, not a claim about total electron/nuclear data or measured device residency.

The cleaned lossless basis source has exactly 26 directory entries plus 26 density-domain bounds. It occupies 106,313,320 bytes and contains no 175-entry directory. Independent offline material helper samples are not runtime EM states.

```bash
# Build analytic_density_metadata.cpp against Geant4 11.3.2 as documented in
# docs/em_analytic_density_20261005.md. Use fresh output directories.
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/software/geant4-11.3.2/lib \
python3 extensions/tools/unified_em/build_continuous_density.py \
  --bases data/em/ion_em_26_composition_bases.npz \
  --helper out/em_analytic_density_20261005/metadata \
  --output out/continuous_reproduction
# Build GPU with CARBON_DOSE_FP32=ON and CARBON_DOSE_FP64=OFF.
python3 scripts/validate_em_continuous_density.py --phase broadbeam \
  --candidate-root out/continuous_reproduction \
  --reference-root data/archive/ion_em_analytic_expanded_20261005 \
  --output out/continuous_validation
```

Generation metadata and component/GPU raw evidence are retained under `out/em_continuous_density_20261005/`. An archived package is never an implicit runtime fallback. Component changes alone do not establish the strict TOPAS gamma gates (broadbeam local 1%/1 mm >99%, minibeam local 3%/0.2 mm >99%, search 0.1×DTA). RT07575 is a head case, independently of the lung minibeam presets.

## Certified evaluation and validation evidence

The GPU compiler removes the archived density-pair branches. Production transport rejects any ion EM package other than the 26-composition/468-record continuous model. Host diagnostics retain historical readers for explicitly selected references.

At load time, omitted Chebyshev coefficient magnitudes certify truncation uniformly over the fitted density domain. In 2,276,742 checks, maximum log-range difference is 3.72063×10⁻⁷ and rate-factor difference is 4.98295×10⁻⁵. The resulting mean-loss relative difference is at most 0.003043%. Most rows require one coefficient. Independent 64-point split-interval veto quadrature over 22,256 cases bounds measured moment relative differences by 3.425×10⁻⁷ (M1) and 4.192×10⁻⁷ (M2). These checks concern implementation approximations; they do not erase the underlying model differences.

Against the expanded analytical model, 2,428,650 finite-step checks reach 0.950578% mean-loss difference, 0.260410% range difference and 0.381468% stopping difference across the full tested energy band. Near-zero delta support can have 100% relative changes; there is no tight global relative-delta guarantee.

Two reversed-seed CT broadbeam pairs (68 batches) using the specialized/truncated build give:

| Case | Expanded control TOPAS local 1%/1 mm | Continuous TOPAS | Continuous versus control | Relative dose L1 |
|---|---:|---:|---:|---:|
| Proton, 25,063,000 primary/arm | 99.882928% | 99.883408% | 99.999520% | 0.148175% |
| Carbon, 12,963,818 primary/arm | 99.795077% | 99.794424% | 100% | 0.015599% |

Search is 0.1 mm. Runtime, energy, EM/electron completion and queue-overflow gates pass. The final bounded-veto/zero-density-effect-pow build subsequently repeats both candidate broadbeam arms against verified controls: the TOPAS gamma values remain identical. Its independent moment, final default-smoke and full-plan checks are complete.

A low-statistics head screen (11,903,520 primary/arm) had only 77.3716% matched-GPU local 3%/0.2 mm and 4.0108% relative dose L1, despite a small summed-dose change. This does not establish that the discrepancy is entirely Monte Carlo noise. The full 236,086,480-primary comparison is required.

## Performance scope

Final-build single screens, three ordinary batches per case, give 163,307 primary/s proton CT broadbeam, 77,173/s carbon CT broadbeam, 26,235/s proton lung CT minibeam, 9,102/s carbon lung CT minibeam and 78,240/s proton RT07575 head minibeam. These are unlocked-clock screens, not precise paired speedup estimates. The continuous model is slower than the old expanded lookup in the available comparisons. Its reduced storage must not be reported as improved throughput. A generic broadbeam run reduced tracked allocation from 4.3356 to 3.0774 GiB; that is version-specific allocation evidence, not total device residency.

The cleaned 26-entry source reproduces the active pair byte-identically (cmp, no digest). Former expanded analytical binaries and their old 175-entry generation metadata are archived. Electron packages are unchanged.

## Final RT07575 head full-plan result

The final build completes 119 batches and 236,086,480 primary (0.1% of the 4 Gy plan), with all 487 positive spots sampled. Ion maximum step is 0.1 mm. Runtime/EM/electron completion/energy/zero-overflow checks pass in every batch; maximum relative energy residual is 1.198×10⁻⁷. Combined run wall time is 3,015.11 seconds (50.25 minutes), including per-process setup; the main 95-batch arm sustains 78,696 primary/s.

The absolute 3D local 3%/0.2 mm comparison uses search 0.02 mm, BODY including GTV and reference dose ≥10% of reference maximum, without dose fitting or registration:

| Measure | Expanded analytical control | Continuous26 |
|---|---:|---:|
| TOPAS local gamma | 98.956666% | 98.903314% |
| GTV local gamma | 98.409441% | 98.233068% |
| Mask summed-dose bias versus TOPAS | +0.170090% | +0.173473% |

Neither arm passes the strict >99% TOPAS gate. The continuous model is lower by 0.05335 percentage points. Continuous versus matched expanded GPU control passes 99.624917%, with relative dose L1 0.901438% and mask summed-dose change +0.003378%. Pointwise local differences have median 0.764985%, 95th percentile 2.819115% and maximum 11.142468%; this is not a pointwise ≤1% claim.

The increased statistics substantially improve the low-statistics matched comparison, but do not prove all residual differences are noise. The default architecture change is implemented; universal absolute minibeam acceptance remains unresolved. Raw evidence is in out/em_continuous_density_20261005/head_0p1_final/.

## Final default and broadbeam closeout

The actual installed data/em/ paths pass two batches for each of the eight default presets, plus two independent head smoke batches (18 batches total). Uniform-water broadbeam correctly has an empty electron selection; all nonempty selections retain completion and coverage checks.

Final-build candidate broadbeam arms use the same formal statistics and seeds as the verified expanded controls. Package byte identity, physics config and history counts are checked before reuse. All 34 new broadbeam batches pass runtime gates. Final TOPAS local 1%/1 mm remains 99.883408% proton and 99.794424% carbon (search 0.1 mm), both >99%. Matched control is 99.999520% proton and 100% carbon; relative dose L1 is 0.148175% and 0.015585%. Final raw results are in out/em_continuous_density_20261005/broadbeam_final/ and default_final_smoke/.

The completed final-build checks cover 171 batches (119 head full plan + 34 broadbeam + 18 smoke). No queue overflow or runtime gate failure occurs. This establishes the requested runtime/package migration and broadbeam accuracy, while leaving the absolute head minibeam gate and carbon-minibeam accuracy unresolved. No throughput gain is claimed.
