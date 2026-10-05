# 26-composition analytical ion-density experiment — 2026-10-05

Historical expanded-interface report: superseded as the default by [continuous-density runtime](em_continuous_density_20261005.md). Its recorded measurements remain unchanged.

The methods documents were committed and pushed first as `1020c50`. This subsequent experiment evaluates 25 Schneider composition responses plus Water_75eV, without target-density response residuals. RT07575 is a head case; lung CT minibeam presets are separate runtime regressions.

The installed default is byte-identical to the validated v3 pair. Previous sparse-residual data is archived in `data/archive/ion_em_density_residual_20261005/`; original native extraction remains separately archived. Basis and independent formula metadata are installed under `data/em/`. See [package inventory](ion_em_packages_manifest.json). Historical “current” labels below refer to the previous sparse-residual control, not the newly installed default.

## Model and scope

Only one original response per composition is extracted (median-density node, all 18 supported ions). The reference26 archive contains those 26 responses and the lightweight material/density descriptor directory. Target stopping/range splines, delta moments, cuts and residual fits are not copied into the prediction. An independent Geant4 material helper calculates the fixed0.05mm range-cut energy, electron-density coefficient and Sternheimer parameters from composition and density. These derived scalars are sampled at175 density positions solely for the current GPU compatibility interface; they are not learned from target EM responses. The cut helper uses Geant4 range-to-energy conversion, not a claim of a closed-form cut formula.

The response correction includes restricted Bethe cut terms, the density-effect difference, the proper alpha charge normalization, and the Bragg cut floor. Effective-charge/model/fluctuation identities remain composition-specific. Restricted DEDX splines are rebuilt; range is reintegrated with the Geant4 midpoint procedure and inverse range is reconstructed. The hard-delta rate is rebuilt on its target-cut Geant4 log grid; its first two moments are regenerated with the existing native-spectrum quadrature. Low-energy heavy-ion correction preserves the base corrected stopping scaled by density rather than separately scaling its subtractive correction. The actual Geant4 spline recurrence is used.

The user authorized this model as default on2026-10-05: the generated EMJOINT1/EMDMOMT2 files expand to the existing175-state GPU layout. It does not reduce GPU residency or establish faster transport. Electron birth/charged/photon packages, electron step controls, scattering, nuclear physics and dose precision remain unchanged. A later26-state GPU implementation would require its own correctness and performance validation, including continuous-density evaluation.

## Component and packaging checks

The raw26-base archive is206,602,854 bytes. Lossless ZIP encoding is106,313,510 bytes; every array is bitwise identical, and regeneration from the lossless archive produces byte-identical EM and delta files. These are basis-file sizes, not the total runtime data size. Independently recalculated cuts match native175-state cut values within5.19e-8 relative error. Pure density scaling alone reaches27.6% restricted-stopping and20.5% range error and is rejected.

The final candidate loads through the normal C++ parser and retains the shared-grid certificate. The finite-step check covers all175 materials and18ions, logarithmic0.0001–300MeV/u queries and0.001/0.1/0.5mm caps. Maximum sampled range change is0.120755%, stopping change0.205270%, and mean-loss change0.880381%; at>=10MeV/u these are0.044540%,0.038503%,0.321978%. These field/mean bounds are not pointwise dose guarantees. Delta moment relative error can be larger close to its zero-support threshold (2.77% maximum in the sampled full band,0.138% at>=10MeV/u). Full transport/dose checks are required. The rejected standard-spline first prototype and cut-floor refinement are documented in `failed.md`.

All four CT presets and a separate RT07575 head minibeam smoke pass two100,000-primary batches, including quality, energy, EM, electron completion/coverage and zero-overflow checks. The head smoke samples485 of487 positive spots due to rounding; formal full-plan allocations separately require all487.

## Full-plan acceptance

Broadbeam uses absolute3D local1%/1mm, search0.1mm, full-reference>=10% maximum mask. Both arms use the same current FP32-dose binary, complete spot source, statistics and seeds in reversed run order; primary and secondary ion fluctuations are ON for both particles. Ion cap is0.5mm. Proton pools25,063,000 histories per arm; carbon12,963,818.

| Case | Current vs TOPAS | Analytic vs TOPAS | Analytic vs current | Relative dose L1 |
|---|---:|---:|---:|---:|
| Proton CT broadbeam |99.881009%|99.882928%|100.000000%|0.026486%|
| Carbon CT broadbeam, fluctuations ON |99.795512%|99.795077%|99.999782%|0.004760%|

Both broadbeam candidates exceed the strict>99% gate. Carbon's current-control result also independently qualifies the new fluctuations-ON default; historical OFF acceptance is kept separate. All68 broadbeam batches pass runtime gates. Resident timing is not a resolved gain from this representation, because the GPU table layout is still expanded and clocks are unlocked.

RT07575 head proton minibeam completed the required0.1% of the4Gy full plan:236,086,480 primary,119batches of1,983,920, all487 positive spots. It uses the existing Urban/FP32 electronics, group32, ion cap0.1mm and absolute3D local3%/0.2mm, search0.02mm, BODY including GTV and reference>=10% maximum mask. The reused equal-history current-package control has matching seeds and spot allocations; all non-package/weight-path config fields were checked identical before each launch.

| Head minibeam0.1% | Current package | Analytic candidate |
|---|---:|---:|
| TOPAS local3%/0.2mm |98.937265%|98.956666%|
| GTV local3%/0.2mm |98.351719%|98.409441%|
| Mask dose-sum bias vs TOPAS |+0.168163%|+0.170090%|

Analytic versus current passes99.999461% (one failure among185,559 evaluation voxels), with relative doseL1=0.342630% and mask dose-sum change+0.001923%. Absolute local difference quantiles are median0.239721%,p95=1.279684%,p99=2.217505%,maximum7.798806%. Spatial gamma agreement is not a guarantee of <=1% differences in every voxel. The two smaller equal-statistics screens are preserved separately; they are not the0.1% formal result. All119 candidate full-plan batches and the487-spot campaign allocation checks pass; in total197 final-version batches (68broadbeam,119head,10smoke) pass runtime gates.

Both broadbeams pass their absolute TOPAS gates. Head minibeam remains below the strict>99% gate for both current and analytic packages. The candidate has high fidelity to current transport but is not qualified as an independently accepted universal CT default; the subsequent user-authorized default promotion does not change these accuracy results, and no attribution of the remaining TOPAS discrepancy solely to Monte Carlo noise is made. Carbon minibeam remains a separate lung-CT runtime smoke with dose acceptance deferred.

Timing totals for the head candidate are45.49 minutes, with resident throughput86992 primary/s; the reused current control is87294 primary/s. These are different-time, unlocked-clock observations, not a resolved speed comparison. The interface still expands175states on the GPU. Removing that expansion and measuring device-memory/performance changes is a separate implementation.

## Reproduction

Use fresh output paths. Build the existing GPU binary with `CARBON_DOSE_FP32=ON`, `CARBON_DOSE_FP64=OFF`.

```bash
python3 extensions/tools/unified_em/synthesize_analytic_density.py extract \
  --source data/archive/ion_em_native_20261005/unified_em_v1.bin \
  --output out/reproduction/composition_bases.npz --lossless
python3 extensions/tools/unified_em/synthesize_analytic_density.py material-inputs \
  --bases out/reproduction/composition_bases.npz \
  --output out/reproduction/material_inputs.txt
# Compile analytic_density_metadata.cpp against the extraction Geant4 installation.
# Example installation below is local; set paths for your own Geant4 11.3.2.
g++ -std=c++17 -O2 -I/software/geant4-11.3.2/include/Geant4 \
  extensions/tools/unified_em/analytic_density_metadata.cpp \
  -L/software/geant4-11.3.2/lib -Wl,-rpath,/software/geant4-11.3.2/lib \
  -lG4processes -lG4materials -lG4particles -lG4global -lG4intercoms \
  -o out/reproduction/metadata
LD_LIBRARY_PATH=/software/geant4-11.3.2/lib out/reproduction/metadata \
  out/reproduction/material_inputs.txt out/reproduction/metadata.txt
OPENBLAS_NUM_THREADS=1 python3 extensions/tools/unified_em/synthesize_analytic_density.py synthesize \
  --bases out/reproduction/composition_bases.npz \
  --metadata out/reproduction/metadata.txt \
  --output out/reproduction/candidate.bin --delta-output out/reproduction/delta.bin
python3 scripts/validate_em_analytic_density.py --phase broadbeam \
  --output out/reproduction/validation --em out/reproduction/candidate.bin \
  --delta out/reproduction/delta.bin
python3 scripts/validate_em_analytic_density.py --phase head-0p1 \
  --output out/reproduction/validation --em out/reproduction/candidate.bin \
  --delta out/reproduction/delta.bin
```

The head campaign requires local existing control artifacts and the actual optimized head-plan inputs. Raw evidence is under `out/em_analytic_density_20261005/`. [Machine-readable report](em_analytic_density_20261005.json).
