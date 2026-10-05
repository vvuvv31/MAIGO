# Ion EM composition bases and density corrections

RT07575 is a head case. This experiment factors the existing ion EM extraction into Water_75eV plus 25 Schneider composition bases, with sparse density corrections. It does not replace CT materials with scaled water and does not claim that all density dependence is linear.

## Format and physical scope

`EMFAC001` contains the original 175 material/density descriptors, 18 species, 3,150 record scalars, native stopping/range spline segments, 26 composition base tables, and sparse corrections. `EMDFAC01` factors the two delta-loss moments using the same bases. Each composition base is an existing median-density node. Leading density powers scale stopping and rates by density and range inversely; the remaining response uses multiplicative correction coefficients, or absolute coefficients where the base is zero. Coefficients interpolate in original energy-knot ordinal. Native energy grids, zero support, model switches, fluctuation flags, production-cut values and spline intervals remain exact. The standard GPU energy interpolation is unchanged.

The default representation reconstructs the original dense arrays once on the host. It reduces storage/download volume, not device memory or the number of GPU material nodes. The compiler still consumes the 175-node extraction to obtain corrections; it does not establish that a fresh extraction of only 26 materials can recover the same physics without additional density information. Electron birth and charged/photon response packages are unchanged.

## Component verification

EM plus delta data shrink from 1,593,660,068 to 302,266,988 bytes (81.033% smaller). EM is 264,095,640 bytes and delta is 38,171,348 bytes. Compilation takes 64.7 seconds on the local host. Across 336,338,275 node fields and 51,744,350 delta values, maximum relative reconstruction error is 0.005%. The 1,219,050 finite-step mean checks span all materials and species at stored/intermediate energies and 0.001/0.1/0.5 mm trial lengths: maximum mean change is 0.00621%, range change zero. These component bounds are not dose-error guarantees.

The additive residual encoder and the decoder arithmetic at exact correction knots were rejected and repaired; evidence and retry conditions are in `failed.md`.

## Full-plan validation protocol

Both arms use the same current binary and identical physics, seeds and whole-plan allocations; only the two ion EM paths change. GPU dose atomics are FP32, with FP64 scoring disabled. Two independent seeds reverse arm order. Audits require accepted quality, finite nonnegative dose, closed energy accounting, completed electron queues, zero overflow and no EM/electron domain failures.

Proton/carbon CT broadbeam use current RT07575 head presets and ion cap 0.5 mm. Each arm pools 25,063,000 proton or 12,963,818 carbon histories. Proton head minibeam uses current shared Urban/FP32 physics, group32, ion cap 0.1 mm, and the actual RT07575 head CT, source/collimator geometry and optimized weights. Its 487 positive spots come from the 870-spot optimized plan. Source support and largest-remainder allocations are explicitly checked. The initial screen pooled47,614,080 incident protons per arm (0.02017% of the4Gy reference). At the user-required0.1%, each arm now pools236,086,480 histories across119 batches of1,983,920. This is only48 protons above the nearest integer0.1% of the236,086,432,317.4-proton4Gy plan. The screen uses two12-batch seeds in reversed arm order; a third identical seed per arm supplies95 additional batches, factorized then native. This is a complete spot plan at reduced Monte Carlo histories, not a full-prescription simulation.

Gamma is absolute 3D local, trilinear spherical Cartesian search, with no fitted scaling or registration. Broadbeam uses 1%/1 mm, search 0.1 mm, all reference voxels >=10% maximum. Head minibeam uses 3%/0.2 mm, search 0.02 mm, BODY including GTV and reference >=10% maximum. The pass gate is strictly >99%. The minibeam reference is the per-fraction 4 Gy optimized `RT07575_minibeam_GPU/inputs/physical_dose.mhd`; the 16-fraction total is not mixed into normalization. Carbon minibeam is a separate lung CT runtime smoke; dose acceptance is deferred.

## Results and decision

| CT full plan | Local gamma criterion | Native vs TOPAS | Factorized vs TOPAS | Factorized vs native | Gate |
|---|---|---:|---:|---:|---|
| Proton broadbeam | 1% / 1 mm | 99.882208% | 99.881009% | 100.000000% | Passed |
| Carbon broadbeam | 1% / 1 mm | 99.786810% | 99.787680% | 100.000000% | Passed |
| RT07575 head proton minibeam,0.1% plan | 3% / 0.2 mm | 98.923254% | 98.937265% | 100.000000% | Neither arm passes TOPAS |

All68 broadbeam batches,238 head-minibeam batches,16 initial four-CT smoke batches and4 final-build head smoke batches pass runtime quality/energy/EM/electron/completion/zero-overflow gates. Carbon minibeam remains a runtime-only smoke. The maximum recorded physical relative energy residual is below 1e-4 in every batch.

For head minibeam at0.1%, GTV gamma is98.2106% native and98.3517% factorized. Relative L1 difference to native in the acceptance mask is0.22277%; summed-dose change is−0.0005867%. Absolute local differences have median0.1435%,95th percentile0.8651%,99th percentile1.5118%,maximum4.9478%. Thus the table error bound must not be interpreted as a pointwise dose bound: small table perturbations can alter subsequent stochastic histories. A100% gamma to native does not mean identical voxel doses.

The preliminary47,614,080-history screen gave95.7809% native and95.6865% factorized. Its equal-history two-seed pooled relative-noise RMS diagnostic was2.4693%/2.4694%. Increased histories improve gamma, but neither this trend nor the small native/factorized gap proves that noise explains all remaining TOPAS discrepancies. That two-seed diagnostic applies to the screen only, not the final236M result.

Proton/carbon broadbeam relative L1 differences to native are0.02280%/0.00332%, with99th-percentile local differences0.2878%/0.0442% and maxima21.618%/5.106%. Gamma permits spatial agreement and does not promise a bound on every voxel.

The representation passes component checks and both broadbeam gates, but **head proton minibeam is not formally accepted against TOPAS** even at the requested0.1% particle count. Both the native baseline and the candidate fall below the strictly>99% requirement. The user subsequently authorized this storage representation as the runtime default based on its fidelity to native transport. All eight presets now load the compressed pair at the stable `data/em/` paths; the native binaries and metadata are archived under `data/archive/ion_em_native_20261005/`. The TOPAS head accuracy gate remains failed, independently of this representation decision. The candidate agrees with matched native transport at the prescribed gamma criterion, which supports its representation fidelity but cannot substitute for independent TOPAS acceptance. Further baseline accuracy/statistics investigation is separate from this completed0.1% comparison.

| CT full plan | Native resident primary/s | Factorized resident primary/s |
|---|---:|---:|
| Proton broadbeam | 206,720 | 206,121 |
| Carbon broadbeam | 134,613 | 134,366 |
| Head proton minibeam,0.1% plan | 87,255 | 87,294 |

Broadbeam timing pools two reversed-order seeds. Head timing additionally includes the95-batch factorized-then-native supplement. Both exclude the first batch of each process. Each head arm takes about45.4 minutes including all119 batches and process setup; the supplemental run is about36.1 minutes per arm. Clocks are unlocked, so small timing changes are not resolved gains or regressions. Warm-cache native EM load/verification takes about0.72–0.76s; factorized about1.29–1.35s for broadbeam. Host reconstruction adds cost and leaves device upload/layout unchanged. There is no measured throughput acceleration from this prototype. Cold-storage behavior was not tested.

The [companion JSON](em_density_factorization_20261005.json) records package sizes, component bounds, all acceptance comparisons, runtime smokes, timing and a two-seed noise diagnostic. Raw evidence is under`out/em_density_factorization_20261005/`.


## Reproduction

From the repository root, with unused output paths:

```bash
OPENBLAS_NUM_THREADS=1 python3 extensions/tools/unified_em/factor_density.py \
  --output out/em_density_factorization_20261005/compact_em.bin \
  --delta-output out/em_density_factorization_20261005/compact_delta.bin

g++ -std=c++20 -O2 -DCARBON_EM_EXACT_INDEX=1 -Iinclude \
  tests/unified_em_density_factorization.cpp src/unified_em_package.cpp \
  src/delta_moments_data.cpp -o out/em_density_factorization_20261005/component_check

out/em_density_factorization_20261005/component_check \
  data/archive/ion_em_native_20261005/unified_em_v1.bin data/em/unified_em_v1.bin \
  data/archive/ion_em_native_20261005/unified_em_delta_moments_v2.bin data/em/unified_em_delta_moments_v2.bin

python3 scripts/validate_em_density_factorization.py --phase smoke
python3 scripts/validate_em_density_factorization.py --phase full
# Supplement the preliminary head comparison to0.1% of the full plan:
python3 scripts/validate_em_density_factorization.py --phase head-0p1
# Recompute reports from existing completed runs, without GPU transport:
python3 scripts/validate_em_density_factorization.py --phase analyze
```

Build `carbon_mc` with `CARBON_DOSE_FP32=ON` and `CARBON_DOSE_FP64=OFF` before GPU validation. The harness uses the local benchmark/reference artifacts named in its source. Output directories must be new; failed or completed evidence is not overwritten. Logs, batch manifests, runtime records and gamma reports are retained under `out/em_density_factorization_20261005/`.

Default promotion: stable runtime filenames now contain EMFAC001/EMDFAC01, with native sources archived. The regression harness explicitly points its native arm at the archive and its compressed arm at active data, avoiding mislabeled A/A comparisons. Default-preset smoke evidence is recorded in the companion JSON.
