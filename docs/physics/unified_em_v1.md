# Unified water / Schneider ion EM package

All eight presets use the continuous-density ion EM runtime installed on 2026-10-05. It contains 26 compositions (Water_75eV and 25 Schneider compositions), 18 ion species and 468 records. The production GPU accepts only this representation: there is no 175-density-state directory, host reconstruction or density-pair interpolation. Electron birth/charged/photon data remain separate, with their existing 82-node material registry.

## Active and archived data

| Role | EM | Delta moments |
|---|---|---|
| Default runtime | data/em/unified_em_v1.bin (EMCNT001), 225,707,600 bytes | data/em/unified_em_delta_moments_v2.bin (EMDMOMT2), 30,750,944 bytes |
| Former expanded analytical interface | data/archive/ion_em_analytic_expanded_20261005/ | Same archive |
| Former sparse density residual representation | data/archive/ion_em_density_residual_20261005/ | Same archive |
| Original native extraction | data/archive/ion_em_native_20261005/ | Same archive |

The active pair totals 256,458,544 bytes, 83.908% less than the 1,593,660,068-byte expanded interface. Archives are explicit historical references, never default runtime fallbacks. The lossless generation source data/em/ion_em_26_composition_bases.npz has exactly 26 directory entries and occupies 106,313,320 bytes. It reproduces both active binaries byte-for-byte.

The 18 species are p, d, t, He3/4/6, Li6/7, Be6/7/9/10, B8/10/11 and C10/11/12. Reference extraction is TOPAS/Geant4 11.3.2, standard_opt4, with Water_75eV (I=75 eV). Energy coverage remains species/material dependent; missing supported-domain data rejects transport. Dedicated data handle additional recoil species.

## Continuous-density treatment

HU selects the Schneider composition and actual density using the existing TOPAS mapping; different compositions are not mixed. One native response spline per composition/species supplies the reference. Restricted stopping applies analytical production-cut and density-effect corrections. A 213,824-byte scalar grid (257 samples per composition) interpolates independently calculated cut and material parameters. These are scalar samples, not complete stopping/range/delta response states.

Restricted range is a degree-23 Chebyshev integral model in log density, evaluated on reference energy knots, with Newton inverse range. It is fitted from the analytical 26-base model, not the old 175 target responses. Load-time truncation bounds the sum of omitted coefficient magnitudes uniformly over the fitted density interval. Delta rate retains a fitted correction for the Geant4 threshold spline. Native-spectrum moments retain finite-size/spin veto: a bounded second-order expansion applies to small-form-factor spin-zero cases; other cases retain log Gauss-Legendre integration. This is a new approximation, not lossless density scaling.

Production GPU arithmetic and dose atomics use FP32. Ion loss fluctuations remain enabled for all eight presets. Nuclear/scattering and electron transport selections are unchanged. Ion maximum steps remain 0.5 mm broadbeam and 0.1 mm minibeam.

## Verification and regeneration

Run extensions/tools/unified_em/verify_unified_em_data.py and verify_delta_moments.py for the active files. Generate a fresh pair with extensions/tools/unified_em/build_continuous_density.py using the 26-composition source and independent Geant4 material helper; see the [reproduction commands and evidence](../em_continuous_density_20261005.md). The older factor_density.py and synthesize_analytic_density.py are historical offline generators with explicit archived sources, not the active runtime generator.

## Validation status

Continuous-model CT broadbeam local 1%/1 mm is 99.8834% proton and 99.7944% carbon, search 0.1 mm; both pass >99%. The final build, including the bounded moment fast path, repeats both candidate arms and passes against verified controls; all eight defaults also pass two-batch runtime smokes. The final RT07575 head 0.1% minibeam campaign is complete: continuous TOPAS local 3%/0.2 mm is 98.9033%, expanded control 98.9567%, search 0.02 mm; both remain below >99%. Matched expanded-GPU comparison is 99.6249%, relative dose L1 0.9014%. Accuracy limitations must remain explicit. Carbon-minibeam and independent water dose acceptance remain separate.

Screened throughput is lower than the former expanded interface. Storage reduction is established; a throughput gain is not. Full details and version-specific measurements are in the [continuous-density report](../em_continuous_density_20261005.md) and [methods](../../mm.md).
