# Frozen C12-water MCS candidate, 2026-09-19

This record freezes the first interval-constrained `fermi_eyges_tail` candidate.
It is a research option only; formal 150/250/300 MeV/u configurations retain
`legacy_highland`.

## Parameters

- `core_scattering_energy_MeV = 9.9`
- `tail_rate_per_mm = 0.0025`
- `tail_scattering_energy_MeV = 2.4`
- `minibeam_water_primary_mcs_max_segment_mm = 0.1`
- C12 in native water only; fragments, Copper, stopping, and electron response
  are unchanged.
- The selected model replaces the legacy per-step Highland, low-energy scale,
  and synthetic tail. It does not stack with the legacy `0.20/0.90` settings.

Builds use `CARBON_DOSE_FP32=ON`, `CARBON_DOSE_FP64=OFF`, and forced-disabled
integrity checks. Both `CARBON_ENABLE_MINIBEAM=ON` and `OFF` CUDA SYCL builds
completed after the conditional-compilation repair.

## SHA-256 manifest

```text
bd1a41d1493471124fd4757e49e7fddc6b2930288051275bd7ba62b0c0d28590  src/detail/sycl_device_math.inc
dfd36232505406304b0492a898287ec1da3d8f2194c9f508a6ff4e4669215e66  src/transport_sycl.cpp
3417aea3402c51b3c5eb7b523e0ac93da37317433cdb096a614be911045dfc8b  /mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/gpu_fe_candidate/config.yaml
8c466f276ae2abe6240f9cc85d134b102f66b335741fd5c59f5ff167dea0c6ed  /mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/input/water_entrance_primary_c12.phsp
0afd27b7a1fb004ea26f180260c8c262a4ee9ad0964931857d782ebe556a7e85  /mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/input/water_entrance_primary_c12_gpu.csv
f03b6e3a0d6c0648ece766dd8bd4e70d5d71352b3b7151ff84edcb8f5f24207c  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/dose.bin
589b45fd3c1c6594e5ea3b14efc028597f8960f9ca2cb50334e8f5325c6ec48b  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/output/primary_040.phsp
c91425616c65f5293733a26973386ab1175d0536791248ed083a970615b0b745  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/output/primary_060.phsp
d3634c910932646aaec6354d711db6e224b10f26ac16c195b67e873665be5e11  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/output/primary_080.phsp
edb6f65f6751b05ff51ac0329cd1a69bd6033fa144c4e749c8f73aad5522d47c  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/output/primary_100.phsp
660c39352b7dd7cbc78775536c4a1432b3fce648484a6cf78d064ea1dd789435  /mnt/sda/wuwei/minibeam_water_planes_e250_em12800k/topas_6529/output/primary_120.phsp
```

The source hashes describe the candidate at the time of this freeze. Later
documentation-only changes do not alter them; any physics or transport change
requires a new freeze record.

## Regression outputs

- Corrected bridge diagnostics and 64-replicate track-block bootstrap:
  `/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/phase_comparison_fe_candidate_bridge_bootstrap/`
- Dose comparison after the diagnostic repair:
  `/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/dose_comparison_fe_candidate_bridge/`
- Independent sampler validation:
  `/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/water_mcs_sampler_validation.json`
- Frozen-parameter 150/250/300 MeV/u full-chain legacy/candidate screening:
  `/mnt/sda/wuwei/minibeam_water_fullchain_regression_20260919/`

The last three propagation intervals reuse tracks from the 40--60 mm fitting
population. They were not used to tune the constants, but they are correlated
propagation checks rather than statistically independent samples.

The 256k full-chain screening did not justify promotion. The candidate changes
2-D L1 from `17.672/18.197/17.224%` to `17.923/18.009/17.227%` at
150/250/300 MeV/u; lateral L1 changes from `7.287/5.598/5.327%` to
`7.078/5.296/5.383%`. Thus the 250 MeV/u gain partially transfers, while 150
is mixed and 300 is neutral to slightly worse. Formal configs were restored to
the default legacy model after these A/B runs.

## Independent 150 and 300 MeV/u water replay

The retry condition for an independent multi-energy water check was exercised
without tuning the frozen constants. Existing full-chain water-entrance phase
spaces supplied `26,425` parent-0 C12 tracks at 150 MeV/u and `34,557` at
300 MeV/u. Each entrance set was replayed through pure-EM water in both engines.
One local Slurm job (`6603`, 192 threads, 60 GB request, 6.1 GB peak RSS) scored
total/electron/non-electron dose and every requested C12 plane in a single run.
The cancelled predecessor `6602` never started; its 160 GB request could not be
scheduled while the node had only about 70 GB unallocated.

The candidate and legacy GPU paths used identical entrance particles and seeds.
No stopping, Copper, fragment, or electron-response parameter was changed. GPU
energy-balance errors were `1.00e-9` and `3.86e-9`; nuclear interactions and
queue overflows were zero.

Candidate interval ratios, GPU/TOPAS:

| MeV/u | interval (mm) | angle variance | displacement variance | covariance | q68 | q99 | q99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 10--20 | 0.997 | 1.003 | 1.005 | 1.013 | 0.996 | 0.864 |
| 150 | 20--30 | 0.995 | 1.024 | 1.024 | 1.017 | 1.003 | 0.906 |
| 150 | 30--40 | 1.013 | 1.015 | 1.002 | 1.010 | 0.942 | 1.000 |
| 150 | 40--50 | 0.947 | 0.963 | 0.961 | 0.989 | 0.967 | 0.870 |
| 300 | 20--40 | 1.045 | 1.009 | 1.023 | 1.026 | 1.028 | 1.001 |
| 300 | 40--60 | 0.990 | 1.017 | 1.020 | 1.024 | 0.993 | 1.027 |
| 300 | 60--80 | 0.999 | 1.029 | 1.015 | 1.019 | 0.984 | 1.004 |
| 300 | 80--100 | 0.997 | 1.019 | 1.016 | 1.023 | 0.955 | 0.972 |
| 300 | 100--120 | 1.224 | 1.020 | 1.022 | 1.024 | 0.993 | 0.898 |
| 300 | 120--140 | 0.947 | 0.997 | 1.001 | 1.016 | 0.975 | 1.055 |
| 300 | 140--160 | 1.001 | 1.012 | 1.001 | 1.006 | 0.960 | 0.930 |

The isolated `1.224` variance ratio at 100--120 mm is not accompanied by a
q99 excess; it is driven by very sparse extreme events, especially the
80--100 MeV/u group. It is not evidence for increasing the global core. With
64 stable-identity block-bootstrap replicates, the aggregate q99.9 confidence
intervals overlap at every interval. At 150 MeV/u the q99.9 point estimates are
mostly low, but only about 19--25k tracks remain per interval and the outer
0.1% contains roughly 19--25 tracks.

The legacy comparison shows that the structural replacement, rather than the
common entrance alone, produces the agreement. Legacy angle-variance ratios at
150 MeV/u fall from `0.672` to `0.221` over 10--50 mm; at 300 MeV/u they fall
to `0.488` in the final interval. Candidate ratios stay near unity except for
the sparse-tail fluctuation above. Legacy displacement and covariance show the
same cumulative deficit, while its q99.9 is generally excessive.

Absolute total-dose ratios are `1.000009` and `1.000024`, IDD L1 is
`0.3671%` and `0.2696%`, and the TOPAS/GPU Bragg bins are `51.625/51.625 mm`
and `169.375/169.125 mm`. The candidate fixed-region dose ratios are more useful
than local extrema at this particle count: all sampled peak/shoulder/valley
integrals are within `2.8%` of TOPAS. The 150 MeV/u 50 mm valley ratio improves
from legacy `0.869` to `0.994`; at 300 MeV/u the candidate valley ratios span
`0.976--1.019`. The apparent 12--16% 2-D L1 is dominated by sparse-voxel noise
from only 26--35k input histories and is not a precision acceptance metric.

TOPAS electron-carrier dose is `5.33%` and `8.00%` of the total at 150 and
300 MeV/u; carrier partitions close with L1 `2.68e-8` and `2.75e-8`. Together
with the matched C12 fixed-region fluence and dose, these results keep primary
C12 water scattering ahead of electron-response changes for this candidate.

Outputs are under
`/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/{e150,e300}/`.
At this stage the candidate was supported as a multi-energy water-only research
model but was not yet promoted: the earlier 256k full-chain A/B was mixed and
these small replay samples could not resolve an energy-dependent q99.9 kernel.
The later high-statistics acceptance below supersedes that provisional status;
the constants `9.9/0.0025/2.4` remain frozen.

Additional immutable inputs:

```text
0c29fad8792c6a2f3e373612d9ce673ece001149226b40536bd71ff738f7a8e2  config/beam_minibeam_water_replay_e150_256k_fe_candidate.yaml
9c5e8fde9d008bd06c1a55cd3891e55c4c117ab495a7e4d404b7e0822f1798c5  config/beam_minibeam_water_replay_e300_256k_fe_candidate.yaml
bf0144893f80381013138f41c83c05c3cef6263767743560db8ad379792b702d  e150/input/water_entrance_primary_c12.phsp
9146760cef82e543a84ce5c8024270a8864ce3ce33fe5725d14a32fad131be44  e300/input/water_entrance_primary_c12.phsp
9b771e7acbbbbcf3d1acf6459b8998db6e36a5c6687b89c61b09d2b6e5394a55  e150/topas/dose.bin
71921911e1edcc9071ea66eb1fdd9257bead86536214bbbae5eba4d44dbc2820  e300/topas/dose.bin
```

## High-statistics upstream and full-chain acceptance

Two upstream tests removed the compensation concern that blocked promotion.
With both engines restricted to EM, 12.8M incident histories gave a normalized
three-dimensional `(slit residual, energy, outward angle)` shape TV of `0.647%`
at the water entrance. A separate 10,000,128-history full-physics comparison
gave `0.638%`, surviving-C12 yield ratio `1.00445`, energy mean/std ratios
`1.00104/0.99773`, and x-width ratio `1.00019`. Thus neither Copper EM/slit
transport nor nuclear survival selection can account for the downstream dose
residual. The remaining angular discrepancy is confined mainly to a small
wide-angle tail.

The frozen candidate was then run through the complete GPU chain and compared
with the existing 10M TOPAS references:

| MeV/u | total | 2-D L1 | IDD L1 | lateral L1 | Bragg PVDR | fixed Bragg peak/valley |
|---:|---:|---:|---:|---:|---:|:---:|
| 150 | 0.99905 | 3.393% | 0.848% | 1.725% | 1.032 | 1.004/0.988 |
| 250 | 0.98632 | 3.481% | 1.390% | 1.692% | 1.004 | 0.989/1.004 |
| 300 | 0.98186 | 3.551% | 1.827% | 1.951% | 0.956 | 0.987/1.004 |

The strict 250 MeV/u same-seed A/B reduced 2-D L1 from `3.610%` to
`3.481%` and changed the local Bragg peak/valley/PVDR ratios from
`0.995/1.035/0.962` to `1.004/1.001/1.004`, while total dose and IDD were
unchanged. Candidate wall time was `116.8 s` versus `108.4 s` for legacy.

This evidence satisfies the documented retry condition. The formal
150/250/300 validation configurations now select `fermi_eyges_tail` with a
0.1 mm internal maximum segment. The remaining entrance-valley and absolute
dose deficits are assigned to fragment/normalization isolation; they must not
be used to retune the frozen primary-scattering constants.
