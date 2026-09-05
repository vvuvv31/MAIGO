# Path A: FRED / paper carbon physics

Goal: electromagnetic transport aligned with FRED 3.76, nuclear fragmentation
from De Simoni et al. (the `fred_paper` sampler), benchmarked against
FLUKA/FRED water tanks — not Geant4 INCLXX packages.

## What this tree implements now

| Process | Implementation |
|---|---|
| Inelastic XS (water) | Paper C–C fit × Kox(Np,Nt)/Kox(C,C); H from ICRU. Built in `CrossSectionTable::from_fred_paper_water`. |
| Target sampling | P(H)=2σ_H/(2σ_H+σ_O) |
| Fragment yields / kinematics | Existing Table 1 + Eq. 12 2D PDF in `sycl_inelastic_device.inc` |
| Nuclear elastic | C-12 on free proton, isotropic CoM (paper Sec. 4). Elastic σ_H is a therapy-range placeholder until ENDF is imported. |
| Energy straggling | `vavilov_landau`: Landau/Moyal thin, Gaussian thick (not the packed FRED LUT). |
| MCS | Highland 13.6 MeV × `multiple_scattering_scale` (1.40 ≈ 14.1/13.6 × f_mcs). 2GR tables exist in FRED `libFred.data` but are not wired yet. |

Config: [`config/beam_200MeVu_fred_paper.yaml`](../config/beam_200MeVu_fred_paper.yaml)

```yaml
nuclear_model: fred_paper
enable_nuclear_elastic: true
energy_straggling_model: vavilov_landau
multiple_scattering_scale: 1.40
```

`nuclear_model: fred_paper` ignores the Geant4 macroscopic CSV and builds the
paper water table at the configured density.

## Event libraries (`libEvents_C12_*`)

FRED 3.76 loads optional correlated event files at runtime
(`fred::loadEventLibraries()`, `libEvents_C12_H1.dat`, `_C12.dat`, `_O16.dat`).
Those files are **not** inside `libFred.data` (that archive is HU, dE/dx,
Vavilov LUT, MCS 2GR, activation, nuc text). GPU code falls back to
`SamplFrag` / `SamplETheta` (the same 2D PDF as the paper) when the library
offset is zero.

Until those `.dat` files are obtained from a full FRED data install, Path A
keeps the analytic paper sampler.

## TOPAS 提取（进行中）

脚本与说明：[`startup/path_a_extract/README.md`](../startup/path_a_extract/README.md)。  
数据根目录：`/mnt/sda/wuwei/path_a_topas_extract/`（`sbatch` 作业 206/207/208）。

- **206 `c12_fluc`**：C-12 水 EM-only 涨落，32 个 (E, xρ) 点，50k/点。  
- **207 `c12_el_lib`**：H1/C12/O16 弹性 XS（0–400 MeV/u）+ 95/200/300/400 事件（50 万/点）。100 MeV/u 的 H1/O16 另见已完成的 `/mnt/sda/wuwei/c12_h1o16_elastic_100/`。  
- **208 `c12_thin_frag`**：1 mm 薄靶 INCLXX 相关末态，H1/C12/O16 × 95/200/300/400，1M/点。95 MeV/u H1 已得到约 1.15×10^5 条 reaction 记录。

完成后已编入仓库：

- `data/packages/c12_G4_WATER_fluctuation_50k.csv`（32 点 × 50k，inverse-CDF）
- `data/c12_elastic_h1_water_topas_11_3_2.csv` 与 `include/carbon/detail/c12_elastic_h_xs.hpp`
- `data/packages/c12_{H1,C12,O16}_95MeVu_events.bin`（薄靶 INCLXX 相关末态，GPU 抽 H/O）

## Still missing vs FRED 3.76

1. Import `libEvents_C12_{H1,C12,O16}` when available; use them for those
   targets and keep Table 1 for N/P/Ca.
2. Unpack FRED `Vavilov_v01.lut` (format is a packed blob, not a plain text
   LUT in the 3.76 archive) and sample it instead of the analytic mix.
3. 2GR MCS (`data/mcs/*_2GR.txt` in `libFred.data`) plus lateral displacement.
4. ENDF C+p elastic angular/energy tables (replace the placeholder σ_el).
5. Homogeneous-water vs FRED/FLUKA IDD and halo at 100–400 MeV/u.

## What not to mix in

Do not enable INCLXX reaction packages, `dose_output_scale` ≠ 1, or
energy-dependent yield knots on this path. Paper mode already copies 95 MeV/u
Table 1 at all energies (`fill_energy_dependent_inclusive_weights`).
