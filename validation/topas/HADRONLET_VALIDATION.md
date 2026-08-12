# HadronLET water validation

## Installed reference

The `myHadronLET` and `myHadronLET_Denominator` scorers were built from
`Villadslj/Topas-Extension` commit
`a667d80a55f78f9c46ee9fb7e2ed8042af015cbd` and installed in:

```text
build/opentopas-hadronlet-install/bin/topas
```

The build uses local OpenTOPAS 4.2.3, Geant4 11.3.2, and at most 40 compile
and simulation threads. The configured build and source directories are
`build/opentopas-hadronlet-build` and `/tmp/topas-extension-reference`.

The upstream dose-weighted scorer evaluates, per bin:

```text
LET_d = sum(Edep * (electronic dE/dx / density))
        / sum(Edep + kinetic energy of step-born electrons)
```

Only charged particles with atomic number `Z >= 1` contribute. The GPU does
not explicitly transport delta electrons. Instead, the
`CarbonDeltaElectronEnergy` diagnostic measures
`E_delta/(Edep+E_delta)` directly in the matching TOPAS/Geant4 physics list.
`prepare_topas_delta_fraction.py` smooths the extensive numerator and
denominator, maps depth to projectile energy, and writes an energy-dependent
fraction table. This correction is used only by the LET numerator; it does not
change dose transport or the existing electronic build-up model.

The GPU accumulates all four raw LET moments in FP64 even in FP32-dose builds.
This is required for history-count convergence: in distal bins, many small
step deposits otherwise fall below the ULP of a million-scale FP32
denominator.

## Switches and commands

YAML accepts either `scorerLET: true|false` or
`enable_let_scoring: true|false`. The command line can override it:

```bash
build/oneapi-release/carbon_mc \
  --config config/beam_200MeVu_letd.yaml \
  --device cuda --histories 500000 \
  --scorer-let --let-output out/letd_water/gpu_letd.csv
```

The benchmark script selects the scorer state itself and supports `off`, `on`,
or alternating `ab`:

```bash
python3 validation/scripts/benchmark_gpu_let_scorer.py \
  --mode ab --histories 500000 --repetitions 3
```

Run TOPAS (40 threads) and compare:

```bash
validation/topas/run_hadronlet_topas.sh \
  carbon_200MeVu_water_letd_smoke.txt
python3 validation/scripts/compare_water_letd.py \
  --gpu out/letd_water/benchmark_500k/gpu_letd_on_2.csv
```

Use `carbon_200MeVu_water_letd.txt` instead of the smoke file for 100k TOPAS
histories.

Generate or refresh the delta-electron table:

```bash
validation/topas/run_topas.sh delta-electron-smoke
python3 validation/scripts/prepare_topas_delta_fraction.py
```

The GPU validation YAML points to both Geant4 11.3.2 tables:

```yaml
stopping_power_file: data/stopping_power_water_geant4_11_3_2.csv
let_delta_electron_fraction_file: data/let_delta_electron_fraction_water_geant4_11_3_2.csv
```

## Initial results (2026-07-26)

The 10k-history TOPAS smoke run completed successfully in 92.0 s execution
time on 40 threads and produced both primary-C12 and all-hadron LET_d.

For the 500k-history CUDA A/B (three repetitions), median throughput was:

- scorer off: 45,807 histories/s
- scorer on: 44,953 histories/s
- throughput change: -1.87%
- equivalent runtime increase: +1.90%

Thus LET scoring is measurably slower, but only by about 2% in this
transport/cascade-heavy case.

Against the initial 10k TOPAS reference, using bins whose corresponding TOPAS
energy deposition exceeds 1% of maximum:

- primary C-12 median absolute relative LET_d difference: 7.92%
- all-hadron median absolute relative LET_d difference: 7.94%
- primary LET_d peak depth: TOPAS 87.75 mm, GPU 88.25 mm
- all-hadron LET_d peak depth: TOPAS 87.25 mm, GPU 87.75 mm

The comparison is an initial validation, not a final convergence result:
TOPAS has only 10k histories, and the GPU delta-electron
term is a proxy rather than explicit electron production. In addition, the
first GPU implementation scores continuous charged-particle loss only, while
TOPAS `GetTotalEnergyDeposit()` on a positive-length ion step can also contain
local energy released by a nuclear interaction. These effects must be
separated before tuning the underlying LET model.

## Improved results (2026-07-26)

The delta-electron diagnostic was rerun with 10k histories and 40 TOPAS
threads. The measured fraction is about 0.1102 at 200 MeV/u; the earlier
linear electronic-build-up proxy supplied only 0.03 at the same energy. The
retired stopping-power comparison showed no table-driven entrance bias, but
publication inputs now use only the Geant4 11.3.2 extract.

The high-accuracy water configuration additionally uses:

- `maximum_step_mm: 0.1`, matching the TOPAS phantom step limit;
- `maximum_relative_energy_loss: 0.001`;
- `straggling_scale: 1.0`;
- automatic secondary queue sizing, eliminating particle-count-dependent
  fragment overflow.

The formal 100k-history TOPAS HadronLET reference completed in 853.1 s on 40
threads. The 500k CUDA result with FP64 LET moments, compared on bins where
the corresponding TOPAS energy deposit exceeds 1% of its maximum, gives:

| Quantity | Initial median abs. rel. | Improved median abs. rel. | Improved MAE | Improved RMSE | Mean rel. bias |
|---|---:|---:|---:|---:|---:|
| Primary C-12 LET_d | 7.92% | **0.044%** | 0.0835 | 0.481 | +0.0003% |
| All-hadron LET_d | 7.94% | **2.79%** | 1.36 | 2.66 | +0.94% |

The primary result is stable between 100k and 500k GPU histories. For all
hadrons, the median 100k-to-500k difference on the TOPAS mask is 1.51%; its
remaining TOPAS discrepancy is mainly in the low-dose fragment tail beyond
the Bragg peak. From 0--80 mm, the regional median absolute differences are
0.93--1.82%.

Final comparison outputs:

```text
out/letd_water/comparison_fp64_moments_500k_topas100k/metrics.json
out/letd_water/comparison_fp64_moments_500k_topas100k/gpu_vs_topas_water_letd.png
```

The smaller high-accuracy steps dominate runtime. At 500k histories the
scorer-on transport took 58.8 s (8.50k histories/s). In a two-repeat 100k
A/B with FP64 moments, median throughput was 8.53k histories/s with the scorer
off and 8.08k histories/s with it on: a 5.55% runtime increase.

## Same-version SOBP, 3D and energy sweep (2026-07-26)

The reaction and cascade samplers were regenerated with the same TOPAS
4.2.p3 / Geant4 11.3.2 installation used by HadronLET. The 400 MeV/u,
100k-history reference covers the full 0--400 MeV/u slowing-down interval:

- 209,326 compiled multi-projectile cascade interactions and 1,673,741 products;
- 73,729 primary C-12 reactions and 723,244 correlated products.

For the 21-layer, 5--10 cm SOBP at 0.5 mm depth resolution (100k histories on
each backend), the high-dose platform comparison was:

| Quantity | Median abs. rel. | P95 abs. rel. | Mean rel. bias |
|---|---:|---:|---:|
| Primary C-12 LET_d | 1.03% | 3.32% | -0.76% |
| All-hadron LET_d | 1.16% | 3.58% | -0.93% |

TOPAS used 816.8 s on 40 threads; CUDA used 1.37 s. A true 50x50x50,
3 mm isotropic FP64-moment LET scorer was then added. CUDA required 2.56 s;
TOPAS required 852.0 s. In the irradiated 3x3 cm, 5--10 cm SOBP volume the
mean relative biases were only -0.59% (primary) and -0.67% (all hadrons), but
the voxel-wise median absolute differences were 5.87% and 5.52%. The latter
is dominated by TOPAS voxel statistics: 100k histories are divided among
1,600 high-dose voxels.

Species-resolved depth scoring identifies the residual fragment-model trend
inside the SOBP. Median absolute relative differences over 5--10 cm were:

| Species | Median abs. rel. | Mean rel. bias |
|---|---:|---:|
| Primary C-12 | 1.03% | -0.76% |
| Secondary carbon | 4.27% | -0.31% |
| Boron | 6.02% | -1.77% |
| Beryllium | 9.59% | -2.37% |
| Lithium | 18.87% | -11.86% |
| Helium | 8.04% | -7.75% |
| Hydrogen (p/d/t) | 15.01% | -15.03% |

Thus the remaining all-hadron error is not a primary-carbon stopping-power
problem. It is concentrated in light-fragment stopping/energy spectra, most
clearly Li and Z=1 fragments.

The 100/200/300/400 MeV/u water sweep used GPU 100k and TOPAS 50k histories.
Primary C-12 median absolute relative differences were 0.046%, 0.046%,
0.156% and 0.666%; all-hadron values were 1.32%, 4.07%, 12.36% and 7.80%.
The high-energy all-hadron degradation occurs mainly in the post-Bragg
fragment tail and is consistent with the species diagnosis.

Key outputs:

```text
out/letd_sobp/comparison_g4_11_3_2/
out/letd_sobp/3d/comparison/
out/letd_sobp/species_comparison_g4_11_3_2/
out/letd_energy_sweep/comparison_g4_11_3_2/
```

## Isotope-specific fragment stopping tables (2026-07-26)

`IonStoppingPowerNtuple` extracts deterministic Geant4 electronic stopping
powers for the 32 isotopes present in the same-version cascade package:
H-1/2/3, He-3/4/6/8, Li-6/7/8/9, Be-4/6/7/9/10, B-8/10/11/12,
C-9/10/11/12/13/14, N-12/14/15, O-15/16 and F-19. Each isotope is tabulated
on a uniform 0.1 MeV/u grid from 0.01 to 400.01 MeV/u in Water_75eV. The
low-energy extension is required for the Bragg-tail contribution of light
fragments; the former 1 MeV/u lower bound biased H/He LET low. The generated
GPU input is:

```text
data/ion_stopping_power_water_geant4_11_3_2.csv
```

Recreate it with:

```bash
validation/topas/run_hadronlet_topas.sh \
  carbon_ion_stopping_power_water_g4_11_3_2.txt
python3 validation/scripts/prepare_ion_stopping_power_tables.py \
  validation/topas/output/ion_stopping_power_water_g4_11_3_2.phsp \
  data/let_delta_electron_fraction_water_geant4_11_3_2.csv \
  data/ion_stopping_power_water_geant4_11_3_2.csv \
  --carbon-stopping-output \
    data/stopping_power_water_geant4_11_3_2.csv \
  --carbon-delta-output \
    data/let_delta_electron_fraction_water_geant4_11_3_2.csv
```

The YAML selection is explicit and defaults to the legacy path:

```yaml
# Fast legacy approximation (the default).
use_particle_specific_stopping_power: false
particle_stopping_power_file: data/ion_stopping_power_water_geant4_11_3_2.csv
```

Set the boolean to `true` to use isotope-specific tables. Missing isotopes
fall back safely to the C-12 effective-charge approximation. In heterogeneous
media the code applies the exact water ratio
`dE/dx(species)/dE/dx(C-12)` to the existing local C-12 material table. This
retains the CT/material calibration rather than incorrectly treating every
voxel as water.

The table also contains the raw Geant4 restricted/unrestricted stopping-power
ratio at the matching 0.05 mm electron production range. That deterministic
ratio underestimates the separately scored high-energy C-12 delta-electron
fraction (at 400 MeV/u: 0.1347 versus 0.1583). Therefore the runtime column
preserves the measured C-12 curve and applies only each isotope's relative
Geant4 correction; it does not silently replace the validated C-12 scorer
correction.

For the 100k-history SOBP, isotope tables changed the 5--10 cm all-hadron
median absolute relative difference from 1.179% to 1.121% (4.9% relative
improvement). MAE changed from 0.8298 to 0.8280 and RMSE from 1.0671 to
1.0641. Species-level changes were mixed:

| Species | C-12 scaling | Isotope tables | Relative change |
|---|---:|---:|---:|
| Boron | 6.02% | 5.44% | -9.7% |
| Beryllium | 9.59% | 8.96% | -6.6% |
| Lithium | 18.87% | 17.29% | -8.4% |
| Helium | 8.04% | 11.26% | +40.0% |
| Hydrogen | 15.01% | 19.53% | +30.1% |

The broad-energy all-hadron medians changed as follows:

| Energy | C-12 scaling | Isotope tables | Relative change |
|---|---:|---:|---:|
| 100 MeV/u | 1.317% | 1.263% | -4.1% |
| 200 MeV/u | 4.071% | 4.270% | +4.9% |
| 300 MeV/u | 12.362% | 11.411% | -7.7% |
| 400 MeV/u | 7.803% | 7.838% | +0.4% |

Thus exact electronic stopping tables are physically preferable, but they do
not by themselves remove the all-hadron error. The remaining H/He and
high-energy tail discrepancy is dominated by fragment production spectra,
transport/cascade correlations and scorer composition rather than the
effective-charge approximation.

Contrary to the initial expectation that more tables would be slower, an
interleaved five-repeat Titan RTX benchmark measured 65.6k histories/s for
C-12 scaling and 67.4k histories/s for isotope tables: **+2.7% throughput**
(about 2.6% less runtime). The lookup avoids two per-step exponential
effective-charge evaluations and costs only about 1 MiB extra device memory.
This result is hardware dependent; use
`validation/scripts/benchmark_particle_stopping_power.py` on another GPU.

Comparison and timing outputs:

```text
out/letd_sobp/particle_table_benchmark/
out/letd_energy_sweep/comparison_particle_tables_g4_11_3_2/
```

### Low-energy table and isotope diagnosis

The p/d/t/He-3/He-4 optional diagnostic is enabled only when
`light_isotope_let_output_file` is non-empty. It adds five pairs of FP64
moments without changing production output when disabled. A 1M-history GPU
SOBP and independent 10k-history TOPAS scorers showed that isotope
denominators and composition already agreed within about 3%; the mismatch
was in dose-weighted stopping power, not fragment yield.

Extending the tables below 1 MeV/u and scoring the locally deposited
sub-cutoff tail changed the SOBP-platform GPU/TOPAS LET ratios:

| Isotope | Old 1--400 MeV/u table | 0.01--400 MeV/u table |
|---|---:|---:|
| proton | 0.812 | 0.938 |
| deuteron | 0.837 | 0.991 |
| triton | 0.969 | 1.153 |
| He-3 | 0.997 | 1.061 |
| He-4 | 0.875 | 0.979 |

Triton and He-3 are smaller components and retain a production-spectrum
error; no case-specific scale was applied. For the total all-hadron LET over
the 50--100 mm SOBP, the median absolute relative difference improved from
about 1.21% to **0.785%**, with mean relative bias **-0.23%** and P95
**2.16%**. The 1M Titan RTX runtime increased from about 14.0 s to 15.45 s
(roughly 10%). Lowering the secondary transport cutoff from 0.1 to 0.01 MeV
did not improve these metrics and increased runtime to 16.66 s, so 0.1 MeV
remains the selected cutoff.

The broad monoenergetic comparison also improved, but its high-energy
fragment tail remains limited by the sampled cascade model:

| Energy | Previous all-hadron median error | Low-energy table |
|---|---:|---:|
| 100 MeV/u | 1.263% | 1.207% |
| 200 MeV/u | 4.270% | 3.719% |
| 300 MeV/u | 11.411% | 9.926% |
| 400 MeV/u | 7.838% | 7.245% |

Key outputs:

```text
out/letd_sobp/particle_table_1M/exact/lowenergy_compare/
out/letd_energy_sweep/comparison_particle_tables_lowenergy_g4_11_3_2/
```
