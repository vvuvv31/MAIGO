# Parent Z/A × energy × depth cascade package

Date: 2026-07-27

## Implementation

- Cascade binary format v3 adds `reference_depth_mm` to every correlated
  interaction.
- Events are reordered as:
  `projectile Z/A × 2 MeV/u energy bin × 10 mm depth bin × exact energy`.
- The device sampler locates a cell by binary search and preserves the full
  correlated final state.
- Empty cells expand to at most ±26 MeV/u and ±80 mm.
- v1/v2 packages remain readable and use the legacy energy-only sampler.
- YAML switch:

```yaml
cascade_condition_on_reference_depth: false
```

Absolute reference-depth conditioning is deliberately off by default because
the reference depth belongs to the 400 MeV/u water run.

Package:

```text
validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_conditioned_3d.bin
```

## 100k monoenergetic A/B

All-hadron LET_d, TOPAS dose ≥1% mask:

| Case | 300 MeV/u median \|rel\| | 400 MeV/u median \|rel\| |
|---|---:|---:|
| v2, energy only, 4 generations | 6.949% | 4.025% |
| v3, energy bins only | 6.865% | 4.139% |
| v3, energy × absolute depth | 7.195% | 3.968% |

The absolute-depth mode improves the same-source 400 MeV/u median slightly,
but worsens 300 MeV/u. This is direct evidence of source-energy overfitting,
so it is available only as a diagnostic switch and is not enabled in
production configurations.

The v3 energy-bin-only control is neutral overall: it slightly improves the
300 MeV/u median/mean/RMSE but worsens P95 and 400 MeV/u.

## SOBP generalization

For the 50–100 mm SOBP, v3 energy-bin-only gives:

- median absolute relative error: 0.852%;
- mean relative bias: −0.159%;
- P95 absolute relative error: 2.959%.

The v2 reference was 0.818%, −0.166%, and 3.073%, respectively. Differences
are small and mixed; v3 is therefore not made the default package.

## Reproduction

```bash
python3 validation/scripts/compile_cascade_package.py \
  --metadata validation/results/topas_400MeVu_cascade_g4_11_3_2_100k.metadata.json \
  --interactions validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz \
  --products validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz \
  --output validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_conditioned_3d.bin \
  --output-metadata validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_conditioned_3d.metadata.json

python3 validation/scripts/run_gpu_letd_energy_sweep.py \
  --binary build/oneapi-release/carbon_mc \
  --histories 100000 --energies 300 400 \
  --particle-specific --maximum-cascade-generations 4 \
  --cascade-package validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_conditioned_3d.bin \
  --condition-cascade-depth \
  --output-dir out/letd_energy_sweep/gpu_conditioned_cells_v2
```

## Conclusion

Parent Z/A and incident-energy conditioning are physically transferable.
Absolute depth is not: it encodes the reference source energy and geometry.
The remaining distal LET error should be pursued through projectile-resolved
interaction-rate and transported-parent population diagnostics, not by making
absolute-depth conditioning the production default.
