# Physics tables (TOPAS / Geant4 alignment)

## Cross sections (C-12 inelastic in Water_75eV)

| File | Source |
|------|--------|
| `c12_inelastic_cross_sections_water_geant4_11_1_3.csv` | **Preferred.** Remote TOPAS 4.1.p1 / Geant4 11.1.p3 via `CarbonCrossSectionNtuple` |
| `c12_inelastic_cross_sections_water_geant4_11_3_2.csv` | Legacy (TOPAS 4.2.p3 / G4 11.3.2). Macro water Σ matches 11.1.3 to machine precision on this grid. |

Extraction:

```bash
# remote rebuild + extract
python validation/scripts/_remote_rebuild_extensions.py
python validation/scripts/_remote_cross_sections.py run
python validation/scripts/prepare_topas_cross_sections.py \
  --output-csv data/c12_inelastic_cross_sections_water_geant4_11_1_3.csv \
  --metadata data/c12_inelastic_cross_sections_water_geant4_11_1_3.metadata.json
```

At 200 MeV/u: Σ_water = 0.00474216 mm⁻¹, MFP = 210.874 mm.

## Stopping power (C-12 in Water_75eV)

| File | Source |
|------|--------|
| `stopping_power_water_geant4_11_1_3.csv` | **Preferred.** G4EmCalculator electronic dE/dx via `CarbonStoppingPowerNtuple` on remote TOPAS 4.1.p1 / G4 11.1.p3 |
| `stopping_power_water.csv` | Legacy development Bethe–Bloch + Hubert table (still used by older configs) |

Extraction:

```bash
python validation/scripts/_remote_stopping_power.py
python validation/scripts/prepare_topas_stopping_power.py
```

At 200 MeV/u: electronic dE/dx ≈ 16.127 MeV/mm (legacy table ≈ 16.098 MeV/mm, ~+0.18%).

Note: CSDA range column in the full table may be zero unless `/process/eLoss/CSDARange true` is enabled in TOPAS; GPU only needs dE/dx.

## Reaction / cascade packages

| Beam energy | Primary reaction package | Cascade package |
|-------------|--------------------------|-----------------|
| ≤ 200 MeV/u | `topas_200MeVu_cascade_aligned_primary_3d.bin` (0–200, 201 bins) | `topas_200MeVu_cascade_100k_3d.bin` |
| > 200 MeV/u | `topas_400MeVu_cascade_aligned_primary_3d.bin` (0–400, 401 bins) | `topas_400MeVu_cascade_100k_3d.bin` |

Neutral (mode D): `topas_200MeVu_neutral_development.bin` (from 200 MeV/u 100k n-tuple; n/γ spectra cover high-E secondaries as well).

## Version policy

- Validation reference host: TOPAS **4.1.p1** / Geant4 **11.1.p3** at `v@192.168.31.5`.
- Prefer `*_geant4_11_1_3.*` tables for any new multi-energy work.
- Do not mix package tables from different Geant4 major/minor without documenting it.
