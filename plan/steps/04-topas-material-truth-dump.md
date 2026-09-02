# Step 04 — Build TOPAS Schneider material truth dump

## Objective

Ask the actual configured TOPAS/Geant4 stack what material it constructed for each Schneider section. This is the source of truth for density, composition-derived properties, and element atom densities.

## Extension

Implement `CarbonSchneiderMaterialDump` under `/home/wuwei/topas` source and build trees. Keep a repository copy of extension source/build instructions under `startup/extensions/`. Run after geometry/material initialization.

For each test voxel/material output stable JSON fields:

```text
schema_version, topas_version, geant4_version, physics_list
input_HU, expected_section, material_name
density_g_cm3, mean_excitation_energy_eV, radiation_length_g_cm2
elements[]: name, Z, atomic_mass_g_mol, mass_fraction, atom_density_per_cm3
```

Use numeric values from `G4Material`/material properties, not duplicated formulas from MAIGO.

## Synthetic geometry

Generate a tiny 3D CT phantom, never a 1D scorer:

- 25 representative-HU voxels, one per section.
- Additional boundary cases immediately below, exactly at, and immediately above each of the 24 internal HU edges. Since DICOM stores integer HU, use exact adjacent integers where `nextafter` is not representable and separately test floating parser boundaries in Step 03.
- Use adequate transverse extent and a 3D scorer if dose is enabled, though no dose result is required here.

## Operational rules

Build and run locally. TOPAS execution must be an `sbatch` job with logs/data under `/mnt/sda/wuwei`. Stay within the global 192-thread/160-GiB cap. Record transient `InvalidAccount`, wait 1--3 minutes, then recheck.

## Acceptance

- Exactly one unambiguous record exists per requested HU.
- Each record has 13 components in declared order (zero fractions may be retained) and finite physical properties.
- Repeated runs produce identical material values apart from timestamps.
- Output and metadata hashes are recorded; raw job files stay outside Git.

## Commit intent

`feat(topas): dump Schneider material physics`
