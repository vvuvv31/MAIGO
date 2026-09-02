# Step 06 — Deterministically extract C12 material/element inelastic cross sections

## Objective

Extract interaction rates from the initialized TOPAS/Geant4 hadronic process without estimating cross sections from Monte Carlo counts.

## Scope

Only projectile C12, all 25 Schneider sections, all 13 declared target elements, and the transport-aligned grid spanning 0.5--430 MeV/u. No final-state events and no secondary projectiles.

## Extension behavior

Implement `CarbonIonInelasticXsDump` after physics initialization. Locate the exact inelastic process attached to C12 by process type/name and fail if zero or multiple ambiguous candidates exist. For every energy/section:

1. Query the full material macroscopic inelastic rate directly.
2. Query microscopic element cross section using the same projectile, energy, material context, and active data store/model selection.
3. Obtain element atom density from the actual `G4Material`.
4. Compute each partial macroscopic rate and their sum.
5. Emit direct total, partial sum, and mass-normalized rates at 1 g/cm3.

Required row fields:

```text
projectile_Z, projectile_A, energy_MeVu, total_energy_MeV
section_id, material_name, density_g_cm3
target_name, target_Z, target_atomic_mass_g_mol
microscopic_sigma_barn, atom_density_per_cm3
partial_macro_per_mm, summed_macro_per_mm, direct_material_macro_per_mm
mass_partial_per_mm_at_1g_cm3, mass_total_per_mm_at_1g_cm3
process_name, dataset/model identity if available
```

## Critical correctness checks

- Confirm Geant4 API energy argument is total kinetic energy, while file grid is MeV/u.
- Verify barn-to-mm and cm-to-mm conversions with a hand calculation in a unit test.
- Direct material rate and sum of element partial rates must agree to `1e-5` relative or the extractor fails.
- Do not query a pure-element material and assume its microscopic XS equals the material-context query without documenting Geant4 dataset behavior.
- Natural isotope handling must be recorded. If the API selects/isotope-averages internally, state that; do not invent a single A.

## Acceptance

Expected row cardinality is `number_of_energies * 25 * 13`; no duplicates, gaps, NaN, negative values, or unit ambiguity. All totals close, and two independently computed sample rows match by hand.

## Commit intent

`feat(topas): extract material-resolved C12 inelastic cross sections`
