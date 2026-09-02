# Step 07 — Compile and audit the Schneider C12 mass-rate table

## Objective

Convert raw long-form TOPAS output into a strict repository data product while preserving element partial rates for later target selection.

## Outputs

Produce:

1. `data/schneider/c12_schneider_inelastic_mass_xs.csv` with exactly:

```text
energy_MeV_per_u,
section_00_mass_xs_per_mm_at_1g_cm3,...,
section_24_mass_xs_per_mm_at_1g_cm3
```

2. A versioned binary/long-form audited partial-rate product for `[section][target][energy]`, with explicit dimensions and target Z order.
3. Sibling metadata files with every field required by the README.

## Compiler rules

- Input is the Step 06 raw manifest, never ad hoc concatenated CSVs.
- Enforce one and only one row per key.
- Sort by numeric energy, section, and declared target order.
- Verify section total equals sum of partials and compiled CSV equals raw mass total.
- Preserve zeros. Do not replace small elements with oxygen, drop sparse targets, smooth curves, or interpolate missing nodes.
- Refuse mixed TOPAS/Geant4/physics-list/Schneider hashes.
- Write atomically and calculate data SHA256 after final serialization.

## Tests

Test duplicate/missing key, shuffled input, unit mismatch, mixed provenance, negative value, wrong section count, wrong target set, and tampered metadata hash.

## Acceptance

Compiler output is deterministic byte-for-byte; recompile twice and compare SHA256. The existing `from_schneider_csv()` can parse the total table and sees exactly 25 tables with the intended energy grid.

## Commit intent

`feat(ct): compile Schneider primary-carbon nuclear rates`
