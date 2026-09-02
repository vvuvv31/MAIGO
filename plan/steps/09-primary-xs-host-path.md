# Step 09 — Make the 25-section primary XS host path strict

## Objective

Wire the validated C12 table through the existing configuration/loader/resampling design with explicit units and fail-fast behavior.

## Required implementation

Use the existing key `ct_schneider_cross_section_file`; do not add an alias or second CT XS configuration system. On CT + Schneider primary nuclear mode:

1. Resolve the path relative to the config consistently with other input files.
2. Load using `CrossSectionTable::from_schneider_csv()`.
3. Require exactly 25 section tables, identical energy nodes, coverage of the configured transport range, finite/nonnegative values, and the exact column schema.
4. Resample each section to the transport energy grid with documented interpolation and endpoint policy. Production must fail if requested energy is outside validated coverage; clamping is not acceptable unless existing global transport bounds make it unreachable and this is asserted.
5. Store mass rate units as `per mm at 1 g/cm3`; density multiplication occurs exactly once at runtime.
6. If CT nuclear attenuation is requested without a valid Schneider table, fail during config/load. Do not fall back to water or four-class XS.

## Compatibility

Non-CT water behavior and explicit legacy CT compatibility configs may remain, but logs must state `legacy-four-class` versus `schneider-25`. Production Schneider config must never call `ct_material_class()` for nuclear rate indexing.

## Tests and acceptance

Test 25 valid columns, 24/26 columns, swapped header, section ID 24, out-of-range ID, non-monotonic energies, insufficient range, relative-path resolution, density unit sanity, and no-table fail-fast. Full CPU tests and frozen water regression pass.

## Commit intent

`feat(ct): load section-resolved primary C12 nuclear rates`
