# Step 10 — Upload and index the primary C12 table on the GPU

## Objective

Ensure device memory contains the same 25 x energy table and retrieves the correct section/rate without layout or density errors.

## Layout contract

Choose and document one contiguous layout, preferably `[section][energy]` for P2. Define a single host/device index helper and static/runtime bounds checks. Device inputs include pointer, section count (=25), energy count, minimum energy, and inverse energy spacing or explicit interpolation contract.

At a CT point:

```text
section = CtGrid.material_id[voxel]
mass_rate = interpolate(device_table[section], E_MeVu)
Sigma_per_mm = density_g_cm3 * mass_rate_per_mm_at_1g_cm3
```

No additional reference-density division or four-class conversion is allowed.

## Required diagnostics/tests

- Host/device lookup equivalence for all 25 sections at exact nodes, midpoints, and endpoints.
- Sentinel table test where each cell encodes section and energy index, catching transposition.
- Density scaling at 0.001, 0.3, 1.0, and 2.0 g/cm3.
- Invalid section/pointer/dimension must fail before kernel launch, not index arbitrary memory.
- Log table dimensions, byte count, source SHA256, and mode once per run.

Run SYCL/GPU tests locally outside sandbox on RTX 2080 Ti. Inspect allocation/free paths for leaks and zero-size allocations.

## Do not change

Do not touch CINEL final-state selection, water target fractions, secondary kernel, or species behavior.

## Acceptance

Every device lookup matches host within `1e-6` relative (or exact for sentinel values), compute-sanitizer/device diagnostics show no invalid access, and frozen water regression passes.
