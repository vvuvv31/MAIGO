# Step 02 — Implement the Schneider material data model and parser

## Objective

Parse the complete TOPAS Schneider definition once on the host and represent 13 elements plus 25 fixed-composition sections without inflating every voxel.

## Required design

Add host-side types in `include/carbon/ct_grid.hpp` (names may match repository style but semantics must match):

```cpp
struct SchneiderElement { std::uint8_t z; double atomic_mass_g_mol; std::string name; };
struct SchneiderMaterialSection {
    int hu_min_inclusive;
    int hu_max_exclusive;
    std::array<double, 13> mass_fraction;
};
struct SchneiderMaterialTable {
    std::array<SchneiderElement, 13> elements;
    std::array<SchneiderMaterialSection, 25> sections;
};
```

Avoid hard-wiring atomic mass where Geant4 truth may differ without documenting the source. If the TOPAS file contains names but not Z/A, use a reviewed constant mapping for exactly the declared 13 names and reject unknown/duplicate names.

Extend parsing in `src/ct_grid.cpp` for:

- `SchneiderElements`
- `SchneiderMaterialsWeight1` through `25`
- `SchneiderHUToMaterialSections`
- existing density formula and `DensityCorrection`

Keep `CtGrid::density_g_per_cm3` and `material_id` per voxel. Do not copy the 25 x 13 table into voxels or change CCTG solely to duplicate composition.

## Mandatory validation

- Exactly 13 unique supported elements in file order.
- Exactly 26 strictly increasing HU boundaries and exactly 25 weight rows, with no duplicate/missing row number.
- Every weight finite and nonnegative.
- Each row sum within `1e-6` absolute of 1.0. Do not silently normalize malformed rows.
- Boundary semantics are `[low, high)` and HU 2996/out-of-domain behavior is explicit and tested.
- Density arrays/count fields agree and density is finite/positive over the supported HU domain.

## Do not change

- Do not derive nuclear rates here.
- Do not add per-voxel composition.
- Do not delete legacy CCTG compatibility or four-class helpers yet; isolate them and ensure the new production path can avoid them.
- Mean excitation energy and radiation length are not authoritative merely because they can be estimated from weights. They are validated against TOPAS in Steps 04--05.

## Acceptance

The parser loads the repository Schneider file into 13 elements/25 sections, reports exact boundaries and weights, rejects malformed variants, and leaves water behavior unchanged.

## Commit intent

`feat(ct): parse Schneider elemental material composition`
