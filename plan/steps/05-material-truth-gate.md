# Step 05 — Compare MAIGO parsing against TOPAS truth

## Objective

Automate the section-by-section comparison and stop the workstream if HU mapping, density, or composition differs.

## Comparator

Add a repository script that consumes the TOPAS JSON and the Schneider source, independently invokes/reads the MAIGO parser result, and writes machine-readable JSON plus CSV. Required comparisons:

- HU -> section ID and interval.
- Density in g/cm3.
- Element name/Z/order.
- Mass fraction.
- Derived atom density using the same documented atomic mass as the record.
- Mean excitation energy and radiation length as observed reference values; do not force MAIGO to manufacture these until the corresponding transport steps.

## Fixed gates

```text
section mapping mismatch count = 0
density relative difference < 1e-5 for every record
mass-fraction absolute difference < 1e-6 for every element/section
mass-fraction row-sum error <= 1e-6
```

Atom-density differences must be reported. If atomic masses differ between the parser constants and Geant4, resolve the source/version mismatch explicitly; do not loosen composition gates or hide it by renormalizing.

## Acceptance

- Comparator has unit tests including a deliberately shifted HU edge and swapped element columns; both must fail.
- All fixed gates pass against the exact TOPAS truth file.
- Formal product `data/schneider/schneider_materials_geant4_11_3_2.json` and metadata contain hashes and versions from README.
