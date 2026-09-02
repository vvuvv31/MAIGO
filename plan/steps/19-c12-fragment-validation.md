# Step 19 — Validate C12 fragmentation across Schneider materials

## Objective

Validate target-dependent C12 final states after primary attenuation, stopping, and MCS are already correct.

## Matrix

Use at least lung-like, soft tissue, trabecular-ish bone, and dense bone homogeneous slabs at 100, 200, and 300 MeV/u, then the 25-section staircase. TOPAS and GPU share source, geometry, Schneider file, physics provenance, scoring bins, and history normalization.

## Compare

- Primary survival and first-interaction depth/energy.
- Interaction target-element fractions versus partial-rate prediction and TOPAS truth.
- Integrated yield and 3D-dose/IDD for major species/isotopes.
- Fragment energy and angular distributions at production and selected depths.
- Total/local/escaped energy ledger and terminal categories.

## Fixed gates

Maintain prior primary gates. Additionally:

```text
target interaction mix agrees statistically with expected partial rates
major species integral relative difference < 3% for this phase
unsupported target/package lookup = 0
secondary overflow = 0 (only produced fragments are transported without secondary inelastic cascade)
energy ledger closes under the existing documented tolerance
```

The final <2% species goal is deferred until secondary projectile coverage. Do not weaken the 3% gate after seeing a material.

## Acceptance

Automated per-case and aggregate reports pass. Discrepancies are attributed to rate, target selection, event payload, or downstream EM transport using counters—not hidden by retuning water or global yield factors.

## Commit intent

`test(ct): validate C12 fragmentation in Schneider materials`
