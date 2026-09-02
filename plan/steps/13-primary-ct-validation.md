# Step 13 — Validate the C12 primary CT milestone

## Objective

Demonstrate that HU/material mapping, current stopping, voxel stepping, and section-resolved C12 reaction length agree with TOPAS before adding fragmentation.

## Phantom suite

1. Homogeneous lung-like, soft-tissue, and dense-bone Schneider slabs.
2. Five-material set adding air and trabecular-ish bone for range/dose checks.
3. A 25-section staircase with one sufficiently thick segment per section.
4. Boundary-stress variants with beam oblique to the voxel grid.

Run at least 100, 200, and 300 MeV/u where meaningful. TOPAS uses the same Schneider file, geometry, source definition, physics list, and 3D scoring grid.

## Metrics and fixed P2 gates

```text
primary survival integral relative difference < 1%
first-interaction-depth NRMSE < 2%
range/Bragg position difference < max(0.5 mm, one voxel size)
incident = survived + inelastic + other-terminal categories exactly
section mapping mismatch = 0
secondary/replay count = 0
overflow count = 0
```

Report total primary dose, but if stopping is still approximate, do not claim final CT dose validation; classify discrepancies for Steps 14--15.

## Reproducibility

Shard high statistics. Store raw output under `/mnt/sda`, manifest every shard, merge counts/dose sums before normalization, and reject overflowed/incomplete shards.

## Acceptance

An automated report exits zero only when every case meets every fixed gate. Mark P2 achieved in README. Do not start CINEL03 here.

## Commit intent

`test(ct): validate Schneider primary C12 attenuation`
