# Step 08 — Independently validate extracted XS with TOPAS thin slabs

## Objective

Validate deterministic process queries against real transport attenuation, without using the MC fit as the production XS source.

## Matrix

Run C12 in at least lung-like, soft-tissue, and dense-bone Schneider materials at 100, 200, and 300 MeV/u. Use homogeneous 3D slab geometry and a 3D scorer; derive any depth curve by transverse summation.

## Measurement

Record primary histories entering the slab, first inelastic interaction depth/process/energy, and surviving unreacted primaries at several depths. Fit or directly compare:

```text
N(x) = N0 exp(-integral Sigma(E(x), material) dx)
```

Do not blindly use constant `Sigma(E0)` if ionization loss changes energy materially. Either make the slab thin enough with a quantified energy-change bound or integrate the extracted energy-dependent rate along the independently observed/known energy profile.

## Statistics and sharding

Precompute histories needed for <=0.5% relative statistical uncertainty where practical. Split jobs; allocate CPU/RAM proportional to expected runtime while respecting 192 threads/160 GiB total. On overflow, reduce shard size and rerun affected shards. Merge counts and exposure, never average fitted Sigma values without weights.

## Fixed gate

For every matrix point:

```text
direct Geant4 XS prediction lies within MC statistical 2 sigma
absolute systematic relative difference < 1%
first-interaction process contamination is reported and excluded by explicit rule
overflow count = 0
```

## Acceptance

A script regenerates tables/plots from raw shard manifests, reports confidence intervals and the above gates, and exits nonzero on any failure. Visual agreement alone is not sufficient.
