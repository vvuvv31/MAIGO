# Step 18 — Select material target and replay CINEL03 on the GPU

## Objective

Use validated element partial rates to sample the interacting target, then replay the matching C12 correlated event.

## Runtime sequence

```text
voxel section + density + C12 energy
 -> interpolate 13 mass partial rates
 -> multiply all by density
 -> total = sum(partials)
 -> consume/sample nuclear optical depth with total
 -> categorical target from partial/total
 -> CINEL03 lookup(C12, target_Z, energy)
 -> replay one complete correlated event
```

Density cancels from target fractions but remains in total hazard; implement this without applying density twice. A compact sparse target representation is optional only after a dense reference implementation passes. If compacted, preserve exact declared target Z and validate omitted entries are exactly zero.

## Device layout

Document dimensions/strides. Prefer one shared uniform energy grid and precompiled total plus CDF/partials. Never perform 13 unrelated binary searches. CDF must end at 1 within numerical tolerance; zero-total cells are explicitly invalid/uncovered.

## Diagnostics

Count interactions by section, target Z, energy bin, lookup success/failure, replayed events, and unsupported target. Production aborts on unsupported target/package. Audit counters may continue only under an explicit non-production flag.

## Tests

- Synthetic one-hot and known-ratio target tables.
- Statistical categorical goodness-of-fit.
- Density changes total distance but not target fractions.
- Same oxygen library reused across multiple sections.
- Missing C/N/Ca package fails rather than selecting O.
- CPU/GPU lookup equivalence and boundary optical-depth tests.

## Scope guard

Apply to C12 primary only. Do not enable secondary projectile inelastic reactions yet.

## Acceptance

Rate and target distributions meet analytic/statistical expectations, no unsupported lookup occurs in complete C12 tests, and all P1--P3 gates remain passed.

## Commit intent

`feat(ct): sample Schneider elemental targets for C12 on GPU`
