# Step 12 — Add explicit primary-only CT validation mode and observables

## Objective

Measure attenuation without allowing fragmentation or secondary cascade to contaminate P2.

## Mode contract

Add one clearly named validation configuration mode, or compose existing flags if they already express it unambiguously:

```text
projectile = C12
Schneider CT enabled
section-resolved nuclear rate enabled
on first inelastic: score event, terminate primary
no correlated-event replay
no charged/neutral secondary creation or transport
```

This mode is not a production final-state model. Its output/log must say `primary-attenuation-only`. Do not reinterpret an unsupported CINEL lookup as this mode.

## Required observables

- Primary survival versus depth.
- First interaction depth, energy MeV/u, section ID, voxel density, and sampled target only if validated partial rates are used (P2 total-rate mode may omit target).
- Incident/terminated/escaped primary counts and conservation identity.
- Total primary dose and 3D voxel dose. IDD is produced offline by summing x/y bins.
- Bragg peak/range metric with a fixed algorithm.

## Tests

Zero XS gives 100% survival/no interactions. Huge XS interacts near entrance. Disabling nuclear physics reproduces frozen EM-only output. Primary-only mode produces zero secondary count and zero final-state replay count.

## Acceptance

Counters close exactly, output schema is tested, 3D-to-IDD reduction is reproducible, and secondary overflow cannot occur because no secondaries are generated.
