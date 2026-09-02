# Step 16 — Define and test CINEL03 element-target schema

## Objective

Generalize correlated final-state lookup from water-specific H/O to arbitrary Schneider target elements while keeping material rates separate from event payloads.

## Key contract

```text
event key = projectile_Z, projectile_A, target_element_Z, energy_node
material section is NOT part of the event key
```

Payload retains one complete correlated interaction: parent final status/energy/vector, all charged daughters with Z/A/KE/direction, neutron/gamma records per established policy, local process deposit, energy closure, actual target isotope A when available, and source event provenance.

## Versioning

Use a new magic/version (`CINEL03` or a formally bumped generic package). Never reinterpret existing CINEL02 bytes. Specify endianness, integer widths, offsets, bounds, target element registry, energy interpolation/selection semantics, empty cell behavior, and checksum.

## Separation

- Material-rate product determines total hazard and selected target element.
- CINEL03 provides a correlated event for that projectile/target/energy.
- Package has no Schneider section and no density.

## Required tests

Round-trip, deterministic serialization, corrupt/truncated offsets, unsupported version, missing target, duplicate key, invalid isotope, energy closure, parent status, and compatibility rejection. Missing target must hard-fail production; audit mode may increment a named counter and stop/kill according to an explicit non-production rule.

## Do not change

Do not migrate the frozen water production package in place, tune event yields, or add target aliases.

## Acceptance

Schema specification and tests are complete before any large campaign starts. A tiny synthetic package replays the exact correlated event on CPU and GPU.

## Commit intent

`feat(cinel): add versioned elemental-target event package`
