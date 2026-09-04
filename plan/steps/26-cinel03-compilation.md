# Step 26 — CINEL03 compilation (versioned v2 package)

## Objective

Compile validated raw events into a new versioned CINEL03 package without
touching v1 artifacts.

## Inputs

- Step-25 accepted raw files + validation report

## Outputs (new files only)

- `data/schneider/cinel03_secondary_targets_v2.bin`
- `data/schneider/cinel03_secondary_targets_v2.metadata.json`
- `data/schneider/cinel03_secondary_targets_v2.channels.json`

## Compiler rules

- Sort by (projectile_Z, projectile_A, target_Z, energy).
- Reject ambiguous duplicates, empty nodes, illegal offsets/counts.
- Compute energy_min/max/max_gap from real nodes; forging a domain from
  configuration is forbidden.
- No target alias under any circumstance.
- Record every raw input SHA256, compiler commit, and full command line.
- Write to a temporary path; atomically rename only after all checks pass.

## Acceptance

v2 package loads in the existing reader; channel inventory equals the union
of kept v1 nodes plus validated new nodes.

## Prohibitions

- Do not overwrite `cinel03_secondary_targets.bin` or its metadata.
- Do not change the file schema version silently; bump formally if changed.
- Do not touch frozen water/CINEL02 packages.

## Commit intent

`feat(cinel): compile versioned v2 secondary event package`
