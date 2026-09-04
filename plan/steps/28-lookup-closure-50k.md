# Step 28 — 50k lookup closure (generation=1)

## Objective

Prove zero lookup failures on real patient CT at small scale before any
dose comparison.

## Inputs

- Step-27 audited v2 package wired as `ct_schneider_secondary_cinel03_file`
- 50k-histories real-CT config (RT06423, generation=1)

## Outputs

- `energy_ledger.json` (named Schneider diagnostics) + `quality_report.json`

## Hard gates

- primary hazards = primary replayed
- secondary hazards = secondary replayed + secondary stopped-before-replay
- missing/gap/domain/empty = 0 (primary and secondary)
- lookup failure energy = 0
- unsupported projectile tracks = 0
- overflow = 0
- born strict closure (primary and secondary)
- accepted = true

## Acceptance

All gates pass on the 50k run; evidence JSONs archived with config + input
SHA256 values.

## Prohibitions

- No Gamma, no 20-shard, no threshold changes to make the gate pass.

## Commit intent

`test(ct): 50k secondary lookup closure on v2 package`
