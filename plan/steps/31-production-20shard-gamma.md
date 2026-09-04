# Step 31 — 20-shard production Gamma

## Objective

Full-statistics production dose and Gamma after Step-30 passes.

## Inputs

- Step-30 validated configuration and D-stage physics

## Outputs

- 20 per-shard dose grids + summed aggregate + Gamma vs TOPAS + per-shard
  quality reports (all accepted = true)

## Rules

- Single-shard gates green first; any queue overflow invalidates the shard
  (reduce histories, rerun, merge only overflow-free shards).
- Same paired conditions and reporting set as Step 30.

## Acceptance

Aggregate Gamma reported with all shards accepted = true and zero overflow.

## Prohibitions

- No 20-shard run before Step-30 gates pass.

## Commit intent

`test(ct): 20-shard production Gamma on v2 package`
