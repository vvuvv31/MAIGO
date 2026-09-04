# Step 25 — Raw campaign validation

## Objective

Accept or pinpoint-rerun every raw node file before compilation.

## Inputs

- Step-24 raw files + per-job logs + manifest

## Outputs

- `schneider-secondary-v2/raw_validation.json` (per node: pass/fail +
  reason + raw SHA256)

## Rules (every node must verify)

- File exists and is non-empty; TOPAS exited normally, no FatalException.
- projectile/target/energy match the manifest entry.
- Accepted interaction event count meets the request.
- Product offsets/counts legal; all charged products have legal Z/A or an
  explicit classification.
- Event energy closure finite and within the established threshold.
- No NaN/Inf; provenance complete; raw SHA256 recorded.

## Acceptance

All planned nodes pass, or each failure has a tracked pinpoint rerun that
subsequently passes.

## Prohibitions

- Failed nodes may only be rerun at the same (projectile, target, energy);
  never silently dropped, never substituted with a neighboring target.

## Commit intent

`test(ct): validate v2 secondary raw campaign nodes`
