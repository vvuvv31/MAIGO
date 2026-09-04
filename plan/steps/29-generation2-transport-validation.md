# Step 29 — Generation-2 transport validation

## Objective

Prove tertiary products are queued, transported, and terminally closed on
the v2 package with all Step-28 gates still passing.

## Inputs

- Step-27 v2 package; 50k real-CT config with
  `cinel02_max_secondary_inelastic_generations = 2`

## Outputs

- `energy_ledger.json` + `quality_report.json` for the generation-2 run

## Hard gates

- secondary_charged_queued > 0 (tertiary tracks actually start)
- All Step-28 lookup/energy/born/overflow gates continue to pass
- accepted = true

## Overflow rule

If generation-2 overflows the secondary queue, reduce histories per shard
and rerun; results with lost particles are never accepted.

## Acceptance

Tertiary transport demonstrated with zero overflow and accepted = true.

## Prohibitions

- No Gamma until this step passes; no generation-3 scope creep in this step.

## Commit intent

`test(ct): generation-2 tertiary transport on v2 package`
