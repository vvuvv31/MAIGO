# Step 20 — Expand material-dependent nuclear transport to secondary projectiles

## Objective

Only now unfreeze the CT secondary nuclear path and extend coverage in measured importance order.

## Priority calculation

Run representative synthetic and real-CT audit transport without enabling unsupported reactions. Accumulate optical depth demand keyed by projectile Z/A, target Z, section, and energy. Rank coverage by fraction of total nuclear optical depth, not intuition.

Implement in phases:

```text
A: B/Be/Li projectiles x highest-importance targets
B: He and Z=1 projectiles
C: remaining transportable charged isotopes x all 13 targets
```

Be6 remains excluded/TopasCompatKill exactly according to the frozen registry policy unless a separate approved physics change says otherwise.

## Data and runtime

Extract deterministic partial rates and correlated event campaigns using the same schemas and gates as C12. Do not reuse C12 rates for secondaries. Extend device tables with a documented projectile index/stride. Missing target/package coverage hard-fails production; no O alias.

## Scheduler and overflow

Use pilot-based local sbatch resource allocation within 192 CPUs/160 GiB. Shard aggressively. Any secondary overflow invalidates the shard; reduce histories and rerun before merge.

## Acceptance

- Intermediate releases cover >=99.9% of measured optical-depth demand and explicitly remain non-production if any registry key is absent.
- Final production products cover every transportable registry isotope, all 13 elements, and required energy nodes, except explicitly frozen kill species.
- `cinel03_secondary_targets.bin` metadata enumerates exact coverage.
- Major species integral reaches the final <2% target on the validation suite, with statistical uncertainty reported.

## Commit intent

`feat(ct): enable secondary-ion Schneider nuclear transport`
