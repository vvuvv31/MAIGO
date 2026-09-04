# Step 22 — Reachable demand manifest

## Objective

Produce the authoritative machine-readable list of secondary
(projectile Z/A, target Z, energy) campaign tasks required to close all
Schneider CT lookup failures observed in real diagnostic transport.

## Inputs

- `out/reachable_channel_audit.json` (169 channels, failure modes, demand proxy)
- `data/schneider/secondary_inelastic_rates_v1.bin` (+ metadata)
- `data/schneider/cinel03_secondary_targets.bin` (+ metadata, channels JSON)
- Raw campaign inventory under `/mnt/sda/wuwei/cinel03-campaigns/`
- Full-shard diagnostics (20-shard strict run: gap 7.64M, below 319k,
  above 51k, missing-target 175, stopped-before-replay 254k)

## Outputs

- `/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/campaign_manifest.json`
  (new version directory; never reuse v1 paths)
- Each task: projectile_Z/A, target_Z, energy_MeV_per_u, requested_events,
  priority, source_of_demand, expected_output, random_seed, TOPAS_config_hash

## Rules

1. Exact 13 projectiles x 13 targets = 169 channels, including p+H.
2. New missing p+H channel required; no alias to any other target.
3. Within each channel's reachable domain, adjacent nodes must satisfy
   gap <= 5.0 MeV/u (fixed threshold, never relaxed).
4. Low end must cover the true minimum collision energy (no clamp).
5. High end must cover the true maximum collision energy.
6. Keep all valid existing nodes; fill gaps only, never regenerate.
7. Priority ordered by real nuclear optical-depth demand.
8. Lowest-demand channels must still be covered eventually; aliasing to O
   or any other target is forbidden.

## Independent audit (before any TOPAS submission)

- missing_channels = 0
- max_gap_planned <= 5.0 on every channel
- planned_below_domain_demand = 0
- planned_above_domain_demand = 0

## Acceptance

Manifest + audit JSON on disk with all four audit equalities holding.

## Prohibitions

- Do not submit TOPAS jobs before the audit passes.
- Do not change the 5 MeV/u gap rule, physics parameters, or frozen
  water/CINEL02 data.
- Do not run Gamma or 20-shard production in this step.

## Commit intent

`docs(ct): add secondary coverage campaign plan and demand manifest`
