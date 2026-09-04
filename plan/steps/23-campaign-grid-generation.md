# Step 23 — Campaign grid generation

## Objective

Convert the Step-22 manifest into per-channel energy node grids that are
dense where data is sparse and lean where data already exists.

## Inputs

- Step-22 `campaign_manifest.json`
- Existing node lists per channel (`*.channels.json`)

## Outputs

- Per-channel grid file under `schneider-secondary-v2/grids/`
  (one JSON per channel: old_nodes, inserted_nodes, final_nodes,
  max_gap, observed_query_min/max, optical_depth_fraction)

## Rules

1. Keep every original node.
2. For each adjacent pair (E0, E1): n = ceil((E1-E0)/5); insert equidistant
   nodes so the realized max gap is strictly <= 5 MeV/u.
3. Extend low/high ends from the real collision-energy distribution;
   endpoints must be effectively reachable, not padded arbitrarily.
4. Domain endpoints must become real generated TOPAS nodes; editing a JSON
   to claim range without generating data is falsification and fails audit.
5. Report per channel: old_nodes, inserted_nodes, final_nodes, max_gap,
   observed_query_min/max, optical_depth_fraction.

## Acceptance

Every channel grid satisfies max_gap <= 5.0 with node counts that keep the
total campaign within the sbatch budget (<= 192 CPUs, <= 160 GiB).

## Prohibitions

- No uniform 5 MeV/u full-domain grid that explodes data volume without
  demand justification.
- No threshold relaxation, no aliasing, no water-path changes.

## Commit intent

`docs(ct): add per-channel campaign energy grids`
