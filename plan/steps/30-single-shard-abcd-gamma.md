# Step 30 — Single-shard A/B/C/D Gamma

## Objective

Attribute dose increments across physics stages on ONE shard with strictly
paired conditions.

## Stages (same binary, spots, histories, seed, CT, MCS, stopping, ledger,
output grid, coordinate transform, Gamma sample points)

- A: frozen water / old-physics baseline
- B: Schneider stopping + primary attenuation
- C: Schneider + complete primary CINEL03, no secondary nuclear reactions
- D: Schneider + complete secondary CINEL03, generation=2

## Outputs (per stage)

- 3D dose grid + `quality_report.json` (each must be accepted = true) +
  config with all input SHA256 values

## Report

- Total deposited MeV/primary, primary survival, fragment/species dose, IDD
  (IDD by transverse summation of the 3D dose only), high-dose Pearson r,
  Global/Local 3%/3mm and 2%/2mm, spatial failure map, nominal + LS scales,
  A->B / B->C / C->D increments

## Acceptance

D accepted = true with an explainable improvement over C.

## Prohibitions

- No 1D scorer; no dose/range/physics tuning for Gamma; no
  accepted=false Gamma conclusions; no 20-shard before this passes.

## Commit intent

`test(ct): single-shard A/B/C/D paired Gamma on v2 package`
