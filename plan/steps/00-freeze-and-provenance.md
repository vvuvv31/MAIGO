# Step 00 — Freeze water/CINEL and record provenance

## Objective

Create an immutable, reproducible reference for the current water and secondary-species behavior before CT work. This step changes no physics.

## Required actions

1. Record `git rev-parse HEAD`, branch name, `git status --short`, compiler/SYCL versions, GPU model/driver, TOPAS version, Geant4 version, active physics list, and SHA256 of `data/HUtoMaterialSchneider.txt`, every active CINEL package, stopping table, and baseline configuration.
2. Because the worktree is already dirty, produce two manifests: `HEAD-manifest.json` and `working-tree-manifest.json`. The latter lists every modified/untracked file and hashes it. Do not commit or discard someone else's changes.
3. Identify the exact existing water regression commands and run a small deterministic CPU test plus the established local GPU water validation. Use current settings unchanged.
4. Store numerical summaries and hashes under `/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-00/`. If existing large baselines under the old `plan/artifacts` were removed, regenerate only the minimal authoritative summaries; do not restore the old plan tree.
5. Add a regression test/config guard showing that merely enabling CT support code while no CT is configured does not change water output for the same seed.

## Do not change

- No cross sections, stopping powers, CINEL packages, replay semantics, species registry, RNG calls, tolerances, or scorer definitions.
- Do not enable secondary cascade for later CT tests here.
- Do not create a branch automatically or reset the dirty worktree. Record the branch; branch creation is a human workflow decision.

## Acceptance

- Both manifests parse as JSON and all referenced hashes re-compute exactly.
- Water regression passes with the repository's existing tolerance and no threshold change.
- Same seed/config without CT produces bitwise-identical deterministic summary if the code promises determinism; otherwise document and pass the existing statistical criterion.
- Evidence explicitly identifies the dirty CT material-rate changes and the fact that secondary-species work is frozen.

## Commit intent

`chore(ct): freeze Schneider workstream provenance`
