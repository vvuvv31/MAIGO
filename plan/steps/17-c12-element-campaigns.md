# Step 17 — Generate C12 correlated events for all 13 target elements

## Objective

Build unbiased pure-element TOPAS campaigns for C12 on H, C, N, O, Mg, P, S, Cl, Ar, Ca, Na, K, and Ti.

## Campaign design

- Projectile: C12 only.
- Target: one declared element per campaign; use Geant4 natural elemental composition unless a validated extraction requires isotope-specific materials.
- Energy nodes: match the package/runtime grid over 0.5--430 MeV/u, with pilot statistics first.
- Score only real selected nuclear interactions and preserve entire correlated events.

Each event records all fields required by Step 16 plus target element and actual isotope A if Geant4 exposes it. Record rejected events and reasons; never silently drop energy-closure failures.

## Pilot before production

Run a small pilot for every target at low/mid/high energy. Audit process identity, event count, product registry coverage, energy/momentum diagnostics, file growth, runtime, and overflow. Fix extractor/schema bugs before production submission.

## Scheduler rules

All jobs are local `sbatch`. Data/logs go under `/mnt/sda/wuwei`. Across active jobs stay <=192 CPUs and <=160 GiB. Allocate resources roughly proportional to energy/observed pilot runtime. Recheck brief `InvalidAccount` after 1--3 minutes. Shard large jobs; overflowed shards are invalid and rerun smaller.

## Compiler/audit

Compile `cinel03_c12_targets.bin`; require all 13 targets and all required energy cells. Report events/cell, effective sample size, isotope mix, rejected fraction, closure distribution, and SHA256 provenance.

## Acceptance

Coverage is complete with no target alias, every source shard is overflow-free, all event fields validate, and the package audit exits zero. Low-stat cells are not marked complete merely because a key exists.

## Commit intent

`feat(topas): extract C12 elemental-target CINEL events`
