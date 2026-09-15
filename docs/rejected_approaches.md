# Rejected acceleration approaches (do not retry without new evidence)

All below were measured on the local RTX 2080 Ti against the unified-EM
baseline. "Rejected" means: slower, physics-breaking, or unfixable without
changing validated distributions. Code for batch/macro/RNG experiments was
removed from the tree; benchmark records stay frozen for provenance.

## 1. Delta batching (Ts-split, Poisson + per-event spectra)

- Idea: span K delta-clock ticks, Poisson count + exact per-event spectra.
- Result: **-5% to -19% throughput** (fewer steps, but added Poisson
  integrals, bounds, branches per step outweighed savings).
- Evidence: `docs/delta_batch_sampling.md`,
  `benchmark/delta_batch_20260914/results.json`.
- Code removed: `CARBON_EM_DELTA_BATCH`, `em_delta_batch_cut_scale`,
  `unified_delta_split`, `DeltaSpectrum`/`sample_delta_batch`,
  `delta_batch_candidate.hpp`, `tests/delta_batch_candidate.cpp`.

## 2. Quantile tables for delta spectra

- Idea: tabulated spectrum quantiles to skip accept/reject.
- Result: **+10.9% speed but IDD max diff 11.4%** (frozen fluctuation
  templates lost energy dependence), R80 shifts over gate.
- Evidence: `docs/rt07575_quantile_optimization.md`,
  `benchmark/delta_quantile_20260914/`.
- No tree code remains.

## 3. Macro-step with h-scaled continuous + exact delta walk (K-tick)

- Idea: one h-scaled fluctuation over H + exact clock walk.
- Result: ~2x kernel, but **Bragg peak +2.3% on sharp water** (integral
  exact, R80 fixed). Host slowdown reproduced +1.8% in 11 s.
- Root causes (bisected): stale prepare on rising-S edge (+4.3% direction),
  stale proposal/threshold tick positions (-1.4% direction). Only fully-fresh
  per tick (= micro) is exact, which saves nothing.
- Evidence: `docs/em_macro_step.md`,
  `benchmark/benchmark20260914_b1b2_unified/` (K=4/8 IDD).
- Code reverted (`em_macro_ticks` removed).

## 4. Adaptive macro (high-E macro, low-E micro)

- Idea: restrict §3 to plateau to dodge peak bias.
- Result: **26% steps macro-ized still +0.97% peak**; bias spread over the
  whole track with sign changes (non-monotonic in E_cut). No viable cut.
- Evidence: host `slowdown` adaptive scan (11 s per config).

## 5. Grouped micro (exact per-tick EM, shared tail only)

- Idea: identical EM calls, share MCS/nuclear/advance/geometry per span.
- Result: accuracy within noise, but **speed ~= K1 on water, slower on
  RT07575** (tail saving < loop/codegen overhead; +5% K1-path pollution
  until revert).
- Verdict: tail-only sharing is not a lever.

## 6. RNG / ALU micro-opts (all ~0 or negative: memory-latency bound)

- Philox 4-output buffering: **-5%** (register spill at 255 regs + warp
  refill divergence). Per-step buffer additionally corrupts the stream.
- Exact single-draw Poisson inversion (validated chi2 < 21): **0%**.
- Global `-ffast-math`: +2.8% (finite-math broke host parsing separately).
- `maxrregcount=128`: 0% (spill cancelled occupancy).
- Direct `prepare`/`loss` field interpolation without a full `UnifiedEmNode`
  temporary (2026-09-15, vs compact+index production): audit-identical, RT07575
  elapsed about 3% slower, secondary kernel slower; reverted. Turing still
  rounds ~203 regs to an 8-warp SM cap; this did not cross the 168–192
  register band that would allow 10–12 warps.
- Sticky last-interval search plus independent stopping/range cubics
  (2026-09-15): audit-identical, resume 272→304 bytes, RT07575 elapsed about
  9% slower (primary and secondary). Extra hint state and hit/miss branches
  outweighed the already-narrow exact-index binary search; reverted.
- Add birth-voxel Schneider section to the secondary regrouping key,
  `(section, species, energy)` with 26×19×16 buckets (2026-09-15): quality
  pass, zero overflow, dose within atomic-order noise, but RT07575 secondary
  about 3% slower than the accepted `(species, energy)` key. Section/density
  changes along the track diluted the locality benefit and added a CT material
  load plus more atomic buckets per regroup; reverted. Only the species × 16
  energy key was kept.
- Splitting the secondary inelastic final-state block into a separate kernel
  (2026-09-15, plan in `gpustructure.md`): stopped before implementation.
  Removing the entire event block in a diagnostic stub only changed the
  secondary kernel from 178 to 174 registers (stack 2096→1984 B); primary
  unchanged at 168. Neither 32-thread (11 warps) nor 128-thread (2 blocks)
  occupancy crosses a threshold, so the split cannot raise occupancy, and the
  event path runs only ~1.3e6 times against >1e9 secondary steps. Production
  unchanged. Evidence: `scratch/opt_gpu_20260915/binary_inelastic_stub`.
- Splitting the EM node/segment binary-search keys into contiguous float
  arrays (candidate A from `gpustructure.md`, 2026-09-15): SASS sampling showed
  the search compares (`LD.E.SYS` stride 52 B node / 24 B segment) are ~11% of
  the secondary kernel's long_scoreboard samples, but the separated-key build
  (`CARBON_EM_SPLIT_SEARCH_KEYS=ON`) was ~4% slower on the secondary kernel
  (15.36 vs 14.73 s) with dose within atomic noise. The binary search still
  performs the same number of dependent loads; the extra key stream adds a
  second global read per step and the 52 B/24 B records were already L2-resident.
  Kept OFF (default); code isolated behind the compile switch.
- Adding a 2-bit mantissa sub-index to each exact-index exponent bucket
  (2026-09-16): table audit (`tools/audit_em_index_widths.py`) shows the current
  8-bit exponent index already leaves a median candidate interval of 1 for the
  binary search (mean 1.31 comparisons/query). A 10-bit exponent+2-mantissa
  index only lowers this to 1.13 comparisons/query while multiplying the index
  space and upload cost ~4x. Not implemented.
- Dummy-RNG separation: Philox ALU ~= 4% of primary step; the 2.8x
  constant-dummy effect was divergence elimination, not RNG math.
- Interval narrowing (`CARBON_EM_EXACT_INDEX` style): -1.1%.
- `no_delta`: 3.9x but unphysical (deletes delta energy). Never a candidate.
- All reverted; `rng.hpp`/sampler match HEAD.

## 7. Legacy EM on GPU (removed)

- `em_model: legacy` rejected on cuda/nvidia/gpu/default devices;
  serial/cpu backends (incl. `transport_cpu` and `carbon_tests`) retain
  legacy routing. Joint water EM (`g4_joint_water_v1`) fully deleted
  (firewalled as unknown key).
- Legacy-vs-unified gap (current binary, scale 1.0):
  legacy peak systematically low by 0.4-4.5pp, direct MARE 0.2-0.6%,
  R80 <= 0.06mm; legacy ~1.5x faster on b1.
- Evidence: `benchmark/benchmark20260914_legacy/`.

## What remains viable

Overhead cuts that keep EM stepping exact (kernel specialization for
registers/occupancy, species/range grouping, workgroup tuning, lookup
fusion), or approximations with explicit error budgets (§9.2 short-range
termination after share measurement, roulette with full weight machinery).
Sharing EM evaluations across ticks is closed (items 1-5).
