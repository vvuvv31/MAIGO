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
- Lowering the secondary continue-tail threshold below 8192 (2026-09-16):
  per-round instrumentation shows the finish-tail round and all under-covered
  rounds together are <1% of the secondary kernel, and the 8192/2048/512 sweep
  made Elapsed slower (2048 +0.6%, 512 +1.5%) with dose unchanged. Kept 8192.
- Compile-time specialization of production-off scoring flags
  (enable_charged_origin_voxel_scoring forced false, 2026-09-16): secondary
  named kernel 178->174 registers but the launched wrapper stayed at 191, and
  RT07575 timing did not improve (Elapsed 27.93->28.24 s). The runtime branches
  are already cheap; specialization alone does not cross a resource threshold.
- MPS and EM step-scale relaxation (2026-09-16): MPS ON/OFF over 1/3/6
  concurrent groups on A6000 differed by <0.3% (191 regs/block prevents SM
  co-residency). Relaxing em_*_step_scale to 2x/4x (source cap raised to [1,4],
  research_step gates removed) bought at most ~5% Elapsed for up to 6.16% of
  peak same-voxel dose deviation; step counts fell only 5-8%. Both stopped.
- Relaxing the 1% combined mean-loss step guard to 5% (2026-09-16): the real
  binding step limit. Primary steps -22.4%, secondary -12.9%, Elapsed -11%, but
  same-voxel dose differs by up to 57.4% of peak and the peak depth moves +2.0 mm
  (R80/R50 unchanged). Not a viable throughput lever; restored to 1%.
- The controlled step limits maximum_step_mm and maximum_relative_energy_loss are
  non-binding (the primary unified path overwrites the latter), so they cannot be
  relaxed for speed either. Step size is set by dose accuracy, not a slack knob.
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


## Universal conditional-sum lookup (2026-09-16; not promoted)

Keeping the original Poisson N and replacing only the explicit transfer sum with a normalized conditional quantile table (N=2..24, R<=128, exact fallback elsewhere) improves the isolated complete restricted-fluctuation sampler by only 6.36–7.56% over four interleaved replay rounds. The approximately 10.5 MiB table also introduces interpolation error. No full-transport or patient-accuracy benefit has been demonstrated; do not integrate into production. Evidence and numerical error bounds over the tested grid are in `docs/em_cost_evaluation_20260916.md`, with executable experiments under `scratch/em_cost_20260916/fluctuation/conditional/`. This result does not reject a future joint compound-Poisson distribution sampler, which is a different, untested candidate.


## Joint Universal compound-Poisson lookup (2026-09-16; rejected for integration)

A joint Poisson-count plus transfer-sum quantile table with explicit zero-event mass, bounded lambda/R coverage and exact fallback improved isolated sampler throughput by 11–13% after compression to 1.7 MiB. However, a secondary-only research transport candidate on RT07575 failed the end-to-end gate: five interleaved runs gave median Elapsed 27.645→28.060 s and secondary kernel 14.731→15.075 s. Run integrity passed with zero overflow, but dose accuracy is unresolved after changed random consumption; no patient acceptance is implied. Do not integrate or expand this candidate based on microbenchmark gains. Evidence: `docs/em_cost_evaluation_20260916.md` and `scratch/em_cost_20260916/fluctuation/compound/compact/transport_results.json`.


## 2026-09-16：EM 覆盖边界缓存未达到推广收益

独立候选将 covers 所需节点能区端点缓存到 UnifiedEmState。真实包 1,294,650 次检查等价；完整 RT07575 五次穿插审计一致，剂量差处于 baseline 重复波动量级，零 overflow。两轮候选相对前后 baseline 均值的 primary 耗时约 −3%、secondary 约 +0.8%，Elapsed 仅 −0.6%。未证明稳定且实用的端到端收益，停止推广；不据此否定所有索引缓存。原始日志、脚本和哈希在 `scratch/em_cost_20260916/coverage_cache/`。基线已有 research_step 旧保护断言失败，见 `docs/em_cost_evaluation_20260916.md` 第五轮。
