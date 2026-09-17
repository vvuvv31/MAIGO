# EM macro-step (K-tick span): investigated and reverted — 2026-09-14

## Outcome

Macro-stepping the unified EM (spanning K delta-clock ticks per step) is
**not viable**: sharing prepare/rate-update across ticks introduces a
systematic Bragg-peak shape bias on sharp (zero-spread) water cases that
exceeds tolerance, and the exactly-equivalent grouped form saves nothing.
All macro code was reverted; production stepping is unchanged single-tick.
Kept: `enable_secondary_unified_em` switch (separate, validated).

## What was tried

Per macro step H = min(native, K x tick, 0.005 E/S), floor one tick:
one h-scaled continuous draw + exact clock walk (re-armed exponentials,
per-event spectra, carryover preserved). K=1 path was ulp-identical to
baseline; K=8 gave ~2x kernel speedup on RT07575/b3/b4.

## Why it fails (b1/b2 unified benchmark caught it)

b1_200 unified K=8: peak **+2.3%** vs TOPAS (K=1: -0.07%), integral exact,
R80 fixed — pure shape narrowing. Host full-slowdown reproduced +1.8% in
11 s. Bisection:

- stale prepare over the span (stopping evaluated at step-start E on the
  rising-S edge underestimates upstream loss, pushing dose into the peak):
  estimated +4.3%;
- stale proposal/threshold (tick positions shift): estimated -1.4%;
- only fully-fresh per tick (= micro) is exact — which saves nothing.

Grouped form (exact per-tick EM, shared tail only): accuracy within noise
but speed ~= K1 on water and slower on RT07575 (codegen/register pressure
also polluted the K1 path +5% until revert).

## Secondary unified EM switch (kept)

- `enable_secondary_unified_em` (default off): off = CSDA fallback.
- Throughput (with old K=8): RT07575 1M 30771 -> 48079 hps; b3/b4 +23-27%.
- Therapy dose unchanged (gamma identical); out-of-mask low-dose region +9%
  relative (+1.6% of total). See benchmark notes before graduating default.

## Lesson

b3/b4/RT07575 (spread + heterogeneity) masked the shape bias; only sharp
zero-spread b1/b2 revealed it. All dose benchmarks must run unified EM.
Speedup must come from overhead cuts that keep EM stepping exact
(RNG reuse, kernel specialization), not from sharing EM across ticks.

## RNG cost (priority 1 closes as not-a-lever)

- Philox-10 per draw, 1/4 outputs used. Dummy experiments on RT07575:
  constant-0.5 dummy (removes Philox + divergence + fixes iterations):
  per-step 16.9 -> 6.0 ns (2.8x). Per-(history,draw) hash dummy (removes
  Philox ALU only, keeps draws/divergence/iterations): primary 7.45 -> 7.14 s
  (+4%). Conclusion: Philox ALU ~= 4%; the 2.8x was divergence + iteration
  effects, not RNG math.
- Buffered 4-output reuse: -5% (register spill at 255 regs + warp refill
  divergence); per-step buffer corrupts the stream (4x step-count artifact).
  Reverted.
- Exact single-draw Poisson inversion (validated chi2 < 21): 0% gain
  (ALU free, memory-latency bound). Reverted.
- Verdict: stop all RNG/ALU work. Remaining levers are memory latency,
  divergence, occupancy (specialization, grouping, tuning).
