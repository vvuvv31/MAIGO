# Carbon CT minibeam native-step continuation — 2026-10-16

The release carbon CT pool size is subsequently updated in
[the primary-pool sizing report](carbon_minibeam_primary_pool_20261016.md).
The measurements below retain their original131072 chunk.

## Retained change

The dedicated CarbonMinibeam CT patient phase now reuses the existing proton
native-step continuation dispatcher. Every 64 native iterations it stores the
complete track state, compacts surviving primaries, and groups them by current
material and energy. The 27-material/16-energy scheduling key changes execution
order only; it does not approximate material density, energy or physics tables.
Below 8192 survivors, or after 64 continuation rounds, the tail finishes all
remaining native transport under the original maximum-step safeguard.

The checkpoint preserves position/direction/energy, primary history and RNG
identity, native Urban and continuous26 state, the ionization clock, shared
nuclear optical depth, hadronic cache, step counters, and pending scores. The
continuous26 table-view pointer is rebound for each resumed kernel. No clock
thinning, resampling, premature scoring or new physics approximation is added.
Copper/air transport, Cu children and patient secondaries retain their physics.
Water configurations and generic carbon minibeams keep their preceding scheduling;
proton broadbeam/minibeam retain their already enabled continuation.

The change reuses the 272-byte checkpoint and two uint32 indices: about 35 MiB
at the 131072 carbon CT chunk used in this benchmark, plus small queue/histogram buffers.
Primary fluctuations stay enabled; secondary/Cu-fragment restricted fluctuations
stay disabled. Patient step cap 0.5 mm, Cu cap 0.05 mm, native delta sampling,
Urban, nuclear channels, calibrated rates/slopes and all input data are retained.

## Full-plan timing

RTX 2080 Ti, all five pipelines, native sm_75 AOT and
`CUDA_DISABLE_PTX_JIT=1`. Original optimized RT06423 head 90°/270° full plan,
unchanged CT/source/placement/weights/normalization. No compilation or other GPU
job overlaps timing. First batch excluded from each campaign. A separate 1.2M
control campaign heats the GPU before the ABBA screen; it contributes neither
timed histories nor independent dose statistics. Confirmation reverses to BABA
with a separate per-spot seed offset. GPU clocks are unlocked.

| Experiment | Control histories/s | Continuation histories/s | Pooled gain | Paired gains |
|---|---:|---:|---:|---|
|Screen: three 400k batches/campaign|21998.7|22816.2|3.716%|+2.778% / +4.654%|
|Confirmation: three 1M batches/campaign|24448.4|25706.8|5.148%|+5.370% / +4.927%|

Confirmation includes 3,999,998 timed histories per implementation. Patient-primary
kernel time decreases 10.030% in this confirmation. Phase time
includes every initial/resumed kernel; queue/grouping and context/output costs
remain in the whole-call wall time. The screen and confirmation use different
batch sizes and are not pooled into one rate. These are scoped measurements,
not a fixed gain for other cases or GPUs.

## Matched implementation checks

All 48 full-plan field-batch runtime/energy/coverage/completion/zero-overflow
checks pass across the screen and confirmation. Full-plan source allocations,
per-spot seeds, non-energy integer ledgers and discrete EM/Urban/nuclear/elastic/
live-ion audits match. All eight presets and generic carbon CT/water minibeams
also pass two matched 20k-primary batches per implementation. Maximum relative
dose L1 difference across these comparisons: 2.86967e-07; maximum full-plan
relative energy residual: 6.87027e-06. FP32 dose atomics can accumulate in a
different order. Preset checks establish implementation consistency, not speed.

The exact tested full executable/static library are installed in
`tmp/remove_electron_20261015/build/`. Frozen sources, inputs, timing, audit
outputs and binary/source hashes are under
`tmp/carbon_mb_continuation_20261016/`, with `COMPLETE.json` as the index.
This is not new TOPAS acceptance; carbon minibeam accuracy remains deferred.
The paused RT06423 billion-history campaign was not resumed.

## Rejected preparation-only candidate

Deferring the unused low-energy Cu ion dispersion correction passed 106,954,752
paired component draws bitwise, but a separate warmed ABBA full-plan comparison
gave only +0.0337% overall, with mixed paired directions. It remains private and
is not combined with continuation. Evidence:
`tmp/copper_lazy_ion_dispersion_20261016/`.
