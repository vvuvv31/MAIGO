# Carbon CT minibeam primary pool sizing — 2026-10-16

The release carbon CT minibeam preset now uses `history_chunk_size: 2097152`.
The first experiment below raised it from 131072 to 1048576; the subsequent
larger-batch confirmation below raised it again. Larger primary blocks collect more copper-exit survivors
into one patient continuation pool and require fewer independent terminal tails.
Custom YAML chunk values remain authoritative; historical field configurations
are retained. To reproduce the optimized full-plan setting, use the same chunk
value in both field YAMLs.

The accepted transport source and executable are unchanged. Native 64-step
continuation retains its 8192-survivor/64-round completion thresholds, complete
track/RNG/shared optical-depth/pending-score state, Urban and continuous26,
primary fluctuations, Cu sampling, patient 0.5 mm and Cu 0.05 mm caps, all nuclear
channels and FP32 dose atomics. Source allocation, seeds, geometry, weights and
campaign normalization remain unchanged. Other release presets are unchanged.
The larger pool requires additional GPU memory; every measured run completes
under the existing device-memory budget on the 11 GB RTX 2080 Ti.

## Full-plan measurements

Original RT06423 Head2 optimized 90°/270° full plan, RTX 2080 Ti, full five-pipeline
native sm_75 AOT, `CUDA_DISABLE_PTX_JIT=1`. Each campaign runs three resident
batches of approximately one million histories; the first batch is excluded.
Each paired experiment times 3,999,998 histories per arm. No compilation or
competing GPU work overlaps timing. GPU clocks are unlocked.

| Experiment | Candidate chunk | Control histories/s | Candidate histories/s | Gain | Paired gains |
|---|---:|---:|---:|---:|---|
| ABBA screen | 524288 | 25755.9 | 32008.6 | 24.277% | +24.057% / +24.497% |
| Independent-seed BABA confirmation | 1048576 | 25590.3 | 32876.6 | 28.473% | +29.155% / +27.794% |

Confirmation uses per-spot seed offset 217139. Patient-primary kernel time falls
52.581% (64.821688 → 30.737824 s). Whole-call timing includes all queue,
state exchange, host synchronization and output costs. This is a scoped result
for RT06423 on this GPU, not a fixed gain for other cases or devices.

A separate single-campaign 1048576 probe measured 33132.1 histories/s and selected
the larger candidate. Its comparison with 524288 is exploratory; the paired
confirmation above establishes the reported gain against 131072. A redundant
partial thermal-preparation run is excluded from timing and dose statistics;
the GPU was already warm from the preceding full-plan experiments.

## Implementation checks

All 48 paired full-plan field-batches and six size-probe field-batches pass energy,
EM, coverage, completion and zero-overflow checks. Matched source allocations,
seeds, non-energy ledger integers and discrete Urban/elastic/nuclear/cascade/live-ion
counts agree. Maximum relative dose L1 difference is 7.17314703e-08.
Two matched 20k batches per arm also exercise the release carbon CT minibeam
configuration. These are GPU implementation checks, not new TOPAS acceptance.
Carbon minibeam dose accuracy remains deferred; the paused billion-history
campaign was not resumed.

## Rejected candidate

Reducing the continuation tail threshold from 8192 to 2048 and raising its round
limit from 64 to 256 slowed a separate ABBA screen by 2.233%, with both paired
comparisons negative. That source and binary remain private. Production retains
8192/64; no new scheduler knob or diagnostic runtime path was added.

Evidence: `tmp/carbon_mb_patient_pool_20261016/COMPLETE.json` and
`tmp/carbon_mb_tail_continuation_20261016/NOT_PROMOTED.json`.

## Subsequent enlarged-batch confirmation

With the accepted slit-air cap 0.2 mm, Cu 0.05 mm and patient 0.5 mm, the
original optimized RT06423 full plan was globally allocated once to each
campaign. Local campaigns use two resident ~3M batches (6M total); A6000
campaigns use two ~6M batches (12M total). Each device runs AB then independent
seed BA, with the first batch excluded from timing. Both arms on each device
use the same native AOT executable and identical source files/seeds. PTX JIT
is disabled; no compilation or competing GPU work overlaps formal timing.

| Device | Pool control → candidate | Control /s | Candidate /s | Gain | Two paired gains |
|---|---|---:|---:|---:|---|
| RTX 2080 Ti | 1,048,576 → 2,097,152 | 47,134.0 | 48,993.3 | +3.945% | +1.788% / +6.145% |
| RTX A6000 | 1,048,576 → 4,194,304 | 88,239.2 | 94,691.9 | +7.313% | +6.737% / +7.894% |

These rates use larger batches than the earlier 3×1M experiment; do not
attribute their full difference to the pool change or compare devices as if
their batch sizes/build selections were identical. Local timed histories are
6M per arm; A6000 timed histories are 12M per arm. Clocks are unlocked.

All 32 field-batch runtime/energy/coverage/completion/zero-overflow audits pass;
sources, non-energy integers and discrete/live-species counters match within
each device. Maximum dose relative L1 is 8.126e-8 locally and 1.480e-7 on A6000.
The release CT carbon-minibeam configuration now uses 2,097,152; 4,194,304 is
the scoped A6000 research setting. Other presets and explicit archived field
YAML values are unchanged. Two matched 20k release batches per arm also pass
(maximum dose L1 6.743e-9); this small check establishes implementation
consistency, not throughput. No paused 1B campaign was resumed.

Evidence: `tmp/carbon_mb_large_pool_20261016/{SUMMARY,COMPLETE}.json`.
