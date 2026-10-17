# Proton CT minibeam Cu/air continuation and primary pool — 2026-10-16

The existing carbon CT beamline continuation is now enabled for the dedicated proton CT minibeam pipeline. `config/proton_ct_minibeam.yaml` uses a 1,048,576-history pool instead of 262,144. Explicit user chunk sizes remain authoritative. Proton water, broadbeam, generic and carbon configuration values are unchanged. Carbon CT retains both its upstream and direct Cu-child continuations; proton direct Cu children retain their existing whole-track persistent queue.

## Transport preservation

The shared upstream state and dispatcher already exist. The code change only extends their eligibility to `ProtonMinibeam && enable_ct_grid && queue_copper_primary_children`. Every segment executes up to 64 complete native loop iterations, then survivors regroup by current Cu/air material and 16 energy bins. Fewer than 8192 survivors, or round 64, finishes the original transport under its original cumulative 100000-step safeguard. No tracks or physical steps are skipped.

The 128-byte checkpoint retains energy, position/direction, separate Cu/air Urban states and boundary flags, air ionization clock/threshold/rate/step, Cu shared nuclear optical depth/step, copper-contact state and energy ledgers. Original stateless spot/history source identity is retained; restored tracks skip pre-block air. Per-step copper loss contexts are rebound normally. Incident sampling, nuclear reactions, primary fluctuations, ionization clocks, full restricted/delta slopes, native Cu sampler arithmetic, scoring flushes and dose normalization are unchanged. Patient/Cu caps remain 0.5/0.05 mm; slit-air settings are unchanged.

The checkpoint plus two indices costs 136 bytes/history: approximately 136 MiB at the new 1M pool, in addition to the existing patient checkpoint and entry buffers. Larger pools merge multiple small native-continuation tails; they do not alter the physical step limit. The 20261010 pool-only failure predates both patient and upstream native continuation, so it is not the current scheduling experiment.

## Experiment design

Three original optimized proton minibeam **full plans** retain their own CT, source optics, spots, weights, placement, collimator and normalization: RT07575 Head1, RT06423 Head2, 20022516 Lung. Default release CT minibeam uses a separate flat-source lung benchmark; these are not interchangeable cases.

RTX 2080 Ti uses full all-five-pipeline native `sm_75` builds; A6000 uses matched proton-minibeam-only native `sm_86` builds. The local control links an exact frozen copy of the latest installed full-pipeline library; the remote control builds that same pre-change source. Both GPUs run concurrently, with 3M histories/campaign locally and 6M remotely, three resident batches each. Per-campaign source allocation is preserved globally; remote input counts are exactly doubled from the local full-campaign integer allocation, retaining all source float strings. Original per-spot seeds are matched within every comparison. All 31 referenced shared data/metadata files and the three transferred CT files have matching local/remote SHA256 values.

All native builds finish before timing. Every GPU execution sets `CUDA_DISABLE_PTX_JIT=1`; no PTX JIT or JIT warm-up is used. First batches are excluded from steady rates; each campaign reuses its GPU context/packages. Whole-call times include scheduling, scoring and output. Clocks are unlocked; no performance claim is made from small release regressions or static kernel resource figures. Different build selections and batch sizes across devices prohibit interpreting their ratio as a pure hardware comparison.

Three experiments separate the changes:

1. Upstream continuation alone, keeping the 262144 pool: AB plus independent-seed BA on both devices and all cases, 108M total executed histories.
2. Pool-only screen on the new continuation binary, 262144→1048576: one AB pair per device/case, 54M executions. This is an incremental screen, not independently reversed pool-only confirmation.
3. **Direct final comparison** of old scheduling/262144 versus continuation/1048576: AB plus independent-seed BA on both devices/all cases, 108M executions. These direct results establish the final combined performance; component gains are not multiplied.

The experiments total 270M executions /180 resident batches. Repeated paired histories are implementation/performance checks, not extra independent dose statistics.

## Direct final steady-state comparison

| Case | GPU | Before histories/s | Final histories/s | Whole-call gain | AB / independent BA gain |
|---|---|---:|---:|---:|---:|
| RT07575 | RTX 2080 Ti | 249,676.3 | 303,315.9 | 21.484% | 20.542% / 22.420% |
| RT07575 | RTX A6000 | 391,904.3 | 472,558.0 | 20.580% | 19.311% / 21.840% |
| RT06423 | RTX 2080 Ti | 262,143.1 | 326,481.5 | 24.543% | 23.781% / 25.315% |
| RT06423 | RTX A6000 | 395,004.0 | 490,343.5 | 24.136% | 23.252% / 25.010% |
| 20022516 | RTX 2080 Ti | 252,264.2 | 304,162.5 | 20.573% | 20.797% / 20.352% |
| 20022516 | RTX A6000 | 363,897.1 | 436,357.9 | 19.912% | 20.204% / 19.624% |

All twelve final device/case/order comparisons have positive gains. The upstream-only experiment gives local Head1/Head2/Lung gains9.460/6.398/3.686%, remote4.414/3.826/1.301%; its matched event counts and dose show unchanged transport. The pool-only screen adds local10.247/13.311/15.089%, remote14.100/19.055/17.128%, primarily through reduced patient-primary time. These figures belong to separate cohorts and must not be combined arithmetically into the final table.

## Runtime and dose checks

All formal sources, nested nonenergy integer ledgers and discrete EM/Urban/nuclear/all-ion-elastic/live-ion audits match, with energy/coverage/completion and zero-overflow gates passing. This includes all three experiment stages. Final direct comparisons have maximum dose relative L1 below 1.9e-7, consistent with FP32 accumulation-order differences. No primary fluctuations, nuclear process, source, physical step or energy-loss model is removed or relaxed.

The final native executable undergoes matched two-batch tests of all eight release presets, using the new proton CT pool default, plus generic proton CT/water and carbon CT minibeams. These are implementation regressions, not new TOPAS gamma acceptance or new high-statistics validation. No paused 1B campaign is resumed.

Exact timings, source/input/native-binary hashes, asset verification, audits and dose outputs remain under ignored `tmp/proton_mb_upstream_continuation_20261016/`: `SUMMARY.json` (upstream-only), `POOL_SUMMARY.json` (incremental screen), `COMBINED_SUMMARY.json` (final paired comparison), `VALIDATION.json` and `COMPLETE.json` (final regression/installation). The measured source/native CLI/library are installed together; prior artifacts are backed up.

## Follow-up with larger resident batches

The same retained candidate and matched pre-change controls were subsequently measured with exactly 20M histories per batch, three consecutive batches per arm/device/case. See the [20M-batch throughput report](proton_minibeam_20m_batch_throughput_20261016.md). Keep its large-batch absolute rates separate from the smaller-batch rates above.
