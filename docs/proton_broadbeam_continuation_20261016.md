# Proton CT broadbeam continuation — 2026-10-16

## Change and scope

Eligible dedicated ProtonBroadbeam CT primary transport now shares the existing proton-minibeam continuation dispatcher. Up to 64 complete native iterations run before checkpointing; unfinished histories are compacted and grouped by cached material section and current energy (27×16 bins). Below 8192 survivors, or after 64 rounds, the final launch completes the remaining native transport. The original maximum-step-count safety guard remains; no trajectory is truncated by the scheduler. Noneligible/custom configurations retain their original dispatch. Carbon and water scheduling is unchanged.

All cross-step RNG counters, source/history identity, ionization and nuclear optical depths, hadronic/EM caches, pending voxel/depth and grid energy sums survive. Resumes skip source sampling and rebind the EM table pointer to the current kernel capture. Primary fluctuations, secondary-straggling-off, continuous26 arithmetic/calibration/slopes, nuclear elastic/inelastic, FE, CT geometry and the 1 mm broadbeam cap are unchanged. Dose atomics remain FP32. The shared 272-byte checkpoint plus two uint32 indices costs about 140 MiB at a 524288-history broadbeam chunk; continuation queue writes are bounded and oversized counts rejected.

## Matched full-plan timing

RTX 2080 Ti, all five production pipelines linked, Release/native sm_75 AOT, CUDA_DISABLE_PTX_JIT=1. Current proton-minibeam continuation/Cu lookup changes are in both arms. Each case uses its own original full-plan CT/source/optics/placement/weights, retaining the archived six-case benchmark inputs. RT07575 sources are generated through TpsSourcePlan with its beam model and upstream-air correction; the two manual plans reproduce the prior benchmark driver's upstream-energy preparation. Retired electron/scorer keys are removed only from these private test YAMLs. No source reweighting to favor one anatomy.

Each campaign has 9M histories allocated once by largest remainder across the full plan, then balanced into three 3M resident batches with independent per-spot batch seeds. ABBA plus independent-seed BABA for every case: 24M warm histories/arm/case, 216M histories simulated across all cases/arms. First batches are excluded only from steady-state timing. Both arms use byte-identical source CSVs. Timings include the transport call, state exchange, ordering, host synchronization, scoring, audits and output; compilation is completed before any formal GPU timing. Clocks are unlocked.

| Case | Control histories/s | Continuation histories/s | Whole-call gain | Primary-time reduction |
|---|---:|---:|---:|---:|
|RT07575 Head1|601637|695598|15.618%|15.105%|
|RT06423 Head2|627886|714476|13.791%|13.704%|
|20022516 Lung|383741|528565|37.740%|30.227%|

## Implementation checks

All 72 full-plan batches pass energy/EM/coverage/completion and zero-overflow checks. Source CSVs, nonenergy discrete/live-ion/nuclear/elastic/EM audit totals match exactly. Matched dose relative L1 ≤3.04049e-07; maximum relative energy residual 3.37542e-07. Every candidate batch passes paired GPU 1%/0 mm at 100% for reference dose >10% Dmax; largest voxel relative difference in that region 3.41739e-06. Thus the paired implementation comparison also satisfies 1%/1 mm without needing spatial search. This is a GPU implementation comparison, not new independent TOPAS acceptance.

All eight release presets and generic proton CT broadbeam/minibeam pass matched two-batch checks, with exact nonenergy counters and dose relative L1 ≤3.12167e-07. These low-statistics checks support implementation consistency and carry no throughput claim.

## Interpretation and artifacts

The larger measured lung benefit supports a shared scheduling inefficiency with the proton-minibeam case. Compaction, segmentation and material/energy regrouping change together, so their individual divergence/locality/hardware contributions are not isolated. These results do not establish the cause or achievable gain in carbon pipelines or other GPUs.

Evidence: tmp/proton_bb_continuation_20261016/{SUMMARY.json,RELEASE_AUDITS.json,runs/}. The compiled source snapshot remains candidate/source; integration differs only by comments and indentation, checked after stripping comments/whitespace. The exact measured executable/library are installed in tmp/remove_electron_20261015/build/. No commit/push performed.
