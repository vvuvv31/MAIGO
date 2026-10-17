# Carbon CT minibeam: native Cu charged-child continuation

This change extends native-step continuation to the **direct charged children of primary copper nuclear events** in the dedicated carbon CT minibeam pipeline. Patient secondaries and later copper cascade generations keep their existing implementations. Proton, carbon water and generic paths retain whole-track Cu-child workers.

## Preserved transport

Every segment executes up to 64 original loop iterations. A 60-byte checkpoint preserves energy, position, direction, native step number, copper segment path, shared nuclear optical depth/active flag, Urban track state and boundary flag. The original task remains immutable and supplies the event product, parent history and RNG identity. Reconstructing its initial direction before restore does not consume random numbers.

Survivors are regrouped into 9 species categories ×16 energy-per-nucleon bins ×2 current Cu/air regions. These keys affect scheduling only; all charged fragment Z/A remain supported. A pool below 8192 survivors or at round 64 finishes the original loop with the original cumulative 100000-step safeguard. The native step number is never reset.

The existing 17408-lane persistent worker remains. It now drains bounded track segments rather than necessarily completing each entire track. Host prefix sums and device scatter reorder the next segment. The initial task count is bounded by the existing queue capacity; survivor storage has one slot per actual task. Completion is checked after all survivor rounds drain, and every original task is started once. Existing downstream and cascade overflow rejection remains mandatory.

Only tracks reaching the patient queue subtract their surviving energy from the original history's beamline-removed ledger, exactly as before. A suspended track neither deposits nor subtracts energy. Source allocation, per-spot seeds, normalization, native Cu hard-delta draws, nuclear reactions, Urban scattering and primary fluctuations are unchanged. Patient/Cu/slit-air caps remain 0.5/0.05/0.2 mm in this experiment.

The checkpoint plus two indices requires 68 bytes per actual direct-child task, approximately 183 MiB at 2.8 million tasks or 365 MiB at 5.6 million tasks, excluding the existing task queue. It is allocated after primary continuation buffers have been freed.

## Matched two-device experiment

Case: **RT06423 Head2 carbon minibeam optimized 90°/270° full plan**, preserving the original source, weights, CT, placement, copper geometry and normalization. This is not the default lung benchmark or RT07575.

Within each device, control/candidate source and build selection are matched. RTX 2080 Ti uses all five compiled pipelines, native `sm_75`, a 2M history pool and a 6M-history/two-batch campaign. A6000 uses carbon-minibeam-only native `sm_86`, a scoped 4M pool and a 12M-history/two-batch campaign. Both GPUs run concurrently. Each device executes AB and independent-seed BA; only the second resident batch contributes to steady throughput. Compilation finishes before timing. Every GPU execution sets `CUDA_DISABLE_PTX_JIT=1`.

72M total histories were executed across both arms/devices/orders; repeated paired histories are implementation/timing comparisons, not additional independent dose statistics. No compilation time is included. Different batch sizes/build selections prohibit treating the cross-device ratio as an isolated hardware comparison. Clocks are unlocked, so measured gains are scoped to these runs.

| Device | Control histories/s | Continuation histories/s | Whole-call gain | Direct-child kernel time reduction |
|---|---:|---:|---:|---:|
| RTX 2080 Ti | 63,647.0 | 66,484.2 | 4.458% | 21.573% |
| RTX A6000 | 112,987.0 | 115,091.3 | 1.862% | 22.001% |

Both orders have positive whole-call gains on both devices. In one measured warm 90° field, approximately 35.2% of direct-child tasks survive 64 iterations, 14.9% survive 512, and 6.3% survive 1024. This is a native-work distribution, not a hardware occupancy measurement. The targeted kernel-time reduction measures the combined segmentation/reordering change; it does not isolate either mechanism or establish a particular hardware stall cause.

All formal source CSVs and nonenergy nested integer counters match. Urban, nuclear, all-ion elastic, native Cu and live-ion audits match. Energy/EM/coverage/completion and zero-overflow gates pass. Maximum formal dose relative L1 is 7.284e-8; maximum physical relative energy residual is 8.838e-7. Residual dose differences are consistent with FP32 accumulation order, not an introduced physics approximation.

The eight release presets and generic carbon CT/water minibeams also undergo matched two-batch implementation regression. These checks are not new TOPAS acceptance; carbon minibeam dose accuracy remains under verification. No paused 1B campaign is resumed.

Raw inputs, native build commands, source/binary hashes, logs, dose outputs, audits and survivor curves are under ignored `tmp/carbon_mb_child_continuation_20261016/`. See `SUMMARY.json`, `VALIDATION.json` and `COMPLETE.json` there for final installation provenance.
