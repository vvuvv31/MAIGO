# L4 throughput experiments — 2026-10-09

The user authorized replacing the T4 with one L4 and continuing into the afternoon.
The L4 has 23,034 MiB, driver 580.82.07 and reported billing of 1.54 CU/hour.
The T4 was stopped after its evidence was copied and checksum-verified locally.
The L4 baseline includes the accepted in-place electron change described in
[t4_throughput_20261009.md](t4_throughput_20261009.md).

The physics packages and inputs were transferred in 8 MiB parts after a large
SSH transfer failed. All five reconstructed archives passed byte-size and
SHA256 verification before extraction. Local evidence is under
`out/l4_throughput_20261009/`; the VM is `maigo-l4-20261009`.

## Campaign

Primary candidates are fixed Cu material/species indices, reuse of the existing
Urban range, per-history integer loss-audit aggregation, in-place Urban output,
64/256-thread workgroups, full-optics source ordering, and lazy post-loss
nuclear-rate lookup. The final candidate only evaluates those rates after the
existing nuclear-collision flag is set; energy arithmetic, collision sampling
and RNG draws remain unchanged.

Each candidate runs matched ABBA comparisons on carbon HU0 (121 spots,
121,000 histories/batch) and proton CT minibeam (default lung,
100,000 histories/batch). Each process reuses one GPU context for four batches;
its first batch is excluded from resident throughput. This is implementation
verification, not new TOPAS acceptance or RT07575 head validation.

The campaign requests the reported maximum graphics clock and records whether
locking succeeds, actual clocks, temperature, power and binary SHA256.
Energy, electromagnetic coverage, completion, zero-overflow, per-species and
source/seed checks are required. Candidate doses must match the control.
The first portable-PTX matched arm is allowed to complete. The following
experiments switch to an isolated `CARBON_CUDA_ARCH=sm_89` build. The original
embedded SYCL images were extracted from the ELF and inspected with
`cuobjdump`: they contain both `sm_75.cubin` and `sm_75.ptx`, explaining the
L4 driver JIT. The compiler is already the locally built Intel LLVM / DPC++
Clang24; driver JIT is the later PTX-to-machine-code stage.

Each new executable must contain `sm_89.cubin` in every embedded SYCL image.
AOT runs set `CUDA_DISABLE_PTX_JIT=1`, so a missing machine-code image is an
error rather than an unnoticed PTX fallback. A matched ABBA comparison first
checks AOT versus the portable baseline; subsequent candidates use the AOT
baseline on both arms. These campaigns are kept in separate directories.
The previous runtime-register-cap queue is cancelled when switching: a runtime
JIT register limit cannot establish a register-cap change to precompiled cubins.

## First portable comparison

Fixed primary Cu material/species indices pass the matched checks, but provide
no useful resident gain: carbon ABBA1.001025 (pairs1.000016/1.002035),
proton CT lung1.000496 (pairs0.999817/1.001175). This candidate is not applied.

The AOT baseline and initial candidates compiled successfully. ELF extraction
and `cuobjdump` verify all three images in each uploaded baseline executable
contain `sm_89.cubin`; the AOT runs completed with PTX JIT disabled. Native cubins remove cold-load
JIT, but did not establish a resident transport throughput gain by themselves.

## Status

The user subsequently prioritized proton/carbon CT broadbeam. That campaign
is complete; see [ct_broadbeam_throughput_20261009.md](ct_broadbeam_throughput_20261009.md)
for matched experiments, rejected candidates and release validation.

The selected change uses 256 primary threads per workgroup only for CT
broadbeam; all other presets retain 128. Reverse-order BABA resident rates are:

| RT07575 head preset | Control primary/s | Selected primary/s | Gain |
|---|---:|---:|---:|
| Proton CT broadbeam | 264,032 | 273,398 | 3.55% |
| Carbon CT broadbeam | 137,132 | 139,802 | 1.95% |

The initial ABBA comparisons also improved both CT cases. All eight preset
implementation regressions and 83 retired-interface rejection checks pass.
Matched CT dose passes local1% at zero distance for 100% of voxels above 10%
of the control maximum, a lower bound for local1%/1mm. This is matched GPU
implementation verification, not new TOPAS acceptance. Material caching was
rejected and is absent from the release source. Water performance work is
deferred. Drive caching was cancelled at the user's request; no Drive copy
has been made.

Complete raw evidence for all seven campaigns was restored locally and
SHA256-verified: 63 archive parts, the complete 1,040,628,929-byte compressed
archive, and all 2,552 original files. Verification is recorded in
`out/l4_throughput_20261009/backup_done.json`; the concise selected-result
inventory is `out/l4_throughput_20261009/result.json`. The L4 session was
terminated successfully after verification; `stop_l4.log` records completion.
