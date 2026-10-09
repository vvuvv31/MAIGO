# CT broadbeam throughput — 2026-10-09

The user prioritized proton/carbon CT broadbeam throughput; water benchmarks
and the remaining minibeam campaign are deferred. Both CT broadbeam presets
use RT07575 head inputs. Their source, weights, placement, geometry, physical
models and dose normalization are preserved.

Tests use one NVIDIA L4 (23,034 MiB, driver580.82.07), requested and observed
2040MHz. The baseline includes the accepted electron in-place change.
Local Intel LLVM / DPC++ Clang24 and CUDA12.6 build `sm_89` cubins before upload.
Each executable's three embedded SYCL images are inspected using `cuobjdump`.
`CUDA_DISABLE_PTX_JIT=1` prevents unnoticed runtime PTX fallback.

## Measurement

The initial four-by200,000-history checks select workload size and identify
coarse host stages; they are not the accepted performance experiment.
Formal runs use four-by1,000,000 histories per process, one reused context and
package set, discarding the first batch from resident timing. ABBA compares
control/candidate/candidate/control, using identical whole-campaign largest
remainder allocation and per-spot seeds. Native batch wall timing includes
output writes. Coarse host timings are nested and must not be summed.

Initial formal baseline means are266,987 primary/s for proton CT and137,030/s
for carbon CT. Electron mean1.110/1.730s versus whole3.745/7.298s. These are
workload-specific rates, not a hardware speed ratio against a different GPU.

| Primary workgroup | Proton CT pooled ratio | Paired ratios | Carbon CT pooled ratio | Paired ratios |
|---|---:|---|---:|---|
|64|0.976407|0.976700 / 0.976114|0.986838|0.987268 / 0.986410|
|256|1.031819|1.026613 / 1.037025|1.025522|1.029097 / 1.021965|

All four comparisons pass matched discrete/EM/completion/coverage/energy and
zero-overflow checks. The matched local1% zero-distance dose pass is a lower
bound on local1%/1mm; exact results are retained in matched_check.json.
This is implementation verification, not a new absolute TOPAS gamma campaign.

The64-thread candidate is rejected. The256-thread result was reproduced
with a final scope restricted to CT broadbeam; six other presets retain128.
The final CT-only scope passes all eight presets' two-by10,000-history
implementation regression (no performance claim), maximum dose relative
L1=3.17706514e-7. Reverse-order BABA reproduces gains: proton1.035473
(pairs1.040814/1.030218), carbon1.019472 (pairs1.015347/1.023612).
The CT-only256-thread change is now applied to the release source and rebuilt.
The83 retired-interface rejection checks pass. Latest reverse-order rates are
proton264,032→273,398/s (+3.55%) and carbon137,132→139,802/s (+1.95%).
Both CT cases have local1% zero-distance pass100% above10% of control maximum,
which is a lower bound for matched local1%/1mm; maximum doseL1 is4.24e-8.

## Exact material-parameter reuse candidate

The isolated candidate retains the eight material scalars and normalized
continuous-density coordinate in `UnifiedEmState`, refreshing them when
selection changes material/density/species. It uses the original FP32 density
round trip and interpolation arithmetic. Energy-dependent stopping, range,
hard rates, conditional terminals, moments, fluctuations and RNG calls remain
unchanged. Scheduler continuation carries the state and rebinds its device
pointer as before. No density nodes, new approximation or missing-data
substitution are introduced.

This adds36B of data before alignment to the state and continuation records.
Static CT-primary resources change226/228→230/232registers and2384→2320B
stack. These resource numbers do not establish dynamic spill cost or speedup.
The broad cache candidate passes matched checks and yields proton1.004899
(pairs0.997219/1.012568, unresolved) and carbon1.027294
(pairs1.032957/1.021671). It is not adopted as-is because its larger state also
extends generic secondary continuation records.

A second candidate compiles material reuse only for the CT broadbeam primary
kernel, alongside the CT-only256-thread workgroup. `UnifiedEmState` selects an
empty cache at compile time for other paths. Host layout audit confirms the
uncached state remains64B and the cached state is96B. It has no YAML switch or
compatibility alias. The direct matched comparison against CT-only256 passes all eight regressions
(maxL1=3.28349015e-7), but adds only proton0.22% (opposite pair signs) and
carbon0.77% (pairs+1.20/+0.34%). It is not selected: the modest incremental
benefit does not justify the larger primary state and extra template/API changes.
Release retains the original continuous26 evaluator and state.

The selected implementation changes only primary workgroup size for CT
broadbeam. Water/minibeam workgroup settings remain128. No package, source,
fluctuation, maximum step, scoring arithmetic or normalization is changed.

Evidence is under `out/l4_throughput_20261009/`, with separate VM campaigns
`maigo_l4_ct_broadbeam_1m` and `maigo_l4_ct_material_cache`. Existing carbon
high-statistics tasks and their binaries/data/results are preserved.

## Reusing the offline L4 build

The existing CMake architecture setting can generate native L4 code:

```bash
cmake -S . -B out/build_l4 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCMAKE_CXX_FLAGS=--cuda-path=/usr/local/cuda-12.6 \
  -DCARBON_CUDA_ARCH=sm_89
cmake --build out/build_l4 -j6
```

The actual measured binary and archive hashes are retained with the isolated
`ct_wg256` candidate under the evidence directory. Runtime measurements were
performed with PTX JIT disabled; precompilation removes first-load JIT waiting,
not the steady-state transport cost. The ordinary local build uses its existing
CUDA target setting.

## Completion and evidence retention

The local release build and 83 interface checks completed successfully. Source
SHA256 verification confirms the release source matches the selected isolated
candidate and the continuous26 header retains its original implementation.
All seven L4 campaigns are restored under
`out/l4_throughput_20261009/remote/maigo_l4_*`, including the final CT candidate
and primary-cache rejection. All 2,552 original files, 63 compressed archive
parts and the complete archive passed SHA256 verification. `backup_done.json`
and `result.json` record verification and selected results. The L4 session was
terminated after the verified backup; `stop_l4.log` confirms termination.
