# Transport pipelines

The eight release presets explicitly select one of four compile-time pipelines
using `transport_pipeline`. The fifth, `generic`, is the default for custom
configurations that omit this key. All sources remain restricted to proton/C12. Each pipeline generates its own
primary and charged-secondary kernel identities. A proton/carbon
pipeline identity describes the source, not the species of its fragments:
generic charged fragment species remain supported in every secondary worker.

| Pipeline | Independent policy and optimization entry |
|---|---|
| Proton broadbeam | `src/pipelines/proton_broadbeam.hpp` |
| Proton minibeam | `src/pipelines/proton_minibeam.hpp` |
| Carbon broadbeam | `src/pipelines/carbon_broadbeam.hpp` |
| Carbon minibeam | `src/pipelines/carbon_minibeam.hpp` |

The proton CT broadbeam explicit local-electron mode uses stage15, or stage16
when a host certificate proves identical CT/scorer grids. Stage16 reuses the
directed scoring cell for material reads; unclamped validity and exact face
ownership remain authoritative. Other configurations keep their existing
stages. All current paths use local ionization deposition; no electron/photon
transport packages or configuration switches remain.

Stages15/16 use a step-start delta first moment for the proton primary, retaining
the current continuous26 rate, variance, restricted-loss slope and fluctuations.
The two nearby-energy delta evaluations are omitted. Secondary particles and
other stages keep the slope correction. The executed specialization sets
`primary_first_order_delta_enabled`, so quality reports declare the approximation
only when it ran. RT07575 head full-plan TOPAS qualification and measured
throughput are recorded in `docs/proton_ct_first_order_delta_20261009.md`;
this qualification applies to the patient-interior broadbeam dose gate.

Policies own primary workgroup size, default history chunk selection, secondary
segment length and tail threshold. Broadbeam policies also select their
projectile-specific CT primary stage. Explicit YAML history chunks remain
strictly authoritative. CT and water continue to select the existing medium
implementation and package behavior within each pipeline; the selector changes implementation, not the physical profile. Unknown values
and source/beam-mode mismatches are rejected before transport.

Custom configurations may omit `transport_pipeline` or set it to `generic`.
Copies of a release YAML retain its explicit dedicated selection; remove that
key or replace it with `generic` to use the general path. Conversely, custom
proton/C12 configurations may explicitly select a matching dedicated path.
`src/pipelines/generic.hpp` keeps the existing broadbeam/reference physics
modes and all package/coverage validation. It does not admit other primary
species or supply missing physical data.

`src/transport_sycl.cpp` contains common helpers and the once-per-batch router.
`transport_pipeline<Pipeline, JointResponse>` composes four shared phases:

| Phase | Shared implementation |
|---|---|
| Input/package cache, allocation, geometry and setup | `src/detail/sycl_pipeline_setup.inc` |
| Primary launch, compact beamline/medium stages and Cu children | `src/detail/sycl_pipeline_primary.inc` |
| Charged fragments, grouping, segmented continuation and tails | `src/detail/sycl_pipeline_secondary.inc` |
| Readback, dose and energy/coverage/completion audits, cleanup | `src/detail/sycl_pipeline_finalize.inc` |

These `.inc` files are lexical phase boundaries, matching the repository's
existing device-helper organization. They are not independent translation units:
phase-local buffers retain exactly the original lifetime and CUDA ordering.
Small reusable physics operations remain in the existing geometry, continuous26
EM, nuclear, Urban, copper and scoring helpers. There are no four copies of the
physical transport loop. Shared phases are still substantial; this first step
creates independently compiled optimization boundaries rather than rewriting
or approximating the physics. Future path-specific step helpers should be
selected through the pipeline type rather than growing runtime species switches.

Minibeams retain two compact primary stages and their separate Cu/
fragment kernels, rather than merging the entire simulation into four kernels.
Carbon water retains its TPS CSV, emittance and generic Urban medium path.
Primary rate/package calibrations, FP32 dose/geometry/loss, fluctuation flags,
step caps, seed allocation and normalization are unchanged by the refactor.

To optimize one path, edit its policy or add a compile-time operation owned by
that pipeline and call it from the corresponding shared phase. Medium-specific
workgroup selection can be tuned in `primary_workgroup(ct)`. Do not specialize
the secondary species to the source or duplicate shared physics tables.

Validate shared changes using `scripts/validate_presets.py` against a frozen
pre-change binary and all eight defaults. Matched source allocation, seeds,
discrete audits, dose, energy, EM coverage, completion and zero overflow must
pass. Performance work additionally requires independent continuous-batch
comparisons after reading `failed.md`; smoke/regression timing is not a
performance measurement. RT07575 head CT broadbeams and lung CT minibeams
remain distinct validation inputs. Absolute TOPAS acceptance is separate.

Refactor evidence: `out/pipeline_refactor_20261009/`.

## Selective compilation

Five CMake options control which pipeline templates and CUDA kernel families
are instantiated. All default to `ON`, so an ordinary release build still
supports all eight presets and custom proton/C12 configurations.

| CMake option | Compiled configurations |
|---|---|
| `CARBON_ENABLE_PROTON_BROADBEAM` | Proton water/CT broadbeam |
| `CARBON_ENABLE_PROTON_MINIBEAM` | Proton water/CT minibeam |
| `CARBON_ENABLE_CARBON_BROADBEAM` | Carbon water/CT broadbeam |
| `CARBON_ENABLE_CARBON_MINIBEAM` | Carbon water/CT minibeam |
| `CARBON_ENABLE_GENERIC` | Custom proton/C12 configurations selecting `generic` |

For proton broadbeam optimization, use a separate build directory. Explicitly
specify `-DCARBON_CUDA_ARCH=sm_75` (根据自己显卡设置: RTX 2080 Ti `sm_75`,
RTX A6000 `sm_86`, L4 `sm_89`) and disable runtime PTX JIT:

```bash
BUILD_DIR=build-proton-broadbeam bash scripts/build.sh \
  -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_ENABLE_PROTON_BROADBEAM=ON \
  -DCARBON_ENABLE_PROTON_MINIBEAM=OFF \
  -DCARBON_ENABLE_CARBON_BROADBEAM=OFF \
  -DCARBON_ENABLE_CARBON_MINIBEAM=OFF \
  -DCARBON_ENABLE_GENERIC=OFF

export LD_LIBRARY_PATH="/home/wuwei/sycl_workspace/llvm/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DISABLE_PTX_JIT=1
./build-proton-broadbeam/carbon_mc --config config/proton_ct_broadbeam.yaml
```

The options select pipeline families, not individual medium presets. A proton
broadbeam build keeps both CT and water behavior and all required charged
charged-fragment transport. Common physics/package helpers remain
available to the enabled family; disabling carbon primary pipelines does not
remove charged fragment species. Selecting a disabled pipeline fails during
configuration validation, before GPU/context/package setup. It never silently
falls back to `generic`. CMake rejects a build with all five options disabled.

`scripts/build.sh` forwards its arguments to CMake and accepts `BUILD_DIR`;
relative build directories are resolved against the repository root. CMake
caches option values per build directory. To restore every family in a directory
where some were disabled, explicitly set all five options to `ON`. Keep selective
builds separate from the normal `build/` release directory.

Selective-build validation (2026-10-09): native sm_75 resource inspection confirms
that a proton-broadbeam-only executable excludes the other four kernel families.
It is 11.1 MiB, compared with 51.8 MiB for the complete build. Both supported
medium presets pass matched two-batch checks (10,000 histories/batch); disabled
families fail before GPU initialization and an all-disabled CMake configuration
is rejected. The default complete build passes all eight matched presets, both
generic beam-mode entry checks and 87 interface checks. Maximum dose relative L1
is 2.91227e-7; all cases have 100% zero-distance local dose pass above 10% peak,
with matching source/discrete audits and accepted energy/coverage/completion/
zero-overflow checks. Evidence: `out/proton_broadbeam_perf_20261009/result.json`.

Two exact proton CT primary rate-query candidates were also benchmarked with
four 1M-history batches per process, discarding the first batch. Schneider-only
dispatch and early hazard evaluation give ABBA throughput ratios 0.997912 and
0.996047, with opposite paired signs under unlocked-clock/temperature drift.
Neither establishes a useful improvement; both were reverted and recorded in
`failed.md`. Selective compilation is a build workflow change, not a transport
speedup or new TOPAS acceptance.

## Validation — 2026-10-09

The release build passes. Dedicated and generic regressions each pass all eight
presets (two batches of10,000 histories per arm): max dose relativeL1
2.70723e-7 / 2.93809e-7, identical discrete audits/source allocation/seeds,
accepted energy/EM/coverage/completion and zero overflow. All16 matched cases
have zero-distance local dose pass100% above10% of control peak, a lower bound
for the corresponding broadbeam local1%/1mm or minibeam local3%/0.2mm gamma.
Generic tests include both an explicit selector and an omitted selector. The87
interface rejection checks and release asset manifest check pass.25 previously
copied RT07575 carbon-minibeam input files were newly registered in the manifest;
no input/package contents changed. Default CT minibeams remain the lung case.

The final three embedded device images contain native sm_75 cubins. Resource
inspection confirms distinct primary and secondary kernel identities for all
four dedicated pipelines and the generic family. Proton/C12 fragment identities
remain generic. The local executable grows from about19MiB to52MiB because the
pipelines compile independently; build/link time also increases. One selected
pipeline reuses the existing context and packages during continuous batches.

RTX2080Ti CT broadbeam formal ABBA, four1M-history batches/process, discard
first batch, PTX JIT disabled: proton control149,461→candidate148,955 primary/s
(ratio0.996612; pairs0.990339/1.002880); carbon72,182→71,677/s
(ratio0.992994; pairs0.985828/1.000141). Matched dose/discrete/energy/EM/
coverage/completion/overflow audits pass. Unlocked clocks/temperature vary;
these small changes do not establish a consistent regression or a speedup.
Water performance was not benchmarked. This is a structure change and matched
GPU implementation check, not new absolute TOPAS acceptance.

The first four-pipeline compile exposed an over-instantiated CT primary stage
in the water broadbeam response mode; restoring the original compile-time
JointResponse guard resolved it before GPU execution. Final evidence and
source/binary hashes are in `out/pipeline_refactor_20261009/result.json`.
