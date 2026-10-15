# Intel Arc native AOT support — 2026-10-15

MAIGO can be built for Intel Level Zero GPUs with `spir64_gen` and an explicit
`CARBON_INTEL_GPU_ARCH`. The tested device is an Arc B580 (12 GB), architecture
`bmg-g21`, on Debian 13 with the Xe driver 26.22.38646.4 and oneAPI 2026.1.
The eight release YAMLs still select CUDA. An Intel configuration uses
`device: intel` and the same physics assets and source inputs.

## Implementation

- CMake selects either CUDA native AOT or Intel native AOT. Intel uses separate
  native images per kernel. Portable SPIR-V/JIT-only targets are rejected.
- Device selection accepts the two explicit backend names, with no fallback.
  Intel requires an Intel Level Zero GPU with FP64, atomic64 and device USM.
- Intel primary kernels access immutable nuclear and EM views through one
  device context. CUDA retains its existing captures and queue ordering.
- The exact CINEL03 lookup is kept out of line on Intel devices. Together with
  per-kernel images this avoids incorrect CT lookup results observed in the
  tested compiler's merged images. Channel selection rules are unchanged.
- Temporary host arrays used by the copper/air float-pair uploader now remain
  alive until both asynchronous transfers complete. Previously, freed array
  storage could be reused for the next table before Level Zero copied it.
  A copper stopping-power query consequently read a nuclear-rate value. The
  fix applies to both backends.

No physics model, source, RNG mapping, track limit, dose precision or scheduling
parameter was changed to support Arc. All experimental step caps, workgroup
changes, tail scheduling, large-register options and ledger-reduction trials
were excluded from the final source.

## Required runtime configuration on the tested system

Use `SYCL_UR_USE_LEVEL_ZERO_V2=0`: the tested V2 adapter rejects native kernel
launches with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`. V1 still executes the AOT
GPU images through Level Zero.

The Xe compute engine initially had `preempt_timeout_us=640000` and
`job_timeout_ms=5000`. Carbon CT minibeam failed with `DEVICE_LOST` and the
kernel log recorded CCS engine resets. Raising **only** the preemption timeout
to `10000000`, keeping the job timeout at `5000`, allowed the unchanged complete
binary to finish two successive 10,000-history batches with all audits passing.
The patient-primary kernels took approximately 1.6–1.9 seconds in the initial
successful two-timeout test; these timings diagnose the timeout and are not a
throughput benchmark.

The remote B580 is now configured with the 10-second preemption timeout; the
setting resets on reboot. The executable does not change driver settings.
See the [README workflow](../README.md#intel-arc-build-and-run) for the explicit
administrator command. Larger workloads must also fit the driver's job timeout;
use appropriate history chunks/batches without truncating physical trajectories.

## Validation

The full five-pipeline native AOT executable was used for the release-preset
checks. Each preset ran two batches of 10,000 histories, with electrons off:

| Preset | Runtime/audit result |
|---|---|
| Proton water broadbeam | Pass |
| Proton CT broadbeam | Pass |
| Carbon water broadbeam | Pass |
| Carbon CT broadbeam | Pass |
| Proton water minibeam | Pass |
| Proton CT minibeam | Pass |
| Carbon water minibeam | Pass |
| Carbon CT minibeam | Pass with the Xe preemption setting above |

Checks cover finite nonnegative dose, nonzero scoring, energy/EM/coverage and
completion audits, and zero queue overflow. The carbon CT minibeam preemption-only
confirmation had physical relative energy residuals of `1.233e-6` and `1.083e-6`.
These are runtime checks, not new TOPAS dose acceptance or performance claims.
The generic proton and carbon pipelines also passed two 10,000-history batches each.
Electron-on Arc execution has not yet been tested.

The CUDA `sm_75` build was tested with PTX JIT disabled against the unchanged
control: all eight presets, two 20,000-history batches per arm, passed with exact
source allocations/discrete audits and maximum dose relative L1 `3.037e-7`.
Generic proton/carbon and two electron-on CUDA cases also passed, as did all 90
retired-interface checks.

Local evidence is under `tmp/intel_arc_support_20261015/`:

- `lifetime_regression/`: final CUDA matched regression and interface checks.
- `remote/arc_checks_final/`: seven release presets and generic runtime checks.
- `remote/arc_carbon_xe_preempt/`: carbon CT minibeam preemption-only confirmation.
- `remote/arc_carbon_xe_preempt.log`: unchanged full executable, default physical
  track limits, original workgroups and secondary-tail scheduling.

The remote verified executable is
`/home/v/MAIGO_arc/build-intel-split/carbon_mc`. Other build directories created
while diagnosing the driver issue contain experiments and are not release builds.

## Full-plan proton broadbeam throughput

On the tested B580, each case ran three consecutive 20,000,000-history batches
with the current implementation. Electron transport was off, primary straggling
on, secondary straggling off, medium nuclear elastic on, and the broadbeam step
cap 1 mm. Original full-plan CT, source and weights were retained.

| Case | Cold histories/s | Subsequent histories/s | Batch wall times (s) |
|---|---:|---:|---|
| RT07575 (Head 1) | 638,764 | 662,277 | 31.044, 30.217, 30.181 |
| RT06423 (Head 2) | 533,996 | 572,149 | 37.236, 37.170, 32.742 |
| 20022516 (Lung 1) | 553,719 | 572,731 | 35.904, 35.069, 34.772 |

Cold throughput includes process startup, package loading and first batch output;
compilation is excluded. Subsequent throughput is 40 million histories divided
by the combined wall time of batches 2 and 3, reusing one GPU context. All nine
batches passed runtime audits with zero overflow. GPU clocks were not locked and
OS page cache was not flushed; Head 2 showed visible batch timing variation.
These are single three-batch measurements, not new TOPAS accuracy acceptance.

The full-plan driver is compiled once for the destination GPU and reused across
cases; CT, weight or history-count changes do not require rebuilding it. The
standalone native AOT link used `-fsycl-max-parallel-link-jobs=8`; this affects
build preparation, not the timed Monte Carlo execution. Local evidence:
`tmp/arc_bb_three_20261015/RESULTS.md` and `COMPLETE_compiled.json`.
