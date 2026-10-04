# Four CT presets: throughput and acceptance, 2026-10-05

This campaign follows cleanup commit `3fc828f` and the secondary elastic-guard correction `5cc096d`. All runs use FP32 dose scoring, with FP64 dose scoring disabled. The delivered performance change is carbon CT minibeam scheduling: **5,165 → 10,048 primary/s**, a **1.946×** resident-batch speedup. No candidate C++ specialization or electron-worker rewrite was promoted. Water performance and dose campaigns remain deferred.

## Final results

Measurements use an RTX2080Ti, the CUDA SYCL Release executable from `5cc096d`, and automatic GPU clocks. Resident throughput excludes each process's first batch; it includes subsequent batch setup, transport and output. Whole-process times include startup and output. These four scenes have different sources and geometries, so their rates are not particle-species-only comparisons.

| CT preset | Histories | Resident primary/s | Dose result | Decision |
|---|---:|---:|---|---|
| Proton broadbeam, RT07575 | 10 × 1,253,150 | 167,692 | 3D local 1%/1 mm: **99.9254%** | Pass; keep existing scheduling |
| Carbon broadbeam, RT07575 | 7 × 925,987 | 104,144 | 3D local 1%/1 mm: **99.8677%** | Pass; keep existing scheduling |
| Proton minibeam, lung 170 MeV | 10 × 2,000,000 | 51,821 | Projected 2D local 3%/0.2 mm: **98.3183%** | Fails >99%; no performance candidate promoted |
| Carbon minibeam, lung 250 MeV/u | 2 seeds × 3 × 200,000 per arm | **10,048** (control 5,165) | Dose accuracy deferred | Scheduling regression passes |

Broadbeam process times are 80.35 s proton and 68.25 s carbon. The proton minibeam process takes 390.48 s. Maximum physical relative energy residuals are respectively 5.08e-7, 3.77e-6 and 3.46e-7. Every reported run passes quality, Unified EM, electron coverage, completion and zero-overflow checks. A runtime quality pass does not mean a TOPAS dose acceptance pass.

Machine-readable timings, gamma metadata, matched comparisons and the final eight-preset regression are in [ct_throughput_20261005.json](ct_throughput_20261005.json).

## Dose scope and limits

The broadbeam presets use the RT07575 plans and retained TOPAS 3D references. The evaluation includes the full reference ≥10% maximum mask, with absolute plan normalization, local 1%/1 mm and a **0.1 mm** spherical Cartesian-lattice search. There is no BODY-only exclusion, fitted dose scale, registration or smoothing. Dose-sum biases in that mask are −0.0382% proton and −0.5120% carbon. Reference arrays are retained under `out/broadbeam_shared_electron_20261004/`.

The minibeam presets use lung CT. Proton's existing comparison uses the TOPAS 200M BODY-projected mean over |y|<15 mm, |x|<15 mm and reference ≥3% maximum. This is a **2D projected gamma**, not a 3D voxel acceptance or RT07575 minibeam full-plan validation. The source, reference, mask and normalization were not changed to improve the pass rate.

The 20M proton control uses the historical PyMedPhys 0.41.0 radial search, scipy linear interpolation, `interp_fraction=10`, and **0.02 mm** search spacing:

| Projected local 3%/0.2 mm scope | Pass rate |
|---|---:|
| Full evaluation field | 98.3183% |
| Peak strips | 99.9605% |
| Valley strips | 99.0052% |

The dose-sum bias across the evaluation mask is −0.01768%. An independent Cartesian-lattice evaluator at 0.02 mm gives 97.1013%; steep minibeam gradients make the search discretization material. Both results are preserved, and neither reaches the user's >99% gate. The historical older-model 200M lung result was 97.9345%. These finite-statistics results do not establish a zero-noise limit. New proton minibeam performance candidates remain unaccepted under the requested gate; physics accuracy needs separate work.

Carbon minibeam dose accuracy remains explicitly unaccepted, as authorized. Its matched-dose regression below tests the scheduling change against the current GPU model, not against TOPAS.

## Accepted carbon minibeam scheduling

The preset now uses existing shared scheduling mechanisms:

```yaml
history_chunk_size: 131072
material_electron_queue_chunk: 131072
secondary_species_grouping: true
secondary_step_chunking: true
```

Control chunks were 16,384 with both secondary switches false. The change preserves group32 births, 16,384 electron lanes, the 576 MiB electron queue budget, physical step limits, species data and per-history RNG. Secondary continuation uses the existing 16-step segments and 8,192 tail threshold.

Final confirmation uses the **same production binary for both arms**. Seed 1001902102 runs control then final; seed 1001902202 reverses that order. Each process runs three batches of 200,000 primaries.

| Metric | Control | Final |
|---|---:|---:|
| Seed 1 resident primary/s | 5,165 | 10,026 |
| Seed 2 resident primary/s | 5,164 | 10,070 |
| Combined resident primary/s | 5,165 | **10,048** |
| Mean resident batch wall time | 38.724 s | **19.904 s** |
| Mean resident primary kernel | 22.320 s | **7.565 s** |
| Mean resident secondary kernel | 6.917 s | **3.498 s** |
| Mean resident electron kernel | 8.507 s | **7.939 s** |
| Whole-process primary/s, including each startup | 4,880 | **9,057** |

The combined rate divides total resident histories by summed resident seconds; it is not the arithmetic mean of run rates. Each 600k final process takes about 66.2 s, versus 122.5–123.4 s control.

The two comparisons preserve 98,997 and 98,998 discrete ledger fields, all reported EM/electron work counters and dose support exactly. Per-batch dose relative L1 is at most 5.01e-8, consistent with FP32 accumulation order. Maximum final physical relative energy residual is 1.294e-6. Queue peak is 3,798,118 tasks against capacity 5,807,497, a 1.53× margin for these cases; that is not a guarantee for arbitrary sources. No task, coverage, or overflow failures occurred.

Chunk-only screens showed about 5,046 / 7,612 / 8,532 primary/s at 16k / 64k / 128k. Subsequent grouping and continuation screens motivated the final configuration. Some early screens used an experimental C12 specialization; the final production-binary pairs above remove that confound.

## Rejected candidates and hardware evidence

All rejected routes and retry conditions are recorded in [failed.md](../failed.md).

- **17,408 electron workers:** tests all four CT presets, matching 68 blocks to 68 SMs instead of 64. Audits and dose match, but throughput changes (+0.65%, −2.14%, −1.78%, −0.95%) show no useful net gain. Unlocked thermal/order variation prevents precise claims about these small changes. Keep 16,384 workers.
- **Compact electron failure state:** retained diagnostics shrink 296→104 bytes, with 100,000 randomized log records matching. Static resources improve from 255→237 registers and 1712→1696 stack bytes, but measured worker time is unchanged. Corrected reverse-order comparisons at 1500 MHz give proton broad +1.086%, carbon broad −0.130%, proton mini +0.289% overall, without a resolved electron-kernel gain. Keep the original record.
- **C12 primary specialization:** exact audits and dose, but an initial +1.74% screen falls to +0.308% in an independent reversed pair at 1500 MHz. No useful gain resolved; retain the generic shared C12 Urban implementation.
- **Secondary tail thresholds 2,048/512:** carbon Urban segmented tails were substantial enough to satisfy the historical retry condition. Same-binary rates at 8,192/2,048/512 are 10,009/9,845/9,946 primary/s. Fewer tail seconds do not repay extra rounds. Keep 8,192; no experimental environment switch was added.

Actual Nsight Compute data for the compact-state trial (100k proton lung launch): worker time 1.240654→1.240934 s, local load sectors 5.281B→5.207B, store sectors 4.061B→4.001B, active-warps occupancy 24.767→24.745%, and active threads per issued warp instruction 11.035→11.012. Local-memory requests are not exclusive evidence of register spills. The remaining electron bottleneck requires a larger reduction in hot state or memory traffic; lowering static registers alone did not improve this workload.

The first locked-clock driver accidentally selected the control executable for both labels. Those `locked_*` runs are A/A evidence only. Only corrected `locked_v2_*` pairs informed candidate decisions. Automatic GPU clocks were restored after profiling.

## Verification and reproduction

The final production binary and current YAMLs pass two-batch regression in all eight presets, including water shared-worker smoke checks. These small runs test implementation consistency, not water dose acceptance. Package checks pass for 171 electron payload files and 218 installed extraction-kit package entries. The active source/config/extractor scan has no SHA256/hashlib/OpenSSL/ZSTD checking references; exactly eight YAML presets remain, with both overlooked legacy source templates removed. Historical TOPAS data and archived experimental records are retained.

`out/ct_throughput_20261005/` preserves frozen binaries, configurations, commands, per-batch dose and quality reports, ledgers, telemetry, profiler reports and scripts. `harness.py` runs native repeated batches; `compare.py` checks discrete work and FP32 dose; `gamma.py` and `independent_mini_gamma.py` retain both gamma methods. `confirmed_carbon_*` contains the final pairs, and `final_preset_regression/` contains the final eight cases.

The supported benchmark tool now uses native repeated batches for every scene instead of attempting a single 6.48M-history carbon broadbeam allocation. It checks every batch and reports process, transport and resident throughput:

```bash
python3 scripts/benchmark_ct_devices.py \
  --root . --binary build/carbon_mc \
  --lib /home/wuwei/sycl_workspace/llvm/build/install/lib \
  --output out/ct_device_smoke --smoke-only
```

Use a new output directory. Omit `--smoke-only` for the full repeated campaign. The default proton minibeam benchmark is 2M histories for timing, not the 20M dose control reported above. GPU profiling/timing jobs must run without competing GPU workloads.
