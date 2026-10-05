# Exact EM search reuse in CT transport — 2026-10-05

The shared EM module now reuses same-step search indices between certified adjacent density nodes. It retains the original interpolation, loss fluctuations, RNG, step limits and FP32 dose. This follows the already accepted electron birth ordering and tissue roulette; neither is changed here. **300k primary/s has not been reached.**

| RT07575 broadbeam | Resident primary/s, control → selected | Change | Whole-process primary/s | Pooled 3D local 1%/1 mm |
|---|---:|---:|---:|---:|
| Proton | 201,356 → **203,021** | +0.83% | 186,232 → 186,364 | **99.8935%**, unchanged |
| Carbon | 129,244 → **138,019** | +6.79% | 116,342 → 122,956 | **99.8699%**, unchanged |

Proton's change is small: paired gains are +0.38% / +1.27%, with automatic GPU clocks and recorded telemetry. Carbon's two independent pairs improve +5.45% / +8.12%, establishing a useful gain. Do not treat differing absolute rates from older campaigns as an additional speedup. Likewise the new gamma values use new seeds; this implementation does not improve dose matching—the matched controls have exactly the same passing fractions.

## Implementation and scope

The actual Unified EM package has 175 material nodes and 18 charged species. All 149 adjacent same-section material pairs share, for each species, the node-energy grid and the stopping/range spline search keys. The loader verifies the keys, counts, mass-number and energy-coordinate ratio once, after normal package validation, and stores the certificate with the immutable package. Existing process-level input caching retains it across batches.

Preparation locates the lower endpoint's three intervals, reuses only those indices for the upper endpoint, and independently evaluates both endpoints' original table values and density scaling. No interpolation coefficients, reciprocal ranges, floating-point expressions or draws are approximated. Inverse-range and hard-rate grids are not reused. A different package whose grids do not meet the certificate uses the two original searches. There is no persistent per-track interval cache, new YAML switch, new response package or research environment override in the selected runtime.

The code is in `include/carbon/unified_em_view.hpp`, `include/carbon/unified_em_package.hpp`, `src/unified_em_package.cpp` and the EM-view setup in `src/transport_sycl.cpp`. It applies to the common primary/secondary ion EM preparation, including other presets; electron transport and electron-response data are unchanged. Carbon minibeam remains outside dose-accuracy acceptance.

## Matched validation

RTX 2080 Ti; FP32 dose ON, FP64 dose OFF. Each arm has two independent seeds (1001903202/1001903302), with the arm order reversed on the second seed. Proton: 10×1,253,150 histories per seed; carbon: 7×925,987. Resident rate uses complete batch walltimes after batch0; whole-process rate includes package loading and outputs. GPU runs are serialized; final full-plan timings have no concurrent compilation or GPU profiler. The native repeated-batch interface explicitly resizes TPS history allocations.

All 68 broadbeam validation batches pass runtime quality, finite nonnegative dose, energy accounting, task completion, response coverage and zero queue overflow. All compared discrete ion/EM/electron work is identical. Maximum dose relative L1 is proton 4.823e-08, carbon 3.515e-08.

Gamma uses the full 3D reference grid at >=10% reference maximum, absolute dose normalization, local 1%/1 mm, spherical Cartesian search step 0.1 mm, linear interpolation, no fitting or registration. Both individual seeds and both pooled comparisons exceed 99%. References and per-seed results are recorded in the JSON and each run's `gamma.json`.

Component checks cover 151,539,660 CPU cases, 1,492,992 valid GPU cases across all18 charged species, and native node/spline boundaries with adjacent FP32 values. Prepared fields, sampled losses and RNG states match exactly. The committed standalone regression checks28,350 boundary cases and four altered-grid/coordinate fallback cases. Run it with:

```bash
g++ -std=c++20 -O2 -DCARBON_EM_EXACT_INDEX=1 -Iinclude   tests/unified_em_shared_prepare.cpp src/unified_em_package.cpp   -o /tmp/maigo_em_shared_test
/tmp/maigo_em_shared_test
```

## Hardware evidence

Separate Nsight Compute runs use one matching kernel from a native100,000-primary batch; carbon secondary is the sixth matching launch. These profile the frozen v1 device implementation, before moving its host certificate check from every batch to package loading. Profiler time is not whole-process throughput.

| Metric, control → shared search | Proton primary | Carbon primary | Carbon secondary |
|---|---:|---:|---:|
| Profiled kernel ms |333.123 →327.031|221.366 →209.161|15.258 →13.262|
| Registers/thread |217 →215|215 →215|212 →223|
| Active threads/instruction |21.103 →20.959|22.572 →22.539|27.367 →27.252|
| Warp instructions, billion |10.332 →10.122|5.598 →5.366|0.354 →0.339|
| Local-load sectors, million |2352.263 →2325.975|824.381 →796.356|60.242 →49.670|
| Local-store sectors, million |525.691 →525.703|185.531 →185.544|37.854 →33.759|
| Long-scoreboard / issued-active ratio |6.233 →5.982|14.137 →13.829|18.949 →15.978|

Carbon secondary improves despite higher registers; occupancy stays approximately23%. These counters support reduced instruction/local-memory requests and dependency waits, not DRAM-bandwidth saturation or proof that every local access is a register spill. Proton's no-instruction wait ratio remains high (11.256→11.638), consistent with its weak whole-run gain. The same-binary v1 OFF→ON carbon screen improves4.05%; production→v1 ON secondary time also includes code-layout changes already visible in the OFF arm.

## Rejected prototypes

- Conservative division-free CT face guard: exact CPU/GPU geometry and CT dose, but no useful throughput gain.
- Passing known primary Z/A through EM templates: exact work, no useful whole-run gain.
- Additionally folding the metadata-proven native-ion mean branch: exact work, no useful gain; fluctuation sampling was preserved.
- Shared-search v1 with per-batch certification: rescanning immutable1.3GB node arrays adds approximately0.315s per resident proton batch in the screen and removes the useful kernel gain. The selected implementation certifies once on package load; later EM setup returns to about0.00008s.

All rejected candidates and explicit retry conditions are recorded in `failed.md`. Earlier failed step-size, block-count and persistent interval-cache sweeps were not repeated.

## Other presets and evidence

The final binary also passes matched three-batch CT minibeam transport/dose comparisons:

| Preset | Histories | Resident rate | Screen change |
|---|---:|---:|---:|
| proton CT minibeam | 500,000 × 3 per arm | 49,147 → 50,524/s | +2.80% |
| carbon CT minibeam | 200,000 × 3 per arm | 9,965 → 10,429/s | +4.66% |

These are single-seed regression screens, not a new minibeam gamma or robust large-statistics performance claim. The previously reported proton minibeam dose gate remains unmet; carbon minibeam dose acceptance remains deferred.

All six other presets pass two-batch regression against previous accepted smoke outputs, with matching electron/EM/transport audits and dose relative L1<1e-6. Water performance and dose have not been newly qualified.

Frozen binaries, complete source patches, native batch outputs, telemetry, component tests and NCU reports are under `out/broadbeam_ions_20261005/`. `validate_shared.py`, `screen_final_minibeam.py`, `regress_other_presets.py` and `summarize_final.py` reproduce the checks. The selected binary is installed as `build/carbon_mc`; no failed prototype is installed. See [machine-readable results](broadbeam_em_search_20261005.json).
