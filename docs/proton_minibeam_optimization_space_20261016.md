# Proton minibeam optimization space — 2026-10-16

## Current cost and remaining opportunities

Before the upstream/pool optimization below, original optimized full-plan warm timing on RTX2080Ti puts Cu/air primary at51.45%/48.69%/36.64%, patient primary32.89%/36.02%/43.19%, other15.66%/15.29%/20.17% for Head1/Head2/Lung. These are combined Cu/air event times, not isolated copper function costs. Native step census shows289.54/253.23/178.45 Cu steps per incident history. Only about10–12% of incident histories execute a patient step; that is not a dose-contribution fraction. Evidence: tmp/proton_mb_continuation_20261015/SUMMARY.json and tmp/proton_mb_track_work_20261015/.

Hypothetically reducing Cu/air time20% increases whole-call throughput11.47%/10.79%/7.91%; halving it gives34.64%/32.18%/22.43%. These are Amdahl scenarios, not measured achievable gains. Native Cu0.05mm and patient0.5mm caps, primary fluctuations and native optical depths remain. Ordinary launch overhead is already small. A large further gain needs a substantially cheaper per-step kernel or a separately validated change in statistical estimator/work. Fixed upstream phase-space reuse amortizes repeated calculations only; weighted source roulette requires complete primary/descendant/scorer/energy-weight propagation and ROI variance validation. Neither is implemented here.

## Rejected audit sharding

A64-way successful-Cu-loss counter avoids a per-history live accumulator and reduces exactly into the canonical uint64 audit slot. Unlike the previously rejected per-history aggregation, it keeps atomic increments per draw. Fullall5 sm75AOT/JITdisabled matched original3M fullplans/three1Mresidentbatches ABBA gave Head1+.156%,Head2+.191%,Lung+.332%; head pairs mixed, no resolved targetCu/air gain. All36batch audits/source/discrete counters pass,maxdoseL1=7.251e-8. Extra allocation/reduction is not retained. Evidence: tmp/proton_mb_cu_shards_20261016/SUMMARY.json.

## Retained native Cu fluctuation math

Only dedicated ProtonMinibeam primary Cu restricted-fluctuation sampling enables existing SYCL native normal/log/exp/sqrt/pow helpers. Universal/Gamma/Poisson algorithms, rejection guards, zero-event atoms, explicit tails and separate hard-delta sampler remain. Carbon primary, proton Cu children, generic Cu callers and patient loss use their previous arithmetic. Both proton water/CT minibeam primaries use the same Cu implementation. No CUDA-specific new intrinsics, data packages, knobs or physics/cap changes. Native functions have finite precision; this is not bitwise trajectory equivalence.

GPU fixture:18 energy/path conditions from5–250MeV and0.0001–0.05mm,1048576 paired and independent samples/condition. Invalid0, paired RNG-count differences0, aggregate paired lossL1=1.7112e-8,maxabs1.0133e-6MeV; independent max mean/varianceZ=1.4822/1.5965. These synthetic inputs check sampler behavior, not production throughput.

Fullall5 RTX2080Ti native sm75 AOT/JITdisabled, original optimized fullplans, primary fluctuationsON/secondaryOFF/patient0.5mm/Cu0.05mm/elastic+inelastic retained. Each campaign3M globally allocated once acrossthree resident batches; firstbatch excluded. ABBA plus independent-seed BABA per case,8Mwarm/arm/case,72M simulated total; no compilation or other GPU jobs overlap timing. Clocks unlocked.

| Case | Control histories/s | Native Cu histories/s | Whole-call gain | Cu/air time reduction |
|---|---:|---:|---:|---:|
|RT07575 Head1|242730|248886|2.536%|4.035%|
|RT06423 Head2|257174|261341|1.620%|3.887%|
|20022516 Lung|249025|252940|1.572%|3.913%|

## Checks and limits

All72fullplan batches pass runtime/energy/coverage/completion/zerooverflow; maximum energy residual9.05424e-08. Per-spot source allocation/seeds are exact. Small floating-point differences cause some different full trajectories and discrete counters; largest matched single-batch doseL1=0.00340612, not accumulation-order-only equivalence.

Head1 uses two independent3M campaigns per arm (6M total), summing each campaign's batch doses once and averaging the two campaign estimates; repeated timing runs are excluded. GPU dose is mapped from the original rotated CCTG frame by reshape(417,35,505), transpose(1,2,0), then reversal of the last axis, matching the archived proton setup. The first private gamma script omitted this mapping; its result was rejected and retained as TOPAS_GAMMA_INVALID_LAYOUT.json. No dose scaling was fitted. Against existing independent TOPAS1B, frozen BODY/reference>=0.4Gy, local3%/0.2mm/search0.02mm: 75.988298%→75.983387% (-0.004912pp), within1pp relative gate. This low-statistics scoped check is not new absolute/high-statistics TOPAS acceptance and does not replace archived1B tables.

Allsix unaffected release presets pass strict matched implementation checks. Both affected proton minibeam presets pass source/runtime audits; finite-arithmetic differences are recorded instead of requiring identical changed trajectories. Generic protonCTMB remains on standard Cu math and passes matched checks. No throughput claim from these20k/batch checks. Evidence: tmp/proton_mb_cu_native_math_20261016/{SUMMARY.json,TOPAS_GAMMA.json,RELEASE_AUDITS.json,fixture.jsonl,runs/}. Exact measured full executable/library installed in tmp/remove_electron_20261015/build/. No commit/push performed.

## Retained shared Cu/air continuation and larger primary pool

The dedicated proton CT minibeam beamline now uses the same state-preserving continuation as carbon CT, with 64 native steps per segment. The default primary pool is 1,048,576; explicit custom sizes remain authoritative. Physical models, native ionization/nuclear clocks, primary fluctuations, Cu native math, patient 0.5 mm / Cu 0.05 mm caps and slit-air settings are unchanged. Proton direct Cu children retain their previous whole-track queue.

Direct final AB and independent-seed BA comparisons against the previous scheduling and 262,144-history pool use the three optimized full plans on both GPUs. Gains are approximately 21.5% / 24.5% / 20.6% on RTX 2080 Ti and 20.6% / 24.1% / 19.9% on A6000 for Head1 / Head2 / Lung. All twelve final paired gains are positive. These are measured combined changes, not multiplied gains from separate experiments. All full-plan source, discrete-event and live-ion audits match; energy, coverage, completion and zero-overflow checks pass, as do the eight-preset and generic implementation checks.

Details, build/batch scope, raw provenance and the unchanged-physics state contract: [proton beamline continuation report](proton_minibeam_beamline_continuation_20261016.md). The cost fractions and Amdahl scenarios at the start of this document describe the earlier baseline.
