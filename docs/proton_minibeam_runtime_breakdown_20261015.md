# Proton CT minibeam runtime breakdown — 2026-10-15

Latest electron-removed runtime469d283, current release physics defaults, original optimized full-plan CT/source/placement/collimator retained for each case. RTX2080Ti native sm75 AOT; CUDA_DISABLE_PTX_JIT=1. All five production pipelines linked from the current electron-removal library; host driver additionally prints existing primary/secondary event timers, without transport edits. No performance candidate or default change.

Each case15M histories, three balanced5M batches. Largest remainder across source campaign counts once; exact source float strings retained, new independent per-batch seeds. One resident context per case. First batch excluded from warm timing; no claim from smoke tests. All nine batches pass runtime energy/coverage/completion/zero-overflow gates. Dose is not a new TOPAS comparison.

| Case | Warm histories/s | Patient primary s/M | Cu/air primary s/M | Medium secondary s/M | Cu primary-child queue s/M | Other incl ordering s/M | Total s/M |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT07575 | 256788 | 1.3927 | 2.1042 | 0.2978 | 0.0039 | 0.0957 | 3.8943 |
| RT06423 | 272084 | 1.4264 | 1.9325 | 0.2189 | 0.0031 | 0.0944 | 3.6753 |
| 20022516 | 225274 | 2.4777 | 1.5187 | 0.2886 | 0.0019 | 0.1521 | 4.4390 |

Primary patient step counts per incident: RT07575 59.6102, RT06423 56.2640,20022516 72.8550. Patient-primary shares35.76%,38.81%,55.82%; upstream primary54.03%,52.58%,34.21%. Effective patient step throughput42.80M/39.44M/29.40Msteps/s. Lung incurs both more steps and lower aggregate steps/s; stage averages do not isolate memory stalls, divergence, or individual physics samplers.

Cu primary-child queue time measures only the separately timed worker; no claim that it covers every copper-fragment operation. Unattributed wall remainder includes setup/transfers/scoring/output/untimed work; not all assigned to secondary. Secondary grouping/chunking both defaultfalse for protonCTMB, so the carbon segmented-tail argument does not transfer. Cu and air are one timing bucket; no separate attribution yet.

Amdahl scenarios (not measured gains): halving patient-primary time alone yields about21.8%,24.1%,38.7% whole-run speedups for Head1,Head2,Lung. A20% patient time reduction gives7.7%,8.4%,12.6%. The earlier20261010 proton1/4 screen used1mm,old runtime and enabled secondary fluctuations; its16% patient share is not representative of these measurements. The original timing runs did not measure the clock-limiter winning fraction; the later census below supplies that count. Patient share does not prove that clock thinning will achieve those scenarios. Prioritize lung for an isolated current-version clock proposal screen only if further optimization is requested, keeping primary fluctuations and evaluating spatial dose independently.

Evidence: `tmp/proton_mb_breakdown_20261015/{RESULTS.json,inputs.json,provenance.json,COMPLETE.json}`. Source config,source CSVs,raw stage logs and per-batch audits retained. No1B campaign resumed.

## Further primary attribution

Existing two warm full-plan batches per case give these counts per incident:

| Case | Cu native loss draws | Air Urban steps | Air ionization proposals / air steps | Patient Urban steps | Source weighted mean MeV |
|---|---:|---:|---:|---:|---:|
|RT07575|289.461|16.031|89.96%|59.610|115.893|
|RT06423|253.408|17.072|90.84%|56.264|106.769|
|20022516|178.494|18.445|91.88%|72.855|85.704|

Cu draw count is the successful native primary-loss audit, not an exact count of all Cu loop entries (e.g. range-terminal/no-fluctuation branches are excluded). All primary fluctuations are enabled. Air clock consumption marks a proposal when the accepted true path reaches the candidate; high proposal frequency shows the clock is frequently reached, not a proof that changing it preserves angular/spatial dose or will save a particular fraction of wall time. Cu and air work counts do not establish separate timing percentages.

Cu repeatedly performs surface/material navigation, nuclear elastic/inelastic rate competition, Urban true/geometric path/scatter, native range/DEDX/restricted loss plus universal-delta draws, and audit updates. Its0.05mm cap remains much tighter than patient's0.5mm. The source-energy ordering is consistent with the Cu-work-count ordering across these different plans; it is not a controlled causal energy experiment. Patient repeatedly performs CT navigation/material-density selection, continuous26 prepared moments/slopes, ionization proposal/1%meanloss/Urban limits, nuclear hazard queries, final scatter, restricted+delta stochastic loss, scorer/audit updates. Actual nuclear event frequency is much lower than per-step query frequency; this does not make rate queries free.

Nsight Compute hardware metrics were unavailable: ERR_NVGPUCTRPERM, RmProfilingAdminOnly=1, sudo requires a password. No global driver setting was changed. Nsight Systems CUDA-only traces succeeded without kernel source changes; one matched5M full-plan batch per case, no new throughput claim from profiled runs. SourceCSV and exact Cu/air/Urban/elastic/cascade audit counts match the original same batch.

| Case | Upstream kernel sum s | Patient kernel sum s | Between-primary-kernel gaps s | Gap share of primary span |
|---|---:|---:|---:|---:|
|RT07575|10.4011|6.7081|.00915|.0534%|
|RT06423|9.4659|6.8911|.00861|.0526%|
|20022516|7.4041|12.0429|.00831|.0427%|

Each phase has20launches/block128; full chunks262144 lanes. Upstream232registers/thread, patient255. These are static kernel resources reported by the trace, not measured occupancy/spill traffic. Do not infer memory-bound behavior or actual spilling from them. Lung full-size patient chunks vary418–738ms, vs Head1 289–385ms; heterogeneous chunk costs are measured, but divergence versus memory-versus-physics causes are not isolated. Queue launch gaps are negligible in these traces.

The strongest supported conclusions are: upstream executes hundreds of Cu loss draws per original history, while patient executes fewer but more expensive CT/EM/Urban steps; lung has more patient steps and lower aggregate steps/s; host launch overhead and species sorting are not the dominant primary cost. Exact percentages for loss/MSC/geometry within either monolithic kernel still require permitted hardware profiling or carefully qualified intrusive profiling. No such percentages are fabricated here.

Evidence: `tmp/proton_mb_breakdown_20261015/primary_work_counts.json`, `tmp/proton_mb_primary_trace_20261015/{RESULTS.json,AUDITS.json}` and per-case .nsys-rep/.sqlite traces.

## Accepted patient-step limiter census

A separate intrusive counting build, frozen469d283 source, uses only the proton-minibeam pipeline and native sm75 AOT/JITdisabled. Each case reuses the exact original timing campaign batch000 source, histories and per-spot seeds (approximately5M incident histories/case). No step ceiling, fluctuation, clock, nuclear, scattering or scoring behavior changed. Counter overhead means these runs are **not throughput benchmarks**. The production source/build is unchanged.

Every accepted primary CT step receives one category: the initial continuous26 range-derived limit competes with the ionization clock; strictly shorter combined1%mean-loss,0.5mmcap,Urban/range,voxel-face and nuclear limits replace that category in execution order. Initial equal clock/range ties belong to the clock; subsequent exact ties retain the existing category. True/geometric conversion alone is not a new limiting process. This describes the **executed final winner**, not the counterfactual number of steps saved by removing a process. Continuous26 near-terminal range limits remain in its initial category; the tiny final `range` category denotes a further Urban range clamp only.

| Final limiter, fraction of accepted CT primary steps | RT07575 Head1 | RT06423 Head2 | 20022516 Lung |
|---|---:|---:|---:|
| CT voxel face |46.3630%|46.2061%|43.3862%|
|0.5mm cap|24.9280%|25.2689%|20.9139%|
|Ionization segmentation clock|16.3281%|15.5437%|25.8046%|
|Combined1%mean-loss guard|9.8310%|10.3473%|8.1295%|
|Continuous26 range-derived step|2.4896%|2.5759%|1.7343%|
|Nuclear candidate|0.0603%|0.0582%|0.0315%|
|Urban MSC further shortening|0|0|0|
|Additional Urban range clamp|<0.000002%|<0.000002%|<0.000002%|

Urban is still evaluated and angular/lateral scattering sampled; a zero final-MSC-limiter count does **not** establish that Urban costs no time. Likewise sparse nuclear winners do not measure the cost of per-step hazard queries.

| Work measure | Head1 | Head2 | Lung |
|---|---:|---:|---:|
|Accepted primary steps/incident|59.4760|56.4388|72.8965|
|Mean accepted true step,mm|0.227748|0.229293|0.214522|
|Sum accepted CT true paths/incident,mm|13.5455|12.9410|15.6379|
|Step fraction at density<0.5g/cm³|29.9661%|39.8829%|69.7143%|
|Step fraction at energy<20MeV|4.5948%|4.9071%|9.7128%|

Denominators include all original incident histories, including those stopped upstream. The CT phase includes low-density/air voxels inside the CT grid; it is not a BODY-only timing/step mask. These figures explain the workload distribution without proving which subroutine causes lung's lower aggregate steps/s. Material, density, source energy, track length and execution divergence differ together across cases.

All three quality/energy/coverage/completion gates pass, zero queue overflows. Original unifiedEM/Cu/air/Urban audits match exactly;205/206/206 non-energy integer ledger fields match per case, including the live tracked-species counters. FP32 dose relativeL1 differences versus the original same-source batch are9.75e-8/1.08e-7/8.33e-8, consistent with accumulation order. No new TOPAS or spatial-accuracy acceptance is implied.

### Consequences for the next optimization

- Carbon's clock-dominated patient result should not be transplanted as a proton performance prediction. Proton's clock wins16–26% of these steps; geometry/cap win64–71%, and another process can take over when a candidate is removed. Counts are not runtime shares.
- Preserve primary fluctuations and current caps. Prioritize reducing work **within the existing accepted step**: reuse a prepared continuous26 point through coverage/preparation/mean/loss where equivalent, and inspect repeated coordinate/index calculations without replacing CT material lookup with a scorer index. Some source-level repeated expressions may already be compiler-eliminated; only a matched implementation experiment can demonstrate a gain.
- `ct_sample` uses directed ceil/floor index selection, while `urban_voxel_axis` also corrects against representable centered faces. The two semantics must remain distinct at float boundaries. Simply using Urban/scorer indices as CT material indices would not be an exact refactor.
- The apparent wasted inelastic rate evaluation before `HadronicRateCache::update` is real at source level, but the analogous stage15 lazy-query candidate already had no stable gain on20261009. Do not repeat it as the main optimization without measured query cost or a materially different design.
- Copper still performs178–289 successful native loss draws per original incident history. Previous unchanged-physics CSV skipping, air-rate reuse and small audit aggregation did not resolve a stable useful gain. Separate Cu/air timing and native-loss/Urban cost remain unmeasured; no exact subfunction cost percentages are assigned from operation counts.

Evidence: `tmp/proton_mb_limiter_census_20261015/{provenance.json,RESULTS.json,AUDITS.json,COMPLETE.json,prepare.py,run.py,check_audits.py}`, private source/build and per-case census logs. No production diagnostics or new runtime configuration keys were added.

## Exact point-reuse screen

A pure proton CT primary same-step point-reuse candidate was tested after the limiter census. Unlike the old carbon point+audit combined experiment, this uses the current density-corrected/electron-removed runtime, changes only point lifetime/reuse, and keeps full restricted/delta slopes, native clock and original minibeam fluctuation sampler. Water and other pipelines remain on their original path.

20022516 lung optimized fullplan,3M histories per campaign,three balanced1M batches, ABBA with firstbatch excluded:4M timed histories/arm. Control209807.320/s, candidate210430.076/s(+0.2968%); paired ratios0.996416/1.009479. Patient event sum9.85546→9.86177s. GPU clocks unlocked; no stable useful benefit. All12batch audits/zerooverflow pass, source and nonenergy integer/live-ion/EM/Urban/nuclear counters exact,maxdoseL1=6.64e-8. Candidate **not integrated**. The earlier15M/case rates use5M batches; absolute rates from these different batch sizes are not a code-speed comparison.

Evidence: `tmp/proton_mb_point_reuse_20261015/` and `runs/20022516/abba/RESULTS.json`.

## Copper post-loss event-query deferral

The next candidate moves the **post-loss** copper elastic nearest-record search and inelastic-rate interpolation into `collision_in_step`. These rates only choose the nuclear process when the accepted true path has reached its candidate. The pre-step total hazard, persistent shared nuclear optical depth/debit, post-loss evaluation energy, branch acceptance, target/final-state generation, RNG dimensions and native primary/Cu fluctuations are preserved. The MCS direction update remains unconditional with respect to collision and precedes the reaction. No package/data or default/cap changes.

This differs from the previously rejected medium inelastic-cache query change: it guards a copper post-loss event-selection calculation, including a full-bank binary search, using an already-resolved candidate boolean. It does not cache or approximate an energy-dependent hazard.

Matched selective proton-minibeam builds, RTX2080Ti sm75 AOT/JITdisabled; each campaign3M optimized full-plan histories overthree balanced1M batches, firstbatch excluded. Head1 ABBA plus independent-seed BABA, Head2/Lung ABBA. Eight paired ratios are positive; clocks unlocked, so no fixed cross-hardware gain or small-regression attribution is claimed.

| Case | Control histories/s | Candidate histories/s | Whole-call improvement | Cu/air primary time reduction |
|---|---:|---:|---:|---:|
|RT07575 Head1|236990|240916|1.657%|2.913%|
|RT06423 Head2|245814|249794|1.619%|2.607%|
|20022516 Lung|204620|205848|0.600%|2.263%|

Head1 has8M warm timed histories/arm; Head2 andLung4M/arm. All48campaign batches accepted with zerooverflows/max relative energy residual1.027e-7. Per-spot source CSVs and non-energy discrete/live-ion/nuclear/Urban/EM counters match. Max single-batch dose relativeL1=8.66e-8. Repeated timing seeds are not additional independent dose statistics; these are implementation comparisons, not a fresh TOPAS simulation or changed-physics acceptance.

The measured source is integrated into the shared Cu primary path with indentation-only differences. It applies to the same Cu event-selection logic in proton/carbon water/CT minibeams and generic; measured throughput claims above cover only proton CT fullplans. Explicit full-build verification is recorded separately below. Point reuse is excluded from the integrated code.

Evidence: `tmp/proton_mb_cu_event_lookup_20261015/{pooled.json,provenance.json,pool.py}`, per-case campaign sources/audits/raw logs, and `integration/source_hashes.json`.

Full five-pipeline native sm75 production build completed. Matched two-batch implementation checks of all four minibeam release presets and generic protonCTMB/carbonwaterMB pass: source allocation/seeds and physical/live-ion event counters match; energy/EM/coverage/completion/zerooverflow pass. Max release-preset dose relativeL1=4.26e-8. These small-batch checks are not throughput evidence. Broadbeam does not execute this Cu block, so no new broadbeam physics test was added. Evidence: `tmp/proton_mb_cu_event_lookup_20261015/integration/{RESULTS.json,minibeams/summary.json,generic.json}`. Formal executable: `tmp/remove_electron_20261015/build/carbon_mc`.

### Full production-build timing confirmation

The original469d283 all-five-pipeline library embedded in the frozen timing driver is compared directly against the newly rebuilt all-five-pipeline library; same host driver,sm75 AOT/JITdisabled, original3M campaign allocation/seeds/case geometry and3×1M resident batches. No GPU timing overlaps compilation or implementation-regression GPU jobs. Head1 andLung use ABBA+independentseedBABA; Head2 ABBA. Every group's firstbatch is excluded from warm timing.

| Case | Full-build control histories/s | Full-build candidate histories/s | Pooled whole-call change | Cu/air primary time reduction |
|---|---:|---:|---:|---:|
|RT07575 Head1|237091|240527|+1.449%|2.885%|
|RT06423 Head2|245896|248524|+1.068%|2.484%|
|20022516 Lung|206836|207795|+0.464%|2.494%|

Head1 group changes+1.113%/+1.787%; one of its four individual pair ratios0.999560 is effectively flat. Head2 two pairs1.003735/1.017620. Lung group changes−0.623%/+1.574%, two pairs in each direction: **its small pooled whole-call gain is not resolved as stable**. Do not substitute the more favorable selective-build rates for these production-build results. The consistent2.5–2.9% reduction in the targeted Cu/air phase is diluted by patient transport, secondaries, setup/scoring/output and unlocked-clock noise. Patient event-time totals are essentially unchanged after pooling; no patient transport optimization is claimed.

All60full-build campaign batches pass runtime/energy/coverage/completion/zerooverflow; source and nonenergy discrete/live-ion/Urban/EM/nuclear counters match. Max matched single-batch dose relativeL1=8.69e-8. No new TOPAS simulation, physical approximation, source normalization or package changes. The shared Cu query deferral is retained; the failed patient-point candidate remains private.

Evidence: `tmp/proton_mb_cu_event_lookup_20261015/{COMPLETE.json,production_timing/pooled.json}`, per-case full-buildABBA/BABA records, and production/integration binary hashes. No commit/push performed in this optimization turn.

## CT proton primary continuation (retained)

Following the Cu-query baseline above, the dedicated proton CT minibeam patient stage now runs up to64 complete native loop iterations per segment, saves all cross-step state and pending scores, compacts unfinished histories, and orders them by current cached material section and energy. Below8192 active histories, or after64 segment rounds, a final kernel finishes all remaining native transport; this is a scheduling tail, not a transport cutoff. Source sampling, per-spot RNG identity, native ionization/nuclear optical depths, fluctuations, slopes, CT geometry and0.5mm patient/0.05mm Cu caps remain unchanged. Pending depth/voxel sums are preserved until the original scoring flush points. Resumed EM views rebind their table pointer to the new kernel capture.

The272-byte state plus two uint32 index arrays cost about70MiB at the default262144-history chunk; queues reject overflow without writing beyond capacity. This is enabled only in the dedicated proton-minibeam CT pipeline. Water, carbon and generic pipelines keep their existing scheduling. No public diagnostic/build/config switches or package changes are introduced.

Matched original full-plan allocations, native RTX2080Ti sm75 AOT with CUDA_DISABLE_PTX_JIT=1, full all-five-pipeline build. Each campaign3M histories/three balanced1M resident batches; firstbatch excluded. Head1 andLung use ABBA+independent-seed BABA, Head2 ABBA. Whole-call timing includes continuation state exchange, sorting, host synchronization and scoring/output; compilation does not overlap formal timing. Clocks unlocked, so these are workload-specific measurements.

| Case | Control histories/s | Continuation histories/s | Whole-call improvement | Patient primary time reduction |
|---|---:|---:|---:|---:|
|RT07575 Head1|236424|241138|1.994%|4.931%|
|RT06423 Head2|248413|255089|2.688%|6.561%|
|20022516 Lung|211230|253667|20.090%|31.850%|

All60 full-plan campaign batches pass source/discrete/live-ion/Urban/EM/nuclear and energy/coverage/completion/zero-overflow checks. Matched dose relativeL1 ≤1.118e-07, maximum relative energy residual 9.054e-08. All eight release presets plus generic protonCTMB/carbonwaterMB pass matched two-batch implementation regression, with exact non-energy ledger counters and dose relativeL1 ≤2.979e-07. These checks establish matched GPU implementation consistency, not a new TOPAS acceptance or smoke-test performance claim.

The prior source-spot-only patient ordering failed (Lung −0.863%, mixedpairs) and remains private; exact same-step point reuse also remains excluded. Dynamic continuation has a larger measured benefit in lung than the two head plans. It does not imply the same speedup for carbon or other GPUs.

Evidence: `tmp/proton_mb_continuation_20261015/{SUMMARY.json,COMPLETE.json,RELEASE_AUDITS.json}`, `production_timing/runs/`, frozen source and native build logs. Latest tested full release executable: `tmp/remove_electron_20261015/build/carbon_mc` (byte-identical installation copy of `full_candidate/build/runtime/carbon_mc`); paired performance drivers remain frozen. No commit/push performed.
