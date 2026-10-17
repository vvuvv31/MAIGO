# RT06423 carbon minibeam optimization screen — 2026-10-15

Baseline is `469d283`, after explicit electron/photon removal. These are isolated
experiments; the production runtime and paused 1B campaign are unchanged.

## Scope and method

RTX 2080 Ti, native sm_75 AOT, `CUDA_DISABLE_PTX_JIT=1`. Both arms compile only
CarbonMinibeam with identical Release flags. Each experiment uses ABBA, four
3M-history campaigns, with three approximately 1M batches per campaign. The
first batch is excluded only from steady-state timing: 12M simulated total,
4M timed per arm. GPU clocks are unlocked; observed temperature reaches 83 C.
Compilation never overlaps measured transport.

The RT06423 physical-dose optimized full plan retains its 90/270 fields,
source energy/optics/placement, CT transforms and collimator (100 mm Cu, 0.7 mm
slits, 3 mm pitch, 270-degree offset 1.5 mm). Original 1B per-spot allocations
are scaled to 3M with largest remainder and then distributed across batches;
all arms use identical source CSVs and per-spot seeds. Primary fluctuations
remain on, patient/Cu-fragment restricted fluctuations off, patient cap 0.5 mm,
Cu cap 0.05 mm, nuclear elastic/inelastic and charged fragments retained.

## Exact implementation candidates

| Candidate | Control warm histories/s | Candidate | Change | Pair ratios | Decision |
|---|---:|---:|---:|---|---|
| Cache Urban material index until exact CT section/density changes | 24,748.73 | 24,727.50 | -0.086% | 0.989890 / 1.008270 | Not promoted |
| Reuse identical step-start continuous26 range in CT Urban | 24,589.36 | 24,772.23 | +0.744% | 0.995207 / 1.019707 | Not promoted |

Both changes retain source allocation, EM/Urban/nuclear/elastic counters and
live species terminal counts exactly, with accepted energy/coverage/completion
and zero-overflow audits. Maximum voxel dose relative L1 changes are 4.0273e-8
and 3.9735e-8 respectively. Patient-primary pooled times improve numerically
by 1.59% and 2.38%, but mixed pair ratios and thermal variation do not establish
a stable whole-plan gain. No production source change is justified by these
screens. Candidates and all logs remain in:

- `tmp/carbon_mb_urban_material_20261015/`
- `tmp/carbon_mb_urban_range_20261015/`

## Research step-segmentation candidate

`tmp/carbon_mb_clock_quarter_20261015/` tests an independent candidate that
multiplies only the CT CarbonMinibeam primary segmentation rate by 0.25. The
scaled rate is used both by the distance calculation and accepted-path clock
debit. The physical continuous26 hard rate, energy-loss moments and slopes,
restricted/delta fluctuations, 1% combined mean-loss guard, Urban laws, nuclear
competition, maximum step caps and copper/air/secondary transport remain.

This changes step discretization and trajectory/RNG evolution; it is not an
exact implementation optimization. Runtime audits must pass, but identical
nuclear counts or voxel doses are not expected. A low-statistics same-source
comparison is a speed and gross-dose screen, not a new TOPAS gamma acceptance.

Completed quarter-clock results:

| Metric | Native clock | Quarter segmentation rate |
|---|---:|---:|
| Pooled warm throughput, histories/s | 24,657.61 | 34,553.00 |
| Warm patient-primary seconds, 4M histories per arm | 71.2067 | 23.9854 |
| Warm Cu/air primary seconds | 50.8811 | 51.3944 |
| Patient primary steps, one matched 3M campaign | 1,036,946,101 | 375,691,353 |
| Patient primary steps per incident history | 345.649 | 125.230 |

Whole-plan throughput increases **40.13%**; the two opposite-order pair ratios
are **1.394070 / 1.408523**. Patient-primary time decreases **66.32%**. All 24
field-batch quality/energy/coverage/completion and zero-overflow audits pass.
This is a controlled short screen on one device/case, not a general speed claim.

After this change, measured wall fractions are patient-primary **20.72%**,
Cu/air primary **44.40%**, queued Cu children **9.19%**, medium secondaries
**17.69%**, and unattributed remainder **8.00%**. Secondary tail is already
included in the secondary kernel total. Upstream work is now the dominant
combined component; further patient-only acceleration has diminishing returns.

### Dose screen and adoption status

For the first matched 3M campaign in each arm, after exact field back-mapping
and summing each normalized field/batch once:

- Total patient deposited energy changes **-0.00997%**.
- Transversely integrated field depth-profile relative RMS differences are
  **0.1783% (90°)** and **0.1526% (270°)**, evaluated at bins above 10% of each
  reference profile maximum. Maximum absolute relative differences are 1.1123%
  and 0.7001%. These are profile differences, not gamma.
- Full-grid dose relative L1 difference is **7.56%**. In the reference-dose
  `>0.4 Gy` mask, voxel relative RMS difference is **11.03%** and mean-dose
  change is **-0.163%**. This mask is not a CST BODY mask.

The ABBA repetitions reuse the same seeds and are not independent dose samples.
The 3M spatial comparison cannot separate sampling differences from systematic
bias. Close integral/profile agreement does not certify minibeam valleys,
interfaces or TOPAS local gamma. The quarter-clock implementation is retained
**only as an isolated research candidate**, with no production/default clock
change and no resumption of the 1B campaign. The two exact-code candidates are
rejected for lack of stable performance benefit.

Evidence: `tmp/carbon_mb_clock_quarter_20261015/{RESULTS.json,dose_screen.json,breakdown.json,provenance.json}`.
The private candidate is the baseline primary phase with the following scoped
line immediately after `unified_primary_clock.update(...)`, before both clock
`distance(...)` and accepted-path `consume(...)`:

```cpp
if constexpr (std::is_same_v<Pipeline, pipelines::CarbonMinibeam>)
    if (enable_ct_grid && in_ct) unified_primary_rate *= 0.25f;
```

This is not a modification of the continuous26 package rate used by the energy
loss law, and it is not an instruction to change production defaults without
assessing the resulting spatial dose approximation.

## Further timing attribution

Existing ABBA warm samples (3,999,998 histories per arm), without new simulation:

| Stage | Default s / million | Quarter-clock s / million |
|---|---:|---:|
| Patient primary | 17.802 | 5.996 |
| Cu/air primary | 12.720 | 12.849 |
| Cu children | 2.617 | 2.661 |
| Medium secondaries | 5.166 | 5.121 |
| Unattributed wall-time remainder | 2.251 | 2.315 |
| Total | 40.555 | 28.941 |

Secondary tail time is INCLUDED in secondary kernel time: default 10.494/20.663 s (50.8%), candidate 10.008/20.482 s (48.9%). The code switches to full completion when the active population falls below the tail threshold; this suggests long-track divergence/low active population as a profiling target, not a proven hardware diagnosis. Compaction plus grouping totals only 0.0394 s per arm, approximately 0.034% of candidate wall time. Eliminating queue organization cannot materially improve throughput.

Warm Cu counters are exactly equal between arms: 873,616,164 primary loss calls, 1,069,650,282 fragment loss calls, 6,330,076 Cu-child tasks. These are work counts, not measurements of the time spent inside loss sampling. Cu and air primary time are not separately instrumented; their individual shares cannot be inferred from these logs.

Candidate Amdahl scenarios (not measured predictions): halving Cu/air-primary plus Cu-child time would improve total throughput by approximately 36.6%; halving remaining patient-primary time by 11.6%; halving medium-secondary time by 9.7%. A zero-cost patient-primary phase still caps this candidate at approximately 43.6k histories/s. Raw attribution: `tmp/carbon_mb_clock_quarter_20261015/secondary_timing_detail.json`.

## Secondary tail specialization screen

An independent 469d283 candidate specializes CarbonMinibeam secondary execution into compile-time tail and segmented paths. Threshold8192, segment16, original primary clock, all physics and source allocations remain unchanged. Tail compilation can eliminate the segment-pause check and suspended-state save; no dynamic threshold sweep. Native sm75 AOT, no JIT.

ABBA4x3M, 4M warm histories per arm: control **24,867.293/s**, candidate **24,614.707/s** (−1.016%). Pair ratios0.978783/1.000874. Control drift and unlocked clocks prevent a precise small-regression attribution; there is no stable speed benefit, so the candidate is not promoted.

Warm secondary kernel time: 20.605703→20.778918 s; included tail time 10.467152→10.549133 s. All24field-batch runtime/energy/coverage/completion/zero-overflow audits pass; source and discrete EM/Urban/nuclear/elastic/live-ion counters match, maximum dose relative L1 3.9862e-8. Production runtime unchanged, 1B remains paused.

Evidence: `tmp/carbon_mb_tail_specialization_20261015/RESULTS.json` and `timing_detail.json`.

## Default-workload bottleneck interpretation

The tail-specialization control warm samples contain 1,382,805,130 patient primary Urban steps and 451,925,676 patient secondary Urban steps over 3,999,998 incident histories: 345.701 and 112.981 steps per incident. Stage rates are approximately19.76M and21.93M steps/s respectively (aggregate device rates, not per-thread instruction latency). Patient primary time is69.9917s, secondary20.6057s. This supports excessive primary work count as a major contributor, without isolating individual sampler or memory costs.

Secondary step counts: proton135,014,288; He4 126,295,850; deuteron91,781,562; triton35,723,796; He3 21,375,614. Together these account for90.77% of secondary steps; step shares are not time shares. Secondary tail accounts for50.8% of secondary time but only6.5% of total runtime. Eliminating only tail checks did not improve throughput.

The primary step is limited jointly by continuous26's proposed step, ionization segmentation distance, a hardcoded1% combined mean-loss guard, the0.5mm cap, Urban/voxel geometry and nuclear competition. Removing explicit electrons did not remove this segmentation. The quarter-clock experiment provides causal evidence for step-count cost but does not certify a physics-equivalent replacement. Cu/air primary remains a combined event timer; Cu sampling, geometry, atomics, and air shares need separate measurement before claiming an individual hardware bottleneck.

## CT primary continuation: quarter versus sixteenth segmentation clock

Isolated RT06423 candidate based on469d283; control is the previously measured quarter-clock binary, candidate changes only the CT CarbonMinibeam primary scalar to0.0625. Both distance and accepted-path debit use that scalar. Physical hard-rate tables, loss means/variances/slopes and primary fluctuations stay unchanged; combined mean-loss guard1%, patient0.5mm/Cu0.05mm, Urban, nuclear models and secondary code stay unchanged. This changes step/RNG/trajectory distributions and is a physics approximation. Current density-fixed/electron-removed runtime,0.5mm cap and RT06423 plan differ from the historical20261010 RT07575/1mm experiment.

RTX2080Ti native sm75 AOT/JIT disabled, ABBA4x3M,4M warm histories per arm:

| Metric | Quarter clock | Sixteenth clock |
|---|---:|---:|
| Warm throughput histories/s | 35,135.594 | 38,124.491 |
| Patient-primary s / million | 5.882 | 3.268 |
| Primary Urban steps / incident | 125.230 | 82.116 |
| Whole time s / million | 28.461 | 26.230 |

Pooled gain8.507%; paired gains7.399%/9.612%. Patient-primary time decreases44.44%; candidate share12.46%. Cu/air primary50.7696s, Cu children10.4617s, medium secondary21.3832s, other9.2332s over4M warm histories. Secondary code is unchanged, but birth states and downstream work can change when primary stepping changes. Unlocked clocks; no hardware-stall attribution. All24field-batch energy/coverage/completion/zero-overflow gates pass; source CSVs match, Cu loss/cascade counters match. Do not sum separate experimental gains or treat this as a fresh native-versus-sixteenth timing pair.

First3M per arm dose screen (repeated timing seeds are not independent dose samples): total deposited energy+0.01996%; integrated depth profiles RMS0.2048%/0.2088% for90/270degrees, maximum1.534%/1.599% above10% profile peak. Whole-grid relative L1=7.442%; reference>.4Gy voxel relative RMS=10.806%, mean-dose change−0.0878%. This mask is not CST BODY. Low statistics cannot separate systematic spatial differences from noise; no TOPAS/local-gamma acceptance. Native default clock remains unchanged;1B campaign remains paused.

Evidence: `tmp/carbon_mb_clock_sixteenth_20261015/{RESULTS.json,breakdown.json,dose_screen.json}`. Candidate remains private pending spatial dose assessment.

## 40M per-arm native versus sixteenth validation

RT06423 original optimized full plan,20 independent full-plan batches per arm, matching per-spot allocations/seeds across arms; distinct from prior screens/paused1B. Largest remainder from original physical weights once for40M; W/40M doses summed once. Interleaved four groups per arm on2080Ti, native sm75 AOT/JITdisabled. All80 field-batch runtime audits pass, zero overflow, max relative energy residual6.873e-6, source allocations exact. Campaign elapsed GPU+output1540.068s native versus1000.519s candidate:25,972.877→39,979.239 histories/s (+53.93%). These include first-batch data initialization in each group; not pure kernel-only rates.

Reference here is native GPU, NOT TOPAS. BODY & native>0.4Gy, search0.02mm, local3%:

| DTA | Actual candidate vs native | Centered paired-batch noise surrogate mean |
|---|---:|---:|
|0.2mm|93.4356%|95.8087%|
|0.1mm|87.9258%|90.2675%|
|0mm|72.3366%|72.6399%|

Target local3%/0.2mm actual85.6977% versus surrogate93.0496% (eight-replicate range91.6973–94.4649%). BODY observed relative dose RMS3.4292% versus paired difference SE RMS3.3881%; target2.5477% versus2.4481%. Full anatomical BODY mean change−0.00114% (approximate95%CI−0.01319…+0.01092%); target mean−0.08510% (−0.18356…+0.01337%). Small integral changes do not certify local spatial equivalence.

Noise surrogate uses eight balanced sign permutations of20 centered paired batch differences with fixed native reference/mask. It retains measured spatial noise correlations, but is conditional, not an independent simulation or a formal acceptance test. The actual spatial gamma is below all eight surrogates in BODY/target at0.2mm: this raises a substantive spatial-bias concern despite noise-dominated voxel RMS. Do not dismiss failures as purely MC or promote1/16 by integral energy agreement. No new TOPAS reference was assembled, no physical normalization fitting, no smoothing of reported doses, and no default changed. Current1/16 candidate is NOT accepted for production; a milder clock candidate needs its own spatial test.

Evidence: `tmp/carbon_mb_clock_validation_40m_20261015/results/{report.json,noise_gamma.json,transverse_profiles.png}`. Profiles now correctly vary patient y at fixed patient x beam-depth for90/270fields. The original1B campaign remains paused. GPU validation work is complete.

### Noise-reference correction and quarter-clock follow-up

The preceding fixed-reference sign-noise surrogate is NOT a definitive expected-gamma baseline: retaining the noisy native reference while injecting a signed dose difference does not preserve the reference/evaluation noise relationship. Its95.81% BODY/93.05%target expectations overstate the contrast. A refined exploratory check swaps labels of whole paired batches, reconstructing BOTH arms around their common mean, retaining the original evaluation mask. Eight balanced permutations give BODY local3%/.2mm94.2231% (94.1038–94.3604), target88.2137% (86.3751–90.3232), versus actual93.4356%/85.6977%. Thus the excess gap is about0.79/2.52pp, not2.37/7.35pp. These are conditional exploratory surrogates, not independent MC or formal acceptance; no p-value claimed. The candidate remains unaccepted, but the previous stronger attribution of local failures should be tempered.

Fixed anatomical target coordinate-quartile analysis found no95% marginal mean-dose CI excluding zero. Along patient x (beam depth), changes−.0869/−.0585/−.0769/−.1148%; along transverse y,−.1054/+.1554/−.3322/−.0979%. Multiple comparisons were not adjusted; absence of a detected mean bias is not proof of local equivalence.

Quarter-clock40M follow-up started in `tmp/carbon_mb_clock_quarter_validation_40m_20261015/`, reusing native40M and exact same inputs/seeds read-only. Only40M new candidate histories run on2080Ti; expected~19–20min plus automatic analysis. Both surrogate methods are saved, with label-swap preferred for interpretation. Default clock remains native, original1B remains paused. Evidence: `results/{label_swap_gamma.json,localization.json}` in the sixteenth validation directory.

### Quarter-clock40M completed

All40M candidate histories completed with exact native40M source counts/seeds and all80pairedfield audits passing, zero overflow,maxrelativeenergyresidual6.893e-6. Candidate fullgroupGPU+output1093.747s,36571.539histories/s; earlier reusednative25972.877/s. Timing is sequential, not a new controlled paired speed estimate.

BODY native-reference>.4Gy local3%/.2/.1/0mm =93.8675/88.6081/73.0491%; refined paired-label-swap means94.2939/88.9435/72.9393%. Target actual87.1492/80.4142/77.1821%,surrogate87.4057/82.0036/79.0691%; target.2 actual lies within eight-surrogate range86.1235–88.6007. BODY actual.2 remains0.4264pp below surrogate mean; surrogates are conditional, not an acceptance test.

BODY differenceRMS3.3760% versus pairedSE3.3634%;target2.5109% versus2.4328%. FixedBODYmean−.00326%(95CI−.01879…+.01227%),target−.07061%(−.14414…+.00291%). Quarter is more conservative than1/16 and the target.2 result is compatible with the surrogate spread, but residuallocalbias cannot be excluded. No newTOPAS comparison/defaultpromotion. Evidence tmp/carbon_mb_clock_quarter_validation_40m_20261015/results/. Both validation campaigns are complete; original1B remains paused.
