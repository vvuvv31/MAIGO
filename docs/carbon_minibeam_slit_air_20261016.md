# Carbon CT minibeam: slit-air step experiment, 2026-10-16

The user approved this configuration-only change after reviewing the timing
and paired-dose results: carbon CT minibeam now defaults to
`minibeam_air_slit_max_step_mm: 0.2`. No transport source or binary changed in
this experiment. The paused RT06423 1B GPU campaign remains paused.

## Scope and rationale

RT06423 head, original optimized carbon minibeam full plan, 90/270 degree fields,
original source/emittance/weights/placement/normalization. This is not the default
lung benchmark or RT07575. RTX 2080 Ti, full five-pipeline native sm_75 executable,
`CUDA_DISABLE_PTX_JIT=1` for every run. Electron transport is absent in this runtime.

The preceding intrusive work census found 413.155 slit-air iterations per incident
history, 65.21% of the logged upstream steps; through-slit histories approach
2000 air steps through the 100 mm collimator. Counts and thread clocks are not
hardware utilization or isolated wall-time measurements. The candidate changes
only the slit-air maximum step from 0.05 to 0.2 mm. Cu stays 0.05 mm; patient stays
0.5 mm. Native clocks, primary fluctuations, carbon Cu hard delta, Urban,
nuclear/elastic/fragment transport and all audits remain enabled. This changes
step discretization and RNG trajectories; it is not an exact scheduling refactor.

## Controlled whole-plan timing

ABBA plus independent-seed BABA; eight campaigns of three resident ~1M batches.
First batch excluded from steady timing, no compilation overlap. 24M transported
history executions, about 8M warm executions/arm. Repeated timing campaigns share
seeds: only 6M distinct histories/arm, not 12M dose statistics/arm. CUDA clocks were
unlocked. These are steady whole-call rates, not cold CLI startup or cross-case
claims.

| Measure | Air 0.05 mm | Air 0.2 mm |
|---|---:|---:|
| Pooled steady throughput | 33,028.9/s | 40,240.9/s |
| Cu/air primary, seconds/M incident | 12.4382 | 6.9394 |
| Patient primary, seconds/M incident | 7.6720 | 7.6643 |

Whole-call throughput **+21.835%**, Cu/air primary time
**-44.209%**.
Four paired speed ratios: 1.209154, 1.226589, 1.217207, 1.220391.
Source CSVs are identical between arms. Runtime, energy, EM, coverage, completion
and zero-overflow checks passed for every field/batch. Repeated same-arm integer
ledgers match exactly; changed cross-arm counters are retained in `AUDIT.json`.

## Independent 40M/arm dose comparison

Twenty independent ~2M batches, matched spot counts/seeds; global Hamilton source
allocation reused without reallocation. Dose scale remains W/40M and batch doses
are summed once. The archived native 40M reference was checked with a 2M current
binary bridge: both fields have exact non-energy counters, maximum dose relative
L1 1.117e-07.
All 40M candidate histories completed, all runtime audits passed, maximum physical
energy residual 6.873e-06; no queue overflow.

| Fixed anatomical ROI | Mean-dose change | Paired batch SE | Approx. 95% interval |
|---|---:|---:|---|
| BODY | -0.01410% | 0.02369% | [-0.06368, +0.03549]% |
| Target | +0.04873% | 0.06191% | [-0.08085, +0.17830]% |

**GPU candidate versus native GPU**, BODY & native dose >0.4 Gy, local gamma,
search step 0.02 mm:

| Criterion | Pass rate |
|---|---:|
| 3% / 0.2 mm | 93.13570% |
| 3% / 0.1 mm | 86.83991% |
| 3% / 0 mm | 64.66756% |

These are not TOPAS gamma or absolute acceptance. Relative voxel-difference RMS
is 4.0730%; paired
batch-SE RMS is 4.0099%.
Noise dominates but this does not prove zero spatial bias.

Eight balanced batch-label swaps reconstruct both arms, preserving source counts
and normalization. On a common BODY & pooled dose >0.4 Gy mask, actual local3%/
0.2mm is 93.02077%, while swaps range
93.07586–93.39148%. This is a descriptive noise comparator, not a
confidence interval, p-value or acceptance gate. The slightly lower actual rate
leaves small spatial changes unresolved.

Exploratory fixed-anatomy depth thirds and six transverse phases (3 mm pitch /
0.5 mm voxel spacing) were also examined. Target phase 2 changes
-0.5501% with paired SE
0.2508%; phase 5 changes
+0.4810% with SE
0.2449%. These correlated
exploratory bins do not establish a systematic effect, but overall ROI means
alone cannot establish equivalence. Phase bins are not labelled peak or valley.

## TOPAS reference and decision

Four completed TOPAS contributions (parts 00–03, 800M histories) were retrieved
read-only into the experiment folder and SHA-256 checked. Part 04 remains
unfinished. Parts distribute spot packets, not independent copies of the whole
plan; multiplying the first four contributions by 1.25 would give the wrong
reference. They were not used for full-plan gamma, and no cluster tasks changed.

The speed improvement is resolved. Absolute TOPAS gamma change was still unknown,
and sub-percent transverse changes remained unresolved when the user explicitly
approved adopting **0.2 mm as the carbon CT minibeam default**. This approval is
not a new accuracy acceptance claim. No other preset was changed, no 1B GPU job
resumed, no commit or push performed.

The user also requested gamma using the four completed TOPAS parts. This is
possible by matching their actual spot allocation on the GPU: parts 00–03
contain 800M histories across 949 spots. A separate matched-subplan GPU run is
required; the prior 40M full-plan GPU dose contains additional spots. Both doses
retain the original full-plan normalization, without multiplying the four TOPAS
contributions by 1.25. Analysis artifacts are kept under
`tmp/carbon_mb_topas_four_parts_20261016/`.

### Completed matched-four-part TOPAS comparison

The matched GPU subplan completed 40M histories (30,000,021 on RTX 2080 Ti,
9,999,979 on RTX A6000), against 800M TOPAS histories in parts 00–03. The
949 selected spots and their relative allocation match the reference; no
full-plan extrapolation, dose fitting, registration or smoothing was applied.
Both native AOT builds used identical frozen runtime source, with PTX JIT
disabled. All field/batch runtime audits passed, no queue overflow occurred,
and the maximum physical energy residual was 8.598e-7.

BODY and selected-part TOPAS dose >0.4 Gy, 385,115 voxels; local gamma with
0.02 mm search spacing:

| Criterion | GPU air 0.2 mm versus TOPAS |
|---|---:|
| 3% / 0.2 mm | 95.94303% |
| 3% / 0.1 mm | 91.25923% |
| 3% / 0 mm | 69.50158% |

Relative voxel-difference RMS is 3.31817%; estimated voxel-SE RMS is
3.20545% GPU, 0.71384% TOPAS and 3.28397% combined. Fixed anatomical BODY
and target mean-dose differences are -0.09852% and +0.18418%, respectively.
The comparable difference and noise RMS suggest substantial statistical
limitation; they do not exclude spatial model bias or establish the change in
TOPAS gamma caused by the air cap. This comparison uses only the 0.2 mm arm
and the selected four-part subplan, rather than the complete 1B plan.
Results, device/binary provenance and transverse profiles are in
`tmp/carbon_mb_topas_four_parts_20261016/results/` and `MC_COMPLETE.json`.

## Evidence

All reproducible scripts, source YAML/CSV copies, timing results, individual
quality/energy ledgers, dose/SE arrays, spatial/noise diagnostics and transverse
profile plot are under `tmp/carbon_mb_air_step_20261016/` (ignored experiment
artifacts). Key files: `PERFORMANCE.json`, `AUDIT.json`,
`validation40m/results/report.json`, `noise_comparison.json`,
`spatial_diagnostics.json`, `transverse_profiles.png`, and
`topas_reference/manifest.json`. Native driver SHA-256:
`ca0423396bef27bc77ca870b757ecb51100ff3a6fd829fa055df12c024584473`.
