# Independent electron optimization execution

The active objective is the supplied `pasted-text-1.txt` electron optimization
proposal. The user's subsequent ordering is authoritative: establish the full
FP32 baseline first, then independently measure selective FP64 transport.
Dose scoring remains FP32 in every build. No target speed is an achieved result.

## Completion checklist

- [x] A1 baseline: full FP32 independent electron runtime, runtime conversion,
  closed queue/energy audits and RT07575 comparison. See
  [precision report](electron_precision_20261003.md). Resident 86,861.91 primary/s;
  electron 12.7896 s per 1,983,920-primary batch. This is research acceptance,
  not all-voxel 1% agreement or independent TOPAS validation.
- [x] 0: matched hardware counters, energy-decade and exact limiter histogram.
  Diagnostic-only build isolated under `out/electron_diagnostics_20261003/build`;
  no diagnostic timing may be promoted as throughput evidence.
- [ ] A1 follow-up: separately test sensitive Urban arithmetic and geometry
  precision starting from that FP32 baseline, with native AlongStep/interface
  and CT dose checks. The [FP64 mean-product/Urban-gate ablation](electron_fp64_gate_20261003.md) has component and ABBA evidence (no resolved whole-batch cost); default remains OFF pending combined validation. Original native AlongStep tests expose a separate 100 eV energy/terminal-representation discrepancy in164/5412 cells, unchanged by this gate. The [geometry ablation](electron_geometry_20261003.md) now passes interface/translation, OFF-control hashes and corrected RT07575 ABBA audits; resident throughput −11.960%, electron time +25.395%, so default OFF. Pooled peak/valley change≤0.00312%, but local outliers remain (BODY max20.838%, central valley max4.255%); this is not all-voxel 1% acceptance. Energy precision remains open.
- [ ] A2: **parameter reuse accepted** ([report](electron_parameter_reuse_20261003.md)); rate-polynomial representation, material constants
  and elastic lookup work; independent component checks then paired ABBA. [Urban material-log precomputation](electron_lookup_20261004.md) is evaluated and rejected for no throughput gain; source reverted.
  [Load-time hard-majorant certification](electron_rate_certificate_20261003.md) is accepted with a continuous FP32 interval proof, unchanged endpoint checks and exact step hashes.
  [Cross-step voxel/material caching](electron_voxel_cache_20261004.md) is now
  accepted and enabled in the standard build: 44.5M exact paired steps and
  RT07575 ABBA, resident throughput +9.988%, electron time −19.529%, unchanged
  integer work and passing dose/energy/completion/overflow checks. Polynomial
  hard-rate evaluation has now been [component-screened](electron_rate_polynomial_20261004.md):
  protected centered Horner passes CPU/GPU numerical and majorant checks, but
  isolated lookup timing resolves no benefit, so no runtime integration or CT
  speed claim. Other remaining A2 work is not marked complete.
- [ ] B1: class-II hard elastic + condensed soft moments/random hinge; establish
  physical/interface validation before CT adoption. [Angular-moment preparation](electron_class2_20261004.md) covers82nodes/3034 energy conditions and49.7M GPU samples; naive uniform90%-soft small-angle premise is not valid at low energies. Runtime condensed/lateral/interface work remains open.
  [Protected loss aggregation with explicit elastic paths](electron_elastic_batch_20261004.md)
  now has24 full charged/photon trajectory conditions,6.291M histories/arm,
  and a+300mm translation control at the same6.291M histories/arm. An initial FP32 origin-cancellation guard
  failure was fixed and logged; actual per-vertex material/scoring index audits
  now have zero violations. Calls fall35.68% in the original fixture (not time);
  terminal-distribution/side screens pass. Peak point estimates<=0.8216%
  (translated<=0.6493%), but
  41/48 conservative1% peak/valley budget checks remain unresolved, and local
  fine-bin differences remain. Follow-up613 mixed-density/interface conditions
  (5.022M histories/arm) pass validity/index/distribution screens;1105/1226
  conservative peak/valley1% budgets remain unresolved. Prepared-original
  fallback reuse preserves full raw hashes in both arms. An experimental,
  default-OFF runtime now passes RT07575 two-batch primary/EM, full photon,
  completion/energy/zero-overflow audits with a documented event-count joint
  retry clock. Charged calls fall47.29%, pooled depth peak/valley changes<=0.19594%,
  but BODY local max47.84%, central peak/valley max6.47/32.68%; no all-voxel1%
  acceptance. Uninstrumented ABBA now finds no useful gain: resident112305→111468
  primary/s (−0.745%, paired−1.882/+0.380%), electron7.72750→7.78665s.
  Matched full-grid NCU shows active threads/instruction7.96→5.78, local
  load/store sectors+41.70/+32.90%, with warp occupancy essentially unchanged.
  Runtime/CMake hooks restored byte-identically, default binary unchanged;
  research artifacts and rejection/retry conditions retained. User requested
  wrap-up; no further experiments started. This tests energy coupling before
  replacing angles and does not finish the broader optimization objective.
- [ ] B2: assess GS alternative with geometry-dependent lateral displacement;
  the previously rejected sphere/voxel shortcut is not reusable unchanged.
  [Fixed-energy condensed spatial screening](electron_condensed_20261004.md)
  rejects single-hinge and two-hinge moment-only closures; four-subsegment
  radial KS improves to0.009918 but has no CT/performance acceptance. Exact
  half-angle GS spatial isolation plus exact one-event position gives maximum
  radial KS0.016151; low-multiplicity spatial treatment remains open. No new
  runtime default or throughput benefit from these prototypes.
  Follow-up exact N<=4 half-trajectory composition reduces radial KS to0.007172,
  with moment/geometry/Poisson-coverage checks passing. It still samples every
  half-angle explicitly; no condensed speed or CT acceptance yet.
  [Full-soft-law fixed-count convolution](electron_multiple_cdf_20261004.md)
  now passes independent spectral/tail and GPU angle/spatial/count screens for
  12 fixed node/energy laws. The coarse construction grid was rejected by an
  independent tail check; final 32769-point grid maximum relative tail error
  is0.2444%. Exact N<=4 positions plus GS high-count displacement, forced inline
  and scalar low-count state reduce isolated segment time about21.3% (57regs,
  0B static stack). This is NOT CT/electron-stage speed or dose acceptance;
  arbitrary-energy, energy-loss/hard-event and interface coupling remain open.
  [RT07575 original-state sampling](electron_segment_feasibility_20261004.md)
  now covers79,941 states from1.316B unchanged charged calls, with exact work,
  quality/energy/completion/overflow checks. Under current0.005 loss and0.05mm
  limits, all-face protection and hard-proposal termination, state-weighted
  mean soft count is1.5394 and mean P(N>4)=6.6877%. This is not a candidate
  trajectory or speed prediction; low-count energy/spatial coupling remains
  essential. Diagnostic default OFF; formal binary unchanged.
- [ ] B3: assess elastic null-collision/energy-loss cap only after B1/B2.
- [ ] C: evaluate scoring-voxel-contained range rejection using measured work,
  with material/face protection and photon/tail accounting. No global cutoff.
  [Observational range-guard screening](electron_range_guard_20261004.md) now
  covers4.55M packets/1.158B steps: 1x CSDA is not exact containment, while
  1.5x/2x still leave photon handoffs requiring full continuation checks.
  The default-OFF RT07575 counterfactual diagnostic now passes unchanged work,
  original dose/quality, energy and completion/overflow checks on3.968M primaries.
  Actual guarded suffixes cover16.6236% of charged calls; full photon/joint
  continuation exposes114 different-voxel deposits and2 escapes. Pooled
  peak/valley change<=0.000109%, but BODY local max3.420% (central valley0.584%):
  neither exact containment nor all-voxel1% is established. No early-termination
  approximation or throughput gain accepted; photon/tail treatment remains open.
- [ ] D: full Mott inverse-CDF/alias sampling; validate angular moments/tails.
  [Full Mott inverse-CDF component](electron_inverse_cdf_20261004.md) now covers
  82 nodes/3034 exact-energy conditions (18.65MB), 49.7M samples per arm, and
  passes theoretical-variance moment/KS/tail screens. Continuous-energy work
  now passes8856 conditions and145.1M samples/arm, including the generated runtime
  entry and node-neighbor tests. RT07575 ABBA rejects this runtime integration:
  resident throughput−3.377%, electron time+4.537%; pooled peak/valley≤0.09163%
  but central valley local max26.60%, with RNG/statistical effects unresolved.
  Runtime edits reverted and default binary unchanged. Further D work requires
  a materially cheaper lookup/state/layout; the fixed-energy microbenchmark
  ratio0.408 must not be reused as whole-CT speed evidence.
- [ ] Final: accepted defaults, dose/transport/EM/energy/overflow validation,
  reproducible performance and explicit record of every rejected candidate.

## Diagnostic definition

`CARBON_ELECTRON_STEP_DIAGNOSTICS=ON` instruments only the FP32 worker.
The default is OFF and it adds no production data members or counters.
There is one matrix count per charged transaction, including early exits;
host code requires its sum to equal the existing charged-call audit.

Energy bins (pre-step MeV): `<1e-4`, `[1e-4,1e-3)`, `[1e-3,1e-2)`,
`[1e-2,1e-1)`, `[1e-1,1)`, `>=1`. Limiter bits are max step=1,
mean-loss cap=2, range=4, voxel face=8, collision flight=16. Exact ties remain
bitmasks; zero means no positive transport step was selected.

Outcome categories: 0 continued without realized collision; 1 elastic;
2 hard proposal rejected by the null test; 3 accepted hard collision (including
hard local deposit); 4 soft/native terminal deposit; 5 escaped;
6 missing coverage; 7 invalid. Flight-limited soft terminal deposits are not
artificially classified as elastic/hard: the physical code never draws that
event choice. Diagnostics consume no random numbers. Counts use 64 sharded
device matrices; the original eight per-transaction audit slots are unchanged.

The immutable initial full-FP32 executable remains `out/electron_fp32_20261003/full_carbon_mc`, SHA256
`c7465f3f11586a38d3ce987a7ad68efdc28613f62bf19007271869f3a0ace3b4`.
Hardware-counter input identities and exact metrics are recorded in
`out/electron_diagnostics_20261003/profile_inputs.json`.

## Diagnostic result

Two RT07575 shards, 3,967,840 distinct primaries, 1,315,941,614 charged calls.
Every histogram closes exactly against the charged audit. All primary/EM and
electron/joint/photon counts equal the frozen full FP32 baseline; task completion
4,917,548 / 4,905,394, zero overflow/coverage/invalid. Maximum relative energy
residual 4.942e-8; full-grid dose L1 6.1403e-8. Instrumentation is observational.

- max_step: 0.2476% of charged calls.
- mean_loss: 19.1227% of charged calls.
- range: 0.0000% of charged calls.
- voxel_face: 0.6196% of charged calls.
- collision_flight: 80.0096% of charged calls.

- Energy <100 eV: 0.0000% of charged calls.
- Energy 100 eV–1 keV: 1.3363% of charged calls.
- Energy 1–10 keV: 6.7861% of charged calls.
- Energy 10–100 keV: 68.4448% of charged calls.
- Energy 0.1–1 MeV: 23.4328% of charged calls.
- Energy >=1 MeV: 0.0000% of charged calls.

Realized elastic events occupy 76.5583% of calls.
Only 28 exact limiter ties occur across both shards; inclusive totals overlap
there. Flight-limited soft terminal deposits remain separate from elastic/hard.
The low-energy (<10 keV) work fraction is only about 8.1%, so eliminating just
that band cannot deliver an order-of-magnitude reduction in step count; runtime
cost is not assumed equal across energies. The 0.05 mm ceiling limits only
0.25% of calls, whereas collision flight and the mean-loss ceiling dominate.
Hardware measurements and the corrected full-grid kernel identity are recorded
in [precision report](electron_precision_20261003.md#完整批次硬件剖析校正).

Current `build/carbon_mc` includes parameter reuse, certified pre-step rates and
cross-step voxel/material caching, SHA256
`ab913f824ddd13968f56a47acd897c9283f0a872a4d2af338c6e2cf9ab47d19a`.
Latest matched ABBA resident throughput100,994→111,082 primary/s;
electron9.68437→7.79307s per1,983,920-primary batch. Whole-process throughput
improves8.740%; FP32 scoring and full-FP32 electron transport remain enabled.
The prior accepted binary is preserved under
`out/electron_voxel_cache_20261004/baseline_carbon_mc` (SHA256
`63255a237a44dd4d88656ba5b8196ea1539baf7d391ddfd8f5297e0b9bd0980a`).
This is an implementation-equivalence acceptance, not new TOPAS validation;
the electron1–2s target and the native100eV AlongStep issue remain unresolved.
The independent FP64 mean-product/Urban-gate experiment has completed isolated
throughput and CT screening against pure-FP32 parameter reuse. It does not resolve
the separate native terminal threshold discrepancy; the combined certificate+gate
configuration has not been validated.
Remaining A1/A2/B/C/D items are still open; no claim of goal completion.

## Current-default full-plan run requested after wrap-up

The user-requested RT075750.1% rerun completed236,086,432 primaries,487 spots,
119 resident batches,group32,using the unchanged accepted full-FP32 binary
`ab913f824ddd13968f56a47acd897c9283f0a872a4d2af338c6e2cf9ab47d19a`.
Process2178.596s (36.31min),108366primary/s including loading; batches2–119
resident108716primary/s; electron total956.427s (mean8.0372s/batch). All119
primary/EM,completion,energy and zero-overflow/coverage/invalid checks pass;
584,285,106 electron packets completed; max relative energy residual1.104e-7.
No legacy electron loading/fallback and no elastic-batch experiment.

Against unchanged historical TOPAS weighted DIJ, BODY∪GTV/10% reference-max
threshold (185559voxels), local gamma1/2/3% at0.2mm is92.1098/97.1923/99.1006%;
local3%/0mm92.2806%. Gamma uses the existing DTA/10 spherical grid and trilinear
interpolation; this is no new TOPAS run or exact continuous gamma.
Patient-coordinate4Gy-scaled dose, audits, hashes, gamma and plots are in
[full-plan report](../benchmark/protonCT/RT07575_minibeam_GPU_fp32_0p1pct_20261004/REPORT_zh.md).
This is a completed requested simulation, not completion of the broader
electron optimization objective or evidence for an unmeasured speed ratio.
