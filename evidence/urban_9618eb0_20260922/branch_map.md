# Urban branch map: Geant4 11.3.2 reference -> MAIGO port -> test
Evidence: `evidence/urban_9618eb0_20260922/`. Anchor: 9618eb0 + B1-B6 fixes
(uncommitted worktree). Reference sources: executed local install
`/software/geant4-11.3.2` (version string `geant4-11-03-patch-02`,
G4VERSION_NUMBER 1132) + upstream v11.3.2 tag text. No local G4 patch:
per-atom C12 cross sections match the tag to 6e-5 over 1-3600 MeV, and the
executed model dump matches the tag defaults.

Executed reference configuration (direct oracle, opt4, GenericIon):
`UrbanMsc, StepLim=Minimal, Rfact=0.2, Gfact=2.5, Sfact=0.6, DispFlag:1,
Skin=3, Llim=1 mm`. TOPAS 4.2.p3 reference log (job 6529) agrees:
muon/hadron step limit 0 (=fMinimal), muhad Rfact 0.2, muhad disp 1, Alg96 1,
ion step function (0.1, 0.001 mm), Urban fluctuations.

## 1. Model registration (premise)

| Reference | MAIGO | Test |
|---|---|---|
| opt4: GenericIon -> G4hMultipleScattering -> default G4UrbanMscModel (isWVI=false, no region override) | `urban_v2` research paths for C12 (+Cu) | oracle `BeginOfRun`: model name `UrbanMsc`; TOPAS log EM dump |

## 2. Constants (G4UrbanMscModel ctor + G4VMscModel)

| Symbol | Ref value | MAIGO | Test |
|---|---|---|---|
| tlimitminfix | 0.01 nm = 1e-8 mm | kUrbanV2TlimitMinFix 1e-8 | E3/B3 |
| tlimitminfix2 | 1 nm = 1e-6 mm | kUrbanV2TlimitMinFix2 1e-6 | subdiv tests |
| tlimitmin (StartTracking) | 10*fix = 1e-7 mm, NEVER recomputed in fMinimal | UrbanV2TrackState.tlimitmin_mm = 1e-7, frozen (B3) | B3 all paths |
| tlimit init | geombig = 1e50 mm (StartTracking) | state init 1e10 (both non-binding; first real boundary sets 0.2*max(range,λ0)) | B6 boundary equivalence |
| facrange (muhad) | 0.2 | kUrbanV2FacRange 0.2 | propose path tests |
| lambdalimit | 1 mm | kUrbanV2LambdaLimit 1.0 | tsmall |
| dtrl | 0.05 | kUrbanV2Dtrl 0.05 | branch tests |
| taubig/tausmall/taulim | 8 / 1e-16 / 1e-6 | same | sampler tests |
| geomMin/minDisplacement2 | 0.05 nm=5e-8 / 2.5e-15 | kUrbanV2GeomMinMm / kUrbanV2MinDisplacement2Mm2 exact | R2 acceptance |
| sFact 0.99 | AlongStepDoIt postSafety | accept_displacement, WITH dispR cap (B2) | R2 updated |
| c_highland | 13.6 MeV | 13.6F | theta0 in oracle bands |
| dispAlg96 | true (default) | 0.73*rmax Alg96 only | failed.md #173 (New rejected) |
| rmax | sqrt(t^2-z^2) | rmax2 = delta*(2t-delta) | R3/delta tests |
| skin | 3 (opt4 global; fMinimal-unused) | n/a (no skin logic on fMinimal path) | — |
| stepmin/tlow/finalr/drr | fMinimal-unused | stepmin computation DELETED (B3) | B3 |

## 3. ComputeTruePathLengthLimit, fMinimal branch

| Reference branch | MAIGO (`copper_urban_v2_limit_step`) | Test |
|---|---|---|
| t = min(input, range) | same | propose tests |
| t < tlimitminfix -> ConvertTrueToGeom, latDisp=false, firstStep KEPT (early return) | exit 0, no_displacement, first_step untouched | B3 path-0 |
| t==range && t<presafety, or t<fix -> same | exit 1 | (covered) |
| distance=range*doverrb < presafety -> ConvertTrueToGeom, no limit | exit 2, doverrb from InitialiseModelCache formula | B3 path-2 |
| fMinimal: at fGeomBoundary: tlimit=max(0.2*max(range,λ0), tlimitmin) | same with frozen tlimitmin; at_boundary flag | propose tests, B6 |
| tlimit < t -> min(t, Randomizetlimit), Gauss(tlimit,0.1*(tlimit-tlimitmin)) floor tlimitmin | Box-Muller, strict draws (B1), frozen floor (B3) | determinism |
| NO ComputeStepmin/ComputeTlimitmin on this branch | none (B3 deleted recompute) | B3/E3 |
| firstStep=false at every exit | state.first_step=false at path 3 (write-only in fMinimal; harmless) | propose test asserts consumed |

## 4. ComputeGeomPathLength / ComputeTrueStepLength

| Reference branch | MAIGO | Test |
|---|---|---|
| t<tlimitminfix2 (1nm) -> g=t | true_to_geom tiny branch + finalize sliver (g<1e-6 -> t=g, delta=0) | subdiv robustness |
| tau<=1e-16 -> z=min(t,λ0) | same + delta=0 | B4 |
| t<range*dtrl, tau<1e-6 -> t(1-tau/2) | same + series delta t²/2λ (B4) | B4 small-t |
| t<range*dtrl else λ(1-e^-τ) | same + double-formula delta (B4) | B4 |
| E<mass or t==range -> 1/range branch + par capture | same + par1/2/3 + double-direct delta (B4) | B4 range branch (par1>0 verified reachable) |
| else lambda1 branch | same + guard + delta | (C12-dead: E<mass always; kept for shape) |
| z=min(z,λ0) | same; delta follows clamp (B4) | B4 |
| ComputeTrueStepLength: g==z shortcut; par1<0 inversion; par1>=0 inversion; clamp [g,t_msc] | finalize_true + delta_out per branch (B4); untruncated -> proposal delta | B4 untruncated/truncated |

## 5. SampleScattering / SampleCosineTheta

| Reference | MAIGO (`copper_urban_v2_sample_full`) | Test |
|---|---|---|
| t>=range -> return, no scatter | same | DoIt gate test |
| internal E prediction: GetEnergy(range-t) if t>5%range else E-t*DEDX | loss-table energy / linear dedx | — |
| t<=fix or t<tausmall*λ0 or E<=1eV -> return | same (1e-6 MeV) | lowE tests |
| tau>=8 -> isotropic | same | — |
| effective tau via lambda1 | same | — |
| xmeanth/x2meanth series vs exp | same | — |
| relloss>0.5 -> SimpleScattering | same | — |
| tsmall=min(tlimitmin,lambdalimit); extreme-small scaling | same with FROZEN tlimitmin (B3) | B3 |
| theta2<tausmall -> cth=1 | same (cth stays 1, omcth 0) | E2 |
| theta0>pi/6 -> SimpleScattering | same | — |
| x / xsi / max(xsi,1.9) / c guards | same | — |
| xmean1<=0.999*xmeanth -> SimpleScattering | DOUBLE gate (B2) | B2/q-audit |
| mixture weights + q | DOUBLE chain + trials (B2); FP32 sampling bodies kept | single-step oracle bands |
| flatArray(2) draws | strict draws rd,rd+1 (B1); rd+2 core/tail | B1 |
| core/tail/isotropic sampling expressions | same (float) + stable omcth per branch | E2, oracle q999 |
| \|cth\|>=1 -> return UNSCATTERED | omcth-based early return (B2) | E2 (no misfire: C/B=1) |
| sth=sqrt((1-cth)(1+cth)); phi=2π*flat | sth from omcth; phi strict rd+3 | E2 |
| displacement iff latDisplasment && tau>=tausmall; Alg96 (0.73*rmax, psi exp(-2.16v), phi±psi) | same, rmax from branch delta; draws rd+4/5 | disp branch stats |
| process gate (AlongStepDoIt): r2>minDisp2; post=0.99*min(safety,dispR); accept/reduce/cancel | accept_displacement WITH cap (B2) | R2 (branch3@far-field) |

## 6. GPIL/DoIt order (G4VMultipleScattering)

GPIL: t=ComputeTruePathLengthLimit(track, currentMinimalStep) -> return geom;
candidate iff t<physStepLimit. DoIt: t=ComputeTrueStepLength(geom);
t=min(t, physStepLimit); sample iff t<range && t>geomMin(5e-8);
displacement gate as above; ProposeTrueStepLength(t).
MAIGO: caller subdivides to fixed ceiling (user limit plays the TOPAS
MaxStepSize role); limiter non-binding inside it; no min()-commutes claim
(comment corrected). MSC-only sub-loop (loss pre-scored on macro step) is a
documented structural difference (§2.6), unchanged by B-fixes.

## 7. RNG addressing proof (§2.9)

Composition steps*1024+segment_index, dims {58,59,70..75,110..115}.
random_u32 counter=[hist_lo,hist_hi,ii_lo,ii_hi^(dim/4)], lane=dim%4.
Claim: no collision for seg<1024/step and steps<4.19M/history.
Proof: (a) seg<1024 -> 64-bit ii injective in (steps,seg) (mixed radix).
(b) Same-counter+lane with different dims needs same lane, same ii_lo, and
B^B'=H^H' with H^H' in {31,28,21,18,3,10,13,9,12,7} (block pairs of the dim
set); smallest magnitude is 3 (blocks 17^18, lanes 70/74 and 71/75),
requiring |Δsteps|=3*2^22=12.6M; m=1,2 need XOR-1/XOR-2 block pairs which do
not exist in the set. |Δsteps|=4.19M (m=1, XOR-0 impossible for B≠B').
Histories with ≥4.19M macro steps are unrunnable (kernel cost per step);
valid runs additionally trip the 1M-seg fatal cap first. (c) seg<1024 is
structural: step_mm<=maximum_step_mm, enforced <=1024*max_segment_mm by the
config guard (51.2 mm urban / 102.4 mm FE); water mode caps at 1.0 mm anyway.
Tests: B6 addressing edges + shipped-config invariant + boundary equivalence.

## 8. Direct-oracle validation status

- Model dump: UrbanMsc/Minimal/0.2/Disp1/Skin3/Llim1 (executed setup).
- Cross sections: installed lib == tag to 6e-5 (C12, H/O, 1-3600 MeV);
  MAIGO port matches spot checks to 5 digits (full grid pin: follow-up test).
- Single-step (0.05 mm, 250 MeV/u, 200k): q50-q999 ratios
  1.0009/1.0035/1.0006/1.0001/1.0066/0.9980, E[th^2] 0.983.
- Post-B1..B6 single-step: IDENTICAL (fixes inert at this point by design).
- Slab/multi-step composition vs oracle: PENDING (Phase D).

## 9. Validation outcomes (2026-09-22, fixed binary)

- B1 endpoint artifact CONFIRMED as the shallow-interval-gap root cause:
  pristine replay carries ~16-25 spurious radian kicks per 1.34M histories
  (40->60 |dth|>0.3: pristine 19 vs TOPAS 3 vs B 2; max 0.97/0.60/0.39);
  B replay intervals match TOPAS to 1.011-1.032 (was 1.241/3.80 shallow).
  Mechanism: float-q rounds to exactly 1.0 while true q is 1+1.6e-8, so
  u0==1.0 (2^-24/draw, E1 preimage on file) takes isotropic in pristine
  but mixture in the reference. An interim "debunk" (u0=1<double-q) was
  wrong: the code compares against float-q. B1 strict uniform removes it.
- B2 verdict: float trials decision-identical to double on the entire
  REACHABLE sampling domain (t<=min(0.05,range/2) grid: 0 flips/20000;
  the 4e-4 drift corner (t=0.05,E~1MeV) cannot sample since t<range
  implies tau<=range/lambda0<3e-3). Unconditional double measured 3-4x
  wall and was removed; kept: strict draws, float gate, omcth protection
  (dead by construction, no misfire per E2 C/B=1), dispR-capped postSafety.
- Dose bulk unchanged B-vs-pristine (<=0.07pp same-seed) as predicted for
  a dozens-of-histories artifact; deep-vs-FE improvement retained;
  mid-depth valley excess unchanged (non-MCS residual, out of scope).
- Composition (host, production path): 1mm E[th^2] 1.052 / varX 0.994;
  10mm 0.988 / 0.970; exit-E +0.04%/+0.6% (CSDA-linear vs straggling).
- G4 own step dependence 0.025/0.05: 0.960 (1mm) / 0.924 (10mm).
- CT non-interference: B-vs-pristine dose sums identical, 1.1e-8 max diff.
- Full-physics production 10M: sums identical, 3.7e-9 max diff.
- Perf: NO controlled claim (P8 clock variance observed; root unavailable).
  Hot-path delta vs pristine is a few ALU ops; double-exp pitfall removed.
