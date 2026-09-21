# Urban v2: Copper MSC parity diagnosis (2026-09-21)

Baseline: `5f9f9b48adc1defad971d406cd015e1a28c78c37` (branch `pristine`, clean tree).
Reference: TOPAS 4.2.p3 / Geant4 11.3.2, module list
`g4em-standard_opt4 g4h-phy_QGSP_BIC_HP g4decay g4ion-inclxx g4h-elastic_HP
g4stopping` (the `run1.txt` for the minibeam references).

This note is the audit of the existing `copper_urban_msc_step` port
(`src/detail/sycl_device_math.inc`) against the *actually active* Geant4
branch, plus an independent attribution of the current residual. It does not
retune any scale.

## 0. Runtime-verified active branch

From the TOPAS log `/mnt/sda/wuwei/minibeam_msc_steps_e250/topas.log`:

```
msc:  for GenericIon  SubType= 10
            UrbanMsc : Emin= 0 eV  Emax= 6 GeV
          StepLim=Minimal Rfact=0.2 Gfact=2.5 Sfact=0.6 DispFlag:1 Skin=3 Llim=1 mm
ionIoni:  for GenericIon  XStype:3  SubType=2
      StepFunction=(0.1, 0.001 mm), integ: 3, fluct: 1, linLossLim= 0.02
    LindhardSorensen : Emin= 0 eV  Emax= 6 GeV  deltaVI
Msc lateral displacement for e+- enabled           1
Msc lateral displacement for muons and hadrons     1
Urban msc model lateral displacement alg96         1
```

So for C12 (GenericIon): model = `UrbanMsc`; step limit = `fMinimal`;
`Rfact=0.2`, `Gfact=2.5`, `Sfact=0.6`; lateral displacement ON; **Alg96 = 1**;
`Skin=3`, `Llim=1 mm`. Ionisation step function `(0.1, 1 um)`.

Source confirmation:
- `G4EmStandardPhysics_option4.cc:164` creates `G4hMultipleScattering("ionmsc")`
  with no model; `G4hMultipleScattering.cc:75` defaults it to `G4UrbanMscModel`.
- `option4:129` `param->SetMuHadLateralDisplacement(true);`.
- `option4:122-128` `SetStepFunctionMuHad(0.1,50um)`, and the e+--only
  `SetMscStepLimitType(fUseSafetyPlus)`, `SetMscSkin(3)`, `SetMscRangeFactor(0.08)`.
  `G4VMscModel::InitialiseParameters` reads `MscMuHadStepLimitType()` (=`fMinimal`,
  `G4EmParameters.cc:175`) and `MscMuHadRangeFactor()` (=`0.2`,
  `G4EmParameters.cc:160`) for non-e-, while `skin`/`facgeom`/`facsafety`/
  `lambdalimit` are shared, so Skin=3 applies to the ion too.
- `G4EmParameters.cc:124-125` defaults `lateralDisplacement=true`,
  `lateralDisplacementAlg96=true`; `option4:129` sets the MuHad flag.
  `G4UrbanMscModel::Initialise` sets `dispAlg96 =
  G4EmParameters::Instance()->LateralDisplacementAlg96()` (=true).

## 1. Priority-ordered diagnosis

### D1 (confirmed source inequivalence, active, unquantified): wrong displacement algorithm

`SampleScattering` (`G4UrbanMscModel.cc:814-819`) chooses

```cpp
if (latDisplasment && currentTau >= tausmall) {
  if(dispAlg96) { SampleDisplacement(sth, phi); }   // ACTIVE
  else          { SampleDisplacementNew(cth, phi); }
}
```

The port implemented only `SampleDisplacementNew` (Alg96=false). The active
Alg96 branch (`G4UrbanMscModel.cc:1049` is New; Alg96 is the earlier
`SampleDisplacement`) is:

```cpp
rmax = sqrt((tPathLength - zPathLength)*(tPathLength + zPathLength));
r    = 0.73*rmax;                                  // deterministic radius
psi  = -log(1 - u0*(1-exp(-cbeta*pi)))/cbeta;      // cbeta = 2.160
Phi  = (u1 < 0.5) ? phi+psi : phi-psi;             // random sign
disp = (r*cos(Phi), r*sin(Phi), 0)
```

So: fixed radius `0.73*rmax`, an exponential azimuth perturbation (not a
two-branch radius sampler), and a `phi ± psi` sign flip. This is a different
random variable from the ported New sampler.

### D2 (confirmed source inequivalence, active, unquantified): wrong `rmax`

Port: `rmax = path_length_mm * sin(theta_sampled)`.
Active: `rmax = sqrt(t^2 - z^2)` with `t` = true path length and `z` =
geometric path length. These agree only in the small-angle limit
(`t^2-z^2 ≈ (t sinθ)^2` is not an identity). `t` and `z` are produced by the
t↔g conversion, which the port does not implement.

### D3 (confirmed source inequivalence, active, unquantified): missing t↔g path conversion

Active pipeline (`G4VMultipleScattering::AlongStepDoIt`):
`ComputeTruePathLengthLimit` -> `ComputeGeomPathLength` (t->z) -> geometry
truncates z -> `ComputeTrueStepLength` (z->t) -> `SampleScattering(t, E)`.

`ComputeGeomPathLength` (`G4UrbanMscModel.cc:671`) for the common branch
(`t < currentRange*dtrl`) gives
`z = t*(1-0.5*tau)` for `tau<taulim`, else `lambda0*(1-exp(-tau))`,
`tau = t/lambda0`; then `z = min(z, lambda0)`. `ComputeTrueStepLength`
(`:732`) inverts it: `t = -lambda0*log(1 - z/lambda0)` for `par1<0`, clamped
to `[z, tPathLength]`. The port uses `z = t = transport_path` and never
produces `delta = t-z`, so D2's `rmax` is unavailable.

### D4 (confirmed source inequivalence, active, unquantified): missing `fMinimal` step limit

Active `steppingAlgorithm = fMinimal` takes the final `else` of
`ComputeTruePathLengthLimit` (`G4UrbanMscModel.cc:641-657`): only at
`fGeomBoundary` is `tlimit` set to `facrange*min(range,lambda0)`; otherwise
`tlimit` is the persistent member (init `1e10 mm`), and the step is
`min(tPathLength, Randomizetlimit())`. The port has no MSC limiter and relies
on the caller's fixed `minibeam_copper_max_step_mm`.

### D5 (confirmed source inequivalence, active, unquantified): no boundary-safe displacement

`G4VMultipleScattering::AlongStepDoIt:325-365`:
`postSafety = 0.99*ComputeSafety(fNewPosition, dispR)`; accept if
`dispR <= postSafety`; else reduce to `postSafety` if `postSafety > geomMin`;
else cancel. The port adds the raw displacement unconditionally after
straight-ray navigation. Along-ray distance to boundary is not isotropic
safety.

### D6 (confirmed source inequivalence, active, unquantified): step-end energy, not step-start

`SampleScattering:773-779` predicts the energy at the step end:
`kinEnergy = currentKinEnergy`; if `t > range*dtrl` then
`GetEnergy(range-t)`, elif `t > 0.01*range` then
`kinEnergy -= t*GetDEDX(...)`. The port passes the step-start `energy_MeV`
and computes `theta0` from it. The `invbetacp` geometric-mean branch in
`ComputeTheta0` (`:977`) is a further refinement.

### D7 (FP32 robustness, unquantified): `sin` from `cth`, moments near 1

`SampleScattering` uses `sth = sqrt((1-cth)*(1+cth))`; the port does the same
but then builds `rmax = path*sth`. With D1/D2 the correct `delta=t-z` form is
`rmax^2 = delta*(2*t-delta)`, which avoids the cancellation. `SimpleScattering`
has a small denominator `2*xmeanth - 3*x2meanth + 1` and `xmean1`, `xmean2`,
`prob`, `qprob` near 1; these need error-bounded evaluation (log1p/expm1) and
explicit branch accounting rather than silent clamps.

### D8 (range, unquantified): `currentRange` provenance

Active uses `G4VMscModel::GetRange` -> `ionisation->GetRange` ->
`theRangeTableForLoss` (restricted-loss range), with `dtrl`, `massRatio`,
`reduceFactor`, cuts. The port's `range_mm` argument is `energy/stopping`
(CSDA-like) and is only used for the (now-inactive) New-displacement branch.
For v2 `range` must feed the step limit, `dtrl` branch and `GetEnergy`.

### D9 (fast path, unquantified): air and straight-ray

The kernel's `in_copper == false` branch advances straight with air stopping
and no scattering. Geant4 scatters in air too. The audit's structural note
already flags that "straight to boundary" is not the reference. This is a
separate parity item and must be measured, not assumed, before expanding.

### D10 (metric hygiene)

The "3-5%" is not one number. The audited observables are: full-chain EM-only
12.8M 2-D L1 `2.8091%`, IDD L1 `0.4887%`, lateral-integral L1 `1.1785%`,
total dose `1.003424`; fixed-valley dose ratios `0.950` at entrance rising to
`1.109/1.167` at 80/100 mm. The same-source water replay attributes the
depth-dependent valley residual mainly to water EM/MCS, not Copper. The
single-spot entrance-contrast error (FE `+3.27`/Urban `+2.31` pp at 1 mm) is
a different observable from the 256-spot 2-D L1 and must not be mixed with it.

## 2. Independent attribution (not "all from Copper MCS")

Established elsewhere in this repo and re-checked here:

1. `GenericIon -> UrbanMsc` is runtime-confirmed (D0), so "opt4 means GS for
   ions" is false; the model choice is correct.
2. The per-step Copper angle tracks `ComputeTheta0` with a constant ratio
   ~1.24 across a 10x step range, so the angular core law is right.
3. Total deposited dose is identical across FE/Urban/Urban-0.03mm; the
   residual is lateral redistribution only.
4. The same-source water replay reproduces the mid-depth residual from a
   common entrance, assigning the dominant part to water, not Copper.

Therefore the current 3-5% cannot be attributed to Copper MSC alone. D1-D8
are real source inequivalences that must be fixed and measured; D9/D10
bound the attribution.

## 3. Status labels

| item | source-inequivalent | active here | contribution quantified |
|---|---|---|---|
| D1 displacement alg | yes | yes | no |
| D2 rmax | yes | yes | no |
| D3 t<->g | yes | yes | no |
| D4 fMinimal limiter | yes | yes | no |
| D5 boundary safety | yes | yes | no |
| D6 step-end energy | yes | yes | no |
| D7 FP32 | possible | yes | no |
| D8 range provenance | yes | partial | no |
| D9 air fast path | yes | yes | no |

## 4. Minimal urban_v2 implementation and measured effect

`copper_urban_v2_msc_step` (`src/detail/sycl_device_math.inc`) adds D1/D2/D3/D6
and the D5 boundary-safe displacement. Selected only by
`minibeam_copper_mcs_model: urban_v2` (research-only; production default is
unchanged). The caller passes the geometric step, `range`, restricted `dedx`
and an isotropic `safety`.

Component oracle (`/tmp/urban_v2_oracle.py`, CPU double):
- `lambda0(250 MeV/u) = 1.9749e6 mm`.
- `rmax` from the v2 form `sqrt(delta*(2t-delta))` equals the Geant4
  `sqrt((t-z)(t+z))` exactly (rel_err 0) for z = 0.03/0.05/0.25 mm.
- Alg96 `psi` has a z-independent mean `E|psi| = 0.4588`.
- **The correct Alg96 `rmax` is ~5x smaller than the old port's
  `path*sin(theta)`** (e.g. z=0.25 mm: 8.9e-5 mm vs 1.9e-5 mm at z=0.03 mm,
  ratio ~5). The old port over-displaced.

Single central spot, 250 MeV/u, EM-only, 10M, entry contrast error
(GPU/TOPAS-1, pp):

| depth | FE | urban 0.25 | urban 0.03 | urban_v2 |
|---:|---:|---:|---:|---:|
| 0.88 | 3.27 | 2.31 | 3.15 | 2.39 |
| 4.88 | 1.95 | 1.31 | 2.15 | **0.45** |
| 9.88 | 1.69 | 0.40 | 1.73 | **-0.47** |
| 19.88 | 0.39 | -0.55 | 2.04 | -2.08 |
| 39.88 | 1.64 | 1.11 | 2.17 | -1.12 |
| 79.88 | 0.36 | -0.36 | 0.67 | -3.07 |
| 119.88 | -1.36 | -1.79 | -0.82 | -3.64 |

urban_v2 is closest at 5-10 mm but over-scatters at depth (the step-end energy
and true-path conversion add scattering). The residual is now sign-changing
with depth, consistent with the water-dominated attribution in D10.

## 5. Corrected conclusions in urban.md

- `urban.md` says the port implements `SampleDisplacementNew`. The **active**
  Geant4 branch is Alg96 `SampleDisplacement` (MuHad displacement ON, alg96=1),
  so the shipped `urban` model used the wrong displacement algorithm (D1).
- `urban.md`'s `rmax = path*sin(theta)` is not the Geant4 form (D2).
- `urban.md`'s conclusion that "step length is not the residual" is confirmed,
  but it did not separate the displacement/energy/path-conversion errors.

## 6. Not run / missing dependencies

- No paired TOPAS water-entry or Copper pre-entry replay was executed here;
  the available phase spaces are single-run aggregates, not identity-matched.
- No step-policy matrix (matched 0.25/0.10/0.05/0.03 mm) beyond the two steps
  reported; Gate 3 needs a matched-policy TOPAS run.
- No finite-slit P(exit|d) scan (Gate 4) and no per-species Cu-touched split
  (Gate 5); those need new TOPAS runs with the MSC ntuple.
- Gate 6 downstream dose is unchanged (production water/nuclear frozen).

## 7. Gate 2/3 measured results (250 MeV/u Cu slab, 256k, TOPAS full list)

Exit phase space, GPU/TOPAS. TOPAS Cu `MaxStepSize = 0.05 mm`.

Angle-variance / tail ratios vs matched Copper step:

| thickness | model | step 0.05 | step 0.10 | step 0.25 |
|---:|---|---:|---:|---:|
| 1 mm | urban | thvar 1.035, q999 1.072 | 1.105, 1.161 | 1.218, 1.344 |
| 1 mm | urban_v2 | 1.037, 1.074 | 1.111, 1.164 | 1.234, 1.353 |
| 10 mm | urban | 1.027, 1.076 | 1.089, 1.176 | 1.165, 1.313 |
| 10 mm | urban_v2 | 1.031, 1.077 | 1.097, 1.181 | 1.186, 1.323 |

`[FACT]` Matching the step policy (0.05 mm, TOPAS's `MaxStepSize`) collapses the
angle variance to within 3% and q99.9 to within ~7%. At 0.25 mm the ratio is
~1.17-1.22, which is exactly the Urban
`(c1+c2*ln(h/X0))^2` ratio between h=0.25 (0.635) and h=0.05 (0.524), i.e.
1.21. This is the missing `fMinimal` step limit (D4) and the `theta0`
step-length dependence (D3/D6) manifesting as a slab error.

`[INFERENCE]` The Copper Urban angular law is correct once the step is
matched. The 0.25 mm production step over-scatters the slab by ~20%. The
single-spot collimator entry contrast was *better* at 0.25 mm than at 0.03 mm,
so that contrast gain is a compensating error, not evidence that 0.25 mm is
the correct step. Production must not select the step by the dose contrast.

`[FACT]` Lateral x-variance: at 1 mm TOPAS rms is 0.0021 mm vs GPU 0.021 mm
(ratio ~100); at 10 mm 0.0745 vs 0.0810 mm (ratio 1.09). The GPU slab geometry
uses the minibeam slit block, not the TOPAS 60x60xt box, so this x comparison
is geometry-confounded and is not used for the Copper MSC verdict.

`[FACT]` Energy spread ratio is 1.02-1.03 across all steps and models,
consistent with the earlier straggling validation; the step policy does not
affect it.

## 8. Remaining open gates

- Gate 4/5 (finite slit, Cu-touched split) still need TOPAS MSC-ntuple runs at
  the matched step policy.
- The 1 mm lateral x discrepancy above needs a slab config that reproduces the
  TOPAS box geometry before it can be attributed.

## 9. Gate 3 collimator step-policy matrix (single spot, 250 MeV/u, EM-only, 10M)

Entry contrast error (GPU/TOPAS - 1, pp):

| depth | U 0.25 | U 0.10 | U 0.05 | V2 0.25 | V2 0.10 | V2 0.05 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.88 | 2.31 | 2.44 | 2.89 | 2.39 | 2.59 | 2.80 |
| 9.88 | 0.40 | 1.01 | 1.61 | -0.47 | 0.04 | 1.75 |
| 39.88 | 1.11 | 1.77 | 2.33 | -1.12 | 0.50 | 1.70 |
| 119.88 | -1.79 | -1.40 | -1.21 | -3.64 | -2.23 | -1.63 |

`[FACT]` The slab-correct step (0.05 mm) makes the collimator entry contrast
*worse* (+2.89 pp at 1 mm vs +2.31 pp at 0.25 mm). Since the slab shows 0.05 mm
is the physically correct Copper scattering, the 0.25 mm contrast gain is a
compensating error. `[INFERENCE]` The +2.3-2.9 pp collimator entry residual is
therefore not a Copper MSC amplitude error; it must come from the source,
slit-edge geometry or scoring. This matches the independent water-dominated
attribution in D10 and closes the Copper-MSC-only hypothesis for the residual.

## 10. Gate 1 component truth: angular sampler vs TOPAS per-step ntuple

Double-precision port of `SampleCosineTheta` + `SimpleScattering`
(`benchmark/carbonminibeam/urban_sampler_oracle.py`) vs the TOPAS Copper
per-step ntuple (`minibeam_msc_steps_e250`, 31.5M steps). Ratio of simulated
to measured angle quantiles:

| E (MeV/u) | h (mm) | n | q50 | q68 | q90 | q99 | q99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200.7 | 0.0500 | 36714 | 1.013 | 1.005 | 0.996 | 0.984 | 0.995 |
| 105.3 | 0.0500 | 18976 | 1.013 | 1.010 | 1.004 | 1.013 | 1.003 |
| 50.8 | 0.0500 | 10228 | 0.978 | 0.978 | 0.953 | 0.949 | 0.922 |
| 200.8 | 0.0288 | 23735 | 1.004 | 0.994 | 0.974 | 0.969 | 0.990 |

`[FACT]` The angular sampler matches the TOPAS per-step distribution within
~2% above 100 MeV/u and within ~8% at 50 MeV/u (a small low-energy deficit).
This closes Gate 1 for the angular kernel.

## 11. Verdict

`[FACT]` The Copper Urban MSC port is algorithmically correct after the v2
fixes (Alg96 displacement, t<->z, step-end energy): the component angular
sampler matches TOPAS within 2-8% and the matched-step slab angle variance
matches within 3%. The production `urban` model used the wrong displacement
algorithm and an unmatched step, which are real inequivalences.

> 2026-09-21 correction: "algorithmically correct" is WITHDRAWN as a premise.
> At the time it was written the v2 production path still (a) passed
> `safety_mm=0.0F` from `transport_sycl.cpp:4238`, cancelling every accepted
> displacement (`test_urban_v2_regression.py` R1 reproduces恒零); (b) gated
> process displacement reduction with the single model constant
> `geommin=1e-3 mm`, conflating the `G4VMultipleScattering` geomMin and
> minDisplacement2 families (R2: raw=1e-4/safety=1e-5 wrongly cancels);
> (c) formed `delta=float(t)-float(g)`, which cancels to 0 or wrong sign for
> the production g=0.03-0.10 mm steps (R3); (d) kept `true_path` inside the
> sampler while the caller advanced position and energy loss with the
> geometric length (R4). Items (a)-(d) are fixed structurally this round
> (endpoint isotropic safety, distinct-threshold acceptance helper,
> branch-saved stable delta, proposal/finalization contract), but the
> geomMin/minDisplacement2 NUMERIC values remain UNVALIDATED placeholders
> pending a real `AlongStepDoIt` reference trace, and the range provenance
> (`GetRange->ionisation->theRangeTableForLoss`) plus the t/range energy
> branches are still OPEN. No "done" label below may be reused as a premise
> until those traces exist.

`[INFERENCE]` The remaining single-spot collimator entry contrast residual
(+2.3-2.9 pp) is not a Copper MSC amplitude error: it persists and grows when
the physically correct 0.05 mm step is used, and the same-source water replay
assigns the depth-dependent valley residual to water. The next target is the
source/slit-edge geometry and scoring, not the Copper scattering kernel.

## 12. Gate 5 full-slit: water-entry direct/touched split (250 MeV/u, EM-only)

TOPAS cross run (`minibeam_slit_cross_e250`, 20k histories) flags
Copper-touched tracks by TrackID via the per-step MSC ntuple; the GPU phase
space carries `copper_touched`.

| quantity | TOPAS | urban 0.25 | urban_v2 0.05 |
|---|---:|---:|---:|
| direct fraction | 0.5797 | 0.5796 | 0.5809 |
| touched fraction | 0.4203 | 0.4204 | 0.4191 |
| direct E mean / std (MeV) | 2993.2 / 35.57 | 2992.4 / 36.03 | 2992.4 / 36.03 |
| touched E mean / std (MeV) | 1999.2 / 783.0 | 2029.5 / 767.3 | 2022.3 / 765.1 |
| direct theta q50/q90/q99 (mrad) | 4.243/7.746/10.743 | 4.173/7.633/11.085 | 4.173/7.633/11.085 |
| touched theta q50/q90/q99 (mrad) | 14.000/38.426/75.215 | 13.789/39.246/78.136 | 13.485/38.019/76.443 |

`[FACT]` The direct/touched split matches to 0.1 pp, the conditional energy to
~1.5%, and the conditional water-entry angles to ~2-4% for both urban and
urban_v2. The water-entry conditional phase space is therefore already
matched; the collimator entry contrast residual in section 9 is not visible in
these conditional observables.

## 13. Completed gate summary

| gate | status | evidence |
|---|---|---|
| 0 reference freeze | done | runtime log, configs, hashes recorded |
| 1 component truth | done | sampler q50-q99.9 within 2-8% |
| 2 slab | done | 1/10 mm, 150/250/300 MeV/u |
| 3 step policy | done | slab + collimator step matrix |
| 4 finite slit | partial | split/energy/angle done; no signed-distance scan |
| 5 full slit | done | direct/touched split and conditionals |
| 6 downstream | not run | water/source/nuclear frozen by design |

> 2026-09-21 correction (replaces every "done" above as a premise):
> Gate 1 INVALID_FIXTURE as a full-component claim: the oracle
> (`urban_sampler_oracle.py`) is angle-only; it never exercises the SYCL
> production helper, displacement/path/safety, or the missing low-energy
> sigma `cpositron` branch (scope now documented in the oracle file's
> successor `test_urban_v2_regression.py` R3).
> Gate 2 FAIL (open): the 1 mm lateral-x ~100x gap was excluded as
> "geometry-confounded" without the required TOPAS-box-geometry control slab.
> Gate 3 FAIL (open): there is no real `fMinimal` limiter in the GPU path;
> TOPAS user `MaxStepSize=0.05 mm` must NOT be read as "MSC limited to
> 0.05 mm" (limiting process among user MaxStep / ionisation / MSC /
> geometry / stopping is unconfirmed by trace). The D4 min/max statements in
> section 1 are UNVERIFIED against `G4UrbanMscModel.cc:641-657` source.
> Gate 4 NOT_RUN: sections 14/16/17 record three invalid scan attempts, not
> evidence; the section-17 "pencil-like" inference from a single 0.05 mm
> translation is INSUFFICIENT (no expanded-config/log/return-code/output
> audit; navigation control `test_slit_navigation_control.cpp` now passes on
> the host but shared entry records do not yet exist).
> Gate 5 INVALID_FIXTURE: the TrackID-only TOPAS/GPU join contradicts the
> 2026-09-19 failed.md entry (MT worker blocks reorder records); required
> key is `(run,event,track)` + shard namespace. The 0.1 pp split and the
> conditional energy/angle agreements are therefore NOT established.
> Gate 6 NOT_RUN (unchanged).

## 14. Gate 4 finite-slit scan attempt (2026-09-21)

TOPAS runs locally (`/home/wuwei/topas/topas-build/topas`, 32 threads, ~1 min
per 20k-history run). The cross setup was rerun with the PBS spot `x` offset at
0/0.25/0.5/1.0/2.0 mm. The water-entry C12 yield is 2558/2501/2553/2571/2541,
i.e. offset-independent. The `PencilBeamScanning` spot `x` is therefore not a
transverse offset at the aperture plane; a correct finite-slit scan needs either
an aperture transverse translation or a source position offset in the
`BeamPosition` group. Gate 4 remains open: the direct/touched split and
conditionals (section 12) are done, but the signed-distance `P(exit|d)` scan is
not.

No fabrication: the 5 offset runs exist under `/tmp/topas_gate4/out_x*` but do
not constitute a valid scan.

## 15. Final status

> 2026-09-21: this table is SUPERSEDED by the correction under section 13.
> Retained verbatim for history; do not cite as premise.

| gate | status |
|---|---|
| 0 reference freeze | done (runtime log, configs, hashes) |
| 1 component truth | done (sampler 2-8%) |
| 2 slab 1/10 mm x 150/250/300 | done |
| 3 matched step policy | done (slab + collimator) |
| 4 finite slit | split/energy/angle done; signed-distance scan blocked by source-offset mechanism |
| 5 full slit | done (split 0.1 pp, conditionals 1.5-4%) |
| 6 downstream | not run (water/source/nuclear frozen by design) |

## 16. Gate 4 second attempt

A `d:Ge/PBSBeamFrame/TransX` offset (0/0.25/0.5/1.0 mm) also leaves the
water-entry yield unchanged (2558/2500/2560/2576), so the group translation is
not applied to the source in this configuration either. A valid finite-slit
scan must translate the `Aperture` component itself (or use a source that
accepts an aperture-plane transverse offset). Gate 4's signed-distance scan
remains open; no fabricated result is reported.

## 17. Gate 4 third attempt: aperture translation

`d:Ge/Aperture/TransX` does change the yield: 0 mm -> 2558, but 0.05/0.10/
0.15/0.20/0.25/0.30 mm all -> 0. The beam at the aperture is therefore an
effectively pencil-like beam at x=0 (aperture half-width 0.25 mm); any
translation kills it. A finite-slit `P(exit|d)` curve cannot be resolved with
this source. A broad transverse beam at the aperture (or a scanned narrow beam
sampled across d) is required. Gate 4 remains open; the direct/touched split
and conditional phase space (section 12) are the available evidence.
