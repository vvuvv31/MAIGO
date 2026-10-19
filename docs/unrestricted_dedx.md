# Unrestricted electronic stopping in all transport regions

All 18 continuous26 charged-ion species use the native unrestricted electronic
`full` response for total mean ionization loss, electronic LET and medium
CSDA/Urban queries. Electron/photon transport stays absent. Source definitions,
normalization, nuclear channels, primary fluctuations, disabled secondary
straggling, FP32 scoring and forced step caps retain their existing settings.
Medium ionization segmentation is replaced by unrestricted range-based stepping.

## Mean and fluctuations

A segment has one total mean electronic loss. For water/CT, non-ion models
(including protons) use step-start unrestricted stopping when the first-order
loss does not exceed the record's native `linear_limit`. Ion models retain
their charge-dependent midpoint evaluation in that branch. Both use residual
unrestricted CSDA range inversion above `linear_limit`, and deposit the remaining
kinetic energy at the range endpoint. The non-ion/ion distinction is the
package's Geant4 model flag, not the particle's electric charge or CT density.
Cu/air
use step-start full stopping for losses up to 2% of kinetic energy and the
full-range inverse for larger losses.
One shared Universal/Ion sampler receives that full mean and the kinematic
maximum electron transfer `Tmax` as both transfer limits. Convert the native
exported reference-cut dispersion to its unrestricted value using

```text
ratio = (Tmax / beta^2 - 0.5*Tmax)
      / (Tmax / beta^2 - 0.5*min(reference_cut, Tmax))
V_full = V_reference * density_scale * ratio
```

The native ion charge-fluctuation multiplier is already in the exported
variance and is retained. Species/model selection, excitation parameters,
low-energy ion corrections and finite-loss corrections remain active. Universal
thin-layer sampling retains its native bounded Poisson tail. The former
proton-broadbeam dense-tail Gamma replacement, separate aggregate delta sampler,
delta first-moment approximations and soft/hard budget split are removed.
Loss is bounded by available kinetic energy; secondary straggling off returns
the full mean without a random loss draw.

## Stepping

The water/patient step proposal uses the unrestricted residual range `R`,
its native step fraction `f` and final range `x`. For `R <= x`, propose `R` to
terminate the stopping tail. Otherwise, use the smaller of

```text
f*R + x*(1-f)*(2-x/R)
R(E) - R((1-mean_loss_fraction)*E)
```

The second term is a range difference, rather than `fraction*E/S` with a
hard-delta activation condition. Geometry, the forced step cap, Urban
true/geometric path conversion and nuclear interaction distances still compete
with this proposal. The configured broadbeam mean-loss fraction is retained;
minibeams retain the existing internal 1% value. The medium ionization proposal
clock and its continuation state are removed. Shared nuclear optical depths
are independent and remain active.

## CSDA and density

At package load, integrate `1/S_full(E)` over the original linearly interpolated
native full-response nodes, using an analytic logarithmic primitive and double
precision accumulation. Retain FP32 cumulative range nodes on the device;
within a native interval, inversion uses the corresponding analytic exponential.
The native square-root continuation supplies the integral below the first
response node; transport energy coverage and cutoffs are unchanged.
No external range file or new package schema is required.

At actual density, stopping uses the existing continuous Sternheimer law and
linear density scale. A proof of identical density-effect laws, or inactive
density effects over the complete energy interval, permits exact linear range
scaling. Otherwise, integrate the reciprocal-stopping residual in log energy
with eight-point Gauss quadrature and log panels no wider than 0.125. Inversion
uses a safeguarded Newton solve; failed convergence returns an invalid value
for the existing EM audits to reject. This is a continuous-density evaluation,
not a fallback to neighboring density response nodes.

The old restricted-range correction, restricted inverse solver, CT primary
restricted-range-node cache and Poisson-partition mean correction are removed.
Exact actual-density material-field memoization and elastic-rate caches remain.
The immutable EMCNT001 file still carries native restricted fields, splines
and rate coefficients as part of its unchanged schema. The loader validates
them on the host. Disk-only node/record/descriptor types stay private to the
reader; unused restricted/rate fields are projected out before GPU upload.
Spline and continuous range/rate coefficient buffers are discarded after
validation rather than retained by the package cache. Full stopping, native dispersion and continuous material fields
supply the active model. `delta_moments.bin`, its configuration key, reader,
preset dependencies and offline generation/verification are removed.

## Scope

Cu/air upstream electronic loss, including charged fragments, now uses the
required `BLEM0001` package (`minibeam_em_package_file`). Native Geant4
`ComputeElectronicDEDX` supplies full means; native Universal/Ion
`Dispersion(tcut=Tmax)` supplies the variance per unit true path. The full
mean is sampled once. There is no separate hard-delta energy draw, finite-cut
mean selector or air ionization clock. Urban scattering/MFP and nuclear
packages remain independent. Electronic energy lost upstream remains a
beamline sink and is never scored as patient dose.

Additional generic recoil isotopes retain unrestricted electronic plus
nuclear transport stopping from ELRSP001 and electronic-only LET from LERSP001.
Their CSDA/inverse is derived from their full total-stopping table; Urban
restricted loss/range vectors no longer drive recoil stepping. Nuclear
stopping is preserved rather than silently discarded.
The TOPAS scorer and its immutable response format are unchanged.

Earlier dose/throughput measurements describe their frozen implementations;
this switch is a new mean-loss model revision. Numerical consistency checks and
transport audits do not replace matched full-plan TOPAS dose comparisons.

## Earlier mean-only checks on 2026-10-18

These results predate the total-fluctuation and stepping switch above; they
validate the earlier unrestricted-mean revision only.

A mathematical sweep covered 149,760 states: all 18 registered ions, all 26
compositions, five density coordinates and 64 energies. It returned no invalid
states. Maximum relative energy error after range/inverse round-trip was
1.84e-6; maximum deterministic mean-loss discrepancy was 1.16e-7. A subset
checked against independent log-energy numerical integration differed by at
most 1.99e-4. This is a numerical consistency check, not a dose accuracy bound.

RT07575 **head** broadbeam whole-plan tests used 16 million GPU primaries per
projectile in four independent 4-million batches. Proton ran on RTX 2080 Ti
(sm_75), carbon on RTX A6000 (sm_86), both AOT with PTX JIT disabled. Archived
source, physical normalization and whole-campaign largest-remainder spot
allocation were retained. Batch doses were summed once without dose fitting.

| RT07575 whole plan | GPU histories | Existing TOPAS histories | BODY local 1%/1 mm |
|---|---:|---:|---:|
| Proton broadbeam | 16,000,000 | 250,626,500 | 99.993% |
| Carbon broadbeam | 16,000,000 | 129,638,170 | 99.896% |

The ROI is BODY intersected with TOPAS dose greater than 10% of global TOPAS
Dmax; gamma search spacing is 0.1 mm. Both tests exceed the 99% broadbeam gate.
All eight accepted batches passed energy/EM/coverage/completion audits with
zero queue overflows. Maximum relative energy closure residuals were 2.97e-7
(proton) and 3.72e-6 (carbon). An initial single 16-million carbon batch was
rejected for secondary queue overflow and excluded entirely; only the complete
four-batch rerun contributes to these results.

Each of the eight release presets also completed a 2,000-history integration
run. All eight quality reports were accepted, with no failures or queue
overflows. These runs check package integration and live transport audits;
their small statistics are unsuitable for dose or throughput claims.

The new GPU statistics are lower than the TOPAS statistics, so these results
do not replace the historical equal-statistics tables. They do not qualify
minibeam dose or LET agreement. Temporary build and numerical/full-plan check
artifacts are in `tmp/unrestricted_dedx_20261018/` and are not release inputs.

## Total-fluctuation and stepping checks on 2026-10-18

The subsequent switch described above passed these component checks:

- 149,760 composition/species/density/energy states, including unrestricted
  range/inverse, deterministic and fluctuating loss, and the new step proposal:
  no invalid states; maximum inverse relative error 1.84e-6.
- Native Geant4 11.3.2 unrestricted fluctuation comparison: proton, alpha and
  C12 in water, lung and cortical bone at three energies and three mean-loss
  fractions, 81 cases with 120,000 samples per arm. The native dispersion
  conversion agreed to 2.23e-16 relative error in double precision; the largest
  two-sample CDF distance was 0.0068. This checks the unrestricted sampler, not
  equivalence to finite-production-cut TOPAS transport.
- 240,000 paired finite-cut shared-sampler draws retained exactly the old
  loss values and RNG states, checking that the engine rename and removal of
  the medium-only dense-tail option leave native Cu sampling unchanged.

All eight release presets completed 2,000-history integration checks with
accepted quality reports and zero queue overflows. Four primary whole-plan
comparisons used 16 million GPU histories each, split into four resident batches:

| Case / mode | Existing TOPAS histories | Current gamma | Matched prior gamma |
|---|---:|---:|---:|
| RT07575 proton broadbeam | 250,626,500 | local 1%/1 mm: **99.994%** | — |
| RT07575 carbon broadbeam | 129,638,170 | local 1%/1 mm: **99.896%** | — |
| RT07575 proton minibeam | 1,000,000,000 | local 3%/0.2 mm: **85.683%** | 85.660% |
| 20022516 proton minibeam | 3,000,000,000 | local 3%/0.2 mm: **70.155%** | 70.289% |

Broadbeam ROI is BODY and TOPAS >10% global TOPAS Dmax, with 0.1-mm search;
minibeam ROI is BODY and TOPAS >0.4 Gy, with 0.02-mm search. Both broadbeams
pass the >99% gate. The matched minibeam changes are +0.023 and −0.133 percentage
points, within the 1-point relative criterion. Those absolute minibeam values
are limited by the 16-million GPU statistics; they do not replace billion-history
validation tables or establish high-statistics equivalence.

The minibeam baseline is the preceding unrestricted-mean revision with its
soft/delta fluctuation split and ionization clock, at identical histories,
spot counts, source floats, seeds, placement and normalization. The new sampler
consumes random numbers differently, so histories are not trajectory-paired.
All 24 complete whole-plan batches across the four current campaigns and two
matched baselines passed energy/EM/coverage/completion audits without queue
loss or overflow. Current broadbeam maximum energy residuals were 2.92e-7
(proton) and 3.67e-6 (carbon); current minibeam maxima were below 7.44e-8.

RTX 2080 Ti handled proton comparisons and RTX A6000 handled carbon broadbeam;
both used explicit native CUDA AOT images and disabled PTX JIT. All temporary
sources, outputs, numerical checks and comparison records are under
`tmp/unrestricted_transport_20261018/`. No new TOPAS or LET campaign was run,
and carbon-minibeam dose qualification remains separate.


## All-region extension (2026-10-18)

Cu/air full response is extracted independently for 324 native charged species.
The immutable package stores full electronic stopping and native full Ion and
Universal dispersion at float energy nodes. An adaptive refinement resolves
native model transitions. The first-node sqrt(E) stopping continuation is
explicit; there is no missing-data replacement or charge-scaled species fallback.

The host integrates reciprocal piecewise-linear stopping analytically using
`linear_stopping_path`; the inverse uses its exponential primitive. CSDA
prefixes are accumulated and stored in double precision to keep adjacent
native transition knots distinct; physical energies, transport/scoring and
GPU dose atomics remain FP32. The native Urban package supplies MFP and charge
factors for scattering only. Its unused restricted vectors remain validated
file-format fields; there is no runtime loss/range accessor for them.

Retired keys: `minibeam_copper_straggling_model`,
`minibeam_copper_primary_native_mean_loss`, both Cu straggling scales,
`minibeam_air_ionization_rate_file`, and the three beamline stopping-CSV keys.
Custom configurations must supply `minibeam_em_package_file`; there are no
legacy-key aliases. The old Cu split engine, discrete-delta sampler, native
cut/spin constants and species-ratio CSV reader are removed.

Native held-out response validation and CSDA checks are recorded under
`tmp/unrestricted_all_regions_20261018/`. Earlier tables in this document
refer to the preceding medium-only implementation, not the all-region revision.


### All-region verification

The Cu/air package contains 5,340,581 native response nodes (85,458,416 bytes).
Independent held-out maximum relative errors are 1.051e-4 for stopping,
1.164e-4 for Ion variance and 7.129e-5 for Universal variance, below 5e-4.
CSDA/inverse checks cover 5,832 states (maximum inverse error 1.79e-7).
Native Geant4 comparison covers 72 proton/C12/alpha/deuteron × Cu/air ×
energy × mean-loss cases, with 120,000 samples per arm and maximum two-sample
CDF distance 0.00629. All 388,800 shared-wrapper draws were valid; 3,888
fluctuations-disabled cases returned the full mean without RNG draws.
The independent 8,192-step Urban air scattering check also passed.

All eight presets passed 2,000-history integration checks, including energy,
EM, coverage/completion and zero-overflow audits. Four matched whole-plan
screenings used 16 million GPU histories each in four resident batches:

| Case / mode | All-region gamma | Previous medium-only gamma |
|---|---:|---:|
| RT07575 proton broadbeam, local 1%/1 mm | 99.994% | 99.994% |
| RT07575 carbon broadbeam, local 1%/1 mm | 99.896% | 99.896% |
| RT07575 proton minibeam, local 3%/0.2 mm | 86.012% | 85.683% |
| 20022516 proton minibeam, local 3%/0.2 mm | 70.200% | 70.155% |

BODY masks and thresholds/search spacings match the preceding section.
Broadbeam results exceed 99%; matched minibeam changes are +0.329 and
+0.045 percentage points. The low-statistics minibeam values do not replace
the historical billion-history validation. Spot allocation, source strings,
seeds, placement and normalization are fixed between revisions; different
sampler RNG consumption means individual trajectories are not paired.
All 16 campaign batches were accepted with no lost queue energy, overflow
or EM lookup failures. Maximum relative energy residuals were 2.92e-7 and
3.67e-6 for proton/carbon broadbeams and below 8.72e-8 for both minibeams.

RTX 2080 Ti ran the head proton checks; RTX A6000 ran lung proton minibeam
and carbon broadbeam. Both used explicit native AOT images with PTX JIT
prohibited. Evidence and hashes are in
`tmp/unrestricted_all_regions_20261018/validation.json`. Final report-text
corrections remove obsolete claims about restricted fragment loss and retained
hard-delta draws; they do not change transport or the comparison results.
No new TOPAS campaign or full-statistics dose/LET qualification was performed.


## Runtime cleanup after the all-region switch

The shared fluctuation engine now accepts one `Tmax` limit and cannot be
called with a finite production cut. Its native excitation/ionization tail
still samples the full energy-loss distribution; it never creates electron
tracks. The reference cut in the medium descriptor remains necessary to
convert the stored native dispersion to its unrestricted value.

The fixed EMCNT001 disk format is unchanged and remains strictly validated.
Runtime nodes now contain six active floats (24 bytes instead of 52), records
52 bytes instead of 104, and continuous descriptors 36 instead of 64.
Restricted stopping, legacy range/rate fields, correction/model fields and
spline directories no longer enter these GPU objects or interpolation returns.
For the installed package this removes 107,665,688 bytes of device table
payload and 25,533,720 bytes of retained host spline/coefficient buffers.
These are layout savings, not measured throughput gains.

The obsolete `probe_analytic_density.py` experiment and broken
`check_proton_primary_point.cpp` explicit-delta API check were removed.
The production disk-field reconstruction needed by `build_continuous_density.py`
now lives directly in `synthesize_analytic_density.py`, with no dependency on
that experiment. It still reconstructs fields required by the current native
file format; deleting those disk fields would require a separate package
format migration. Source comments and extension instructions no longer claim
that the deleted delta-sidecar path is active.

Matched host checks cover 134,784 medium loss cases and 11,664 Cu/air draws;
outputs and RNG states are bit-for-bit equal before and after cleanup.
The offline packer's native-field reconstruction also matches exactly over
360,828 nodes. Evidence is under `tmp/unrestricted_cleanup_20261018/`.

Native sm_75 compilation and all eight 2,000-history matched GPU integration
checks passed with PTX JIT disabled, accepted energy/EM/coverage/completion audits
and zero queue overflow. The largest dose-array difference from the preceding
all-region revision was 4.511e-07 of peak dose, consistent with FP32
accumulation differences. These short checks establish implementation integrity,
not throughput or new high-statistics TOPAS qualification.

## Shared step-start evaluation (2026-10-18)

The step-start unrestricted response now shares one native energy-interval lookup for stopping, fluctuations and the analytic CSDA primitive. No cross-step cache, continuation state, package format or physical approximation is added. Matched RT07575 full-plan implementation checks and limited performance results are documented in [the shared-evaluation report](unrestricted_shared_eval_20261018.md).
