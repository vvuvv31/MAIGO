# Actual-cut composition26 electron repair, 2026-10-06

The user authorized a complete switch away from the83 material-density-node
electron transport banks. That switch is **not complete**. Current GPU defaults
still use the83-node charged/photon payloads. No GPU dose or throughput campaign
has run with a complete26-electron replacement in this stage.

## Qualified component changes

- `electron_composition_kinematics.hpp` implements the electron Moller energy
  rejection law inFP32 and retains the TOPAS opt4 `G4DeltaAngle` shell-motion
  generator using the existing26-composition shell package. The primary direction
  follows momentum subtraction from the SAME sampled secondary direction.
  Eight difficult compositions pass101 native conditions (maximum moment
  z3.16871, joint-CDF z3.78755). The remaining18 compositions pass173 conditions
  (maximum moment z2.67260, joint-CDF z4.29934). These are kinematics checks,
  not hard-rate, GPU performance or dose acceptance.
- `penelope_composition_model.py` reconstructs actual-cut oscillator selection
  from native composition oscillator parameters and the original100-point
  cross-section energy grid. The98 offline probability checks have maximum
  absolute difference4.96e-10. Preserve native log interpolation and its
  last-oscillator residual probability; do not renormalize it at the queried
  energy. The FP32 implementation in`electron_penelope_composition.hpp` passes
  21 native primary energy/angle conditions (maximum moment z1.69449,
  joint-CDF z2.75340). Atomic relaxation remains an explicit energy budget,
  **not local-deposition permission**; this is not a complete terminal sampler.
- Native actual-cut lambda reconstruction at one reference density, then mass
  scaling, passes82 independent process/density conditions and42,476 queries:
  maximum peak-relative error0.068702%, active-rate relative error0.787718%.
  This supports the one-composition/actual-cut architecture; it does not qualify
  the new polynomial GPU evaluator or a full replacement bank.

## Rejected shortcuts and remaining work

Ten million low-cut Penelope events versusone million direct target-cut events
reject simple floor-law filtering at section1/E0.04MeV/cut0.0196604674MeV:
energy-weighted mean fraction0.559605 versus0.641591, mean cosine0.628966
versus0.496706, joint-CDF difference0.181294. This is a component distribution
error, not an18% dose-error claim. Use actual-cut oscillator/channel sampling.

The free-electron Moller angle also fails: opt4 explicitly uses bound-shell
motion. The repaired sampler above includes it. Bremsstrahlung recoil is
correlated with photon energy/direction; preserve the full distribution rather
than replacing coarse cells by their mean. The header includes the native
Modified Tsai angular law and momentum-subtraction formula, but the new
brems energy spectrum is not yet qualified.

The native probe now exports exact cut-dependent lambda-grid cubic intervals
and Penelope composition oscillators/selection probabilities. The experimental
native-grid rate builder partitions the physical cut coordinate at grid-bin,
model-switch, resonance, recoil-cut and close-channel support boundaries.
Initial pilots without oscillator support boundaries did not complete their
fit budget; their large offline raw CSV files are NOT runtime package sizes.
The FP32 rate view is experimental; no default loader uses it. The repaired
pilot6 also factors the native Penelope-to-Moller matching pole just below
cut0.05MeV. All26 compositions generate:5,095,354B compressed coefficient
files and5,183,720B arrays, **rate component only**. The expanded FP32 screen
has248 source/off-grid cases and255,192 queries, maximum peak-relative error
0.0850294%, active-rate relative error4.849857%, zero zero-support leakage.
It fails the1% active-rate gate. A one-reference-density mass response still
needs continuous Penelope density/oscillator corrections; the earlier smaller
native screen cannot override this failure.

## User-authorized staged GPU experiment

The user requested GPU full-plan testing before completing all remaining
physics components. `scripts/validate_electron_hybrid26_fullplan.py` freezes
the experimental FP32 binary, runs a two-batch smoke audit, then runs the
RT07575 **head** plan at0.1%:119 batches and236,086,480 primaries, matched
to the existing birth26v3 control seeds and per-spot allocations.
`CARBON_ELECTRON_RATE26_FILE` explicitly enables the5,188,176-byte MAIERG26
rate package; `CARBON_ELECTRON_MOLLER26=1` enables the repaired correlated
Moller sampler above100keV. Production defaults are unchanged.

This is a hybrid experiment: MAIBCA26 birth plus26 hard-rate descriptors and
high-energy Moller outcomes, retaining83-node low-energy ionization rates
and charged terminals,
bremsstrahlung/photon terminals and the existing joint response. Rates use
the existing interpolated terminal cuts. The active-rate component failure
is still recorded; dose testing does not retrospectively qualify it. This
campaign cannot measure the effect of deleting all83-node payloads or
establish complete26-electron memory savings. Full-plan results are recorded
under `out/electron_composition26_20261006/gpu_hybrid26_v2/`.

Remaining prerequisites for the complete switch: qualify actual-cut rates and
their sampling majorants, complete Seltzer-Berger energy sampling and atomic
relaxation, build the26-composition photon component, remove83-node runtime
loading, then run CT audits and dose comparisons. Continue FP32 scoring.
RT07575 remains a head full-plan case; its existing absolute minibeam TOPAS
gate below99% must be reported separately from replacement-versus-control
fidelity. Carbon-minibeam accuracy remains deferred under the user's scope.

Geant4 model references: v11.3.2 `G4MollerBhabhaModel`, `G4DeltaAngle`,
`G4ModifiedTsai`, `G4SeltzerBergerModel`, `G4PenelopeIonisationModel`,
`G4PenelopeIonisationXSHandler`, `G4PenelopeCrossSection`.
Machine-readable evidence: `electron_composition26_20261006.json`.

The repaired GPU smoke passes100,000 primaries in2 batches (maximum
relative physical energy residual1.62115e-8). The first matched full-plan
subset passes23,807,040 primaries in12 batches, all487 active spots;
resident rate74,492.5 primary/s. Its preliminary BODY/TOPAS>=10%max
local3%/0.2mm gamma (search0.02mm) versus birth26v3 control is100%;
relativeL1 is0.00725793%, dose-sum change+0.000009229%. Absolute TOPAS
passes92.173379% versus control92.170684% at this reduced count. These
are preliminary subset results, not the completed0.1% campaign or a
complete26 replacement/default promotion.

The first full-plan integration (v1) stopped at s2 batch6 with one low-energy
ionization terminal coverage miss. New actual-cut rates can activate an event
where both retained terminal bins are empty. Its first12-batch gamma above is
only v1 preliminary evidence and is NOT a completed or accepted campaign.
All v1 outputs are preserved under gpu_hybrid26_low_terminal_failure_v1/.
V2 keeps below100keV ionization rates and terminals together; it uses26
rates for above100keV ionization and bremsstrahlung, with a combined bound
covering the retained low-energy rate. No absent event is scored locally or
ignored. Completing the low-energy26 replacement requires consistent
actual-cut Penelope outcomes and relaxation, or a separately declared and
validated approximation to them.

## Completed v2 head full-plan result

All119 batches/236,086,480 primaries at0.1% complete; all487 active spots,
matched seeds/configurations/allocations, zero queue overflow, zero charged/
photon/joint coverage/invalid failures and zero ion-EM audit failures.
Maximum relative physical energy residual1.2004311e-7. FP32 dose, group32,
ion maximum step0.1mm are retained. Gamma uses BODY includingGTV and
TOPAS>=10% maximum, absolute plan weights, local3%/0.2mm, search0.02mm.

| Measure | Birth26v3 control | Hybrid26v2 |
|---|---:|---:|
| TOPAS local gamma pass |98.899002%|98.897925%|
| GTV local gamma pass |98.233068%|98.229861%|
| Dose-sum bias versus TOPAS |+0.17371072%|+0.17371395%|
| Wall seconds, all runs |3165.777|3269.404|
| Resident primary/s, excluding first batch of each run |74,773.4|72,421.6|

Matched GPU gamma is100%; relativeL1 difference0.00264842%, dose-sum
change+0.0000032221%. Local absolute-difference percentiles95/99/99.9/max
are0.00987477/0.01692493/0.03893471/0.56303961%. This hybrid preserves
matched GPU dose very closely but BOTH absolute TOPAS gates fail>99%.
The control runs were reused rather than interleaved; the observed3.15%
lower resident rate does not show a throughput gain and is not a precise
causal performance estimate. Hybrid26 adds a5.19MB rate component while
retaining shared83 payloads; it does not establish full-package memory savings.

Evidence: gpu_hybrid26_v2/full_0p1_analysis/comparison.json and dose_arrays.npz,
plus three execution.json files and batch quality reports. Experimental opt-in
remains available; production defaults are unchanged. The failed v1 proves
low-energy rates and conditional terminal support must be replaced together.
This experiment supports staging birth/high-energy outcomes, not omitting
untested atomic-relaxation or photon physics. Further low-energy/relaxation
and brems/photon approximation effects require separate component and dose
experiments before retiring the83-node package.
