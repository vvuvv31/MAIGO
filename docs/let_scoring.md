# Electronic LET scoring

The optional `letd_output_prefix: letd` setting adds patient/phantom voxel LET
scoring to the current ion transport. It does not enable electron transport,
draw random numbers, or change dose normalization.
Release presets leave it unset. The retired `LET: true` interface is not restored.
Voxel scoring must be enabled. The standalone campaign driver can set the same
`TransportConfig::letd_output_prefix` member before running a batch.

For each finite charged-ion segment, let `S_e` be unrestricted **linear**
electronic stopping power at the **pre-step** kinetic energy and `dl` the
accepted **true** path length. The outputs are:

- `<prefix>_numerator.mhd/raw`: `sum(S_e^2 dl)`, in MeV²/mm.
- `<prefix>_denominator.mhd/raw`: `sum(S_e dl)`, in MeV.
- `<prefix>.mhd/raw`: their ratio, in keV/µm (numerically equal to MeV/mm).

This is the stopping-power track-length estimator of dose-averaged electronic
LET. It uses expected electronic energy loss as its dose weight; it is not an
average of sampled `dE/dl`, an average over electrons, or a mass stopping power
normalized by density. GPU `S_e` uses the native unrestricted electronic
`full` response in continuous26, evaluated at the same pre-step energy, with
linear density scaling and the continuous Sternheimer density-effect correction.
The same unrestricted response now sets the total mean ionization loss and
CSDA range used by medium transport. Medium fluctuations use one unrestricted Universal/Ion draw
with that mean; LET still scores expected stopping at pre-step energy rather than
the sampled loss. Enabling LET alone does not change transport or its random draws. Native low-energy model transitions remain
limited by the existing response-node interpolation; this is not a claim of
exact stopping at every energy.
Generic recoils use a separate **electronic-only** recoil table selected with
`letd_recoil_electronic_stopping_file`. The transport `ELRSP001` table contains
electronic plus nuclear stopping and must not be reused as electronic LET.
The LET-only `LERSP001` bank must match its isotope and energy coverage; it
does not replace the transport stopping table or change any trajectory.
It is loaded only when LET and medium all-ion elastic are both enabled.

```yaml
letd_output_prefix: letd
letd_recoil_electronic_stopping_file: data/let/schneider/recoil_electronic_stopping.bin
```

The eight dose-only presets do not need this bank. LET runs require an
explicitly supplied bank whose isotope/grid coverage matches their transport
recoil bank; additional transported isotopes may require regenerating both.

All transported charged ions contribute, including primary ions and transported
fragments/recoils. Copper/air upstream transport is excluded from the patient
scorer. Nuclear vertex deposits, zero-length cutoff/local deposits, untracked
neutral energy and killed/untransported products have no finite scored ion path
and are excluded. Match the TOPAS LET kinetic-energy scoring threshold to the GPU transport
cutoff (0.1 MeV total kinetic energy in the release configurations). Changing
that cutoff changes the population represented by LET.

Zero-denominator voxels are written as zero, meaning no scored track; the saved
denominator must be used to distinguish them from measured LET values. Finite,
nonnegative moments and consistent zero denominators are required at output.
Dose retains its existing FP32 atomics and quality/energy/overflow audits. LET
uses independent FP32 atomics, so changing execution order can change the last
few bits of dose without changing trajectories or random numbers.

## TOPAS counterpart

`extensions/topas/let/ElectronicLETMoment` records the same two additive moments.
Set `Moment = 2` for the numerator and `Moment = 1` for the denominator, both on
`Patient`, with `MinimumKineticEnergy = 0.1 MeV` and the same required
`StoppingPowerFile`. The scorer reads an immutable `ESLUT001` native unrestricted
electronic stopping response, uses pre-step energy, true step length, and track
weight, and includes particles with atomic number and mass at least one.
Its binary fields are numeric MeV/mm quantities; TOPAS labels these raw moments
as unitless, so use the units above.

Do **not** call `G4EmCalculator::ComputeElectronicDEDX` inside a transport scorer,
even at event end. It selects and modifies live Geant4 model caches. The
immutable reader leaves transport models and random numbers untouched.

The response is extracted in a **separate process**, with the same Geant4 version,
physics list, projectile identities, actual material composition, density and I.
`ESLUT001` stores those material properties and total-kinetic-energy/stopping
pairs. The reader rejects material mismatches, missing isotope/material pairs
and out-of-range energies; it does not fall back to live queries or extrapolate.
Supply a matched offline response covering the entire scored geometry and all
charged isotopes that can be transported, including rare residuals and boundary
air where applicable. A central-spot response is insufficient for a full plan.
The case-specific extraction/probing scripts and experimental response assets
are not part of the runtime or the release presets.

Example parameters after preparing a complete matched response:

```text
s:Sc/LETNum/StoppingPowerFile = "/path/to/native_stopping_response.bin"
s:Sc/LETDen/StoppingPowerFile = "/path/to/native_stopping_response.bin"
```

Build it in a separate output directory while preserving the original build:

```bash
python3 extensions/topas/let/build.py \
  --topas-build /path/to/existing/topas-build \
  --output /path/to/new/let-build
```

For electronic recoil bank regeneration, add `--with-recoil-extractor` to
this build command, then prepare a one-history extraction input with
`extensions/tools/general_ion_elastic/prepare_campaign.py --recoil-only
--recoil-electronic --hu-file ... --dicom-dir ... --output ...`. The patient
must instantiate all 25 Schneider section probes. Run the generated input
with the newly built TOPAS executable. The new `LERSP001` payload
uses `ComputeElectronicDEDX`; ordinary recoil extraction remains `ELRSP001`
with `ComputeTotalDEDX`.
For nonelastic residuals that are not natural target isotopes, explicitly add
`--stopping-isotope 8:15` (repeatable) to **both** extraction commands. The
resulting total/electronic banks must have identical isotope and energy coverage.
Natural-target coverage alone does not cover every transported charged residual.
Extending the total transport bank is a separate data change; keep its existing
projectile calibrations and update the matching electronic bank together.

The builder retains the existing extension archive and source extension, adds
the two-moment scorer and, if absent, registers the repository's existing
`AllChargedUrbanPhysics` module. A module is only used when selected in the
TOPAS parameter file. Never overwrite an executable used by a running campaign.

## Native reference step control

Pre-step stopping is a discretized estimator. A native heavy ion can lose a
large fraction of its energy in one step, especially near termination, so a
coarse TOPAS dose calculation is not automatically a converged LET reference.
Converge both additive moments for the selected reference step controls before
comparing LET. No LET scale factor is applied.

The builder registers optional `IonLETStepPhysics`. To use it, append
`"IonLETStepPhysics"` **after** the EM constructor in `Ph/Default/Modules` and
increase that vector's count, then set all three required parameters:

```text
u:Ph/IonLETStepPhysics/RangeFraction = 0.005
d:Ph/IonLETStepPhysics/FinalRange = 0.1 nm
b:Ph/IonLETStepPhysics/IncludeProtons = "False"
```

These are Geant4's native range-relative loss-step controls, not a spatial
maximum step or an energy cutoff. They are applied during physics initialization,
independently of the immutable scorer. Light-ion controls include deuteron,
triton, He3 and alpha; GenericIon has its own control. `IncludeProtons = "True"`
also changes the native MuHad group (including other hadrons), so make this
choice explicit and check convergence for the reference configuration. This
module is not enabled automatically by LET scoring or in existing dose-only
campaigns. Match the physics list,
source, cutoff and geometry, then verify voxel moments at adequate statistics
before reporting a new full-plan LET gamma.

## Merging and comparison

Sum the numerator and denominator separately across batches, fields and devices,
then divide **once**. Do not average LET images, scale LET by prescription dose,
or multiply normalized dose by the number of batches. Apply exactly the same
spatial mapping to both moments before combining rotated/packed GPU fields.
Use matched campaign-wide spot allocations and independent seeds; record native
GPU architecture, executable/source hashes, patient inputs and physics lists.
Differences between TOPAS and GPU transport remain model differences; matching
this scorer definition does not establish LET accuracy by itself.

Historical `myHadronLET` outputs use a different restricted/deposition weighting
and density normalization. They must not be substituted for this matched scorer.
