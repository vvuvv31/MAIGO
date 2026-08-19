# Materials and Methods

This note describes the physical models and their numerical implementation in
MAIGO, a condensed-history Monte Carlo for configured charged-ion dose and dose-averaged
linear energy transfer (LET\(_d\)). The code is research software. It is not a
Geant4 physics-list clone and is not intended for clinical treatment planning.
Reference comparisons use OpenTOPAS 4.2 with Geant4 11.3.2.

## 1. Computational framework

Transport is a class-II condensed-history scheme for charged ions. Continuous
electromagnetic energy loss, energy-loss straggling and multiple Coulomb
scattering (MCS) are applied along a finite step. Discrete nuclear inelastic
events are sampled analogously from macroscopic cross sections. When an
inelastic event is accepted, the outgoing charged fragments are not generated
by an in-kernel cascade generator. They are drawn from precomputed,
correlated final-state packages recorded from TOPAS/Geant4 (INCL++). Charged
secondaries are queued and transported with the same electromagnetic kernel.
A second generation of fragment-induced inelastic events (cascade) is optional.

The production path is a SYCL kernel (`src/transport_sycl_legacy.cpp`) that
runs on Intel Level Zero or NVIDIA CUDA. A serial CPU backend implements only
a one-dimensional CSDA / straggling / primary-attenuation subset and is not
used for three-dimensional patient calculations. Random numbers are
counter-based: each history and physics channel has a deterministic stream
indexed by seed, history index and channel, so a given configuration is
reproducible across devices.

All energy-dependent electromagnetic and nuclear data used at run time are
tables. They are generated offline from the same Geant4 11.3.2 build that
drives the TOPAS reference and loaded as CSV or binary packages
(`data/`, `startup/`). The GPU never calls Geant4 processes.

## 2. Geometry and patient model

A patient volume is a CCTG voxel grid: per-voxel mass density
\(\rho\) (g cm\(^{-3}\)) and a Schneider section index. Version-3 grids also
store the section \(\langle Z/A\rangle\) relative to water and the Bragg
mean excitation energy \(I\). Voxel faces are the geometric boundaries for
step clamping. Homogeneous-region face clamps can be skipped
(`ct_skip_homogeneous_face_clamp`) without changing the material model.

Clinical calculations keep the CT in a fixed patient frame and rotate the
source. For the head plans that match the existing TOPAS scripts, the
historical `tps_90` packing is still used: the CT is reoriented so that
patient \(\pm X\) becomes transport depth (GPU \(+Z\)), and the world TPS-0°
beam (source at \(y=-\mathrm{SAD}\), direction \(+Y\)) is mapped through
TOPAS `Patient/RotZ` then `Trans`. TOPAS applies the rotation first and the
recorded translation afterwards; the GPU uses the same order
(`transform_tps_90_pose_to_ct`). Pencil-beam scanning (PBS) spots are
optionally placed with virtual scanning magnets (VSAD \(X,Y\) and source-plane
distance \(D\)): the source-plane offset is
\(x_\mathrm{src}=x_\mathrm{iso}(S_X-D)/S_X\) (and likewise for \(Y\)), and
the ray is aimed at isocentre. That geometry is not equivalent to rotating
an unrotated CT by a TPS beam-angle number.

Uniform water, axial slabs and a single heterogeneous insert remain available
for phantom work. They use the same electromagnetic and nuclear kernels with
material tables selected by layer or insert rather than by Schneider section.

## 3. Source

A plan is a list of spots. Each spot carries kinetic energy (total ion
kinetic energy or MeV/u), a history count or MU-proportional allocation,
optional Gaussian energy spread, and a bi-Gaussian emittance
(\(\sigma_{x,y}\), \(\sigma_{x',y'}\), correlations) in the beam frame.
Two input paths are supported:

- TOPAS Time Feature `spots_*.txt` (per-spot `Trans` / `RotX` / `RotY` /
  histories);
- TPS / PBS CSV (`spots.csv` plus optional `beam_model.csv` optics).

Primary particles are launched in one batched SYCL kernel. The history
count of a production full plan is the integer L4 allocation used by TOPAS,
not a renormalized fluence.

## 4. Charged-particle stepping

A step length is

\[
\Delta s=\min\Bigl(\Delta s_\max,\;
\frac{\varepsilon\,E}{S(E)}\Bigr),
\]

where \(\Delta s_\max\) is `maximum_step_mm`, \(\varepsilon\) is
`maximum_relative_energy_loss`, \(E\) is the current kinetic energy and
\(S(E)\) is the local stopping power. The step is further clipped to the
next voxel face (unless a homogeneous skip applies) and to the remaining
distance to the energy cutoff. Production CT runs typically use
\(\Delta s_\max=0.1\,\mathrm{mm}\) and \(\varepsilon=10^{-3}\). Larger
\(\Delta s_\max\) and a condensed secondary step
(`secondary_condensed_step_mm`) trade spatial resolution for throughput;
they do not change the tabulated physics.

## 5. Continuous energy loss

The mean electromagnetic loss on a step is the condensed CSDA increment

\[
\Delta E_\mathrm{cont}=S_{p,m}(E)\,\Delta s,
\]

with \(S_{p,m}\) interpolated from Geant4 11.3.2 tables for projectile
species \(p\) and local material \(m\). The primary uses
`primary_stopping_power_file`; its identity is explicit in
`primary_atomic_number`, `primary_mass_number`, and optionally
`primary_rest_mass_MeV` (zero retains the historical
\(A\times931.49410242\) MeV approximation). Fragments use isotope-specific water ratios
(`ion_stopping_power_*`) applied to the continuous carbon table when
`use_particle_specific_stopping_power` is on; missing isotopes fall back to
water or to an effective-charge scaling of the carbon table.

In CT, the ionization model is chosen in this order:

1. A configured HU / Schneider-section mass-stopping-power LUT sampled from
   TOPAS (`ct_hu_stopping_power_lut_file`).
2. Otherwise, if `ct_use_density_mass_spr` is true, a continuous mass SPR
   \(\mathrm{SPR}(\rho,E)\) built from air, lung, water and bone carbon
   tables (moqui-style).
3. Otherwise an analytic Schneider-section Bethe factor from the stored
   \(\langle Z/A\rangle\) and \(I\).

Nuclear cross sections and reaction packages are selected by Schneider
section independently of the ionization backend. Soft tissue uses Schneider
section 7 as the residual class; lung and bone can load dedicated packages.

There is no explicit \(\delta\)-electron, bremsstrahlung or photon transport
in the production charged kernel. Optional `electronic_buildup_fraction`
is a local suppression of unrestricted \(dE/dx\) near the entrance and is
off by default. It is not equivalent to transporting knock-on electrons.

## 6. Energy-loss straggling

When `enable_energy_straggling` is on, the actual loss on a primary step is
sampled from a condensed total-loss distribution whose variance is the
heavy-particle collision integral with \(T_\mathrm{cut}=T_\mathrm{max}\):

\[
\sigma^2=K\,m_e\,z_\mathrm{eff}^2\,\Bigl(\frac{Z}{A}\Bigr)
\rho\,\Delta s\cdot
\frac{T_\mathrm{max}/\beta^2-T_\mathrm{max}/2}{2m_e}.
\]

Here \(K=0.307075\,\mathrm{MeV}\,\mathrm{cm}^2\,\mathrm{g}^{-1}\),
\(z_\mathrm{eff}\) is the Barkas-type effective charge of the projectile,
and \(T_\mathrm{max}\) is the kinematic maximum energy transfer to a free
electron. The relativistic multiplier tends to one in the Bohr limit. The
sampled loss is drawn as a Gaussian about \(\Delta E_\mathrm{cont}\) and
clamped to \([0,\min(2\Delta E_\mathrm{cont},E)]\).

This is the same physical purpose as Geant4 ion fluctuations
(`g4em-standard_opt4`) but not the same algorithm: there is no
Gaussian / Vavilov / Urban regime switch and no explicit hard-collision
split. Fragment straggling is a separate switch
(`enable_secondary_energy_straggling`) and is off in current CT
production.

## 7. Multiple Coulomb scattering

MCS uses a Highland projected RMS angle

\[
\theta_\mathrm{rms}
=\frac{13.6\,\mathrm{MeV}\,z}{\beta p}\,
\sqrt{\frac{x}{X_0}}
\Bigl[1+0.038\ln\Bigl(\frac{x z^2}{X_0\beta^2}\Bigr)\Bigr],
\]

implemented in `include/carbon/multiple_scattering.hpp`. The areal density
\(x=\rho\Delta s\) converts a geometric step to radiation lengths. Two
independent Gaussians in the beam transverse plane are applied and the
direction is renormalized.

By default CT production sets `enable_ct_material_mcs: false`, so every
voxel uses the water mass radiation length \(X_0=36.08\,\mathrm{g}\,\mathrm{cm}^{-2}\).
Material-conditioned \(X_0\) values measured from Geant4 air, lung, water
and compact bone are available when that switch is on. There is no
independent hadronic-elastic process; some nuclear angular structure
enters only through the precomputed inelastic final states.

## 8. Nuclear inelastic interactions

### 8.1 Interaction probability

The macroscopic inelastic cross section \(\Sigma(E,m)\) is interpolated from
the configured primary inelastic table and per-projectile cascade tables. On
a step the analog interaction probability is

\[
P_\mathrm{int}=1-\exp\bigl[-\Sigma(E,m)\,\Delta s\bigr].
\]

A uniform random number decides whether the step is interrupted. The
interaction site is taken at the end of the accepted step (standard analog
MC along a short condensed step). Schneider-section tables supersede the
four-class air/lung/water/bone tables when present.

### 8.2 Correlated final states

MAIGO does not run a nuclear generator on the device. Offline, a user-selected
inelastic model records correlated events into binary packages
(`CRPKG` reaction, `CCAS` cascade). Each sampled interaction stores:

- incident energy per nucleon and a reaction depth;
- the Geant4 step-local deposit \(E_\mathrm{loc}=\)
  `G4Step::GetTotalEnergyDeposit()`;
- every charged product with \(Z\), \(A\), kinetic energy and a
  three-dimensional direction.

At run time the code selects the energy bin of the projectile, draws one
precomputed event uniformly from that bin, rescales product kinetic
energies to the actual projectile energy, and rotates the recorded
directions into the current projectile frame. The local deposit is scored
at the interaction point. Using Geant4’s step-local deposit avoids treating
the kinematic imbalance \(E_\mathrm{in}-\sum E_\mathrm{out}\) (reaction
\(Q\) value and mass defect) as dose.

Unsupported products, recoils below the transport cutoff and any energy
that cannot be queued are added to a residual nuclear-heat tally so that
the energy ledger remains closed. Package layout v1 is required;
older files without 3-D directions or local deposit are rejected. The package
compiler records the source model as provenance but does not require INCL++ or
TOPAS/Geant4 branding; users are responsible for making the configured primary
Z/A, stopping table, cross section, and package physically consistent.

### 8.3 Charged secondaries and cascade

Accepted charged products enter a device secondary queue and are
transported in generation batches. Each secondary continues CSDA (and
optional straggling), Highland MCS, and analog inelastic sampling from
projectile-specific cascade cross sections. `maximum_cascade_generations`
(production value 2) caps how many subsequent inelastic generations are
followed. Kinetic energy below
`secondary_local_deposit_cutoff_MeV` is deposited locally and the track
stops.

Queue capacity is a numerical, not physical, parameter. An overflow
truncates charged secondaries and must be zero for a validation run.

### 8.4 Scope of ion generalization

The shared algorithm is voxel traversal, step control, continuous loss,
straggling, MCS, reaction sampling, queues, RNG, dose, and LET scoring.
Primary mass/charge and all primary physics data are configurable, so proton,
helium, carbon, oxygen, and neon runs can use their own prepared datasets.
This is a transport interface generalization, not a claim of cross-ion physics
validation: package v1 does not embed primary identity, scorer species columns
remain fixed (Z>6 products are `secondary_other_charged`), and hadronic
elastic, decay, and arbitrary-species scoring are not implemented. The Copper
minibeam calibration is explicitly restricted to C-12.

### 8.5 Neutrals, decay and electrons

Neutrons and photons can be pushed to an optional package-driven neutral
queue (`enable_neutral_transport`). Production CT configurations leave this
off and set the local kerma fraction to zero, so those particles neither
transport nor deposit. There is no general electron / positron / photon
cascade, no `G4DecayPhysics` lifetime module, no radioactive-decay chain
and no at-rest hadronic capture process. Those TOPAS modules therefore
have no one-to-one GPU counterpart.

## 9. Scoring

### 9.1 Absorbed dose to medium

Energy is accumulated from continuous loss, secondary tracks and
reaction-local deposits. Voxel dose is

\[
D_v=\frac{E_{\mathrm{dep},v}}{\rho_v V_v}.
\]

This is the same quantity as TOPAS `DoseToMedium`. Because electrons,
photons, neutrons and decay products are not transported in the production
GPU configuration, their contribution is missing except insofar as it was
already folded into the recorded Geant4 local deposit of an inelastic
event. Production comparisons use an unscaled scorer
(`dose_output_scale=1`). Outputs are dense MetaImage MHD/RAW on the
transport grid; a `tps_90` grid is remapped to native patient
\((z,y,x)\) before gamma evaluation.

### 9.2 Dose-averaged LET

With `scorerLET` / `LET: true` the kernel accumulates dose-weighted
moments

\[
\mathrm{LET}_d=\frac{\sum_i L_i\,\Delta E_i}{\sum_i\Delta E_i},
\]

separately for primary \(^{12}\mathrm{C}\) and all charged hadrons.
\(L_i\) is a restricted electronic stopping power. An optional
delta-electron fraction table
(`let_delta_electron_fraction_*`) converts unrestricted table \(dE/dx\)
into a restricted value; it changes the LET definition only, not the
transported dose. Tracks that fall below the transport cutoff still
contribute a terminal LET moment so that the Bragg peak is not
systematically low. The scorer target matches a dose-weighted HadronLET
definition; the transport underneath is the surrogate described above.

Gamma analysis against TOPAS uses every BODY voxel whose reference dose is
at least 10% of the BODY maximum, with 3%/3 mm, 2%/2 mm, 1%/1 mm and
3%/0 mm global and local criteria on a half-voxel trilinear refinement.

## 10. Numerical implementation notes

On NVIDIA devices, production builds typically use FP32 dose and LET
atomics (`CARBON_DOSE_FP32`). An energy-balance identity is reported after
every run:

\[
E_\mathrm{in}
=E_\mathrm{dep}+E_\mathrm{esc}
+E_\mathrm{beamline}
+E_\mathrm{untracked}.
\]

Relative imbalance of order \(10^{-6}\)–\(10^{-7}\) is typical. Primary and
secondary kernel times are printed separately so that step-size and
secondary-cutoff studies can be attributed.

Random seeds for multi-shard high-statistics runs are offset per shard; dose
is summed and LET is combined as a dose-weighted mean.

## 11. Correspondence to the TOPAS reference

The TOPAS full-plan physics list is the seven-module Geant4 modular set
`g4em-standard_opt4`, `g4h-phy_QGSP_BIC_HP`, `g4decay`, `g4ion-inclxx`,
`g4h-elastic_HP`, `g4stopping` and `g4radioactivedecay`. The GPU covers the
charged-ion chain that dominates carbon-ion dose:

| Process | Implementation | Relation to TOPAS |
|---|---|---|
| Continuous \(dE/dx\) | Geant4 tables + CT SPR | Same quantity, tabulated |
| Energy straggling | Relativistic condensed total-loss | Same purpose, not G4IonFluctuations |
| MCS | Highland, default water \(X_0\) | Approximate |
| \(^{12}\mathrm{C}\) inelastic | \(\Sigma(E)\) + INCL++ event package | Package surrogate, not runtime INCL++ |
| Fragment transport | Isotope \(dE/dx\) + MCS + cascade package | Same purpose, two generations |
| Neutrals, \(e^\pm/\gamma\), decay | Off / absent | Not covered |

Agreement with TOPAS on a given plan is therefore a validation of this
surrogate on that geometry, energy range, material map and scorer
definition. It is not a claim of process-by-process equivalence to the
Geant4 list.

## References

1. Geant4 Collaboration. *Physics Reference Manual*.
2. TOPAS. *Modular Physics Lists*.
3. Highland VL. Some practical remarks on multiple scattering. *Nucl. Instrum. Methods* 1975; 129:497–499.
4. MAIGO implementation: `include/carbon/{stopping_power,straggling,multiple_scattering,reaction_package,transport_config}.hpp`, `src/transport_sycl_legacy.cpp`, `startup/`.
