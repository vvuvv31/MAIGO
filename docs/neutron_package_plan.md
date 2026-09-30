# Optional neutron package: implementation status and contract

Status: experimental GPU neutron path connected at user request, 2026-09-27.
The Water_75eV/Cu/air bank contains 491,520 conditional collisions and 1,849,482
products on 111 energy nodes, 1e-11–300 MeV. A separate rate extraction has
32,777 nodes per material. These are draft data, **not runtime validated**.

`enable_neutron_transport: false` retains the existing path. Setting it true
now runs an explicitly experimental GPU pass in `run_mode: research`. This
uses the independent v2 rates and sparse correlated collision bank, repeated
neutron collisions, physical water/Cu/slit-air/world flight and straight CSDA
charged recoils. Full descendant transport and physical accuracy are incomplete.
This supersedes the staged runtime rejection described in earlier milestones.

Raw data and draft packages: `/mnt/sda/wuwei/neutron_package_20260927/`.
Extraction scripts and audits: `evidence/neutron_package_20260927/`.
The standalone Geant4 physics constructors match the reference neutron modules;
this is native Geant4 extraction, not a completed TOPAS transport validation.

## Purpose and first scope

Complete the missing neutron contribution identified by the proton 170 MeV
minibeam entrance study. Primary comparison metrics are peak/valley depth dose
and integrated dose over 0–20 mm; retain the full TOPAS reference. Do not add
measured TOPAS component fractions to GPU dose.

The first geometry scope should cover Water_75eV, natural copper and the actual
beamline air composition. Schneider/CT and carbon-source energy coverage need
separate validation. Package coverage is expressed in neutron kinetic energy,
not source energy per nucleon. Select the upper bound after auditing neutron
birth energies in the intended proton source range. Cover moderation through
thermal energies and capture; do not reuse the charged-particle 0.1 MeV cutoff.

## Experimental YAML

```yaml
enable_neutron_transport: false
neutron_package_file: /mnt/sda/wuwei/neutron_package_20260927/independent_rates_package/package.json
run_mode: research
neutron_world_half_width_mm: 3000
neutron_max_collisions: 4096
```

Disabled: retain present physics and avoid allocating neutron tables/queues.
Planned enabled behavior: validate manifest, checksums, material identities, supported geometry,
energy coverage and schema, then load tables and run the neutron queue.
Reject missing or incompatible packages and queue/coverage failures. An accepted
run must not silently discard unsupported neutron states.

The existing `minibeam_copper_neutral_cross_section_file` field is only parsed
in the current tree; it does not enable neutron transport. Do not describe it
as a working switch or reuse it for a different contract without migration.

## Data that can be reused

`Cinel03ProductRecord` and `Cinel03DeviceProduct` retain PDG, kinetic energy,
local direction, weight and nuclear identity. Interaction position and event
rotation are available at the actual birth. Copper and water kernels currently
skip neutral products. Capture neutron births before those skips, with stable
history/lineage/RNG identity; never infer a source from the missing dose map.

## Native extraction contract

Pin TOPAS/Geant4 11.3.2, the selected neutron HP/high-energy models, data-library
versions, material isotopes and temperature. Run extraction/builds with local
Slurm. Collect native macroscopic rates (1/mm) and correlated conditional final
states for elastic, inelastic and capture processes, including outgoing
neutrons, charged recoils/fragments, photons, local deposit and Q-value terms.

Preserve process/model transitions and low-energy structure. Grid interpolation
must not smooth away resonance behavior or independently rescale correlated
reaction energies. Retain null outcomes where applicable. Audits must separate
kinetic-energy balance from nuclear Q-value release.

## Runtime contract

1. Enqueue primary/secondary nuclear neutron births in copper and water.
2. Sample collision optical depth and preserve the remaining optical depth
   across copper/slit-air/water boundaries using material-specific rates.
3. Sample the selected process and its native correlated final state.
4. Requeue surviving/new neutrons; dispatch charged descendants to the existing
   charged transport with neutron ancestry, and score native local deposits.
5. Handle capture/inelastic photons and their electron descendants through an
   explicit validated path. Merely dropping photons cannot reproduce the full
   TOPAS neutron-lineage dose. Current charged/electron response support must
   not be assumed to provide a general photon transport engine.
6. Track escaped, deposited, surviving, Q-value and overflow energies separately.
   Use FP32 GPU scoring/atomics, with FP64 dose OFF.

The historical first-interaction neutral mode is not a sufficient substitute:
moderation, repeated scattering and capture require continuation transport.

## Acceptance sequence

- Native material rate and final-state audits, especially n–H elastic,
  low-energy moderation and capture.
- Neutron pencil/slab comparisons at representative energies and material
  interfaces, including angular/energy correlations and secondary dose.
- Switch-off regression for the existing proton/carbon configurations.
- Switch-on 170 MeV proton minibeam comparison against full pure-Urban TOPAS;
  inspect the separately scored neutron-lineage dose and 0–20 mm depth shape.
- Queue overflow, unsupported domain and energy-accounting gates must pass.

Evidence motivating this work:
`evidence/proton_entrance20_20260927/REPORT_zh.md` and
`evidence/proton_entrance20_20260927/transport_scope_audit.json`.

## Findings that constrain implementation

- Coarse log-energy/linear-rate interpolation is rejected. Relative to the dense
  samples, copper elastic p95 error is about 96.5%; capture p95 is about 205%.
  A dense rate grid needs an independent midpoint/resonance audit. A dense rate
  grid alone does not solve conditional final-state sampling between event nodes.
- Explicit model registry seams include 19.5, 19.9 and 20 MeV. Preserve these
  transitions when generating adaptive nodes.
- Final states include photons, electrons, isotope-specific recoils, excited
  nuclei and rare pions. Product counts from a conditional bank are not physical
  production fractions. Unsupported products cannot be silently discarded.
- `kinetic_gain` is measured final kinetic energy plus deposit minus incident
  kinetic energy. Its serialization closure is not an independent nuclear-Q or
  dose validation.

## GPU birth input milestone

The opt-in `neutron_birth_output_file` now captures all five active CINEL03
neutron birth sites in homogeneous-water minibeam runs. It preserves source
history, derived RNG identity, common event rotation, true GPU position, energy,
weight, material and parent ZA. `neutron_birth_capacity` defaults to one million
and overflow rejects the run. This capture does not yet transport neutrons.

The 100k 170 MeV matched-seed run collected 35,553 births (33,418 Cu / 2,135
water). On/off comparisons to the frozen binary have dose relative L1 below
1.7e-8 and pass queue/quality/Urban checks. This establishes the birth input
path, not the planned neutron transport switch. See the evidence report.

Correction to the initial birth geometry audit: x=-71.53 mm is outside the
scoring ROI (+/-50 mm), but inside its Water_75eV parent (+/-2000 mm), matching
the GPU physical water extent. All captured births pass the full physical
volume hierarchy check. No boundary transport change is needed on that basis.
`birth_geometry_audit_corrected.json` supersedes the earlier conclusion.

Latest rate progress is in `evidence/neutron_package_20260927/rate_status.json`.
The 1,992-node water draft has no unresolved sampled midpoint checks; the air
and copper tables still require further work. None of these statuses certify
conditional events, photon transport, or full neutron dose.

## Native replay reference milestone

Four independent 250k GPU proton runs produced 364,722 neutron births. Their
full-geometry TOPAS replays are normalized to the original incident protons,
not neutron count. The reproducible reference uses one TOPAS worker; adding
neutron/gamma component scorers leaves all four total maps byte-identical.
32-worker phase-space replays did not pass the matched-seed total-map check,
so their components are not mixed with the one-worker reference.

For central pooled valleys, 0–20 mm, native transport of GPU births differs
from full TOPAS neutron-lineage dose by +12.12 +/-10.41 percentage points
(1 SEM). The finite source bank and conditional event-library bias are not
validated by this agreement. Dose within that neutron lineage is 70.39%
protons, 4.96% electrons/positrons, 0.048% neutron local deposit, 0.006% photon
local deposit, and 24.60% other species including recoils/fragments. Thus
proton-only descendants or local deposition cannot be the full backend.
Photons can contribute through electron descendants even though their direct
local-deposit fraction is small. GPU neutron transport remains unimplemented.

## Independent macroscopic grid milestone

The v2 draft separates the collision event bank from the refined macroscopic
rate grid. `extensions/tools/neutron/assemble.py` preserves the v1 collision
binary and creates a checksummed `macroscopic.bin`, including per-node SEM,
query counts and outgoing interval audit status. The C++ audit loader checks
both binaries, process coverage, common energy grids, interval termination and
event-bank energy coverage. Runtime activation remains rejected.

Slurm 18796 compiled the reader and loaded both the original bank and the new
`independent_rates_package`: 491,520 events, 1,849,482 products and 76,284
independent macroscopic records (19,071 energy nodes across three materials).
This snapshot uses completed Water curvature and Cu/Air extended extractions;
the Cu/Air statistical refresh is still pending. Unresolved intervals remain
explicitly marked in the binary. Passing this structural audit does not certify
rates, conditional event interpolation or GPU neutron dose.

## First integrated experiment and known approximations

The user requested integration before complete validation. The production
loader remains strict, while the research path explicitly loads draft data.
`src/neutron_transport_sycl.cpp` runs on the same GPU queue after the charged
pass and adds only its calculated water deposit to total depth/voxel dose.
`neutron_transport.json` itemizes input, Q/thermal kinetic gain, water/outside
deposit, escape, unsupported particles, energy coverage and truncation.
`neutron_deposit_MeV.mhd` is the separate contribution in MeV, not Gy. Existing
charged category maps exclude it; category closure includes this independent
map. Dose uncertainty and LET components do not yet include neutron lineage.

- Fine rates interpolate linearly in log energy, preserving stored FP32 seams.
  Statistical-limit interval queries are counted and permitted in this experiment.
- Conditional nodes are selected stochastically in log energy. All outgoing
  kinetic energies and native local deposit share the factor
  `(actual incident E + recorded gain)/(bank incident E + recorded gain)`.
  This preserves correlated directions and kinetic bookkeeping but is an
  unvalidated approximation to final states, thresholds and momentum balance.
- Neutrons retain sampled optical depth across boundaries and may return to
  water. The world is configured explicitly; water extent is +/-2000 mm in
  the reference, distinct from the +/-50 mm scorer.
- Charged descendants use stopping tables with straight paths and continuous
  mean loss, without MCS, straggling, nuclear reactions or relocated delta rays.
  Water preferentially uses the available recoil table; otherwise isotope CSV
  stopping is used. Below the first stopping node, `S(E) = S0*sqrt(E/E0)`;
  remaining energy below 1 eV is deposited locally. Air currently supports
  proton stopping; missing species are recorded as unresolved energy.
- Photons, electrons and other unsupported products remain explicit energy
  sinks; they are not deposited locally or replaced by fitted missing fractions.
- Each source neutron uses a 64-entry descendant stack. Overflow remains an
  error; collision/step limits retain terminal energy in the diagnostic ledger.

First Slurm build 18799 and direct 100k-proton run produced 35,553 neutron
births and 905,418 collisions. The first output failed because charged-only
category closure omitted the newly scored neutron map. This was corrected
by including the independent neutron contribution, without loosening closure
tolerances. Original outputs and binary are preserved.

## Recoil extension (latest)

The research path now accepts `neutron_recoil_stopping_manifest` and
`neutron_elastic_two_body`. The native three-material extension covers the 131
charged ZA species in the neutron event bank and is queried only after
existing tables. Eligible exclusive stationary-target elastic events at
incident and bank energies >=100 keV reconstruct two-body kinematics using
the sampled CM transfer. Thermal and non-exclusive events retain the bank
path. The paired 1M run eliminates missing-ion stopping energy, but still
leaves incomplete descendants and a pooled-valley neutron-dose deficit of
28.89 +/-12.65% against same-birth native transport. See
`evidence/neutron_package_20260927/RECOIL_EXTENSION_zh.md`.


## Direct-depositor maps and optional water recoil Urban (2026-09-27)

The GPU now writes `neutron_proton_MeV`, `neutron_other_charged_MeV` and
`neutron_local_MeV` raw/MHD maps. These disjoint diagnostic maps partition the
existing neutron contribution; do not add them to total dose a second time.
The new YAML `neutron_water_recoil_urban: false` is opt-in and defaults off.
When true, native package-covered water charged descendants reuse the existing
Urban global proposal/finalization and physical water/replica-cell geometry.
Their independent random stream does not consume neutron collision randoms.
Uncovered ZA/energy steps retain CSDA and are counted. Copper/air descendants
remain straight CSDA, and photon/electron transport is still absent.

Four same-birth cohorts locate the clearest pooled-valley discrepancy in
proton deposition: -20.93 +/-4.63 percentage points of TOPAS neutron-lineage
total dose. Water Urban alone does not repair it (-30.21 +/-13.37% total
neutron deficit versus -28.89 +/-12.65% before). This trial does not exclude
uncovered low-energy scattering, copper descendants, production spectra,
straggling or nuclear reactions. See `GPU_COMPONENTS_zh.md` and
`WATER_RECOIL_URBAN_zh.md` in `evidence/neutron_package_20260927`.


## Optional neutron-born proton capture

`neutron_proton_birth_output_file` enables a read-only CSV of protons born
inside the neutron pass. `neutron_proton_birth_capacity` defaults to 2,000,000;
overflow rejects the run. The switch requires neutron transport. Source history
is retained across plan shard aggregation. In this CSV, `rng_stream` identifies
the root neutron, `product_index` is the charged-descendant ordinal within that
root, `collision` is its neutron collision ordinal, and `parent_Z/A=0/1` denotes
the neutron parent. `generation` and `role` are reserved zero. Material sections
-1/-2/-3 denote physical water/copper/air. All rows are protons; do not interpret
this CSV as the original neutron birth source solely from its shared column names.

The capture does not advance any RNG or change dose/transport. Four 250k runs
capture 653,433 protons. Replaying these protons through native TOPAS is a
source/transport diagnostic, not an extra dose correction. Normalize all maps
by original incident protons, not the number of captured records.


## Cartesian boundary repair (2026-09-27)

`neutron_exact_plane_boundaries` now defaults to **true**. At a selected
Cartesian plane, the neutron/CSDA endpoint is snapped to the exact face before
crossing it. Zero-plane nudges use a normal FP32 value so device arithmetic
cannot flush the displacement back to zero. Positive near-face intersections
are no longer discarded by the old 1e-6 mm plane threshold. Optical depth is
still carried across boundaries. Cylinder and rotated-slit intersection
formulas are unchanged; this is not a new general geometry engine.

The old `advance(p,d,t)` plus one `nextafter` could leave cancellation residue
on the incident side of z=0. The subsequent material classification and tiny
plane-distance suppression could let a neutron traverse physical water as air.
The 1M paired study records 135,871 material classifications changed by face
snapping. Water-born protons recover from 621,702 to 756,873 versus native
762,735; entrance 0–20 mm counts recover from 158,155 to 215,562 versus 213,333.
Pooled-valley neutron-dose bias changes from -28.89% to -7.79 +/-15.30%.
Full total-dose uncertainty is still substantial at 1M.

Set the switch false only to reproduce the frozen old geometry behavior.
Water recoil Urban remains optional/default off; the boundary repair does not
retune cross sections, stopping power, MCS amplitudes or dose normalization.
