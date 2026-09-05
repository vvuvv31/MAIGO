# Materials and Methods

Revision: 2026-09-05. Current research draft, not a clinical or general Geant4-equivalence claim.
[中文](mm_zh.md). The [physics specification](docs/TOPAS_GPU_Physics_Model.md) is authoritative
for implementation details. Original drafts, including exploratory proton/multi-ion material,
are preserved in the [archive](docs/archive/README.md), not promoted to validated CT capabilities.

## 1. Computational framework and reference

MAIGO is a C++20/SYCL condensed-history charged-ion Monte Carlo.
The patient backend is [transport_sycl.cpp](src/transport_sycl.cpp); the serial backend is a subset.
This benchmark uses a local NVIDIA RTX 2080 Ti (sm_75).
The GPU interpolates transport tables and replays correlated nuclear final states without
running Geant4 or an intranuclear cascade generator.

Reference: TOPAS 4.2.p3 / Geant4 11.3.2 with g4em-standard_opt4,
g4h-phy_QGSP_BIC_HP, g4decay, g4ion-inclxx, g4h-elastic_HP, g4stopping,
g4radioactivedecay. Module names do not identify the active model for every projectile.
Reference C12 was observed to use INCLXX; BIC in the list does not establish a C12
BIC-versus-INCLXX discrepancy.

Counter-based streams index histories and process channels. Reproducibility requires
the frozen executable, inputs, seeds and allocations, not an assumption of bitwise
equivalence across devices/builds.

## 2. Materials, source and geometry

HU maps to continuous density and one of 25 Schneider sections with 13-element compositions.
Composition is a shared LUT, not copied into each voxel. Probe-material representative
densities must not replace actual CT densities. CCTG/DICOM and scorer geometry are checked together.

Current source generation uses TPS spot CSV histories, beam-model emittance, energy spread,
virtual scanning magnets and explicit TOPAS placement. Each replica's integer spot histories
are matched before GPU sharding. TOPAS passive RotZ gives component world-to-patient coordinates
R(+RotZ) × (world − Trans); vectors rotate without translation.
Source, placement and CT packing are separate operations. Dose is restored to patient coordinates
using the frozen mapping, without dose-fitted registration. See [planning](docs/planning.md).

Water remains a separate route; CT H/O event channels do not replace all water stopping,
cross-section, MCS or fluctuation inputs.

## 3. Electromagnetic transport

Steps are bounded by configured maximum length, relative energy loss, geometry, nuclear
optical depth and termination. Frozen three-case limits are 0.5 mm and 0.005.
Primary C12 uses Schneider mass stopping/range with local density.
Secondaries use water-ion stopping with CT material factors and local density; this is
not a complete directly extracted isotope-by-section stopping dataset.
Upstream air energy loss is separate.

The benchmark uses Highland MCS with section X0 and local density for both primary and
secondary tracks, not the exact Geant4 msc algorithm.
Primary straggling is enabled with the existing frozen scale 1.2;
secondary straggling is disabled by default. These are reproducibility settings,
not permission to fit physics parameters to dose.

## 4. Nuclear replay and secondary scope

Minimum stack: Schneider v2.1 SCHNRATE/SCHN2RAT v3 rates, CINPKG04 v4 primary and
14-projectile secondary packages, SCHNSTOP v1. Hashes are pinned by [AGENTS](AGENTS.md)
and the [manifest](data/schneider/v2_1_data_manifest.json).

Hazard is local density times valid elemental partial-rate sum. Per-channel domains
mask invalid partials at query time. Lookup requires exact projectile Z/A and target Z.
For a valid bracket, P(upper)=(Eq−E0)/(E1−E0); gaps >5 MeV/u are rejected except
exact-node hits. A correlated event is sampled within the node, retaining its energies;
directions rotate into the incident frame. There is no closest-target alias or global
product-KE rescaling.

Post-EM null candidates retain energy and continue without replay/local dumping.
Nulls and failures are reported separately. Finite-step energy skew and masked domains
remain declared approximations, not proof of unrestricted physical coverage.

Charged products enter queues for EM and supported nuclear transport.
The frozen generation setting is 2, not unlimited cascade.
He6/B8/C10 have the declared out-of-scope EM-only nuclear policy; Be6 handling is separate.
Overflow invalidates a run. Independent nuclear elastic is disabled in this benchmark
and is not implicitly included in inelastic events. General neutral/decay transport is not validated.

## 5. Electron response

The strict-dose stack includes a TOPAS-derived transverse redistribution of part of primary
C12 loss in section 0, extracted at 150/200/225 MeV/u.
It relocates already-accounted loss, not additional energy. Source eligibility, destination
material and scorer escape constrain its scope; it is not general electron transport.

The three-case executable includes the entrance-mask candidate, not validation of the
separate longitudinal candidate currently in the worktree. General density/geometry/birth
conditioning and joint-response gates remain under [plan2](plan2/README.md).

## 6. Scoring and quality

Dose(Gy) = Edep(MeV) × 1.602176634e−13 / mass(kg), with
mass = density(g/cm3) × volume(mm3) × 1e−6.
Output is cumulative 3D DoseToMedium. IDD is derived by transverse summation, not a 1D scorer;
isocenter profiles are interpolated lines, not IDD.

Optional LET_d = sum(L × dE)/sum(dE); shards merge both moments before division.
LET is disabled in this three-case dose benchmark, which supplies no new LET validation.
Origin-dose outputs are diagnostics, not independent TOPAS species references.

Checks include finite outputs, exact particle/candidate accounting, zero overflow,
data provenance and energy closure. Explicit sinks check deposited = in-grid + outside-grid
and voxel sum = in-grid with implemented relative tolerance 1e−3.
Global closure is not exact nuclear mass/Q closure at every vertex.
See [scoring contract](docs/scoring_validation.md) for fields and compatibility conditions.

## 7. Evaluation and limitations

The 2026-09-05 dataset has 60 accepted shards, 457,898,870 histories and zero overflow.
Numerical results reside in the [current index](docs/results.md) and frozen artifacts.

Evaluation uses nominal cumulative Gy without LS scaling or fitted registration.
The mask includes every reference voxel ≥10% of whole-volume Dmax, with no BODY mask.
Global tolerances use Dmax; local tolerances use the reference voxel dose.
Criteria: 3%/3mm, 2%/2mm, 1%/1mm, 3%/0mm.
Nonzero distances use a 0.5 mm spherical lattice and trilinear interpolation,
not an analytic continuous minimum; 3%/0mm compares the same voxel.

Results apply only to frozen inputs/executable. Strict local and low-density/interface
residuals remain; neither a unique electron cause nor full physics equivalence is established.
This benchmark does not close plan2, upgrade packages or establish clinical readiness.

## References and traceability

- [FRED paper interpretation / source pointers](docs/FRED_Carbon_Fragmentation_Model.md)
- [Physics specification](docs/TOPAS_GPU_Physics_Model.md) and [code map](docs/structure.md)
- [Frozen benchmark and provenance](benchmark/topas10x/gpu_current_20260905.md)
