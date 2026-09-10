# Materials and Methods

Revision: 2026-09-10. Current research draft, not a clinical or general Geant4-equivalence claim.
[中文](mm_zh.md). The [physics specification](docs/TOPAS_GPU_Physics_Model.md) describes the September 5 baseline;
subsequent implementation changes are identified below and checked against source code.
Original drafts, including exploratory proton/multi-ion material,
are preserved in the [archive](docs/archive/README.md), not promoted to validated CT capabilities.

This revision reviews the source/build graph, configuration and data loaders, TOPAS
extraction tools, tests, benchmark artifacts and research plans through source checkpoint
`f6245c5` (2026-09-09). Existing uncommitted transport changes are not a frozen release.
Unless explicitly labelled otherwise, numerical CT results and benchmark settings below
refer to the Git-tracked September 5 dataset, not to every current code path.

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

Current homogeneous-water runs share the CINEL03 nuclear transport framework with CT,
using native, SHA-pinned G4_WATER composition and H/O elemental rates reconstructed
from the material partial-rate tables. Water is not assigned a fictitious Schneider section.
Its stopping, density, radiation length and fluctuation inputs remain material-specific;
CT tables do not silently replace them. Legacy CINEL02 event/rate configuration keys
are rejected by the unified-water route. Allowing water `run_mode: production` does not
establish TOPAS accuracy: run quality retains an explicit incomplete-validation note.
See the [unification record](plan2/water_ct_unification.md) and
[material rate interface](include/carbon/material_nuclear_rates.hpp).

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

Since September 7, primary and secondary CINEL03 replay also sample one common
uniform azimuth about the incident axis per event (Philox dimension 60). The same
rotation acts on every product, preserving relative angles and correlated kinematics;
independent per-product azimuths would destroy those correlations. This code change
postdates the September 5 executable. See [rotation helper](include/carbon/cinel03_event_rotation.hpp).

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

The September 5 three-case executable includes the entrance-mask candidate. Later
longitudinal, joint and ordered full-family response implementations are separate
experiments; that benchmark does not validate them. Ordered replay preserves the
recorded path and ancestry rather than replacing a trajectory by its endpoint chord.

A further material-response candidate loads SHA-pinned birth distributions, raw electron
segments and continuation indices for water or Schneider CT. It transports an energy-valued
statistical packet W, distinct from the sampled electron kinetic energy. Each packet has
one terminal energy owner: deposit, patient/scorer escape or explicitly uncovered energy.
Finite source exhaustion requests continuation; it is not patient escape. Missing photon
continuation is reported separately and is not deposited locally. Same-section density
interpolation mixes measured conditional distributions; full density and interface accuracy
remain unvalidated. Electron birth material and boundary traversal must use consistent
voxel ownership. The runtime supports device-resident or host-mapped response banks
with explicit memory budgets.

The material candidate is mutually exclusive with the older delta-tail, joint and explicit
electron modes and requires 3D scoring with LET disabled. Run quality still reports
`unvalidated_material_electron_response`; closure alone cannot remove this failure.
See [packet transport](include/carbon/electron_packet_transport.hpp),
[density sampling](include/carbon/material_electron_density.hpp) and
[quality gates](src/run_quality.cpp). General density/geometry/birth conditioning and
promotion gates remain under [plan2](plan2/README.md).

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

## 8. Later evaluation and performance work

The local, untracked report `benchmark/topas10x/gpu_current_20260909.md` records a later
60-shard three-case run, refined Gamma and an experimental RT07575 electron full20 A/B.
These local artifacts are not supplied by this two-file documentation update and are not
a replacement for the published September 5 evidence. Their experimental electron
results do not promote the minimum physics stack; reproduction requires the corresponding
executable, configuration, raw-dose hashes and quality reports.

The later `tools/evaluate_gamma_adaptive.py` tool (source checkpoint `f6245c5`) first
reproduces the frozen 0.5 mm pass mask, then searches only failures on a 0.25 mm lattice
using the same trilinear interpolant. Coarse passes are retained, and 3%/0mm is unchanged.
It checks dose hashes/history counts and writes a separate `gamma_refined.json` without
overwriting the frozen result. Refined and coarse rates must be labelled separately:
search refinement changes numerical evaluation, not transported dose. Neither finite
lattice is an exact continuous minimum. The tool is a later local commit, not part of
this documentation-only publication.

Performance reporting separates primary-plus-secondary GPU kernel time, the program's
transport elapsed time, and complete process wall time including response loading and I/O.
Histories/s must identify its denominator, statistics, chunk size, scorer settings and
hardware; a small smoke test cannot establish full-plan acceleration against TOPAS.

The [tracked 64-history short-range comparison](benchmark/benchmark20260909/short_range_64_comparison/README.md)
uses patient 20022516 and thresholds 0, 0.1 and 0.25 mm. Raw dose is bitwise equal,
overflow is zero, but all runs remain rejected for unvalidated material response.
Primary kernel times are 61.2625, 61.5884 and 61.5946 s: no speedup was demonstrated.
The shortcut requires a complete childless terminal tail contained in the current voxel;
the default threshold is zero. The later cached-tail implementation remains a candidate,
not an accepted performance result.

## 9. Reproduction and validation procedure

Build settings are specified by [CMake](CMakeLists.txt) and [presets](CMakePresets.json):
C++20, SYCL enabled, NVIDIA target `nvptx64-nvidia-cuda`, local `sm_75`.
Freeze executable SHA, parsed configuration, source/CT transforms, all data pins,
spot allocations, seeds, scorer options and shard manifests. Run
`python3 tools/verify_schneider_v2_1_data.py` before every Schneider CT campaign;
missing files, hash/size/schema mismatch or incomplete 14-projectile coverage are hard failures.
Large binaries and external raw responses are not guaranteed to be present in a fresh clone.

GPU work runs only on the local RTX 2080 Ti. TOPAS extraction uses local `sbatch`,
raw data under `/mnt/sda/wuwei`, and source/build/extensions under `/home/wuwei/topas`.
All jobs together are limited to 192 CPU threads and 160 GB memory, allocated by workload.
Split large GPU history sets; any secondary overflow requires smaller shards and reruns.
Merge only individually acceptable shards and verify aggregate energy and history accounting.

Data promotion requires explicit TOPAS/Geant4 provenance, manifest and host/device lookup
checks, 50k closure gates, paired one-shard Gamma, and zero-overflow full validation.
Independent phantom/domain/geometry tests are also required for electron candidates.
Historical test fixtures and old documentation are not evidence that current tests passed;
this methods update itself runs no new transport or physical-accuracy validation.

## References and traceability

- [FRED paper interpretation / source pointers](docs/FRED_Carbon_Fragmentation_Model.md)
- [Physics specification](docs/TOPAS_GPU_Physics_Model.md) and [code map](docs/structure.md)
- [Frozen benchmark and provenance](benchmark/topas10x/gpu_current_20260905.md)
