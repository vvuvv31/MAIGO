# Materials and Methods

Revision: 2026-09-14. Describes the current working tree and named presets, not a frozen release or a claim of clinical/Geant4 equivalence. [中文](mm_zh.md).
Historical numerical results remain bound to their original executable, data and configuration; they do not validate every current option.

## 1. Framework and active configuration

MAIGO is a C++20/SYCL condensed-history charged-ion Monte Carlo, running locally on
an NVIDIA RTX 2080 Ti (`sm_75`). The GPU interpolates TOPAS/Geant4-derived tables
and replays nuclear events rather than executing an intranuclear cascade generator.
The patient implementation is [transport_sycl.cpp](src/transport_sycl.cpp); the serial backend supports a subset.

The reference uses TOPAS 4.2.p3 / Geant4 11.3.2, including standard_opt4 EM,
QGSP_BIC_HP, ion-INCLXX, elastic, stopping and decay modules. A module list alone
does not identify the model attached to every projectile; C12 was observed to use INCLXX.
All-ion elastic comparisons require the matched extension described in Section 5.

The two current unified-EM production entry points are
[water](config/unified_water_production.yaml) and [RT07575](config/rt07575_unified_em_production.yaml):

| Setting | Explicit production-preset value / behavior |
|---|---|
| `em_model` | `g4_material_joint_v1`, one water/Schneider package |
| `enable_secondary_unified_em` | `true`: all 18 supported charged species use unified EM |
| Primary / secondary fluctuations | Both enabled, `straggling_scale: 1.0` |
| `ct_secondary_exact_faces` | `true` |
| `secondary_species_grouping` | `true`; restore original scheduling with `false` |
| Inelastic / secondary transport | Enabled; secondary inelastic generation limit 2 |
| Independent all-ion elastic | Bank omitted in these two production presets; research option |
| Electron dose | Local deposition of sampled delta energy; no electron packet tracking |

`primary_em_model: legacy` in these presets does **not** select legacy stopping:
`em_model: g4_material_joint_v1` selects the unified path. The old primary-only
selector is not an extra model to stack on top. Conversely, `em_model` alone does
not enable unified secondary EM: that switch defaults to false in the configuration
structure. The full-secondary performance experiments explicitly enabled it.
Legacy and research YAML files are not automatically upgraded by updating the executable.
Legacy EM (`em_model: legacy`) is rejected on GPU devices; serial/cpu backends
retain legacy routing (incl. `transport_cpu` and CPU tests). The joint water EM
(`g4_joint_water_v1`) implementation was deleted; its YAML keys are rejected.

Execution of unified EM and species grouping has been authorized. Low-density
production-cut behavior, patient Gamma and the scheduling-dose difference remain
under investigation; execution-quality acceptance is not accuracy acceptance.

## 2. Materials, source and geometry

CT HU maps to actual voxel density and one of 25 Schneider composition sections,
using 13 elemental constituents. CT does not use a four-class material fallback.
Water uses native SHA-pinned G4_WATER, not a fictitious Schneider section.
The unified EM package covers water, all 25 sections and 18 charged species;
its density-node queries preserve material-specific stopping and cut behavior.

TPS spot CSV histories, energy spread, emittance, scanning magnets and patient
placement define the source. Integer spot allocations must match the reference
before sharding. TOPAS passive RotZ uses `R(+RotZ) × (world − Trans)`; vectors
rotate without translation. CT packing, source placement and dose mapping are
checked separately, without dose-fitted registration. See [source geometry](docs/planning.md).

## 3. Electromagnetic transport

### 3.1. One step in the current unified model

1. Read species, kinetic energy, material section and density. Prepare restricted stopping,
   range, ion corrections, fluctuation inputs and delta proposal rate from the unified package.
2. Select the shortest allowed distance from the native EM StepFunction, geometry,
   nuclear candidate and sampled delta candidate. Low-energy stopping/termination
   rules also apply. The native EM limit replaces the legacy maximum-step and
   relative-loss limits on this path.
3. Compute restricted mean energy loss using stopping or range inversion, apply
   ion corrections and sample the enabled restricted-loss fluctuation law.
4. Consume the delta clock along the actual step. At a candidate, perform the
   post-loss rate acceptance test and sample the electron spectrum/form-factor veto.
5. Deduct continuous loss plus accepted delta energy once, score their local dose,
   advance and apply configured MCS, then resolve any nuclear candidate and queue products.

This is the physical dependency order; geometry/scoring and process-specific
branches are interleaved in the kernel. Particle streams retain their history and
process identity; regrouping does not replace Philox or restart its counters.

### 3.2. Restricted mean energy loss and density

Let `T` be total kinetic energy (MeV), `E=T/A` energy per nucleon, and `h` length (mm).
The current package retains native restricted stopping, range and inverse-range
spline segments. Each selected density node is evaluated with the actual/node
density ratio; prepared quantities and mean losses are then interpolated between
bracketing density nodes. This is more than a water stopping curve multiplied by density.

For native StepFunction parameters `f` and `r_final`, with remaining range `R`:

```text
h_EM = f R + r_final (1 − f) (2 − r_final/R),  R > r_final
h_EM = R,                                    otherwise
```

Parameters are read per ion from the package; the primary C12 data use
`f=0.1`, `r_final=0.001 mm`. Do not assign these values to every species without
checking its record. The package also supplies the linear-loss threshold.

For a sufficiently small estimated restricted loss, use `S_restricted(T) × h`.
For larger losses, obtain the outgoing energy by inverse range, retaining the
pre-step mass/charge scaling during inversion. The implemented ion correction
uses an intermediate energy and includes a low-energy replacement. This is not
the old predictor-midpoint total-stopping algorithm. Available kinetic energy
bounds the result; the package's lowest kinetic energy can trigger final stopping.

The older `ct_primary_midpoint_stopping` and material-secondary stopping paths remain
for non-unified transport. Their tables store density-normalized total stopping:

```text
S1(section,ion,E) = S_extracted / [rho_reference/(1 g/cm³)]
S(T,section,rho) = S1(section,ion,T/A) × rho/(1 g/cm³)
```

That density scaling does not double-count density. Those tables and switches do
not determine the unified model's restricted mean loss. Extra unregistered heavy
recoils retain their dedicated stopping path. See [unified lookup/mean loss](include/carbon/unified_em_view.hpp).

### 3.3. Fluctuations and explicit delta proposals

Restricted continuous loss uses the packaged IonFluc or Universal/Urban-related
sampler as applicable. Both primary and supported secondary ions enable native
fluctuations with scale 1.0. The former CT scale 1.2 and secondary-CSDA choices are
historical configurations, not these production presets.

The delta clock samples optical depth `tau = −ln(U)`. Over a segment it consumes
proposal optical depth; locally the candidate distance is `tau_remaining / Sigma_proposal`.
A candidate is not necessarily an accepted electron: the post-loss rate test and
spectrum/form-factor acceptance still apply. Material/density changes invalidate
cached rates while retaining unconsumed optical depth. A mean free path is a
statistical scale, not a fixed collision distance.

```text
DeltaT = sampled restricted continuous loss + accepted delta kinetic energy
T_out  = T_in − DeltaT
```

Restricted stopping excludes above-threshold transfers already treated explicitly.
The delta threshold is material/density dependent; the water example near 57 keV
must not be imposed on every CT section. Delta energy is currently deposited
locally. No proportional electron response is stacked on top of this budget.
`em_macro_ticks` is not a supported current switch; rejected macro-tick and RNG
experiments do not describe the production algorithm.

### 3.4. Coulomb multiple scattering

The configured Highland approximation uses section radiation length and local density:

```text
t = rho × (h/10) / X0_mass
C = max(0, 1 + 0.038 ln(t Z²/beta²))
theta0 = 13.6 MeV × Z/(beta p c) × sqrt(t) × C
```

Deflections are generated in the local transverse frame and rotated into the track
frame, with spatial/directional updates according to the selected transport branch.
The scale defaults to 1.0 and is not tuned in the two production presets.
This is not Geant4's full msc implementation and does not replace hadronic elastic.
See [MCS](include/carbon/multiple_scattering.hpp).

## 4. Inelastic nuclear interactions

**MAIGO: sample the collision, then replay a TOPAS-derived correlated event.**

1. **Locate the collision.** The local macroscopic rate is `Sigma = rho × sum(elemental mass-rate partials)`.
   Primary C12 consumes a sampled optical depth `tau = -ln(U)` along its path;
   the remaining optical depth determines the collision distance. Supported secondary
   ions use their own projectile-dependent rates. A mean free path is a statistical
   scale, not a fixed collision distance.
2. **Select the target and event.** Choose the target element according to its partial
   rate. At the post-EM energy, select one of the two bracketing energy nodes and
   sample a complete CINEL03 event. Retain the correlated product energies and angles;
   rotate the event into the incident frame with one shared random azimuth.
3. **Transport the products.** Replace the incident track with the sampled final state.
   Supported charged fragments enter GPU queues for electromagnetic transport and
   eligible further inelastic reactions. Unsupported channels, energy cutoffs and
   generation limits have explicit accounting; queue overflow invalidates the run.

The current minimum is the pinned Schneider v2.1 stack with a 14-projectile secondary
nuclear registry. Missing channels are not replaced by a nearby target, and product
energies are not globally rescaled. If no valid target remains after EM loss, a null
candidate continues with its remaining energy instead of replaying or depositing it locally.
The frozen generation setting is 2; He6/B8/C10 follow the declared EM-only nuclear policy.

**Difference from TOPAS**

| Aspect | MAIGO | TOPAS / Geant4 reference |
|---|---|---|
| Collision probability | Interpolated extracted elemental rate tables with explicit validity domains | Cross-section datasets and process tracking of the configured physics list |
| Nuclear final state | Samples a finite bank of precomputed correlated events at discrete energy nodes | Invokes the applicable nuclear model at the interaction state to generate products |
| Model execution | No online intranuclear cascade calculation | Reference C12 was observed to invoke INCLXX; the active model depends on projectile and energy |
| Subsequent transport | Supported charged species and bounded nuclear generations; neutral/decay scope is limited | Transports products using the enabled particle processes and tracking cuts |

Reusing TOPAS-derived events preserves the sampled event correlations and avoids online
nuclear-model computation. It does not make the two engines identical: finite event
statistics, energy-node sampling, stepping and secondary coverage remain differences.
The shared event-azimuth rotation was added after the September 5 frozen benchmark.
See [CINEL03 lookup](include/carbon/inelastic_package_v3.hpp) and
[GPU transport](src/transport_sycl.cpp).

## 5. Elastic nuclear interactions

**Available in research configurations for Schneider CT and unified water:** all 18
supported charged species, including primary C12, against all 13 Schneider target
elements. Existing default production configurations still omit this process while
matched-reference validation is in progress. Coulomb MCS remains a separate EM process.

1. **Locate the collision.** Primary C12 consumes sampled nuclear optical depth using
   the sum of elastic and inelastic macroscopic rates, then selects the reaction type
   by their relative rates. Secondary elastic and inelastic collision distances compete;
   elastic remains active beyond the nonelastic generation limit. Geometry, EM loss and
   energy cutoff can shorten the step before either collision.
2. **Sample the target and transfer.** Select the element by its material-specific
   partial rate and the neighboring energy node by rate-weighted interpolation. A joint
   TOPAS sample provides the target isotope, its nuclear mass and `t/tmax`. The current
   energy and a random azimuth determine the relativistic two-body final state. This
   preserves energy and momentum rather than assuming isotropic scattering for every target.
3. **Transport the recoil.** Recoils within the existing 18-species registry enter normal
   charged transport. Additional natural target isotopes use a 37-isotope material-specific
   total-stopping table and MCS, without subsequent explicit nuclear reactions. That total
   stopping includes condensed nuclear stopping. Below-cutoff residual energy deposits locally;
   queue overflow or missing required data rejects the run.

| Aspect | GPU research implementation | Matched TOPAS reference |
|---|---|---|
| Elastic rate and final state | Finite, material-specific rate tables and isotope/transfer samples | Online Geant4 cross-section and final-state model evaluation |
| Models represented | p: hElasticCHIPS; d/t/He3/alpha: hElasticLHEP; other supported ions: NNDiffuseElastic | Same attached elastic models, with `CarbonIonElasticPhysics` enabled |
| Recoil coverage | Existing 18 species plus EM-only additional target recoils | Applicable processes for the generated particles |
| Coulomb scattering | Condensed Highland treatment | Configured Geant4 EM processes |

**Why new TOPAS references are necessary.** The historical `topas10x` physics list had
elastic processes for p, d, t, He3 and alpha, but not GenericIon/C12. Adding
`CarbonIonElasticPhysics` enables the missing ion elastic process. The matched reruns
retain the previous EM, inelastic, stopping and decay modules, patient geometry, source,
3D dose grid and total histories; LET scorers are omitted for this dose validation.

The baseline bank has 137 energy nodes and 512 samples per projectile/target/node.
Tests cover relativistic kinematics, host/device lookup, water closure and a 6,481,909-history
RT07575 shard with zero overflow. Independent 2048-sample and denser-grid banks also pass
shard closure, but low-energy cross-section onset interpolation, additional recoil
contributions and full-statistics matched dose validation remain open. These checks do
not yet establish production acceptance.

Use [CT research configuration](config/rt07575_elastic_research.yaml) or
[water research configuration](config/unified_water_elastic_research.yaml); the legacy
`enable_nuclear_elastic` switch is not the activation method for this path. New elastic
packages are not included in Release 11.3.2. See [implementation and data](docs/all_ion_elastic.md),
[validation evidence](evidence/step-31/elastic-production-validation-20260911/README.md) and
[September 12 reference migration](evidence/step-31/elastic-migration-20260912/README.md).

## 6. Electrons and neutral products

The active unified EM model samples delta transfers but does not track those electrons
spatially. Restricted and delta losses contribute local dose; local deposition itself
is an approximation, particularly for interfaces, lateral tails and minibeam valleys.
There is no active electron packet kernel to disable for another large speed gain.

The repository also contains old section-0 delta-tail redistribution and material
electron-family/packet replay candidates. Packets move an already budgeted energy
weight through recorded states and continuations; they are not a second energy loss
charged to the ion. They are not enabled by the unified production presets and
cannot simply be stacked on the restricted-plus-delta model. Material-response
accuracy remains unvalidated. See [packet transport](include/carbon/electron_packet_transport.hpp)
and [electron plans](plan2/README.md).

Neutrons, photons and decay products do not all have a complete production transport
chain. Unsupported energy, escape and compatibility sinks are recorded explicitly;
small integrated energy does not prove a negligible local halo/valley contribution.
Do not describe the current model as complete Geant4 transport of all secondaries.

## 7. Scoring and dose comparison

```text
Dose(Gy) = Edep(MeV) × 1.602176634e−13 / voxel_mass(kg)
voxel_mass(kg) = rho(g/cm³) × volume(mm³) × 1e−6
```

Use cumulative 3D DoseToMedium. Obtain IDD by lateral summation of the 3D scorer;
for heterogeneous voxels convert dose back to deposited energy with voxel mass
before forming an energy-deposition IDD. A bare sum of Gy is not an energy sum.
Lateral profiles and core/halo widths are separate observables. LET remains a
separate option; the present dose performance comparisons disable LET.

Current CT Gamma evaluation:

- Build the BODY mask from RTSTRUCT and evaluate only reference voxel centers inside it.
  Record the ROI identity, rasterization/interpolation method and mask hash.
- With the current 10% dose cutoff, evaluate `BODY ∩ {Dref >= 0.1 Dmax}`;
  `Dmax` is the full-volume reference maximum. BODY membership and dose threshold are distinct.
- Global tolerance uses `Dmax`; local tolerance uses the reference dose at the query voxel.
  Compare 3%/3 mm, 2%/2 mm, 1%/1 mm and 3%/0 mm at matched histories and geometry.
- For positive DTA, use search spacing `DTA/10`: 0.3, 0.2 and 0.1 mm respectively,
  with trilinear interpolation. BODY selects reference query points; do not zero
  the evaluated dose outside BODY. A discrete search is not an analytic continuous minimum.
- Zero DTA compares the same voxel and has no search-step dependence.

No fitted dose normalization or registration is used. The older September 5
whole-volume/no-BODY/0.5-mm results remain historical; some old evaluator scripts
still implement that convention and must not be used as the current protocol unchanged.
A current implementation example is [BODY evaluation](benchmark/benchmark20260912/RT06423_replica01/evaluate.py).

Quality checks cover finite values, provenance, sampling audits, energy accounting
and zero queue overflow. An overflow invalidates the shard: split and rerun.
Global energy closure does not establish spatial-dose accuracy or close every nuclear Q value.

## 8. Species grouping and measured throughput

`secondary_species_grouping: true` creates a GPU index permutation for each secondary
generation: 18 species buckets plus other products. Histogram, prefix sum and scatter
leave particle records, parent histories and RNG streams intact. Descendants are
processed in the next generation. The implementation does not include the extra
split kernels. Additional index memory is approximately 4 bytes per queue slot.

The two production YAML files explicitly opt in; the configuration-structure default
is false for compatibility. Logs print the mode and grouping time. The quality report
retains `secondary_species_grouping_accuracy_pending` after the authorized integration.

RT07575, 1M histories, full secondary unified EM **and research elastic enabled**, two runs:

| Scheduling | Mean histories/s | Relative gain |
|---|---:|---:|
| Original | 11,154 | — |
| Species grouping | 14,691 | 31.7% |
| Grouping + isolated split | 14,828 | 32.9%; split not integrated |

This denominator includes grouping overhead. These numbers are not measurements of
the two production presets with elastic omitted. Grouping changed maximum voxel dose
by 0.0203% of peak; original-schedule repeats differed by 0.0047%. Counts and EM audits
matched, but the origin of the larger difference remains unresolved. Authorization
to integrate is not a passed dose-equivalence gate.

Production-entry checks used RT07575 200k and water 50k, with grouping on/off,
full secondary EM and no independent elastic. All four runs passed execution quality,
zero overflow and matching step/audit/reaction counts; maximum dose differences were
0.0000133% and 0.000172% of peak. These do not resolve the elastic-enabled 1M discrepancy.
See [study and integration records](benchmark/benchmark20260914/secondary_schedule/README.md).

Report kernel time, program transport elapsed and full process wall time separately.
Loading, preprocessing, transfers, finalization and output affect the last two;
do not label their difference as a measured individual physics-process cost.

## 9. Approximation options for substantial acceleration — proposals only

No approximation below was enabled by this documentation update. The target is
reduced computational work with a declared error budget; 50k histories/s has not been achieved.
After grouping, the 1M elastic study still spent about 37.7 s in primary kernels and
26.2 s in secondary kernels. A 50k/s target allows 20 s total. Eliminating secondary
work alone cannot reach it: primary transport must also become substantially cheaper.

| Candidate | Work removed | Potential and main limitation |
|---|---|---|
| Joint EM block propagator | Many continuous-loss, delta-clock and scattering microsteps | Highest broad potential; difficult joint-distribution and geometry validation |
| Short-range recoil/fragment terminal kernel | Many low-energy steps ending inside one voxel | More localized scope; contribution to runtime must be measured first |
| Secondary weighted roulette | Transport only a sampled subset, increase surviving weights | Higher raw histories/s may come with proportionally worse variance |
| Secondary CSDA/mean-only fast mode | Secondary delta clocks and stochastic loss detail | Explicitly biased alternative model; cannot count as full unified EM |
| Less frequent MCS preparation/updates | Repeated angular/displacement work | Separate scattering length changes finite-step physics; unlikely alone to supply a multi-fold gain |

### 9.1. Primary research direction: a joint EM transition, not delta-only batching

Build a conditional propagator from the current microstep model for
`(species, material, density, energy, block length)`. It must represent **together**
restricted loss, delta loss, outgoing energy, angular/displacement changes and
intrablock dose placement. The nonlinear change of stopping and rate with energy
couples these quantities; matching delta mean and variance alone is insufficient.

A block cannot jump past the first competing nuclear collision or a material boundary.
Energy-dependent integrated nuclear hazard must be consistent with the sampled
energy trajectory; evaluating the nuclear rate only at the endpoint is not equivalent.
An endpoint-only spatial check is insufficient for a path that can leave and re-enter
its voxel. Begin with short blocks in homogeneous regions, away from range end and
interfaces, with an explicit fallback/transition rule that does not reject and resample
boundary-crossing outcomes into a biased contained-path distribution.

Near the Bragg peak, low-density production-cut onset and interfaces, retain the
original stepping initially. Sample total loss and intrablock deposition consistently;
placing all block energy at its start/end would add a separate spatial approximation.
A small interpolated joint distribution or reduced conditional model may remove
repeated work, but its table size, lookup cost and fallback frequency can erase the gain.
No speed factor is established for this proposal.

This differs from the rejected compound-Poisson/delta-quantile candidates:
those retained or approximated only parts of the joint propagation while changing
continuous-fluctuation and MCS step lengths. Re-running them unchanged is not proposed.

### 9.2. Lower-scope candidate: short-range termination

For additional EM-only heavy recoils, estimate whether the entire remaining
transport is negligible relative to distance to each face and the dose-gradient scale.
Replace many steps by a terminal deposition distribution or, in a declared tighter
limit, local deposition. A CSDA range is a mean estimate, not a hard upper bound:
range tails and material escape need a bound/error budget. Preserve residual energy
exactly; do not apply this rule indiscriminately to all low-energy protons or alphas.
For particles that can still undergo nuclear reactions, quantify the omitted reaction
probability before any shortcut. Minibeam use requires a bound relative to beam/valley
width, not merely a coarse voxel size. Measure step/time share by species and range
before implementation; no existing breakdown proves this alone has a large gain.

### 9.3. Weighted sampling and deliberately simplified secondary EM

With roulette survival probability `p`, surviving track weight becomes `w/p`.
The expectation is preserved only if every descendant, dose/LET tally, escape and
energy ledger propagates the weight correctly. Current transport has unit-weight
queue/scoring paths; this is not a drop-in configuration switch. Realized per-history
energy closure and estimator bookkeeping need redesign; ordinary overflow/loss
checks must remain separate from statistical roulette fluctuations.

Judge roulette by time to fixed uncertainty (for example `1/(time × variance)`),
not raw primary histories/s. Correlated loss of rare fragments can worsen local Gamma,
halo or valley precision. A mean-only secondary model instead introduces bias:
it must be labelled a separate approximate mode, with no claim of full-unified accuracy.
A blanket removal of secondary fluctuations, nuclear elastic, or neutral dose is
particularly unsuitable as a minibeam/valley default.

### 9.4. What not to repeat, and how to accept a new approximation

- RNG dummy gains do not establish faster physical sampling. Reverted buffering and
  macro-tick experiments are not production options; secOFF/K speed figures cannot
  serve as full-unified baselines.
- The per-electron Poisson batching experiment was slower. The delta-only quantile
  candidate gained about 10.9% in one RT07575 run but failed peak/R80 screens;
  it was not a successful precision-preserving replacement. See
  [batching](docs/delta_batch_sampling.md) and [quantile results](docs/rt07575_quantile_optimization.md).
- Current electron deposition is already local and nuclear final states already use
  table replay. Disabling a nonexistent full-electron track or replacing an online
  nuclear cascade cannot supply additional savings in this implementation.

First measure how much time and how many steps the eligible region/species consumes.
If fraction `f` of runtime can be accelerated by `s`, total speedup is at most
`1 / [(1−f) + f/s]`, before new lookup/scheduling overhead. Then validate conditional
loss distributions and their correlations, range/end-state distributions, and energy accounting.
Use unchanged default TOPAS for 100/200/300 MeV/u, b3/b4 interfaces and RT07575;
350/400 MeV/u are supplementary. Compare IDD peak/R80, core/halo widths, lateral
profiles and BODY local/global Gamma at matched histories and across independent seeds.
For approximations, identical event counts are not generally expected; statistical
agreement and dose bias replace scheduling-only bitwise/count equivalence tests.

Provisional continuation gates may reuse the earlier peak-error increase ≤0.3 percentage
points and R80 displacement ≤0.1 mm screens; they are screening criteria, not current
clinical acceptance. Prespecify allowed Gamma degradation and uncertainty before testing.
Minibeam extension additionally needs valley dose/PVDR and spatial-tail validation.
Do not tune the reference TOPAS to match the approximation.

## 10. Reproduction and data

Build C++20/SYCL for local `nvptx64-nvidia-cuda`, `sm_75`. Freeze the executable,
resolved YAML, data hashes, CT/source transforms, histories/spot allocation, seed,
scoring and shard manifest. Large physics binaries and external CT/source inputs
are required in addition to a clone; never infer current data coverage from an older release tag.

Before Schneider CT runs execute `python3 tools/verify_schneider_v2_1_data.py`.
The exact v2.1 nuclear/stopping bundle remains the minimum; no water/four-class or
older-schema fallback is allowed. Unified EM uses `data/em/unified_em_v1.bin`, SHA256
`8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`.
New propagator/recoil approximation tables would be new candidate data, not an
automatic extension of that package's authorization or validation.

GPU jobs stay local. If no TOPAS environment is specified, use local `sbatch` and
`/mnt/sda/wuwei`; an explicitly named remote CPU host/cluster is allowed under the
repository rules. Aggregate limits are 192 CPU threads and 160 GB RAM. Split runs
when needed and rerun every overflowed shard; merge only accepted shards.

This revision updates documentation and proposals; it adds no new physics validation
or transport approximation. Current production-integration checks are recorded in Section 8.

## References

- [FRED carbon model analysis](docs/FRED_Carbon_Fragmentation_Model.md)
- [Unified EM data/model](docs/physics/unified_em_v1.md)
- [Older primary-water model, removed from code](docs/physics/water_joint_em_v1.md)
- [Historical physics specification](docs/TOPAS_GPU_Physics_Model.md)
- [Scoring contract](docs/scoring_validation.md) and [archived drafts](docs/archive/README.md)
