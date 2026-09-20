# Materials and Methods

Revision: 2026-09-20. Describes the current working tree and named presets, not a frozen release or a claim of clinical/Geant4 equivalence. [中文](mm_zh.md).
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
Package extraction sources, compilers and frozen hashes live in
[extensions/](extensions/README.md).

Two configuration families are active:

1. **Broad-beam / CT production.** Unified-EM production entry points are
   [water](config/unified_water_production.yaml) and
   [RT07575](config/rt07575_unified_em_production.yaml).
2. **Copper minibeam validation.** Field configurations such as
   [250 MeV/u, 256k](config/beam_minibeam_field3cm_copper_e250_256k.yaml)
   (and the 150/300 MeV/u siblings) enable Copper collimator transport plus
   water transport. They are research validation presets, not a clinical TPS.

| Setting | Water / RT07575 production | Copper minibeam field presets |
|---|---|---|
| `em_model` | `g4_material_joint_v1` | `g4_material_joint_v1` |
| `enable_secondary_unified_em` | `true` | omitted; defaults **false** |
| Primary / secondary fluctuations | Both enabled, `straggling_scale: 1.0` | Primary enabled; secondary straggling omitted, defaults false |
| `multiple_scattering_model` | explicit `highland` (code default) | `fermi_eyges`, `fermi_eyges_species: all_charged` |
| `ct_secondary_exact_faces` | `true` (CT) | not a CT grid |
| `secondary_species_grouping` | `true` | not the minibeam performance target |
| `secondary_step_chunking` | `true`: 16 complete iterations, then survivor compaction | as configured |
| EM table search | `CARBON_EM_EXACT_INDEX=ON` | same build |
| Inelastic / secondary transport | Enabled; generation limit 2 | Water generation limit 2; Copper cascade generations 3 |
| Independent all-ion elastic | Omitted in production YAML | Copper discrete elastic off |
| Electron dose | Local deposition of sampled delta energy; no electron packet tracking | Same |
| Minibeam-only loss scale | not applied | `minibeam_water_primary_stopping_power_scale: 0.9958` on primary C12 only |

`primary_em_model: legacy` in the production presets does **not** select legacy stopping:
`em_model: g4_material_joint_v1` selects the unified path. The old primary-only
selector is not an extra model to stack on top. Conversely, `em_model` alone does
not enable unified secondary EM: that switch defaults to false in the configuration
structure. Legacy and research YAML files are not automatically upgraded by updating the executable.
Legacy EM (`em_model: legacy`) is rejected on GPU devices; serial/cpu backends
retain legacy routing (incl. `transport_cpu` and CPU tests). The joint water EM
(`g4_joint_water_v1`) implementation was deleted; its YAML keys are rejected.

Execution of unified EM and species grouping has been authorized for the CT/water
production path. Low-density production-cut behavior, patient Gamma and the
scheduling-dose difference remain under investigation; execution-quality acceptance
is not accuracy acceptance. Minibeam field matching is a separate, unfinished
validation (Section 11).

Dose scoring for GPU performance and minibeam development is FP32
(`CARBON_DOSE_FP32=ON`). FP32 atomic accumulation-order differences are expected
and are not, by themselves, a reason to reject a candidate.

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

Minibeam field geometry is a 15-slit Copper collimator (slit width 0.5 mm, pitch
3.6 mm, thickness 60 mm) upstream of a water phantom. The GPU transports
primaries through air and Copper, then water; absorbing-geometry replay can start
just inside water. Copper density 8.96 g/cm³ and mass radiation length
12.8628 g/cm² are explicit. Copper stopping, inelastic rates and per-energy
fragment-cascade packages are extracted TOPAS/Geant4 11.3.2 tables, not Schneider
CT sections.

## 3. Electromagnetic transport

### 3.1. One step in the current unified model

Following user acceptance on 2026-09-15, water and all 18 supported Schneider ions use two-moment Gamma delta aggregation with an analytic partition correction.

1. Prepare species/material/density-specific restricted stopping/range, ion corrections, native fluctuations and delta moments.
2. Choose the current step length as in Section 3.1.1.
3. Correct the restricted linear-loss branch for Poisson partitioning as below; do not apply it again to range inversion. Retain native restricted fluctuations at scale 1.
4. Sample one Gamma aggregate for the step's delta loss, debit energy and deposit locally. No individual electron collision or delta-clock step limit is sampled.
5. Advance, apply configured MCS, resolve nuclear candidates and queue products. Nuclear optical depth remains a separate mechanism.

Philox streams retain particle identity and continuation counters. The 1% guard bounds estimated mean loss, not the sampled loss.

On the minibeam primary-C12 water path only, the sampled total loss is then multiplied
by the frozen scale 0.9958 and clipped to available kinetic energy. The same draw's
`continuous` and `delta` parts are scaled by `deposited/unscaled` so the partition is
preserved. This scale is not applied to secondary C12, fragments, Copper, or the
broad-beam production presets. A 250 MeV/u full-chain A/B against the current TOPAS
reference favoured 0.9958 over 1.0 on 2-D/IDD L1 and Bragg depth; it is a table/sampling
correction relative to that reference, not a Geant4 stopping-table rewrite.

#### 3.1.1. How the current step length is chosen

The GPU first proposes an electromagnetic length, then shortens it for geometry
and a possible nuclear collision. Energy loss and MCS use that final `h`.
Implementation: `step_from_range` in
[unified_em_view.hpp](include/carbon/unified_em_view.hpp) and the primary/secondary
loops in [transport_sycl.cpp](src/transport_sycl.cpp).

**Unified EM (water, RT07575 and current minibeam field YAML).** The YAML keys
`maximum_step_mm` and `maximum_relative_energy_loss` are **replaced**, not
combined with a min, by Geant4's native StepFunction. Remaining restricted
CSDA range \(R\) comes from the package spline at the step-start energy and
local density. Parameters are `f = dRoverRange` and
`r_final = finalRange`, stored per ion as `step_fraction` and `final_range`:

```text
h_EM = f R + r_final (1 − f) (2 − r_final/R),  R > r_final
h_EM = R,                                    otherwise
```

`f = 0.1` for every ion in the v1 package. `r_final` is species-dependent,
read from `G4VEnergyLossProcess::finalRange` at extraction:

| ion | `r_final` |
|---|---|
| p | 0.05 mm |
| d, t, He-3, He-4 | 0.02 mm |
| He-6 and \(Z\ge 3\), including C12 | 0.001 mm |

Far from stopping, \(h_{\mathrm{EM}}\approx fR\) (about 10% of remaining range
for C12, plus a ~0.002 mm offset). At \(R=r_{\mathrm{final}}\) the two branches
match in value and first derivative (\(h=R\), \(h'=1\)), so the stepper does not
jump. For \(R\le r_{\mathrm{final}}\) the proposal is the remaining range.
This is a geometric EM proposal, not a fixed millimetre step and not the YAML
relative-loss cap. High-energy C12 steps can be many millimetres; near stopping
they become micrometres.

If delta stopping is positive, also require

```text
h ≤ 0.01 T / (S0 + D0)
```

This bounds the **mean** combined restricted-plus-delta loss, not the sampled
random loss. It is a step-length cap, not the `linear_limit` that later chooses
the continuous-mean branch (Section 3.2.1).

**Legacy EM only.** The proposal is
`min(maximum_step_mm, maximum_relative_energy_loss × T / S)`. GPU production
rejects `em_model: legacy`; serial/CPU backends still have that path.

**Then shorten `h` by the minimum of:**

- depth-bin faces of the 1-D scorer (present in water and minibeam; 0.25 mm
  bins in current minibeam YAML);
- exact CT voxel faces when the track is in the CT grid;
- slab, insert, phantom and Copper material boundaries when those geometries
  are active;
- a nuclear collision inside this candidate step: remaining optical depth
  \(\tau=-\ln U\) is consumed over `h`; if \(\Sigma h \ge \tau\), `h` is cut
  to \(\tau/\Sigma\) and a collision is flagged. Otherwise \(\tau\) is reduced
  and transport continues. The collision distance is not the only step limit
  computed in advance.

Lateral scoring voxels do **not** clamp transport in production
(`voxel_scorer_clamps_transport` is off). Fermi–Eyges internal 0.1 mm segments
split MCS inside an already chosen `h`; they do not replace StepFunction.

**Transport stops** when kinetic energy falls to `energy_cutoff_MeV`, the track
leaves the phantom, or the primary/secondary step cap is reached. Residual
energy at cutoff is scored in the current voxel.

### 3.2. Restricted mean energy loss and density

Let `T` be total kinetic energy (MeV), not MeV/u; `E=T/A` is energy per nucleon;
`h` is the already chosen step from Section 3.1.1 (mm). Continuous loss is the
restricted ionization on that step, not a separate step-length formula.

#### 3.2.1. How primary and secondary continuous loss is formed

After `h` is fixed, ionization is split at the material electron production cut:

- energy transfers **below** the cut → restricted / **continuous** (`draw.continuous`);
- energy transfers **above** the cut → aggregate **delta** (`draw.delta`, Section 3.3).

The two are sampled separately and added. Continuous loss is not
\(S_{\mathrm{total}} h\), and it is not the YAML `maximum_relative_energy_loss`
limiter from Section 3.1.1.

**Unified EM** (`em_model: g4_material_joint_v1`) uses one restricted-mean then
fluctuation sequence for a primary ion and for any secondary that has Unified EM
enabled. The package stores native restricted stopping, range and inverse-range
splines. Adjacent density nodes are evaluated at (actual density)/(node density)
and interpolated; this is not a water stopping curve multiplied by
\(\rho/(1\,\mathrm{g/cm^3})\). The package also supplies Geant4's native
`linLossLimit`, stored per ion as `linear_limit`. Read from `unified_em_v1.bin`
(3,150 material/ion records; 175 density nodes share one value per ion):

| ion | `linear_limit` |
|---|---|
| p, d, t (\(Z=1\)) | 0.01 |
| He-3 and heavier, including C12 | 0.02 |

This is **not** millimetre length. "Small" / "large" means whether the estimated
restricted loss \(S_0 h\) is a small fraction of the current total kinetic energy
\(T\). It is also **not** the 1% step guard \(h\le 0.01\,T/(S_0+D_0)\) in
Section 3.1.1 (that cap uses restricted **plus** delta stopping and limits \(h\)
before this branch). For C12 the linear branch is \(S_0 h \le 0.02\,T\). The
`0.02` default in `primary_restricted_mean_candidate` is unused on the unified
path; the call passes `r.linear_limit`.

1. Look up restricted stopping \(S_0\), remaining restricted range \(R\), inverse
   range and ion-correction terms at species, material, local density and \(T\).
2. **Small step (linear branch):** \(h<R\) and \(S_0 h \le\) `linear_limit` \(\times T\).
   Restricted mean starts from \(S_0 h\) and receives the Poisson-partition
   correction in Section 3.3. **Large step (range inversion):** estimated restricted
   loss above that fraction, or \(h\ge R\). Outgoing energy comes from inverse range
   at \(R-h\) (if \(h\ge R\), the track stops and the continuous mean is the remaining
   \(T\)). The range branch is not partition-corrected again. Pre-step mass/charge
   scaling is kept during inversion.
3. An ion correction is evaluated at an intermediate energy. That midpoint includes
   mean delta loss, \(T_{\mathrm{mid}}=\max(0.5T,\,T_{\mathrm{mid}}-\tfrac12 D_0 h)\),
   and a low-energy replacement for \(Z>2\). This is not the older
   predictor-midpoint on **total** stopping.
4. The mean is clipped to available kinetic energy; the package's lowest kinetic
   energy can stop the track.
5. If energy straggling is on, IonFluc or Universal/Urban is sampled around that
   restricted mean (`straggling_scale: 1.0` on current production presets). The
   sample is `draw.continuous`. If straggling is off, the mean is used as the
   continuous loss.

On the minibeam **primary C12 water** path only, the sampled total
`continuous+delta` is then multiplied by the frozen 0.9958 and both parts are
rescaled by `deposited/unscaled`. Secondary C12, fragments, Copper and the
broad-beam production presets do not use this factor.

**Which tracks use that sequence.**

| Track | Continuous loss |
|---|---|
| Primary C12 in water or Schneider CT, Unified EM | Sequence above |
| Secondary ions with `enable_secondary_unified_em: true` (water and RT07575 production YAML) | Same sequence |
| Water secondaries on current minibeam **field** YAML | Switch omitted, defaults **false**. Midpoint on the particle-specific water table; homogeneous water is already absolute stopping and is **not** multiplied by density. Midpoint energy per nucleon is \(\max(0.01\,\mathrm{MeV/u},\,(T-\tfrac12 S h)/A)\). Packaged fluctuation, if secondary straggling is on, applies to secondary C12 only. `minibeam_water_secondary_c12_enable_unified_em` can put **C12 only** onto Unified EM; it is default-off. |
| Copper collimator (primary C12 and fragments) | Not in the Unified package. Extracted Copper table, predictor then midpoint \(S\), optional condensed straggling. |
| CT secondaries without Unified EM | Water table with the same mass-stopping factor or density scale at step start and midpoint. |
| Unregistered heavy recoils | Dedicated recoil stopping; generic recoils also have a 5% relative-loss step cap. |

Implementation: `mean` / `unified_em_loss` in
[unified_em_view.hpp](include/carbon/unified_em_view.hpp); primary and secondary
loops in [transport_sycl.cpp](src/transport_sycl.cpp).

The older `ct_primary_midpoint_stopping` and material-secondary stopping paths remain
for non-unified transport. Their tables store density-normalized total stopping:

```text
S1(section,ion,E) = S_extracted / [rho_reference/(1 g/cm³)]
S(T,section,rho) = S1(section,ion,T/A) × rho/(1 g/cm³)
```

That density scaling does not double-count density. Those tables and switches do
not determine the unified model's restricted mean loss. Extra unregistered heavy
recoils retain their dedicated stopping path. See [unified lookup/mean loss](include/carbon/unified_em_view.hpp).

He-4 and other ions with total kinetic energy above the default 600 MeV Geant4 EM
table limit must use an explicit higher `EMRangeMax` (10 GeV in the 2026-09-19
water-slab campaign). A 600 MeV table is not a valid He-4 300 MeV/u reference.

### 3.3. Restricted fluctuations, delta aggregation and partition correction

`S0` is restricted stopping, `D0=M1` is delta mean loss per length, and `lambda_native` is density-scaled native GetLambda. Restricted fluctuations retain the applicable IonFluc or Universal/Urban sampler. For the restricted linear branch:

```text
x = lambda_native h
F(x) = 1 - 2/x + 2(1-exp(-x))/x²    [F(0)=0]
S_eff = S0 - h/2 (S0 F(x) + D0) dS0/dT
mu_delta = h max(0, D0 - (mean_restricted + D0 h)/2 dD0/dT)
v_delta = h M2(T)
Gamma shape = mu_delta²/v_delta; scale = v_delta/mu_delta
```

`F` scales continuous self-drift only; the delta-jump drift remains complete. The ion-correction intermediate energy also includes mean delta loss. This constant-rate, locally linear Poisson-partition approximation compensates the mean-loss change when discrete delta partitions are removed; it has no TOPAS-fitted coefficient.
Means and samples are bounded by available kinetic energy; zero moments skip sampling. Gamma does not preserve the discrete zero-collision atom or higher moments.
The companion table integrates the native spectrum with spin, form-factor and magnetic acceptance, using step-start M2 for variance. Transfers excluded from restricted stopping enter the delta aggregate exactly once. Thresholds remain material/density dependent.
No electron spatial tracking or old electron response is added. `em_macro_ticks` is not a supported current switch.

On the unified minibeam water path, `use_water_electron` is off (`EmMode==1`).
Sampled `draw.delta` therefore remains in the local deposit of the transporting ion.
That is a production-cut mapping to TOPAS explicit electrons, not identity of carrier
labels. Diagnostic ROI tallies record `continuous_sampled`, `delta_sampled`,
`continuous_after_scale`, `delta_after_scale` and `local_total_deposit` from the
same draw without changing RNG or the transport final state.

### 3.4. Coulomb multiple scattering

Both models run **after** the Unified-EM energy-loss draw for the current step.
Nuclear optical depth is independent. They share the same mass radiation length
\(X_0\): G4_WATER in homogeneous water, Schneider 25-section LUT in CT, converted
with local density. Implementation:
[multiple_scattering.hpp](include/carbon/multiple_scattering.hpp),
device helpers in [sycl_device_math.inc](src/detail/sycl_device_math.inc),
applied in [transport_sycl.cpp](src/transport_sycl.cpp).

Two implementations share one YAML selector
([broad-beam FE](docs/broad_beam_fermi_eyges.md)):

```yaml
enable_multiple_scattering: true
multiple_scattering_model: highland   # or fermi_eyges / fe
fermi_eyges_species: c12              # c12 / c12_he4 / c12_he4_pdt / all_charged
fermi_eyges_parameter_set: species_water
fermi_eyges_max_segment_mm: 0.1
```

The code default is `highland`. Water and RT07575 production YAML set Highland
explicitly so older runs do not switch silently. Minibeam field YAML currently
selects `fermi_eyges` with `all_charged`. When `multiple_scattering_model` is
present it is authoritative and also disables older minibeam-only FE keys.

#### Highland: end-of-step angular kick only

Projected RMS angle:

```text
t = rho × (h/10) / X0_mass
C = max(0, 1 + 0.038 ln(t Z²/beta²))
theta0 = 13.6 MeV × Z/(beta p c) × sqrt(t) × C
```

Optionally multiplied by `multiple_scattering_scale` (production default 1.0).
On the minibeam primary-C12 water path, a low-energy transition further scales
`theta0` linearly down to 0.20 for \(E<180\,\mathrm{MeV/u}\). An older minibeam
Highland option can mix a Gaussian core with a rarer wider Gaussian; that is
not the FE model.

Two independent Box–Muller samples give \(\theta_x,\theta_y\sim\mathcal{N}(0,\theta_0^2)\).
They are rotated into the track frame and **only the direction is updated**.
The current step still advances along the **pre-scatter** direction:

```text
position += direction_old × step
```

There is no lateral displacement and no \(y\)–\(\theta\) correlation for that
step. Geometrically the step is a straight segment; the new direction applies
from the next step. This is not Geant4 Urban/Wentzel msc and does not replace
hadronic elastic.

#### Fermi–Eyges: correlated displacement plus Poisson tail

`ion_fermi_eyges_transport_step` splits the physical step into internal
segments of at most `fermi_eyges_max_segment_mm` (0.1 mm). Segment-midpoint
energy interpolates the already-sampled step loss:

```text
E(s) = E0 − (s + h/2)/L × dE
```

Each segment calls `water_ion_fermi_eyges_tail_step`; the next segment uses the
new direction. The returned displacement is relative to a straight drift along
the input direction. The caller then does:

```text
position += direction_old × L + displacement
direction = new_direction
```

Ions outside `fermi_eyges_species` keep Highland. C12 uses the frozen water
candidate `(core, rate, tail) = (9.9 MeV, 0.0025 mm⁻¹, 2.4 MeV)`. Protons,
deuterons, tritons and He-4 use independent pure-water TOPAS fits at
50/150/300 MeV/u with linear interpolation
([calibration](benchmark/fermi_eyges_species_water/calibration_manifest.json));
endpoint values are held outside that interval. He-3 and heavier fragments
still fall back to the C12 constants. The 50–300 MeV/u interval does not cover
the `<50 MeV/u` fragments that dominate part of Bragg fragment dose. Selecting
scopes beyond `c12` remains experimental, especially on Schneider materials.

**Gaussian core.** Scattering power uses the current material \(X_0\):

```text
T = (Es × Z/(beta p))² / X0_mm
Var(theta) = T L
y = (L/2) theta + eta,   Var(eta) = T L³ / 12
```

so \(\mathrm{Cov}(y,\theta)=T L^2/2\). \(\theta_x,\theta_y\) and the independent
\(\eta_x,\eta_y\) are Box–Muller Gaussians. `Es` is `core_MeV`.

**Poisson tail.** Event count \(N\sim\mathrm{Poisson}(\lambda L)\). \(\lambda\)
is fitted per millimetre of water and scaled by the water/\(X_0\) ratio in
other materials. Knuth sampling is untruncated. Each event is uniform along
the segment, with kick \(\mathcal{N}(0,(E_{\mathrm{tail}} Z/(\beta p))^2)\);
the remaining path converts the kick into extra displacement. On a typical
0.1 mm water step the mean is about \(2.5\times10^{-4}\), so most steps have
\(N=0\).

**Diagnostic planes.** If a scoring plane falls inside the step, the code
samples a Brownian bridge for \((\theta(t),y(t))\) **conditional on the already
drawn segment endpoint**. Linear interpolation of start/end states is not used;
it would give variance \(f^2 TL\) instead of \(fTL\) ([failed.md](failed.md)).

| | Highland | Fermi–Eyges |
|---|---|---|
| Lateral displacement this step | none | yes, correlated with exit angle |
| Angular law | single Gaussian \(\theta_0\) | narrow core + rare wide tail |
| Energy inside the step | whole-step, start \(E\) | 0.1 mm segments, mid-segment \(E\) |
| Position update | \(\mathbf{r}+\hat n_{\mathrm{old}} L\) | \(\mathbf{r}+\hat n_{\mathrm{old}} L+\boldsymbol{\delta}\) |
| Not | Geant4 full msc | Urban shoulder; 1 mm thin-slab quantiles are not fully fit |

Minibeam Copper uses a separate `fermi_eyges_tail` entry (scale 1.0; fragment
MCS scale 0.785) with the same process class. Failed MCS/electron widening
routes are listed in [failed.md](failed.md) and are not restored.

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
energies are not globally rescaled. Primary and secondary post-EM null candidates
retain the track and its remaining energy without a nuclear replay or nuclear local
energy dump. The primary branch clears the collision flag and completes the current
EM step, including scoring. The former analytic fragmentation fallback and its
empirical parameters have been removed. Nuclear transport requires the validated
Schneider/unified-water CINEL03 path. See the [removal report](docs/fred_cleanup.md).
The frozen water generation setting is 2; He6/B8/C10 follow the declared EM-only nuclear policy.

Copper minibeam nuclear reactions use extracted C12+Cu rates and a per-energy
fragment-cascade package (gap-filled INCL++ samples). Cascade generation is 3 in
Copper; water generation remains a separate namespace so Copper survivors enter
water at generation 0. Discrete Copper General Ion Elastic is off.

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
| Coulomb scattering | Condensed Highland or FE treatment | Configured Geant4 EM processes |

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
`enable_nuclear_elastic` switch is not the activation method for this path. Frozen
AllIonElastic sources used by the accepted 2026-09-11 packages are in
[extensions/topas/all_ion_elastic/frozen_production](extensions/topas/all_ion_elastic/frozen_production).
See [implementation and data](docs/all_ion_elastic.md).

## 6. Electrons and neutral products

The active unified EM model samples aggregate delta loss but does not track those electrons
spatially. Restricted and delta losses contribute local dose; local deposition itself
is an approximation, particularly for interfaces, lateral tails and minibeam valleys.
There is no active electron packet kernel to disable for another large speed gain.
Empirical Gaussian delta relocation (sigma = 0.5 mm) was tested and rejected
([failed.md](failed.md)).

The repository also contains old section-0 delta-tail redistribution and material
electron-family/packet replay candidates. Packets move an already budgeted energy
weight through recorded states and continuations; they are not a second energy loss
charged to the ion. They are not enabled by the unified production presets or the
current minibeam field YAML, and cannot simply be stacked on the restricted-plus-delta model.
See [packet transport](include/carbon/electron_packet_transport.hpp).

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

No fitted dose normalization or registration is used. Quality checks cover finite
values, provenance, sampling audits, energy accounting and zero queue overflow.
An overflow invalidates the shard: split and rerun. Global energy closure does not
establish spatial-dose accuracy or close every nuclear Q value.

### 7.1. CT Gamma

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

The older September 5 whole-volume/no-BODY/0.5-mm results remain historical.

### 7.2. Minibeam fixed ROI

Minibeam lateral scoring uses a 1000 × 1000 plane (0.1 mm × 0.25 mm). Canonical
fixed ROIs, with coordinates from grid metadata and pitch 3.6 mm, are:

```text
peak:     |folded x| < 0.25 mm
shoulder: 0.25 mm ≤ |folded x| < 0.9 mm
valley:   0.9 mm ≤ |folded x| ≤ 1.8 mm
```

within the central field `|x| ≤ 18 mm`. One-millimetre depth planes are smoothed
along depth; Bragg ±2 mm integrals use unsmoothed arrays. Incident-history
normalization is absolute Gy after scaling by history counts; surviving-particle
renormalization is not used.

TOPAS `ChargedOriginDoseToMedium` is origin classification: nuclear tracks use
their own Z/A, while electrons inherit a known charged ancestor. GPU charged-origin
maps are depositing-ion Z class with condensed electrons included. Those comparable
classes must not be added to unmatched TOPAS-only classes (`neutral_origin`,
`unclassified`). GPU/TOPAS residual contribution is

```text
Δ_i = (D_GPU,i − D_TOPAS,i) / D_TOPAS,total
```

Uncertainty is unknown unless per-incident-history ROI moments or independent
shards exist. `|D|/√N_histories` is not a valid ROI error. TOPAS multi-threaded
EventID is not a source-file row number.

Optional diagnostics (off by default, must not change production dose):
energy-band ROI for p/d/t/He-4; primary-C12 ROI ledger; water-entrance phase
space; in-water primary planes. See
[analyze_minibeam_residual_attribution.py](benchmark/carbonminibeam/analyze_minibeam_residual_attribution.py)
and [minibeamresult.md](minibeamresult.md).

### Full-history three-case comparison (2026-09-15 campaign)

The September 15 three-case CT comparison remains a historical campaign record.
Gamma follows the BODY, 10% cutoff and DTA/10 protocol above; unfinished pass
rates are not reported as validation. TOPAS includes `CarbonIonElasticPhysics`,
while current GPU production omits independent nuclear elastic.

## 8. Species grouping and measured throughput

Current production algorithm (2026-09-15): the accepted candidate ran RT07575's 1/20 shard (6,481,909 primaries, two subdivisions) in 68.61 s wall / 65.20 s program elapsed, 99.4k histories/s, zero overflow. b1 peak errors at 100/200/300 MeV/u were −0.0715%, −0.1098%, +0.0424%. Patient BODY Gamma remains unvalidated. Older timings below and Section 9 proposals are historical research; delta aggregation is now enabled together with partition correction. CT Fermi–Eyges remains optional research; forcing 0.10 mm real FE material-refresh steps on all CT C12 was rejected ([failed.md](failed.md)).

### Current memory use and shard sizing (2026-09-15)

Primaries launch in `history_chunk_size` batches, but the secondary queue is not emptied
after each primary batch: all primaries in the shard finish before secondary generations
are transported. The queue has a fixed **32,000,000**-entry capacity. Continuation also
allocates **272 bytes of state** per current-generation secondary plus indices/flags.
Shard histories therefore affect peak secondary memory; the roughly 4.4 GB observed
during the primary phase is not the whole-run peak.

| RT07575 diagnostic | Primaries | Primary batch | Full wall s | Program histories/s | Sampled peak MiB |
|---|---:|---:|---:|---:|---:|
| Larger primary batch | 3,240,955 | 131,072 | 34.65 | 99,216 | 7,817 |
| Larger shard | 4,861,226 | 34,816 | 48.88 | 102,989 | 9,517 |

Both passed quality checks with zero overflow and unchanged production physics.
Memory was sampled every 0.25 s and may miss shorter transients. The earlier approximately
3.24M-history / 34,816-batch runs took 34–35 s; single measurements do not establish a
speedup confidence interval. The larger primary batch showed no clear benefit, so the
production batch remains **34,816**. This CT benchmark targets at most **4.9M primaries**
per remaining shard; this is a scheduling choice, not a universal safe limit or a physics
change. Different CTs and spectra still require checks. Allocation failures or secondary
overflows exclude the attempt from dose and trigger subdivision. Diagnostic doses are
excluded, and repartitioning conserves every spot's integer history total.

The production integration replay of RT07575's 6,481,909 primaries took **68.76 s wall** /
**65.45 s program elapsed**, approximately **99.0k histories/s**, consistent with the
accepted candidate above.

### Default secondary continuation (2026-09-14)

Both production presets now enable `secondary_step_chunking: true` alongside species
grouping. A launch performs up to 16 complete secondary loop iterations, saves the
surviving tracks and stably compacts their indices. Queues below 8192 finish directly.
No physical step is shortened or merged. Energy, position/direction, RNG counter,
material cache, pending deposits and diagnostics persist across launches;
terminal scoring happens only when the track actually terminates. Each generation
finishes before its descendants start. Setting the option to `false` restores full-track
launches without continuation buffers; species grouping is controlled separately.

The September 14 prototype used 336 bytes of state plus about 20 bytes of indices/flags per
secondary; its 2.88M-track pool needed about 1.03 GB extra memory. The current state is
272 bytes per secondary (16-byte aligned compact resume), as described above. Allocation failure stops the run; reduce histories per shard and merge
outputs. Primary transport also skips a duplicate mean-loss lookup unless its optional
audit needs it; the physical loss sampler remains unchanged.

Validated prototype RT07575 1M throughput was 29.5–29.6k histories/s without independent
elastic (+36.6–39.7% over the mean-lookup-optimized baseline), and 28.2–28.6k with
all-ion elastic (+34.7–39.2%). Counts, EM audits and steps matched, with zero overflow.
The maximum elastic dose difference was 0.000955% of peak (about 0.00337% locally at
the worst voxel); the user accepted it for production. This stable difference exceeds
self-repeat noise and its origin remains unproven. The quality report records
`secondary_step_chunking_accepted`; these GPU scheduling checks are not a new TOPAS
Gamma validation. The existing low-density and patient-Gamma caveats remain.
The production-source ON/OFF check measured 21,559 → 29,535 histories/s (+37.0%)
for RT07575 1M without independent elastic, with a maximum dose difference of
0.0000966% of peak.

`secondary_species_grouping: true` creates a GPU index permutation for each secondary
generation: 18 species buckets plus other products. Histogram, prefix sum and scatter
leave particle records, parent histories and RNG streams intact. Descendants are
processed in the next generation. Additional index memory is approximately 4 bytes per queue slot.

The two production YAML files explicitly opt in; the configuration-structure default
is false for compatibility. Logs print the mode and grouping time. The quality report
retains `secondary_species_grouping_accuracy_pending` after the authorized integration.

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
- Do not retune primary MCS or enable electron packets to compensate minibeam
  valley residuals until homologous C12 trajectory, energy-loss partition and
  spatial scoring are separated (Section 11).

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

Production EM additionally requires `data/em/unified_em_delta_moments_v2.bin`. After installing core data, run `python3 tools/build_delta_moments.py`, then `python3 tools/verify_unified_em_data.py`. This pinned derivative covers all material/ion nodes; older releases do not include it.

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

The [extensions kit](extensions/README.md) regenerates EM, CINEL03, Schneider rates,
all-ion elastic and Copper packages against TOPAS 4.2.p3 / Geant4 11.3.2. A different
TOPAS/Geant4 version is a new package version. Large `.bin`/`.cinpkg` artifacts are
not stored in Git; hashes are in `extensions/package_manifest.json`.

This revision updates documentation to the 2026-09-20 working tree. It adds no new
physics validation of CT Gamma or minibeam valley matching.

## 11. Minibeam validation protocol (current)

Minibeam development freezes primary water loss scale 0.9958, Copper/slit geometry,
packages and seed unless a controlled A/B names the change. Secondary C12 FE and
secondary 0.9958 remain off as production claims; FE for p/d/t/He-4 water is an
experimental field-YAML option, not a completed CT calibration.

Homologous C12 water transport uses the same parent-0 C12 entrance set in GPU and
TOPAS. Empty-history TOPAS phase-space runs are not interchangeable with
no-empty runs: the 250 MeV/u 256k file `topas_6363` (empty histories on, 256000
histories) is not comparable to the 30878-particle GPU replay, whereas `topas_6364`
(empty off, 30878 emissions, weight 1, `PhaseSpaceMultipleUse=1`) is. Do not
mechanically scale dose by survivor/incident. GPU-entry versus TOPAS-entry through
the same GPU kernel is an **entrance-replacement sensitivity**, not a cross-engine
comparison.

On that homologous protocol, 250 MeV/u full-physics 40 mm valley GPU/TOPAS was
0.9867 and 300 MeV/u EM-only Bragg valley was 1.0037, with C12 crossings, energy
and angle at 40 mm and 160 mm close on the EM-only pair. Full-chain valley residuals
are larger and are not attributed to primary MCS from occupancy fractions.
Next kernel changes require a documented split among entrance, C12 transport,
energy evolution and scoring location ([minibeamresult.md](minibeamresult.md) §37).

## References

- [FRED carbon model analysis](docs/FRED_Carbon_Fragmentation_Model.md)
- [Unified EM data/model](docs/physics/unified_em_v1.md)
- [Older primary-water model, removed from code](docs/physics/water_joint_em_v1.md)
- [Historical physics specification](docs/TOPAS_GPU_Physics_Model.md)
- [Broad-beam Fermi–Eyges](docs/broad_beam_fermi_eyges.md)
- [Scoring contract](docs/scoring_validation.md) and [archived drafts](docs/archive/README.md)
- [Minibeam running notes](minibeamresult.md)
- [Rejected routes](failed.md)
- [TOPAS extraction kit](extensions/README.md)
