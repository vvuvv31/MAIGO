# Materials and methods

Revision: 2026-10-05; implementation baseline `ebff981`. [中文](mm_zh.md). This describes the physics processes, approximations, scoring and execution of the current eight presets. See the [cleanup regression](docs/repository_cleanup_20261005.md) and [CT throughput/acceptance report](docs/ct_throughput_20261005.md). See also the [CT broadbeam electron optimization](docs/broadbeam_300k_20261005.md) and the [latest exact EM search reuse](docs/broadbeam_em_search_20261005.md). Both historical CT broadbeam configurations pass the requested dose gate; carbon CT broadbeam now enables primary/secondary fluctuations and requires dose requalification; the historical lung proton minibeam result falls below it. The ion-step defaults below have since changed: minibeam is now 0.1 mm and water broadbeam 0.5 mm. These changed configurations have not yet been requalified for dose or throughput; CT broadbeam remains at its measured 0.5 mm setting. Historical measurements apply to their recorded executable, source, geometry and statistics.

## Eight presets

The supported configuration matrix contains exactly eight entries. Presets select projectile and geometry; electron propagation is shared.

| Preset in `config/` | Ion scattering | Electron birth selection | Recorded verification (before step-default update) |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Current 3D local 1%/1 mm 99.8935%; passed |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Historical fluctuations OFF: local1%/1mm99.8699%; dose acceptance with fluctuations ON pending |
| `proton_ct_minibeam.yaml` | Urban | All primary ionization in CT | Shared Urban/FP32 logic; lung CT does not yet pass the new >99% gate |
| `carbon_ct_minibeam.yaml` | Urban | All primary ionization in CT | Scheduling 1.946× faster; transport audits passed; dose pending |
| `proton_water_broadbeam.yaml` | Gaussian Fermi–Eyges | Empty selection in unit-density water | Two-batch smoke passed; water campaign deferred |
| `carbon_water_broadbeam.yaml` | Gaussian Fermi–Eyges | Empty selection in unit-density water | Two-batch smoke passed; water campaign deferred |
| `proton_water_minibeam.yaml` | Urban | All primary ionization in water | Shared-worker smoke passed; water dose acceptance pending |
| `carbon_water_minibeam.yaml` | Urban | All primary ionization in water | Shared-worker smoke passed; water dose acceptance pending |

The two CT broadbeam presets use RT07575 plan inputs. The CT minibeam presets use the lung CT, with 170 MeV protons and 250 MeV/u C12 respectively. These YAMLs are not the RT07575 minibeam full plan; running that plan requires its actual geometry, spots, normalization and reference. Carbon water minibeam retains its 250 MeV/u spot source and 1.2% relative energy spread.

The default [composition-base/density-correction ion EM representation](docs/em_density_factorization_20261005.md) reduces the ion EM plus delta-moment files by 81.033%, using 26 composition bases and sparse corrections derived from the original 175 density nodes. Native cut scalars and stopping/range splines remain exact; reconstructed fields differ by at most 0.005% in the component check. This prototype reconstructs the original dense GPU tables on the host, so it does not reduce GPU memory. The presets now load compressed EMFAC001/EMDFAC01 at their existing `data/em/` paths; native sources and metadata are archived in `data/archive/ion_em_native_20261005/` for offline generation/regression. Its RT07575 validation uses the **head** CT and full optimized spot plan, independently of the lung CT minibeam presets. Full-plan results and Monte Carlo counts are reported separately in the linked report. Full-plan results: factorized proton/carbon broadbeam local1%/1mm are99.8810%/99.7877% (historical configurations passed; carbon used fluctuations OFF, so this does not qualify the new ON default). RT07575 **head** proton minibeam now uses0.1% of the4Gy plan:236,086,480 primary per arm, all487 positive spots. Local3%/0.2mm is98.9373% factorized versus98.9233% native, so neither passes>99%; search spacing is0.1×DTA. Factorized versus matched native passes100%, with relative doseL1 of0.2228%. Head resident throughput is87.3k/s for both arms, with no resolved gain or GPU-memory reduction. The user authorized the compressed encoding as default; the independent TOPAS head accuracy gate remains failed.

All eight updated presets pass two-batch runtime smokes with FP32 dose; carbon CT broadbeam ON gamma remains pending. See the [fluctuation default checks](docs/ion_straggling_defaults_20261005.md).

## Materials, source and transport geometry

MAIGO is a C++20/SYCL Monte Carlo implementation combining condensed charged-particle transport with TOPAS/Geant4-derived tables and sampled conditional final states. The extraction/reference stack is TOPAS 4.2.p3 with Geant4 11.3.2. The GPU does not run a complete Geant4 physics list: finite response libraries, interpolation, restricted particle coverage and the approximations below are part of the method. Validation applies to the stated configuration and reference.

CT inputs provide voxel dimensions, origin, density and Schneider composition indices. Material-dependent transport uses the local composition and physical density, rather than treating all CT voxels as scaled water. Ion EM tables cover 175 material/density entries and 18 charged species; the electron bank has a separate 82-node Schneider registry. These are different discretizations. The registered ion species are p, d, t, He3/4/6, Li6/7, Be6/7/9/10, B8/10/11 and C10/11/12. Additional elastic recoils require their dedicated stopping data. See [CT geometry](include/carbon/ct_grid.hpp), [charged species](include/carbon/charged_species.hpp) and [EM extraction](extensions/tools/unified_em/prepare.py).

The water stopping reference is **TOPAS Water_75eV**, with mean excitation energy 75 eV. `data/water_unified/g4_water_material.json` supplies composition/radiation-length information; its G4_WATER identity does not make the stopping reference 78 eV. `minibeam_water_primary_stopping_power_scale` remains 1.0. Electron material section25 denotes water; the ion EM registry uses section−1 for water. They must not be interchanged.

Sources are either the configured flat field or a spot list. TPS sources allocate histories from spot weights, sample the configured energy spread and correlated transverse position/angle distributions, and transform beam coordinates into the patient frame using the plan pose and isocenter. The `mu` and `histories` weight modes have different allocation semantics; changing only `number_of_histories` does not rescale a history-weighted plan. Carbon source energies stated in MeV/u are converted to total kinetic energy for transport. Upstream air loss and source covariance corrections apply when selected by the preset; they do not provide an explicit upstream electron shower. See [TPS source](src/tps_source.cpp).

CT material faces limit transport steps, including oblique incidence. Scoring and physical geometry are separate: a particle outside the scoring ROI can still be inside the transport world. The water electron worker receives one Water_75eV cell spanning the physical water box, independently of the finer scoring mesh; this does not activate Schneider ion physics in water. Boundary crossings update the actual material and density. Numerical boundary nudges are navigation operations, not extra physical energy loss.

## Ion electromagnetic processes

### Mean ionization loss and range

The common EM package supplies restricted stopping power, range/inverse-range information, charge and fluctuation parameters, and above-cut delta-electron loss moments for each supported species/material. Native spline data reconstruct the stopping/range response; values from adjacent density nodes are interpolated within the same composition. Small losses use stopping power times accepted **true path**. Larger losses use range inversion and the extracted ion corrections. A range-terminal step deposits the remaining energy. This is condensed transport, not explicit sampling of every atomic excitation.

The total ionization loss is the restricted continuous component plus an aggregate above-cut delta component. Both are bounded by the remaining kinetic energy. The delta mean includes the local energy-dependence correction. Ion energy removal and spatial electron deposition are separate operations: the birth response partitions the removed energy afterwards; an individual delta draw does not create a list of physical electron tracks. See [mean loss](include/carbon/primary_em_mean_candidate.hpp) and [runtime EM state/loss](include/carbon/unified_em_view.hpp).

### Energy-loss fluctuations

With straggling enabled, restricted loss uses the extracted ion/universal fluctuation model and its material production cut, dispersion and kinematic maximum transfer. Its thin/thick-step regimes are not a single Gaussian approximation. The independent aggregate delta draw is moment-matched Gamma with shape `mean²/variance` and scale `variance/mean`; zero variance returns the mean. This reproduces two aggregate moments, not the exact compound-Poisson zero-collision probability or higher moments. See [restricted sampler](include/carbon/restricted_fluctuation_candidate.hpp) and [delta sampler](include/carbon/delta_loss_sampling.hpp).

**All eight presets enable primary and secondary ion energy-loss fluctuations.** These are also enabled by default in TransportConfig. All four minibeams enable copper primary and charged-fragment loss fluctuations. The previous carbon CT broadbeam acceptance used fluctuations OFF; its historical gamma/throughput values do not validate the new ON configuration. Carbon water minibeam previously disabled copper-fragment fluctuations, now enabled.

### Ion multiple Coulomb scattering

Broadbeam uses the Gaussian Fermi–Eyges branch. In each approximately constant-scattering-power segment, the projected angular/lateral covariance is `A0 = T L`, `A1 = T L²/2`, `A2 = T L³/3`; correlated normal draws give both angular deflection and lateral displacement. Scattering power depends on charge, momentum, velocity and the local material radiation length. Species/energy coefficients originate from water slab fits; unparameterized species use the common coefficients. The Gaussian branch incorporates the fitted tail contribution into the second moment and does not sample a separate Poisson tail. These fits are an approximation, not native Urban scattering. Their historical G4_WATER slab provenance concerns the scattering fit and does not replace the Water_75eV stopping reference. See [Fermi–Eyges implementation](src/detail/sycl_device_math.inc).

Minibeam uses the common **Urban v2 ion** path. Material-dependent native parameters set scattering/path limits; geometry limits the geometric advance; loss uses the corresponding accepted true path; angular and lateral draws are committed after the step has been accepted. Primary and supported charged fragments use this machinery with their own species data. A nominal maximum step is an upper bound, not a fixed transport increment. Both CT minibeams enable `urban_ct_density_scaling`: the native material context is scaled by actual/reference density for path, stopping/range and mean-free-path handling. This fixed-cut approximation does not encode all density-effect or production-cut energy shifts, so shared Urban code alone is not a claim of exact native-density equivalence. See [Urban integration](src/detail/sycl_urban_global.inc) and [Urban package](include/carbon/urban_mcs_package.hpp).

### Effective preset settings

These are the effective settings after `transport_profile` expansion in [configuration parsing](src/config.cpp), not a list of every YAML key. The ion kinetic cutoff is 0.1 MeV **total energy**, not 0.1 MeV/u. Electron cuts are separate and material-dependent.

| Preset | Ion maximum step, mm | Ion MCS | Primary / secondary ion straggling | Independent all-ion nuclear elastic bank | Neutron worker |
|---|---:|---|---|---|---|
| Proton CT broadbeam | 0.5 | Gaussian FE | On / on | On | Off |
| Carbon CT broadbeam | 0.5 | Gaussian FE | On / on | Off | Off |
| Proton water broadbeam | 0.5 | Gaussian FE | On / on | On | Off |
| Carbon water broadbeam | 0.5 | Gaussian FE | On / on | Off | Off |
| Proton CT minibeam | 0.1 | Urban v2 | On / on | On | Off |
| Carbon CT minibeam | 0.1 | Urban v2 | On / on | On | Off |
| Proton water minibeam | 0.1 | Urban v2 | On / on | On | On, experimental |
| Carbon water minibeam | 0.1 | Urban v2 | On / on | On | Off |

The CT broadbeam mean-loss step fractions are 0.02 for proton and 0.05 for carbon; water broadbeam uses the 0.01 default. Boundary, range and interaction limits can shorten these steps. The default `maximum_step_mm` is 0.5 mm for broadbeam and 0.1 mm for minibeam, for both projectiles and both geometries. All eight YAML presets specify these values. If the key is omitted, `load_config()` selects the default from `minibeam` and records it in the canonical configuration; an explicit value overrides the default. This setting limits phantom primary and charged-secondary ion transport; the separate copper cap remains 0.05 mm. Common algorithms therefore do not imply identical numerical controls across all eight presets.

## Nuclear interactions and charged secondaries

### Discrete nuclear elastic scattering

Where `all_ion_elastic_file` is present, material/species-dependent elastic rates and sampled angular outcomes update projectile energy/direction and create the target recoil. Recoils covered by the charged-species registry or dedicated recoil stopping tables are queued above cutoff; sub-cutoff kinetic energy is deposited locally. Unsupported above-cutoff recoils are recorded in the untracked-energy ledger. Queue overflow is a failed run, not permission to discard energy.

This discrete nuclear channel is distinct from electromagnetic multiple scattering. In particular, `enable_nuclear_elastic: false` alone does **not** disable the independently selected all-ion bank. Both carbon broadbeam presets omit this bank; all proton presets and both carbon minibeams select it. See [elastic data/sampling](include/carbon/all_ion_elastic.hpp) and [primary/secondary dispatch](src/transport_sycl.cpp).

### Inelastic reactions and fragment cascades

Macroscopic reaction rates depend on projectile, energy, local composition and density. In the current Schneider v3 path, elemental partial rates are masked to their supported channel domains, interpolated in energy, and summed; target probabilities use the same partials as the interaction hazard. Unsupported channels are not replaced by oxygen or a nearest element. See [target/rate sampling](include/carbon/schneider_target_sampler.hpp).

CINEL03 stores a correlated native nuclear event keyed by `(projectile Z, A, target Z, collision energy)`. An in-domain query selects one of the two bracketing energy nodes with linear probability, then samples an event at that node. The allowed node gap is at most 5 MeV/u. **No per-event kinetic-energy rescaling is applied in this ion CINEL03 path.** One common random azimuth rotates the entire final state into the incident frame, preserving within-event directional correlations. The finite-node energy mismatch and nuclear residual accounting remain explicit; sampled fragments are not independently reconstructed from marginal spectra. See [lookup contract](include/carbon/inelastic_package_v3.hpp) and [event rotation](include/carbon/cinel03_event_rotation.hpp).

The package provenance names proton primary Binary Cascade (`QGSP_BIC_HP`), C12 primary `FTFP_INCLXX`, and secondary-ion extraction with `g4em-standard_opt4 + g4ion-binarycascade`. These labels describe the extraction sources, not a single complete physics list executing on the GPU. The live files are inventoried in [the package manifest](docs/package_manifest.json).

Process-local deposit is scored at the collision. Supported charged products above cutoff enter the secondary queue and undergo EM transport, scattering and eligible subsequent nuclear reactions; sub-cutoff charged energy deposits locally. The current patient/water CINEL03 dispatcher terminates the reacting incoming track and transports the recorded direct products; stored parent fields should not be interpreted as generic parent-continuation support. Patient/water secondary inelastic generation limits are 2; the copper fragment limit is 3. Reaching a generation limit disables further inelastic branching, not the remaining charged-particle EM transport. The configured `em_only` policy for out-of-scope secondary nuclear channels and the finite isotope/event coverage are explicit model limitations.

Neutral products, unsupported products, escapes and nuclear residuals have separate audit fields. Diagnostic neutral/unsupported breakdowns must not be added again to the combined nuclear sink. Nuclear photons are not automatically passed to the electron-response photon worker. With the neutron module disabled, nuclear neutron energy is accounted as untransported, not scored as local dose.

### Copper collimator and upstream transport

Minibeam tracks pass through the configured copper slit geometry and air gaps before entering the phantom. Copper uses density 8.96 g/cm³ and its own stopping, Urban scattering, elastic and inelastic data; collimator attenuation is therefore more than a geometric aperture. Its charged step cap is 0.05 mm. The proton presets use the `g4_universal_delta` copper proton-loss option; the carbon presets retain `legacy_clipped_gaussian` for that option. Copper fragment straggling is enabled in all four minibeams. Upstream electron showers are not supplied by the patient/water primary birth bank. Copper-specific settings are retained in each [minibeam YAML](config/).

### Optional neutron transport

Only proton water minibeam enables the experimental neutron worker by default. It transports nuclear neutron births in the water/copper/air world using tabulated elastic, inelastic, capture and fission channels, with boundary-aware optical-depth flights. Eligible exclusive elastic events above 0.1 MeV use two-body kinematics at the actual energy. Other bank events retain correlated final states with energy rescaling and explicit kinetic-gain accounting. Charged recoils use stopping data; optional water recoil Urban transport is disabled in this preset. Neutron-generated photons/electrons remain unresolved and are separately accounted. This is not a validated general CT neutron solver. See [neutron implementation](src/neutron_transport_sycl.cpp).

## Shared electron and photon transport

### Birth partition and energy-packet estimator

The primary ionization loss is partitioned using the energy-weighted birth yield for the projectile, material, density and energy: the residual deposits locally, and the represented electron energy enters a task. The proton and C12 birth banks differ; downstream electron/photon physics is shared. The current primary-birth implementation does not add explicit electron births from secondary ions or upstream copper/air cascades. Nuclear recoil deposits are not folded into the primary birth response.

The task carries **represented energy W** and a sampled physical electron kinetic energy **E**. They are not interchangeable. Group32 uses weighted reservoir sampling to combine birth contributions into a representative task; it does not transport 32 analog electrons. Birth energy and polar angle are sampled jointly, with azimuth relative to the ion direction. See [birth distribution](include/carbon/electron_birth_bank.hpp) and [task state](include/carbon/electron_birth_task.hpp).

The worker follows a single energy-weighted branch through the cascade. For a soft loss ΔE, the standard estimator deposits W at a sampled position with probability ΔE/E and terminates; otherwise it continues with physical energy E−ΔE and unchanged W. At a hard interaction, an energy-weighted conditional CDF selects local deposition, a continuing electron, a secondary electron or a photon, again retaining W. Thus the expected dose follows the extracted branching response without storing a full shower. It does not reproduce event multiplicity or all shower correlations. Grouping and roulette affect variance; finite response statistics and interpolation can also affect accuracy. See [packet estimator](include/carbon/electron_charged_packet.hpp) and [FP32 worker](src/detail/sycl_independent_electron_fp32.inc).

### Independent charged-electron steps

At the current material and energy, the worker evaluates restricted stopping, range, elastic rate and hard-ionization/bremsstrahlung rates. The ordinary step is limited by **0.05 mm**, a **0.005 mean-loss fraction**, remaining range, the next physical voxel face and a proposed interaction distance. These electron controls are distinct from the ion step cap. Material production cuts and low-energy endpoints come from the operators; they are not the ion 0.1 MeV cutoff.

Soft loss uses extracted electron Urban fluctuation parameters. Elastic deflections use the Geant4 GS single-elastic law including Mott corrections. This is neither ion Urban MCS nor a new low-density condensed-MCS approximation. Hard events use certified majorant proposals, rate-based acceptance/null collisions and the extracted conditional ionization/bremsstrahlung outcomes. Boundary crossings continue from the resulting state in the new material; missing support and iteration-limit failures are reported, not converted into successful local deposition. See [step algorithm](include/carbon/electron_charged_step.hpp), [elastic law](include/carbon/electron_elastic.hpp) and [Urban loss](include/carbon/electron_urban_fluctuation.hpp).

Within a composition, mass stopping/range and mass interaction rates interpolate between actual density nodes and are converted back using physical density. Hard final-state node mixing uses process-rate weights. The charged model uses the lower-node elastic angular law and composition Urban constants; this choice and finite density interpolation are approximations, not an assertion that every density response is identical. See [density model](include/carbon/electron_charged_density.hpp).

### Electron density nodes and compression

The active shared charged bank has **82 Schneider density nodes plus one Water_75eV node**, covering25 tissue compositions and water; these are not82 separate compositions. Birth distributions, restricted stopping, hard rates and conditional final states are extracted at actual density nodes and interpolated/mixed within a composition. Joint already density-scales its spatial advance, which does not make all electron responses linear in density.

The ion representation is likewise **26 composition bases plus sparse corrections for175 density states**, reconstructed on the host. Electrons can use this storage approach: share composition Urban/elastic parameters, encode mass-normalized scalar bases plus density corrections, and compress birth/collision joint CDFs while retaining normalized probability, threshold support, energy/angle correlation and process-rate weighting. Removing the other density information entirely is an unvalidated new approximation.

A read-only audit of the installed bank finds Schneider section1 at densities0.05946 and0.87963g/cm³ has electron cuts1.60075 and52.6970keV. At100keV its restricted mass stopping is0.32525 versus0.41212MeV·cm³/(mm·g), a26.7% difference including the cut dependence. Multiplying a single table by density cannot reproduce this. Share redundant composition data and preserve density corrections first; conditional-distribution compression needs independent validation. The electron runtime bank has not been replaced by this proposed representation.

### Joint-response acceleration

Covered Schneider states may use a compact correlated joint operator for short advances. The whole displacement support must be inside a region of the same composition **and density before sampling**; eligibility cannot depend on the drawn outcome. Each composition uses its minimum indexed density as the reference and scales lengths/support by `reference density / physical density`, without enlarging support beyond the reference. Charged segments remain ≤0.25 mm. Near an interface or outside joint support, the same packet continues in the independent charged worker. Photon continuations use actual-geometry photon transport, not a stored source-sphere flight.

Joint attempts occur at birth, after photon-to-electron conversion and periodically during charged continuation. Water_75eV has no joint operator and uses independent steps. Joint scaling and finite sampled operators require dose validation; they are not an exact replacement theorem. The runtime does not fall back to the retired spatial trajectory package. See [joint operator](include/carbon/electron_joint_pilot.hpp).

### Photon branches

The common photon bank covers photoelectric absorption, Compton scattering and Rayleigh scattering. Pair production is outside the present sub-MeV support. A sampled optical depth `τ = −ln(U)` is consumed by `μ × distance` through the actual sequence of CT/water cells until collision or physical escape. A material boundary does not itself deposit energy or resample the remaining optical depth. Photons are not subject to the ion or electron step caps.

At collision, process rates select the interaction and its energy-weighted conditional outcome selects local deposit, a continuing photon or an electron returned to the charged worker. Mass attenuation scales with physical density; conditional final states mix bracketing density nodes with geometric interpolation weights, an explicit finite-node approximation. Energy is scored at the sampled interaction/deposit location, not spread along a photon flight. See [optical transport](include/carbon/photon_optical_transport.hpp) and [bank sampling](include/carbon/photon_response_bank.hpp).

### Broadbeam selection and statistical acceleration

All four minibeams enable primary electron births throughout their phantom. Broadbeam uses:

```yaml
broadbeam_electron_tracking: true
broadbeam_electron_density_threshold: 0.1
broadbeam_electron_halo_mm: 5.0
```

The CT selection is a **box dilation** of density ≤0.1 g/cm³ by 5 mm, evaluated at the sampled birth position. Outside this selection, the electron contribution deposits locally. An accepted packet can leave the selection during transport. Uniform unit-density water has an empty selection, so its broadbeam electron contribution remains local under this policy. Tracking OFF removes the independent bank from the run. Neither a proton-only spatial tail nor a carbon-only replay path is needed.

Both CT broadbeams use group32, birth-index ordering by sampled energy/density, low-density survival probability 0.5 and selected-tissue survival probability 0.5. Surviving packets receive W/p and retain downstream electron/photon transport; rejected packets contribute to the signed roulette ledger. This preserves the expectation of the existing estimator while changing variance. These ordering/roulette settings do not apply to minibeam or water. External air is not simply discarded or forced to local deposition, and the research expected-value and low-density condensed-scattering alternatives are not the preset baseline.

## Dose scoring, numerics and execution

Ion local loss, nuclear local deposits and sub-cutoff charged energy are accumulated in their scoring cells; electron packets score at the sampled deposit position. Repeated contributions in one cell may be buffered before an atomic update. The voxel-dose output is **dose to medium** using the actual CT voxel mass (or water mass), not a stopping-power conversion to dose to water:

```text
mass_kg = density_g_cm3 × voxel_volume_mm3 × 1e−6
D_Gy = accumulated_energy_MeV × dose_output_scale × 1.602176634e−13 / mass_kg
```

The CT scorer must match the CT grid; the implementation floors mass density at 1e−6 g/cm³. A depth aggregate is not a replacement for the mass-aware CT voxel dose. Raw scored energy/dose is accumulated over histories; the dose writer does not automatically divide by the history count. Per-primary or full-plan comparisons must explicitly use the recorded spot allocation, total histories and `dose_output_scale`. See [scoring/output and audits](src/io.cpp).

GPU voxel scoring/atomics are **FP32** (`CARBON_DOSE_FP32=ON`, `CARBON_DOSE_FP64=OFF`). The independent electron/photon production implementation uses FP32 transport and geometry with integer CDF draws. This does not mean every host calculation, table on disk, ion local accumulator or audit is FP32. Reference helper headers containing `double` are not evidence that the selected electron kernel uses FP64.

Primary/charged-ion random streams use counter-based Philox4x32-10 with history, step and dimension identifiers; secondary identity is independent of its atomic queue slot. The electron worker has a task-owned 64-bit linear-congruential state and integer CDF draws; the neutron module uses its own xorshift stream. Thus task ordering can change FP32 atomic summation order without changing the task's transport stream. See [ion RNG](include/carbon/rng.hpp), [electron FP32 implementation](include/carbon/electron_fp32.hpp) and [neutron worker](src/neutron_transport_sycl.cpp).

Primary, charged-secondary and electron work uses bounded queues/chunks. Species grouping and finite-step secondary continuation improve scheduling; continuous batches reuse immutable packages and geometry in one process. The latest EM optimization certifies equal energy/stopping/range search grids once on load and reuses three same-step interval indices across density endpoints. It retains interpolation values and random draws; inverse-range and hard-rate searches remain separate, and unequal grids use separate searches. This exact implementation optimization is distinct from response interpolation, FE scattering, finite nuclear libraries and the electron energy-packet approximation.

Run acceptance requires completion, EM/transport coverage, finite states, energy/roulette accounting and zero queue overflow as well as dose checks. Unsupported-energy ledgers, the finite cascade depth and untransported nuclear neutral products must remain visible. Gamma agreement in one geometry does not validate omitted channels, out-of-domain sources or all eight presets.

## Data layout

The electron registry is [docs/electron_packages_manifest.json](docs/electron_packages_manifest.json). There are three packages:

| Package | Runtime contents | Payload bytes |
|---|---|---:|
| `data/electrons/proton/` | Proton `birth_bank.bin` | 328,620,496 |
| `data/electrons/carbon/` | C12 `birth_bank.bin` | 1,033,490,328 |
| `data/electrons/shared/` | `joint.bin`, charged and photon operators/indices | 3,837,722,550 |

Total unique payload is 5,199,833,374 bytes; JSON manifests are excluded. Only the chosen projectile birth package and the shared package load for a run. Loader accounting for the 83-node package reports approximately 3.127 GB proton or 3.597 GB carbon device data, excluding other physics, queues and scorers; these are not total process memory or peak loading footprints.

Birth, charged and photon banks contain 82 Schneider density nodes plus one Water_75eV node. Electron material section25 denotes water; Schneider sections remain0–24. Water's charged/photon domain reaches0.8 MeV, while Schneider remains0.65 MeV. Water primary birth data reach250 MeV for protons and300 MeV/u for C12; Schneider birth limits are250 and255 MeV/u respectively. A source outside actual material/energy support is not validated by this table. Missing runtime support is an error, not a reason to substitute water or local deposition.

Current schemas are MAIBIR02 birth, MAIJNB02 joint, MAIECB02 charged index and MAIPHB02 photon index. The indices contain section, density and relative payload path. Validation checks dimensions, sizes, materials, projectile identity, finite values, sampling support and certified collision-rate bounds. The runtime and current extractors have no file-digest checks. Retired configuration keys are rejected; no digest-field or old-package compatibility entrypoint is provided.

Other packages remain necessary: unified EM and delta moments; projectile nuclear rates/events; elastic/recoil data where selected; Urban tables for minibeam; copper/air tables and optional neutron data. The live dependency inventory is [docs/package_manifest.json](docs/package_manifest.json). Large binaries and patient inputs remain local under ignored `data/` and `inputs/`; a Git checkout alone is not a complete runnable dataset.

## Build and run

Use a CUDA-capable SYCL compiler/plugin. For RTX2080Ti:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_ENABLE_MINIBEAM=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=ON -DCARBON_DOSE_FP64=OFF
cmake --build build -j 6
python3 scripts/check_electron_packages.py
```

Use the architecture of the actual GPU when building for another device. The latest measured implementation is committed as `ebff981`; its frozen executable is `out/broadbeam_ions_20261005/shared_v2_carbon_mc`. It includes the earlier secondary elastic-guard repair (`5cc096d`). The guard-only and pre-cleanup executables are historical controls, not the latest baseline. Local `out/` binaries are not distributed by Git; rebuild the recorded source and preserve the effective configuration when reproducing a result.

Run continuous batches in one process:

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

Use a new output directory. Each batch has its own seed, dose and quality report. Packages and immutable geometry are reused in the same process. Plan weights are normalized across the total requested histories; do not multiply an already normalized plan dose by the batch count again. To run distinct configurations sequentially with a shared session, use `--plan-manifest FILE`; `--plan-preflight` checks the manifest before transport.

## Verification and throughput work

For nonzero DTA, local gamma at reference position r is the minimum over evaluated positions x of `sqrt((|x−r|/DTA)² + ((D_eval(x)−D_ref(r))/(δ × D_ref(r)))²)`, where δ is the fractional dose criterion. The pass fraction counts eligible reference points with gamma ≤1. The current broadbeam implementation uses Cartesian candidate offsets inside the search sphere; the historical minibeam radial evaluator is identified separately below. Search spacing is 0.1 × DTA.

The four CT presets use these acceptance gates:

| CT preset | Dose gate | Gamma search step |
|---|---|---|
| Proton and carbon broadbeam | Local 1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local 3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | Dose gate deferred | Report separately if evaluated |

Freeze the reference, dose normalization, coordinate mapping, dose threshold and evaluation mask before comparison. Report them with the result. The current RT07575 broadbeam comparison uses the full 3D evaluation region above 10% of reference maximum, absolute normalization and no dose fitting or registration. A smaller search step is not permission to change the mask or denominator. Zero-DTA gamma remains a noise-sensitive diagnostic.

The following measurements predate the step-default update. CT broadbeam keeps the measured 0.5 mm setting; minibeam rates and gamma below are historical, not measurements at the new 0.1 mm cap. RTX2080Ti broadbeam results use two independent seeds and pooled dose; minibeam rows retain the preceding larger-statistics study:

| Preset | Histories | Resident primary/s | TOPAS local gamma |
|---|---:|---:|---|
| Proton CT broadbeam | 2 × 12,531,500 | 203,021 | 1%/1 mm: **99.8935%**, pass |
| Carbon CT broadbeam | 2 × 6,481,909 | 138,019 | 1%/1 mm: **99.8699%**, pass |
| Proton CT minibeam | 20,000,000 | 51,821 | Projected 2D 3%/0.2 mm: **98.3183%**, below gate |
| Carbon CT minibeam | Two seeds, each 3 × 200,000 per arm | **10,048**, control 5,165 | Accuracy deferred |

Latest matched controls give proton 201,356→203,021/s (+0.8%, small) and carbon 129,244→138,019/s (+6.8%). Both pooled gamma values are identical to their same-seed controls. The common EM preparation now reuses certified same-step search indices across density endpoints; normal interpolation, loss/RNG and transport are unchanged. Grid certification runs once when the immutable package loads. See the [report](docs/broadbeam_em_search_20261005.md) and [results](docs/broadbeam_em_search_20261005.json).

The [preceding electron optimization](docs/broadbeam_300k_20261005.md) established birth ordering and tissue survival0.5, with +22.8%/+22.0% resident gains and improved BODY statistical efficiency. Its earlier rates/gamma use different seeds and timing conditions; they must not be compared directly to infer an extra gain. The current work preserves that estimator. 300k/s is still not reached. Minibeam table entries retain the earlier larger-statistics study; new three-batch exact-work screens give proton49,147→50,524/s and carbon9,965→10,429/s, without a new gamma acceptance.

The proton minibeam figure uses the historical PyMedPhys 0.41.0 radial search (`interp_fraction=10`) on the BODY-projected lung dose, |y|<15 mm, |x|<15 mm, reference ≥3% maximum. It is not a 3D voxel or RT07575 full-plan acceptance. A Cartesian-lattice cross-check gives 97.1013% at the same 0.02 mm spacing; neither evaluator passes >99%. Peak and valley strips give 99.9605% and 99.0052% with PyMedPhys, but do not replace the full-field gate. The shared EM search update preserves this dose conclusion; the new minibeam checks establish implementation consistency, not a new gamma acceptance.

Carbon CT minibeam now uses `history_chunk_size` and `material_electron_queue_chunk` of 131,072, with existing species grouping and 16-step secondary continuation enabled. Group32, 16,384 electron lanes, the 576 MiB queue budget, physical steps and per-history RNG stay fixed. Same-binary reverse-order confirmation gives **1.946×** resident throughput, primary kernel 22.320→7.565 s and secondary 6.917→3.498 s per 200k batch. All compared discrete transport/EM/electron fields match; maximum dose relative L1 is 5.01e-8. Queue demand peaks at 3,798,118 / 5,807,497 tasks for this scene. This validates scheduling consistency, not carbon minibeam dose accuracy.

The shared-worker count, compact failure state, C12 specialization and smaller secondary-tail candidates did not deliver a resolved useful gain and were not promoted. Actual worker profiling showed substantial local-memory traffic and about 11 active threads per issued warp instruction; reducing static registers alone did not improve runtime. See [the full report](docs/ct_throughput_20261005.md) and [machine-readable results](docs/ct_throughput_20261005.json).

The EM search update completes 68 full broadbeam validation batches, 12 CT minibeam comparison batches and two-batch regression for each of the other six presets. All compared ion/EM/electron counts match. Broadbeam dose relative L1 is at most4.823e-8; the other-presets smoke maximum is1.081e-7. Quality, energy, completion, coverage and zero-overflow checks pass. These small runs are not water gamma or steady-state throughput measurements. Water birth extraction used 64 native histories per projectile and needs a separate dose/statistical campaign. The earlier cleanup control-flow regression and its larger-statistics repair are recorded in the [cleanup correction](docs/repository_cleanup_20261005.md#larger-statistics-correction).

Measure cold start, first batch and resident batches separately. Report primaries/s, kernel times, task counts and peak memory with GPU model, compiler, source, seed and history counts. Compare like-for-like source histories, not electron tasks or accepted collimator primaries. `scripts/benchmark_ct_devices.py` uses native repeated batches for all four CT presets, validates each batch, and supports `--smoke-only`; the reproduction command is in the CT report.

Search [failed.md](failed.md) before each candidate; require documented new evidence before retrying a rejected route. FP32 atomic order differences alone do not reject scheduling changes, but energy, transport/EM, completion and queue-overflow audits must pass. TOPAS references and historical experimental evidence are retained. Water performance and dose campaigns remain deferred until CT validation.
