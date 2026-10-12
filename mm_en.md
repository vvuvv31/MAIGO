# II. Materials and Methods

**Manuscript theme: GPU Monte Carlo: From Broadbeam to Minibeam.** [中文版](mm_zh.md). Revised October 12, 2026.

This study implements proton and carbon-ion broadbeam dose calculation within a shared GPU Monte Carlo framework and extends the framework to minibeams formed by a copper collimator. The methods are presented in three stages: the broadbeam transport foundation, the geometric and physical extensions required for minibeams, and a common evaluation of dose agreement and computational efficiency. The dosimetric endpoint is absorbed dose to medium.

## II.A. Study design: from broadbeam to minibeam

### II.A.1. Shared computational framework and scope

MAIGO is a GPU Monte Carlo dose engine implemented in C++20 and SYCL, with charged-particle transport executed through the CUDA backend. Primary particles are restricted to protons (Z=1, A=1) and carbon ions (Z=6, A=12). Charged fragments produced by nuclear interactions are transported according to their own species. The engine combines condensed-history electromagnetic transport with physics data and correlated nuclear final-state libraries extracted from TOPAS/Geant4, thereby avoiding online intranuclear-cascade calculations on the GPU. Offline physics-data generation is separate from patient dose calculation.

Within a common physics and scoring framework, patient broadbeam transport provides the foundation, and explicit copper-collimator and air-gap transport extends the calculation to minibeams. Particle type, medium, and irradiation mode define eight configurations: all combinations of proton/carbon, water/CT, and broadbeam/minibeam. Four dedicated pipelines serve proton broadbeam, proton minibeam, carbon broadbeam, and carbon minibeam configurations; a generic pipeline supports other proton or C12 configurations. The pipelines share geometry, electromagnetic, nuclear, and scoring operations. A pipeline's particle designation identifies the primary source and does not restrict its charged-fragment species.

The study evaluates physical absorbed dose. LET, biological dose, and relative biological effectiveness are outside the endpoints of this methods draft. Nuclear neutral products are not explicitly transported; their energy is recorded in separate untracked-energy accounts and is not substituted with local dose. Explicit electron transport is optional and includes electromagnetic photon transport when enabled.

### II.A.2. Physics-data generation and coverage

The extraction tools correspond to TOPAS 4.2.3 and Geant4 11.3.2. Each comparison records the actual reference executable version, physics modules, package hashes, source, and geometry. Physics-list names alone are not used to establish process equivalence. Extraction tools, build scripts, and module descriptions are provided in `extensions/`. Production data are organized by materials, electromagnetic interactions, nuclear reactions, elastic interactions, stopping powers, scattering, and ionization.

Packages explicitly identify species, materials, and valid energy ranges. Before transport, the engine checks formats, metadata, required channel support, and coverage. Missing data or queries outside the permitted domain are not replaced by neighboring materials, retired density nodes, or local deposition. Installed assets are fixed by a SHA256 inventory, and configuration dependencies are listed in `docs/preset_dependencies.json`. Each simulation preserves its configuration, integer history allocation, random seeds, executable hash, and quality reports.

## II.B. Broadbeam GPU transport foundation

### II.B.1. Water and heterogeneous-medium representation

Broadbeam and minibeam calculations share the medium representation. Reference water is Water_75eV, with a mean excitation energy of 75 eV. Patient HU values are mapped to actual mass density and one of 25 tissue compositions using the Schneider method[1]. Together with reference water, these define 26 electromagnetic compositions. Electromagnetic properties are evaluated at the actual density using continuous-density laws. Voxel geometry, density, and material labels are prepared offline and read by the runtime.

### II.B.2. Continuous energy loss, fluctuations, and stepping

All configurations use the continuous26 electromagnetic package. It contains species- and composition-specific native restricted stopping powers, ranges, inverse ranges, fluctuation parameters, and hard-ionization parameters, evaluated at the actual CT density. The implementation retains explicitly calibrated approximations for hard rates and conditional final states. Extraction from Geant4 does not establish step-by-step equivalence with Geant4 transport.

For each step, the electromagnetic state is prepared from material, density, species, and kinetic energy. Geometric boundaries, the configured maximum step, native electromagnetic limits, mean-energy-loss constraints, nuclear interaction distances, and any discrete proposal clocks enabled by the pipeline then compete to determine the step. Small restricted losses are calculated from stopping powers with the corresponding stepping corrections; larger losses use range inversion. Restricted losses and above-threshold delta-ray losses use their respective data without counting the same energy twice.

Release configurations retain primary energy-loss fluctuations and disable restricted continuous and aggregated delta-loss fluctuations for charged secondaries in water and patient media, using the corresponding mean losses. Secondary transport, scattering, and nuclear reactions remain. Termination energies, nuclear-generation limits, and other stepping parameters are stored with each simulation configuration. Scoped paired validation has been completed for RT07575 CT proton broadbeam, proton minibeam, and carbon broadbeam; carbon minibeam is under verification. These results do not establish dose acceptance for water or other patients. See [secondary-straggling validation status](docs/secondary_straggling.md).

For primaries with fluctuations enabled, the fast broadbeam pipeline aggregates above-threshold ionization losses into a single stochastic energy transfer per step, retaining the mean and variance together with the applicable restricted continuous fluctuations. For nonzero mean μ and variance v, the moment-matched Gamma approximation uses shape μ²/v and scale v/μ. It does not preserve the zero-collision probability or higher moments of individual electron collisions. Energy-dependent stepping corrections accompany the restricted-loss calculation; eliminating individual collision sampling does not eliminate ionization energy loss.

The dedicated proton CT broadbeam primary path with local electron-energy deposition uses the step-start delta first moment while retaining the restricted-loss slope, current rate, variance, and fluctuations. Nearby-energy evaluations of the delta slope are omitted. This approximation applies only to the corresponding dedicated path and is not automatically applied to secondaries or other paths. Runtime audits identify whether it was executed.

Broadbeam transport uses a Gaussian Fermi–Eyges approximation to sample angular deflection and lateral displacement jointly from scattering moments[2]. Multiple Coulomb scattering and nuclear elastic scattering are queried and sampled separately. The current proton and carbon CT broadbeam configurations both use a 5% fast mean-energy-loss limit. Maximum ion steps are uniformly 1.0 mm for broadbeam and 0.5 mm for minibeam, with a 0.05 mm copper-collimator cap. Configuration loading overrides the corresponding YAML values.

Broadbeam transport establishes the shared patient-transport foundation. Extending it to minibeams requires the outgoing distribution generated by the collimator and the submillimeter lateral dose structure within the patient to be represented together. These requirements motivate the medium-specific stepping and scattering treatment described in Section II.C.

### II.B.3. Nuclear interactions and charged fragments

Inelastic rates are constructed from the elemental composition and actual density. Particles consume sampled nuclear optical depth along their paths. At a candidate interaction, the target element is selected according to its partial rate. A correlated event is sampled from finite final-state libraries at neighboring energy nodes and rotated as a whole into the incident-particle frame, preserving within-event correlations between product energies and directions. Supported charged products enter secondary queues and continue electromagnetic transport and permitted subsequent nuclear reactions.

Release configurations limit patient secondary inelastic generations to two. Reaching the limit prevents new nuclear branches but does not remove electromagnetic transport of existing charged fragments. Species explicitly assigned an electromagnetic-only nuclear policy follow that policy. Untracked neutral energy, locally scored energy, and outgoing energy are recorded separately.

Independent all-ion elastic packages contain reaction rates, target isotopes, and momentum-transfer samples. Scattering and recoil states are generated at the current energy using two-body kinematics. Registered species continue normal transport; additional target recoils use the corresponding recoil stopping-power path. The four proton configurations and two carbon minibeam configurations currently load independent elastic/recoil packages; the two carbon broadbeam configurations do not. Actual elastic behavior therefore depends on both the loaded package and executed path, rather than a single elastic boolean setting.

## II.C. Extension from broadbeam to minibeam

### II.C.1. Copper collimator and air gaps

Relative to broadbeam irradiation without a copper slit collimator, minibeam calculations introduce an upstream high-density medium, slit sidewalls, and air gaps. Explicit upstream transport preserves the joint outgoing distribution of position, direction, energy, and particle species before transport continues within the patient.

Minibeams are formed using a copper slit collimator. Its geometry specifies copper thickness, slit width, pitch, number, length, rotation, source-to-collimator distance, and collimator-to-patient distance. Copper, slit air, and the downstream air gap are distinct media. Material boundaries constrain particle steps at copper–slit interfaces. Dimensions are taken from each simulation's inputs; identical geometry is not assumed across all release configurations or patient plans.

For example, the 20022516 proton minibeam full plan uses a 60 mm copper thickness, 15 slits, a 0.5 mm slit width, a 3.6 mm pitch, and a 50 mm slit length. The source plane is 450 mm upstream of isocenter, and the collimator-center-to-isocenter distance is approximately 192.361 mm. These dimensions apply specifically to that case's validation inputs.

Upstream transport includes primary particles in copper, primaries scattered through copper, and charged nuclear products. Tracks are not removed solely because their mean range appears insufficient to reach the downstream face: lateral entry into a nearby slit remains possible. Absolute dose is normalized to the original incident particles, rather than particles transmitted through the collimator or entering the patient.

### II.C.2. Minibeam scattering, fluctuations, and boundaries

Minibeam calculations use material- and species-specific data from the Geant4 Urban model, including scattering mean free paths, ranges, and geometric/true-path treatment. Patient, copper, and air transport use their corresponding medium data. The CT Urban path retains its current density-scaling approximation. Scattering updates are performed in a local orthonormal particle frame and transformed back to world coordinates.

Primary energy-loss fluctuations in copper remain enabled; restricted fragment fluctuations are disabled by default. Proton minibeams use `g4_universal_delta`, whereas carbon minibeams use `g4_ion_delta`. Carbon copper fragments retain hard-delta Poisson and spectrum sampling; disabling restricted fluctuations does not remove all stochastic energy transfer. Copper and air retain the corresponding native proposal clocks and material-boundary constraints. Patient steps are also limited by electromagnetic, scattering, and voxel-boundary constraints; the configured maximum step is therefore not the mean realized step.

| Medium and irradiation mode | Maximum primary/ion step | Scattering path | Default explicit electron transport |
|---|---:|---|---|
| CT broadbeam, proton and carbon | 1.0 mm | Gaussian Fermi–Eyges | Off |
| Water broadbeam, proton and carbon | 1.0 mm | Gaussian Fermi–Eyges | Off |
| CT minibeam, proton and carbon | 0.5 mm | Urban | Off |
| Water minibeam, proton and carbon | 0.5 mm | Urban | Off |
| Minibeam copper collimator | 0.05 mm | Copper-specific scattering and energy loss | Follows the corresponding electron configuration |

The combined minibeam mean-loss limit and native stepping parameters are determined by the executed branch. A general relative-energy-loss YAML key is not assumed to control all pipelines identically. Carbon minibeam research configurations using a 1 mm patient cap or scaled ionization-segmentation clocks are recorded separately and are not release defaults. Disabled secondary fluctuations are now the default, but carbon-minibeam dose validation remains incomplete. A broadbeam Gaussian-scattering hypothesis is not used as a direct explanation of Urban minibeam discrepancies.

### II.C.3. GPU decomposition of upstream and patient transport

All four minibeam configurations use compact scheduling to separate copper/air transport from transport inside the medium. Primary position, direction, energy, and random identity are passed across the handoff surface, so the source distribution downstream of the collimator is generated by upstream transport. In proton and carbon CT minibeams, charged children produced by primaries in copper are processed by a separate queue, preserving their source, product, and random identities. Carbon water minibeams retain their existing medium path.

This decomposition permits upstream and patient costs to be measured separately and execution pipelines to be optimized by primary species and irradiation mode. It does not replace the upstream calculation with a source containing only direct slit-transmitted primaries, and it does not introduce fixed phase-space reuse.

### II.C.4. Optional spatial electron transport

Minibeam peak–valley structure and heterogeneous interfaces require separate assessment of the local ionization-energy-deposition approximation. The framework retains the following optional electron/photon paths to evaluate spatial redistribution in both broadbeam and minibeam calculations.

Explicit electron transport is enabled with `enable_electron_transport: true`. When disabled, electron/photon packages are not loaded, and ionization energy follows the existing local-deposition path. This is a spatial-dose approximation, not removal of electromagnetic energy loss from the ions. Enabling transport requires complete, projectile-matched strict26 birth, charged-particle, photon, and spatial joint-response packages.

The electron algorithm uses energy-packet estimators, birth merging, and energy-weighted successor sampling to reduce the cost of transporting complete electron trees. Charged collisions, atomic relaxation, elastic scattering, bremsstrahlung, and electromagnetic photon interactions use the corresponding composition data. Applicable homogeneous-medium segments are accelerated using spatial joint responses. Joint responses remain subject to package-validity and geometric-step constraints; electron geometry, energy loss, and scoring retain FP32 arithmetic.

The optional CT broadbeam electron path additionally uses selective tracking in low-density regions and their neighborhood. The default research selection comprises densities no greater than 0.1 g/cm³ and a 5 mm halo, combined with birth sampling. Region selection requires an additional explicit setting. Ionization by patient nuclear secondary ions retains the current local-deposition policy. Uniform-water broadbeam configurations do not define a corresponding electron-selection region; enabling one master switch alone does not establish complete spatial electron transport.

Electron-on/off comparisons use identical sources, weights, geometry, and incident-particle counts. Paired seeds isolate implementation differences, while independent batches estimate dose variance. Nuclear and elastic integer counters verify consistency of the upstream ion path when electron transport is enabled. This design measures spatial electron redistribution; small local differences are not generalized to all materials or all minibeam valleys.

## II.D. Shared GPU execution and dose scoring

### II.D.1. Continuous batches, random numbers, and queues

Primaries are launched in internal history chunks. Charged fragments may undergo species grouping and segmented continuation according to configuration-specific scheduling settings.

Continuous batches reuse one GPU context and physics-package cache, reducing initialization and transfer overhead. Total histories per batch and internal kernel chunk size are controlled separately. Increasing the former primarily reduces external batch, output, and initialization counts and does not guarantee faster individual kernels. Secondary and copper-product queues have explicit capacities. Overflow invalidates the batch; products are not truncated before publishing dose.

Random streams use counter-based Philox and are tied to spot, history, and descendant-track identities, with state retained across continuations. Matched implementation checks use identical source allocations and seeds. Statistical studies use independent seeds; repeated runs with the same seeds are not counted as additional statistics.

### II.D.2. Plan-weight allocation and absolute dose

Let W_j denote physical incident particles per fraction for spot j, W=Σ_j W_j the total plan weight, and N the total simulated histories in a GPU campaign. Ideal allocations N W_j/W are converted to integers n_j using the largest-remainder method, ensuring Σ_j n_j=N. Allocation is performed once for the entire campaign, after which each spot's n_j is balanced across continuous batches. Zero-weight spots receive no histories.

Deposited energy is accumulated using FP32 atomics on the GPU. Voxel mass and dose are

\[
 m_i=\rho_i V_i\times10^{-6}\ \mathrm{kg},\qquad
 D_i=\frac{E_{\mathrm{dep},i}\times1.602176634\times10^{-13}}{m_i}\frac{W}{N},
\]

where ρ is in g/cm³, V in mm³, and deposited energy in MeV. Full-plan dose is obtained by summing batch contributions with the same W/N scale; it is not multiplied again by the number of batches. Scoring and postprocessing retain simulated energy and absolute dose scaling separately. Files may be accumulated at higher precision during postprocessing without changing GPU FP32 atomic scoring.

## II.E. Validation from basic models to patient full plans

### II.E.1. Geometry, patients, and evaluation grids

Water and patient CT studies separately assess homogeneous-medium transport, collimator effects, and dose redistribution in heterogeneous media. Broadbeam evaluation addresses patient-interior dose, whereas minibeam evaluation additionally examines lateral peak–valley structure at different depths. Single-spot probes help distinguish model discrepancies from reference statistical noise.

Reference water is Water_75eV, with a mean excitation energy of 75 eV. CT HU values are mapped to mass density and one of 25 Schneider tissue compositions[1], giving 26 electromagnetic compositions together with water. Offline preprocessing generates a binary grid containing voxel spacing, origin, density, and material labels. The runtime reads this grid directly and does not perform DICOM conversion. Actual density enters the continuous-density electromagnetic evaluation; composition effects are not replaced by a single density scaling of water stopping powers.

Patient and reference dose grids are aligned in physical coordinates, with axis order, spacing, and origin explicitly recorded. BODY, target, and organ-at-risk masks are generated from case-specific structure data and preserved with the dose grid. No dose fitting, peak alignment, or post hoc spatial registration is performed. Where CT and scoring grids coincide, a directed voxel index may be reused for material access while preserving voxel-face ownership.

Patient full-plan validation uses RT07575, RT06423, and 20022516. RT07575 is a head case; RT06423 and 20022516 are used for heterogeneous lung studies. The flat-source lung benchmark in the release CT minibeam configurations is distinct from patient full plans and is not reported as RT07575 head validation. Each full plan binds CT, source, weights, placement, collimator, and absolute normalization as one input set.

### II.E.2. Beam sources and irradiation plans

Plan sources specify spot positions, energies, and incident-particle weights. Central input energies are expressed in MeV/u; TPS CSV energies are total kinetic energies in MeV, with conversion according to A. The beam-optics model supplies position and angular standard deviations, correlations, and energy spread, interpolated at each spot energy. Correlated position and direction are sampled in an orthonormal beam frame and transformed to world coordinates using scan-magnet positions, the source plane, and patient placement.

The RT07575 carbon minibeam optimized plan uses two crossfiring fields at 90° and 270°. Other cases and modes retain their own spot distributions, energy layers, and beam-optics models; head-plan source parameters are not transferred to lung plans. Optical CSV data are treated as simulation inputs and are not described as measured beam calibration without independent measurement evidence.

### II.E.3. Physical dose influence matrices and optimization

TOPAS spotwise simulations generate the physical dose influence matrix (DIJ). Each column preserves its simulated history count, spot index, and dose units before conversion to dose per incident particle. Assembly records whether sparsification or threshold truncation was applied. Truncation before weighting can remove substantial low-dose tails; an unchecked truncated matrix is therefore not used as a complete minibeam dose reference.

For influence matrix A and nonnegative incident-particle weights w, plan dose is d=Aw. Proton and carbon optimization generally use the MATLAB entry points `run_admm_physical_p.m` and `run_admm_physical_c.m`, respectively, with ADMM and conjugate-gradient subproblems for case-specific target and organ-at-risk objectives. A retained case-specific Python optimization workflow is labeled separately. ROI mappings, objective types, weights, dose limits, volume parameters, and stopping settings are archived with the plan.

The recent preliminary lung plans described here use ten ADMM iterations and normalize target D95 to 4 Gy per fraction over 16 fractions. Ten iterations constitute a fixed computational budget and do not establish convergence; convergence flags and target uniformity are reported separately. Exported weights represent physical incident particles per fraction, rather than MU or GPU batch multipliers.

### II.E.4. TOPAS references and statistical design

TOPAS references use the corresponding source model, CT materials, geometry, and DoseToMedium scoring. Actual activation of `g4em-standard_opt4`, hadronic, ion-inelastic, nuclear-elastic, stopping, and decay modules is recorded[3,4]. Minibeam references record `AllChargedUrbanPhysics` to establish the charged-particle Urban configuration. Matching carbon nuclear elastic physics requires checking actual GenericIon elastic-process registration; an ordinary hadronic-elastic module name is insufficient. Historical carbon references missing this process are labeled as incompletely matched.

Two reference types are distinguished: the original TOPAS DIJ weighted by optimized spot weights, and an independent TOPAS full-plan simulation rerun with those weights. The former shares its noise realization with optimization and is useful for algorithm checks, but does not replace independent recalculation. Untruncated dose and dose truncated separately per spot are also distinguished. Reference provenance accompanies each results table.

The 20022516 proton minibeam statistical-sensitivity study uses an unthresholded TOPAS DIJ comprising 1216 spots with 10⁶ histories per spot; 451 spots have nonzero optimized weights. GPU campaigns simulate 366,360,520 and 10⁹ incident protons. The 10⁹ campaign consists of 50 near-equal batches of approximately 2×10⁷ histories, with the same plan and physics settings and independent seeds. Three major contributing spots are additionally simulated with independent TOPAS runs of 2×10⁶ histories and variance output, and GPU runs of 10⁷ histories per spot. Single-spot analysis uses 10% of that spot's reference maximum, explicitly distinguished from the full-plan prescription threshold.

## II.F. Dose agreement and statistical uncertainty

### II.F.1. Gamma and regional dose metrics

Three-dimensional local gamma is evaluated at reference position r[5]:

\[
 \gamma(\mathbf r)=\min_{\mathbf r'}\sqrt{
 \frac{\|\mathbf r'-\mathbf r\|^2}{\Delta r^2}
 +\frac{[D_{\mathrm{GPU}}(\mathbf r')-D_{\mathrm{ref}}(\mathbf r)]^2}
 {[\delta D_{\mathrm{ref}}(\mathbf r)]^2}}.
\]

GPU dose is queried by trilinear interpolation, and γ≤1 is counted as passing. Broadbeam evaluation uses local 1%/1 mm with 0.1 mm search spacing and a prespecified pass-rate requirement greater than 99%. Minibeam evaluation primarily uses local 3%/0.2 mm with 0.02 mm search spacing, supplemented by 3%/0.1 mm and same-voxel 3%/0 mm comparisons. Zero DTA is evaluated as a same-voxel dose comparison. A numerical grid search is not an analytic continuous minimum.

The main full-plan minibeam evaluation includes BODY voxels with reference dose at least 10% of the prescription: 0.4 Gy for a 4 Gy fraction. The local dose tolerance remains 3% of each reference voxel's dose, rather than 3% of prescription. Some historical RT07575 carbon studies use 0.12 Gy (3% of prescription) or dedicated irradiation masks and are labeled separately. Existing broadbeam acceptance records use BODY voxels above 10% of the reference maximum. Pass rates from different thresholds are not treated as interchangeable.

Evaluation counts and pass rates are reported separately for BODY and target. Target mean dose, D95, D2, dose differences, and local relative-difference RMS are also calculated. Lateral minibeam peaks and valleys and depth-specific one-dimensional profiles retain their original dose scale. BODY restricts reference evaluation points; GPU dose outside BODY is not artificially set to zero within the search neighborhood.

### II.F.2. Batch variance and reference noise

For B independent full-plan batches with near-equal history counts, let d_b be the dose contribution of batch b, already normalized by the full campaign's W/N. Total dose is D=Σ_b d_b, with estimated variance

\[
 \widehat{\sigma}_D^2=\frac{B}{B-1}
 \left[\sum_b d_b^2-\frac{(\sum_b d_b)^2}{B}\right].
\]

Within a fixed ROI, the ratio of voxel standard error to reference dose is summarized by its RMS, median, and percentiles. This estimate concerns GPU transport sampling noise and excludes TOPAS reference noise, finite nuclear-library uncertainty, and model bias.

Where available, repeated TOPAS simulations output Sum and Variance. If H is the spot history count and Variance denotes the reported single-history sample variance, mean dose per incident particle is Sum/H and its variance is Variance/H. For independently simulated spots, weighted reference dose and voxel variance are

\[
 D_{\mathrm{ref},i}=\sum_j W_j\bar d_{ij},\qquad
 \sigma_{\mathrm{ref},i}^2=\sum_j W_j^2\frac{V_{ij}}{H_j}.
\]

The variance convention is checked against the actual scorer header, and dose and variance columns are stored separately. For independent GPU and reference simulations, voxel variance of their difference is σ_GPU²+σ_ref². Spatial correlation prevents sums of voxel variances from directly yielding standard errors for regional means, DVH metrics, or gamma pass rates. Those endpoints require independent batches, full recalculations, or appropriate resampling.

Noise attribution combines independent TOPAS-versus-TOPAS repeats, independent GPU batches, and weighted libraries with fixed sampling per spot. Spatial smoothing is diagnostic only and does not modify formal dose or gamma calculations. Predictions using GPU noise as a TOPAS proxy explicitly state the assumed similarity of per-history variance and spatial covariance; predicted pass rates are not reported as measured acceptance.

## II.G. Computational efficiency and runtime quality control

Performance is reported separately for the actual NVIDIA RTX 2080 Ti, RTX A6000, and other devices used. GPU model, memory, driver, compilation target, executable hash, and concurrent workload are recorded. Cold-start time extends from process launch through the first complete batch dose output and includes initialization and data transfer. Subsequent-batch throughput divides subsequent incident histories by their corresponding runtime while reusing the same GPU context. Filesystem-cache conditions and portable PTX versus device-specific AOT builds are identified.

Paired optimization studies use identical sources and seeds in A/B/B/A or other interleaved orders. The first batch may be excluded from warm timing while retaining its physics audits. End-to-end time and primary, secondary, electron, and upstream-collimator stage times are reported separately. Temperature and frequency drift under unlocked clocks are measurement limitations; startup smoke checks do not establish performance gains. When sampling or physical approximations change, time to a comparable statistical error is evaluated in addition to incident histories per second.

Each batch is checked for source completion, charged-species terminal states, electromagnetic and nuclear coverage, finite nonnegative dose, energy accounting, queue completion, and zero overflow. Matched GPU implementation checks and independent TOPAS physics acceptance are reported separately. Proton minibeam electron-package replacement uses a decrease of no more than one percentage point in local 3%/0.2 mm pass rate against a matched old baseline as an implementation-related criterion; absolute TOPAS gamma is evaluated separately. Independent absolute carbon minibeam dose acceptance remains incomplete, and passing matched scheduling checks does not replace that assessment.

## References

1. Schneider W, Bortfeld T, Schlegel W. Correlation between CT numbers and tissue parameters needed for Monte Carlo simulations of clinical dose distributions. *Phys Med Biol.* 2000;45:459–478.
2. Eyges L. Multiple scattering with energy loss. *Phys Rev.* 1948;74:1534–1535.
3. Perl J, Shin J, Schümann J, Faddegon B, Paganetti H. TOPAS: An innovative proton Monte Carlo platform for research and clinical applications. *Med Phys.* 2012;39:6818–6837.
4. Agostinelli S, et al. Geant4—a simulation toolkit. *Nucl Instrum Methods Phys Res A.* 2003;506:250–303.
5. Low DA, Harms WB, Mutic S, Purdy JA. A technique for the quantitative evaluation of dose distributions. *Med Phys.* 1998;25:656–661.

## Reproducibility materials (to be organized as supplementary material)

Physics-package generation is described in [extensions/README.md](extensions/README.md), pipeline architecture and selective compilation in [docs/transport_pipelines.md](docs/transport_pipelines.md), electron opt-in behavior in [docs/optional_electron_transport_20261011.md](docs/optional_electron_transport_20261011.md), and installed data layout in [docs/data_layout_20261007.md](docs/data_layout_20261007.md). Performance candidates and unsuccessful experiments are recorded in `failed.md` and the corresponding study documents.

Before submission, each study requires a frozen executable version, configuration, package hashes, and input inventory, together with patient-data provenance and ethics approval or exemption information; case-specific energy ranges, fields, spots, and collimator parameters; and actual independent-reference history counts and statistics. Unconfirmed ethics information, measured clinical beam calibration, or independent high-statistics acceptance is not represented as completed. Measured gamma, throughput, and noise belong in Results. Model applicability, finite event libraries, and the effects of untracked nuclear neutral products belong in Discussion.
