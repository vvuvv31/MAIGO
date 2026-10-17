# II. Materials and Methods

Manuscript theme: GPU Monte Carlo: From Broadbeam to Minibeam. [中文版](mm_zh.md).

## II.A. Study design, materials, and beam inputs

### II.A.1. Computational framework and scope

MAIGO was developed as a GPU Monte Carlo dose engine for proton and carbon-ion broadbeam irradiation and extended to minibeams formed by a copper slit collimator. The engine uses data-driven condensed-history charged-particle transport, combining frequent soft interactions into transport steps and sampling nuclear interactions and the applicable discrete processes stochastically. Separating the physical model from its parallel implementation is also characteristic of established proton and carbon-ion GPU Monte Carlo engines[1–4]. The extension studied here concerns continuous transport through the collimator, air, and patient, together with evaluation of submillimeter lateral dose structure. Upstream collimator modeling and verification are treated separately from transport within the patient[5].

Incident particles are protons (Z=1, A=1) or C12 ions (Z=6, A=12). Supported charged nuclear fragments are transported subsequently. The scored quantity is absorbed dose to medium; LET, RBE, and biological dose are outside the present scope. The combinations of projectile, water or CT medium, and broadbeam or minibeam irradiation define eight standard configurations. Patient full-plan calculations retain their case-specific sources, weights, and geometries rather than substituting the flat-source phantom configuration.

### II.A.2. Material representation and physics data

Patient CT images are converted offline to a regular voxel grid containing its origin, voxel spacing, mass density, and material identifiers. Hounsfield units are mapped to density and one of 25 tissue elemental compositions using the Schneider method[6]. A Water_75eV reference with a mean excitation energy of 75 eV provides the twenty-sixth electromagnetic composition. Electromagnetic parameters are evaluated for the tissue composition and actual density, including density-dependent effects; tissues are not represented solely by mass-density scaling of water.

Physics data are generated with offline tools based on OpenTOPAS 4.2.3 and Geant4 11.3.2[7,8]. They include restricted ion stopping powers, ranges and inverse ranges, energy-loss fluctuation and hard-ionization parameters, multiple-scattering parameters, nuclear interaction rates, and final-state data. The continuous-density ion electromagnetic tables are termed continuous26. Nuclear inelastic interactions are represented by finite correlated event libraries; nuclear elastic data contain total rates, target isotopes, and momentum-transfer samples. The GPU queries these data rather than evaluating the Geant4 intranuclear cascade online.

Each package specifies its projectile, composition, and energy coverage. Required channels and query ranges are checked before simulation; another material or local energy deposition is not substituted for missing required physics. Inputs, physics-data checksums, program version, source allocation, and seeds are retained for each calculation. Extraction and reference-dose versions are recorded separately because a shared data origin does not imply stepwise-equivalent transport.

### II.A.3. Spot sources and physical-dose weights

Scanning plans specify spot positions, energies, and weights for each field. An energy-dependent beam model supplies position and angular standard deviations, position–angle correlations, and energy spread. Correlated positions and directions are sampled in a local beam frame and transformed to world coordinates using the source plane, scanning-magnet parameters, field orientation, and patient placement. Ion energies are expressed in MeV/u; plan inputs stored as total kinetic energy in MeV are converted using the mass number.

For reoptimized physical-dose plans, A denotes the dose-influence matrix per incident particle and w the nonnegative spot particle weights, giving d=Aw. Case-specific target and organ-at-risk objectives determine the weights. Optimization defines the common irradiation source: GPU and independent TOPAS recalculations use the same final weights, rather than separately optimized weights. Historical carbon broadbeam plans retaining their original TPS weights are distinguished from reoptimized plans and are not collectively described as reoptimized 4 Gy plans.

## II.B. Shared ion transport and broadbeam calculation

### II.B.1. Step selection, mean energy loss, and fluctuations

Each charged track carries position, direction, kinetic energy, species, and random identity. Electromagnetic parameters are prepared from the current composition, density, and energy. Geometric boundaries, the maximum step, mean-loss limits, scattering-related limits, nuclear interaction distances, and applicable ionization proposal clocks compete to determine the accepted step. Geometry queries and scoring-voxel assignment are handled separately to distinguish material boundaries from scoring boundaries. The maximum step is therefore not the mean realized step.

Continuous loss is computed from composition-specific restricted stopping powers. Small losses use the corresponding stepping approximation; larger losses are obtained from residual-range inversion. Restricted losses and above-threshold delta energy transfers are evaluated separately to avoid double counting. All standard configurations retain primary energy-loss fluctuations. Charged secondaries in water and patient media use mean restricted and aggregate delta losses by default, while their motion, scattering, and permitted nuclear interactions remain active.

Aggregate delta transfers are sampled from their mean μ and variance v. Where the two-moment Gamma approximation applies,

\[
 k=\frac{\mu^2}{v},\qquad \theta=\frac{v}{\mu}.
\tag{1}
\]

This approximation preserves the first two moments but not the zero-collision probability or all higher moments of individual transfers. In the dedicated proton CT broadbeam path, the dense restricted-fluctuation tail with Poisson mean at least 4 is also replaced by a moment-matched Gamma draw; low-count tails retain explicit sampling. This treatment is not extended to carbon primaries or minibeams.

For dedicated CT broadbeam transport, proton primaries use step-start restricted stopping powers and delta first moments without their within-step energy-slope corrections. Carbon primaries retain both corrections. Step-start approximations for charged carbon fragments are controlled separately. The executed branches and existing calibration approximations for hard-interaction rates and conditional final states are retained with the calculation records.

Ionization energy in water and patient media is deposited locally in the corresponding scoring voxel, without explicit electron or electromagnetic-photon transport. This is a spatial dose approximation that omits nonlocal redistribution by ionization secondaries. Its applicability is assessed through heterogeneous-medium and minibeam dose comparisons, rather than inferred from agreement in mean energy loss. Hard-delta energy in copper follows the beamline removal convention described in Section II.C.

### II.B.2. Broadbeam scattering and stepping parameters

Broadbeam multiple Coulomb scattering follows a Gaussian Fermi–Eyges approximation, jointly sampling angular deflection and lateral displacement from scattering moments[9]. Coulomb scattering and strong-interaction nuclear elastic scattering are treated separately. The standard broadbeam ion step cap is 1.0 mm. The fast CT broadbeam path limits mean energy loss to 5% per step. These limits compete with geometry and interaction distances.

| Transport region | Maximum ion step | Principal scattering treatment |
|---|---:|---|
| Water/CT broadbeam | 1.0 mm | Gaussian Fermi–Eyges |
| Water/CT minibeam | 0.5 mm | Urban data-driven transport |
| Minibeam copper collimator | 0.05 mm | Copper-specific Urban and nuclear elastic scattering |

These engine-enforced step caps are distinct from TOPAS secondary-production range cuts and ion termination energies. Slit air in CT carbon minibeams has a separate 0.2 mm cap; other air regions follow their respective input limits and boundary handling.

### II.B.3. Nuclear competition and charged fragments

Macroscopic elastic and inelastic rates are formed from target-element number densities and the corresponding cross sections. Primary and charged secondary particles in water and patient media share a nuclear optical-depth algorithm. Each track independently samples

\[
 \tau=-\ln U,\qquad
 \Sigma_{\mathrm{tot}}=\Sigma_{\mathrm{el}}+\Sigma_{\mathrm{in}},
\tag{2}
\]

where U is uniform. Optical depth is consumed along accepted paths using the current step's rate approximation. At an interaction, elastic or inelastic scattering is selected in proportion to its rate, followed by target selection from target-resolved rates. Residual optical depth is preserved across material boundaries and scheduler continuations, avoiding resampling caused solely by geometry crossing or execution order.

An inelastic event is drawn using the package-defined energy nodes and correlated final-state library and rotated as a whole into the current incident direction. This retains correlations within the sampled event. Supported charged products enter transport queues. Patient-medium secondary inelastic generations are limited to two. Reaching this limit suppresses new inelastic branches but preserves electromagnetic and elastic transport of existing tracks. Species explicitly assigned an electromagnetic-only policy follow that coverage policy.

Nuclear elastic scattering uses sampled momentum transfer and two-body kinematics to generate the outgoing projectile and recoil. Kinematic masses follow the conventions of the corresponding packages, including the calibrated C12 A×amu convention, rather than interchanging them with nuclear-mass conventions. Registered charged species use the ordinary transport model; additional target recoils use their dedicated electronic-plus-nuclear stopping powers. Medium nuclear elastic scattering is enabled in all eight standard configurations, with copper nuclear elastic scattering also enabled in all four minibeam configurations. Nuclear neutral products are not transported. Their energy is recorded separately as untracked energy and is not deposited at the interaction site. The finite event library, secondary-generation limit, and omission of neutral transport remain model differences from full TOPAS transport.

## II.C. Extension to copper-collimated minibeams

### II.C.1. Collimator geometry and upstream transport

The minibeam extension adds a copper slit collimator, slit air, and the downstream air gap to the broadbeam patient-transport framework. Geometry inputs specify copper thickness and external shape; slit count, width, pitch, length, phase, and rotation; and relative source, collimator, and patient positions. The collimator is fixed in the field frame rather than translated with the scanning spot. Copper–air sidewalls and the patient entrance participate in geometric step selection.

The proton minibeam full plans use 60 mm of copper with 15 slits, a 0.5 mm slit width, 3.6 mm pitch, and 50 mm slit length. The RT07575 carbon minibeam plan uses 100 mm of copper with 17 slits, a 0.7 mm width, 3.0 mm pitch, and 100 mm full slit length. Its 90° and 270° crossfiring fields retain their individual slit-array placements; the nominal center array for the 270° field has a half-pitch phase offset relative to the 90° field. Other carbon plans use their actual executed geometry inputs rather than dimensions inferred from RT07575. Distances from the collimator exit to the CT bounding box, skin, and isocenter are defined separately.

Upstream transport includes directly transmitted primaries, primaries scattered through copper, and charged nuclear products. Their joint position, direction, energy, and species distribution at patient entry is generated by this transport. A source containing only direct slit transmission or a fixed reusable phase-space cache is not substituted for the collimator. Original histories without transmission remain in the incident-particle normalization.

### II.C.2. Minibeam electromagnetic transport and copper nuclear interactions

Minibeam scattering uses material- and species-dependent Urban data, including scattering mean free paths and true-to-geometric path conversion. In CT, restricted stopping powers, ranges, and inverse ranges are evaluated from continuous26 at the actual density. The pre-step charge correction is fixed within a step, while scattering mean free paths and radiation lengths follow their respective density scaling. Copper, air, and additional recoil isotopes outside the registry use their corresponding data. Scattering is sampled in a local particle frame and transformed back to world coordinates.

Restricted primary fluctuations in copper are evaluated by a shared sampling engine. Z/A, mass, and effective charge select the proton Universal or ion fluctuation parameters and corrections; a common entry does not imply identical physical parameters. Copper primary fluctuations remain enabled and restricted fragment fluctuations are disabled by default. The native copper-fragment path in carbon configurations retains hard-delta Poisson and energy-spectrum sampling. Native fragment transport is controlled independently of the primary fluctuation selection and retains its own calibration.

Energy loss, scattering, nuclear elastic scattering, and inelastic reactions jointly determine the copper-exit distribution. Hard-delta energy in copper is removed from the charged beam and recorded in the beamline energy ledger. The corresponding electrons are not explicitly transported and their energy is not added to patient dose. Patient transport retains the local ionization deposition described in Section II.B. Increasing a geometric step cap does not remove the existing ionization segmentation clock.

## II.D. GPU implementation and dose scoring

### II.D.1. Execution structure, random numbers, and resident batches

The engine is implemented in C++20/SYCL. NVIDIA performance calculations use the CUDA backend with native machine code compiled ahead of time for each target device; runtime PTX JIT compilation is disabled. Four dedicated pipelines serve proton/carbon broadbeam/minibeam transport, with a generic pipeline for custom proton or C12 configurations. They share geometry, energy-loss, nuclear-interaction, random-number, and scoring functions. Dedicated source pipelines do not restrict the species of nuclear fragments.

Minibeam execution separates copper/air transport from transport within the medium. Handoff states preserve position, direction, energy, random identity, and required continuation information. Charged products from copper are handled through separate queues. Secondary species grouping and segmented continuation are enabled only in the relevant configurations; reordering does not change track identity. Queue capacity is allocated in advance, and any overflow invalidates the affected batch.

A counter-based Philox generator binds random streams to spot, history, and descendant-track identities, preserving identity across queue reordering and continuation. Independent statistical batches use distinct seeds. Consecutive batches reuse one GPU context and resident material and physics data. Internal transport chunks and statistical/output batch sizes are controlled separately.

### II.D.2. Integer spot allocation and absolute dose

Let W_j denote the physical particle weight of spot j for one irradiation, W=Σ_j W_j the total physical incident particles, and N the number of simulated histories. Integer spot counts n_j are assigned from N W_j/W by the largest-remainder method, ensuring Σ_j n_j=N. The campaign-wide allocation is subsequently partitioned among devices and batches. When matching archived TOPAS calculations, the actual archived integer spot counts take precedence.

Deposited energy is accumulated using FP32 GPU atomics. For voxel density ρ_i in g/cm³, volume V_i in mm³, and accumulated deposited energy E_i in MeV,

\[
 m_i=\rho_i V_i\,10^{-6}\ {\rm kg},\qquad
 D_i=\frac{E_i\,1.602176634\times10^{-13}}{m_i}\frac{W}{N}.
\tag{3}
\]

Batch and device contributions are summed after applying the same W/N factor; no additional batch-count multiplier is applied. References scored as cumulative dose for the simulated particles are compared using their corresponding source counts and cumulative scale, without imposing prescription normalization. Batch files are merged at higher precision on the host, which does not eliminate within-batch FP32 accumulation error.

## II.E. Full-plan reference calculations and comparison design

### II.E.1. Cases, grids, and simulated histories

Full-plan comparisons include RT07575 (Head 1), RT06423 (Head 2), and 20022516 (Lung 1), covering head anatomy and heterogeneous low-density lung. Broadbeam calculations omit the copper slit collimator, whereas minibeam calculations retain their plan-specific collimators and fields. CT, source optics, spot energies and weights, placement, geometry, and normalization are matched within each comparison. The table defines the currently completed matched full-plan dataset. Counts are simulated incident histories for each engine, not the total physical particles prescribed.

| Case | Proton broadbeam | Carbon broadbeam | Proton minibeam | Carbon minibeam |
|---|---:|---:|---:|---:|
| RT07575 / Head 1 | 250,626,500 | 129,638,170 | 1,000,000,000 | 1,000,000,000 |
| RT06423 / Head 2 | 124,266,340 | 151,087,660 | 1,000,000,000 | 1,000,000,000 |
| 20022516 / Lung 1 | 361,050,622 | 177,173,040 | 3,000,000,000 | Not included in the completed dataset |

Completed plans use voxel spacings of 0.5×0.5×2.0 mm³, with calculation axes potentially permuted by patient placement. Dimensions, origins, and axis ordering follow each plan’s accompanying reference headers. Comparisons are performed in physical coordinates, retaining the original grid and dose scale without peak alignment, post hoc registration, or dose fitting. A search increment smaller than the voxel spacing provides denser interpolation sampling, not additional spatial resolution.

### II.E.2. TOPAS references and model matching

TOPAS uses the corresponding materials, source, and geometry with DoseToMedium scoring[7]. Reference physics lists are archived per plan and include electromagnetic, hadron/ion inelastic, nuclear elastic, stopping, and decay modules. Minibeam references use AllChargedUrbanPhysics for the corresponding charged-particle scattering configuration. Carbon references additionally require verification of actual GenericIon nuclear elastic registration. Geant4 data versions, electromagnetic energy limits, production cuts, copper geometry, and step settings are obtained from executed inputs and logs rather than inferred from module names. Production cuts and maximum transport steps are reported separately. For example, the lung proton minibeam reference uses a 0.05 mm production cut and a 10 GeV electromagnetic table limit, whereas the three historical carbon broadbeam references retain a 600 MeV electromagnetic table limit; these settings are not assumed to be identical across references.

The current full-plan comparison uses unthresholded TOPAS dose scored after independent transport with plan-weighted source allocation. Weighted dose-influence matrices used during optimization are separately identified auxiliary data and do not replace full recalculation. Original-TPS-weight carbon broadbeam references retain their cumulative simulated-dose scale; optimized minibeams retain the corresponding per-irradiation scale. Differences between native TOPAS transport and the GPU finite event library, local ionization deposition, mean secondary losses, and omitted nuclear neutral transport are retained when interpreting dose discrepancies.

## II.F. Dose agreement, statistical uncertainty, and computational performance

### II.F.1. Evaluation regions and local gamma

Three-dimensional local gamma uses TOPAS as the reference[10]:

\[
 \gamma(\mathbf r)=
 \min_{\mathbf r'}
 \left[
 \frac{\|\mathbf r'-\mathbf r\|^2}{\Delta r^2}+
 \frac{[D_{\rm GPU}(\mathbf r')-D_{\rm ref}(\mathbf r)]^2}
 {[\delta D_{\rm ref}(\mathbf r)]^2}
 \right]^{1/2}.
\tag{4}
\]

GPU dose is queried by trilinear interpolation, with γ≤1 indicating a pass. The BODY mask selects reference evaluation points; GPU dose outside BODY within the search domain is not set to zero. Evaluation masks are fixed from the final reference and are not reselected from low-statistics repeats.

| Mode | Reference-dose threshold within BODY | Reported local criteria | Spatial search increment |
|---|---|---|---:|
| Broadbeam | >10% of global reference Dmax | 3%/3 mm, 1%/1 mm, 3%/0 mm | 0.1 mm |
| Minibeam | >0.4 Gy, i.e., 10% of a 4 Gy prescription | 3%/0.2 mm, 3%/0.1 mm, 3%/0 mm | 0.02 mm |

The threshold selects the evaluation region only; the local dose tolerance remains the specified percentage of each point's reference dose. Zero-DTA analysis compares doses at the same position and is a local relative-dose-difference test, without the distance division in Eq. (4). The predefined broadbeam criterion is a local 1%/1 mm pass rate >99%. Minibeam submillimeter pass rates are reported directly rather than assessed using the broadbeam criterion. Exploratory analyses using other thresholds or geometric irradiation masks are identified separately.

Additional measures include spatial dose-difference distributions, target mean dose and dose–volume indices, and lateral profiles at different depths to examine minibeam peaks, valleys, and distal structure. Dose differences contain both random fluctuations and systematic discrepancies and are not treated as pure Monte Carlo uncertainty.

### II.F.2. Uncertainty in both calculations

GPU statistical uncertainty is estimated from independent full-plan batches with matched source composition. For B equal-statistics batches whose contributions d_b have already been scaled by the campaign-wide W/N, D=Σ_b d_b and

\[
 \widehat{\sigma}_{D}^{\,2}
 =\frac{B}{B-1}
 \left[\sum_b d_b^2-\frac{D^2}{B}\right].
\tag{5}
\]

For unequal batch sizes, define a_b=N_b/N, Y_b=d_b/a_b, and q=Σ_b a_b². The implemented weighted batch estimator is

\[
 \widehat{\sigma}_{D}^{\,2}
 =\frac{q}{1-q}
 \left[\sum_b a_bY_b^2-D^2\right].
\tag{6}
\]

Equation (6) reduces to Eq. (5) for equal-sized batches. It is used here for batches with nearly equal history counts and matched source composition, without assuming equal variance for arbitrarily imbalanced batches or different sources. Merging across devices retains batch counts, source proportions, and independent seeds.

TOPAS references with native per-history variance use those statistics, converted to final-dose variance according to the actual scorer definition. Variances from independent spots or recalculations are combined using squared dose-scaling factors. The three historical carbon broadbeam references lack stored native variance. For each, a matched 300,000-history TOPAS pilot preserves geometry, physics list, energy spectrum, and relative weights. Its standard error is extrapolated to the full history count by √(N_pilot/N_full), with both quantities expressed on the same full-plan dose scale. The original full-reference dose supplies the relative-error denominator, and the extrapolation is cross-checked against variance between original independent replicas. These entries are identified as estimates; pooled variance across mixed spots can be conservative for a fixed stratified spot allocation.

For a fixed evaluation region R, the root-mean-square relative standard error is

\[
 u_{\rm RMS}=100
 \sqrt{\frac{1}{|R|}
 \sum_{i\in R}\frac{\widehat{\sigma}_i^2}{D_{{\rm ref},i}^2}}
 \ \%.
\tag{7}
\]

GPU and TOPAS uncertainties are reported separately and, for independent simulations, combined through σ_diff²=σ_GPU²+σ_TOPAS². Percentiles and fractions of voxels below selected uncertainty limits are additionally reported where relevant. This is a regional summary of voxelwise one-standard-error values, not uncertainty in the regional mean or a guarantee for every voxel. Spatial correlations prevent direct conversion of voxel variances into confidence intervals for gamma pass rates or dose–volume indices. Finite-library uncertainty and physical-model bias are not included in these random-transport estimates.

### II.F.3. Timing and run validity

Performance is reported separately for RTX 2080 Ti, RTX A6000, and L4 devices where used. Executables contain native device code compiled ahead of time for the corresponding architecture, with runtime PTX JIT disabled. Compilation is excluded from dose-calculation time. Cold start extends from new-process launch to the first saved dose batch, including input reading, device initialization, physics-data upload, and first-batch calculation. Steady-state throughput is the total incident histories in subsequent batches divided by their cumulative wall time within the same process, including the corresponding audits and dose output. Kernel-only timing does not replace end-to-end throughput. If the file-system cache is not explicitly cleared, process cold start is not interpreted as a cold physical-disk cache.

Large campaigns distribute histories across GPUs according to measured throughput while preserving campaign-wide source allocation and normalization. Per-device and aggregate multi-device throughput are reported separately. TOPAS single-node 56-thread throughput is total histories divided by cumulative node computation time. Historical CPU records are distinguished from contemporaneous measurements. Results with different grids, physics, statistical precision, or timing boundaries are not pooled into one speedup.

Primary, copper/air upstream, and charged-secondary stage times provide the computational breakdown. When an algorithm changes the stochastic estimator, time to comparable statistical precision is also considered. Each batch is checked for completed histories, species transport status, physics-table coverage, energy accounting, finite nonnegative dose, completed queues, and zero overflow. Run audits and matched GPU implementation checks establish computational validity; independent TOPAS dose comparisons assess agreement between models.

## References

1. Jia X, Schümann J, Paganetti H, Jiang SB. GPU-based fast Monte Carlo dose calculation for proton therapy. *Phys Med Biol.* 2012;57:7783–7797. [doi:10.1088/0031-9155/57/23/7783](https://doi.org/10.1088/0031-9155/57/23/7783).
2. Qin N, Pinto M, Tian Z, et al. Initial development of goCMC: a GPU-oriented fast cross-platform Monte Carlo engine for carbon ion therapy. *Phys Med Biol.* 2017;62:3682–3699. [doi:10.1088/1361-6560/aa5d43](https://doi.org/10.1088/1361-6560/aa5d43).
3. De Simoni M, Battistoni G, De Gregorio A, et al. A data-driven fragmentation model for carbon therapy GPU-accelerated Monte-Carlo dose recalculation. *Front Oncol.* 2022;12:780784. [doi:10.3389/fonc.2022.780784](https://doi.org/10.3389/fonc.2022.780784).
4. Lee H, Shin J, Verburg JM, et al. MOQUI: an open-source GPU-based Monte Carlo code for proton dose calculation with efficient data structure. *Phys Med Biol.* 2022;67:174001. [doi:10.1088/1361-6560/ac8716](https://doi.org/10.1088/1361-6560/ac8716).
5. Feng H, Holmes JM, Vora SA, et al. Modelling small block aperture in an in-house developed GPU-accelerated Monte Carlo-based dose engine for pencil beam scanning proton therapy. *Phys Med Biol.* 2024;69:035003. [doi:10.1088/1361-6560/ad0b64](https://doi.org/10.1088/1361-6560/ad0b64).
6. Schneider W, Bortfeld T, Schlegel W. Correlation between CT numbers and tissue parameters needed for Monte Carlo simulations of clinical dose distributions. *Phys Med Biol.* 2000;45:459–478. [doi:10.1088/0031-9155/45/2/314](https://doi.org/10.1088/0031-9155/45/2/314).
7. Perl J, Shin J, Schümann J, Faddegon B, Paganetti H. TOPAS: An innovative proton Monte Carlo platform for research and clinical applications. *Med Phys.* 2012;39:6818–6837. [doi:10.1118/1.4758060](https://doi.org/10.1118/1.4758060).
8. Agostinelli S, et al. Geant4—a simulation toolkit. *Nucl Instrum Methods Phys Res A.* 2003;506:250–303. [doi:10.1016/S0168-9002(03)01368-8](https://doi.org/10.1016/S0168-9002(03)01368-8).
9. Eyges L. Multiple scattering with energy loss. *Phys Rev.* 1948;74:1534–1535. [doi:10.1103/PhysRev.74.1534](https://doi.org/10.1103/PhysRev.74.1534).
10. Low DA, Harms WB, Mutic S, Purdy JA. A technique for the quantitative evaluation of dose distributions. *Med Phys.* 1998;25:656–661. [doi:10.1118/1.598248](https://doi.org/10.1118/1.598248).

---

**Companion material, outside the manuscript Methods text:** [literature and writing analysis](docs/gpu_mc_methods_literature.md), [physics implementation comparison](docs/topas_gpu_physics_comparison.md), [data extraction](extensions/README.md), and [full-plan and uncertainty records](docs/full_plan_validation_20261017/README.md).

**Author information required before submission:** case provenance and ethics approval or exemption; complete per-plan source, energy-layer, field, grid, and collimator tables; optimization objectives and solver settings; and frozen reference-physics, compiler/driver, and CPU-model inventories. Clinical beam commissioning and independent acceptance of every configuration are not asserted without those records. Dose pass rates, throughput, and measured uncertainty values belong in Results; this section defines how they are obtained.
