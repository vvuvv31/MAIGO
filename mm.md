# II. Materials and Methods

Manuscript theme: GPU Monte Carlo: From Broadbeam to Minibeam. [中文版](mm_zh.md).

## II.A. Study design, materials, and beam inputs

### II.A.1. Computational framework and scope

MAIGO was developed as a GPU Monte Carlo engine for proton and carbon-ion broadbeam dose calculation and extended to minibeams formed by a copper slit collimator. Condensed-history electromagnetic transport is combined with stochastic nuclear interactions and subsequent transport of charged fragments. These computational strategies are also used in established GPU proton and carbon-ion engines[1–4]. The minibeam extension explicitly transports particles through the collimator, air, and patient to resolve the lateral dose structure. Upstream collimator modeling is distinguished from transport within the patient, as in GPU aperture-modeling studies[5].

Incident particles are protons (Z=1, A=1) or C12 ions (Z=6, A=12). Supported charged nuclear fragments are transported subsequently. The scored quantities are absorbed dose to medium and, when requested, dose-averaged electronic linear energy transfer (LET). RBE and biological dose are outside the present scope. The combinations of projectile, water or CT medium, and broadbeam or minibeam irradiation define eight standard configurations. Patient full-plan calculations retain their case-specific sources, weights, and geometries; the default CT minibeam phantom configuration is a separate benchmark.

### II.A.2. Material representation and physics data

Patient CT images are converted offline to a regular voxel grid containing its origin, voxel spacing, mass density, and material identifiers. Hounsfield units are mapped to density and one of 25 tissue elemental compositions using the Schneider method[6]. A Water_75eV reference with a mean excitation energy of 75 eV provides the twenty-sixth electromagnetic composition. Electromagnetic parameters are evaluated for the tissue composition and actual density, including density-dependent effects; tissues are not represented solely by mass-density scaling of water.

Physics data are generated offline using OpenTOPAS 4.2.3 and Geant4 11.3.2[7,8]. The active transport data comprise unrestricted electronic stopping powers, energy-loss fluctuation parameters, multiple-scattering parameters, nuclear interaction rates, and final-state data. Consistent continuous-slowing-down approximation (CSDA) ranges and inverse ranges are derived from the stopping response. The continuous-density ion electromagnetic model, termed continuous26, covers the 25 tissue compositions and water. Copper and upstream air have independently extracted electronic response data. Nuclear inelastic interactions are represented by finite correlated event libraries; nuclear elastic data contain total rates, target isotopes, and momentum-transfer samples. The GPU samples these data rather than evaluating a Geant4 intranuclear cascade online.

Each package specifies its projectile, composition, and energy coverage. Required channels and query ranges are checked before simulation; another material or local energy deposition is not substituted for missing required physics. Inputs, physics-data checksums, program version, source allocation, and seeds are retained for each calculation. Extraction and reference-dose versions are recorded separately because a shared data origin does not imply stepwise-equivalent transport.

### II.A.3. Spot sources and physical-dose weights

Scanning plans specify spot positions, energies, and weights for each field. An energy-dependent beam model supplies position and angular standard deviations, position–angle correlations, and energy spread. Correlated positions and directions are sampled in a local beam frame and transformed to world coordinates using the source plane, scanning-magnet parameters, field orientation, and patient placement. Ion energies are expressed in MeV/u; plan inputs stored as total kinetic energy in MeV are converted using the mass number.

For reoptimized physical-dose plans, A denotes the dose-influence matrix per incident particle and w the nonnegative spot particle weights, giving d=Aw. Case-specific target and organ-at-risk objectives determine the weights. Optimization defines the common irradiation source: GPU and independent TOPAS recalculations use the same final weights, rather than separately optimized weights. Historical carbon broadbeam plans retaining their original TPS weights are distinguished from reoptimized plans and are not collectively described as reoptimized 4 Gy plans.

## II.B. Shared ion transport and broadbeam calculation

### II.B.1. Unrestricted electronic stopping and mean energy loss

Each charged track carries position, direction, total kinetic energy E, nuclear identity (Z,A), and random-stream identity. The electromagnetic response is selected for that species, the current material composition, and the actual density. Package energy coordinates expressed in MeV/u are evaluated at E/A. The unrestricted linear electronic stopping power, \(S_e(E)\), includes the expected energy transferred to ionization secondaries over the full kinematically allowed interval. Its native species-dependent response includes effective-charge behavior; a single proton stopping curve multiplied by a universal charge-squared factor is not used for all ions.

For fixed local composition and density, the CSDA range and the finite-path mean loss are related by

\[
 R(E)=\int_0^E\frac{dE'}{S_e(E')},\qquad
 \overline{\Delta E}(E,\ell)
 =E-R^{-1}\!\left(\max[R(E)-\ell,0]\right),
\tag{1}
\]

where \(\ell\) is true path length. Reference ranges integrate the reciprocal of the piecewise-linear stopping response analytically within each energy interval; the corresponding analytic inverse is used within that interval. The prescribed low-energy continuation supplies the integral below the first response node. At the actual CT density, the Sternheimer density-effect correction is evaluated continuously. Linear density scaling of range is used only where the density-effect laws permit it; otherwise, the reciprocal-stopping correction is integrated and the inverse is solved with a safeguarded iteration. No neighboring-density material is substituted.

For small fractional losses in water and patient media, a midpoint stopping evaluation approximates the finite-path mean: a step-start estimate predicts the midpoint energy, at which stopping is reevaluated. Larger losses and range-ending paths use Eq. (1). Stopping, residual range, inverse range, and mean loss therefore refer to the same unrestricted response. Additional heavy recoils outside the ordinary ion registry instead use their isotope-specific total electronic-plus-nuclear stopping response, as described in Section II.B.6.

Electronic energy loss within water or the patient is deposited in the corresponding scoring voxel. Electrons and electromagnetic photons are not explicitly transported. This local-ionization approximation preserves the electronic energy removed from the ion but omits its subsequent nonlocal redistribution. It is particularly relevant to low-density tissues, interfaces, and minibeam valleys, where the range of ionization secondaries may be comparable to the spatial dose structure. Nuclear neutral energy is treated separately and is not converted into a local electronic deposit.

### II.B.2. Primary energy-loss fluctuations

Primary energy-loss fluctuations use algorithms adapted from the Geant4 Universal and Ion fluctuation models[8]. One shared sampling engine receives the total electronic mean loss, accepted true path, material excitation parameters, particle mass and charge, and both native variance responses. For a projectile of mass M and Lorentz factor \(\gamma\), the kinematic electron-transfer limit is

\[
 T_{\max}=
 \frac{2m_ec^2\beta^2\gamma^2}
 {1+2\gamma m_e/M+(m_e/M)^2}.
\tag{2}
\]

This maximum is used as the fluctuation transfer limit, without a separate restricted-loss and hard-delta partition. For medium data exported at a reference transfer cut \(T_c\), the variance at the current density is converted to its unrestricted value using

\[
 V_{\mathrm{full}}=V_{\mathrm{ref}}
 \frac{T_{\max}/\beta^2-T_{\max}/2}
 {T_{\max}/\beta^2-\min(T_c,T_{\max})/2}.
\tag{3}
\]

Here, \(\beta\) is particle speed divided by the speed of light, and \(V_{\mathrm{ref}}\) includes density scaling and the native charge-fluctuation multiplier. The path variance is obtained by multiplying the variance per unit length by the accepted true path. Copper and air provide the unrestricted variances directly, rather than through this reference-cut conversion.

Hydrogen-1 uses Universal sampling. Other ions use the response-selected Ion branch and its native transition to Universal sampling at higher kinetic energy. Universal sampling distinguishes thick-absorber and thin-layer regimes. The thick regime uses a truncated Gaussian or a moment-matched Gamma distribution according to the mean-to-standard-deviation ratio. The thin regime combines excitation and ionization contributions, retaining bounded explicit Poisson sampling of the ionization tail and a Gaussian treatment of its dense component. These internal samples define a total loss; they do not create electron tracks.

The low-energy Ion branch uses its charge-dependent dispersion and finite-loss correction. According to the mean-to-standard-deviation ratio, it samples a truncated Gaussian, Gamma, or uniform distribution. In a Gamma branch, mean \(\mu\) and variance V determine shape \(k=\mu^2/V\) and scale \(\theta=V/\mu\); this branch belongs to the adapted fluctuation model and is not an independently added delta-loss distribution. For a sampled path, the final loss is bounded by the available kinetic energy; a path that exhausts the residual range terminates without an additional fluctuation draw. Thus, a shared engine does not imply identical proton and C12 distributions: the mass, charge, model selection, material parameters, and dispersions remain particle dependent.

Primary fluctuations are enabled in all standard configurations, including copper. Charged-secondary straggling is disabled by default in both the patient and upstream regions, so their electronic loss equals its finite-path mean. This approximation changes the secondary loss distribution while retaining secondary motion, Coulomb scattering, nuclear interactions, and the source energy spread.

### II.B.3. Step selection and boundary handling

At each step, material and energy-dependent responses are evaluated before competing path limits are applied. Let f be the native range fraction, \(r_f\) the native final-range parameter, and \(\eta\) the allowed mean-loss fraction. For \(R(E)>r_f\), the electromagnetic proposal is

\[
 \ell_{\rm EM}=\min\!\left[
 fR(E)+r_f(1-f)\!\left(2-\frac{r_f}{R(E)}\right),\;
 R(E)-R((1-\eta)E)
 \right].
\tag{4}
\]

For \(R(E)\le r_f\), the proposal is the remaining range. This final-range treatment terminates the stopping tail rather than taking indefinitely shorter fractional-energy steps. Fast CT broadbeam transport uses \(\eta=0.05\), whereas minibeam patient transport uses \(\eta=0.01\). Other broadbeam settings retain their recorded mean-loss fraction. The parameters f and \(r_f\) are supplied by the species/material model, independently of the imposed spatial step cap.

The electromagnetic proposal competes with the maximum step, material and scoring boundaries, scattering limits, and nuclear interaction distance. Fermi–Eyges transport uses geometric segment lengths; Urban transport distinguishes true path from the geometric chord. In the latter case, boundary-limited geometric motion is converted to a true path before energy loss and nuclear optical depth are evaluated. If a nuclear interaction further shortens that true path, the geometric motion and scattering update are finalized for the shortened path. The angular update is committed only after these limits have been resolved.

| Transport region | Maximum ion step | Principal scattering treatment |
|---|---:|---|
| Water/CT broadbeam | 1.0 mm | Gaussian Fermi–Eyges |
| Water/CT minibeam | 0.5 mm | Urban data-driven transport |
| Minibeam copper collimator | 0.05 mm | Copper-specific Urban and nuclear elastic scattering |

These engine-enforced step caps are distinct from TOPAS secondary-production range cuts and ion termination energies. Slit air in CT carbon minibeams has a separate 0.2 mm cap; other air regions follow their respective input limits and boundary handling.

Continuous losses are assigned to the traversed source voxel, while nuclear-vertex deposits use the collision location. Tracks reaching the patient-grid boundary are recorded as escaped; exterior energy is not deposited in a clamped edge voxel. The standard total-kinetic-energy termination threshold is 0.1 MeV. Residual kinetic energy at an in-medium termination is deposited locally, with a separate distinction from finite-path LET scoring. There is no electron-production-cut ionization clock.

### II.B.4. Broadbeam multiple Coulomb scattering

Broadbeam scattering uses a Gaussian Fermi–Eyges approximation[9]. For one transverse axis, angular deflection \(\theta\) and lateral displacement y are sampled jointly. For a segment of length L and constant local scattering power \(\mathcal T\), their covariance is

\[
 \begin{pmatrix}
 \langle\theta^2\rangle & \langle\theta y\rangle\\
 \langle\theta y\rangle & \langle y^2\rangle
 \end{pmatrix}
 =\mathcal T
 \begin{pmatrix}L&L^2/2\\L^2/2&L^3/3\end{pmatrix}.
\tag{5}
\]

The two transverse axes are sampled in a local frame and transformed to world coordinates. The endpoint displacement therefore retains its correlation with the outgoing direction; scattering is not represented by an angular kick alone. Long transport steps can be divided into scattering segments, with energy at each segment midpoint estimated from the accepted energy loss. This scattering segmentation is distinct from the electromagnetic transport step and its maximum length.

The Gaussian scattering power contains a core term proportional to \([E_sZ/(\beta pc)]^2/X_0\), where \(X_0\) is the local radiation length. In the standard Gaussian mode, the second moment of the calibrated tail component is added to this power rather than sampling individual tail events. Its nominal rate is scaled from water by the ratio of inverse radiation lengths. Proton, deuteron, triton, and alpha coefficients use species-specific homogeneous-water fits at 50, 150, and 300 MeV/u, with piecewise-linear interpolation and endpoint values outside that interval. C12 and ions without a separate fit use the fixed core energy 9.9 MeV, nominal water tail rate 0.0025 mm⁻¹, and tail scattering energy 2.4 MeV. These are effective scattering parameters, not an online evaluation of the full Geant4 angular distribution. Coulomb multiple scattering is separate from the strong-interaction nuclear elastic process below.

### II.B.5. Nuclear rates and elastic–inelastic competition

For nuclear channel c, the macroscopic rate is formed from target number densities and microscopic cross sections,

\[
 \Sigma_c(E)=\sum_t n_t\sigma_{c,t}(E),\qquad
 P(t\mid c,E)=\frac{n_t\sigma_{c,t}(E)}{\Sigma_c(E)}.
\tag{6}
\]

Composition-specific mass rates are scaled by the actual density to obtain rates per millimeter. Inelastic channels are masked to the explicitly supported projectile–target and energy domains; missing required coverage is not filled by another target or a local deposit. Elastic and inelastic rates are evaluated independently.

Primary and charged secondary particles in water and patient media use the same optical-depth algorithm. Each track independently samples

\[
 \tau=-\ln U,\qquad
 \Sigma_{\mathrm{tot}}=\Sigma_{\mathrm{el}}+\Sigma_{\mathrm{in}},
\tag{7}
\]

where U is uniform. Over an accepted path, the optical-depth increment is approximated as \(\Sigma_{\rm tot}\ell\) using the current rate. If that increment exhausts the remaining depth, the path is shortened to \(\ell_{\rm nuc}=\tau_{\rm remaining}/\Sigma_{\rm tot}\). The process is then selected with probabilities \(\Sigma_{\rm el}/\Sigma_{\rm tot}\) and \(\Sigma_{\rm in}/\Sigma_{\rm tot}\), followed by target selection from the corresponding partial rates. Residual optical depth is preserved across boundaries and execution continuations. Each particle owns its own random optical depth; sharing the algorithm does not correlate different tracks' interaction distances.

Primary inelastic proposals also use an integral-rate cache. The proposal rate is refreshed at material or density changes, after kinetic energy falls below 80% of its cached value, and after a proposed interaction. At a candidate, the post-loss inelastic rate supplies a rejection test against the cached proposal rate. Rejected candidates leave the surviving ion available for continued transport. Elastic rates do not enter that inelastic rejection denominator. This is a piecewise-rate transport approximation, rather than continuous integration of the energy-dependent cross section along each slowing trajectory.

### II.B.6. Nuclear elastic scattering and recoil transport

Nuclear elastic scattering is enabled in all eight standard configurations. After selecting the target element, adjacent energy-node contributions to the elastic rate determine the probability of selecting an angular sample from either node. A stored sample supplies the target isotope and normalized momentum transfer, equivalent to a center-of-mass scattering cosine. A uniform azimuth and relativistic two-body kinematics then generate both outgoing momenta at the current incident energy. In units with c=1,

\[
 s=(m_p+m_t)^2+2m_tE,\qquad
 p_*=\frac{m_t\sqrt{E(E+2m_p)}}{\sqrt{s}},\qquad
 E_R=\frac{p_*^2(1-\cos\theta_*)}{m_t},\qquad
 E'_p=E-E_R.
\tag{8}
\]

Here \(m_p\) and \(m_t\) denote projectile and target masses, \(p_*\) the center-of-mass momentum, and \(E_R\) the recoil kinetic energy. The sampled angular distribution remains essential: conservation laws determine the final-state kinematics once the angle is specified, but do not determine its probability. Package-specific masses are retained, including the calibrated C12 A×amu convention where required.

Recoils in the ordinary charged-ion registry use its corresponding transport response and allowed nuclear channels. Additional target isotopes use dedicated unrestricted electronic-plus-nuclear stopping and a CSDA range derived from that total response. They are not assigned invented discrete nuclear rates. Electronic LET uses an independent electronic-only stopping table for those isotopes, excluding nuclear stopping from its numerator and denominator. Recoils below the termination threshold deposit their remaining energy locally; higher-energy transported recoils can redistribute dose before stopping.

### II.B.7. Nuclear inelastic final states and charged fragments

The installed primary proton event library records Binary Cascade events generated with QGSP_BIC_HP; the primary C12 library was generated with FTFP_INCLXX. The patient-secondary library was extracted with EM option 4 and ion Binary Cascade physics. These are the extraction configurations recorded with the libraries, distinct from the executed full-plan TOPAS reference lists described in Section II.E. Nuclear model identity is therefore retained per dataset rather than treating all Geant4-derived events as interchangeable.

Inelastic final states are indexed by exact projectile (Z,A), target element, and incident energy per nucleon. For a query \(e=E/A\) between adjacent nodes \(e_0\) and \(e_1\), stochastic interpolation selects the upper node with

\[
 P(e_1\mid e)=\frac{e-e_0}{e_1-e_0},\qquad
 P(e_0\mid e)=1-P(e_1\mid e).
\tag{9}
\]

An event is then drawn uniformly from that node's library. Queries must lie within the exact channel domain, with a bracketing gap no larger than 5 MeV/u. Individual product energies are not rescaled to force an event generated at a node to match the query energy. The node mismatch and energy-accounting residual are retained in the model audit. This sampling preserves the library's event correlations but introduces finite-library and incident-energy-discretization approximations.

One common random azimuth is applied to the complete event before its product directions are rotated into the current incident frame. Multiplicity and within-event energy–angle correlations are therefore retained, rather than drawing each product independently from separate marginal distributions. The recorded process-local deposit is scored at the interaction vertex. Supported charged products above cutoff enter subsequent transport with their own species, energy, direction, generation, and random identity; charged products below cutoff are deposited locally.

Patient-medium secondary inelastic generations are limited to two. Reaching this limit suppresses new inelastic branches but preserves electromagnetic and permitted elastic transport of existing particles. Species assigned an electromagnetic-only policy retain that policy rather than acquiring unsupported reaction channels. The implemented species policy does not propagate Be6 residuals, and their energy remains in the untracked balance. A general radioactive-decay or capture chain is not simulated.

Nuclear neutral products, including neutrons and nuclear photons, are not transported. Their energy and the remaining untracked reaction balance are recorded separately from deposited dose and escaped charged-particle energy; they are not added at the reaction vertex as surrogate local dose. The finite event library, energy-node mixture, secondary-generation limit, species policies, and omitted neutral transport remain explicit differences from native TOPAS transport.

## II.C. Extension to copper-collimated minibeams

### II.C.1. Collimator geometry and upstream transport

The minibeam extension adds a copper slit collimator, slit air, and the downstream air gap to the broadbeam patient-transport framework. Geometry inputs specify copper thickness and external shape; slit count, width, pitch, length, phase, and rotation; and relative source, collimator, and patient positions. The collimator is fixed in the field frame rather than translated with the scanning spot. Copper–air sidewalls and the patient entrance participate in geometric step selection.

The proton minibeam full plans use 60 mm of copper with 15 slits, a 0.5 mm slit width, 3.6 mm pitch, and 50 mm slit length. The RT07575 carbon minibeam plan uses 100 mm of copper with 17 slits, a 0.7 mm width, 3.0 mm pitch, and 100 mm full slit length. Its 90° and 270° crossfiring fields retain their individual slit-array placements; the nominal center array for the 270° field has a half-pitch phase offset relative to the 90° field. Other carbon plans use their actual executed geometry inputs rather than dimensions inferred from RT07575. Distances from the collimator exit to the CT bounding box, skin, and isocenter are defined separately.

Upstream transport includes directly transmitted primaries, primaries scattered through copper, and charged nuclear products. Their joint position, direction, energy, and species distribution at patient entry is generated by this transport. A source containing only direct slit transmission or a fixed reusable phase-space cache is not substituted for the collimator. Original histories without transmission remain in the incident-particle normalization.

### II.C.2. Urban scattering and true-to-geometric path conversion

Minibeam patient transport and upstream primary transport use material- and species-dependent Urban scattering data. Inputs include the transport mean free path, radiation length, and angular-distribution parameters, together with the current unrestricted range. In CT, the electronic response is evaluated at the actual density, while scattering mean free paths and radiation lengths retain their own density scaling. Additional recoil isotopes use their matched total-stopping context. The water or CT ion stopping response is not replaced by copper data when a track passes between these regions.

Urban transport accounts for the difference between the microscopic true path \(\ell\) and the geometric advance g. In the constant-transport-mean-free-path limit,

\[
 g=\lambda_{\rm tr}\left(1-e^{-\ell/\lambda_{\rm tr}}\right),
\tag{10}
\]

where \(\lambda_{\rm tr}\) is the transport mean free path. The implemented conversion also includes the energy-dependent mean-free-path branches. The true path controls electronic loss and nuclear optical depth; the geometric advance and sampled lateral displacement control motion. Near a slit sidewall, material boundary, or scoring face, the geometric distance is limited first and converted back to an admissible true path. This distinction prevents a geometric chord from being used as though it represented the complete scattering path.

The angular sampler uses the Urban small-angle core and non-Gaussian tail, with its native transitions for very small or large scattering thickness. The angular width depends on radiation length and the model's pre/post-energy estimate. Angular deflection and lateral displacement are sampled in a particle-local frame, transformed to world coordinates, and constrained by geometric safety. Nuclear shortening of a proposed step precedes the final angular/lateral sample. Persistent boundary and range-limit state is retained between steps rather than reinitialized at each scheduler handoff. This is a data-driven implementation of Urban transport, not execution of the native Geant4 process manager.

### II.C.3. Copper and air electronic energy loss

Copper and upstream air use independently extracted unrestricted electronic stopping powers and both native fluctuation dispersions. The response covers the charged species produced by the copper event library, including its heavy fragments. For mean losses up to 2% of kinetic energy, loss is evaluated from step-start stopping; larger losses use CSDA range inversion. Both calculations use the same response. Below the first energy node, the prescribed square-root stopping continuation supplies the corresponding range integral; missing species or energies above coverage invalidate the calculation.

The same Universal/Ion engine used in the patient samples upstream primary losses, with particle-dependent mass, charge, excitation parameters, and the maximum kinematic transfer. Copper/air variances are extracted directly at that transfer limit. Primary fluctuations remain enabled and charged-fragment fluctuations are disabled by default. Thus, turning off fragment straggling removes the full stochastic electronic-loss draw; it does not leave a separately sampled hard-delta term. Scattering and nuclear transport are selected independently of the fluctuation branch.

Electronic energy lost in copper or upstream air is removed from the charged beam and recorded as a beamline loss, without adding it to patient dose or generating explicit electron tracks. Energy deposited by ions after entering the patient follows the local-ionization convention in Section II.B.1. Histories that stop upstream retain their place in the original incident-particle normalization. The 0.05 mm copper cap and separate slit-air limits constrain transport even when the electronic response would permit a longer step.

### II.C.4. Copper nuclear interactions and charged products

Copper nuclear elastic scattering is enabled in all four minibeam configurations. Primary interaction distance is sampled using the combined copper elastic and inelastic rate. After electronic slowing, the applicable rates determine the process at the candidate interaction. An elastic event samples a copper-isotope angular record and applies the two-body kinematics of Section II.B.6. The outgoing beam ion continues; the copper recoil energy is retained in the upstream energy balance rather than scored as patient dose. This upstream recoil treatment is distinct from explicit target-recoil transport inside the patient.

Copper inelastic events are drawn from the copper-specific correlated library, with its exact projectile/target lookup and energy-node selection. Direct charged products and recorded charged continuations are followed through the remaining copper and air; their directions share the event rotation. Subsequent copper reactions are limited by a separate upstream generation limit, set to three in the standard minibeam configurations. Existing particles continue electromagnetic transport when further inelastic branching is suppressed. Copper-local deposits, neutral products, and remaining untracked reaction energy stay in the beamline balance.

Carbon minibeams use independently extracted, species-resolved Urban and elastic data for copper fragments. The proton minibeam upstream-fragment path instead retains its Highland-based angular update; it is not silently identified with the carbon native-fragment path. Both use unrestricted electronic loss and retain their configured inelastic continuation. This distinction concerns copper-born products and does not change the common Urban transport of registered charged ions within the patient.

Energy loss, Coulomb scattering, nuclear elastic scattering, and inelastic fragmentation jointly produce the patient-entry distribution. The full joint position–direction–energy–species state is passed downstream. Consequently, particles scattered out of copper and surviving charged reaction products can contribute to the patient dose, while purely geometric slit transmission is insufficient to represent the upstream model. No electron, neutral-particle, or unmodeled decay transport is implied by this charged-particle handoff.

## II.D. GPU implementation and scored quantities

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
\tag{11}
\]

Batch and device contributions are summed after applying the same W/N factor; no additional batch-count multiplier is applied. References scored as cumulative dose for the simulated particles are compared using their corresponding source counts and cumulative scale, without imposing prescription normalization. Batch files are merged at higher precision on the host, which does not eliminate within-batch FP32 accumulation error.

### II.D.3. Dose-averaged electronic LET

Electronic LET is scored using a stopping-power track-length estimator. For each finite charged-ion segment k in voxel i, \(S_{e,k}\) is the unrestricted linear electronic stopping power at the pre-step kinetic energy and \(\ell_k\) is the accepted true path length. The two additive moments and their ratio are

\[
 A_i=\sum_{k\in i}S_{e,k}^{\,2}\ell_k,\qquad
 B_i=\sum_{k\in i}S_{e,k}\ell_k,\qquad
 L_{d,i}=\frac{A_i}{B_i}.
\tag{12}
\]

The ratio has units of MeV/mm, numerically equal to keV/µm. The electronic-loss expectation supplies the dose weight; sampled energy loss divided by step length is not substituted for stopping power. The estimator is not normalized by material density. Stopping for registered ions uses the same unrestricted response as transport. Additional recoils use a separate electronic-only stopping table matched to the isotope and energy coverage of their electronic-plus-nuclear transport table.

All transported charged ions, including primary ions, fragments, and recoils, contribute within the patient or water scoring medium. Upstream copper/air segments, nuclear-vertex deposits, zero-path termination deposits, and untracked neutral energy do not contribute. The LET scoring population uses the same 0.1 MeV total-kinetic-energy threshold as the standard ion transport cutoff. LET is therefore defined for scored finite tracks and is not obtained by dividing a stopping-weighted sum by the complete absorbed-dose image.

Both moments are accumulated with FP32 GPU atomics, saved independently, and summed across fields, batches, and devices before taking their ratio. LET images are neither averaged directly nor multiplied by prescription-dose normalization. A zero denominator denotes no scored track and is retained as such during analysis. Enabling the scorer does not change transport physics or random draws, although additional atomic operations can change accumulation order and the last bits of the scored dose.

## II.E. Full-plan reference calculations and comparison design

### II.E.1. Cases, grids, and simulated histories

Full-plan dose comparisons include RT07575 (Head 1), RT06423 (Head 2), and 20022516 (Lung 1), covering head anatomy and heterogeneous low-density lung. Broadbeam calculations omit the copper slit collimator, whereas minibeam calculations retain their plan-specific collimators and fields. CT, source optics, spot energies and weights, placement, geometry, and normalization are matched within each comparison. The table specifies the archived full-plan dose-reference dataset and its matched historical GPU statistics. Calculations using the unrestricted-loss revision are identified by their own version and history count; historical GPU agreement is not assigned to a newer implementation. Counts are simulated incident histories, not the total physical particles prescribed.

| Case | Proton broadbeam | Carbon broadbeam | Proton minibeam | Carbon minibeam |
|---|---:|---:|---:|---:|
| RT07575 / Head 1 | 250,626,500 | 129,638,170 | 1,000,000,000 | 1,000,000,000 |
| RT06423 / Head 2 | 124,266,340 | 151,087,660 | 1,000,000,000 | 1,000,000,000 |
| 20022516 / Lung 1 | 361,050,622 | 177,173,040 | 3,000,000,000 | Not included in the completed dataset |

Completed plans use voxel spacings of 0.5×0.5×2.0 mm³, with calculation axes potentially permuted by patient placement. Dimensions, origins, and axis ordering follow each plan’s accompanying reference headers. Comparisons are performed in physical coordinates, retaining the original grid and dose scale without peak alignment, post hoc registration, or dose fitting. A search increment smaller than the voxel spacing provides denser interpolation sampling, not additional spatial resolution.

### II.E.2. TOPAS references and model matching

TOPAS uses the corresponding materials, source, and geometry with DoseToMedium scoring[7]. Reference physics lists are archived per plan and include electromagnetic, hadron/ion inelastic, nuclear elastic, stopping, and decay modules. Minibeam references use AllChargedUrbanPhysics for the corresponding charged-particle scattering configuration. Carbon references additionally require verification of actual GenericIon nuclear elastic registration. Geant4 data versions, electromagnetic energy limits, production cuts, copper geometry, and step settings are obtained from executed inputs and logs rather than inferred from module names. Production cuts and maximum transport steps are reported separately. For example, the lung proton minibeam reference uses a 0.05 mm production cut and a 10 GeV electromagnetic table limit, whereas the three historical carbon broadbeam references retain a 600 MeV electromagnetic table limit; these settings are not assumed to be identical across references.

The current full-plan comparison uses unthresholded TOPAS dose scored after independent transport with plan-weighted source allocation. Weighted dose-influence matrices used during optimization are separately identified auxiliary data and do not replace full recalculation. Original-TPS-weight carbon broadbeam references retain their cumulative simulated-dose scale; optimized minibeams retain the corresponding per-irradiation scale. Differences between native TOPAS transport and the GPU finite event library, local ionization deposition, mean secondary losses, and omitted nuclear neutral transport are retained when interpreting dose discrepancies.

### II.E.3. Matched TOPAS LET reference

LET comparisons use a two-moment TOPAS scorer implementing Eq. (12), with track weights included in each moment. The scorer evaluates unrestricted electronic stopping from an immutable response extracted in a separate process using the reference Geant4 version, physics list, particle identities, and actual material compositions, densities, and excitation energies. Coverage includes the charged isotopes and materials encountered by the full plan. Material mismatches, missing isotopes, and energies outside the response are rejected. Stopping is not queried through mutable live Geant4 models during scoring, so the stopping lookup does not alter transport-model state or random-number consumption. The scoring threshold is matched to 0.1 MeV total kinetic energy.

Pre-step stopping is a step-discretized estimator. The corrected carbon broadbeam reference uses native ionization range-step controls with a range fraction of 0.005 and final range of 0.1 nm, applied to light ions and GenericIon while leaving proton step controls unchanged. These range-relative controls are distinct from a spatial maximum step, production cut, or kinetic-energy cutoff. The executed step settings and nuclear elastic registration are retained with each reference. Dose agreement alone does not demonstrate LET step convergence; additive-moment stability under step refinement and Monte Carlo uncertainty must be examined separately.

The LET campaign is separate from the archived dose-only dataset. RT07575 defines the four-mode comparison design, with completed modes, reference versions, and incident histories reported individually. The corrected carbon broadbeam reference is divided into five independent replicas of 25,927,634 histories each, for a planned total of 129,638,170. Preliminary comparisons use one completed replica and a GPU calculation with the same 25,927,634 histories and integer spot allocation. Incomplete references are identified explicitly; planned totals do not represent completed four-mode validation.

## II.F. Dose and LET agreement, statistical uncertainty, and computational performance

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
\tag{13}
\]

GPU dose is queried by trilinear interpolation, with γ≤1 indicating a pass. For pass-rate evaluation, the spatial search is confined to a sphere of radius \(\Delta r\), since points farther away cannot pass. The BODY mask selects reference evaluation points; GPU dose outside BODY within the search domain is not set to zero. Evaluation masks are fixed from the reference used for each reported comparison and are not reselected from low-statistics repeats.

| Mode | Reference-dose threshold within BODY | Reported local criteria | Spatial search increment |
|---|---|---|---:|
| Broadbeam | >10% of global reference Dmax | 3%/3 mm, 1%/1 mm, 3%/0 mm | 0.1 mm |
| Minibeam | >0.4 Gy, i.e., 10% of a 4 Gy prescription | 3%/0.2 mm, 3%/0.1 mm, 3%/0 mm | 0.02 mm |

The threshold selects the evaluation region only; the local dose tolerance remains the specified percentage of each point's reference dose. Zero-DTA analysis compares doses at the same position and is a local relative-dose-difference test, without the distance division in Eq. (13). The predefined broadbeam criterion is a local 1%/1 mm pass rate >99%. Minibeam submillimeter pass rates are reported directly rather than assessed using the broadbeam criterion. Exploratory analyses using other thresholds or geometric irradiation masks are identified separately.

Additional measures include spatial dose-difference distributions, target mean dose and dose–volume indices, and lateral profiles at different depths to examine minibeam peaks, valleys, and distal structure. Dose differences contain both random fluctuations and systematic discrepancies and are not treated as pure Monte Carlo uncertainty.

### II.F.2. LET comparisons and one-dimensional profiles

Three-dimensional LET gamma uses the same definition as Eq. (13), replacing dose by \(L_d\), with TOPAS LET as the local reference. The evaluation region is BODY intersected with TOPAS dose greater than 10% of its global maximum. Dose defines the mask only; the percentage tolerance is relative to each point's reference LET. Broadbeam LET criteria are 3%/3 mm, 1%/1 mm, and 3%/0 mm, with a 0.1 mm search increment for nonzero distance criteria. Interpolation and spatial search are applied to the GPU LET field. Zero-GPU-denominator voxels within the reference mask are retained as failures; any zero-reference-LET points are identified separately because a local percentage tolerance is undefined there.

Lateral LET profiles at selected geometric depths and central-ray profiles along depth complement voxelwise comparisons. All curves use the same physical coordinates, reference entrance, dose mask, and grid mapping. For the RT07575 carbon broadbeam example, geometric depth is measured from the central-ray BODY entrance along the incident beam direction; it is not water-equivalent depth or a separately shifted entrance for every lateral ray. Lateral profiles retain 0.5 mm sampling. The depth comparison shows both the raw central ray and a transverse pooled profile spanning 2.5×10 mm², retaining 0.5 mm depth sampling without smoothing along depth. Pooled LET is obtained by summing numerator and denominator over the same selected voxels before division. Pooling reduces visible statistical variation but does not replace the voxelwise gamma analysis.

### II.F.3. Dose uncertainty in both calculations

GPU statistical uncertainty is estimated from independent full-plan batches with matched source composition. For B equal-statistics batches whose contributions d_b have already been scaled by the campaign-wide W/N, D=Σ_b d_b and

\[
 \widehat{\sigma}_{D}^{\,2}
 =\frac{B}{B-1}
 \left[\sum_b d_b^2-\frac{D^2}{B}\right].
\tag{14}
\]

For unequal batch sizes, define a_b=N_b/N, Y_b=d_b/a_b, and q=Σ_b a_b². The implemented weighted batch estimator is

\[
 \widehat{\sigma}_{D}^{\,2}
 =\frac{q}{1-q}
 \left[\sum_b a_bY_b^2-D^2\right].
\tag{15}
\]

Equation (15) reduces to Eq. (14) for equal-sized batches. It is used here for batches with nearly equal history counts and matched source composition, without assuming equal variance for arbitrarily imbalanced batches or different sources. Merging across devices retains batch counts, source proportions, and independent seeds.

TOPAS references with native per-history variance use those statistics, converted to final-dose variance according to the actual scorer definition. Variances from independent spots or recalculations are combined using squared dose-scaling factors. The three historical carbon broadbeam references lack stored native variance. For each, a matched 300,000-history TOPAS pilot preserves geometry, physics list, energy spectrum, and relative weights. Its standard error is extrapolated to the full history count by √(N_pilot/N_full), with both quantities expressed on the same full-plan dose scale. The original full-reference dose supplies the relative-error denominator, and the extrapolation is cross-checked against variance between original independent replicas. These entries are identified as estimates; pooled variance across mixed spots can be conservative for a fixed stratified spot allocation.

For a fixed evaluation region R, the root-mean-square relative standard error is

\[
 u_{\rm RMS}=100
 \sqrt{\frac{1}{|R|}
 \sum_{i\in R}\frac{\widehat{\sigma}_i^2}{D_{{\rm ref},i}^2}}
 \ \%.
\tag{16}
\]

GPU and TOPAS uncertainties are reported separately and, for independent simulations, combined through σ_diff²=σ_GPU²+σ_TOPAS². Percentiles and fractions of voxels below selected uncertainty limits are additionally reported where relevant. This is a regional summary of voxelwise one-standard-error values, not uncertainty in the regional mean or a guarantee for every voxel. Spatial correlations prevent direct conversion of voxel variances into confidence intervals for gamma pass rates or dose–volume indices. Finite-library uncertainty and physical-model bias are not included in these random-transport estimates.

### II.F.4. LET uncertainty

LET uncertainty must retain covariance between its numerator and denominator. An auxiliary GPU campaign uses independent, nearly equal-statistics full-plan groups with matched source proportions and a total history count equal to that of the principal calculation. Let \(A_{ib}\) and \(B_{ib}\) be the two voxel moments in group b, let K be the number of groups, and define \(L_{d,i}=\sum_b A_{ib}/\sum_b B_{ib}\). A batch-based first-order ratio-variance estimate is

\[
 \widehat{\sigma}_{L_{d,i}}^{\,2}
 =\frac{K}{K-1}
 \frac{\sum_b\left(A_{ib}-L_{d,i}B_{ib}\right)^2}
 {\left(\sum_b B_{ib}\right)^2}.
\tag{17}
\]

The carbon broadbeam auxiliary calculation uses 20 independent groups. These histories provide an uncertainty estimate at the principal calculation's statistics and are not added to its dose or LET. The estimate is approximate for a finite number of groups, especially where rare high-LET recoils dominate. Voxelwise relative standard errors, regional RMS values, percentiles, and fractions below specified limits are reported within a fixed reference-dose mask.

A single TOPAS LET image does not provide the covariance required for voxelwise LET standard errors. Reference LET uncertainty is estimated only when independent paired numerator/denominator replicas or equivalent covariance statistics are available. Equal GPU and TOPAS history counts are not assumed to imply equal precision. Where both estimates are available, independent GPU and TOPAS LET variances are added for the difference; model bias, response interpolation, step discretization, and FP32 accumulation error remain separate from Monte Carlo standard error.

### II.F.5. Timing and run validity

Performance is reported separately for each GPU model, with RTX 2080 Ti, RTX A6000, and L4 measurements identified individually. Executables contain native device code compiled ahead of time for the corresponding architecture, with runtime PTX JIT disabled. Compilation is excluded from calculation time. Cold start extends from new-process launch to the first saved batch, including input reading, device initialization, physics-data upload, calculation, and output. Steady-state throughput is the total incident histories in subsequent batches divided by their cumulative wall time within the same process, including audits and output. Kernel-only timing does not replace end-to-end throughput. If the file-system cache is not explicitly cleared, process cold start is not interpreted as a cold physical-disk cache. Dose-only and dose-plus-LET timings are distinguished; their scoring overhead is compared with matched histories, source, batch size, physics, and timing boundaries.

Large campaigns distribute histories across GPUs according to measured throughput while preserving campaign-wide source allocation and normalization. Per-device and aggregate multi-device throughput are reported separately. TOPAS single-node throughput is total histories divided by cumulative node computation time, with 56-thread cluster and 96-thread local calculations identified separately. Historical CPU records are distinguished from contemporaneous measurements. Results with different grids, physics, statistical precision, or timing boundaries are not pooled into one speedup.

Primary, copper/air upstream, and charged-secondary stage times provide the computational breakdown. When an algorithm changes the stochastic estimator, time to comparable statistical precision is also considered. Each batch is checked for completed histories, species transport status, physics-table coverage, energy accounting, finite nonnegative dose, completed queues, and zero overflow. LET additionally requires finite nonnegative additive moments and consistent zero denominators. Run audits and matched GPU implementation checks establish computational validity; independent TOPAS dose and LET comparisons assess agreement between models.

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

**Companion material, outside the manuscript Methods text:** [literature and writing analysis](docs/gpu_mc_methods_literature.md), [physics implementation comparison](docs/topas_gpu_physics_comparison.md), [unrestricted-loss model](docs/unrestricted_dedx.md), [LET estimator and matched scorer](docs/let_scoring.md), [data extraction](extensions/README.md), and [full-plan and uncertainty records](docs/full_plan_validation_20261017/README.md).

**Author information required before submission:** case provenance and ethics approval or exemption; complete per-plan source, energy-layer, field, grid, and collimator tables; optimization objectives and solver settings; full species-dependent scattering coefficients and electromagnetic range-step parameters; frozen reference-physics, compiler/driver, and CPU-model inventories; and completed LET reference statistics and step-refinement records. Clinical beam commissioning and independent acceptance of every configuration are not asserted without those records. Dose and LET pass rates, throughput, and measured uncertainty values belong in Results; this section defines how they are obtained.
