# II. Materials and Methods

**GPU Monte Carlo: From Broadbeam to Minibeam** ([中文版](mm_zh.md)).

---

## II.A. Computational Framework, Material Representation, and Beam Modeling

### II.A.1. Overview of the Transport Architecture and Study Scope

We developed **MAIGO**, a cross-platform GPU Monte Carlo engine that integrates proton and carbon-ion broadbeam and multislit minibeam dose calculation within a common pencil-beam scanning (PBS) framework. The transport core couples condensed-history electromagnetic transport with discrete nuclear interactions and explicit tracking of multigeneration charged fragments [1–4]. Minibeam calculations separate transport through the copper multislit collimator and downstream air gap from voxelized patient transport, transferring the joint position–direction–energy–species state at patient entry. This architecture combines explicit aperture modeling [5] with staged GPU execution to extend broadbeam patient transport to the submillimeter peak–valley structure of minibeam radiation therapy (MBRT).

The engine supports primary proton ($Z=1, A=1$) and carbon-12 ($^{12}\mathrm{C}$, $Z=6, A=12$) beams, together with subsequent transport of all supported charged nuclear fragments and elastic target recoils. Scored physical quantities comprise absorbed dose to medium ($D_m$) and, optionally, dose-averaged electronic linear energy transfer ($\mathrm{LET}_d$, denoted $L_d$). The study evaluates physical dose, electronic LET, and computational performance. The orthogonal combinations of primary projectile ($^1\mathrm{H}$ vs. $^{12}\mathrm{C}$), phantom medium (homogeneous water vs. heterogeneous patient CT), and delivery mode (broadbeam vs. copper-collimated minibeam) define eight standardized release configurations. Patient full-plan validations preserve all case-specific clinical beam optics, spot weights, patient placement, collimator geometries, and prescription normalizations, whereas the default CT minibeam preset serves as an independent heterogeneous lung phantom benchmark.

### II.A.2. Voxelized Material Representation and Offline Physics Tables

Patient CT datasets are preprocessed offline into a regular three-dimensional Cartesian grid specified by its physical origin, voxel dimensions, mass density $\rho$, and discrete tissue composition index. Hounsfield units (HU) are converted to mass density and mapped to 25 stoichiometric human tissue elemental compositions according to the Schneider segmentation scheme [6]. Supplementing these 25 tissues with a reference water composition having a mean excitation energy of $I = 75\text{ eV}$ ($\mathrm{Water\_75eV}$) yields 26 distinct material compositions. MAIGO evaluates material-specific electromagnetic and nuclear responses from each voxel’s tissue composition and continuous mass density, incorporating both composition and density into heterogeneous transport.

All underlying physical cross sections and response functions are extracted offline from Geant4 11.3.2 and OpenTOPAS 4.2.3 [7,8]. The runtime physics database is organized into modular binary packages comprising:
1. **Continuous-density ion electromagnetic tables (`continuous26`)**: Unrestricted electronic stopping powers, energy-loss straggling dispersions, and Sternheimer density-effect parameters across all 26 media, from which consistent continuous-slowing-down approximation (CSDA) ranges and inverse ranges are evaluated at runtime.
2. **Beamline electromagnetic tables (`BLEM0001`)**: Independently extracted unrestricted electronic stopping powers and full energy-loss fluctuation dispersions for copper ($\mathrm{G4\_Cu}$) and air ($\mathrm{G4\_AIR}$) covering primary ions and all copper-generated charged fragment species.
3. **Multiple Coulomb scattering (MCS) tables**: Precomputed species- and material-dependent Urban MCS parameters, transport mean free paths, and radiation lengths for water, CT tissues, copper, and air, paired with effective Gaussian Fermi–Eyges scattering parameters.
4. **Nuclear interaction tables and correlated event libraries**: Macroscopic elastic and inelastic cross-section tables (`SCHNRATE` and `SCHN2RAT` v3), isotope-resolved momentum-transfer distributions for nuclear elastic scattering, stopping-power tables for heavy target recoils (`ELRSP001` and `LERSP001`), and correlated inelastic final-state event libraries (`CINPKG04`) that replace online intranuclear cascade evaluations on the GPU.

Initialization checks projectile identity $(Z, A)$, material composition, and energy coverage in the physics tables. Simulations operate within the data domains of the required channels. Physics-data versions, software revision, integer spot allocations, and random seeds are recorded to reproduce the source and physics configuration.

### II.A.3. Pencil-Beam Scanning Source Model and Spot-Weight Optimization

Clinical treatment plans are defined by a sequence of fields, energy layers, and individual scanning spots, each parameterized by its nominal kinetic energy, transverse isocenter coordinates, and particle weight. An energy-dependent phase-space beam model characterizes the optical properties at the nozzle exit (or upstream source plane) via transverse spatial standard deviations $(\sigma_x, \sigma_y)$, angular divergences $(\sigma_{\theta_x}, \sigma_{\theta_y})$, position–angle covariance terms, and Gaussian energy spread $\sigma_E$. For each history, correlated transverse positions and direction cosines are sampled in the local spot coordinate system and projected into the global patient coordinate frame according to the virtual source-to-axis distances (SAD), gantry/couch rotations, and isocenter offset. Incident ion energies are parameterized in kinetic energy per nucleon ($\mathrm{MeV/u}$); treatment planning system (TPS) inputs tabulated in total kinetic energy ($\mathrm{MeV}$) are converted using the integer mass number $A$.

Physical-dose inverse optimization represents voxel dose as $\mathbf{d} = \mathbf{A}\mathbf{w}$, where $\mathbf{A}$ is the dose-influence matrix per incident particle and $\mathbf{w} \ge \mathbf{0}$ contains the spot particle weights. Weights are optimized for case-specific target coverage and organ-at-risk (OAR) sparing. The final vector is fixed for both MAIGO and TOPAS full-plan forward calculations, establishing a common irradiation source. Reoptimized plans use a $4\text{ Gy}$ physical-dose prescription; carbon-ion broadbeam plans retaining treatment planning system (TPS) weights use the original cumulative dose scale.

---

## II.B. Ion Transport Physics in Homogeneous Water and Heterogeneous CT

### II.B.1. Unrestricted Electronic Stopping Power, Continuous Density Effect, and CSDA Range

Each charged particle track maintains its 3D position $\mathbf{r}$, unit direction vector $\mathbf{u}$, total kinetic energy $E$, nuclear identity $(Z, A)$, and deterministic random-number stream state. Electromagnetic tables parameterized in specific energy $e = E/A$ ($\mathrm{MeV/u}$) are queried using the exact species identity, local material composition, and voxel mass density $\rho$. MAIGO employs the unrestricted linear electronic stopping power $S_e(E; \rho)$, which integrates energy transfers to ionization electrons up to the kinematic maximum $T_{\max}$. Native species-resolved stopping tables retain each ion’s effective-charge ($Z_{\mathrm{eff}}$) dependence and Barkas/Bloch corrections.

In heterogeneous CT media, the unrestricted stopping power at actual voxel density $\rho$ is evaluated by combining mass-density scaling of the reference composition response with a continuous Sternheimer density-effect correction:

$$
S_e(E; \rho) = \frac{\rho}{\rho_0}\, S_e(E; \rho_0) - C_{\mathrm{mat}}(\rho)\,\frac{Z_{\mathrm{eff}}^2}{\beta^2}\left[\delta(X; \rho) - \delta(X; \rho_0)\right],
\tag{1a}
$$

where $\rho_0$ is the base density of the tissue composition, $\beta = v/c$ is the relativistic velocity, $X = \log_{10}(\beta\gamma)$, and $\delta(X; \rho)$ is the continuous Sternheimer parameterization interpolated across the voxel density $\rho$. For a fixed local composition and density, the unrestricted CSDA range $R(E)$ and the finite-step mean energy loss $\overline{\Delta E}(E, \ell)$ over a true path length $\ell$ are defined by

$$
R(E) = \int_0^E \frac{\mathrm{d}E'}{S_e(E')}, \qquad
\overline{\Delta E}(E, \ell) = E - R^{-1}\!\left(\max\!\left[R(E) - \ell,\, 0\right]\right).
\tag{1b}
$$

Stopping power, range, and inverse range share a piecewise-linear response. The reference CSDA range $R_0(E)$ is constructed analytically by integrating the reciprocal of the piecewise-linear stopping response over each energy knot interval $[E_i, E_{i+1}]$, where $S_e(E)$ varies linearly from $S_i$ to $S_{i+1}$:

$$
\Delta R_i = \int_{E_i}^{E_{i+1}} \frac{\mathrm{d}E'}{S_e(E')} = \frac{E_{i+1} - E_i}{S_i}\,\frac{\ln(1 + q_i)}{q_i}, \qquad q_i = \frac{S_{i+1} - S_i}{S_i},
\tag{1c}
$$

with a fourth-order Taylor expansion applied for $|q_i| < 10^{-3}$ to prevent floating-point cancellation in single precision (FP32). The inverse mapping $R_0^{-1}$ within each interval is evaluated in closed form via the exponential inverse $\mathrm{expm1}$, while stopping below the lowest energy knot $E_0$ follows the prescribed $\sqrt{E}$ continuation. Whenever the kinetic energy lies below the density-effect onset (where $\delta(X; \rho) = \delta(X; \rho_0) = 0$), the actual range simplifies exactly to $R(E; \rho) = (\rho_0 / \rho) R_0(E)$. Above the density-effect onset, the residual integral of the reciprocal-stopping difference is evaluated via composite 8-point Gauss–Legendre quadrature in logarithmic energy, and the inverse range $R^{-1}$ is solved to a relative tolerance of $2\times 10^{-6}$ using a safeguarded Newton–bisection iteration.

During particle stepping in water and patient CT, the trial linear energy loss $\Delta E_{\mathrm{lin}} = S_e(E)\,\ell$ is compared against the native linear-loss threshold $\xi_{\mathrm{lin}} E$ stored in the EM package. For fractional losses below $\xi_{\mathrm{lin}}$, non-ion models (including protons) set $\overline{\Delta E} = S_e(E)\,\ell$ following the short-step convention of `G4VEnergyLossProcess`. For heavier ions ($Z > 1$) below $\xi_{\mathrm{lin}}$, the engine retains a midpoint-energy predictor–corrector evaluation, querying $S_e(E - \tfrac{1}{2} S_e(E)\ell)\,\ell$ to account for rapid effective-charge variation. Steps exceeding $\xi_{\mathrm{lin}} E$ and range-terminating steps evaluate the exact CSDA range inversion in Eq. (1b). Heavy target recoils outside the standard 18-ion transport registry use dedicated isotope-specific total (electronic plus nuclear) stopping tables (`ELRSP001`), as detailed in Section II.B.6.

Water and patient CT transport use the local-ionization energy-deposition approximation: unrestricted electronic energy loss is scored in the current transport voxel, and secondary $\delta$-ray electrons and bremsstrahlung/atomic-relaxation photons are not transported as separate tracks. Electronic energy transfer is thus incorporated into condensed-history ion stepping. Dose evaluation covers head and low-density lung cases, with transverse minibeam profiles used to assess the peak–valley structure.

### II.B.2. Stochastic Energy-Loss Straggling

Energy-loss straggling for primary protons and carbon ions is modeled using a unified stochastic sampling engine adapted from the Geant4 11.3.2 `G4UniversalFluctuation` and `G4IonFluctuations` algorithms [8]. For an incident ion of rest mass $M$, speed $\beta c$, and Lorentz factor $\gamma$, the maximum energy transferable to a free electron in a single collision is

$$
T_{\max} = \frac{2 m_e c^2 \beta^2 \gamma^2}{1 + 2\gamma(m_e/M) + (m_e/M)^2},
\tag{2}
$$

where $m_e c^2 = 0.51099891\text{ MeV}$ is the electron rest energy. Because MAIGO operates in an unrestricted condensed-history scheme without discrete $\delta$-ray production cuts, $T_{\max}$ serves directly as the upper energy transfer limit for straggling. For `continuous26` medium tables whose base variance per unit length $V_{\mathrm{ref}}$ was extracted at a finite reference cut $T_c$, the runtime engine analytically restores the full unrestricted Bohr/Bethe energy-loss variance per unit length $V_{\mathrm{full}}$ at the current voxel density via

$$
V_{\mathrm{full}} = V_{\mathrm{ref}} \, \frac{T_{\max}/\beta^2 - T_{\max}/2}{T_{\max}/\beta^2 - \min(T_c, T_{\max})/2},
\tag{3}
$$

where $V_{\mathrm{ref}}$ already incorporates local mass-density scaling and the native effective-charge fluctuation factor. Multiplying $V_{\mathrm{full}}$ by the accepted true step length $\ell$ yields the step variance $\sigma_E^2$. For upstream copper and air (`BLEM0001`), the unrestricted variance at $T_{\max}$ is tabulated directly offline and requires no cut-conversion factor.

The fluctuation engine selects the appropriate physical regime based on particle identity and velocity:
1. **Universal fluctuation regime ($^1\mathrm{H}$ at all energies and $Z > 1$ ions above the native transition energy $E > (10\text{ MeV}/m_p)\,Z M$)**: In the thick-absorber regime ($\overline{\Delta E} \ge 10\,T_{\max}$), energy loss is drawn from a two-sided truncated Gaussian distribution on $[0, 2\overline{\Delta E}]$ when $\overline{\Delta E}/\sigma_E \ge 2$, or from a moment-matched Gamma distribution with shape parameter $k = (\overline{\Delta E}/\sigma_E)^2$ when $\overline{\Delta E}/\sigma_E < 2$. In the thin-absorber regime ($\overline{\Delta E} < 10\,T_{\max}$), the energy loss is decomposed into atomic excitation ($a_1$) and continuum ionization ($a_3$) components following the Urban two-level atom model. To avoid numerical instability when the expected number of ionization collisions $a_3 > 8$, the ionization component is split algebraically into a Gaussian-sampled dense soft part and a bounded explicit Poisson tail with mean $p_3 = 8 a_3 / (8 + a_3) \le 8$, from which single-collision transfers are drawn according to a $1/E^2$ distribution over $[\alpha E_0, T_{\max}]$.
2. **Low-energy Ion fluctuation regime ($Z > 1$ ions below the transition threshold)**: The raw ion dispersion is corrected for finite-step velocity deceleration when $\overline{\Delta E} > 0.2 E$. Depending on the signal-to-noise ratio $\mathrm{SN} = \overline{\Delta E}/\sigma_E$, the actual energy loss is sampled from a truncated Gaussian ($\mathrm{SN} \ge 2$), a Gamma distribution ($0.1 < \mathrm{SN} < 2$), or a uniform distribution on $[0, 2\overline{\Delta E}]$ ($\mathrm{SN} \le 0.1$).

Sampled energy loss is bounded by the available kinetic energy $E$; range-terminating steps deposit the residual kinetic energy. All eight configurations retain primary energy-loss straggling in the patient/water phantom and copper collimator to model primary range straggling and Bragg-peak broadening. Charged secondary fragments use finite-path mean electronic energy loss ($\Delta E = \overline{\Delta E}$), together with stochastic nuclear interactions and multiple scattering.

### II.B.3. Step-Size Selection, Voxel Boundary Crossing, and Track Termination

At the beginning of each transport step, the local material composition, mass density, and electromagnetic response are evaluated to determine the maximum permissible step length across competing physical and geometric processes. Let $f$ denote the species-specific range fraction, $r_f$ the final-range parameter, and $\eta$ the maximum fractional mean energy loss per step. When the residual CSDA range satisfies $R(E) > r_f$, the electromagnetic step proposal $\ell_{\mathrm{EM}}$ is governed by the dual range-guided and energy-loss constraint:

$$
\ell_{\mathrm{EM}} = \min\!\left[
f R(E) + r_f(1 - f)\!\left(2 - \frac{r_f}{R(E)}\right),\;
R(E) - R\!\left((1 - \eta)E\right)
\right].
\tag{4}
$$

Once the particle slows to $R(E) \le r_f$, the proposal collapses to $\ell_{\mathrm{EM}} = R(E)$, and the track terminates at the end of that step. In heterogeneous CT broadbeam transport, the fractional energy-loss limit is set to $\eta = 0.05$, whereas minibeam patient transport enforces a tighter $\eta = 0.01$ to preserve high spatial fidelity across steep lateral dose gradients.

The electromagnetic proposal $\ell_{\mathrm{EM}}$ competes with the region-specific hard step cap $\ell_{\max}$ (Table 1), the distance to the next voxel or material boundary $\ell_{\mathrm{geom}}$, the multiple-scattering step limit, and the sampled nuclear interaction distance $\ell_{\mathrm{nuc}}$.

**Table 1.** Enforced maximum ion step lengths ($\ell_{\max}$) and principal multiple Coulomb scattering (MCS) models across transport regions.

| Transport Region | Maximum Ion Step $\ell_{\max}$ | Principal Multiple Scattering Model |
|---|---:|---|
| Water / CT broadbeam | $1.0\text{ mm}$ | Correlated Gaussian Fermi–Eyges |
| CT minibeam (patient volume) | $0.5\text{ mm}$ | Data-driven Urban (primaries); segmented Fermi–Eyges (charged secondaries, $0.1\text{ mm}$ segment cap) |
| Water minibeam (phantom volume) | $0.5\text{ mm}$ | Data-driven Urban (primaries and charged secondaries) |
| Minibeam copper collimator | $0.05\text{ mm}$ | Copper-specific Urban MCS + discrete nuclear elastic scattering |
| Minibeam slit air (CT carbon preset) | $0.2\text{ mm}$ | Air-specific Urban MCS |

The interaction of geometry and scattering depends on the active MCS formalism:
- Under **Fermi–Eyges transport**, the step length is evaluated directly along the geometric ray, and lateral displacement is applied at the end of each scattering sub-segment.
- Under **Urban transport**, the algorithm explicitly distinguishes the curved microscopic true path length $\ell$ from the straight geometric chord length $g$. Geometric distances to voxel faces or collimator sidewalls are first computed in Euclidean space and converted to an admissible true path length $\ell_{\mathrm{geom}}$. Energy loss and nuclear optical depth are then integrated along the true path $\ell = \min(\ell_{\mathrm{EM}}, \ell_{\max}, \ell_{\mathrm{geom}})$. If a discrete nuclear interaction occurs at $\ell_{\mathrm{nuc}} < \ell$, the true step is truncated to $\ell_{\mathrm{nuc}}$, and the corresponding geometric advance and angular deflection are recomputed for the shortened step prior to final state commitment.

Continuous ionization energy loss is scored in the originating voxel of the step, whereas local energy deposition from a discrete nuclear interaction is scored at the post-step collision vertex. Particles exiting the outer bounding box of the patient/phantom grid are terminated and recorded in the boundary-escape energy ledger. Tracks slowing below the global kinetic energy cutoff $E_{\mathrm{cut}} = 0.1\text{ MeV}$ (total kinetic energy) are terminated in place and deposit their residual kinetic energy locally as a zero-path terminal deposit, which is excluded from finite-track LET accumulation.

### II.B.4. Multiple Coulomb Scattering via the Fermi–Eyges Theory

In broadbeam configurations—and for charged secondary fragments inside the patient volume in dedicated CT minibeam presets—multiple Coulomb scattering is modeled using the correlated Gaussian Fermi–Eyges theory [9]. Within a homogeneous sub-step of length $L$ characterized by linear scattering power $\mathcal{T}$, the projected angular deflection $\theta$ and lateral spatial displacement $y$ along each of two orthogonal transverse axes follow a bivariate normal distribution with covariance matrix

$$
\boldsymbol{\Sigma}_{\mathrm{FE}} =
\begin{pmatrix}
\langle \theta^2 \rangle & \langle \theta y \rangle \\
\langle \theta y \rangle & \langle y^2 \rangle
\end{pmatrix}
= \mathcal{T}
\begin{pmatrix}
L & L^2/2 \\
L^2/2 & L^3/3
\end{pmatrix}.
\tag{5}
$$

Correlated pairs $(\theta_x, x)$ and $(\theta_y, y)$ are sampled via Cholesky decomposition in the instantaneous local frame of the particle and rotated into global Cartesian coordinates, ensuring exact phase-space correlation between lateral beam broadening and angular divergence. When a transport step exceeds the configured scattering sub-segment limit (e.g., the $0.1\text{ mm}$ segment limit for CT minibeam charged secondaries), the step is subdivided into uniform sub-segments and $\mathcal{T}$ is evaluated at the midpoint kinetic energy of each segment.

The effective linear scattering power $\mathcal{T}$ combines a Rossi–Greisen Gaussian core with an effective second-moment contribution from large-angle Rutherford single-scattering tails:

$$
\mathcal{T}(E) = \left(\frac{E_s Z}{\beta p c}\right)^2 \frac{1}{X_0(\rho)} + \lambda_{\mathrm{tail}}(\rho)\,\left(\frac{E_{\mathrm{tail}} Z}{\beta p c}\right)^2,
\tag{5b}
$$

where $p$ is the relativistic momentum, $X_0(\rho)$ is the local radiation length derived from the Schneider tissue composition and voxel density, and $\lambda_{\mathrm{tail}}(\rho) = \lambda_{\mathrm{tail}, \mathrm{water}} \, [X_{0,\mathrm{water}} / X_0(\rho)]$ scales the nominal tail rate by inverse radiation length. For light ions ($^1\mathrm{H}$, $^2\mathrm{H}$, $^3\mathrm{H}$, and $^4\mathrm{He}$), the effective core energy parameter $E_s(E/A)$ and tail parameters are obtained by piecewise-linear interpolation of homogeneous water fits calibrated against Geant4 at $50$, $150$, and $300\text{ MeV/u}$ (clamped to endpoint values outside this interval). For $^{12}\mathrm{C}$ and heavier fragments assigned to the same calibration group, the engine adopts calibrated constants $E_s = 9.9\text{ MeV}$, $\lambda_{\mathrm{tail},\mathrm{water}} = 0.0025\text{ mm}^{-1}$, and $E_{\mathrm{tail}} = 2.4\text{ MeV}$. Strong-interaction nuclear elastic scattering is modeled independently as a discrete stochastic process (Section II.B.6).

### II.B.5. Macroscopic Nuclear Cross Sections and Elastic–Inelastic Competition

Discrete nuclear interactions in water and CT media encompass both coherent nuclear elastic scattering ($\mathrm{el}$) and non-elastic nuclear fragmentation/inelastic reactions ($\mathrm{in}$). For reaction channel $c \in \{\mathrm{el}, \mathrm{in}\}$, the macroscopic interaction cross section (linear interaction rate per unit path length, $\mathrm{mm}^{-1}$) at kinetic energy $E$ is summed over the elemental constituents $t$ of the local tissue composition:

$$
\Sigma_c(E; \rho) = \sum_{t} n_t(\rho)\,\sigma_{c,t}(E) = \rho \sum_{t} \left(\frac{\Sigma_{c,t}}{\rho}\right)_{\!\mathrm{tab}}\!(E), \qquad
P(t \mid c, E) = \frac{n_t(\rho)\,\sigma_{c,t}(E)}{\Sigma_c(E; \rho)},
\tag{6}
$$

where $n_t(\rho)$ is the atomic number density of target element $t$ at voxel density $\rho$, $\sigma_{c,t}(E)$ is the microscopic cross section, and $P(t \mid c, E)$ is the conditional probability of interacting with target nucleus $t$. Tabulated mass interaction rates $(\Sigma_{c,t}/\rho)_{\mathrm{tab}}$ are precomputed on a uniform $0.5\text{ MeV/u}$ grid spanning $0.1$–$460.1\text{ MeV/u}$ (921 nodes) across all 25 Schneider tissues and water. Inelastic partial rates are strictly masked to the validated projectile–target $(Z, A, t)$ energy domains stored in the binary rate tables.

Both primary ions and charged secondary fragments track nuclear interaction distances using a persistent dimensionless optical depth $\tau$:

$$
\tau = -\ln U, \qquad \Sigma_{\mathrm{tot}}(E; \rho) = \Sigma_{\mathrm{el}}(E; \rho) + \Sigma_{\mathrm{in}}(E; \rho),
\tag{7}
$$

where $U \sim \mathcal{U}(0, 1)$ is drawn independently for each particle track. Along a candidate step of true path length $\ell$, the remaining optical depth is decremented by $\Delta\tau = \Sigma_{\mathrm{tot}}\,\ell$. If $\Delta\tau < \tau$, the particle completes the full step without a nuclear collision, and the residual optical depth $\tau \leftarrow \tau - \Delta\tau$ is preserved across voxel boundaries and kernel continuation stages. If $\Delta\tau \ge \tau$, the step is truncated to the nuclear interaction point $\ell_{\mathrm{nuc}} = \tau / \Sigma_{\mathrm{tot}}$, after which $\tau$ is resampled. At the collision vertex, the interaction channel is selected between elastic and inelastic scattering with probabilities $\Sigma_{\mathrm{el}}/\Sigma_{\mathrm{tot}}$ and $\Sigma_{\mathrm{in}}/\Sigma_{\mathrm{tot}}$, respectively, and the target nucleus $t$ is sampled from the discrete cumulative distribution of $P(t \mid c, E)$.

To reduce redundant global-memory cross-section table lookups during primary ion transport, primary inelastic proposals utilize a per-track majorant rate cache $\widehat{\Sigma}_{\mathrm{in}}$. The cached rate is initialized and refreshed whenever the particle enters a new material composition or density, when its kinetic energy drops below $80\%$ of the energy at which the cache was last evaluated, or immediately following a candidate nuclear event. At a proposed inelastic vertex, the exact post-slowing inelastic rate $\Sigma_{\mathrm{in}}(E_{\mathrm{post}})$ is evaluated and accepted with probability $\Sigma_{\mathrm{in}}(E_{\mathrm{post}}) / \widehat{\Sigma}_{\mathrm{in}}$; rejected proposals are treated as null collisions (delta-tracking), allowing the primary ion to continue transport seamlessly.

### II.B.6. Nuclear Elastic Scattering and Target-Recoil Transport

Coherent nuclear elastic scattering is enabled across all eight release presets. Once an elastic collision and target element $t$ are selected at incident specific energy $e = E/A$, the engine brackets $e$ between adjacent table nodes $[e_0, e_1]$ and stochastically selects node $k \in \{0, 1\}$ in proportion to each node's partial elastic rate contribution. An empirical sample is then drawn uniformly from the 512 pre-extracted records at node $k$, providing the specific target isotope mass $m_t$ and the normalized squared four-momentum transfer $q^2 / q_{\max}^2 = \tfrac{1}{2}(1 - \cos\theta_*)$, where $\theta_*$ is the center-of-mass polar scattering angle. Combining $\cos\theta_*$ with a uniform azimuthal angle $\phi \sim \mathcal{U}(0, 2\pi)$, exact relativistic two-body kinematics (in natural units $c = 1$) determines the laboratory-frame recoil kinetic energy $E_R$ and scattered projectile kinetic energy $E'_p$:

$$
s = (m_p + m_t)^2 + 2 m_t E, \qquad
p_* = \frac{m_t \sqrt{E(E + 2 m_p)}}{\sqrt{s}}, \qquad
E_R = \frac{p_*^2 (1 - \cos\theta_*)}{m_t}, \qquad
E'_p = E - E_R,
\tag{8}
$$

where $m_p$ is the projectile rest mass (retaining the package-calibrated mass convention, such as $12\text{ u}$ for $^{12}\mathrm{C}$ where specified) and $p_*$ is the center-of-mass momentum magnitude. The three-momentum vectors of both the scattered projectile and the recoiling target nucleus are rotated from the current projectile direction into the global coordinate frame.

Target recoils whose nuclear identity $(Z_R, A_R)$ belongs to the standard 18-species charged-ion registry enter the secondary transport queue and undergo full electromagnetic, multiple-scattering, and nuclear transport. Heavy target recoil isotopes outside the standard 18-ion registry (e.g., $^{12}\mathrm{C}$–$^{40}\mathrm{Ca}$ or $^{16}\mathrm{O}$ tissue recoils) are transported using a dedicated all-ion recoil stopping package (`ELRSP001`) that tabulates the combined unrestricted electronic plus nuclear stopping power $S_{\mathrm{tot}}(E) = S_e(E) + S_n(E)$ and its consistent CSDA range; this recoil group uses continuous slowing-down transport. When scoring dose-averaged electronic LET ($L_d$), these heavy recoils query a companion electronic-only stopping table (`LERSP001`) so that nuclear stopping power $S_n(E)$ is strictly excluded from both the numerator and denominator of the electronic LET estimator. Recoils born with $E_R \le E_{\mathrm{cut}} = 0.1\text{ MeV}$ deposit their energy locally at the collision vertex.

### II.B.7. Correlated Nuclear Inelastic Final-State Sampling and Fragment Transport

Non-elastic nuclear interactions sample complete events from pre-generated Geant4 intranuclear cascade final-state libraries (`CINPKG04`), preserving multiparticle final-state correlations in a GPU-queryable representation indexed by projectile $(Z, A)$, target element $t$, and incident energy per nucleon $e_k$. The installed primary proton event library was generated offline using the Geant4 Binary Cascade model (`QGSP_BIC_HP`), the primary $^{12}\mathrm{C}$ library using the Liège Intranuclear Cascade model (`FTFP_INCLXX`), and the secondary fragment library using `G4IonBinaryCascadePhysics` paired with `G4EmStandardPhysics_option4`.

For an inelastic interaction occurring at specific energy $e = E/A$ bracketed by library energy nodes $e_0 \le e \le e_1$ (with a maximum permissible node spacing of $5\text{ MeV/u}$), stochastic linear interpolation selects the upper node $e_1$ with probability

$$
P(e_1 \mid e) = \frac{e - e_0}{e_1 - e_0}, \qquad P(e_0 \mid e) = 1 - P(e_1 \mid e),
\tag{9}
$$

A complete interaction record is drawn uniformly from the selected node’s event pool, retaining its recorded product energies and momentum correlations. Equation (9) makes the expected selected-node incident energy equal to the query energy $e$. The incident-energy offset from node selection and the reaction Q-value residual are recorded separately in the energy ledger.

A single uniform azimuthal rotation angle $\Phi \sim \mathcal{U}(0, 2\pi)$ is applied coherently to all reaction products of the selected event before rotating their momentum vectors from the incident particle frame into global coordinates. The shared rotation preserves product multiplicity, transverse momentum balance, and energy–angle correlations within each fragmentation event. Recorded sub-threshold local vertex energy is deposited immediately in the collision voxel. Outgoing charged fragments with kinetic energy above $E_{\mathrm{cut}} = 0.1\text{ MeV}$ are pushed to the secondary transport queue with their exact $(Z, A)$, kinetic energy, direction vector, generation counter $g \leftarrow g + 1$, and deterministic descendant RNG key; sub-cutoff charged products deposit their kinetic energy locally.

Secondary inelastic cascades in water and patient CT use a maximum generation depth of $g_{\max} = 2$. Charged fragments at this limit, or with species policies that disable inelastic branching, continue continuous slowing-down, multiple scattering, and permitted nuclear elastic scattering until stopping or escape. $^6\mathrm{Be}$ residuals terminate at production, with their kinetic energy recorded separately as untracked energy. Neutral nuclear products, including neutrons and de-excitation $\gamma$ rays, are not transported further; their kinetic energy and untracked reaction residuals are recorded separately and excluded from patient dose.

---

## II.C. Extension to Multislit Copper-Collimated Minibeams

### II.C.1. Multislit Collimator Geometry and Explicit Upstream Phase-Space Generation

In the minibeam configurations, a divergent or parallel multislit copper collimator and an intervening downstream air gap are interposed between the scanning nozzle exit and the patient surface. The geometric model specifies the collimator outer dimensions, thickness along the beam axis, slit count, slit aperture width, center-to-center pitch, slit length, array lateral phase offset, azimuthal rotation, and the exact axial coordinates of the collimator exit, CT bounding box, skin surface, and treatment isocenter. The collimator remains stationary in the field coordinate system while pencil-beam spots scan across its upstream entrance face.

Two clinical collimator designs are implemented for full-plan evaluations:
1. **Proton minibeam collimator**: A $60\text{ mm}$-thick copper block containing $15$ parallel slits of $0.5\text{ mm}$ aperture width, $3.6\text{ mm}$ center-to-center pitch, and $50\text{ mm}$ slit length.
2. **Carbon-ion minibeam collimator (RT07575 head plan)**: A $100\text{ mm}$-thick copper block containing $17$ slits of $0.7\text{ mm}$ aperture width, $3.0\text{ mm}$ center-to-center pitch, and $100\text{ mm}$ slit length. For the opposing $90^\circ$ and $270^\circ$ lateral fields of the RT07575 plan, the $270^\circ$ slit array incorporates a half-pitch ($1.5\text{ mm}$) lateral phase shift relative to the $90^\circ$ field to interleave opposing peak and valley dose distributions.

MAIGO transports incident histories and their charged nuclear products through the air slits, copper septa, and downstream air gap to generate the patient-entry phase space. The resulting distribution includes directly transmitted primaries, primaries scattered from copper sidewalls, and charged nuclear fragments generated in copper and upstream air, carrying collimator-induced beam modulation and secondary-particle contributions into patient transport. Histories stopped or absorbed in copper remain part of the total incident history count $N$ used for absolute dose normalization.

### II.C.2. Data-Driven Urban Multiple Coulomb Scattering and True-to-Geometric Path Conversion

Because accurate prediction of minibeam peak-to-valley dose ratios (PVDR) and penumbral sharpness requires high-fidelity modeling of single- and plural-scattering tails near slit edges, MAIGO implements a data-driven adaptation of the Geant4 Urban multiple Coulomb scattering model (`G4UrbanMscModel`) [8] for all upstream copper/air primary transport and all minibeam patient primary transport. Within the patient/phantom volume, water minibeam presets and the generic minibeam pipeline also apply Urban MCS to charged secondary fragments, whereas the dedicated proton and carbon CT minibeam pipelines employ a hybrid scheme: Urban MCS for patient primaries (with $\ell_{\max} = 0.5\text{ mm}$) coupled with segmented Fermi–Eyges scattering ($\ell_{\max} = 0.5\text{ mm}$, scattering sub-segment cap $0.1\text{ mm}$) for patient-medium charged secondaries (including upstream-born fragments once they cross into the patient CT grid).

The offline-extracted Urban package (`ct_copper.bin` and `air.bin`) tabulates species- and material-dependent transport mean free paths $\lambda_{\mathrm{tr},1}(E)$, radiation lengths $X_0$, and angular distribution shape parameters across 52 charged ion species in water, all 25 Schneider CT tissues, and copper (and for $^1\mathrm{H}$ and $^{12}\mathrm{C}$ in air). Urban transport couples scattering and geometric advancement through a mapping between the true path length $\ell$ and geometric chord length $g$. In the regime of locally constant transport mean free path $\lambda_{\mathrm{tr},1}$, the forward mapping and its exact inverse are given by

$$
g(\ell) = \lambda_{\mathrm{tr},1}\left(1 - e^{-\ell/\lambda_{\mathrm{tr},1}}\right), \qquad
\ell(g) = -\lambda_{\mathrm{tr},1}\ln\!\left(1 - \frac{g}{\lambda_{\mathrm{tr},1}}\right),
\tag{10}
$$

supplemented by the native Geant4 energy-dependent $\lambda_{\mathrm{tr},1}(E)$ branch when stopping substantially alters the mean free path along the step. Whenever a particle approaches a copper–air slit sidewall, collimator exit plane, or CT voxel boundary at Euclidean distance $d_{\mathrm{geom}}$, the geometric advance is first bounded by $g \le d_{\mathrm{geom}}$ and inverted via Eq. (10) to obtain the maximum admissible true path $\ell_{\mathrm{geom}}$. Continuous energy loss and nuclear interaction probabilities are evaluated along the true path $\ell$, coupling boundary advancement and path integration within the same scattering geometry.

At the conclusion of the true step (after accounting for any step shortening due to discrete nuclear collisions), the polar deflection cosine $\cos\theta$ is sampled from the composite Urban distribution, which blends a Gaussian small-angle core with a power-law single-scattering tail parameterized by the step thickness $\ell / \lambda_{\mathrm{tr},1}$, radiation length $X_0$, and pre-/post-step kinetic energy mean. Correlated lateral displacement is applied in the local particle frame, constrained by the geometric safety radius, and rotated into global coordinates. Boundary-crossing flags and range-step conditioning states are preserved across steps and kernel handoffs.

### II.C.3. Unrestricted Electronic Energy Loss and Straggling in Copper and Air

Charged particles traversing the copper collimator and air regions query the dedicated `BLEM0001` beamline electromagnetic package, which provides unrestricted electronic stopping powers $S_e(E)$ and native unrestricted straggling variances $V_{\mathrm{full}}(E; T_{\max})$ across all charged isotopes present in the copper fragmentation registry. For steps whose trial linear loss satisfies $S_e(E)\,\ell \le 0.02 E$, mean energy loss is computed linearly as $S_e(E)\,\ell$; for larger fractional losses, exact CSDA range inversion (Eq. 1b) is used with a $\sqrt{E}$ stopping continuation below the lowest energy node.

Primary ions in copper and air use the shared Universal/Ion fluctuation engine of Section II.B.2 with native particle mass, charge, material excitation energy, and $T_{\max}$ parameters. Charged secondary fragments use finite-path mean electronic energy loss. Upstream electronic losses follow the local-deposition approximation and enter the beamline energy ledger separately from patient-grid dose. Step caps are $0.05\text{ mm}$ in copper and $0.2\text{ mm}$ in slit air for CT carbon minibeams.

### II.C.4. Nuclear Interactions and Fragment Cascade in the Copper Collimator

Both coherent nuclear elastic scattering and non-elastic nuclear fragmentation are modeled explicitly within the copper collimator across all four minibeam presets. Primary ions sample their distance to the next copper nuclear interaction using the combined macroscopic cross section $\Sigma_{\mathrm{tot},\mathrm{Cu}} = \Sigma_{\mathrm{el},\mathrm{Cu}} + \Sigma_{\mathrm{in},\mathrm{Cu}}$, followed by post-slowing channel selection. In a copper nuclear elastic event, the center-of-mass scattering angle is drawn from copper-specific momentum-transfer tables and applied via relativistic two-body kinematics (Eq. 8); the scattered beam ion continues transport, while the kinetic energy transferred to the heavy copper target nucleus is booked directly to the upstream beamline energy ledger without tracking the sub-micrometer copper recoil.

Copper inelastic collisions sample complete correlated final states from a dedicated copper cascade event library (`cascade.cinpkg`) indexed by projectile $(Z, A)$ and specific energy $e = E/A$. All surviving charged reaction products are rotated coherently by a common azimuthal angle and tracked through the remainder of the copper block and downstream air gap. Multi-generation nuclear cascades within the copper collimator are tracked up to $g_{\max,\mathrm{Cu}} = 3$ inelastic generations, beyond which surviving charged fragments continue electromagnetic and elastic transport without further inelastic branching.

The upstream angular transport of copper-born charged fragments is tailored to each projectile pipeline:
- In **carbon-ion minibeams**, copper fragments are transported using independently extracted, species-resolved copper Urban MCS tables and species-resolved copper nuclear elastic scattering (`minibeam_copper_fragment_native_transport`).
- In **proton minibeams**, upstream copper fragments (such as secondary deuterons, tritons, and $\alpha$ particles) utilize a Highland-based Gaussian multiple-scattering update paired with unrestricted electronic energy loss.

Every primary ion and copper/air-born charged fragment that reaches the downstream patient entrance plane transfers its complete six-dimensional phase-space state $(\mathbf{r}, \mathbf{u}, E)$, nuclear identity $(Z, A)$, generation index, and deterministic RNG state into the patient transport queue.

---

## II.D. GPU Parallelization, Memory Management, and Scoring

### II.D.1. SYCL/CUDA Execution Pipeline, Queue Compaction, and Counter-Based Random Numbers

MAIGO is implemented in C++20 and Khronos SYCL, supporting NVIDIA GPUs through the oneAPI Unified Runtime CUDA backend and Intel Xe GPUs through Level Zero. Device kernels are compiled ahead of time into native code for the target architecture, with runtime just-in-time compilation disabled. Four compile-time-specialized primary pipelines serve proton broadbeam, carbon-ion broadbeam, proton minibeam, and carbon-ion minibeam calculations; a generic pipeline supports custom ion configurations. The pipelines share secondary transport kernels supporting multispecies charged nuclear fragments.

To mitigate SIMT warp/subgroup divergence caused by heterogeneous track lifetimes, nuclear branching, and collimator absorption, minibeam execution decouples particle transport into sequential device stages connected by preallocated global-memory ring/slot queues:
1. **Upstream beamline stage**: Tracks incident primaries from the nozzle through the multislit copper collimator and air gap, depositing surviving primaries at the patient entrance checkpoint and pushing copper-born charged fragments into dedicated upstream child queues.
2. **Upstream fragment cascade stage**: Transports copper-born charged products through the remaining collimator thickness and air gap across up to three inelastic generations.
3. **Patient primary transport stage**: Resumes surviving primary tracks from the patient entrance into the voxelized water or CT grid, optionally employing counting-sort slot compaction (`bucket_order`) to group active tracks by spatial/material locality.
4. **Patient secondary fragment stage**: Transports all patient-born and upstream-transmitted charged secondaries in fixed-step segments (`kSecondarySegmentSteps`), compacting surviving long-lived tracks when active queue occupancy drops below a tail threshold.

Device queues are preallocated before kernel launch, with hardware atomic counters recording queue occupancy and completion across stages.

Device-side random numbers use the counter-based Philox-4x32-10 generator. Draws are keyed by batch seed, spot index, primary history index, descendant track identity, and a monotonically increasing counter, making each track’s random stream independent of thread scheduling and queue reordering. Consecutive batches reuse the GPU context and resident material and physics tables, amortizing initialization and physics-table upload costs across the campaign.

### II.D.2. Largest-Remainder Integer Spot Allocation and Absolute Dose Normalization

Let $W_j > 0$ denote the physical particle weight (number of clinical particles per fraction/field) assigned to pencil-beam spot $j \in \{1, \dots, M\}$, and let $W = \sum_{j=1}^M W_j$ denote the total physical particle weight of the treatment plan. For a Monte Carlo simulation campaign comprising $N$ total incident histories, integer history counts $n_j$ are allocated across spots via the largest-remainder method:

$$
n_j = \left\lfloor \frac{N W_j}{W} \right\rfloor + \mathbb{I}\!\left(j \in \mathcal{R}_{N - \sum_k \lfloor N W_k / W \rfloor}\right), \qquad \sum_{j=1}^M n_j = N,
\tag{11a}
$$

Here, $\mathcal{R}_K$ is the index set of the $K$ largest fractional remainders. The allocation sums to $N$, with the remaining histories assigned to spots closest to the next integer count. This campaign-wide allocation is partitioned across GPUs and statistical batches. Paired TOPAS comparisons use the reference calculation’s integer spot counts.

During transport, energy deposition $\Delta E$ (in $\mathrm{MeV}$) is accumulated into global device memory using 32-bit floating-point (`FP32`) atomic additions. Given voxel mass density $\rho_i$ ($\mathrm{g/cm^3}$), voxel volume $V_i$ ($\mathrm{mm^3}$), and total accumulated energy $E_i$ ($\mathrm{MeV}$) in voxel $i$, the mass $m_i$ (in $\mathrm{kg}$) and absolute absorbed dose to medium $D_i$ (in $\mathrm{Gy}$) are computed as

$$
m_i = \rho_i V_i \times 10^{-6}\text{ kg}, \qquad
D_i = \frac{E_i \times 1.602176634 \times 10^{-13}\text{ J/MeV}}{m_i}\,\frac{W}{N}.
\tag{11b}
$$

When a campaign is partitioned across $B$ batches (or multiple GPUs) with $N_b$ histories in batch $b$ ($\sum_{b=1}^B N_b = N$), each batch output is scaled by the campaign-wide normalization factor $W / N$ so that the final plan dose is obtained by simple high-precision (FP64) host summation $D_i = \sum_{b=1}^B d_{i,b}$.

### II.D.3. Track-Length Estimator for Dose-Averaged Electronic LET

Dose-averaged electronic linear energy transfer ($L_d$) is scored concurrently with absorbed dose using an unrestricted stopping-power track-length estimator. For each finite transport step $k$ of a charged ion traversing voxel $i$ with accepted true path length $\ell_k > 0$, let $S_{e,k}$ denote the unrestricted linear electronic stopping power ($\mathrm{MeV/mm}$, numerically equivalent to $\mathrm{keV/\mu m}$) evaluated at the pre-step kinetic energy and actual voxel density $\rho_i$. The engine accumulates two extensive voxel moments, $A_i$ and $B_i$, whose ratio yields the voxelwise dose-averaged electronic LET $L_{d,i}$:

$$
A_i = \sum_{k \in i} S_{e,k}^{\,2}\,\ell_k, \qquad
B_i = \sum_{k \in i} S_{e,k}\,\ell_k, \qquad
L_{d,i} = \frac{A_i}{B_i}.
\tag{12}
$$

The weight in Eq. (12) is the expected electronic energy loss $S_{e,k}\,\ell_k$, linking LET scoring to the stopping response used for ion transport. Both $S_{e,k}$ and $L_{d,i}$ are linear stopping quantities evaluated at the actual voxel density and expressed in $\mathrm{keV/\mu m}$. Registered ions query the `continuous26` unrestricted electronic stopping tables, and additional heavy target recoils query the electronic-only `LERSP001` tables. Both LET moments are constructed from electronic stopping power.

All transported charged particles within the water or patient CT scoring grid—including primary ions, secondary nuclear fragments, and transported elastic recoils above the $0.1\text{ MeV}$ kinetic energy cutoff—contribute to $A_i$ and $B_i$. Upstream copper/air trajectory segments, discrete nuclear-vertex local energy deposits, zero-path terminal cutoff deposits, and untracked neutral reaction products are excluded from the LET moments. Both $A_i$ and $B_i$ are accumulated on the GPU via `FP32` atomic additions, written to disk independently for each field and batch, and summed in double precision across all fields, batches, and devices prior to evaluating the quotient $A_i / B_i$. Voxels with $B_i = 0$ (no traversed charged-particle track) are flagged explicitly and excluded from ratio averaging.

---

## II.E. Clinical Treatment Plans and Reference Monte Carlo Benchmarks

### II.E.1. Patient Cohort, Voxel Grids, and Simulation Statistics

Clinical full-plan evaluation uses two head CT datasets—**RT07575** (Head 1) and **RT06423** (Head 2)—and one lung CT dataset, **20022516** (Lung 1). The head cases assess plan-dose agreement in complex anatomy, and the lung case assesses transport in low-density heterogeneous tissue. Table 2 lists the history counts for paired proton and carbon-ion calculations in broadbeam and copper-collimated minibeam modes. Each MAIGO–TOPAS pair shares the CT grid, Schneider material mapping, beam optics, spot positions and energies, final weights, placement, collimator geometry, and absolute dose normalization.

**Table 2.** Total simulated incident particle histories ($N$) for full-plan patient dose validations across clinical cases and beam delivery modes.

| Clinical Case | Anatomical Site | Proton Broadbeam | Carbon-12 Broadbeam | Proton Minibeam | Carbon-12 Minibeam |
|---|---|---:|---:|---:|---:|
| **RT07575** | Head & Neck 1 (Head 1) | $250,626,500$ | $129,638,170$ | $1,000,000,000$ | $1,000,000,000$ |
| **RT06423** | Head & Neck 2 (Head 2) | $124,266,340$ | $151,087,660$ | $1,000,000,000$ | $1,000,000,000$ |
| **20022516** | Thoracic (Lung 1) | $361,050,622$ | $177,173,040$ | $3,000,000,000$ | —$^{\mathrm{a}}$ |

$^{\mathrm{a}}$— denotes a case–modality pair not reported in this table. The default CT carbon minibeam configuration corresponds to a separate lung benchmark phantom.

Full-plan dose grids use $0.5 \times 0.5 \times 2.0\text{ mm}^3$ voxels, with dimensions, physical origin, and axis order defined by each plan’s reference header. Three-dimensional dose and LET are compared voxelwise in the shared patient coordinates, native grid, and absolute plan-dose scale.

### II.E.2. Independent TOPAS Reference Simulations and Physics Alignment

Reference dose is calculated with `DoseToMedium` scoring in OpenTOPAS 4.2.3 / Geant4 11.3.2 [7,8]. The physics list includes standard electromagnetic processes (`g4em-standard_opt4`), hadron and ion inelastic interactions (`g4h-phy_QGSP_BIC_HP`, `g4ion-inclxx`), hadron elastic scattering (`g4h-elastic_HP`), stopping (`g4stopping`), and decay (`g4decay`, `g4radioactivedecay`). Minibeam references apply Urban scattering to charged particles through `AllChargedUrbanPhysics`. Carbon-ion references register nuclear elastic scattering for `GenericIon` through `CarbonIonElasticPhysics`, which wraps `G4IonElasticPhysics`.

Electromagnetic table limits, secondary production cuts, and step settings are recorded with each reference input. The lung proton minibeam reference uses a $0.05\text{ mm}$ production cut and a $10\text{ GeV}$ EM table upper limit; the three carbon-ion broadbeam references use a $600\text{ MeV}$ upper limit. Reference doses are generated by independent full-plan TOPAS transport driven by the final plan weights. TOPAS explicitly transports secondary electrons and photons above the production cut. MAIGO’s local-ionization deposition, secondary mean energy loss, nuclear final-state sampling, and neutral-product treatment are defined in Sections II.B–C.

### II.E.3. Two-Moment TOPAS Reference Scoring for Dose-Averaged LET

The TOPAS reference uses the two-moment LET estimator of Eq. (12), accumulating $A_i = \sum_k w_k S_{e,k}^2 \ell_k$ and $B_i = \sum_k w_k S_{e,k} \ell_k$, where $w_k$ is the track statistical weight and the scoring threshold is $0.1\text{ MeV}$ in total kinetic energy. The scorer queries pre-extracted read-only unrestricted electronic stopping tables generated with the reference Geant4 11.3.2 physics list, particle species, Schneider tissue compositions, densities, and mean excitation energies. This establishes a common LET definition and scoring endpoint across the two engines.

To resolve stopping-power gradients near the Bragg peak, the carbon-ion broadbeam TOPAS LET reference uses `dRoverRange = 0.005` and `finalRange = 0.1 nm` for light ions and `GenericIon`, with the reference proton step controls retained. The RT07575 single-replica paired LET comparison uses $25,927,634$ incident histories and matching integer spot allocations in MAIGO and TOPAS. LET uncertainty from independent replicas is estimated as described in Section II.F.4.

---

## II.F. Quantitative Evaluation Metrics, Statistical Uncertainty, and Computational Efficiency

### II.F.1. Three-Dimensional Local Gamma Index and Dosimetric Endpoints

Spatial and dosimetric agreement between the GPU-calculated dose distribution $D_{\mathrm{GPU}}(\mathbf{r}')$ and the reference TOPAS dose distribution $D_{\mathrm{ref}}(\mathbf{r})$ is quantified using the three-dimensional local gamma index [10]:

$$
\gamma(\mathbf{r}) = \min_{\mathbf{r}'} \sqrt{
\frac{\|\mathbf{r}' - \mathbf{r}\|^2}{\Delta r^2} +
\frac{\left[D_{\mathrm{GPU}}(\mathbf{r}') - D_{\mathrm{ref}}(\mathbf{r})\right]^2}{\left[\delta D_{\mathrm{ref}}(\mathbf{r})\right]^2}
}, \qquad
\delta D_{\mathrm{ref}}(\mathbf{r}) = \left(\frac{\Delta D_{\%}}{100}\right) D_{\mathrm{ref}}(\mathbf{r}),
\tag{13}
$$

Here, $\Delta r$ is the distance-to-agreement (DTA) criterion and $\Delta D_{\%}$ is the percentage dose tolerance relative to the local reference dose. Trilinear interpolation evaluates the GPU dose field within a sphere of radius $\Delta r$ around each reference point for subvoxel-distance gamma searches. Evaluation points are voxels inside `BODY` that exceed the dose threshold; interpolation uses the full GPU dose field, including neighboring voxels outside `BODY`. Table 3 lists the evaluation masks, local criteria, and search increments.

**Table 3.** Three-dimensional local gamma evaluation masks, acceptance criteria, and sub-voxel interpolation search steps.

| Delivery Mode | Evaluation Mask within `BODY` | Reported Local Gamma Criteria ($\Delta D_{\%}/\Delta r$) | Interpolation Search Step |
|---|---|---|---:|
| **Broadbeam** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\, D_{\mathrm{ref},\max}$ | $3\%/3\text{ mm}$, $1\%/1\text{ mm}$, $3\%/0\text{ mm}$ | $0.10\text{ mm}$ |
| **Minibeam** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.4\text{ Gy}$ ($10\%$ of $4\text{ Gy}$ prescription) | $3\%/0.2\text{ mm}$, $3\%/0.1\text{ mm}$, $3\%/0\text{ mm}$ | $0.02\text{ mm}$ |

An evaluation point passes when $\gamma \le 1$. The $3\%/0\text{ mm}$ endpoint is defined separately as the same-position local dose-difference test $|D_{\mathrm{GPU}}-D_{\mathrm{ref}}|/D_{\mathrm{ref}} \le 0.03$. The prespecified broadbeam acceptance criterion is a local $1\%/1\text{ mm}$ pass rate above $99\%$. Minibeams use $3\%/0.2\text{ mm}$ and $3\%/0.1\text{ mm}$ criteria with a $0.02\text{ mm}$ search increment to match the submillimeter slit scale. Transverse profiles at multiple depths assess peak width, valley dose, and PVDR; depth-dose profiles assess distal falloff; and DVH indices assess target and OAR dose. Same-position dose-difference distributions and batch uncertainties characterize voxelwise agreement.

### II.F.2. Voxelwise LET Gamma Evaluation and Spatial Profile Analysis

Three-dimensional local gamma evaluation for dose-averaged electronic LET ($L_d$) is computed analogously to Eq. (13) by substituting $L_{d,\mathrm{GPU}}(\mathbf{r}')$ and $L_{d,\mathrm{ref}}(\mathbf{r})$ for dose, with the local tolerance set to $\delta L_{d,\mathrm{ref}}(\mathbf{r}) = (\Delta L_{\%}/100)\,L_{d,\mathrm{ref}}(\mathbf{r})$. The evaluation region is defined as the intersection of the `BODY` contour with the broadbeam dose mask $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\,D_{\mathrm{ref},\max}$. Broadbeam LET agreement is reported at local $3\%/3\text{ mm}$, $1\%/1\text{ mm}$, and $3\%/0\text{ mm}$ criteria using a $0.10\text{ mm}$ trilinear interpolation search step on the GPU LET field. Any voxel within the reference dose mask where the GPU LET denominator vanishes ($B_i = 0$) is retained as a gamma failure.

LET spatial analysis uses transverse profiles at fixed geometric depths and depth profiles along the beam axis. In the RT07575 carbon-ion broadbeam case, depth begins at the central-ray entry into `BODY` and is sampled at the native $0.5\text{ mm}$ interval. Central-ray profiles characterize local LET structure, and profiles pooled over a $2.5 \times 10\text{ mm}^2$ transverse ROI characterize regional depth trends. ROI LET is computed as $\sum_{i\in\mathrm{ROI}}A_i/\sum_{i\in\mathrm{ROI}}B_i$ within each depth slice.

### II.F.3. Statistical Uncertainty Estimation for Absorbed Dose

Voxelwise Monte Carlo statistical uncertainty on the GPU is evaluated from $B$ independent full-plan batches, each preserving the campaign-wide spot weight proportions. For $B$ equal-history batches whose individual voxel dose contributions $d_{i,b}$ are scaled such that $D_i = \sum_{b=1}^B d_{i,b}$, the unbiased sample variance of the total voxel dose $D_i$ is

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{B}{B - 1} \left[ \sum_{b=1}^B d_{i,b}^{\,2} - \frac{D_i^{\,2}}{B} \right].
\tag{14}
$$

When batches have slightly unequal history counts $N_b$ (e.g., due to throughput-proportional multi-GPU partitioning with $\sum_{b=1}^B N_b = N$), let $a_b = N_b / N$ denote the history fraction, $Y_{i,b} = d_{i,b} / a_b$ the full-plan-equivalent batch dose estimate, and $q = \sum_{b=1}^B a_b^2$. The weighted batch variance estimator is given by

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{q}{1 - q} \left[ \sum_{b=1}^B a_b Y_{i,b}^{\,2} - D_i^{\,2} \right],
\tag{15}
$$

which reduces identically to Eq. (14) when $a_b = 1/B$.

TOPAS uncertainty is estimated from per-history scoring variance or matched pilot simulations. Per-history variances are converted to the plan-dose scale using the normalization factor. For the three carbon-ion broadbeam references, pilot simulations with $N_{\mathrm{pilot}}=300,000$ histories preserve the reference geometry, physics list, energy spectrum, and relative spot weights. Pilot standard errors are extrapolated to the full history count as $\widehat{\sigma}_{D_{\mathrm{ref},i}}=\widehat{\sigma}_{\mathrm{pilot},i}\sqrt{N_{\mathrm{pilot}}/N_{\mathrm{full}}}$ and cross-checked against inter-replica variance.

Across any fixed evaluation mask $\mathcal{R}$ containing $|\mathcal{R}|$ voxels, the overall statistical precision is summarized by the root-mean-square (RMS) relative standard error:

$$
u_{\mathrm{RMS}} = 100 \sqrt{ \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} \frac{\widehat{\sigma}_{D_i}^{\,2}}{D_{\mathrm{ref},i}^{\,2}} } \;\; (\%),
\tag{16}
$$

supplemented by voxelwise uncertainty percentiles and the fraction of voxels satisfying $u_i < 1\%$ or $u_i < 2\%$. Because GPU and TOPAS simulations use independent random streams, the statistical variance of the voxelwise dose difference $\Delta D_i = D_{\mathrm{GPU},i} - D_{\mathrm{ref},i}$ is given by $\widehat{\sigma}_{\Delta D_i}^{\,2} = \widehat{\sigma}_{D_{\mathrm{GPU},i}}^{\,2} + \widehat{\sigma}_{D_{\mathrm{ref},i}}^{\,2}$.

### II.F.4. Ratio-Variance Estimation for Dose-Averaged LET

Because $L_{d,i} = A_i / B_i$ is a ratio of two correlated random variables accumulated along the same particle tracks, its statistical variance accounts for the covariance $\mathrm{Cov}(A_i, B_i)$. To evaluate voxelwise LET statistical uncertainty at the full-plan history count $N$, an auxiliary GPU calculation partitions the $N$ histories into $K$ independent, equal-statistics full-plan replicate groups ($K = 20$ for the carbon-ion broadbeam benchmark). Let $A_{i,b}$ and $B_{i,b}$ denote the accumulated numerator and denominator moments in replicate group $b \in \{1, \dots, K\}$, with pooled estimator $L_{d,i} = \sum_{b=1}^K A_{i,b} \big/ \sum_{b=1}^K B_{i,b}$. Applying a first-order Taylor expansion (delta method) for ratio estimators across independent groups yields the covariance-preserving variance estimator:

$$
\widehat{\sigma}_{L_{d,i}}^{\,2} = \frac{K}{K - 1} \, \frac{\sum_{b=1}^K \left( A_{i,b} - L_{d,i}\,B_{i,b} \right)^2}{\left( \sum_{b=1}^K B_{i,b} \right)^2}.
\tag{17}
$$

For TOPAS LET references partitioned into independent replicas (each saving paired $A_{i,b}$ and $B_{i,b}$ volumes), the reference LET standard error is computed analogously via Eq. (17). Voxelwise relative LET standard errors, regional RMS uncertainties within the $D_{\mathrm{ref}} > 0.10\,D_{\mathrm{ref},\max}$ mask, and quantiles are reported alongside the 3D LET gamma pass rates.

### II.F.5. Computational Throughput Benchmarking and Runtime Integrity Audits

Computational performance is evaluated separately across GPU architectures—including NVIDIA GeForce RTX 2080 Ti (`sm_75`), NVIDIA RTX A6000 (`sm_86`), NVIDIA L4 (`sm_89`), and Intel Arc B580 (`bmg-g21`)—using AOT-compiled native binaries with runtime JIT compilation disabled. Two end-to-end wall-clock timing metrics are reported:
1. **Cold-start latency**: Total elapsed wall-clock time from OS process invocation to completion of the first saved output batch, encompassing configuration parsing, GPU context initialization, host-to-device physics package uploads, kernel execution, runtime quality audits, and disk output.
2. **Steady-state throughput ($\mathrm{histories/s}$)**: Total simulated incident histories across subsequent resident batches divided by their cumulative end-to-end wall-clock time (including device-to-host readback, quality audits, and file I/O) within the same persistent process.

Stage-level kernel execution timers decompose GPU compute time across primary transport, upstream copper/air beamline transport, and secondary fragment transport, and quantify the marginal overhead of enabling concurrent LET scoring ($D_m + L_d$ vs. $D_m$ only) under identical history counts and batch sizes. In multi-GPU campaigns, total histories $N$ are distributed across local and remote GPUs in proportion to their measured steady-state throughput so that concurrent device completion times are equalized, while preserving the single campaign-wide largest-remainder spot allocation (Eq. 11a). Reference multi-threaded CPU TOPAS throughput ($\mathrm{histories/s}$) is measured on dual-socket server nodes (56-thread cluster nodes and 96-thread local workstations) from total simulated histories divided by cumulative node wall-clock time under matched geometry and physics configurations.

Each batch checks track completion, queue capacity, physics-table coverage, and energy balance, saving the checks with dose and LET outputs. Required conditions are completion of all launched tracks, zero queue overflow, in-domain physics queries, and finite non-negative dose and LET moments. The energy ledger separates continuous deposition, nuclear-vertex deposition, terminal cutoff deposition, boundary escape, upstream loss, neutral products, and reaction residuals to check energy accounting by species and transport stage.

---

## References

1. Jia X, Schümann J, Paganetti H, Jiang SB. GPU-based fast Monte Carlo dose calculation for proton therapy. *Phys Med Biol.* 2012;57(23):7783–7797. [doi:10.1088/0031-9155/57/23/7783](https://doi.org/10.1088/0031-9155/57/23/7783).
2. Qin N, Pinto M, Tian Z, et al. Initial development of goCMC: a GPU-oriented fast cross-platform Monte Carlo engine for carbon ion therapy. *Phys Med Biol.* 2017;62(9):3682–3699. [doi:10.1088/1361-6560/aa5d43](https://doi.org/10.1088/1361-6560/aa5d43).
3. De Simoni M, Battistoni G, De Gregorio A, et al. A data-driven fragmentation model for carbon therapy GPU-accelerated Monte-Carlo dose recalculation. *Front Oncol.* 2022;12:780784. [doi:10.3389/fonc.2022.780784](https://doi.org/10.3389/fonc.2022.780784).
4. Lee H, Shin J, Verburg JM, et al. MOQUI: an open-source GPU-based Monte Carlo code for proton dose calculation with efficient data structure. *Phys Med Biol.* 2022;67(17):174001. [doi:10.1088/1361-6560/ac8716](https://doi.org/10.1088/1361-6560/ac8716).
5. Feng H, Holmes JM, Vora SA, et al. Modelling small block aperture in an in-house developed GPU-accelerated Monte Carlo-based dose engine for pencil beam scanning proton therapy. *Phys Med Biol.* 2024;69(3):035003. [doi:10.1088/1361-6560/ad0b64](https://doi.org/10.1088/1361-6560/ad0b64).
6. Schneider W, Bortfeld T, Schlegel W. Correlation between CT numbers and tissue parameters needed for Monte Carlo simulations of clinical dose distributions. *Phys Med Biol.* 2000;45(2):459–478. [doi:10.1088/0031-9155/45/2/314](https://doi.org/10.1088/0031-9155/45/2/314).
7. Perl J, Shin J, Schümann J, Faddegon B, Paganetti H. TOPAS: an innovative proton Monte Carlo platform for research and clinical applications. *Med Phys.* 2012;39(11):6818–6837. [doi:10.1118/1.4758060](https://doi.org/10.1118/1.4758060).
8. Agostinelli S, Allison J, Amako K, et al. Geant4—a simulation toolkit. *Nucl Instrum Methods Phys Res A.* 2003;506(3):250–303. [doi:10.1016/S0168-9002(03)01368-8](https://doi.org/10.1016/S0168-9002(03)01368-8).
9. Eyges L. Multiple scattering with energy loss. *Phys Rev.* 1948;74(10):1534–1535. [doi:10.1103/PhysRev.74.1534](https://doi.org/10.1103/PhysRev.74.1534).
10. Low DA, Harms WB, Mutic S, Purdy JA. A technique for the quantitative evaluation of dose distributions. *Med Phys.* 1998;25(5):656–661. [doi:10.1118/1.598248](https://doi.org/10.1118/1.598248).
