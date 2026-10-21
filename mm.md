# II. Materials and Methods

**GPU Monte Carlo: From Broadbeam to Minibeam** ([中文版](mm_zh.md)).

We developed **MAIGO**, a cross-platform GPU Monte Carlo engine for proton and carbon-ion pencil-beam scanning (PBS), extending broadbeam patient transport to copper-collimated minibeams. A shared transport core combines condensed-history electromagnetic transport, discrete nuclear interactions, and multigeneration charged-fragment tracking [1–4]. The minibeam extension explicitly transports particles through the multislit collimator and air gap [5], then transfers their joint position–direction–energy–species state to the patient. This coupling carries collimator-induced beam modulation and nuclear products into the calculation of submillimeter peak–valley dose structure.

Material-dependent cross sections and response functions are extracted offline from Geant4 11.3.2 and OpenTOPAS 4.2.3 [7,8] and queried during transport. The study considers primary protons ($Z=1, A=1$) and carbon-12 ions ($^{12}\mathrm{C}$, $Z=6, A=12$) in water and patient CT, with absorbed dose to medium ($D_m$) and dose-averaged electronic LET ($L_d$) as scoring quantities.

---

## II.A. Ion Transport Physics

The shared models below describe ion slowing, scattering, and nuclear reactions in water and heterogeneous patient tissues. Their material-specific extension to copper and air is given in Section II.B.

### II.A.1. Continuous Energy Loss and Straggling

Continuous slowing of charged ions is described by the unrestricted electronic stopping power $S_e(E;\rho)$ at the particle's kinetic energy $E$, local material composition, and mass density $\rho$. The stopping response includes species-dependent effective-charge and density-effect corrections; the reference water uses a mean excitation energy of $75\text{ eV}$. For a true step length $\ell$, the mean energy loss is related to the continuous-slowing-down approximation (CSDA) range by

$$
\begin{aligned}
R(E) &= \int_0^E \frac{\mathrm{d}E'}{S_e(E')}, \\
\overline{\Delta E}(E,\ell)
&= E-R^{-1}\!\bigl(\max\{R(E)-\ell,0\}\bigr).
\end{aligned}
\tag{1}
$$

For short steps, protons and heavier ions share a midpoint-energy predictor–corrector: the stopping response is evaluated at $E-\tfrac{1}{2}S_e(E)\ell$ and multiplied by the true step length $\ell$ to account for its variation along the step. When the trial loss exceeds the native fractional linear-loss threshold, or the step exhausts the residual range, the mean loss is evaluated by CSDA range inversion using Eq. (1).

Primary energy-loss straggling is sampled with a unified engine adapted from the Geant4 `G4UniversalFluctuation` and `G4IonFluctuations` models [8]. For an ion of rest mass $M$, speed $\beta c$, and Lorentz factor $\gamma$, the maximum transferable energy to a free electron is

$$
T_{\max} = \frac{2 m_e c^2 \beta^2 \gamma^2}{1 + 2\gamma(m_e/M) + (m_e/M)^2},
\tag{2}
$$

where $m_e$ is the electron rest mass. Unrestricted transport includes electronic energy transfers up to $T_{\max}$, corresponding to an energy-transfer cutoff $T_c=T_{\max}$ in the Geant4 dispersion expression. The baseline energy-loss variance over a true step length $\ell$ is therefore

$$
\sigma_{E,0}^2=
2\pi r_e^2m_ec^2N_{\mathrm{el}}
\frac{Z_h^2}{\beta^2}\,T_{\max}\ell
\left(1-\frac{\beta^2}{2}\right),
\tag{3}
$$

where $r_e$ is the classical electron radius, $N_{\mathrm{el}}$ is the local electron number density, and $Z_h$ is the projectile charge number in units of the elementary charge. Equation (3) gives the unrestricted baseline dispersion; the low-energy Ion model additionally uses the native effective-charge and charge-fluctuation corrections.

Protons and higher-velocity ions use the Universal fluctuation model. Thin-absorber losses combine stochastic atomic-excitation and ionization contributions; thick-absorber losses use truncated Gaussian or moment-matched Gamma sampling according to the mean-to-standard-deviation ratio. Lower-energy heavy ions use the Ion fluctuation model, including the finite-step deceleration correction and its Gaussian, Gamma, or uniform sampling regimes. Each primary step receives one full-loss draw based on the mean loss and unrestricted dispersion. Charged secondary fragments use the finite-step mean loss while retaining stochastic multiple scattering and nuclear interactions.

The sampled loss is bounded by the available kinetic energy; range-terminating steps deposit the residual kinetic energy. The full ionization loss is deposited locally in the current water or patient CT voxel. This treatment retains primary range straggling and Bragg-peak broadening within condensed-history ion transport.

### II.A.2. Multiple Coulomb Scattering

In broadbeam configurations—and for charged secondary fragments inside the patient volume in CT minibeams—multiple Coulomb scattering is modeled using the correlated Gaussian Fermi–Eyges theory [9]. Within a homogeneous sub-step of length $L$ characterized by linear scattering power $\mathcal{T}$, the projected angular deflection $\theta$ and lateral spatial displacement $y$ along each of two orthogonal transverse axes follow a bivariate normal distribution with covariance matrix

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
\tag{4}
$$

Correlated pairs $(\theta_x, x)$ and $(\theta_y, y)$ are sampled via Cholesky decomposition in the instantaneous local frame of the particle and rotated into global Cartesian coordinates, reproducing the angle–displacement covariance in Eq. (4). When a transport step exceeds the configured scattering sub-segment limit (e.g., the $0.1\text{ mm}$ segment limit for CT minibeam charged secondaries), the step is subdivided into uniform sub-segments and $\mathcal{T}$ is evaluated at the midpoint kinetic energy of each segment.

For broadbeam transport and patient charged secondaries in CT minibeams, the effective linear scattering power $\mathcal{T}$ combines a Rossi–Greisen Gaussian core with a calibrated tail second-moment contribution. This contribution is incorporated into the Gaussian covariance without sampling separate tail events:

$$
\mathcal{T}(E) = \left(\frac{E_s Z}{\beta p c}\right)^2 \frac{1}{X_0(\rho)} + \lambda_{\mathrm{tail}}(\rho)\,\left(\frac{E_{\mathrm{tail}} Z}{\beta p c}\right)^2,
\tag{5}
$$

where $p$ is the relativistic momentum, $X_0(\rho)$ is the local radiation length derived from the Schneider tissue composition and voxel density, and $\lambda_{\mathrm{tail}}(\rho) = \lambda_{\mathrm{tail}, \mathrm{water}} \, [X_{0,\mathrm{water}} / X_0(\rho)]$ scales the nominal tail rate by inverse radiation length. For light ions ($^1\mathrm{H}$, $^2\mathrm{H}$, $^3\mathrm{H}$, and $^4\mathrm{He}$), the effective core energy parameter $E_s(E/A)$ and tail parameters are obtained by piecewise-linear interpolation of homogeneous water fits calibrated against Geant4 at $50$, $150$, and $300\text{ MeV/u}$ (clamped to endpoint values outside this interval). For $^{12}\mathrm{C}$ and heavier fragments assigned to the same calibration group, the engine adopts calibrated constants $E_s = 9.9\text{ MeV}$, $\lambda_{\mathrm{tail},\mathrm{water}} = 0.0025\text{ mm}^{-1}$, and $E_{\mathrm{tail}} = 2.4\text{ MeV}$. Strong-interaction nuclear elastic scattering is modeled independently as a discrete stochastic process (Section II.A.3).

For minibeam primaries, multiple scattering is modeled using a GPU implementation of the Urban model [8,11], with material- and species-dependent transport mean free paths, radiation lengths, and angular parameters tabulated from Geant4. The model samples angular deflection and correlated lateral displacement while accounting for the difference between the true particle path and its geometric advance.

Near CT interfaces and copper–air boundaries, the geometric step is limited by the distance to the boundary, while energy loss and nuclear interaction probabilities are evaluated along the corresponding true path. This coupling maintains a consistent description of scattering and geometric transport across tissue interfaces and the multislit collimator. Copper-born charged fragments also use species-resolved Urban transport within copper, as described in Section II.B.2.

### II.A.3. Nuclear Interaction

Nuclear elastic and inelastic interactions are sampled using a shared optical-depth scheme for primary ions and charged secondary fragments. For channel $c \in \{\mathrm{el},\mathrm{in}\}$, the macroscopic interaction rate is obtained from the local target composition and density:

$$
\begin{aligned}
\Sigma_c(E;\rho) &= \sum_t n_t(\rho)\sigma_{c,t}(E), \\
\tau &= -\ln U, \\
\Delta\tau &= [\Sigma_{\mathrm{el}}(E;\rho)+\Sigma_{\mathrm{in}}(E;\rho)]\,\ell.
\end{aligned}
\tag{6}
$$

Here, $n_t$ is the atomic number density of target element $t$, $\sigma_{c,t}$ is its microscopic cross section, $U$ is uniform on $(0,1)$, and $\ell$ is the true step length. The residual optical depth is carried across material boundaries until a nuclear vertex is reached. The channel and target are then sampled in proportion to their interaction rates. Cached primary inelastic proposals are corrected by null-collision rejection using the post-slowing interaction rate.

Elastic final states are generated from projectile- and target-specific isotope and angular distributions extracted from Geant4 [8]. The sampled center-of-mass scattering angle and a uniform azimuth define the outgoing projectile and recoil through relativistic two-body kinematics. Recoils in the registered charged-fragment set undergo electromagnetic and nuclear transport. Additional target-recoil isotopes undergo continuous slowing down using unrestricted electronic plus nuclear stopping, $S_{\mathrm{tot}}=S_e+S_n$, and the corresponding CSDA range; their electronic LET is scored using $S_e$.

Inelastic reactions sample complete correlated events from Geant4 final-state libraries indexed by projectile species, target element, and incident energy per nucleon [8]. The primary proton library uses Binary Cascade (`QGSP_BIC_HP`), the primary carbon library uses INCLXX (`FTFP_INCLXX`), and secondary fragment libraries use `G4IonBinaryCascadePhysics`. At intermediate energies, a neighboring energy node is selected with linear interpolation probabilities, followed by uniform sampling of a complete event. A common random azimuthal rotation is applied to all products, preserving event multiplicity and energy–angle correlations. Energy offsets from node selection and reaction Q-value residuals are accounted for separately.

Supported charged products above $0.1\text{ MeV}$ enter secondary transport; sub-cutoff charged products and recorded local vertex energy are deposited at the reaction site. Secondary inelastic branching in water and patient CT is followed to a maximum generation depth of two, after which charged fragments continue slowing down, multiple scattering, and permitted nuclear elastic scattering. Neutral products, including neutrons and de-excitation photons, and residual species outside the transported set are accounted for as untracked energy and excluded from patient dose.

### II.A.4. Step Control and Boundary Transport

At the beginning of each transport step, the local material composition, mass density, and electromagnetic response are evaluated to determine the maximum permissible step length across competing physical and geometric processes. Let $f$ denote the species-specific range fraction, $r_f$ the final-range parameter, and $\eta$ the maximum fractional mean energy loss per step. When the residual CSDA range satisfies $R(E) > r_f$, the electromagnetic step proposal $\ell_{\mathrm{EM}}$ is governed by the dual range-guided and energy-loss constraint:

$$
\ell_{\mathrm{EM}} = \min\!\left[
f R(E) + r_f(1 - f)\!\left(2 - \frac{r_f}{R(E)}\right),\;
R(E) - R\!\left((1 - \eta)E\right)
\right].
\tag{7}
$$

Once the particle slows to $R(E) \le r_f$, the proposal collapses to $\ell_{\mathrm{EM}} = R(E)$, and the track terminates at the end of that step. In heterogeneous CT broadbeam transport, the fractional energy-loss limit is set to $\eta = 0.05$, and minibeam patient transport uses $\eta = 0.01$.

The electromagnetic proposal $\ell_{\mathrm{EM}}$ competes with the region-specific hard step cap $\ell_{\max}$ (Table 1), the distance to the next voxel or material boundary $\ell_{\mathrm{geom}}$, the multiple-scattering step limit, and the sampled nuclear interaction distance $\ell_{\mathrm{nuc}}$.

**Table 1.** Enforced maximum ion step lengths ($\ell_{\max}$) and principal multiple Coulomb scattering (MCS) models across transport regions.

| Transport Region | Maximum Ion Step $\ell_{\max}$ | Principal Multiple Scattering Model |
|---|---:|---|
| Water / CT broadbeam | $1.0\text{ mm}$ | Correlated Gaussian Fermi–Eyges |
| CT minibeam (patient volume) | $0.5\text{ mm}$ | Data-driven Urban (primaries); segmented Fermi–Eyges (charged secondaries, $0.1\text{ mm}$ segment cap) |
| Water minibeam (phantom volume) | $0.5\text{ mm}$ | Data-driven Urban (primaries and charged secondaries) |
| Minibeam copper collimator | $0.05\text{ mm}$ | Copper-specific Urban MCS + discrete nuclear elastic scattering |
| Minibeam slit air (CT carbon minibeam) | $0.2\text{ mm}$ | Air-specific Urban MCS |

The interaction of geometry and scattering depends on the active MCS formalism:

- Under **Fermi–Eyges transport**, the step length is evaluated directly along the geometric ray, and lateral displacement is applied at the end of each scattering sub-segment.
- Under **Urban transport**, the algorithm explicitly distinguishes the curved microscopic true path length $\ell$ from the straight geometric chord length $g$. Geometric distances to voxel faces or collimator sidewalls are first computed in Euclidean space and converted to an admissible true path length $\ell_{\mathrm{geom}}$. Energy loss and nuclear optical depth are then integrated along the true path $\ell = \min(\ell_{\mathrm{EM}}, \ell_{\max}, \ell_{\mathrm{geom}})$. If a discrete nuclear interaction occurs at $\ell_{\mathrm{nuc}} < \ell$, the true step is truncated to $\ell_{\mathrm{nuc}}$, and the corresponding geometric advance and angular deflection are recomputed for the shortened step prior to final state commitment.

Continuous ionization energy loss is scored in the originating voxel of the step, whereas local energy deposition from a discrete nuclear interaction is scored at the post-step collision vertex. Particles exiting the outer bounding box of the patient/phantom grid are terminated and recorded in the boundary-escape energy ledger. Tracks slowing below the global kinetic energy cutoff $E_{\mathrm{cut}} = 0.1\text{ MeV}$ (total kinetic energy) are terminated in place and deposit their residual kinetic energy locally as a zero-path terminal deposit, which is excluded from finite-track LET accumulation.

---

## II.B. Extension to Minibeam Transport

### II.B.1. Multislit Collimator Geometry

In the minibeam configurations, a divergent or parallel multislit copper collimator and an intervening downstream air gap are interposed between the scanning nozzle exit and the patient surface. The geometric model specifies the collimator outer dimensions, thickness along the beam axis, slit count, slit aperture width, center-to-center pitch, slit length, array lateral phase offset, azimuthal rotation, and the exact axial coordinates of the collimator exit, CT bounding box, skin surface, and treatment isocenter. The collimator remains stationary in the field coordinate system while pencil-beam spots scan across its upstream entrance face.

Two clinical collimator designs are implemented for full-plan evaluations:

1. **Proton minibeam collimator**: A $60\text{ mm}$-thick copper block containing $15$ parallel slits of $0.5\text{ mm}$ aperture width, $3.6\text{ mm}$ center-to-center pitch, and $50\text{ mm}$ slit length.
2. **Carbon-ion minibeam collimator (RT07575 head plan)**: A $100\text{ mm}$-thick copper block containing $17$ slits of $0.7\text{ mm}$ aperture width, $3.0\text{ mm}$ center-to-center pitch, and $100\text{ mm}$ slit length. For the opposing $90^\circ$ and $270^\circ$ lateral fields of the RT07575 plan, the $270^\circ$ slit array incorporates a half-pitch ($1.5\text{ mm}$) lateral phase shift relative to the $90^\circ$ field to interleave opposing peak and valley dose distributions.

### II.B.2. Copper and Air Transport

Charged particles traversing the copper collimator and air regions query material-specific beamline electromagnetic tables, which provide unrestricted electronic stopping powers $S_e(E)$ and native unrestricted straggling variances $V_{\mathrm{full}}(E; T_{\max})$ across all charged isotopes present in the copper charged-fragment set. For steps whose trial linear loss satisfies $S_e(E)\,\ell \le 0.02 E$, mean energy loss is computed linearly as $S_e(E)\,\ell$; for larger fractional losses, exact CSDA range inversion (Eq. 1) is used with a $\sqrt{E}$ stopping continuation below the lowest energy node.

Primary ions in copper and air use the shared Universal/Ion fluctuation engine of Section II.A.1 with native particle mass, charge, material excitation energy, and $T_{\max}$ parameters. Charged secondary fragments use finite-path mean electronic energy loss. Upstream electronic losses follow the local-deposition approximation and enter the beamline energy ledger separately from patient-grid dose. Step caps are $0.05\text{ mm}$ in copper and $0.2\text{ mm}$ in slit air for CT carbon minibeams.

Both coherent nuclear elastic scattering and non-elastic nuclear fragmentation are modeled explicitly within the copper collimator. Primary ions sample their distance to the next copper nuclear interaction using the combined macroscopic cross section $\Sigma_{\mathrm{tot},\mathrm{Cu}} = \Sigma_{\mathrm{el},\mathrm{Cu}} + \Sigma_{\mathrm{in},\mathrm{Cu}}$, followed by post-slowing channel selection. In a copper nuclear elastic event, the center-of-mass scattering angle is drawn from copper-specific momentum-transfer tables and applied via the relativistic two-body kinematics described in Section II.A.3; the scattered beam ion continues transport, while the kinetic energy transferred to the heavy copper target nucleus is booked directly to the upstream beamline energy ledger without tracking the sub-micrometer copper recoil.

Copper inelastic collisions sample complete correlated final states from a dedicated copper cascade event library indexed by projectile $(Z, A)$ and specific energy $e = E/A$. All surviving charged reaction products are rotated coherently by a common azimuthal angle and tracked through the remainder of the copper block and downstream air gap. Multi-generation nuclear cascades within the copper collimator are tracked up to $g_{\max,\mathrm{Cu}} = 3$ inelastic generations, beyond which surviving charged fragments continue electromagnetic and elastic transport without further inelastic branching.

Primary ions in copper and air use the Urban scattering model described in Section II.A.2. In both proton and carbon-ion minibeams, copper-born charged fragments traversing copper use a shared transport model with independently extracted, species-resolved Urban MCS tables, copper nuclear elastic scattering, and unrestricted electronic stopping. Fragment responses are selected by their own nuclear identity $(Z, A)$ and energy.

### II.B.3. Coupling to Patient Transport

MAIGO transports incident histories and their charged nuclear products through the air slits, copper septa, and downstream air gap to generate the patient-entry phase space. The resulting distribution includes directly transmitted primaries, primaries scattered from copper sidewalls, and charged nuclear fragments generated in copper and upstream air, carrying collimator-induced beam modulation and secondary-particle contributions into patient transport. Histories stopped or absorbed in copper remain part of the total incident history count $N$ used for absolute dose normalization.

Every primary ion and copper/air-born charged fragment that reaches the downstream patient entrance plane transfers its joint phase-space state $(\mathbf{r}, \mathbf{u}, E)$, nuclear identity $(Z, A)$, generation index, and deterministic RNG state into the patient transport queue.

Patient primary ions use the Urban model to represent angular deflection and lateral displacement near slit-generated dose gradients. In CT minibeams, charged secondaries use segmented Fermi–Eyges scattering, including upstream-born fragments after entry into the patient. The outer step cap is $0.5\text{ mm}$ for both groups, with a $0.1\text{ mm}$ scattering-segment cap for secondaries. Water and generic minibeam transport use Urban scattering for both primary and secondary charged particles. These assignments use the common scattering operators described in Section II.A.2.

---

## II.C. Dose and LET Scoring

### II.C.1. Absorbed Dose and Plan Normalization

Let $W_j > 0$ denote the physical particle weight (number of clinical particles per fraction/field) assigned to pencil-beam spot $j \in \{1, \dots, M\}$, and let $W = \sum_{j=1}^M W_j$ denote the total physical particle weight of the treatment plan. For a Monte Carlo simulation campaign comprising $N$ total incident histories, integer history counts $n_j$ are allocated across spots via the largest-remainder method:

$$
\begin{aligned}
h_j &= \frac{NW_j}{W},
& K &= N-\sum_{k=1}^M\lfloor h_k\rfloor, \\
n_j &= \lfloor h_j\rfloor+\mathbf{1}_{\{j\in\mathcal{R}_K\}},
& \sum_{j=1}^M n_j &= N.
\end{aligned}
\tag{8a}
$$

Here, $h_j$ is the ideal history count, $K$ is the number of histories remaining after flooring, $\mathcal{R}_K$ is the index set of the $K$ largest fractional remainders $h_j-\lfloor h_j\rfloor$, and $\mathbf{1}$ is the indicator function. This preserves the total history count while keeping each spot allocation within one history of its ideal value. This campaign-wide allocation is partitioned across GPUs and statistical batches. Paired TOPAS comparisons use the reference calculation’s integer spot counts.

During transport, energy deposition $\Delta E$ (in $\mathrm{MeV}$) is accumulated into global device memory using 32-bit floating-point (`FP32`) atomic additions. Given voxel mass density $\rho_i$ ($\mathrm{g/cm^3}$), voxel volume $V_i$ ($\mathrm{mm^3}$), and total accumulated energy $E_i$ ($\mathrm{MeV}$) in voxel $i$, the mass $m_i$ (in $\mathrm{kg}$) and absolute absorbed dose to medium $D_i$ (in $\mathrm{Gy}$) are computed as

$$
\begin{aligned}
m_i &= \rho_iV_i\times 10^{-6}\,\mathrm{kg}, \\
D_i &= \frac{E_i\times 1.602176634\times 10^{-13}\,\mathrm{J/MeV}}{m_i}
\frac{W}{N}.
\end{aligned}
\tag{8b}
$$

When a campaign is partitioned across $B$ batches (or multiple GPUs) with $N_b$ histories in batch $b$ ($\sum_{b=1}^B N_b = N$), each batch output is scaled by the campaign-wide normalization factor $W / N$ so that the final plan dose is obtained by simple high-precision (FP64) host summation $D_i = \sum_{b=1}^B d_{i,b}$.

### II.C.2. Dose-Averaged Electronic LET

Dose-averaged electronic linear energy transfer ($L_d$) is scored concurrently with absorbed dose using an unrestricted stopping-power track-length estimator. For each finite transport step $k$ of a charged ion traversing voxel $i$ with accepted true path length $\ell_k > 0$, let $S_{e,k}$ denote the unrestricted linear electronic stopping power ($\mathrm{MeV/mm}$, numerically equivalent to $\mathrm{keV/\mu m}$) evaluated at the pre-step kinetic energy and actual voxel density $\rho_i$. The engine accumulates two extensive voxel moments, $A_i$ and $B_i$, whose ratio yields the voxelwise dose-averaged electronic LET $L_{d,i}$:

$$
\begin{aligned}
A_i &= \sum_{k\in i}S_{e,k}^{\,2}\ell_k, \\
B_i &= \sum_{k\in i}S_{e,k}\ell_k, \\
L_{d,i} &= \frac{A_i}{B_i},\qquad B_i>0.
\end{aligned}
\tag{9}
$$

The weight in Eq. (9) is the expected electronic energy loss $S_{e,k}\,\ell_k$, linking LET scoring to the stopping response used for ion transport. Both $S_{e,k}$ and $L_{d,i}$ are linear stopping quantities evaluated at the actual voxel density and expressed in $\mathrm{keV/\mu m}$. Registered ions query the material-dependent unrestricted electronic stopping tables, and additional heavy target recoils query the electronic-only recoil stopping tables. Both LET moments are constructed from electronic stopping power.

All transported charged particles within the water or patient CT scoring grid—including primary ions, secondary nuclear fragments, and transported elastic recoils above the $0.1\text{ MeV}$ kinetic energy cutoff—contribute to $A_i$ and $B_i$. Upstream copper/air trajectory segments, discrete nuclear-vertex local energy deposits, zero-path terminal cutoff deposits, and untracked neutral reaction products are excluded from the LET moments. Both $A_i$ and $B_i$ are accumulated on the GPU via `FP32` atomic additions, written to disk independently for each field and batch, and summed in double precision across all fields, batches, and devices prior to evaluating the quotient $A_i / B_i$. Voxels with $B_i = 0$ (no traversed charged-particle track) are flagged explicitly and excluded from ratio averaging.

### II.C.3. TOPAS Reference Scoring

TOPAS absorbed dose is scored with `DoseToMedium`. The TOPAS reference uses the two-moment LET estimator of Eq. (9), accumulating $A_i = \sum_k w_k S_{e,k}^2 \ell_k$ and $B_i = \sum_k w_k S_{e,k} \ell_k$, where $w_k$ is the track statistical weight and the scoring threshold is $0.1\text{ MeV}$ in total kinetic energy. The scorer queries pre-extracted read-only unrestricted electronic stopping tables generated with the reference Geant4 11.3.2 physics list, particle species, Schneider tissue compositions, densities, and mean excitation energies. This establishes a common LET definition and scoring endpoint across the two engines.

---

## II.D. GPU Implementation

### II.D.1. Transport Specialization and Queue Scheduling

MAIGO is implemented in C++20 and Khronos SYCL, supporting NVIDIA GPUs through the oneAPI Unified Runtime CUDA backend and Intel Xe GPUs through Level Zero. Device kernels are compiled ahead of time into native code for the target architecture, with runtime just-in-time compilation disabled. Four compile-time-specialized primary pipelines serve proton broadbeam, carbon-ion broadbeam, proton minibeam, and carbon-ion minibeam calculations. A generic pipeline supports custom ion configurations. All pipelines share secondary transport kernels for multispecies charged nuclear fragments.

To mitigate SIMT warp/subgroup divergence caused by heterogeneous track lifetimes, nuclear branching, and collimator absorption, minibeam execution decouples particle transport into sequential device stages connected by preallocated global-memory particle queues:

1. **Upstream beamline stage**: Tracks incident primaries from the nozzle through the multislit copper collimator and air gap, storing surviving primary states at the patient entrance checkpoint and pushing copper-born charged fragments into dedicated upstream child queues.
2. **Upstream fragment cascade stage**: Transports copper-born charged products through the remaining collimator thickness and air gap across up to three inelastic generations.
3. **Patient primary transport stage**: Resumes surviving primary tracks from the patient entrance into the voxelized water or CT grid, optionally employing counting-sort compaction to group active tracks by spatial/material locality.
4. **Patient secondary fragment stage**: Transports all patient-born and upstream-transmitted charged secondaries in segments containing a fixed number of transport steps, compacting surviving long-lived tracks when active queue occupancy drops below a tail threshold.

Device queues are preallocated before kernel launch, with hardware atomic counters recording queue occupancy and completion across stages.

### II.D.2. Random Number Generation and Batch Execution

Device-side random numbers use the counter-based Philox-4x32-10 generator. Draws are keyed by batch seed, spot index, primary history index, descendant track identity, and a monotonically increasing counter, making each track’s random stream independent of thread scheduling and queue reordering. Consecutive batches reuse the GPU context and resident material and physics tables, amortizing initialization and physics-table upload costs across the campaign.

Multi-GPU campaigns partition histories in proportion to measured steady-state throughput to balance device completion times. The campaign-wide integer spot allocation in Eq. (8a) is preserved across devices and batches; independent history streams are combined using the dose and LET moment normalization of Section II.C.

### II.D.3. Energy Accounting and Transport Completion

Each batch checks track completion, queue capacity, physics-table coverage, and energy balance, saving the checks with dose and LET outputs. Required conditions are completion of all launched tracks, zero queue overflow, in-domain physics queries, and finite non-negative dose and LET moments. The energy ledger separates continuous deposition, nuclear-vertex deposition, terminal cutoff deposition, boundary escape, upstream loss, neutral products, and reaction residuals to check energy accounting by species and transport stage.

Projectile identity, material composition, and energy coverage are checked at initialization. Physics-data versions, software revision, integer spot allocations, and random seeds are recorded with each calculation to reproduce its source and physics configuration.

---

## II.E. Clinical Validation

### II.E.1. Clinical Cases and Treatment Plans

Clinical full-plan evaluation uses two head CT datasets—**RT07575** (Head 1) and **RT06423** (Head 2)—and one lung CT dataset, **20022516** (Lung 1). The head cases assess plan-dose agreement in complex anatomy, and the lung case assesses transport in low-density heterogeneous tissue. Table 2 lists the history counts for paired proton and carbon-ion calculations in broadbeam and copper-collimated minibeam modes. Each MAIGO–TOPAS pair shares the CT grid, Schneider material mapping [6], beam optics, spot positions and energies, final weights, placement, collimator geometry, and absolute dose normalization.

**Table 2.** Total simulated incident particle histories ($N$) for full-plan patient dose validations across clinical cases and beam delivery modes.

| Clinical Case | Anatomical Site | Proton Broadbeam | Carbon-12 Broadbeam | Proton Minibeam | Carbon-12 Minibeam |
|---|---|---:|---:|---:|---:|
| **RT07575** | Head & Neck 1 (Head 1) | $250,626,500$ | $129,638,170$ | $1,000,000,000$ | $1,000,000,000$ |
| **RT06423** | Head & Neck 2 (Head 2) | $124,266,340$ | $151,087,660$ | $1,000,000,000$ | $1,000,000,000$ |
| **20022516** | Thoracic (Lung 1) | $361,050,622$ | $177,173,040$ | $3,000,000,000$ | —$^{\mathrm{a}}$ |

$^{\mathrm{a}}$— denotes a case–modality pair not reported in this table. The default CT carbon minibeam configuration corresponds to a separate lung benchmark phantom.

Full-plan dose grids use $0.5 \times 0.5 \times 2.0\text{ mm}^3$ voxels, with dimensions, physical origin, and axis order defined by each plan’s reference header. Three-dimensional dose and LET are compared voxelwise in the shared patient coordinates, native grid, and absolute plan-dose scale.

Clinical treatment plans are defined by a sequence of fields, energy layers, and individual scanning spots, each parameterized by its nominal kinetic energy, transverse isocenter coordinates, and particle weight. An energy-dependent phase-space beam model characterizes the optical properties at the nozzle exit (or upstream source plane) via transverse spatial standard deviations $(\sigma_x, \sigma_y)$, angular divergences $(\sigma_{\theta_x}, \sigma_{\theta_y})$, position–angle covariance terms, and Gaussian energy spread $\sigma_E$. For each history, correlated transverse positions and direction cosines are sampled in the local spot coordinate system and projected into the global patient coordinate frame according to the virtual source-to-axis distances (SAD), gantry/couch rotations, and isocenter offset. Incident ion energies are parameterized in kinetic energy per nucleon ($\mathrm{MeV/u}$); treatment planning system (TPS) inputs tabulated in total kinetic energy ($\mathrm{MeV}$) are converted using the integer mass number $A$.

Physical-dose inverse optimization represents voxel dose as $\mathbf{d} = \mathbf{A}\mathbf{w}$, where $\mathbf{A}$ is the dose-influence matrix per incident particle and $\mathbf{w} \ge \mathbf{0}$ contains the spot particle weights. Weights are optimized for case-specific target coverage and organ-at-risk (OAR) sparing. The final vector is fixed for both MAIGO and TOPAS full-plan forward calculations, establishing a common irradiation source. Reoptimized plans use a $4\text{ Gy}$ physical-dose prescription; carbon-ion broadbeam plans retaining treatment planning system (TPS) weights use the original cumulative dose scale.

### II.E.2. TOPAS Reference Simulations

Reference simulations use OpenTOPAS 4.2.3 / Geant4 11.3.2 [7,8] with the scorers defined in Section II.C.3. The physics list includes standard electromagnetic processes (`g4em-standard_opt4`), hadron and ion inelastic interactions (`g4h-phy_QGSP_BIC_HP`, `g4ion-inclxx`), hadron elastic scattering (`g4h-elastic_HP`), stopping (`g4stopping`), and decay (`g4decay`, `g4radioactivedecay`). Minibeam references apply Urban scattering to charged particles through `AllChargedUrbanPhysics`. Carbon-ion references register nuclear elastic scattering for `GenericIon` through `CarbonIonElasticPhysics`, which wraps `G4IonElasticPhysics`.

Electromagnetic table limits, secondary production cuts, and step settings are recorded with each reference input. The lung proton minibeam reference uses a $0.05\text{ mm}$ production cut and a $10\text{ GeV}$ EM table upper limit; the three carbon-ion broadbeam references use a $600\text{ MeV}$ upper limit. Reference doses are generated by independent full-plan TOPAS transport driven by the final plan weights. TOPAS explicitly transports secondary electrons and photons above the production cut. MAIGO’s local-ionization deposition, secondary mean energy loss, nuclear final-state sampling, and neutral-product treatment are defined in Sections II.A–B.

To resolve stopping-power gradients near the Bragg peak, the carbon-ion broadbeam TOPAS LET reference uses `dRoverRange = 0.005` and `finalRange = 0.1 nm` for light ions and `GenericIon`, with the reference proton step controls retained. The RT07575 single-replica paired LET comparison uses $25,927,634$ incident histories and matching integer spot allocations in MAIGO and TOPAS. LET uncertainty from independent replicas is estimated as described in Section II.E.6.

### II.E.3. Dose Agreement

Spatial and dosimetric agreement between the GPU-calculated dose distribution $D_{\mathrm{GPU}}(\mathbf{r}')$ and the reference TOPAS dose distribution $D_{\mathrm{ref}}(\mathbf{r})$ is quantified using the three-dimensional local gamma index [10]:

$$
\begin{aligned}
\varepsilon_D &= \frac{\Delta D_{\%}}{100},\qquad \Delta r>0, \\
\gamma(\mathbf{r})
&= \min_{\substack{\mathbf{r}'\\\|\mathbf{r}'-\mathbf{r}\|\le\Delta r}}
\left[
\frac{\|\mathbf{r}'-\mathbf{r}\|^2}{(\Delta r)^2}
+\frac{\bigl[D_{\mathrm{GPU}}(\mathbf{r}')-D_{\mathrm{ref}}(\mathbf{r})\bigr]^2}
{\bigl[\varepsilon_D D_{\mathrm{ref}}(\mathbf{r})\bigr]^2}
\right]^{1/2}.
\end{aligned}
\tag{10}
$$

Here, $\Delta r>0$ is the distance-to-agreement (DTA) criterion and $\varepsilon_D=\Delta D_{\%}/100$ is the fractional dose tolerance relative to the local reference dose. Trilinear interpolation evaluates the GPU dose field within a sphere of radius $\Delta r$ around each reference point for subvoxel-distance gamma searches. Evaluation points are voxels inside `BODY` that exceed the dose threshold; interpolation uses the full GPU dose field, including neighboring voxels outside `BODY`. Table 3 lists the evaluation masks, local criteria, and search increments.

**Table 3.** Three-dimensional local gamma evaluation masks, acceptance criteria, and sub-voxel interpolation search steps.

| Delivery Mode | Evaluation Mask within `BODY` | Reported Local Gamma Criteria ($\Delta D_{\%}/\Delta r$) | Interpolation Search Step |
|---|---|---|---:|
| **Broadbeam** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\, D_{\mathrm{ref},\max}$ | $3\%/3\text{ mm}$, $1\%/1\text{ mm}$, $3\%/0\text{ mm}$ | $0.10\text{ mm}$ |
| **Minibeam** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.4\text{ Gy}$ ($10\%$ of $4\text{ Gy}$ prescription) | $3\%/0.2\text{ mm}$, $3\%/0.1\text{ mm}$, $3\%/0\text{ mm}$ | $0.02\text{ mm}$ |

An evaluation point passes when $\gamma \le 1$. For zero DTA, the endpoint is the same-position relative dose difference $\eta_D(\mathbf{r})=\lvert D_{\mathrm{GPU}}(\mathbf{r})-D_{\mathrm{ref}}(\mathbf{r})\rvert/D_{\mathrm{ref}}(\mathbf{r})$; the $3\%/0\text{ mm}$ test passes when $\eta_D\le 0.03$. The prespecified broadbeam acceptance criterion is a local $1\%/1\text{ mm}$ pass rate above $99\%$. Minibeams use $3\%/0.2\text{ mm}$ and $3\%/0.1\text{ mm}$ criteria with a $0.02\text{ mm}$ search increment to match the submillimeter slit scale. Transverse profiles at multiple depths assess peak width, valley dose, and PVDR; depth-dose profiles assess distal falloff; and DVH indices assess target and OAR dose. Same-position dose-difference distributions and batch uncertainties characterize voxelwise agreement.

### II.E.4. LET Agreement and Spatial Profiles

Three-dimensional local gamma evaluation for dose-averaged electronic LET ($L_d$) is computed analogously to Eq. (10) by substituting $L_{d,\mathrm{GPU}}(\mathbf{r}')$ and $L_{d,\mathrm{ref}}(\mathbf{r})$ for dose, with fractional tolerance $\varepsilon_L=\Delta L_{\%}/100$ and local absolute tolerance $\varepsilon_L L_{d,\mathrm{ref}}(\mathbf{r})$. The zero-DTA endpoint uses the same-position relative LET difference. The evaluation region is defined as the intersection of the `BODY` contour with the broadbeam dose mask $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\,D_{\mathrm{ref},\max}$. Broadbeam LET agreement is reported at local $3\%/3\text{ mm}$, $1\%/1\text{ mm}$, and $3\%/0\text{ mm}$ criteria using a $0.10\text{ mm}$ trilinear interpolation search step on the GPU LET field. Any voxel within the reference dose mask where the GPU LET denominator vanishes ($B_i = 0$) is retained as a gamma failure.

LET spatial analysis uses transverse profiles at fixed geometric depths and depth profiles along the beam axis. In the RT07575 carbon-ion broadbeam case, depth begins at the central-ray entry into `BODY` and is sampled at the native $0.5\text{ mm}$ interval. Central-ray profiles characterize local LET structure, and profiles pooled over a $2.5 \times 10\text{ mm}^2$ transverse ROI characterize regional depth trends. ROI LET is computed as $\sum_{i\in\mathrm{ROI}}A_i/\sum_{i\in\mathrm{ROI}}B_i$ within each depth slice.

### II.E.5. Dose Uncertainty

Voxelwise Monte Carlo statistical uncertainty on the GPU is evaluated from $B$ independent full-plan batches, each preserving the campaign-wide spot weight proportions. For $B$ equal-history batches whose individual voxel dose contributions $d_{i,b}$ are scaled such that $D_i = \sum_{b=1}^B d_{i,b}$, the unbiased sample variance of the total voxel dose $D_i$ is

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{B}{B - 1} \left[ \sum_{b=1}^B d_{i,b}^{\,2} - \frac{D_i^{\,2}}{B} \right].
\tag{11}
$$

For unequal batch history counts $N_b$, let $a_b=N_b/N>0$ be the history fraction and $Y_{i,b}=d_{i,b}/a_b$ the full-plan-equivalent batch estimate, so that $D_i=\sum_b a_bY_{i,b}$. For independent, source-matched batches with a common per-history variance, $\operatorname{Var}(Y_{i,b})=v_i/N_b$. The variance of the total dose is estimated by

$$
\begin{aligned}
\widehat{\sigma}_{D_i}^{\,2}
&= \frac{1}{B-1}\sum_{b=1}^B a_b\bigl(Y_{i,b}-D_i\bigr)^2 \\
&= \frac{1}{B-1}\sum_{b=1}^B
\frac{\bigl(d_{i,b}-a_bD_i\bigr)^2}{a_b}.
\end{aligned}
\tag{12}
$$

Equation (12) applies for $B>1$ and reduces to Eq. (11) for $a_b=1/B$.

TOPAS uncertainty is estimated from per-history scoring variance or matched pilot simulations. Per-history variances are converted to the plan-dose scale using the normalization factor. For the three carbon-ion broadbeam references, pilot simulations with $N_{\mathrm{pilot}}=300,000$ histories preserve the reference geometry, physics list, energy spectrum, and relative spot weights. Pilot standard errors are extrapolated to the full history count as $\widehat{\sigma}_{D_{\mathrm{ref},i}}=\widehat{\sigma}_{\mathrm{pilot},i}\sqrt{N_{\mathrm{pilot}}/N_{\mathrm{full}}}$ and cross-checked against inter-replica variance.

Across any fixed evaluation mask $\mathcal{R}$ containing $|\mathcal{R}|$ voxels, the overall statistical precision is summarized by the root-mean-square (RMS) relative standard error:

$$
\begin{aligned}
u_i &= 100\frac{\widehat{\sigma}_{D_i}}{D_{\mathrm{ref},i}}, \\
u_{\mathrm{RMS}}
&= \sqrt{\frac{1}{\lvert\mathcal{R}\rvert}\sum_{i\in\mathcal{R}}u_i^2}.
\end{aligned}
\tag{13}
$$

Both $u_i$ and $u_{\mathrm{RMS}}$ are expressed in percent. Uncertainty summaries also include voxelwise percentiles and the fractions with relative standard errors below $1\%$ or $2\%$. Because GPU and TOPAS simulations use independent random streams, the statistical variance of the voxelwise dose difference $\Delta D_i = D_{\mathrm{GPU},i} - D_{\mathrm{ref},i}$ is given by $\widehat{\sigma}_{\Delta D_i}^{\,2} = \widehat{\sigma}_{D_{\mathrm{GPU},i}}^{\,2} + \widehat{\sigma}_{D_{\mathrm{ref},i}}^{\,2}$.

### II.E.6. LET Uncertainty

Because $L_{d,i} = A_i / B_i$ is a ratio of two correlated random variables accumulated along the same particle tracks, its statistical variance accounts for the covariance $\mathrm{Cov}(A_i, B_i)$. To evaluate voxelwise LET statistical uncertainty at the full-plan history count $N$, an auxiliary GPU calculation partitions the $N$ histories into $K$ independent, source-matched full-plan groups with history counts $N_b$ and fractions $a_b=N_b/N>0$ ($K = 20$ for the carbon-ion broadbeam benchmark). Let $A_{i,b}$ and $B_{i,b}$ denote the accumulated numerator and denominator moments in replicate group $b \in \{1, \dots, K\}$, with pooled estimator $L_{d,i} = \sum_{b=1}^K A_{i,b} \big/ \sum_{b=1}^K B_{i,b}$. With $K>1$ and a common per-history covariance matrix for the paired moments, a first-order Taylor expansion (delta method) gives the variance estimate

$$
\widehat{\sigma}_{L_{d,i}}^{\,2}
\approx \frac{1}{K-1}
\frac{\displaystyle\sum_{b=1}^K
\frac{\bigl(A_{i,b}-L_{d,i}B_{i,b}\bigr)^2}{a_b}}
{\displaystyle\left(\sum_{b=1}^K B_{i,b}\right)^2}.
\tag{14}
$$

For equal-history groups, Eq. (14) reduces to $\widehat{\sigma}_{L_{d,i}}^{\,2}\approx [K/(K-1)]\sum_b(A_{i,b}-L_{d,i}B_{i,b})^2/(\sum_b B_{i,b})^2$. For TOPAS LET references partitioned into independent replicas (each saving paired $A_{i,b}$ and $B_{i,b}$ volumes), the reference LET standard error is computed analogously via Eq. (14). Voxelwise relative LET standard errors, regional RMS uncertainties within the $D_{\mathrm{ref}} > 0.10\,D_{\mathrm{ref},\max}$ mask, and quantiles are reported alongside the 3D LET gamma pass rates.

---

## II.F. Computational Performance

Computational performance is evaluated separately across GPU architectures—including NVIDIA GeForce RTX 2080 Ti (`sm_75`), NVIDIA RTX A6000 (`sm_86`), NVIDIA L4 (`sm_89`), and Intel Arc B580 (`bmg-g21`)—using AOT-compiled native binaries with runtime JIT compilation disabled. Two end-to-end wall-clock timing metrics are reported:

1. **Cold-start time**: Total elapsed wall-clock time from OS process invocation to completion of the first saved output batch, encompassing configuration parsing, GPU context initialization, host-to-device physics package uploads, kernel execution, runtime quality audits, and disk output.
2. **Steady-state throughput ($\mathrm{histories/s}$)**: Total simulated incident histories across subsequent resident batches divided by their cumulative end-to-end wall-clock time (including device-to-host readback, quality audits, and file I/O) within the same persistent process.

Stage-level kernel execution timers decompose GPU compute time across primary transport, upstream copper/air beamline transport, and secondary fragment transport, and quantify the marginal overhead of enabling concurrent LET scoring ($D_m + L_d$ vs. $D_m$ only) under identical history counts and batch sizes. Reference multi-threaded CPU TOPAS throughput ($\mathrm{histories/s}$) is measured on dual-socket server nodes (56-thread cluster nodes and 96-thread local workstations) from total simulated histories divided by cumulative node wall-clock time under matched geometry and physics configurations.

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
11. Urbán L. A model for multiple scattering in GEANT4. CERN; 2006. [CERN report](https://cds.cern.ch/record/1004190).
