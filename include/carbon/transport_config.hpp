#pragma once

#include "carbon/slab_phantom.hpp"
#include "carbon/particle.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

enum class RunMode : std::uint8_t {
    smoke,
    research,
    production,
};

[[nodiscard]] const char* run_mode_name(RunMode mode) noexcept;

enum class MaterialPhysicsMode : std::uint8_t {
    Water = 0,
    SchneiderCt = 1,
};

[[nodiscard]] const char* material_physics_mode_name(MaterialPhysicsMode mode) noexcept;

struct TransportConfig;
void validate_schneider_ct_startup(const TransportConfig& config);

// Compact per-spot source parameters consumed by one batched SYCL launch.
// history_begin/history_end describe the half-open range in the flattened plan.
//
// Layout note: all floats live in a single contiguous `floats[]` array (not
// separate named float members). Some SYCL device compilers have been observed
// to disagree with the host on offsetof() when uint64_t is followed by a long
// run of individual float fields, which silently zeroed source_origin_y while
// leaving source_origin_x correct. A plain float array has a trivial layout.
struct PrimarySpotBatchEntry {
    std::uint64_t history_begin{0};
    std::uint64_t history_end{0};
    std::uint64_t random_seed{0};
    // Packed floats — keep order stable; indices documented below.
    // 0 initial_energy_MeV
    // 1 beam_energy_spread
    // 2-3 emittance_sigma_x/y_mm
    // 4-5 emittance_sigma_x/y_prime
    // 6-7 emittance_correlation_x/y
    // 8-10 source_origin_x/y/z_mm
    // 11-13 beam_ux_x/y/z
    // 14-16 beam_uy_x/y/z
    // 17-19 beam_uz_x/y/z
    static constexpr int k_float_count = 20;
    float floats[k_float_count]{
        0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F,
    };

    float& initial_energy_MeV() noexcept { return floats[0]; }
    [[nodiscard]] float initial_energy_MeV() const noexcept { return floats[0]; }
    float& beam_energy_spread() noexcept { return floats[1]; }
    [[nodiscard]] float beam_energy_spread() const noexcept { return floats[1]; }
    float& emittance_sigma_x_mm() noexcept { return floats[2]; }
    [[nodiscard]] float emittance_sigma_x_mm() const noexcept { return floats[2]; }
    float& emittance_sigma_y_mm() noexcept { return floats[3]; }
    [[nodiscard]] float emittance_sigma_y_mm() const noexcept { return floats[3]; }
    float& emittance_sigma_x_prime() noexcept { return floats[4]; }
    [[nodiscard]] float emittance_sigma_x_prime() const noexcept { return floats[4]; }
    float& emittance_sigma_y_prime() noexcept { return floats[5]; }
    [[nodiscard]] float emittance_sigma_y_prime() const noexcept { return floats[5]; }
    float& emittance_correlation_x() noexcept { return floats[6]; }
    [[nodiscard]] float emittance_correlation_x() const noexcept { return floats[6]; }
    float& emittance_correlation_y() noexcept { return floats[7]; }
    [[nodiscard]] float emittance_correlation_y() const noexcept { return floats[7]; }
    float& source_origin_x_mm() noexcept { return floats[8]; }
    [[nodiscard]] float source_origin_x_mm() const noexcept { return floats[8]; }
    float& source_origin_y_mm() noexcept { return floats[9]; }
    [[nodiscard]] float source_origin_y_mm() const noexcept { return floats[9]; }
    float& source_origin_z_mm() noexcept { return floats[10]; }
    [[nodiscard]] float source_origin_z_mm() const noexcept { return floats[10]; }
    float& beam_ux_x() noexcept { return floats[11]; }
    [[nodiscard]] float beam_ux_x() const noexcept { return floats[11]; }
    float& beam_ux_y() noexcept { return floats[12]; }
    [[nodiscard]] float beam_ux_y() const noexcept { return floats[12]; }
    float& beam_ux_z() noexcept { return floats[13]; }
    [[nodiscard]] float beam_ux_z() const noexcept { return floats[13]; }
    float& beam_uy_x() noexcept { return floats[14]; }
    [[nodiscard]] float beam_uy_x() const noexcept { return floats[14]; }
    float& beam_uy_y() noexcept { return floats[15]; }
    [[nodiscard]] float beam_uy_y() const noexcept { return floats[15]; }
    float& beam_uy_z() noexcept { return floats[16]; }
    [[nodiscard]] float beam_uy_z() const noexcept { return floats[16]; }
    float& beam_uz_x() noexcept { return floats[17]; }
    [[nodiscard]] float beam_uz_x() const noexcept { return floats[17]; }
    float& beam_uz_y() noexcept { return floats[18]; }
    [[nodiscard]] float beam_uz_y() const noexcept { return floats[18]; }
    float& beam_uz_z() noexcept { return floats[19]; }
    [[nodiscard]] float beam_uz_z() const noexcept { return floats[19]; }
};

struct TransportConfig {
    // Flat configuration schema used by the current strict key/value parser.
    // Files that predate this field are interpreted as schema v1.
    std::uint32_t config_schema_version{1};
    // Stable, comment-free, key-sorted representation of the parsed input and
    // imported ion-physics manifest. This is intended for run provenance and
    // hashing; CLI overrides are not represented here.
    std::string canonical_config_text{};
    // Legacy configurations default to research so existing approximation
    // behavior remains visible but non-production. Production enables strict
    // post-transport quality gates.
    RunMode run_mode{RunMode::research};
    double quality_maximum_relative_energy_residual{1.0e-4};
    double quality_maximum_absolute_energy_residual_MeV{1.0e-6};
    bool quality_reject_any_queue_overflow{true};
    bool quality_reject_nan_or_inf{true};
    // Hard voxel/ledger closure: scored 3D voxel energy must not exceed the
    // global deposited total beyond tolerance (catches double-counted steps).
    bool quality_reject_voxel_over_total{true};
    // Explicit grid-split closure (total ≈ in + outside, voxel ≈ in-grid).
    bool quality_reject_grid_closure{true};
    // Optional single-file manifest owning the primary-ion identity and all
    // ion-dependent water physics data. Run controls remain in the main YAML.
    std::filesystem::path ion_physics_file{};
    std::size_t number_of_histories{10'000};
    // Fallback for single-beam runs. topas_spots_file(s) and tps_spots_file
    // supply per-spot energy and therefore do not require this value in YAML.
    double initial_energy_MeVu{200.0};
    // Relative RMS beam energy spread (TOPAS BeamEnergySpread percent / 100).
    // 0.01 = 1% → sample E ~ N(E0, (0.01*E0)^2) at primary birth.
    double beam_energy_spread{0.0};
    // Primary ion identity. Kinetic energies remain expressed per nucleon.
    int primary_atomic_number{6};
    int primary_mass_number{12};
    // Zero preserves the historical A * nucleon-mass approximation.
    double primary_rest_mass_MeV{0.0};
    // Legacy water-phantom names for the transport/scorer z extent. In a CT
    // run load_config() derives both from voxel_bins_z/voxel_size_z_mm, which
    // in turn default to the native patient CT header.
    double phantom_length_mm{400.0};
    double depth_bin_width_mm{0.5};
    // Explicit unified EM selection; legacy preserves historical input behaviour.
    std::string em_model{"legacy"};
    std::filesystem::path em_package_file{};
    std::string em_package_sha256{};
    // Research-only step extension, applied away from stopping/cut-onset regions.
    double em_primary_step_scale{1.0};
    double em_secondary_step_scale{1.0};
    std::string primary_em_model{"legacy"};
    double maximum_step_mm{0.5};
    double maximum_relative_energy_loss{0.005};
    std::uint32_t maximum_primary_steps{2000000U};
    double energy_cutoff_MeV{0.1};
    // Charged secondaries below this total kinetic energy are stopped and their
    // remaining energy is deposited locally. 0 uses energy_cutoff_MeV.
    double secondary_local_deposit_cutoff_MeV{0.0};
    // Ablation: when >0, charged reaction/cascade products with Z >= this value
    // deposit kinetic energy as local nuclear heat at the interaction point
    // instead of being queued for charged secondary transport. 0 disables
    // (default: transport all supported charged secondaries). Z_min=3 deposits
    // Li and heavier as local heat while transporting H/He fragments.
    int secondary_heavy_local_deposit_z_min{0};
    // Optional condensed-history step for charged secondaries in a uniform
    // phantom. 0 preserves the legacy behavior where every physics step stops
    // at a depth-score boundary. A positive value lets MCS/straggling advance
    // over several depth bins while continuous dose is split back into the
    // original bins by track length.
    double secondary_condensed_step_mm{0.0};
    // Uniform-water density and CT-outside fallback; CT voxels use their own
    // material density from CCTG.
    double water_density_g_per_cm3{1.0};
    // When true, axial slabs override uniform water.
    // Density-only mode: SP/XS water tables × local density (water-equivalent).
    // Material mode: per-layer SP/XS CSV (absolute MeV/mm and macro 1/mm); density
    // used for MCS/straggling only. Empty layers => uniform.
    bool enable_layered_phantom{false};
    std::vector<SlabLayer> slab_layers{};
    // Optional absolute material tables, one path per slab layer (same length).
    std::vector<std::filesystem::path> slab_stopping_power_files{};
    std::vector<std::filesystem::path> slab_cross_section_files{};
    // Optional mass radiation length per layer (g/cm2). Empty preserves the
    // historical all-water MCS model.
    std::vector<double> slab_radiation_lengths_g_per_cm2{};
    // Lateral/3D insert (7b): AABB of alternate material inside water background.
    bool enable_hetero_insert{false};
    HeteroInsert hetero_insert{};
    std::filesystem::path insert_stopping_power_file{};
    std::filesystem::path insert_cross_section_file{};
    // Water default preserves existing insert configurations.
    double insert_radiation_length_g_per_cm2{36.08};
    // 7c: CT voxel grid (exclusive with layered/hetero insert).
    bool enable_ct_grid{false};
    std::filesystem::path ct_grid_file{};
    // Optional TOPAS Schneider HU table. Used when ct_grid_file is a DICOM
    // folder. Empty searches data/HUtoMaterialSchneider.txt.
    std::filesystem::path ct_schneider_file{};
    // DICOM origin: "centered" matches the historical CCTG converter
    // (volume centred, first-slice centre at z=0). "dicom" keeps native
    // LPS voxel low edges from ImagePositionPatient.
    std::string ct_dicom_origin_mode{"centered"};
    // When true (default), skip CT voxel-face step clamps in homogeneous regions.
    // Set false for isolated performance A/B against full face clamping.
    bool ct_skip_homogeneous_face_clamp{true};
    // Optional absolute SP/XS for CT materials 0..3 (air/lung/water/bone).
    // Empty → water table × local density for all materials.
    std::filesystem::path ct_air_stopping_power_file{};
    std::filesystem::path ct_lung_stopping_power_file{};
    std::filesystem::path ct_water_stopping_power_file{};
    std::filesystem::path ct_bone_stopping_power_file{};
    std::filesystem::path ct_air_cross_section_file{};
    std::filesystem::path ct_lung_cross_section_file{};
    std::filesystem::path ct_water_cross_section_file{};
    std::filesystem::path ct_bone_cross_section_file{};
    // Top-level physics routing mode: Water vs SchneiderCt
    MaterialPhysicsMode material_physics_mode{MaterialPhysicsMode::Water};

    // Canonical key names for water and Schneider CT physics
    std::filesystem::path water_cinel_package_file{};
    // Derived routing: native water material, shared CINEL03 framework. No legacy fallback.
    bool unified_water_nuclear_transport{true};
    std::filesystem::path unified_water_material_file{};
    std::string unified_water_material_sha256{};
    std::filesystem::path water_reaction_rate_file{};
    std::filesystem::path ct_schneider_c12_cinel03_file{};
    std::filesystem::path ct_schneider_secondary_cinel03_file{};
    std::filesystem::path ct_schneider_primary_rate_file{};
    std::filesystem::path ct_schneider_secondary_rate_file{};
    // C12 elastic rates on the Schneider grid (SCHNELXS v1, same layout as
    // SCHNRATE v3). Empty = elastic disabled (current production behavior).
    std::filesystem::path ct_elastic_section_rate_file{};
    std::string ct_elastic_section_rate_sha256{};
    // Diagnostic master switch for section elastic (default off;
    // smoke-gated until halo/profile validation promotes it).
    bool ct_elastic_diagnostic{false};
    // Legacy C12 isotropic-CM diagnostic, not a TOPAS angular distribution.
    // Use the separate all-ion bank for physical target-specific sampling.
    bool ct_elastic_all_targets{false};
    // Independent all-projectile TOPAS elastic bank; never uses the C12 isotropic fallback.
    std::filesystem::path all_ion_elastic_file{};
    std::string all_ion_elastic_sha256{};
    std::filesystem::path elastic_recoil_stopping_file{};
    std::string elastic_recoil_stopping_sha256{};
    // v2.1 physics bundle manifest (REQUIRED whenever any Schneider rate
    // file is binary version 3; REFUSED with v1 files: no mixing).
    std::filesystem::path ct_schneider_physics_bundle_file{};
    // Out-of-scope projectile nuclear policy (Schneider CT only). The ONLY
    // allowed value is "em_only": registry-unknown projectiles keep charged
    // EM transport with secondary nuclear reactions disabled; no other value
    // and no default fallback exist (empty refuses startup).
    std::string secondary_out_of_scope_nuclear_policy{};
    std::filesystem::path ct_schneider_stopping_power_file{"data/schneider/schneider_stopping_v1.bin"};
    std::filesystem::path ct_schneider_radiation_length_file{"data/schneider/schneider_radiation_lengths.json"};
    // Optional TOPAS-derived primary C12 delta-electron transverse tail. Empty
    // preserves the condensed local-deposit path exactly. Schneider section 0 only.
    std::filesystem::path ct_schneider_delta_tail_file{};
    // Optional TOPAS-derived primary C12 forward (longitudinal) delta kernel,
    // supplementing the transverse tail. Requires ct_schneider_delta_tail_file;
    // empty disables the forward move exactly. Schneider section 0 only.
    std::filesystem::path ct_schneider_delta_longitudinal_file{};
    // Frozen diagnostic only. Per-patient calibration is forbidden.
    double ct_schneider_delta_longitudinal_scale{1.0};
    // Explicit unvalidated 175 MeV/u uniform section-0 three-density experiment.
    bool ct_longitudinal_homogeneous_density_diagnostic{false};
    // Unvalidated 175 MeV/u EM-only air/soft-tissue mass-column experiment.
    bool ct_longitudinal_interface_mass_diagnostic{false};
    // Independent correlated bulk response, isolated EM-only interface pilot.
    std::filesystem::path ct_electron_joint_response_diagnostic_file{};
    bool ct_electron_joint_patient_experiment{false};
    // Explicit opt-in: allow the joint patient experiment in research mode
    // (smoke stays allowed as before; production stays forbidden). Quality
    // still reports unvalidated_electron_joint_response, so runs remain
    // non-accepting until the joint data passes production validation.
    bool ct_electron_joint_allow_research{false};
    // Production speed switch (default ON = current diagnostic behavior).
    // false skips reporting-only joint diagnostic atomics (queries,
    // redistributed/escaped/untracked energy, path replays). Miss/lookup
    // counters stay always-on: they feed fail-closed quality gates.
    // Skipped counters never feed transport, so dose is bit-identical.
    bool electron_joint_diagnostics{true};
    std::string ct_electron_joint_response_sha256{};
    std::string ct_electron_joint_response_metadata_sha256{};
    // Unvalidated pure Water_75eV EM-only experiment; never a CT alias.
    std::filesystem::path water_electron_response_diagnostic_file{};
    std::string water_electron_response_sha256{};
    bool water_electron_high_energy_diagnostic{false};
    std::string water_electron_response_metadata_sha256{};
    bool water_electron_nuclear_diagnostic{false};
    std::filesystem::path material_electron_response_index_file{};
    std::string material_electron_response_index_sha256{};
    std::uint64_t material_electron_response_device_budget_MiB{8192};
    std::string material_electron_response_memory_mode{"device"};
    double material_electron_short_range_mm{0.0}; // research-only, zero disables
    std::uint64_t material_electron_response_host_budget_MiB{98304};

    // Optional energy-dependent mass XS for every Schneider section. When set
    // on a CCTG v2/v3 grid, this supersedes the legacy four-class XS tables.
    std::filesystem::path ct_schneider_cross_section_file{};
    std::filesystem::path ct_cinel02_rate_file{};
    std::filesystem::path ct_hu_stopping_power_lut_file{};
    // Optional CT validation mode. Supported: "none", "primary-attenuation-only".
    // "primary-attenuation-only" enforces C12 primary nuclear attenuation validation,
    // terminating primaries on first inelastic collision and disabling all secondary creation/transport.
    std::string ct_validation_mode{};
    // moquimc-style continuous mass SPR(rho, E) for CT ionization and LET.
    // When true and no HU LUT is configured, voxel density selects the
    // water-relative electronic mass SPR instead of the analytic
    // Schneider-section Bethe factor.
    bool ct_use_density_mass_spr{true};
    // When true, fragment-species EnergyDeposit for Z in
    // [restrict_fragment_species_z_min, restrict_fragment_species_z_max]
    // uses (1 − δ) × unrestricted dE so the ion column matches TOPAS
    // G4Step::GetTotalEnergyDeposit (δ rays leave the ion scorer).
    // Transport and the aggregate energy ledger stay unrestricted.
    // 0,0 leaves every fragment species unrestricted (production default).
    bool restrict_fragment_species_energy_deposit{false};
    int restrict_fragment_species_z_min{0};
    int restrict_fragment_species_z_max{0};
    // Blend light-ion (Z<=2) reaction/cascade birth directions toward the
    // projectile axis: dir = normalize((1-f)*package_dir + f*projectile_dir).
    // 0 keeps the correlated INCL++ angular package; values in (0,1] narrow
    // the light-fragment cone (diagnostic residual-closure lever for multi-spot
    // mottling / distal secondary fill). Does not change product KE.
    double reaction_light_ion_forward_mix{0.0};
    // Global multiplier on CT mass-scaled / material stopping power (default 1).
    // Used to absorb residual WEPL calibration vs full Geant4 material SP.
    double ct_stopping_power_scale{1.0};
    double scorer_area_mm2{90'000.0};
    // Output-only absolute dose calibration. This scales Gy scorers while
    // preserving raw deposited-energy tallies and transport energy balance.
    // Keep at 1.0 unless an independent open-field calibration is available.
    double dose_output_scale{1.0};
    bool enable_voxel_scoring{false};
    bool enable_charged_origin_voxel_scoring{false};
    // Historical mode stops every physics step at lateral scorer faces.
    // Disable to keep scoring resolution from changing MCS/transport; energy
    // is then assigned to the voxel containing the step start.
    bool voxel_scorer_clamps_transport{false};
    // Dense scorer geometry in patient coordinates. X-Y is the axial plane;
    // Z is the inferior-superior slice direction. For CT runs these values are
    // read from the CCTG header unless explicitly supplied for compatibility.
    std::size_t voxel_bins_x{60};
    std::size_t voxel_bins_y{60};
    std::size_t voxel_bins_z{0};
    double voxel_origin_x_mm{0.0};
    double voxel_origin_y_mm{0.0};
    double voxel_origin_z_mm{0.0};
    double voxel_size_x_mm{5.0};
    double voxel_size_y_mm{5.0};
    double voxel_size_z_mm{0.0};
    bool enable_energy_straggling{false};
    // Smoke-only read-only primary query histogram; no RNG or physics changes.
    bool enable_primary_loss_query_audit{false};
    // Smoke ablation: transport terminal-generation charged products with EM
    // while retaining the existing cap on their nuclear interactions.
    bool enable_terminal_generation_em_transport{false};
    // gaussian_clamped / legacy_calibrated: historical Gaussian + [0, 2μ] cap.
    // moment_matched: positive Gamma/Gaussian sampler that keeps mean/variance
    // without a 2μ clamp; scale tables are rejected.
    std::string energy_straggling_model{"gaussian_clamped"};
    std::filesystem::path energy_straggling_package_file{};
    // Optional homogeneous-water CSDA residual-range energy loss on CPU/SYCL.
    // Disabled preserves each backend's existing local-loss calculation.
    bool enable_csda_range_energy_loss{false};
    // When enabled, primary straggling is sampled on fixed physical blocks so
    // subdividing a transport step does not create new independent Gaussians.
    bool enable_step_stable_straggling{false};
    // Physical sampling length for step-stable primary straggling. 0 disables
    // the mode and preserves the historical per-step sampler.
    double straggling_sampling_length_mm{0.0};
    // Historical validation applied straggling only to primary ion.
    // Enable this separately to apply Bohr straggling to charged fragments.
    bool enable_secondary_energy_straggling{false};
    // Unified EM for secondary charged fragments (native fluctuations, delta
    // clock, material-specific stopping). Off = CSDA fallback
    // (mid_sp x step, packaged fluctuation for C12 only).
    bool enable_secondary_unified_em{false};
    // GPU generation-wise species ordering. Production YAML opts in explicitly.
    bool secondary_species_grouping{false};
    // Pause after 64 complete secondary iterations and compact survivor indices.
    // Production presets opt in; false retains full-track scheduling without state buffers.
    bool secondary_step_chunking{false};
    // Scalar fallback for Bohr straggling. If the two optional tables below
    // are populated, the scale is linearly interpolated using the current
    // particle E/A rather than the incident beam energy. Empty tables retain
    // the historical scalar behavior.
    double straggling_scale{1.0};
    std::vector<double> straggling_scale_energies_MeVu{};
    std::vector<double> straggling_scale_values{};
    bool enable_inelastic{false};
    // geant4: load the configured macroscopic table as-is.
    // fred_paper: C-C fit + Kox scaling + ICRU-H; optional H-target elastic.
    std::string nuclear_model{"geant4"};
    // TOPAS-derived correlated multi-generation final states and independent
    // target-conditioned interaction rates (CINPKG03 / CINEL02_RATE_V1).
    std::filesystem::path primary_inelastic_package_v2_file{};
    std::filesystem::path primary_inelastic_rate_v2_file{};
    // Number of secondary generations allowed to undergo CINEL02 inelastic
    // replay. Zero restricts inelastic reactions to the primary projectile.
    // Validated production configurations are limited to at most two.
    std::uint32_t cinel02_max_secondary_inelastic_generations{0};
    // Reproduce TOPAS/Geant4 behavior for unsupported prompt-unstable
    // products. Currently this is only Be-6: count its generated kinetic
    // energy as an explicit compatibility sink and do not queue it.
    bool cinel02_topas_compatibility_mode{false};
    // Reject a run when a sampled collision has no valid replay event or the
    // hazard/outcome counters do not close.
    bool cinel02_strict_match{false};
    bool enable_nuclear_elastic{false};
    std::filesystem::path fred_event_library_h_file{};
    std::filesystem::path fred_event_library_o_file{};
    std::vector<std::filesystem::path> fred_event_library_h_files{};
    std::vector<std::filesystem::path> fred_event_library_c_files{};
    std::vector<std::filesystem::path> fred_event_library_o_files{};
    std::filesystem::path primary_inelastic_cross_section_file{
        "data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv"};
    bool enable_secondary_transport{true};
    bool enable_multiple_scattering{false};
    // highland / fred_2gr. The latter uses the FRED 3.76 tabulated mixture.
    std::string multiple_scattering_model{"highland"};
    std::filesystem::path fred_2gr_mcs_file{
        "data/packages/fred_3_76_mcs_2gr.bin"};
    // zero preserves FRED 3.76 table-domain behavior;
    // kinematic_extrapolation freezes the 236 MeV/u mixture and scales
    // the sampled angle by (beta*p)_236/(beta*p)_E.
    std::string fred_2gr_high_energy_mode{"zero"};
    // Multiplies Highland projected RMS angle for charged MCS (primary +
    // secondary). 1.0 is the historical default. Values >1 increase lateral
    // fill; keep ≤~1.5 without new validation. Not a per-patient fit.
    double multiple_scattering_scale{1.0};
    // Use Geant4 mass radiation lengths for CT air/lung/water/bone classes.
    // Off preserves the historical all-water MCS model exactly.
    bool enable_ct_material_mcs{false};
    // TOPAS-style flat rectangular source in the local beam frame.
    bool enable_flat_source{false};
    double flat_source_half_width_x_mm{0.0};
    double flat_source_half_width_y_mm{0.0};
    // TOPAS-style BiGaussian emittance source.
    // Samples (x,x') and (y,y') from bivariate Gaussians in the local beam frame;
    // x' = dx/dz_local (rad-like). Local frame defaults to world +z beam.
    bool enable_emittance_source{false};
    double emittance_sigma_x_mm{0.0};
    double emittance_sigma_y_mm{0.0};
    double emittance_sigma_x_prime{0.0};
    double emittance_sigma_y_prime{0.0};
    double emittance_correlation_x{0.0};
    double emittance_correlation_y{0.0};
    // Source pose (world mm). Origin is the beam-source point; particles start
    // at origin + ux*x + uy*y and travel along rotated local +uz.
    double source_origin_x_mm{0.0};
    double source_origin_y_mm{0.0};
    double source_origin_z_mm{0.0};
    double beam_ux_x{1.0}, beam_ux_y{0.0}, beam_ux_z{0.0};
    double beam_uy_x{0.0}, beam_uy_y{1.0}, beam_uy_z{0.0};
    double beam_uz_x{0.0}, beam_uz_y{0.0}, beam_uz_z{1.0};
    // Optional upstream minibeam collimator. The first implementation is an
    // exact absorbing-geometry fast path: only straight rays that stay inside
    // one air slit reach the phantom. Copper EM/nuclear transport is added in
    // later modes; the default false preserves every existing source path.
    bool enable_minibeam{false};
    // Detailed collimator counters and beam moments are useful for validation,
    // but their global atomics are redundant in production dose runs. Beamline
    // removed energy is still accumulated for the global energy balance.
    bool enable_minibeam_diagnostics{true};
    std::string minibeam_transport_mode{"absorbing_geometry"};
    std::string minibeam_material{"Copper"};
    double minibeam_radius_mm{60.0};
    double minibeam_thickness_mm{60.0};
    double minibeam_exit_to_phantom_mm{60.0};
    int minibeam_slit_count{15};
    double minibeam_slit_width_mm{0.5};
    double minibeam_slit_pitch_mm{3.6};
    double minibeam_slit_half_length_mm{25.0};
    // Translation of the complete slit array along its periodic local-u axis.
    // Zero preserves the historical centred array; a half-pitch offset is
    // required by some opposed-field patient apertures.
    double minibeam_slit_offset_mm{0.0};
    double minibeam_collimator_angle_deg{0.0};
    // Copper electromagnetic-only development mode. The table is absolute
    // electronic dE/dx in MeV/mm at the native material density.
    std::filesystem::path minibeam_copper_stopping_power_file{};
    std::filesystem::path minibeam_air_stopping_power_file{};
    double minibeam_copper_density_g_per_cm3{8.96};
    double minibeam_copper_radiation_length_g_per_cm2{12.8628};
    double minibeam_copper_max_step_mm{0.05};
    bool minibeam_copper_enable_mcs{true};
    // Sample Bohr energy-loss fluctuations step-by-step in Copper.  This is
    // independent of downstream water straggling so legacy minibeam cases
    // remain reproducible unless explicitly enabled.
    bool minibeam_copper_enable_energy_straggling{false};
    double minibeam_copper_straggling_scale{1.0};
    // Frozen against an independent 2150 MeV C-12 Copper-foil benchmark using
    // the same TOPAS/Geant4 release. This scales the projected Highland core.
    double minibeam_copper_mcs_scale{0.785};
    // Residual accumulated Copper-loss correction for primary ion ions that
    // survive to the collimator exit. It is frozen from an independent
    // water-entrance phase-space comparison and deliberately does not alter
    // nuclear interaction probability, survival, or angular transport.
    double minibeam_copper_survivor_energy_loss_scale{1.0};
    // Optional piecewise-linear incident-energy calibration of the same
    // accumulated-loss scale. Empty vectors preserve the scalar behavior.
    // The values are constrained by Copper-touched primary ion phase space,
    // independently of the downstream dose comparison.
    std::vector<double>
        minibeam_copper_survivor_energy_loss_energies_MeVu{};
    std::vector<double>
        minibeam_copper_survivor_energy_loss_scales{};
    // Residual primary range correction in the downstream water phantom.
    // Default 1.0; the minibeam value is frozen from a TOPAS/GPU primary-origin
    // R80 comparison after the water-entrance phase space has been matched.
    double minibeam_water_primary_stopping_power_scale{1.0};
    // Optional low-energy correction to the projected Highland core in the
    // downstream water phantom.  The scale approaches the configured value at
    // zero energy and smoothly returns to one at transition_MeVu.  Primary
    // C-12 continuations and charged fragments are kept separate because their
    // distal lateral-dose sensitivities are different.  Defaults are neutral.
    double minibeam_water_low_energy_mcs_transition_MeVu{0.0};
    double minibeam_water_primary_low_energy_mcs_scale{1.0};
    double minibeam_water_fragment_low_energy_mcs_scale{1.0};
    // Optional entrance response for primary ion histories that touched the
    // Copper collimator. The two smooth terms modify stopping (not scored dose),
    // so retained kinetic energy continues downstream. Zero amplitudes are
    // neutral and preserve all non-minibeam/default behaviour.
    double minibeam_water_touched_primary_surface_boost{0.0};
    double minibeam_water_touched_primary_surface_sigma_mm{0.8};
    double minibeam_water_touched_primary_deficit{0.0};
    double minibeam_water_touched_primary_deficit_center_mm{2.2};
    double minibeam_water_touched_primary_deficit_sigma_mm{1.0};
    bool minibeam_copper_enable_nuclear_attenuation{false};
    std::filesystem::path minibeam_copper_cross_section_file{};
    std::filesystem::path minibeam_copper_ion_stopping_power_file{};
    std::filesystem::path minibeam_copper_ion_cross_section_file{};
    std::filesystem::path minibeam_copper_neutral_cross_section_file{};
    // Used by spots_geometry_mode=minibeam_topas_y to map TOPAS world Y to
    // canonical GPU depth: gpu_z = world_y - this value.
    double minibeam_water_entrance_world_y_mm{60.0};
    // Non-empty only for a flattened multi-spot SYCL plan. All entries share
    // the physics/scorer fields above; source and energy fields come from here.
    std::vector<PrimarySpotBatchEntry> primary_spot_batch{};
    // TOPAS-format spots plan (spots_*.txt L0–L14). Input-compatible with TOPAS
    // TimeFeature dumps; GPU does NOT simulate timeline — it just runs spots in
    // file order and accumulates absolute energy (MeV), then normalizes by total
    // histories across spots.
    std::filesystem::path topas_spots_file{};
    // Multiple TOPAS files are concatenated in the listed order. This is used
    // for plans split only for parallel TOPAS execution.
    std::vector<std::filesystem::path> topas_spots_files{};
    // Optional one-column optimization weights, one per concatenated spot.
    // number_of_histories becomes the total plan MC budget and is allocated
    // proportionally (rather than overriding every spot with an equal count).
    std::filesystem::path spot_weights_file{};
    // |TransY| used when building spot source pose (TOPAS SAD). Default 450 mm.
    double spots_sad_mm{450.0};
    // "topas": legacy full BeamPosition2 pose (Rx,Ry + SAD).
    // "beam_plus_z": water-IDD convenience — force +z beam at z=0 with lateral
    //   offsets (TransX,TransZ) as (x,y); ignores gantry for depth scoring.
    // The modes below are legacy beam-repacked-CT adapters. New clinical CT
    // configurations should use tpsSource with a native patient-coordinate CT.
    // "tps_90": TPS 90-degree incidence. Spot rotations establish the fixed
    //   TPS 0-degree source direction, while Patient RotZ=90 supplies the plan
    //   angle. The transform maps either patient-X travel direction to GPU +Z;
    //   use the xneg CT packing when Patient RotZ makes the beam travel −X.
    // "tps_gantry_y": TPS 0-degree incidence used by lung plans. The CT is
    //   packed as GPU (x,y,z)=patient (x,z,+/-y), preserving each spot's L7/L8.
    std::string spots_geometry_mode{"topas"};
    double spots_patient_trans_x_mm{0.0};
    double spots_patient_trans_y_mm{0.0};
    double spots_patient_trans_z_mm{0.0};
    double spots_patient_rot_z_deg{0.0};
    double spots_ct_axis_min_mm{0.0};
    // Multiplies TOPAS BiGaussian SigmaX/Y when applying spots (position width).
    // 1.0 = plan values. Modest >1 widens entrance/outer envelope.
    double spots_emittance_sigma_scale{1.0};
    // Multiplies SigmaXprime/Yprime (angular divergence).
    double spots_emittance_prime_scale{1.0};
    // Optional continuous C-12 energy loss through the TOPAS World air gap
    // before the reoriented CT entrance. This is intentionally separate from
    // CT in-grid air material handling. It applies only to TPS spot geometry
    // modes, is disabled by default, and currently omits air MCS/straggling.
    bool spots_enable_upstream_air_energy_loss{false};
    // Smoke-only uniform World-Air MCS covariance; no beam-model file tuning.
    bool spots_enable_upstream_air_mcs{false};
    bool ct_primary_midpoint_stopping_diagnostic{false};
    // Formal predictor-midpoint Schneider stopping (second-order
    // in-step energy loss vs entry-value first order). Default OFF:
    // full20 gamma validation was neutral-to-negative (lung local30
    // 81.35->80.62, refined local11 98.25->98.20; band means improve
    // but tails widen slightly, likely 0.5%-cap interplay near the
    // Bragg peak). Available for research via this key or the
    // diagnostic alias.
    bool ct_primary_midpoint_stopping{false};
    // Scheme-2 secondary stopping: per-ion per-Schneider-section Geant4
    // tables (SCHNIOSP v1). Empty = current water x material-factor path.
    // Values are unrestricted totals (electron contract preserved).
    std::filesystem::path ct_secondary_ion_section_stopping_file{};
    std::string ct_secondary_ion_section_stopping_sha256{};
    std::string ct_secondary_ion_section_stopping_metadata_sha256{};
    bool ct_secondary_exact_faces_diagnostic{false};
    // Promoted formal behavior (validated: lung full20 local30 81.35->91.41,
    // single-shard cost +7%). Default ON. The diagnostic key above remains
    // as a deprecated explicit alias: when present it overrides this field,
    // so frozen runs stay bit-reproducible with an explicit false.
    // Inert without a CT grid (callers branch on enable_ct_grid && sec_in_ct).
    bool ct_secondary_exact_faces{true};
    bool ct_secondary_mcs_off_diagnostic{false};
    bool ct_secondary_schneider_sp_diagnostic{false};
    std::filesystem::path spots_upstream_air_mcs_file{};
    std::string spots_upstream_air_mcs_sha256{};
    std::filesystem::path spots_upstream_air_stopping_power_file{};
    // Keep all clinical geometry in the fixed DICOM LPS patient frame. This is
    // independent of the spot input format (TOPAS TimeFeature or TPS CSV).
    bool enable_tps_coordinate_system{false};
    // TPS CSV source switch. tpsSource/tps_source are legacy YAML aliases.
    bool enable_tps_source{false};
    // Optional CSV columns: spot_id,energy_MeVu,x_mm,y_mm,mu_weight plus
    // energy_spread_percent and emittance parameters. Empty => one central
    // spot using initial_energy_MeVu and the source defaults above.
    // PencilBeamScanning aliases energy_MeV (total ion KE) and weight are
    // accepted when the corresponding legacy columns are absent.
    std::filesystem::path tps_spots_file{};
    // Optional energy-dependent optics table (PBS beam_model.csv). Applied
    // only to spots that omit per-row emittance / energy-spread columns.
    std::filesystem::path tps_beam_model_file{};
    // PBS virtual scanning magnets. Both must be positive to enable the
    // isocenter-aimed scan geometry. Zero keeps the historical parallel SAD
    // offset (x,y placed on the source plane, direction = central ray).
    double tps_virtual_scanning_magnet_x_mm{0.0};
    double tps_virtual_scanning_magnet_y_mm{0.0};
    // Physical source-plane distance D. Zero falls back to tps_sad_mm.
    double tps_virtual_source_to_isocenter_mm{0.0};
    // When true (or when spots_patient_rot_z_deg != 0 with virtual magnets),
    // build the PBS beam in the TOPAS world TPS-0° frame and apply the same
    // Patient Trans/RotZ + tps_90 CT packing as Time Feature plans.
    // TOPAS applies RotZ first, then the recorded Trans — do not replace that
    // with tps_beam_angle_deg on an unrotated CT.
    bool tps_apply_topas_patient_placement{false};
    // "mu": treat the weight column as MU and Hamilton-allocate
    // number_of_histories (existing TPS CSV). "histories": treat weight as
    // an exact history count, optionally scaled by tps_histories_scale.
    std::string tps_spot_weight_mode{"mu"};
    double tps_histories_scale{1.0};
    // Source coordinates are DICOM LPS: +X patient-left, +Y posterior, +Z
    // superior. "dicom_lps" keeps the CT fixed, defines TPS 0 deg as +Y, and
    // rotates the beam frame about +Z: w=(-sin(theta), cos(theta), 0).
    // "topas_patient_rot_z" is accepted as a deprecated alias. "iec61217" is
    // retained for legacy water-phantom configurations.
    std::string tps_angle_convention{"iec61217"};
    // Public YAML name: tps_beam_angle_deg. This is the in-plane beam angle
    // about patient +Z for dicom_lps; tps_gantry_angle_deg is a legacy alias.
    double tps_gantry_angle_deg{0.0};
    double tps_couch_angle_deg{0.0};
    double tps_collimator_angle_deg{0.0};
    double tps_isocenter_x_mm{0.0};
    double tps_isocenter_y_mm{0.0};
    double tps_isocenter_z_mm{0.0};
    double tps_sad_mm{1000.0};
    // HFS, HFP, FFS, or FFP. Axes are converted to the simulation's patient
    // coordinate system; isocenter coordinates are already in that system.
    std::string tps_patient_position{"HFS"};
    // Per-species secondary depth-dose scoring. Off in production; validation
    // mode turns it on. Voxel-only full-plan runs should leave this false.
    bool enable_fragment_species_scoring{false};
    // "production": DoseToMedium + optional LET_d.
    // "validation": extra fluence / survival / reaction / ledger outputs.
    // Requires a binary built with -DCARBON_VALIDATION_SCORERS=ON.
    std::string scorer_mode{"production"};
    // If set in validation mode, default extra CSV/JSON paths are placed here.
    std::filesystem::path validation_output_directory{};

    [[nodiscard]] bool validation_scorers() const noexcept;
    // Dose-averaged electronic LET scorer compatible with the
    // Villadslj/Topas-Extension myHadronLET definition. YAML also accepts the
    // requested camel-case alias `scorerLET`.
    bool enable_let_scoring{false};
    // V1 transports only neutral-package e-/e+ in homogeneous G4_WATER.
    // Ion delta electrons remain condensed in unrestricted ion stopping power.
    bool enable_electron_transport{false};
    std::filesystem::path electron_transport_data_file{};
    std::size_t electron_queue_capacity{0};
    double electron_kinetic_cutoff_MeV{0.01};
    std::uint32_t maximum_electromagnetic_generations{4};
    double maximum_electron_step_mm{0.1};
    double maximum_electron_relative_energy_loss{0.05};
    // Short-range electronic build-up vs unrestricted CSDA local deposit.
    // Of each continuous energy loss, fraction f is re-deposited along +z with
    // MFP electronic_buildup_mfp_mm (delta-ray proxy). f scales with energy:
    //   f(E) = electronic_buildup_fraction * clamp(E_MeVu / 400, 0, 1)
    // so low-E beams are almost unchanged. 0 disables (legacy).
    double electronic_buildup_fraction{0.0};
    double electronic_buildup_mfp_mm{0.5};
    // Minibeam validation can restrict the unresolved delta-electron proxy to
    // primary ion. This avoids applying a carbon-derived energy fraction to
    // fragment species whose restricted/unrestricted stopping split differs.
    bool minibeam_electronic_buildup_primary_only{false};
    // Optional transverse Gaussian sigma [mm] for the electronic fraction in
    // the dense voxel scorer.  The deposited energy is moved stochastically,
    // preserving expectation and using one voxel atomic per aggregated deposit.
    // 0 keeps the legacy track-local voxel scoring.
    double electronic_buildup_lateral_sigma_mm{0.0};
    // Hard soft-cap on estimated SYCL device USM vs device global memory.
    // Default 0.50: queues are scaled down to fit; allocation still aborts if over.
    // Pair with a host watchdog (run scripts) that kills the process if live VRAM
    // exceeds this fraction — prevents Arc driver lockups / host freezes.
    double max_device_memory_fraction{0.50};
    // Primary kernel launch chunk (histories per GPU submit). 0 = auto:
    // CUDA defaults to a small chunk so WSL/Windows can reclaim the GPU between
    // launches (long single kernels freeze WSL). Intel GPU uses a larger default.
    std::size_t history_chunk_size{0};
    // When false, boundary snaps use sycl::nextafter (master). When true, a
    // fixed 1e-5 mm physical nudge is used to escape CT/voxel face thrash.
    // Minibeam 0.1 mm scorer cases should enable this explicitly.
    bool robust_boundary_nudge{false};
    std::uint64_t random_seed{20'260'714};
    std::filesystem::path primary_stopping_power_file{
        "data/stopping_power_water_geant4_11_3_2.csv"};
    // Optional E_delta/(Edep+E_delta) lookup on the stopping-power energy grid.
    // Used only by the HadronLET-compatible scorer; dose transport is unchanged.
    std::filesystem::path let_delta_electron_fraction_file{};
    // Accurate fragment transport: use isotope-specific G4 water dE/dx ratios and
    // delta-electron corrections. False preserves the faster C-12 effective-charge
    // approximation and all existing configurations.
    bool use_particle_specific_stopping_power{false};
    std::filesystem::path particle_stopping_power_file{
        "data/ion_stopping_power_water_geant4_11_3_2.csv"};
    // Optional isotope-specific stopping ratios in representative CT lung,
    // soft-tissue, and bone materials. Each table is normalized to its own
    // C-12 curve, then applied to the local Schneider C-12 stopping power.
    // Missing species fall back to the production water ratio.
    std::filesystem::path ct_lung_particle_stopping_power_file{};
    std::filesystem::path ct_soft_tissue_particle_stopping_power_file{};
    std::filesystem::path ct_bone_particle_stopping_power_file{};
    // MeV energy-deposition scorer outputs (absolute MeV → MeV/primary in writers).
    std::filesystem::path output_file{"out/cpu_depth_dose.csv"};
    std::filesystem::path fragment_species_output_file{};
    std::filesystem::path let_output_file{"out/gpu_letd_depth.csv"};
    std::filesystem::path fragment_species_let_output_file{};
    // Optional p/d/t/He-3/He-4/N/O/F depth LET diagnostics. Empty avoids the
    // extra per-step FP64 atomics in normal production runs.
    std::filesystem::path light_isotope_let_output_file{};
    // Optional selected-species birth-spectrum diagnostics (CSV prefix). Empty
    // disables all birth-spectrum buffers and atomics.
    // Writes: <path>_summary.csv, <path>_mevu.csv, <path>_depth.csv,
    // <path>_costheta.csv, <path>_parent_mevu.csv, <path>_parent_z.csv
    std::filesystem::path fragment_birth_spectrum_output_file{};
    // Prefix/header path for dense primary and all-hadron 3D LET_d MHD maps.
    // Requires both scorerLET and enable_voxel_scoring.
    std::filesystem::path let_voxel_mhd_output_file{};
    std::filesystem::path voxel_dose_output_file{"out/gpu_voxel_dose.csv"};
    std::filesystem::path charged_origin_voxel_output_file{
        "out/gpu_charged_origin_voxel_dose.csv"};
    std::filesystem::path neutral_origin_voxel_output_file{
        "out/gpu_neutral_origin_voxel_dose.csv"};
    // Dose scorers (total Gy over all histories). Empty path disables that dose
    // file; MeV scorers always write when their paths are set. Defaults write Gy
    // alongside MeV.
    std::filesystem::path dose_output_file{"out/cpu_depth_dose_Gy.csv"};
    std::filesystem::path fragment_species_dose_output_file{};
    std::filesystem::path voxel_dose_Gy_output_file{"out/gpu_voxel_dose_Gy.csv"};
    std::filesystem::path charged_origin_voxel_dose_Gy_output_file{
        "out/gpu_charged_origin_voxel_dose_Gy.csv"};
    // Optional dense CT dose-to-medium maps for primary ion and charged
    // secondary origin categories. The writer appends _<category>.mhd/raw.
    std::filesystem::path charged_origin_voxel_mhd_output_prefix{};
    // Dense MetaImage MHD/RAW (total Gy). Empty disables. Skips sparse CSV I/O
    // cost when voxel_dose_Gy_output_file is also empty.
    std::filesystem::path voxel_dose_mhd_output_file{};
    // Diagnostic-only primary track-length fluence map (/mm2). Empty disables
    // both the device buffer and output, so production transport is unchanged.
    std::filesystem::path primary_voxel_fluence_mhd_output_file{};
    std::string device{"serial"};

    [[nodiscard]] double initial_total_energy_MeV() const noexcept {
        return initial_energy_MeVu * static_cast<double>(primary_mass_number);
    }
    [[nodiscard]] bool is_schneider_ct_mode() const noexcept {
        return material_physics_mode == MaterialPhysicsMode::SchneiderCt;
    }
    [[nodiscard]] bool is_water_mode() const noexcept {
        return material_physics_mode == MaterialPhysicsMode::Water;
    }
    void resolve_material_physics_mode() noexcept {
        if (enable_ct_grid || !ct_grid_file.empty()) {
            material_physics_mode = MaterialPhysicsMode::SchneiderCt;
        } else {
            material_physics_mode = MaterialPhysicsMode::Water;
        }
        unified_water_nuclear_transport = is_water_mode();
    }
    [[nodiscard]] double resolved_primary_rest_mass_MeV() const noexcept {
        return primary_rest_mass_MeV > 0.0
                   ? primary_rest_mass_MeV
                   : static_cast<double>(primary_mass_number) * 931.49410242;
    }
    [[nodiscard]] PrimaryIonDefinition primary_ion() const {
        return make_primary_ion_definition(
            primary_atomic_number, primary_mass_number, primary_rest_mass_MeV);
    }

    [[nodiscard]] std::size_t number_of_bins() const;
    [[nodiscard]] std::size_t number_of_voxels() const;
    [[nodiscard]] double scorer_spacing_z_mm() const noexcept {
        return voxel_size_z_mm > 0.0 ? voxel_size_z_mm : depth_bin_width_mm;
    }
    [[nodiscard]] double effective_secondary_local_deposit_cutoff_MeV()
        const noexcept {
        const auto configured = secondary_local_deposit_cutoff_MeV > 0.0
                                    ? secondary_local_deposit_cutoff_MeV
                                    : energy_cutoff_MeV;
        return configured;
    }
    [[nodiscard]] bool uses_fixed_patient_coordinates() const noexcept {
        return enable_tps_coordinate_system || enable_tps_source;
    }
    [[nodiscard]] bool uses_legacy_clamped_straggling() const noexcept {
        return energy_straggling_model == "gaussian_clamped" ||
               energy_straggling_model == "legacy_calibrated";
    }
    [[nodiscard]] bool uses_moment_matched_straggling() const noexcept {
        return energy_straggling_model == "moment_matched";
    }
    [[nodiscard]] bool uses_vavilov_landau_straggling() const noexcept {
        return energy_straggling_model == "vavilov_landau";
    }
    [[nodiscard]] bool uses_fred_paper_nuclear() const noexcept {
        return nuclear_model == "fred_paper";
    }
    [[nodiscard]] bool uses_packaged_fluctuation() const noexcept {
        return energy_straggling_model == "packaged_fluctuation" ||
               energy_straggling_model == "packaged_fluctuation_fraction" ||
               energy_straggling_model == "packaged_fluctuation_fraction_hybrid";
    }
    [[nodiscard]] bool uses_fred_2gr_mcs() const noexcept {
        return multiple_scattering_model == "fred_2gr";
    }
    [[nodiscard]] bool uses_fred_2gr_high_energy_extrapolation() const noexcept {
        return fred_2gr_high_energy_mode == "kinematic_extrapolation";
    }
    [[nodiscard]] int straggling_sampler_id() const noexcept {
        if (uses_moment_matched_straggling()) {
            return 1;
        }
        if (uses_vavilov_landau_straggling()) {
            return 2;
        }
        return 0;
    }
    [[nodiscard]] bool is_primary_attenuation_only_mode() const noexcept {
        return ct_validation_mode == "primary-attenuation-only";
    }
    // Diagnostic-only allocation gate for per-depth primary survival /
    // inelastic-reaction buffers. Write-only diagnostics; no transport,
    // dose, or RNG effect.
    [[nodiscard]] bool needs_primary_survival_buffers() const noexcept {
        return validation_scorers() || is_primary_attenuation_only_mode() ||
               enable_fragment_species_scoring;
    }
    void validate() const;
};

TransportConfig load_config(const std::filesystem::path& path);
std::uint64_t parse_random_seed(std::string_view value);

}  // namespace carbon
