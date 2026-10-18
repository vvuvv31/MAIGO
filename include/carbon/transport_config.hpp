#pragma once

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
    // Fixed provenance schema; not a configurable YAML interface.
    std::uint32_t config_schema_version{1};
    // Stable, comment-free, key-sorted representation of the parsed input and
    // effective profile settings. This is intended for run provenance and
    // hashing; CLI overrides are not represented here.
    std::string canonical_config_text{};
    // The current experimental physics presets use research mode.
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
    std::size_t number_of_histories{10'000};
    // Single-beam energy; tps_spots_file
    // supply per-spot energy and therefore do not require this value in YAML.
    double initial_energy_MeVu{200.0};
    // Relative RMS beam energy spread (TOPAS BeamEnergySpread percent / 100).
    // 0.01 = 1% → sample E ~ N(E0, (0.01*E0)^2) at primary birth.
    double beam_energy_spread{0.0};
    // Source identity; kinetic energies are per nucleon. Non-minibeam unified
    // GPU transport selects and checks package records using this Z/A.
    int primary_atomic_number{6};
    int primary_mass_number{12};
    // Derived transport/scorer dimensions; never parsed as independent YAML keys.
    // Water uses the explicit voxel grid; CT uses the native patient header.
    double phantom_length_mm{0.0};
    double depth_bin_width_mm{0.0};
    // The sole water/Schneider CT energy-loss model.
    std::filesystem::path em_package_file{};
    double broadbeam_maximum_mean_loss_fraction{0.01};
    // Config loading overrides ion caps by mode: broadbeam 1mm, minibeam 0.5mm.
    double maximum_step_mm{1.0};
    double maximum_relative_energy_loss{0.005};
    std::uint32_t maximum_primary_steps{2000000U};
    double energy_cutoff_MeV{0.1};
    // Charged secondaries below this total kinetic energy are stopped and their
    // remaining energy is deposited locally. 0 uses energy_cutoff_MeV.
    double secondary_local_deposit_cutoff_MeV{0.0};
    // Uniform-water density and CT-outside fallback; CT voxels use their own
    // material density from CCTG.
    double water_density_g_per_cm3{1.0};
    // Patient CT voxel grid.
    bool enable_ct_grid{false};
    std::filesystem::path ct_grid_file{};
    // Schneider composition table used to construct native-water nuclear rates.
    std::filesystem::path ct_schneider_file{};
    // When true (default), skip CT voxel-face step clamps in homogeneous regions.
    // Set false for isolated performance A/B against full face clamping.
    bool ct_skip_homogeneous_face_clamp{true};
    // Independent nuclear cross sections for four-class CT diagnostics.
    // Schneider full physics uses the section-resolved nuclear bundle.
    // Top-level physics routing mode: Water vs SchneiderCt
    MaterialPhysicsMode material_physics_mode{MaterialPhysicsMode::Water};

    // Canonical key names for water and Schneider CT physics
    // Derived routing: native water material, shared CINEL03 framework. No legacy fallback.
    bool unified_water_nuclear_transport{true};
    std::filesystem::path unified_water_material_file{};
    std::filesystem::path ct_schneider_primary_cinel03_file{};
    std::filesystem::path ct_schneider_secondary_cinel03_file{};
    std::filesystem::path ct_schneider_primary_rate_file{};
    std::filesystem::path ct_schneider_secondary_rate_file{};
    // Independent all-projectile TOPAS elastic bank; never uses the C12 isotropic fallback.
    std::filesystem::path all_ion_elastic_file{};
    std::filesystem::path elastic_recoil_stopping_file{};
    // Current physics bundle manifest, required for native water/Schneider
    // nuclear transport with strict v3 rates.
    std::filesystem::path ct_schneider_physics_bundle_file{};
    // Out-of-scope projectile nuclear policy (Schneider CT only). The ONLY
    // allowed value is "em_only": registry-unknown projectiles keep charged
    // EM transport with secondary nuclear reactions disabled; no other value
    // and no default fallback exist (empty refuses startup).
    std::string secondary_out_of_scope_nuclear_policy{};
    double scorer_area_mm2{90'000.0};
    // Output-only absolute dose calibration. This scales Gy scorers while
    // preserving raw deposited-energy tallies and transport energy balance.
    // Keep at 1.0 unless an independent open-field calibration is available.
    double dose_output_scale{1.0};
    bool enable_voxel_scoring{false};

    // Dense scorer geometry in patient coordinates. X-Y is the axial plane;
    // Z is the inferior-superior slice direction. For CT runs these values are
    // read from the CCTG v3 header; explicit values are consistency assertions.
    std::size_t voxel_bins_x{60};
    std::size_t voxel_bins_y{60};
    std::size_t voxel_bins_z{0};
    double voxel_size_x_mm{5.0};
    double voxel_size_y_mm{5.0};
    double voxel_size_z_mm{0.0};
    bool enable_energy_straggling{true};

    // Charged secondaries use mean Unified EM losses by default.
    bool enable_secondary_energy_straggling{false};
    // GPU generation-wise species ordering. Production YAML opts in explicitly.
    bool secondary_species_grouping{false};
    // Compile-time geometry/model branch selection for uniform-water Urban
    // minibeam. Runtime eligibility is checked before the first kernel launch.
    bool minibeam_primary_specialization{false};
    // Pause after the compile-time secondary iteration limit (default 16) and compact survivors.
    // Production presets opt in; false retains full-track scheduling without state buffers.
    bool secondary_step_chunking{false};
    bool enable_inelastic{false};
    // geant4: load the configured macroscopic table as-is.
    std::string nuclear_model{"geant4"};
    // Number of secondary generations allowed to undergo current CINEL03 inelastic
    // replay. Zero restricts inelastic reactions to the primary projectile.
    // Validated production configurations are limited to at most two.
    std::uint32_t max_secondary_inelastic_generations{0};
    bool enable_secondary_transport{true};
    bool enable_multiple_scattering{false};
    // Ion MCS for water/CT: `fermi_eyges` (broadbeam; all charged ions with
    // species-water parameters) or global `urban_v2` (minibeam).
    // Explicit dispatch/physics preset. YAML loading defaults non-minibeam
    // configurations to enabled FE for all charged species; explicit choices win.
    std::string transport_profile{"reference"};
    // Implementation selection only; custom configs default to the shared generic pipeline.
    std::string transport_pipeline{"generic"};
    bool fermi_eyges_gaussian_only{false};
    std::string multiple_scattering_model{"urban_v2"};
    // Global Urban v2: exact material/cut couples, every transported charged ion.
    std::filesystem::path urban_mcs_package_file{};
    // Research-only same-composition density scaling; not a CT dose validation.
    bool urban_ct_density_scaling{false};
    // Physical water box outside the scored ROI (required in global water mode).
    double urban_water_half_width_mm{0.0};
    double fermi_eyges_max_segment_mm{0.1};
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
    // Cylindrical Copper collimator matching the TOPAS reference.
    // Copper transport uses minibeam_radius_mm for the outer body and
    // minibeam_slit_length_mm for the finite parallel slits.
    // Distances are specified in the treatment-room convention: the
    // collimator centre is upstream of isocentre and the water entrance is at
    // minibeam_water_entrance_world_y_mm downstream of isocentre.
    double minibeam_collimator_center_to_isocenter_mm{30.0};
    double minibeam_collimator_thickness_mm{60.0};
    double minibeam_slit_length_mm{50.0};
    double minibeam_slit_thickness_mm{60.0};
    // Radius is the production outer-body dimension.
    double minibeam_radius_mm{60.0};
    int minibeam_slit_count{15};
    double minibeam_slit_width_mm{0.5};
    double minibeam_slit_pitch_mm{3.6};
    // Translation of the complete slit array along its periodic local-u axis.
    // Zero preserves the historical centred array; a half-pitch offset is
    // required by some opposed-field patient apertures.
    double minibeam_slit_offset_mm{0.0};
    double minibeam_collimator_angle_deg{0.0};
    // Native unrestricted electronic response for all beamline ions: Cu + air.
    std::filesystem::path minibeam_em_package_file{};
    // Primary beamline air transport, independent of the water MCS selector.
    double minibeam_air_slit_max_step_mm{1.0e6};
    std::filesystem::path minibeam_air_urban_package_file{};
    double minibeam_copper_density_g_per_cm3{8.96};
    double minibeam_copper_radiation_length_g_per_cm2{12.8628};
    double minibeam_copper_max_step_mm{0.05};
    // Split condensed-history steps at the exact cylindrical/slit material
    // boundary before applying Copper or air physics.
    bool minibeam_copper_exact_material_boundaries{false};
    bool minibeam_copper_enable_mcs{true};
    // Extracted discrete General Ion Elastic interactions. Condensed Copper
    // electromagnetic MCS remains independent of this switch.
    bool minibeam_copper_enable_elastic{true};
    // Sample Bohr energy-loss fluctuations step-by-step in Copper.  This is
    // independent of downstream ion straggling; both default to enabled.
    bool minibeam_copper_enable_energy_straggling{true};
    // Species Urban/elastic fragment transport is independent of primary fluctuations.
    bool minibeam_copper_fragment_native_transport{false};
    // Separate Copper fragment control, enabled in all default minibeams.
    bool minibeam_copper_fragment_enable_energy_straggling{false};
    // Amplitude scale for the selected Copper MCS model. The legacy Highland
    // benchmark uses 0.785; the slab-constrained model uses 1.0.
    double minibeam_copper_mcs_scale{0.785};
    // Copper fragments use independently controlled native Urban or cumulative
    // Highland path.  Do not inherit the primary Fermi--Eyges normalization.
    double minibeam_copper_fragment_mcs_scale{0.785};
    // Number of fragment+Cu reactions whose process-faithful correlated
    // final states are replayed after the primary C12+Cu interaction. Zero
    // preserves the historical terminal-absorption approximation. Copper and
    // water generation counters are independent.
    std::uint32_t minibeam_copper_fragment_cascade_generations{0};
    bool minibeam_copper_enable_nuclear_attenuation{false};
    std::filesystem::path minibeam_copper_cross_section_file{};
    // Copper-only TOPAS banks, loaded only for minibeam runs.
    std::filesystem::path minibeam_copper_inclxx_file{};
    std::filesystem::path minibeam_copper_elastic_file{};
    std::filesystem::path minibeam_copper_fragment_elastic_file{};
    std::filesystem::path minibeam_copper_fragment_urban_package_file{};
    std::filesystem::path minibeam_copper_ion_cross_section_file{};
    // Used by spots_geometry_mode=minibeam_pbs_y to map TOPAS world Y to
    // canonical GPU depth: gpu_z = world_y - this value.
    double minibeam_water_entrance_world_y_mm{60.0};
    // Non-empty only for a flattened multi-spot SYCL plan. All entries share
    // the physics/scorer fields above; source and energy fields come from here.
    std::vector<PrimarySpotBatchEntry> primary_spot_batch{};
    double spots_patient_trans_x_mm{0.0};
    double spots_patient_trans_y_mm{0.0};
    double spots_patient_trans_z_mm{0.0};
    double spots_patient_rot_z_deg{0.0};
    double spots_ct_axis_min_mm{0.0};
    // Optional continuous C-12 energy loss through the TOPAS World air gap
    // before the reoriented CT entrance. This is intentionally separate from
    // CT in-grid air material handling. It applies only to TPS spot geometry
    // modes, is disabled by default, and currently omits air MCS/straggling.
    bool spots_enable_upstream_air_energy_loss{false};
    // Exact CT face handling is the current production behavior. Inert in water.
    bool ct_secondary_exact_faces{true};
    std::filesystem::path spots_upstream_air_stopping_power_file{};
    // Keep all clinical geometry in the fixed DICOM LPS patient frame. This is
    // independent of the spot input format (TOPAS TimeFeature or TPS CSV).
    bool enable_tps_coordinate_system{false};
    // TPS CSV source switch, configured with the formal tpsSource key.
    bool enable_tps_source{false};
    // CSV columns: energy_MeV (total ion KE), x_mm, y_mm, weight;
    // optional spot_id, energy_spread_percent and emittance parameters.
    // Empty file => central spot from initial_energy_MeVu and source defaults.
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
    // "iec61217" is the water-phantom convention (carbon water minibeam).
    std::string tps_angle_convention{"iec61217"};
    // Public YAML name: tps_beam_angle_deg. This is the in-plane beam angle
    // about patient +Z for dicom_lps.
    double tps_beam_angle_deg{0.0};
    double tps_couch_angle_deg{0.0};
    double tps_collimator_angle_deg{0.0};
    double tps_isocenter_x_mm{0.0};
    double tps_isocenter_y_mm{0.0};
    double tps_isocenter_z_mm{0.0};
    double tps_sad_mm{1000.0};
    // HFS, HFP, FFS, or FFP. Axes are converted to the simulation's patient
    // coordinate system; isocenter coordinates are already in that system.
    std::string tps_patient_position{"HFS"};
    double max_device_memory_fraction{0.50};
    // Explicit tracked-USM allocation budget in GiB. 0 keeps the historical
    // unbounded behavior; a positive value makes the DeviceMemoryTracker reject
    // an allocation that would exceed it (driver/runtime memory excluded).
    double device_memory_budget_gib{0.0};
    // Secondary particle queue capacity. Default matches the validated
    // production value; larger cards may raise it for big merged shards.
    std::size_t secondary_queue_capacity{32000000};
    // Primary kernel launch chunk (histories per GPU submit). 0 = auto:
    // CUDA defaults to a small chunk so WSL/Windows can reclaim the GPU between
    // launches (long single kernels freeze WSL).
    std::size_t history_chunk_size{0};
    std::uint64_t random_seed{20'260'714};
    // Reference energy grid for nuclear/scattering bookkeeping only;
    // water/CT primary stopping comes exclusively from em_package_file.
    std::filesystem::path primary_stopping_power_file{
        "data/stopping/water/carbon.csv"};
    // Dense MetaImage MHD/RAW (total Gy). Empty disables.
    std::filesystem::path voxel_dose_mhd_output_file{};
    // Optional unrestricted electronic LETd, using the stopping-power
    // track-length estimator sum(S_e^2 dl) / sum(S_e dl). No transport effect.
    std::filesystem::path letd_output_prefix{};
    // LET-only electronic recoil stopping; never replaces transport's total stopping.
    std::filesystem::path letd_recoil_electronic_stopping_file{};

    std::string device{"cuda"};

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

    void require_primary_reference_mass(double reference_mass_MeV, const char* package,
                                        PrimaryMassConvention convention = PrimaryMassConvention::Nuclear) const {
        const double expected = convention == PrimaryMassConvention::Kinematic
            ? primary_kinematic_mass_MeV(primary_atomic_number, primary_mass_number)
            : primary_nuclear_mass_MeV(primary_atomic_number, primary_mass_number);
        if (!std::isfinite(expected) || !std::isfinite(reference_mass_MeV) ||
            !(reference_mass_MeV > 0.0) || std::abs(reference_mass_MeV - expected) > 2.0e-6 * expected)
            throw std::invalid_argument(std::string(package) + " primary mass disagrees with its production mass convention");
    }
    [[nodiscard]] PrimaryIonDefinition primary_ion() const {
        return make_primary_ion_definition(primary_atomic_number, primary_mass_number);
    }

    [[nodiscard]] std::size_t number_of_bins() const;
    [[nodiscard]] std::size_t number_of_voxels() const;
    [[nodiscard]] double scorer_spacing_z_mm() const noexcept {
        return voxel_size_z_mm;
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

    void validate() const;
};

TransportConfig load_config(const std::filesystem::path& path);
std::uint64_t parse_random_seed(std::string_view value);

}  // namespace carbon
