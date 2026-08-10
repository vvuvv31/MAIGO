#pragma once

#include "carbon/slab_phantom.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

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
    // Explicit transport accuracy policy. Public conventional-CT profiles are
    // "fast" and "best". "accurate" is retained only as the internal default
    // for legacy/minibeam configurations that omit a public profile.
    std::string physics_profile{"accurate"};
    std::size_t number_of_histories{10'000};
    double initial_energy_MeVu{200.0};
    // Relative RMS beam energy spread (TOPAS BeamEnergySpread percent / 100).
    // 0.01 = 1% → sample E ~ N(E0, (0.01*E0)^2) at primary birth.
    double beam_energy_spread{0.0};
    int mass_number{12};
    double phantom_length_mm{400.0};
    double depth_bin_width_mm{0.5};
    double maximum_step_mm{0.5};
    double maximum_relative_energy_loss{0.005};
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
    // Optional energy-dependent mass XS for every Schneider section. When set
    // on a CCTG v2/v3 grid, this supersedes the legacy four-class XS tables.
    std::filesystem::path ct_schneider_cross_section_file{};
    // Optional material-conditioned primary C-12 correlated final states.
    // Empty paths preserve the production water package for every CT voxel.
    // Each package is selected only for reactions occurring in the
    // corresponding CT material class. An empty class path falls back to
    // reaction_package_file.
    std::filesystem::path ct_lung_reaction_package_file{};
    std::filesystem::path ct_soft_tissue_reaction_package_file{};
    std::filesystem::path ct_bone_reaction_package_file{};
    // Optional fragment-specific inelastic cross sections extracted from
    // material cascade packages. Only their projectile XS tables are used
    // unless the matching ct_*_cascade_package_file is also set (full
    // material-conditioned final states). Values in the package are
    // macroscopic at the reference density below and are converted to mass
    // cross sections before CT-density scaling.
    std::filesystem::path ct_lung_cascade_cross_section_package_file{};
    std::filesystem::path ct_bone_cascade_cross_section_package_file{};
    double ct_lung_cascade_reference_density_g_per_cm3{1.04};
    double ct_bone_cascade_reference_density_g_per_cm3{1.85};
    // Optional material-conditioned cascade correlated final states.
    // Empty preserves cascade_package_file for every CT voxel. When set,
    // cascade interactions that occur in that CT material class sample
    // final states from the material package (with fallback to the default
    // cascade package if the projectile is missing there). Soft-tissue is
    // the dominant residual class in head CT cases.
    std::filesystem::path ct_lung_cascade_package_file{};
    std::filesystem::path ct_soft_tissue_cascade_package_file{};
    std::filesystem::path ct_bone_cascade_package_file{};
    // Optional short-range redistribution of nuclear residual heat that is
    // otherwise deposited at the interaction point (Q-value / missing product
    // KE). 0 preserves the historical point deposit. A positive value is the
    // exponential MFP along the projectile direction used as a residual-nucleus
    // transport proxy. This does not invent new particles.
    double nuclear_residual_heat_mfp_mm{0.0};
    // Multiplier on nuclear residual heat scored into the dose field after a
    // reaction (parent KE not accounted in sampled products). 1.0 is historical
    // full local residual. Values in (0, 1) reduce entrance-local nuclear heat
    // without changing package kinematics; 0 disables residual heat scoring.
    // Does not apply to continuous ionization loss or queued secondaries.
    double nuclear_residual_heat_scale{1.0};
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
    bool voxel_scorer_clamps_transport{true};
    std::size_t voxel_bins_x{60};
    std::size_t voxel_bins_y{60};
    double voxel_size_x_mm{5.0};
    double voxel_size_y_mm{5.0};
    bool enable_energy_straggling{false};
    // Historical validation applied straggling only to primary C-12.
    // Enable this separately to apply Bohr straggling to charged fragments.
    bool enable_secondary_energy_straggling{false};
    double straggling_scale{1.0};
    bool enable_multiple_scattering{false};
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
    // Residual accumulated Copper-loss correction for primary C-12 ions that
    // survive to the collimator exit. It is frozen from an independent
    // water-entrance phase-space comparison and deliberately does not alter
    // nuclear interaction probability, survival, or angular transport.
    double minibeam_copper_survivor_energy_loss_scale{1.0};
    // Optional piecewise-linear incident-energy calibration of the same
    // accumulated-loss scale. Empty vectors preserve the scalar behavior.
    // The values are constrained by Copper-touched primary C-12 phase space,
    // independently of the downstream dose comparison.
    std::vector<double>
        minibeam_copper_survivor_energy_loss_energies_MeVu{};
    std::vector<double>
        minibeam_copper_survivor_energy_loss_scales{};
    // Residual primary-C12 range correction in the downstream water phantom.
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
    // Optional entrance response for primary C-12 histories that touched the
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
    bool minibeam_copper_enable_reaction_products{false};
    std::filesystem::path minibeam_copper_reaction_package_file{};
    std::filesystem::path minibeam_copper_ion_stopping_power_file{};
    std::filesystem::path minibeam_copper_ion_cross_section_file{};
    // Total gamma/neutron macroscopic cross sections used to reject Copper
    // products that interact before reaching a slit or the collimator exit.
    std::filesystem::path minibeam_copper_neutral_cross_section_file{};
    // Conditional gamma/neutron final states in Copper. These preserve the
    // leading neutral after elastic, Compton, and Rayleigh interactions instead
    // of treating every total-cross-section collision as absorption.
    std::filesystem::path minibeam_copper_neutral_package_file{};
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
    // Diagnostic/legacy: after tps_90 maps the source to the CT entrance plane
    // (GPU x=patient Y, GPU y=patient Z), apply
    //   origin_y += skew * (origin_x - pivot)
    // i.e. patient_Z += skew * (patient_Y - pivot). The neutral default is the
    // production setting; nonzero values remain for explicit diagnostics and
    // backward-compatible reproduction of historical affine A/B runs.
    double spots_lateral_yz_skew{0.0};
    double spots_lateral_yz_skew_pivot_mm{0.0};
    bool spots_lateral_yz_skew_auto_pivot{false};
    // Diagnostic/legacy rigid rotation of the entrance-plane spot map about
    // the same lateral pivot (GPU x=patient Y, GPU y=patient Z):
    //   [dx']   [ cosθ  -sinθ ] [dx]
    //   [dy'] = [ sinθ   cosθ ] [dy]
    // with θ = spots_lateral_yz_rotation_deg (counter-clockwise in GPU xy).
    // The neutral default is the production setting. Nonzero values reproduce
    // historical affine diagnostics; beam-basis lateral components rotate too.
    double spots_lateral_yz_rotation_deg{0.0};
    // Rotation pivot GPU-y (patient Z). Auto-pivot fills this with the
    // history-weighted mean entrance GPU-y when auto_pivot is true.
    double spots_lateral_yz_rotation_pivot_y_mm{0.0};
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
    std::filesystem::path spots_upstream_air_stopping_power_file{};
    // Optional clinical TPS source. Disabled by default so all existing
    // TOPAS/CT examples retain their exact source construction and RNG path.
    // YAML accepts both `tpsSource` (public switch) and `tps_source`.
    bool enable_tps_source{false};
    // Optional CSV columns: spot_id,energy_MeVu,x_mm,y_mm,mu_weight plus
    // energy_spread_percent and emittance parameters. Empty => one central
    // spot using initial_energy_MeVu and the source defaults above.
    std::filesystem::path tps_spots_file{};
    // "iec61217": historical MAIGO convention, local w=-Z and gantry about +Y.
    // "topas_patient_rot_z": matches the CT-plan convention used in this repo:
    // TPS 0 deg is patient +Y and angle theta gives
    // w=(-sin(theta), cos(theta), 0), equivalent to passive Patient/RotZ.
    std::string tps_angle_convention{"iec61217"};
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
    // The current physics tables model carbon ions. Keep this explicit so a
    // proton TPS plan cannot silently run with carbon physics.
    std::string tps_particle_type{"carbon"};
    bool enable_primary_attenuation{false};
    bool enable_secondary_generation{false};
    bool enable_secondary_transport{false};
    bool enable_fragment_cascade{false};
    // Optional v3 package mode. Absolute reference depth is diagnostic and can
    // overfit a source energy; keep disabled unless cross-case validation wins.
    bool cascade_condition_on_reference_depth{false};
    // Diagnostic / residual-closure scales for cascade macroscopic XS sampled
    // from cascade_package_file. Applied after density / material mass-XS
    // factors. 1.0 preserves package rates. Light ions are Z<=2 (p/d/t/He).
    // Secondary carbons are Z==6 fragments continuing a cascade (not primary).
    // These are not patient-specific fits; use only with equal-history A/B.
    double cascade_light_ion_xs_scale{1.0};
    double cascade_secondary_carbon_xs_scale{1.0};
    // Per-species secondary depth-dose scoring. Disable for voxel-only full-plan
    // production to remove an otherwise redundant global atomic per deposit.
    bool enable_fragment_species_scoring{true};
    // Dose-averaged electronic LET scorer compatible with the
    // Villadslj/Topas-Extension myHadronLET definition. YAML also accepts the
    // requested camel-case alias `scorerLET`.
    bool enable_let_scoring{false};
    bool enable_neutral_transport{false};
    // "first_interaction": free path + one package (mode D); continuation residual.
    // "full": re-queue neutral continuations up to maximum_neutral_generations.
    std::string neutral_transport_mode{"first_interaction"};
    // When neutral transport is off, deposit this fraction of born neutron/gamma
    // kinetic energy as an interim kerma (calibrated ~0.298 at ≤200 MeV/u).
    // Not a global dose scale of the charged IDD.
    double neutral_local_kerma_fraction{0.0};
    // Multiplier of kerma fraction at E>=400 MeV/u (linear ramp from 200→400).
    // f(E) = f0 for E<=200; f0 * scale at E>=400. High-E needs more kerma to
    // close the plateau/integral deficit vs TOPAS.
    double neutral_kerma_high_energy_scale{1.0};
    // Exponential mean free path [mm] for distributing that kerma along +z from
    // the birth depth. 0 = legacy local dump in the production bin (overpredicts
    // entrance). Multi-energy charged suite uses ~80 mm to match TOPAS build-up.
    // Kerma is renormalized into the remaining phantom so high-E tails are not lost.
    double neutral_kerma_mean_free_path_mm{0.0};
    // Short-range electronic build-up vs unrestricted CSDA local deposit.
    // Of each continuous energy loss, fraction f is re-deposited along +z with
    // MFP electronic_buildup_mfp_mm (delta-ray proxy). f scales with energy:
    //   f(E) = electronic_buildup_fraction * clamp(E_MeVu / 400, 0, 1)
    // so low-E beams are almost unchanged. 0 disables (legacy).
    double electronic_buildup_fraction{0.0};
    double electronic_buildup_mfp_mm{0.5};
    // Minibeam validation can restrict the unresolved delta-electron proxy to
    // primary C-12. This avoids applying a carbon-derived energy fraction to
    // fragment species whose restricted/unrestricted stopping split differs.
    bool minibeam_electronic_buildup_primary_only{false};
    // Optional transverse Gaussian sigma [mm] for the electronic fraction in
    // the dense voxel scorer.  The deposited energy is moved stochastically,
    // preserving expectation and using one voxel atomic per aggregated deposit.
    // 0 keeps the legacy track-local voxel scoring.
    double electronic_buildup_lateral_sigma_mm{0.0};
    std::uint32_t maximum_cascade_generations{0};
    std::uint32_t maximum_neutral_generations{1};
    std::size_t secondary_queue_capacity{0};
    std::size_t neutral_queue_capacity{0};
    // Hard soft-cap on estimated SYCL device USM vs device global memory.
    // Default 0.50: queues are scaled down to fit; allocation still aborts if over.
    // Pair with a host watchdog (run scripts) that kills the process if live VRAM
    // exceeds this fraction — prevents Arc driver lockups / host freezes.
    double max_device_memory_fraction{0.50};
    // Primary kernel launch chunk (histories per GPU submit). 0 = auto:
    // CUDA defaults to a small chunk so WSL/Windows can reclaim the GPU between
    // launches (long single kernels freeze WSL). Intel GPU uses a larger default.
    std::size_t history_chunk_size{0};
    // Secondary/cascade particles processed per GPU submit. 0 = auto (CUDA small).
    std::size_t secondary_batch_size{0};
    // Persistent secondary workers per GPU submit. 0 preserves the legacy
    // one-work-item-per-track launch. A smaller nonzero worker pool repeatedly
    // fetches tracks from the batch, reducing whole-track warp tail divergence
    // without changing per-particle Philox streams.
    std::size_t secondary_persistent_workers{0};
    // FP32 continuous-loss residual for charged secondaries. When false, energy
    // is subtracted exactly as on master (legacy bitwise path). When true,
    // unrepresented FP32 ULPs are carried so scored energy is not double-counted
    // in the escaped-energy balance. Minibeam production configs enable this.
    bool secondary_fp32_energy_residual{false};
    // When false, boundary snaps use sycl::nextafter (master). When true, a
    // fixed 1e-5 mm physical nudge is used to escape CT/voxel face thrash.
    // Minibeam 0.1 mm scorer cases should enable this explicitly.
    bool robust_boundary_nudge{false};
    // Reorder each secondary batch into contiguous initial-energy-per-nucleon
    // buckets before transport to reduce warp divergence. Off by default until
    // benchmarked on the target GPU.
    bool enable_secondary_energy_sorting{false};
    std::uint64_t random_seed{20'260'714};
    std::filesystem::path stopping_power_file{"data/stopping_power_water.csv"};
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
    std::filesystem::path nuclear_cross_section_file{
        "data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv"};
    std::filesystem::path reaction_package_file{
        "validation/results/topas_200MeVu_reaction_packages_development.bin"};
    std::filesystem::path cascade_package_file{
        "validation/results/topas_200MeVu_cascade_100k.bin"};
    std::filesystem::path neutral_package_file{
        "validation/results/topas_200MeVu_neutral_smoke.bin"};
    // MeV energy-deposition scorer outputs (absolute MeV → MeV/primary in writers).
    std::filesystem::path output_file{"out/cpu_depth_dose.csv"};
    std::filesystem::path fragment_species_output_file{
        "out/gpu_fragment_species_depth_dose.csv"};
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
    // Prefix/header path for dense primary-C12 and all-hadron 3D LET_d MHD maps.
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
    std::filesystem::path fragment_species_dose_output_file{
        "out/gpu_fragment_species_depth_dose_Gy.csv"};
    std::filesystem::path voxel_dose_Gy_output_file{"out/gpu_voxel_dose_Gy.csv"};
    std::filesystem::path charged_origin_voxel_dose_Gy_output_file{
        "out/gpu_charged_origin_voxel_dose_Gy.csv"};
    // Dense MetaImage MHD/RAW (total Gy). Empty disables. Skips sparse CSV I/O
    // cost when voxel_dose_Gy_output_file is also empty.
    std::filesystem::path voxel_dose_mhd_output_file{};
    std::string device{"serial"};

    [[nodiscard]] double initial_total_energy_MeV() const noexcept {
        return initial_energy_MeVu * static_cast<double>(mass_number);
    }

    [[nodiscard]] std::size_t number_of_bins() const;
    [[nodiscard]] std::size_t number_of_voxels() const;
    void validate() const;
};

TransportConfig load_config(const std::filesystem::path& path);

}  // namespace carbon
