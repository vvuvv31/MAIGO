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
    // Lateral/3D insert (7b): AABB of alternate material inside water background.
    bool enable_hetero_insert{false};
    HeteroInsert hetero_insert{};
    std::filesystem::path insert_stopping_power_file{};
    std::filesystem::path insert_cross_section_file{};
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
    // Global multiplier on CT mass-scaled / material stopping power (default 1).
    // Used to absorb residual WEPL calibration vs full Geant4 material SP.
    double ct_stopping_power_scale{1.0};
    double scorer_area_mm2{90'000.0};
    bool enable_voxel_scoring{false};
    bool enable_charged_origin_voxel_scoring{false};
    std::size_t voxel_bins_x{60};
    std::size_t voxel_bins_y{60};
    double voxel_size_x_mm{5.0};
    double voxel_size_y_mm{5.0};
    bool enable_energy_straggling{false};
    double straggling_scale{1.0};
    bool enable_multiple_scattering{false};
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
    std::string spots_geometry_mode{"topas"};
    double spots_patient_trans_x_mm{0.0};
    double spots_patient_trans_y_mm{0.0};
    double spots_patient_trans_z_mm{0.0};
    double spots_patient_rot_z_deg{0.0};
    double spots_ct_axis_min_mm{0.0};
    bool enable_primary_attenuation{false};
    bool enable_secondary_generation{false};
    bool enable_secondary_transport{false};
    bool enable_fragment_cascade{false};
    // Per-species secondary depth-dose scoring. Disable for voxel-only full-plan
    // production to remove an otherwise redundant global atomic per deposit.
    bool enable_fragment_species_scoring{true};
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
    // Reorder each secondary batch into contiguous initial-energy-per-nucleon
    // buckets before transport to reduce warp divergence. Off by default until
    // benchmarked on the target GPU.
    bool enable_secondary_energy_sorting{false};
    std::uint64_t random_seed{20'260'714};
    std::filesystem::path stopping_power_file{"data/stopping_power_water.csv"};
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
