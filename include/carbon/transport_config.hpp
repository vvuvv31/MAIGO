#pragma once

#include "carbon/slab_phantom.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

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
    // TOPAS-style BiGaussian emittance source on the entrance plane (z=0).
    // Samples (x,x') and (y,y') from bivariate Gaussians; x' = dx/dz (rad-like).
    bool enable_emittance_source{false};
    double emittance_sigma_x_mm{0.0};
    double emittance_sigma_y_mm{0.0};
    double emittance_sigma_x_prime{0.0};
    double emittance_sigma_y_prime{0.0};
    double emittance_correlation_x{0.0};
    double emittance_correlation_y{0.0};
    bool enable_primary_attenuation{false};
    bool enable_secondary_generation{false};
    bool enable_secondary_transport{false};
    bool enable_fragment_cascade{false};
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
    std::filesystem::path output_file{"out/cpu_depth_dose.csv"};
    std::filesystem::path fragment_species_output_file{
        "out/gpu_fragment_species_depth_dose.csv"};
    std::filesystem::path voxel_dose_output_file{"out/gpu_voxel_dose.csv"};
    std::filesystem::path charged_origin_voxel_output_file{
        "out/gpu_charged_origin_voxel_dose.csv"};
    std::filesystem::path neutral_origin_voxel_output_file{
        "out/gpu_neutral_origin_voxel_dose.csv"};
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
