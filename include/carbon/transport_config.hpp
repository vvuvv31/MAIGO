#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace carbon {

struct TransportConfig {
    std::size_t number_of_histories{10'000};
    double initial_energy_MeVu{200.0};
    int mass_number{12};
    double phantom_length_mm{400.0};
    double depth_bin_width_mm{0.5};
    double maximum_step_mm{0.5};
    double maximum_relative_energy_loss{0.005};
    double energy_cutoff_MeV{0.1};
    double water_density_g_per_cm3{1.0};
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
    bool enable_primary_attenuation{false};
    bool enable_secondary_generation{false};
    bool enable_secondary_transport{false};
    bool enable_fragment_cascade{false};
    bool enable_neutral_transport{false};
    // "first_interaction": free path + one package (mode D); continuation residual.
    // "full": re-queue neutral continuations up to maximum_neutral_generations.
    std::string neutral_transport_mode{"first_interaction"};
    std::uint32_t maximum_cascade_generations{0};
    std::uint32_t maximum_neutral_generations{1};
    std::size_t secondary_queue_capacity{0};
    std::size_t neutral_queue_capacity{0};
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
