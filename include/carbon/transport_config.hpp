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
    std::uint64_t random_seed{20'260'714};
    std::filesystem::path stopping_power_file{"data/stopping_power_water.csv"};
    std::filesystem::path output_file{"out/cpu_depth_dose.csv"};
    std::string device{"serial"};

    [[nodiscard]] double initial_total_energy_MeV() const noexcept {
        return initial_energy_MeVu * static_cast<double>(mass_number);
    }

    [[nodiscard]] std::size_t number_of_bins() const;
    void validate() const;
};

TransportConfig load_config(const std::filesystem::path& path);

}  // namespace carbon

