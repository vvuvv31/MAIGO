#pragma once
#include "carbon/urban_mcs_package.hpp"

#include <filesystem>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace carbon {

class StoppingPowerTable;

class StoppingPowerTable {
public:
    StoppingPowerTable(std::vector<double> energies_MeVu,
                       std::vector<double> stopping_powers_MeV_per_mm);

    static StoppingPowerTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] double minimum_energy_MeVu() const noexcept;
    [[nodiscard]] double maximum_energy_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> stopping_powers_MeV_per_mm_;
};

// Species-indexed stopping-power calibration ratios on the reference energy grid.
class IonStoppingPowerTables {
public:
    static constexpr std::size_t mass_stride = 32;
    static constexpr std::size_t atomic_number_slots = 10;
    static constexpr std::size_t species_slots =
        mass_stride * atomic_number_slots;

    static IonStoppingPowerTables from_csv(
        const std::filesystem::path& path,
        const StoppingPowerTable& carbon_stopping_power,
        bool normalize_to_file_carbon = false);

    [[nodiscard]] const std::vector<float>& ratios_to_carbon() const noexcept;
    [[nodiscard]] const std::vector<float>& delta_electron_fractions() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& species_present() const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;

private:
    std::size_t energy_grid_size_{};
    std::vector<float> ratios_to_carbon_;
    std::vector<float> delta_electron_fractions_;
    std::vector<std::uint8_t> species_present_;
};

}  // namespace carbon
