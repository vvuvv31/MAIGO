#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace carbon {

enum class LeptonSpecies {
    electron,
    positron,
};

class ElectronTransportTable {
public:
    ElectronTransportTable(
        std::vector<double> kinetic_energies_MeV,
        std::vector<double> electron_collisional_stopping_power_MeV_per_mm,
        std::vector<double> electron_radiative_stopping_power_MeV_per_mm,
        std::vector<double> electron_total_stopping_power_MeV_per_mm,
        std::vector<double> positron_collisional_stopping_power_MeV_per_mm,
        std::vector<double> positron_radiative_stopping_power_MeV_per_mm,
        std::vector<double> positron_total_stopping_power_MeV_per_mm);

    [[nodiscard]] static ElectronTransportTable from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] double collisional_stopping_power(
        LeptonSpecies species, double kinetic_energy_MeV) const noexcept;
    [[nodiscard]] double radiative_stopping_power(
        LeptonSpecies species, double kinetic_energy_MeV) const noexcept;
    [[nodiscard]] double total_stopping_power(
        LeptonSpecies species, double kinetic_energy_MeV) const noexcept;
    [[nodiscard]] double csda_range_mm(
        LeptonSpecies species, double kinetic_energy_MeV) const;

    [[nodiscard]] const std::vector<double>& kinetic_energies_MeV() const noexcept {
        return kinetic_energies_MeV_;
    }
    [[nodiscard]] const std::vector<double>& collisional_stopping_powers(
        LeptonSpecies species) const noexcept;
    [[nodiscard]] const std::vector<double>& radiative_stopping_powers(
        LeptonSpecies species) const noexcept;
    [[nodiscard]] const std::vector<double>& total_stopping_powers(
        LeptonSpecies species) const noexcept;
    [[nodiscard]] const std::vector<double>& cumulative_ranges_mm(
        LeptonSpecies species) const noexcept;

private:
    std::vector<double> kinetic_energies_MeV_;
    std::vector<double> electron_collisional_stopping_power_MeV_per_mm_;
    std::vector<double> electron_radiative_stopping_power_MeV_per_mm_;
    std::vector<double> electron_total_stopping_power_MeV_per_mm_;
    std::vector<double> electron_cumulative_range_mm_;
    std::vector<double> positron_collisional_stopping_power_MeV_per_mm_;
    std::vector<double> positron_radiative_stopping_power_MeV_per_mm_;
    std::vector<double> positron_total_stopping_power_MeV_per_mm_;
    std::vector<double> positron_cumulative_range_mm_;
};

}  // namespace carbon
