#pragma once

#include <filesystem>
#include <cstdint>
#include <cmath>
#include <vector>

namespace carbon {

[[nodiscard]] double ion_effective_charge(int atomic_number, double energy_MeVu);
[[nodiscard]] double stopping_power_scale_from_reference_ion(
    int atomic_number, int reference_atomic_number, double energy_MeVu);

class StoppingPowerTable;

[[nodiscard]] std::vector<float> load_hu_stopping_power_lut(
    const std::filesystem::path& path,
    std::size_t n_sections,
    std::size_t table_size,
    float scale = 1.0F);

[[nodiscard]] std::vector<float> load_ion_species_stopping_power_lut(
    const std::filesystem::path& path,
    std::size_t table_size,
    float scale = 1.0F);

struct DensityMassSprLut {
    std::uint32_t n_rho{0};
    float log_rho_min{0.0F};
    float inv_dlog{0.0F};
    std::vector<float> factors{};
};

[[nodiscard]] DensityMassSprLut build_density_mass_spr_lut(
    const StoppingPowerTable& water,
    const StoppingPowerTable& air,
    const StoppingPowerTable& lung,
    const StoppingPowerTable& bone,
    float scale = 1.0F);

class StoppingPowerTable {
public:
    StoppingPowerTable(std::vector<double> energies_MeVu,
                       std::vector<double> stopping_powers_MeV_per_mm);

    static StoppingPowerTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    // CSDA range in mm from zero energy to energy_MeVu. The table abscissa is
    // kinetic energy per nucleon (MeV/u); values are total-ion stopping power
    // (MeV/mm), so mass_number is required for the A*dE_u/S(E_u) integral.
    [[nodiscard]] double csda_range_mm(double energy_MeVu, int mass_number) const;
    // Invert the CSDA range after a physical path length in mm. Energy is
    // returned in MeV/u and is clamped to zero when the path exhausts range.
    [[nodiscard]] double csda_energy_after_distance_MeVu(
        double initial_energy_MeVu, double distance_mm, int mass_number) const;
    [[nodiscard]] const std::vector<double>& cumulative_ranges_mm() const noexcept;
    [[nodiscard]] double minimum_energy_MeVu() const noexcept;
    [[nodiscard]] double maximum_energy_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> stopping_powers_MeV_per_mm_;
    std::vector<double> cumulative_ranges_mm_;
};

// Flat-array CSDA helpers for future SYCL kernels. Arrays use the same contract
// as StoppingPowerTable: energy is MeV/u, stopping power is total-ion MeV/mm,
// cumulative range is A=1 mm, and mass_number supplies the overall A factor.
inline float csda_range_mm_device(const float* energies_MeVu,
                                  const float* stopping_powers_MeV_per_mm,
                                  const float* cumulative_ranges_mm,
                                  const std::size_t count,
                                  const float energy_MeVu,
                                  const int mass_number) noexcept {
    if (count == 0 || mass_number <= 0 || energy_MeVu <= 0.0F) return 0.0F;
    const auto first_energy = energies_MeVu[0];
    float range = 0.0F;
    if (energy_MeVu <= first_energy) {
        range = energy_MeVu / stopping_powers_MeV_per_mm[0];
    } else if (energy_MeVu >= energies_MeVu[count - 1]) {
        range = cumulative_ranges_mm[count - 1] +
                (energy_MeVu - energies_MeVu[count - 1]) /
                    stopping_powers_MeV_per_mm[count - 1];
    } else {
        std::size_t low = 0;
        std::size_t high = count;
        while (low + 1 < high) {
            const auto middle = (low + high) / 2;
            if (energies_MeVu[middle] <= energy_MeVu) low = middle;
            else high = middle;
        }
        const auto e0 = energies_MeVu[low];
        const auto e1 = energies_MeVu[low + 1];
        const auto s0 = stopping_powers_MeV_per_mm[low];
        const auto slope = (stopping_powers_MeV_per_mm[low + 1] - s0) / (e1 - e0);
        const auto delta_e = energy_MeVu - e0;
        const auto nearly_constant = std::abs(slope) <
            1.0e-7F * std::max(1.0F, std::max(std::abs(s0),
                                                std::abs(stopping_powers_MeV_per_mm[low + 1])));
        const auto integral = nearly_constant
            ? delta_e / s0
            : std::log((s0 + slope * delta_e) / s0) / slope;
        range = cumulative_ranges_mm[low] + integral;
    }
    return static_cast<float>(mass_number) * range;
}

inline float csda_energy_after_distance_device(
    const float* energies_MeVu,
    const float* stopping_powers_MeV_per_mm,
    const float* cumulative_ranges_mm,
    const std::size_t count,
    const float initial_energy_MeVu,
    const float distance_mm,
    const int mass_number) noexcept {
    if (count == 0 || mass_number <= 0 || initial_energy_MeVu <= 0.0F || distance_mm <= 0.0F) {
        return initial_energy_MeVu > 0.0F ? initial_energy_MeVu : 0.0F;
    }
    const auto initial_range = csda_range_mm_device(
        energies_MeVu, stopping_powers_MeV_per_mm, cumulative_ranges_mm,
        count, initial_energy_MeVu, mass_number);
    if (distance_mm >= initial_range) return 0.0F;
    const auto target = (initial_range - distance_mm) / static_cast<float>(mass_number);
    if (target <= cumulative_ranges_mm[0]) {
        return target * stopping_powers_MeV_per_mm[0];
    }
    if (target >= cumulative_ranges_mm[count - 1]) {
        return energies_MeVu[count - 1] +
               (target - cumulative_ranges_mm[count - 1]) *
                   stopping_powers_MeV_per_mm[count - 1];
    }
    std::size_t low = 0;
    std::size_t high = count;
    while (low + 1 < high) {
        const auto middle = (low + high) / 2;
        if (cumulative_ranges_mm[middle] <= target) low = middle;
        else high = middle;
    }
    const auto e0 = energies_MeVu[low];
    const auto e1 = energies_MeVu[low + 1];
    const auto s0 = stopping_powers_MeV_per_mm[low];
    const auto slope = (stopping_powers_MeV_per_mm[low + 1] - s0) / (e1 - e0);
    const auto integral = target - cumulative_ranges_mm[low];
    const auto nearly_constant = std::abs(slope) <
        1.0e-7F * std::max(1.0F, std::max(std::abs(s0),
                                            std::abs(stopping_powers_MeV_per_mm[low + 1])));
    const auto delta_e = nearly_constant
        ? s0 * integral
        : s0 * std::expm1(slope * integral) / slope;
    return e0 + std::min(std::max(delta_e, 0.0F), e1 - e0);
}

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
