#pragma once

#include <filesystem>
#include <cstdint>
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
    [[nodiscard]] double minimum_energy_MeVu() const noexcept;
    [[nodiscard]] double maximum_energy_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> stopping_powers_MeV_per_mm_;
};

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
