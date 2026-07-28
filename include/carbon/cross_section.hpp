#pragma once

#include <filesystem>
#include <vector>

namespace carbon {

class CrossSectionTable {
public:
    CrossSectionTable(std::vector<double> energies_MeVu,
                      std::vector<double> macroscopic_cross_sections_per_mm);

    static CrossSectionTable from_csv(const std::filesystem::path& path);
    static std::vector<CrossSectionTable> from_schneider_csv(
        const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> macroscopic_cross_sections_per_mm_;
};

class IonCrossSectionTables {
public:
    static constexpr std::size_t mass_stride = 32;
    static constexpr std::size_t atomic_number_slots = 10;
    static constexpr std::size_t species_slots =
        mass_stride * atomic_number_slots;

    static IonCrossSectionTables from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& values() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& species_present()
        const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;
    [[nodiscard]] float minimum_energy_MeVu() const noexcept;
    [[nodiscard]] float energy_step_MeVu() const noexcept;

private:
    std::size_t energy_grid_size_{};
    float minimum_energy_MeVu_{};
    float energy_step_MeVu_{};
    std::vector<float> values_;
    std::vector<std::uint8_t> species_present_;
};

class NeutralCrossSectionTables {
public:
    static constexpr std::size_t species_count = 2;

    static NeutralCrossSectionTables from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& values() const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;
    [[nodiscard]] float minimum_log_energy() const noexcept;
    [[nodiscard]] float log_energy_step() const noexcept;

private:
    std::size_t energy_grid_size_{};
    float minimum_log_energy_{};
    float log_energy_step_{};
    // Gamma (PDG 22), then neutron (PDG 2112).
    std::vector<float> values_;
};

}  // namespace carbon
