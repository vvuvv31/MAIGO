#pragma once

#include <filesystem>
#include <vector>

namespace carbon {

class CrossSectionTable {
public:
    CrossSectionTable() = default;
    CrossSectionTable(std::vector<double> energies_MeVu,
                      std::vector<double> macroscopic_cross_sections_per_mm);

    static CrossSectionTable from_csv(const std::filesystem::path& path);
    // Paper C-C fit + Kox(C,O)/Kox(C,C) + ICRU-H macroscopic water table.
    static CrossSectionTable from_fred_paper_water(double density_g_per_cm3 = 1.0);
    static std::vector<CrossSectionTable> from_schneider_csv(
        const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] double interpolate_target_h_fraction(double energy_MeVu) const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;
    [[nodiscard]] const std::vector<double>& target_h_fractions() const noexcept;
    [[nodiscard]] const std::vector<double>& macro_h_per_mm() const noexcept;
    [[nodiscard]] const std::vector<double>& macro_o_per_mm() const noexcept;
    void set_target_h_fractions(std::vector<double> fractions);
    void set_partial_macros(std::vector<double> macro_h, std::vector<double> macro_o);

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> macroscopic_cross_sections_per_mm_;
    std::vector<double> target_h_fractions_;
    std::vector<double> macro_h_per_mm_;
    std::vector<double> macro_o_per_mm_;
};

struct ResampledCrossSectionGrid {
    std::vector<float> macroscopic_per_mm;
    std::vector<float> target_h_fraction;
};

// Resample inelastic data onto the transport grid used by device kernels.
[[nodiscard]] ResampledCrossSectionGrid resample_cross_section_grid(
    const CrossSectionTable& cross_section,
    const std::vector<double>& transport_energies_MeVu);

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
