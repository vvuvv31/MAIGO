#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSchneiderStoppingNumSections = 25;
constexpr std::size_t kSchneiderStoppingNumEnergies = 4302;
constexpr double kSchneiderStoppingEnergyMin = 0.01;
constexpr double kSchneiderStoppingEnergyMax = 430.11;
constexpr double kSchneiderStoppingEnergyStep = 0.1;

#pragma pack(push, 1)
struct SchneiderStoppingHeader {
    char magic[8];             // "SCHNSTOP"
    uint32_t version;          // 1
    uint32_t num_sections;     // 25
    uint32_t num_energies;     // 4302
    double energy_min_mevu;    // 0.01
    double energy_max_mevu;    // 430.11
    double energy_step_mevu;   // 0.1
};
#pragma pack(pop)

class SchneiderStoppingTable {
public:
    SchneiderStoppingTable() = default;

    static SchneiderStoppingTable from_binary(
        const std::filesystem::path& binary_path,
        const std::filesystem::path& metadata_path = {});

    static SchneiderStoppingTable from_csv(
        const std::filesystem::path& csv_path);

    [[nodiscard]] double energy_min_mevu() const noexcept { return energy_min_mevu_; }
    [[nodiscard]] double energy_max_mevu() const noexcept { return energy_max_mevu_; }
    [[nodiscard]] double energy_step_mevu() const noexcept { return energy_step_mevu_; }
    [[nodiscard]] std::size_t num_sections() const noexcept { return kSchneiderStoppingNumSections; }
    [[nodiscard]] std::size_t num_energies() const noexcept { return kSchneiderStoppingNumEnergies; }

    [[nodiscard]] double density(std::size_t section_id) const;
    [[nodiscard]] double mass_stopping_power(std::size_t section_id, std::size_t energy_idx) const;
    [[nodiscard]] double csda_range_mm(std::size_t section_id, std::size_t energy_idx) const;

    [[nodiscard]] double interpolate_mass_stopping(std::size_t section_id, double energy_mevu) const;

    [[nodiscard]] const std::vector<double>& densities() const noexcept { return densities_; }
    [[nodiscard]] const std::vector<double>& mass_stopping_powers() const noexcept { return mass_stopping_powers_; }
    [[nodiscard]] const std::vector<double>& csda_ranges_mm() const noexcept { return csda_ranges_mm_; }

    [[nodiscard]] std::vector<float> to_flat_mass_stopping_float() const;

private:
    double energy_min_mevu_{kSchneiderStoppingEnergyMin};
    double energy_max_mevu_{kSchneiderStoppingEnergyMax};
    double energy_step_mevu_{kSchneiderStoppingEnergyStep};

    std::vector<double> densities_;
    // Flattened array: [section][energy] = section * 4302 + energy
    std::vector<double> mass_stopping_powers_;
    // Flattened array: [section][energy] = section * 4302 + energy
    std::vector<double> csda_ranges_mm_;
};

}  // namespace carbon
