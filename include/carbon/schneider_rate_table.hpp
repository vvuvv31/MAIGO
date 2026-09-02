#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSchneiderNumSections = 25;
constexpr std::size_t kSchneiderNumTargets = 13;
constexpr std::size_t kSchneiderNumEnergies = 860;

constexpr std::array<int32_t, 13> kSchneiderCanonicalZ = {
    1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
};

#pragma pack(push, 1)
struct SchneiderRateHeader {
    char magic[8];             // "SCHNRATE"
    uint32_t version;          // 1
    uint32_t num_sections;     // 25
    uint32_t num_targets;      // 13
    uint32_t num_energies;     // 860
    double energy_min_mevu;    // 0.5
    double energy_max_mevu;    // 430.0
    double energy_step_mevu;   // 0.5
    int32_t target_z[13];      // {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22}
};
#pragma pack(pop)

class SchneiderRateTable {
public:
    SchneiderRateTable() = default;

    static SchneiderRateTable from_binary(
        const std::filesystem::path& binary_path,
        const std::filesystem::path& metadata_path = {});

    [[nodiscard]] double energy_min_mevu() const noexcept { return energy_min_mevu_; }
    [[nodiscard]] double energy_max_mevu() const noexcept { return energy_max_mevu_; }
    [[nodiscard]] double energy_step_mevu() const noexcept { return energy_step_mevu_; }

    [[nodiscard]] static std::size_t target_index_from_z(int target_z);

    [[nodiscard]] double mass_partial_rate(std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const;
    [[nodiscard]] double mass_total_rate(std::size_t section_id, std::size_t energy_idx) const;

    [[nodiscard]] double interpolate_mass_partial(std::size_t section_id, std::size_t target_idx, double energy_mevu) const;
    [[nodiscard]] double interpolate_mass_total(std::size_t section_id, double energy_mevu) const;

    [[nodiscard]] const std::vector<double>& mass_partial_rates() const noexcept { return mass_partial_rates_; }
    [[nodiscard]] const std::vector<double>& mass_total_rates() const noexcept { return mass_total_rates_; }

private:
    double energy_min_mevu_{0.5};
    double energy_max_mevu_{430.0};
    double energy_step_mevu_{0.5};
    // Flattened array: [section][target][energy] = section * (13 * 860) + target * 860 + energy
    std::vector<double> mass_partial_rates_;
    // Flattened array: [section][energy] = section * 860 + energy
    std::vector<double> mass_total_rates_;
};

}  // namespace carbon
