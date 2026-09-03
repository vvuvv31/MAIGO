#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSecondaryNumProjectiles = 13;
constexpr std::size_t kSecondaryNumSections = 25;
constexpr std::size_t kSecondaryNumTargets = 13;
constexpr std::size_t kSecondaryNumEnergies = 860;

#pragma pack(push, 1)
struct SecondaryRateHeader {
    char magic[8];             // "SCHN2RAT"
    uint32_t version;          // 1
    uint32_t num_projectiles;  // 13
    uint32_t num_sections;     // 25
    uint32_t num_targets;      // 13
    uint32_t num_energies;     // 860
    double energy_min_mevu;    // 0.5
    double energy_max_mevu;    // 430.0
    double energy_step_mevu;   // 0.5
};

struct SecondaryProjectileKey {
    int32_t z;
    int32_t a;
};
#pragma pack(pop)

class SecondaryRateTable {
public:
    SecondaryRateTable() = default;

    static SecondaryRateTable from_binary(
        const std::filesystem::path& binary_path,
        const std::filesystem::path& metadata_path = {});

    [[nodiscard]] double energy_min_mevu() const noexcept { return energy_min_mevu_; }
    [[nodiscard]] double energy_max_mevu() const noexcept { return energy_max_mevu_; }
    [[nodiscard]] double energy_step_mevu() const noexcept { return energy_step_mevu_; }
    [[nodiscard]] std::size_t num_projectiles() const noexcept { return num_projectiles_; }

    [[nodiscard]] int projectile_index(int proj_z, int proj_a) const noexcept;
    [[nodiscard]] static std::size_t target_index_from_z(int target_z);

    [[nodiscard]] double mass_partial_rate(std::size_t proj_idx, std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const;
    [[nodiscard]] double mass_total_rate(std::size_t proj_idx, std::size_t section_id, std::size_t energy_idx) const;

    [[nodiscard]] double interpolate_mass_total(std::size_t proj_idx, std::size_t section_id, double energy_mevu) const;

    [[nodiscard]] const std::vector<SecondaryProjectileKey>& projectiles() const noexcept { return projectiles_; }
    [[nodiscard]] const std::vector<double>& mass_partial_rates() const noexcept { return mass_partial_rates_; }
    [[nodiscard]] const std::vector<double>& mass_total_rates() const noexcept { return mass_total_rates_; }

private:
    uint32_t num_projectiles_{13};
    double energy_min_mevu_{0.5};
    double energy_max_mevu_{430.0};
    double energy_step_mevu_{0.5};
    std::vector<SecondaryProjectileKey> projectiles_;
    std::array<int32_t, 13> canonical_targets_{};
    std::vector<double> mass_partial_rates_;
    std::vector<double> mass_total_rates_;
};

}  // namespace carbon
