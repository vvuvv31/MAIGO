#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSchneiderNumSections = 25;
constexpr std::size_t kSchneiderNumTargets = 13;
constexpr std::size_t kSchneiderNumEnergies = 921;

constexpr std::array<int32_t, 13> kSchneiderCanonicalZ = {
    1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
};

#pragma pack(push, 1)
struct SchneiderRateHeader {
    char magic[8];             // "SCHNRATE"
    uint32_t version;          // 3
    uint32_t num_sections;     // 25
    uint32_t num_targets;      // 13
    uint32_t num_energies;     // 921
    double energy_min_mevu;    // 0.1
    double energy_max_mevu;    // 460.1
    double energy_step_mevu;   // 0.5
    int32_t target_z[13];      // {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22}
};

// v3 per-target valid-domain entry appended after the totals payload
// (C12 primary has no projectile axis: 13 entries). Same mask-before-
// interpolation contract as the secondary v3 table.
struct SchneiderRateDomainEntry {
    double energy_min_mevu;
    double energy_max_mevu;
    std::uint8_t has_support;
    std::uint8_t reserved[7];
};
#pragma pack(pop)
static_assert(sizeof(SchneiderRateDomainEntry) == 24,
              "SchneiderRateDomainEntry must be 24 bytes");

// Reads only magic + version from a SCHNRATE binary (no full load).
// Throws on missing file, bad magic, or short header.
std::uint32_t schneider_rate_binary_version(const std::filesystem::path& binary_path);

class SchneiderRateTable {
public:
    SchneiderRateTable() = default;

    // Current SCHNRATE v3 rates; grids and channel domains come from the binary.
    static SchneiderRateTable from_binary(
        const std::filesystem::path& binary_path);

    [[nodiscard]] double energy_min_mevu() const noexcept { return energy_min_mevu_; }
    [[nodiscard]] double energy_max_mevu() const noexcept { return energy_max_mevu_; }
    [[nodiscard]] double energy_step_mevu() const noexcept { return energy_step_mevu_; }
    [[nodiscard]] std::size_t num_energies() const noexcept { return num_energies_; }
    [[nodiscard]] std::uint32_t binary_version() const noexcept { return binary_version_; }

    [[nodiscard]] static std::size_t target_index_from_z(int target_z);

    [[nodiscard]] double mass_partial_rate(std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const;
    [[nodiscard]] double mass_total_rate(std::size_t section_id, std::size_t energy_idx) const;

    [[nodiscard]] const std::vector<double>& mass_partial_rates() const noexcept { return mass_partial_rates_; }
    [[nodiscard]] const std::vector<double>& mass_total_rates() const noexcept { return mass_total_rates_; }
    // Per-target valid domain, 13 entries (required by v3).
    [[nodiscard]] bool has_channel_domains() const noexcept { return !channel_domains_.empty(); }
    [[nodiscard]] const std::vector<SchneiderRateDomainEntry>& channel_domains() const noexcept {
        return channel_domains_;
    }
    [[nodiscard]] const SchneiderRateDomainEntry& channel_domain(std::size_t target_idx) const {
        if (target_idx >= kSchneiderNumTargets || channel_domains_.size() != kSchneiderNumTargets) {
            throw std::out_of_range("SchneiderRateTable::channel_domain: index out of range");
        }
        return channel_domains_[target_idx];
    }
    // v3 masked partial at an energy node: exactly 0 outside the channel
    // domain, raw node value inside.
    [[nodiscard]] double masked_partial_at_node(std::size_t section_id, std::size_t target_idx,
                                               std::size_t energy_idx) const {
        const SchneiderRateDomainEntry& dom = channel_domain(target_idx);
        if (!dom.has_support) {
            return 0.0;
        }
        const double e = energy_min_mevu_ + static_cast<double>(energy_idx) * energy_step_mevu_;
        if (e < dom.energy_min_mevu || e > dom.energy_max_mevu) {
            return 0.0;
        }
        return mass_partial_rate(section_id, target_idx, energy_idx);
    }

private:
    std::uint32_t binary_version_{3};
    std::size_t num_energies_{kSchneiderNumEnergies};
    double energy_min_mevu_{0.1};
    double energy_max_mevu_{460.1};
    double energy_step_mevu_{0.5};
    // Flattened array: [section][target][energy] = section * (13 * num_energies) + target * num_energies + energy
    std::vector<double> mass_partial_rates_;
    // Flattened array: [section][energy] = section * num_energies + energy
    std::vector<double> mass_total_rates_;
    std::vector<SchneiderRateDomainEntry> channel_domains_;
};

}  // namespace carbon
