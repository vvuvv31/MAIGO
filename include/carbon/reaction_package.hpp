#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace carbon {

struct ReactionEnergyBin {
    std::uint32_t reaction_offset{0};
    std::uint32_t reaction_count{0};
};

struct ReactionPackage {
    float incident_energy_MeV_per_u{0.0F};
    float reaction_depth_mm{0.0F};
    std::uint32_t secondary_offset{0};
    std::uint32_t secondary_count{0};
};

struct ReactionSecondary {
    std::int32_t pdg_id{0};
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    float kinetic_energy_MeV{0.0F};
    float direction_z{0.0F};
};

class ReactionPackageTable {
public:
    static ReactionPackageTable from_binary(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<ReactionEnergyBin>& energy_bins() const noexcept;
    [[nodiscard]] const std::vector<ReactionPackage>& reactions() const noexcept;
    [[nodiscard]] const std::vector<ReactionSecondary>& secondaries() const noexcept;
    [[nodiscard]] float minimum_energy_MeV_per_u() const noexcept;
    [[nodiscard]] float energy_bin_width_MeV_per_u() const noexcept;
    [[nodiscard]] std::size_t energy_bin_index(float energy_MeV_per_u) const noexcept;
    [[nodiscard]] const ReactionEnergyBin& energy_bin(float energy_MeV_per_u) const noexcept;

private:
    float minimum_energy_MeV_per_u_{0.0F};
    float energy_bin_width_MeV_per_u_{1.0F};
    std::vector<ReactionEnergyBin> energy_bins_;
    std::vector<ReactionPackage> reactions_;
    std::vector<ReactionSecondary> secondaries_;
};

}  // namespace carbon
