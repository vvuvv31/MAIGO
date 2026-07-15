#pragma once

#include "carbon/reaction_package.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

struct NeutralProjectile {
    std::int32_t pdg_id{0};
    std::uint32_t cross_section_offset{0};
    std::uint32_t cross_section_count{0};
    std::uint32_t interaction_offset{0};
    std::uint32_t interaction_count{0};
};

struct NeutralCrossSectionSample {
    float energy_MeV{0.0F};
    float macroscopic_total_per_mm{0.0F};
};

struct NeutralInteraction {
    float incident_energy_MeV{0.0F};
    float continuation_energy_MeV{0.0F};
    float local_deposit_MeV{0.0F};
    float continuation_direction_x{0.0F};
    float continuation_direction_y{0.0F};
    float continuation_direction_z{0.0F};
    std::int32_t process_type{0};
    std::int32_t process_subtype{0};
    std::uint32_t product_offset{0};
    std::uint32_t product_count{0};
};

static_assert(sizeof(NeutralProjectile) == 20);
static_assert(sizeof(NeutralCrossSectionSample) == 8);
static_assert(sizeof(NeutralInteraction) == 40);

class NeutralPackageTable {
public:
    static NeutralPackageTable from_binary(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<NeutralProjectile>& projectiles() const noexcept;
    [[nodiscard]] const std::vector<NeutralCrossSectionSample>& cross_sections() const noexcept;
    [[nodiscard]] const std::vector<NeutralInteraction>& interactions() const noexcept;
    [[nodiscard]] const std::vector<ReactionSecondary>& products() const noexcept;
    [[nodiscard]] const NeutralProjectile* find_projectile(int pdg_id) const noexcept;

private:
    std::vector<NeutralProjectile> projectiles_;
    std::vector<NeutralCrossSectionSample> cross_sections_;
    std::vector<NeutralInteraction> interactions_;
    std::vector<ReactionSecondary> products_;
};

}  // namespace carbon
