#pragma once

#include "carbon/reaction_package.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

struct CascadeProjectile {
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    std::uint32_t cross_section_offset{0};
    std::uint32_t cross_section_count{0};
    std::uint32_t interaction_offset{0};
    std::uint32_t interaction_count{0};
};

struct CascadeCrossSectionSample {
    float energy_MeV_per_u{0.0F};
    float macroscopic_cross_section_per_mm{0.0F};
};

struct CascadeInteraction {
    float incident_energy_MeV_per_u{0.0F};
    // Reaction depth in the reference phantom.
    float depth_mm{0.0F};
    // Geant4 step-local deposit at the sampled inelastic interaction.
    float local_deposit_MeV{0.0F};
    std::uint32_t product_offset{0};
    std::uint32_t product_count{0};
};

static_assert(sizeof(CascadeProjectile) == 20);
static_assert(sizeof(CascadeCrossSectionSample) == 8);
static_assert(sizeof(CascadeInteraction) == 20);

class CascadePackageTable {
public:
    static CascadePackageTable from_binary(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<CascadeProjectile>& projectiles() const noexcept;
    [[nodiscard]] const std::vector<CascadeCrossSectionSample>& cross_sections() const noexcept;
    [[nodiscard]] const std::vector<CascadeInteraction>& interactions() const noexcept;
    [[nodiscard]] const std::vector<ReactionSecondary>& products() const noexcept;
    [[nodiscard]] const CascadeProjectile* find_projectile(int atomic_number,
                                                           int mass_number) const noexcept;

private:
    std::vector<CascadeProjectile> projectiles_;
    std::vector<CascadeCrossSectionSample> cross_sections_;
    std::vector<CascadeInteraction> interactions_;
    std::vector<ReactionSecondary> products_;
};

}  // namespace carbon
