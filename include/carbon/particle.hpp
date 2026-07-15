#pragma once

#include <cstddef>
#include <cstdint>

namespace carbon {

inline constexpr std::size_t charged_origin_category_count = 8;
inline constexpr std::size_t primary_c12_charged_origin_category = 0;

constexpr std::uint8_t charged_dose_category(const int atomic_number,
                                             const int mass_number) noexcept {
    if (atomic_number == 1 && mass_number == 1) {
        return 5;
    }
    if (atomic_number >= 2 && atomic_number <= 6) {
        return static_cast<std::uint8_t>(6 - atomic_number);
    }
    return 6;
}

constexpr std::size_t charged_origin_category_from_fragment(
    const std::uint8_t fragment_category) noexcept {
    return static_cast<std::size_t>(fragment_category) + 1;
}

struct Particle1D {
    double position_mm{0.0};
    double kinetic_energy_MeV{0.0};
    int particle_type{6012};
    bool alive{true};
};

struct SecondaryParticle3D {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{0.0F};
    std::int32_t pdg_id{0};
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    std::uint8_t origin_category{0};
    std::uint8_t generation{0};
    std::uint16_t reserved{0};
};

static_assert(sizeof(SecondaryParticle3D) == 40);

struct SecondaryGenerationSummary {
    std::uint32_t direct_count{0};
    std::uint32_t queued_count{0};
    std::uint32_t overflow_count{0};
    float queued_energy_MeV{0.0F};
    float neutral_energy_MeV{0.0F};
    float unsupported_charged_energy_MeV{0.0F};
    float overflow_energy_MeV{0.0F};
};

static_assert(sizeof(SecondaryGenerationSummary) == 28);

struct CascadeTransportSummary {
    std::uint32_t interaction_count{0};
    std::uint32_t direct_count{0};
    std::uint32_t queued_count{0};
    std::uint32_t overflow_count{0};
    float incident_energy_MeV{0.0F};
    float queued_energy_MeV{0.0F};
    float neutral_energy_MeV{0.0F};
    float unsupported_charged_energy_MeV{0.0F};
    float overflow_energy_MeV{0.0F};
};

static_assert(sizeof(CascadeTransportSummary) == 36);

}  // namespace carbon
