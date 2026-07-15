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

// Lineage tags stored in SecondaryParticle3D::reserved for ancestor attribution.
inline constexpr std::uint16_t charged_lineage = 0;
inline constexpr std::uint16_t neutron_lineage = 1;
inline constexpr std::uint16_t gamma_lineage = 2;

inline constexpr std::size_t neutral_origin_category_count = 2;
inline constexpr std::size_t neutron_origin_category = 0;
inline constexpr std::size_t gamma_origin_category = 1;

constexpr std::uint16_t neutral_lineage_from_pdg(const int pdg_id) noexcept {
    return pdg_id == 22 ? gamma_lineage : neutron_lineage;
}

constexpr std::size_t neutral_origin_category_from_lineage(
    const std::uint16_t lineage) noexcept {
    return lineage == gamma_lineage ? gamma_origin_category : neutron_origin_category;
}

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
    std::uint16_t reserved{0};  // charged_lineage / neutron_lineage / gamma_lineage
};

static_assert(sizeof(SecondaryParticle3D) == 40);

struct NeutralParticle3D {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{0.0F};
    std::int32_t pdg_id{0};
    std::uint8_t origin_category{0};  // neutron_origin_category / gamma_origin_category
    std::uint8_t generation{0};
    std::uint16_t reserved{0};
};

static_assert(sizeof(NeutralParticle3D) == 36);

struct SecondaryGenerationSummary {
    std::uint32_t direct_count{0};
    std::uint32_t queued_count{0};
    std::uint32_t overflow_count{0};
    float queued_energy_MeV{0.0F};
    float neutral_energy_MeV{0.0F};
    float unsupported_charged_energy_MeV{0.0F};
    float overflow_energy_MeV{0.0F};
    std::uint32_t queued_neutral_count{0};
    float queued_neutral_energy_MeV{0.0F};
    std::uint32_t neutral_queue_overflow_count{0};
    float neutral_queue_overflow_energy_MeV{0.0F};
};

static_assert(sizeof(SecondaryGenerationSummary) == 44);

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
    std::uint32_t queued_neutral_count{0};
    float queued_neutral_energy_MeV{0.0F};
    std::uint32_t neutral_queue_overflow_count{0};
    float neutral_queue_overflow_energy_MeV{0.0F};
};

static_assert(sizeof(CascadeTransportSummary) == 52);

struct NeutralTransportSummary {
    std::uint32_t interaction_count{0};
    std::uint32_t queued_charged_count{0};
    std::uint32_t charged_overflow_count{0};
    std::uint32_t continuation_count{0};
    std::uint32_t neutral_overflow_count{0};
    float local_deposit_MeV{0.0F};
    float queued_charged_energy_MeV{0.0F};
    float charged_overflow_energy_MeV{0.0F};
    float continuation_energy_MeV{0.0F};  // requeued only in "full" mode
    float neutral_overflow_energy_MeV{0.0F};
    float escaped_energy_MeV{0.0F};
    float residual_energy_MeV{0.0F};  // continuation/nested neutrals not re-queued
};

static_assert(sizeof(NeutralTransportSummary) == 48);

}  // namespace carbon
