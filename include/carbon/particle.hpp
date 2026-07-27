#pragma once

#include <cstddef>
#include <cstdint>

namespace carbon {

inline constexpr std::size_t charged_origin_category_count = 8;
inline constexpr std::size_t primary_c12_charged_origin_category = 0;
inline constexpr std::size_t light_isotope_category_count = 5;

// Optional LET diagnostics: p, d, t, He-3, He-4. Other ions return the
// sentinel category_count and are not accumulated.
constexpr std::size_t light_isotope_category(const int atomic_number,
                                             const int mass_number) noexcept {
    if (atomic_number == 1 && mass_number >= 1 && mass_number <= 3) {
        return static_cast<std::size_t>(mass_number - 1);
    }
    if (atomic_number == 2 && (mass_number == 3 || mass_number == 4)) {
        return static_cast<std::size_t>(mass_number);
    }
    return light_isotope_category_count;
}

// Fragment birth-spectrum diagnostics (p/d/t/He-3/He-4) for high-energy cascade
// validation. Generation bins: 0 = primary C-12 direct product, 1 = first cascade
// generation, 2 = generation >= 2.
inline constexpr std::size_t birth_generation_bin_count = 3;
inline constexpr std::size_t birth_mevu_bin_count = 200;  // 0–400 MeV/u @ 2 MeV/u
inline constexpr double birth_mevu_bin_width = 2.0;
inline constexpr double birth_mevu_max = 400.0;
inline constexpr std::size_t birth_cos_bin_count = 20;  // cosθ ∈ [-1, 1]
inline constexpr std::size_t birth_parent_mevu_bin_count = 40;  // 0–400 @ 10 MeV/u
inline constexpr double birth_parent_mevu_bin_width = 10.0;
inline constexpr std::size_t birth_parent_z_bin_count = 9;  // index = clamp(Z, 0..8)

constexpr std::size_t birth_generation_bin(const std::uint8_t generation) noexcept {
    return generation >= 2 ? 2 : static_cast<std::size_t>(generation);
}

constexpr std::size_t birth_mevu_bin(const double kinetic_energy_MeV,
                                    const int mass_number) noexcept {
    if (mass_number <= 0 || !(kinetic_energy_MeV > 0.0)) {
        return 0;
    }
    const double mevu = kinetic_energy_MeV / static_cast<double>(mass_number);
    if (!(mevu > 0.0)) {
        return 0;
    }
    auto bin = static_cast<std::size_t>(mevu / birth_mevu_bin_width);
    if (bin >= birth_mevu_bin_count) {
        bin = birth_mevu_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_cos_bin(const double direction_z) noexcept {
    double c = direction_z;
    if (c < -1.0) {
        c = -1.0;
    }
    if (c > 1.0) {
        c = 1.0;
    }
    auto bin = static_cast<std::size_t>(((c + 1.0) * 0.5) * birth_cos_bin_count);
    if (bin >= birth_cos_bin_count) {
        bin = birth_cos_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_parent_mevu_bin(const double parent_kinetic_energy_MeV,
                                           const int parent_mass_number) noexcept {
    if (parent_mass_number <= 0 || !(parent_kinetic_energy_MeV > 0.0)) {
        return 0;
    }
    const double mevu =
        parent_kinetic_energy_MeV / static_cast<double>(parent_mass_number);
    auto bin = static_cast<std::size_t>(mevu / birth_parent_mevu_bin_width);
    if (bin >= birth_parent_mevu_bin_count) {
        bin = birth_parent_mevu_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_parent_z_bin(const int parent_atomic_number) noexcept {
    if (parent_atomic_number <= 0) {
        return 0;
    }
    if (parent_atomic_number >= static_cast<int>(birth_parent_z_bin_count)) {
        return birth_parent_z_bin_count - 1;
    }
    return static_cast<std::size_t>(parent_atomic_number);
}

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
    // Deterministic Philox stream id (not the atomic queue slot). Primary children
    // derive from history; cascade/neutral children derive from parent stream.
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(SecondaryParticle3D) == 48);

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
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(NeutralParticle3D) == 48);

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
    // Local residual heat deposited at cascade sites (dose map, not untracked).
    float residual_local_MeV{0.0F};
};

static_assert(sizeof(CascadeTransportSummary) == 56);

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
