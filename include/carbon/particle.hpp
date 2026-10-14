#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

namespace carbon {

inline constexpr double ion_nucleon_rest_mass_MeV = 931.49410242;

enum class PrimaryMassConvention { Kinematic, Nuclear };

// Production kinematic convention, also used by primary CINEL03 events.
// Preserve the calibrated C12 A*amu convention separately from nuclear masses.
inline double primary_kinematic_mass_MeV(int z, int a) noexcept {
    if (z == 1 && a == 1) return 938.27208816;
    if (z == 6 && a == 12) return 12.0 * ion_nucleon_rest_mass_MeV;
    return std::numeric_limits<double>::quiet_NaN();
}

// Nuclear mass contract for current Urban, continuous26 EM and elastic banks.
inline double primary_nuclear_mass_MeV(int z, int a) noexcept {
    if (z == 1 && a == 1) return 938.27208816;
    if (z == 6 && a == 12) return 11174.86338798439;
    return std::numeric_limits<double>::quiet_NaN();
}

struct PrimaryIonDefinition {
    int atomic_number{6};
    int mass_number{12};
    double rest_mass_MeV{12.0 * ion_nucleon_rest_mass_MeV};
    double inverse_mass_number{1.0 / 12.0};
};

inline PrimaryIonDefinition make_primary_ion_definition(
    const int atomic_number, const int mass_number) {
    const double mass = primary_kinematic_mass_MeV(atomic_number, mass_number);
    if (!(mass > 0.0) || !std::isfinite(mass))
        throw std::invalid_argument("Production primary must be proton (Z=1,A=1) or C12 (Z=6,A=12)");
    return PrimaryIonDefinition{atomic_number, mass_number, mass,
        1.0 / static_cast<double>(mass_number)};
}

inline constexpr std::uint8_t minibeam_birth_region_copper = 1;

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
    std::uint16_t reserved{0};  // Padding.
    // Deterministic Philox stream id (not the atomic queue slot). Primary children
    // derive from history; cascade children derive from parent stream.
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(SecondaryParticle3D) == 48);

struct ElectronParticle3D {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{0.0F};
    std::int32_t pdg_id{11};  // 11 = e-, -11 = e+
    std::uint8_t origin_category{0};
    std::uint8_t generation{0};
    std::uint16_t reserved{0};
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(ElectronParticle3D) == 48);
inline constexpr float positron_annihilation_reserve_MeV = 1.0219979F;

}  // namespace carbon
