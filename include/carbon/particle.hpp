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

// Other source ions supply both mass conventions explicitly. Never infer a
// nuclear mass from A*amu or change the calibrated proton/C12 conventions.
inline double resolve_primary_mass_MeV(int z, int a,
    PrimaryMassConvention convention, double supplied_mass_MeV = 0.0) {
    if (z <= 0 || a < z)
        throw std::invalid_argument("Primary ion requires Z > 0 and A >= Z");
    const double calibrated = convention == PrimaryMassConvention::Kinematic
        ? primary_kinematic_mass_MeV(z, a) : primary_nuclear_mass_MeV(z, a);
    const double mass = supplied_mass_MeV == 0.0 ? calibrated : supplied_mass_MeV;
    if (!(mass > 0.0) || !std::isfinite(mass))
        throw std::invalid_argument(convention == PrimaryMassConvention::Kinematic
            ? "Specify a finite positive primary_kinematic_mass_MeV for this source ion"
            : "Specify a finite positive primary_nuclear_mass_MeV for this source ion");
    if (std::isfinite(calibrated) && std::abs(mass - calibrated) > 2.0e-6 * calibrated)
        throw std::invalid_argument("Primary mass disagrees with the calibrated proton/C12 convention");
    // An optional matching value is an assertion, not a recalibration.
    return std::isfinite(calibrated) ? calibrated : mass;
}

struct PrimaryIonDefinition {
    int atomic_number{6};
    int mass_number{12};
    double rest_mass_MeV{12.0 * ion_nucleon_rest_mass_MeV};
    double inverse_mass_number{1.0 / 12.0};
};

inline PrimaryIonDefinition make_primary_ion_definition(
    const int atomic_number, const int mass_number,
    const double kinematic_mass_MeV = 0.0) {
    const double mass = resolve_primary_mass_MeV(atomic_number, mass_number,
        PrimaryMassConvention::Kinematic, kinematic_mass_MeV);
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

}  // namespace carbon
