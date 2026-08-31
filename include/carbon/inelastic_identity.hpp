#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace carbon {

// The TOPAS replay device transports fully stripped ions, gamma and neutron.
// Electrons/positrons and pions are recognized only so that the package
// compiler can place them in the explicit unsupported-energy ledger.  They
// are not silently converted into an ion or a neutral transport product.
enum class InelasticSpeciesKind : std::uint8_t {
    invalid = 0,
    ion,
    gamma,
    neutron,
    electron,
    positron,
    charged_pion,
    neutral_pion,
    eta_meson,
};

struct InelasticSpeciesIdentity {
    InelasticSpeciesKind kind{InelasticSpeciesKind::invalid};

    [[nodiscard]] bool valid() const noexcept {
        return kind != InelasticSpeciesKind::invalid;
    }

    [[nodiscard]] bool runtime_supported() const noexcept {
        return kind == InelasticSpeciesKind::ion ||
               kind == InelasticSpeciesKind::gamma ||
               kind == InelasticSpeciesKind::neutron;
    }
};

inline constexpr double inelastic_nucleon_rest_mass_MeV = 931.49410242;
inline constexpr double inelastic_proton_rest_mass_MeV = 938.27208816;
inline constexpr double inelastic_neutron_rest_mass_MeV = 939.56542052;
inline constexpr double inelastic_electron_rest_mass_MeV = 0.51099895;
inline constexpr double inelastic_charged_pion_rest_mass_MeV = 139.57039;
inline constexpr double inelastic_eta_meson_rest_mass_MeV = 547.862;
inline constexpr double inelastic_neutral_pion_rest_mass_MeV = 134.9768;

inline bool inelastic_close(const float value, const double expected,
                            const double relative_tolerance = 1.0e-3,
                            const double absolute_tolerance = 1.0e-3) noexcept {
    if (!std::isfinite(value) || !std::isfinite(expected)) {
        return false;
    }
    return std::abs(static_cast<double>(value) - expected) <=
           std::max(absolute_tolerance,
                    relative_tolerance * std::max(1.0, std::abs(expected)));
}

inline bool inelastic_ground_state(const float excitation_MeV) noexcept {
    return std::isfinite(excitation_MeV) && excitation_MeV >= 0.0F &&
           excitation_MeV <= 1.0e-4F;
}

inline std::int32_t inelastic_ground_state_ion_pdg(const int atomic_number,
                                                   const int mass_number) noexcept {
    if (atomic_number <= 0 || mass_number < atomic_number) {
        return 0;
    }
    const auto pdg = static_cast<std::int64_t>(1000000000LL) +
                     static_cast<std::int64_t>(atomic_number) * 10000LL +
                     static_cast<std::int64_t>(mass_number) * 10LL;
    if (pdg > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        return 0;
    }
    return static_cast<std::int32_t>(pdg);
}

inline bool inelastic_ion_mass_is_explainable(const int mass_number,
                                               const float rest_mass_MeV) noexcept {
    if (mass_number <= 0 || !std::isfinite(rest_mass_MeV) || rest_mass_MeV <= 0.0F) {
        return false;
    }
    // The runtime species model uses A times the nucleon mass.  A broad
    // binding-energy allowance accepts Geant4 ground-state masses (including
    // light nuclei) while rejecting a missing or unrelated particle mass.
    const auto expected = static_cast<double>(mass_number) *
                          inelastic_nucleon_rest_mass_MeV;
    const auto tolerance = std::max(25.0, 0.02 * expected);
    return std::abs(static_cast<double>(rest_mass_MeV) - expected) <= tolerance;
}

inline InelasticSpeciesIdentity classify_inelastic_species(
    const std::int32_t pdg,
    const int atomic_number,
    const int mass_number,
    const float charge_e,
    const float rest_mass_MeV,
    const float excitation_MeV) noexcept {
    if (pdg == 0 || !inelastic_ground_state(excitation_MeV) ||
        !std::isfinite(charge_e) || !std::isfinite(rest_mass_MeV) ||
        rest_mass_MeV < 0.0F) {
        return {};
    }

    if (pdg == 22) {
        if (atomic_number == 0 && mass_number == 0 &&
            inelastic_close(charge_e, 0.0, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, 0.0, 0.0, 1.0e-3)) {
            return {InelasticSpeciesKind::gamma};
        }
        return {};
    }
    if (pdg == 2112) {
        if (atomic_number == 0 && mass_number == 1 &&
            inelastic_close(charge_e, 0.0, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, inelastic_neutron_rest_mass_MeV)) {
            return {InelasticSpeciesKind::neutron};
        }
        return {};
    }
    if (pdg == 2212) {
        if (atomic_number == 1 && mass_number == 1 &&
            inelastic_close(charge_e, 1.0, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, inelastic_proton_rest_mass_MeV)) {
            return {InelasticSpeciesKind::ion};
        }
        return {};
    }
    if (pdg == 11 || pdg == -11) {
        const auto expected_charge = pdg == 11 ? -1.0 : 1.0;
        if (atomic_number == 0 && mass_number == 0 &&
            inelastic_close(charge_e, expected_charge, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, inelastic_electron_rest_mass_MeV)) {
            return {pdg == 11 ? InelasticSpeciesKind::electron
                               : InelasticSpeciesKind::positron};
        }
        return {};
    }
    if (pdg == 211 || pdg == -211) {
        const auto expected_charge = pdg == 211 ? 1.0 : -1.0;
        if (atomic_number == 0 && mass_number == 0 &&
            inelastic_close(charge_e, expected_charge, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, inelastic_charged_pion_rest_mass_MeV)) {
            return {InelasticSpeciesKind::charged_pion};
        }
        return {};
    }
    if (pdg == 111) {
        if (atomic_number == 0 && mass_number == 0 &&
            inelastic_close(charge_e, 0.0, 0.0, 1.0e-3) &&
            inelastic_close(rest_mass_MeV, inelastic_neutral_pion_rest_mass_MeV)) {
            return {InelasticSpeciesKind::neutral_pion};
        }
        return {};
    }
    if (pdg == 221) {
        if (atomic_number == 0 && mass_number == 0 &&
            inelastic_close(charge_e, 0.0, 0.0) &&
            inelastic_close(rest_mass_MeV, inelastic_eta_meson_rest_mass_MeV)) {
            return {InelasticSpeciesKind::eta_meson};
        }
        return {};
    }

    if (atomic_number <= 0 || mass_number < atomic_number ||
        pdg != inelastic_ground_state_ion_pdg(atomic_number, mass_number) ||
        !inelastic_close(charge_e, static_cast<double>(atomic_number), 0.0, 1.0e-3) ||
        !inelastic_ion_mass_is_explainable(mass_number, rest_mass_MeV)) {
        return {};
    }
    return {InelasticSpeciesKind::ion};
}

}  // namespace carbon
