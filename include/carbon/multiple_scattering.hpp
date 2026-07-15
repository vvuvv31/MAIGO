#pragma once

#include <algorithm>
#include <cmath>

namespace carbon {

constexpr double nucleon_rest_mass_MeV = 931.49410242;
constexpr double water_radiation_length_g_per_cm2 = 36.08;
constexpr double highland_energy_constant_MeV = 13.6;

inline double highland_projected_rms_angle_rad(double kinetic_energy_MeV,
                                                int atomic_number,
                                                int mass_number,
                                                double path_length_mm,
                                                double density_g_per_cm3) noexcept {
    if (kinetic_energy_MeV <= 0.0 || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0 || density_g_per_cm3 <= 0.0) {
        return 0.0;
    }

    const auto energy_MeV_per_u = kinetic_energy_MeV / static_cast<double>(mass_number);
    const auto total_energy_MeV_per_u = energy_MeV_per_u + nucleon_rest_mass_MeV;
    const auto momentum_MeV_per_c_per_u =
        std::sqrt(energy_MeV_per_u *
                  (energy_MeV_per_u + 2.0 * nucleon_rest_mass_MeV));
    const auto beta = momentum_MeV_per_c_per_u / total_energy_MeV_per_u;
    const auto momentum_MeV_per_c =
        static_cast<double>(mass_number) * momentum_MeV_per_c_per_u;
    const auto radiation_lengths =
        density_g_per_cm3 * (path_length_mm / 10.0) /
        water_radiation_length_g_per_cm2;
    if (beta <= 0.0 || momentum_MeV_per_c <= 0.0 || radiation_lengths <= 0.0) {
        return 0.0;
    }

    const auto charge = static_cast<double>(atomic_number);
    const auto logarithm_argument =
        radiation_lengths * charge * charge / (beta * beta);
    const auto correction =
        std::max(0.0, 1.0 + 0.038 * std::log(logarithm_argument));
    return highland_energy_constant_MeV * charge /
           (beta * momentum_MeV_per_c) * std::sqrt(radiation_lengths) * correction;
}

}  // namespace carbon
