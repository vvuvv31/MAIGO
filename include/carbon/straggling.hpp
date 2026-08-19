#pragma once

#include "carbon/stopping_power.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace carbon {

inline constexpr std::size_t max_straggling_scale_points = 16;

template <typename Scalar>
inline Scalar interpolate_straggling_scale(
    const Scalar energy_MeVu,
    const std::array<Scalar, max_straggling_scale_points>& energies_MeVu,
    const std::array<Scalar, max_straggling_scale_points>& scales,
    const std::size_t point_count,
    const Scalar fallback_scale) noexcept {
    if (point_count == 0) {
        return fallback_scale;
    }
    if (point_count == 1 || energy_MeVu <= energies_MeVu[0]) {
        return scales[0];
    }
    for (std::size_t index = 1; index < point_count; ++index) {
        if (energy_MeVu <= energies_MeVu[index]) {
            const auto denominator = energies_MeVu[index] - energies_MeVu[index - 1];
            const auto fraction = (energy_MeVu - energies_MeVu[index - 1]) / denominator;
            return scales[index - 1] + fraction * (scales[index] - scales[index - 1]);
        }
    }
    return scales[point_count - 1];
}

inline double bohr_straggling_sigma_MeV(double energy_MeVu,
                                        int atomic_number,
                                        double step_mm,
                                        double density_g_per_cm3,
                                        double scale = 1.0,
                                        double z_over_a_rel_water = 1.0) noexcept {
    constexpr double bethe_K_MeV_cm2_per_g = 0.307075;
    constexpr double electron_mass_MeV = 0.51099895;
    constexpr double water_Z_over_A = 0.55509;
    const auto effective_charge = ion_effective_charge(atomic_number, energy_MeVu);
    const auto step_cm = step_mm / 10.0;
    // z_over_a_rel_water = (Z/A)_mat / (Z/A)_water  (Schneider mass-SP za_rel).
    const auto z_over_a =
        water_Z_over_A * std::max(0.5, std::min(1.5, z_over_a_rel_water));
    const auto variance_MeV2 = bethe_K_MeV_cm2_per_g * electron_mass_MeV *
                               effective_charge * effective_charge * z_over_a *
                               density_g_per_cm3 * step_cm;
    return scale * std::sqrt(std::max(0.0, variance_MeV2));
}

template <typename Scalar>
inline Scalar condensed_total_loss_variance_MeV2(
    const Scalar energy_MeVu,
    const Scalar projectile_mass_MeV,
    const Scalar effective_charge,
    const Scalar step_mm,
    const Scalar density_g_per_cm3,
    const Scalar z_over_a_rel_water = Scalar{1}) noexcept {
    constexpr Scalar electron_mass_MeV = Scalar{0.51099895};
    constexpr Scalar bethe_K_MeV_cm2_per_g = Scalar{0.307075};
    constexpr Scalar water_Z_over_A = Scalar{0.55509};
    constexpr Scalar nucleon_mass_MeV = Scalar{931.49410242};

    const auto gamma = Scalar{1} + energy_MeVu / nucleon_mass_MeV;
    const auto beta_squared = std::max(
        Scalar{0}, Scalar{1} - Scalar{1} / (gamma * gamma));
    if (beta_squared <= Scalar{0} || projectile_mass_MeV <= Scalar{0}) {
        return Scalar{0};
    }
    const auto mass_ratio = electron_mass_MeV / projectile_mass_MeV;
    const auto maximum_transfer_MeV =
        Scalar{2} * electron_mass_MeV * beta_squared * gamma * gamma /
        (Scalar{1} + Scalar{2} * gamma * mass_ratio + mass_ratio * mass_ratio);

    // The transport condenses continuous loss and unresolved hard electron
    // transfers into one local step. Integrating the heavy-particle collision
    // spectrum through Tmax is the G4 dispersion expression with Tcut=Tmax.
    // The multiplier tends to one in the non-relativistic Bohr limit.
    const auto relativistic_multiplier =
        (maximum_transfer_MeV / beta_squared - Scalar{0.5} * maximum_transfer_MeV) /
        (Scalar{2} * electron_mass_MeV);
    const auto z_over_a = water_Z_over_A * std::clamp(
        z_over_a_rel_water, Scalar{0.5}, Scalar{1.5});
    return bethe_K_MeV_cm2_per_g * electron_mass_MeV *
           effective_charge * effective_charge * z_over_a *
           density_g_per_cm3 * (step_mm / Scalar{10}) *
           std::max(Scalar{0}, relativistic_multiplier);
}

inline double clamp_sampled_energy_loss(double mean_loss_MeV,
                                        double sigma_MeV,
                                        double gaussian,
                                        double available_energy_MeV) noexcept {
    return std::clamp(mean_loss_MeV + sigma_MeV * gaussian, 0.0,
                      std::min(2.0 * mean_loss_MeV, available_energy_MeV));
}

}  // namespace carbon
