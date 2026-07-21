#pragma once

#include <algorithm>
#include <cmath>

namespace carbon {

inline double carbon_effective_charge(double energy_MeVu) noexcept {
    constexpr double nucleon_mass_MeV = 931.49410242;
    constexpr double carbon_atomic_number = 6.0;
    const auto gamma = 1.0 + energy_MeVu / nucleon_mass_MeV;
    const auto beta_squared = std::max(0.0, 1.0 - 1.0 / (gamma * gamma));
    const auto beta = std::sqrt(beta_squared);
    return carbon_atomic_number *
           (1.0 - std::exp(-125.0 * beta * std::pow(carbon_atomic_number, -2.0 / 3.0)));
}

inline double bohr_straggling_sigma_MeV(double energy_MeVu,
                                        double step_mm,
                                        double density_g_per_cm3,
                                        double scale = 1.0,
                                        double z_over_a_rel_water = 1.0) noexcept {
    constexpr double bethe_K_MeV_cm2_per_g = 0.307075;
    constexpr double electron_mass_MeV = 0.51099895;
    constexpr double water_Z_over_A = 0.55509;
    const auto effective_charge = carbon_effective_charge(energy_MeVu);
    const auto step_cm = step_mm / 10.0;
    // z_over_a_rel_water = (Z/A)_mat / (Z/A)_water  (Schneider mass-SP za_rel).
    const auto z_over_a =
        water_Z_over_A * std::max(0.5, std::min(1.5, z_over_a_rel_water));
    const auto variance_MeV2 = bethe_K_MeV_cm2_per_g * electron_mass_MeV *
                               effective_charge * effective_charge * z_over_a *
                               density_g_per_cm3 * step_cm;
    return scale * std::sqrt(std::max(0.0, variance_MeV2));
}

inline double clamp_sampled_energy_loss(double mean_loss_MeV,
                                        double sigma_MeV,
                                        double gaussian,
                                        double available_energy_MeV) noexcept {
    return std::clamp(mean_loss_MeV + sigma_MeV * gaussian, 0.0, available_energy_MeV);
}

}  // namespace carbon

