#pragma once

#include "carbon/stopping_power.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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

inline constexpr int straggling_sampler_gaussian_clamped = 0;
inline constexpr int straggling_sampler_moment_matched = 1;
inline constexpr int straggling_sampler_vavilov_landau = 2;
// Thick-layer Gaussian is used when μ/σ is large enough that P(ΔE<0) is
// negligible. Below this, a moment-matched Gamma (thin: compound-Poisson
// Gamma with k<1) keeps the support positive without a 2μ cap.
inline constexpr double straggling_gaussian_kappa_min = 4.0;

inline double clamp_sampled_energy_loss(double mean_loss_MeV,
                                        double sigma_MeV,
                                        double gaussian,
                                        double available_energy_MeV) noexcept {
    return std::clamp(mean_loss_MeV + sigma_MeV * gaussian, 0.0,
                      std::min(2.0 * mean_loss_MeV, available_energy_MeV));
}

template <typename Scalar>
inline Scalar integrate_path_variance_MeV2(
    const Scalar variance_at_start_MeV2,
    const Scalar variance_at_end_MeV2) noexcept {
    return Scalar{0.5} * (std::max(Scalar{0}, variance_at_start_MeV2) +
                          std::max(Scalar{0}, variance_at_end_MeV2));
}

template <typename Scalar>
inline Scalar wilson_hilferty_unit_gamma(const Scalar shape,
                                         const Scalar gaussian) noexcept {
    const auto shifted = Scalar{1} - Scalar{1} / (Scalar{9} * shape) +
                         gaussian / std::sqrt(Scalar{9} * shape);
    const auto cube = shifted * shifted * shifted;
    return shape * std::max(Scalar{1.0e-18}, cube);
}

template <typename Scalar>
inline Scalar sample_unit_gamma(const Scalar shape,
                                const Scalar gaussian,
                                const Scalar extra_uniform) noexcept {
    if (shape >= Scalar{1}) {
        return wilson_hilferty_unit_gamma(shape, gaussian);
    }
    const auto parent = wilson_hilferty_unit_gamma(shape + Scalar{1}, gaussian);
    const auto uniform = std::min(std::max(extra_uniform, Scalar{1.0e-12}),
                                  Scalar{1} - Scalar{1.0e-12});
    return parent * std::pow(uniform, Scalar{1} / shape);
}

template <typename Scalar>
inline Scalar sample_moment_matched_energy_loss(
    const Scalar mean_loss_MeV,
    const Scalar sigma_MeV,
    const Scalar gaussian,
    const Scalar extra_uniform,
    const Scalar available_energy_MeV) noexcept {
    if (!(mean_loss_MeV > Scalar{0}) || !(sigma_MeV > Scalar{0})) {
        return std::clamp(mean_loss_MeV, Scalar{0}, available_energy_MeV);
    }
    const auto kappa = mean_loss_MeV / sigma_MeV;
    Scalar sampled = mean_loss_MeV + sigma_MeV * gaussian;
    const auto use_gaussian =
        kappa >= static_cast<Scalar>(straggling_gaussian_kappa_min) &&
        sampled > Scalar{0};
    if (!use_gaussian) {
        const auto shape = kappa * kappa;
        const auto scale = mean_loss_MeV / shape;
        sampled = scale * sample_unit_gamma(shape, gaussian, extra_uniform);
    }
    return std::clamp(sampled, Scalar{0}, available_energy_MeV);
}

// Condensed fluctuation: Landau/Moyal for thin steps (small κ),
// Gaussian for thick steps. κ uses Vavilov ξ/Tmax ≈ σ² / Tmax².
template <typename Scalar>
inline Scalar sample_vavilov_landau_energy_loss(
    const Scalar mean_loss_MeV,
    const Scalar sigma_MeV,
    const Scalar gaussian,
    const Scalar extra_uniform,
    const Scalar tmax_MeV,
    const Scalar available_energy_MeV) noexcept {
    if (!(mean_loss_MeV > Scalar{0}) || !(sigma_MeV > Scalar{0})) {
        return std::clamp(mean_loss_MeV, Scalar{0}, available_energy_MeV);
    }
    const auto tmax = std::max(tmax_MeV, Scalar{1.0e-6});
    const auto kappa = (sigma_MeV * sigma_MeV) / (tmax * tmax);
    Scalar sampled = mean_loss_MeV + sigma_MeV * gaussian;
    if (kappa < Scalar{0.01}) {
        const auto u = std::min(std::max(extra_uniform, Scalar{1.0e-12}),
                                Scalar{1} - Scalar{1.0e-12});
        const auto lambda = -std::log(-std::log(u)) - Scalar{0.22278};
        const auto xi = sigma_MeV * sigma_MeV / tmax;
        sampled = mean_loss_MeV + xi * lambda;
    } else if (kappa < Scalar{10}) {
        const auto u = std::min(std::max(extra_uniform, Scalar{1.0e-12}),
                                Scalar{1} - Scalar{1.0e-12});
        const auto lambda = -std::log(-std::log(u)) - Scalar{0.22278};
        const auto xi = sigma_MeV * sigma_MeV / tmax;
        const auto landau = mean_loss_MeV + xi * lambda;
        const auto t = (kappa - Scalar{0.01}) / Scalar{9.99};
        sampled = landau * (Scalar{1} - t) + sampled * t;
    }
    return std::clamp(sampled, Scalar{0}, available_energy_MeV);
}

template <typename Scalar>
inline Scalar sample_condensed_energy_loss(
    const Scalar mean_loss_MeV,
    const Scalar sigma_MeV,
    const Scalar gaussian,
    const Scalar extra_uniform,
    const Scalar available_energy_MeV,
    const int sampler) noexcept {
    if (sampler == straggling_sampler_moment_matched) {
        return sample_moment_matched_energy_loss(
            mean_loss_MeV, sigma_MeV, gaussian, extra_uniform,
            available_energy_MeV);
    }
    if (sampler == straggling_sampler_vavilov_landau) {
        const Scalar tmax = std::max(mean_loss_MeV, Scalar{1.0e-3});
        return sample_vavilov_landau_energy_loss(
            mean_loss_MeV, sigma_MeV, gaussian, extra_uniform, tmax,
            available_energy_MeV);
    }
    return std::clamp(mean_loss_MeV + sigma_MeV * gaussian, Scalar{0},
                      std::min(Scalar{2} * mean_loss_MeV, available_energy_MeV));
}

template <typename Scalar>
inline Scalar csda_block_mean_loss_MeV(const Scalar initial_energy_MeV,
                                       const Scalar energy_after_distance_MeVu,
                                       const Scalar mass_number) noexcept {
    return std::clamp(
        initial_energy_MeV - mass_number * energy_after_distance_MeVu,
        Scalar{0}, initial_energy_MeV);
}

template <typename Scalar>
struct StepStableStragglingState {
    std::uint64_t block_index{0};
    Scalar block_remaining_mm{0};
    Scalar block_remaining_loss_MeV{0};
    Scalar block_length_mm{0};
    Scalar sampling_length_mm{0};
    bool block_active{false};

    inline void initialize(const Scalar sampling_length) noexcept {
        sampling_length_mm = sampling_length;
        block_remaining_mm = 0;
        block_length_mm = 0;
        block_remaining_loss_MeV = 0;
        block_active = false;
        block_index = 0;
    }

    inline void begin_block(const Scalar mean_loss_MeV,
                            const Scalar variance_MeV2,
                            const Scalar step_mm,
                            const Scalar scale,
                            const Scalar gaussian,
                            const Scalar available_energy_MeV,
                            const Scalar extra_uniform = Scalar{0.5},
                            const int sampler = straggling_sampler_gaussian_clamped) noexcept {
        const auto effective_length = block_length_mm > Scalar{0}
            ? block_length_mm : sampling_length_mm;
        const auto rate = step_mm > Scalar{0} ? mean_loss_MeV / step_mm : Scalar{0};
        const auto variance_rate = step_mm > Scalar{0}
            ? std::max(Scalar{0}, variance_MeV2 / step_mm) : Scalar{0};
        const auto block_mean = rate * effective_length;
        const auto block_sigma = scale * std::sqrt(variance_rate * effective_length);
        block_remaining_loss_MeV = sample_condensed_energy_loss(
            block_mean, block_sigma, gaussian, extra_uniform,
            available_energy_MeV, sampler);
        block_remaining_mm = effective_length;
        block_active = true;
    }

    inline Scalar consume_loss(const Scalar step_mm,
                                const Scalar available_energy_MeV) noexcept {
        const auto epsilon = std::max(Scalar{1.0e-7}, sampling_length_mm * Scalar{1.0e-6});
        const auto fraction = block_remaining_mm <= epsilon
            ? Scalar{1} : std::min(Scalar{1}, step_mm / block_remaining_mm);
        const auto loss = fraction >= Scalar{1} - Scalar{1.0e-6}
            ? block_remaining_loss_MeV
            : block_remaining_loss_MeV * fraction;
        block_remaining_mm -= step_mm;
        block_remaining_loss_MeV -= loss;
        if (block_remaining_mm <= epsilon) {
            ++block_index;
            block_remaining_mm = Scalar{0};
            block_remaining_loss_MeV = 0;
            block_active = false;
        }
        return std::clamp(loss, Scalar{0}, available_energy_MeV);
    }

    inline void prepare_step(Scalar& step_mm,
                             const Scalar terminal_remaining_mm =
                                 std::numeric_limits<Scalar>::infinity()) noexcept {
        if (sampling_length_mm <= Scalar{0}) return;
        const auto epsilon = std::max(Scalar{1.0e-7}, sampling_length_mm * Scalar{1.0e-6});
        if (block_remaining_mm <= epsilon) {
            block_length_mm = std::min(sampling_length_mm, terminal_remaining_mm);
            block_remaining_mm = block_length_mm;
        }
        step_mm = std::min(step_mm, block_remaining_mm);
    }

    inline void reset_block() noexcept {
        if (block_active || block_remaining_mm > Scalar{0}) ++block_index;
        block_remaining_mm = Scalar{0};
        block_remaining_loss_MeV = Scalar{0};
        block_length_mm = Scalar{0};
        block_active = false;
    }

    inline void consume(const Scalar step_mm) noexcept {
        if (sampling_length_mm <= Scalar{0}) return;
        const auto epsilon = std::max(Scalar{1.0e-7}, sampling_length_mm * Scalar{1.0e-6});
        block_remaining_mm -= step_mm;
        if (block_remaining_mm <= epsilon) {
            ++block_index;
            block_remaining_mm = Scalar{0};
        }
    }
};

// Apply one deterministic Gaussian to a fixed physical sampling block. The
// caller clips the transport step to the remaining block length. For a full
// block, sum(step_fraction) is one and the accumulated variance is exactly the
// block variance, independent of transport-step subdivision.
template <typename Scalar>
inline Scalar step_stable_sampled_energy_loss(
    const Scalar mean_loss_MeV,
    const Scalar step_variance_MeV2,
    const Scalar step_mm,
    const Scalar sampling_length_mm,
    const Scalar gaussian,
    const Scalar available_energy_MeV) noexcept {
    if (step_mm <= Scalar{0} || sampling_length_mm <= Scalar{0}) {
        return std::clamp(mean_loss_MeV, Scalar{0}, available_energy_MeV);
    }
    const auto variance_per_mm = std::max(Scalar{0}, step_variance_MeV2 / step_mm);
    const auto block_sigma = std::sqrt(variance_per_mm * sampling_length_mm);
    const auto fraction = std::clamp(step_mm / sampling_length_mm, Scalar{0}, Scalar{1});
    const auto sampled = mean_loss_MeV + block_sigma * gaussian * fraction;
    return std::clamp(sampled, Scalar{0},
                      std::min(Scalar{2} * mean_loss_MeV, available_energy_MeV));
}

}  // namespace carbon
