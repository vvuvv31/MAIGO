#include "carbon/transport.hpp"
#include "carbon/rng.hpp"
#include "carbon/straggling.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <numbers>
#include <stdexcept>

namespace carbon {

double TransportResult::relative_energy_balance_error() const noexcept {
    if (initial_energy_MeV == 0.0) {
        return 0.0;
    }
    return std::abs(initial_energy_MeV - total_deposited_energy_MeV -
                    escaped_energy_MeV - beamline_removed_energy_MeV -
                    untracked_nuclear_energy_MeV) /
           initial_energy_MeV;
}

double choose_step_mm(double energy_MeV,
                      double stopping_power_MeV_per_mm,
                      double maximum_step_mm,
                      double maximum_relative_energy_loss) {
    if (energy_MeV <= 0.0 || stopping_power_MeV_per_mm <= 0.0 || maximum_step_mm <= 0.0 ||
        maximum_relative_energy_loss <= 0.0) {
        throw std::invalid_argument("choose_step_mm received a nonpositive physical input");
    }
    const auto energy_limited_step =
        maximum_relative_energy_loss * energy_MeV / stopping_power_MeV_per_mm;
    return std::min(maximum_step_mm, energy_limited_step);
}

TransportResult transport_serial(const TransportConfig& config,
                                 const StoppingPowerTable& stopping_power,
                                 const CrossSectionTable& cross_section) {
    config.validate();
    if (config.enable_multiple_scattering) {
        throw std::invalid_argument(
            "Multiple scattering requires the three-dimensional SYCL backend");
    }
    const auto start = std::chrono::steady_clock::now();
    const auto primary_ion = config.primary_ion();
    TransportResult result;
    std::array<double, max_straggling_scale_points> straggling_scale_energies{};
    std::array<double, max_straggling_scale_points> straggling_scale_values{};
    std::array<double, max_straggling_scale_points> primary_xs_correction_energies{};
    std::array<double, max_straggling_scale_points> primary_xs_correction_scales{};
    const auto straggling_scale_point_count =
        config.straggling_scale_energies_MeVu.size();
    for (std::size_t index = 0; index < straggling_scale_point_count; ++index) {
        straggling_scale_energies[index] =
            config.straggling_scale_energies_MeVu[index];
        straggling_scale_values[index] = config.straggling_scale_values[index];
    }
    const auto primary_xs_correction_point_count =
        config.primary_inelastic_xs_correction_energies_MeVu.size();
    for (std::size_t index = 0; index < primary_xs_correction_point_count; ++index) {
        primary_xs_correction_energies[index] =
            config.primary_inelastic_xs_correction_energies_MeVu[index];
        primary_xs_correction_scales[index] =
            config.primary_inelastic_xs_correction_scales[index];
    }
    result.backend = config.enable_energy_straggling ? "serial+straggling" : "serial";
    if (config.enable_primary_attenuation) {
        result.backend += "+attenuation";
        if (config.enable_primary_inelastic_xs_correction) {
            result.backend += "+primary-xs-table";
        } else if (config.primary_inelastic_xs_scale != 1.0) {
            result.backend += "+primary-xs-scale";
        }
    }
    result.deposited_energy_MeV.assign(config.number_of_bins(), 0.0);
    if (config.enable_voxel_scoring) {
        result.voxel_deposited_energy_MeV.assign(config.number_of_voxels(), 0.0);
    }
    const auto center_voxel_offset =
        (config.voxel_bins_y / 2) * config.voxel_bins_x + config.voxel_bins_x / 2;
    const auto voxel_plane_size = config.voxel_bins_x * config.voxel_bins_y;

    struct HistorySummary {
        double escaped_MeV;
        double untracked_nuclear_MeV;
        std::uint64_t steps;
        bool nuclear_interaction;
    };
    const auto simulate_history = [&](std::uint64_t history_id, std::vector<double>& tally,
                                      std::vector<double>& voxel_tally) {
        auto energy_MeV = config.initial_total_energy_MeV();
        if (config.beam_energy_spread > 0.0) {
            const auto uniform1 = std::max(
                static_cast<double>(rng::uniform01(config.random_seed, history_id, 0, 40)),
                1.0e-12);
            const auto uniform2 =
                static_cast<double>(rng::uniform01(config.random_seed, history_id, 0, 41));
            const auto gaussian =
                std::sqrt(-2.0 * std::log(uniform1)) *
                std::cos(2.0 * std::numbers::pi * uniform2);
            energy_MeV *= (1.0 + config.beam_energy_spread * gaussian);
            if (energy_MeV < config.energy_cutoff_MeV) {
                energy_MeV = config.energy_cutoff_MeV;
            }
        }
        const auto history_primary_inelastic_xs_scale =
            interpolate_straggling_scale(
                energy_MeV / static_cast<double>(config.primary_mass_number),
                primary_xs_correction_energies,
                primary_xs_correction_scales,
                primary_xs_correction_point_count,
                config.primary_inelastic_xs_scale);
        auto position_mm = 0.0;
        std::uint64_t steps = 0;
        auto untracked_nuclear_MeV = 0.0;
        auto nuclear_interaction = false;
        const auto attenuation_uniform = std::max(
            static_cast<double>(rng::uniform01(config.random_seed, history_id, 0, 9)),
            1.0e-12);
        auto remaining_interaction_lengths = -std::log(attenuation_uniform);

        while (energy_MeV > config.energy_cutoff_MeV &&
               position_mm < config.phantom_length_mm) {
            const auto energy_MeVu = energy_MeV / static_cast<double>(config.primary_mass_number);
            const auto stopping_power_MeV_per_mm = stopping_power.interpolate(energy_MeVu);
            auto step_mm = choose_step_mm(energy_MeV,
                                          stopping_power_MeV_per_mm,
                                          config.maximum_step_mm,
                                          config.maximum_relative_energy_loss);

            const auto bin =
                std::min(static_cast<std::size_t>(position_mm / config.depth_bin_width_mm),
                         config.number_of_bins() - 1);
            const auto next_bin_boundary_mm =
                static_cast<double>(bin + 1) * config.depth_bin_width_mm;
            step_mm = std::min(step_mm, next_bin_boundary_mm - position_mm);
            step_mm = std::min(step_mm, config.phantom_length_mm - position_mm);
            if (step_mm <= 0.0) {
                throw std::runtime_error("Transport stalled at a depth-bin boundary");
            }

            const auto mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
            auto deposited_MeV = std::min(mean_loss_MeV, energy_MeV);
            if (config.enable_energy_straggling) {
                const auto uniform1 = std::max(
                    static_cast<double>(rng::uniform01(config.random_seed, history_id, steps, 0)),
                    1.0e-12);
                const auto uniform2 =
                    static_cast<double>(rng::uniform01(config.random_seed, history_id, steps, 1));
                const auto gaussian =
                    std::sqrt(-2.0 * std::log(uniform1)) *
                    std::cos(2.0 * std::numbers::pi * uniform2);
                const auto local_scale = interpolate_straggling_scale(
                    energy_MeVu, straggling_scale_energies,
                    straggling_scale_values, straggling_scale_point_count,
                    config.straggling_scale);
                const auto effective_charge = ion_effective_charge(
                    primary_ion.atomic_number, energy_MeVu);
                const auto variance_MeV2 = condensed_total_loss_variance_MeV2(
                    energy_MeVu,
                    primary_ion.rest_mass_MeV,
                    effective_charge, step_mm, config.water_density_g_per_cm3);
                const auto sigma_MeV =
                    local_scale * std::sqrt(std::max(0.0, variance_MeV2));
                deposited_MeV =
                    clamp_sampled_energy_loss(mean_loss_MeV, sigma_MeV, gaussian, energy_MeV);
            }
            tally[bin] += deposited_MeV;
            if (config.enable_voxel_scoring) {
                voxel_tally[bin * voxel_plane_size + center_voxel_offset] += deposited_MeV;
            }
            const auto mid_step_energy_MeVu =
                (energy_MeV - 0.5 * deposited_MeV) /
                static_cast<double>(config.primary_mass_number);
            energy_MeV -= deposited_MeV;
            position_mm += step_mm;
            if (config.enable_primary_attenuation && energy_MeV > config.energy_cutoff_MeV) {
                const auto macroscopic_cross_section_per_mm =
                    cross_section.interpolate(mid_step_energy_MeVu) *
                    history_primary_inelastic_xs_scale;
                remaining_interaction_lengths -=
                    macroscopic_cross_section_per_mm * step_mm;
                if (remaining_interaction_lengths <= 0.0) {
                    untracked_nuclear_MeV = energy_MeV;
                    energy_MeV = 0.0;
                    nuclear_interaction = true;
                }
            }
            ++steps;
        }

        if (energy_MeV > 0.0 && position_mm < config.phantom_length_mm) {
            const auto bin =
                std::min(static_cast<std::size_t>(position_mm / config.depth_bin_width_mm),
                         config.number_of_bins() - 1);
            tally[bin] += energy_MeV;
            if (config.enable_voxel_scoring) {
                voxel_tally[bin * voxel_plane_size + center_voxel_offset] += energy_MeV;
            }
            energy_MeV = 0.0;
        }
        return HistorySummary{energy_MeV, untracked_nuclear_MeV, steps, nuclear_interaction};
    };

    if (config.enable_energy_straggling || config.enable_primary_attenuation) {
        for (std::uint64_t history = 0; history < config.number_of_histories; ++history) {
            const auto summary = simulate_history(
                history, result.deposited_energy_MeV, result.voxel_deposited_energy_MeV);
            result.escaped_energy_MeV += summary.escaped_MeV;
            result.untracked_nuclear_energy_MeV += summary.untracked_nuclear_MeV;
            result.total_steps += summary.steps;
            result.nuclear_interactions += summary.nuclear_interaction ? 1U : 0U;
        }
    } else {
        // Pure CSDA is deterministic, so one trajectory can be scaled exactly.
        std::vector<double> one_history(config.number_of_bins(), 0.0);
        std::vector<double> one_history_voxels;
        if (config.enable_voxel_scoring) {
            one_history_voxels.assign(config.number_of_voxels(), 0.0);
        }
        const auto summary = simulate_history(0, one_history, one_history_voxels);
        const auto history_scale = static_cast<double>(config.number_of_histories);
        std::transform(one_history.begin(), one_history.end(), result.deposited_energy_MeV.begin(),
                       [history_scale](double value) { return value * history_scale; });
        if (config.enable_voxel_scoring) {
            std::transform(one_history_voxels.begin(), one_history_voxels.end(),
                           result.voxel_deposited_energy_MeV.begin(),
                           [history_scale](double value) { return value * history_scale; });
        }
        result.escaped_energy_MeV = summary.escaped_MeV * history_scale;
        result.untracked_nuclear_energy_MeV =
            summary.untracked_nuclear_MeV * history_scale;
        result.total_steps = summary.steps * config.number_of_histories;
    }

    result.initial_energy_MeV = config.initial_total_energy_MeV() *
                                static_cast<double>(config.number_of_histories);
    result.total_deposited_energy_MeV =
        std::accumulate(result.deposited_energy_MeV.begin(), result.deposited_energy_MeV.end(), 0.0);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

}  // namespace carbon
