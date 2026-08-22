#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/minibeam_collimator.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace carbon {

#include "detail/sycl_dose_atomic.inc"

#include "detail/sycl_profile.inc"


#include "detail/sycl_transport_context_impl.inc"

#if !defined(CARBON_ENABLE_MINIBEAM) || defined(CARBON_DEFINE_SYCL_CONTEXT)
#include "detail/sycl_transport_context_methods.inc"
#endif

namespace {

constexpr std::size_t fragment_species_count = 7;

#include "detail/sycl_device_math.inc"

inline float nudge_past_axis_boundary(const float boundary,
                                      const float direction,
                                      const bool robust) noexcept {
    constexpr auto infinity = std::numeric_limits<float>::infinity();
    if (!robust) {
        // Master-compatible snap: one representable float past the face.
        return sycl::nextafter(
            boundary, direction > 0.0F ? infinity : -infinity);
    }
    // One nextafter is not enough when the voxel index is evaluated from
    // (position - origin) / spacing in FP32: cancellation can map several
    // adjacent representable values back to the old voxel.  Move by a tiny
    // physical distance instead (0.01 um, versus the smallest 0.1 mm bin).
    constexpr float boundary_nudge_mm = 1.0e-5F;
    const auto nudged =
        boundary + (direction > 0.0F ? boundary_nudge_mm
                                    : -boundary_nudge_mm);
    if (nudged != boundary) {
        return nudged;
    }
    return sycl::nextafter(
        boundary, direction > 0.0F ? infinity : -infinity);
}



struct MinibeamChargedSurvivor {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    Direction3F direction{};
    bool alive{false};
};

struct MinibeamNeutralSurvivor {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    Direction3F direction{};
    bool alive{false};
};

inline MinibeamNeutralSurvivor transport_minibeam_neutral_product(
    const ReactionSecondary secondary,
    const float reaction_energy_scale,
    const Direction3F parent_direction,
    const float reaction_x_mm,
    const float reaction_y_mm,
    const float reaction_z_mm,
    const float collimator_exit_z_mm,
    const float cosine_angle,
    const float sine_angle,
    const float slit_offset_x_mm,
    const float slit_offset_y_mm,
    const float collimator_radius_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm,
    const float copper_max_step_mm,
    const float* copper_neutral_cross_sections,
    const std::size_t cross_section_grid_size,
    const float minimum_log_energy,
    const float inverse_log_energy_step,
    const NeutralProjectile* copper_neutral_projectiles,
    const std::size_t copper_neutral_projectile_count,
    const NeutralInteraction* copper_neutral_interactions,
    const float energy_cutoff_MeV,
    const std::uint64_t random_seed,
    const std::uint64_t random_stream) noexcept {
    MinibeamNeutralSurvivor result{};
    if ((secondary.pdg_id != 22 && secondary.pdg_id != 2112) ||
        copper_neutral_cross_sections == nullptr ||
        cross_section_grid_size < 2 ||
        copper_neutral_projectiles == nullptr ||
        copper_neutral_projectile_count == 0 ||
        copper_neutral_interactions == nullptr) {
        return result;
    }
    auto energy_MeV =
        secondary.kinetic_energy_MeV * reaction_energy_scale;
    auto direction = rotate_local_direction(
        secondary.direction_x, secondary.direction_y,
        secondary.direction_z, parent_direction);
    if (energy_MeV <= energy_cutoff_MeV ||
        direction.z <= 1.0e-8F) {
        return result;
    }

    const auto species_offset =
        (secondary.pdg_id == 22 ? std::size_t{0} : std::size_t{1}) *
        cross_section_grid_size;
    const auto macroscopic_cross_section =
        [&](const float query_energy_MeV) {
            const auto floating_index =
                (sycl::log(query_energy_MeV) - minimum_log_energy) *
                inverse_log_energy_step;
            auto index = static_cast<int>(sycl::floor(floating_index));
            index = sycl::max(
                0, sycl::min(
                       index, static_cast<int>(cross_section_grid_size) - 2));
            const auto fraction = sycl::clamp(
                floating_index - static_cast<float>(index), 0.0F, 1.0F);
            return copper_neutral_cross_sections[
                       species_offset + static_cast<std::size_t>(index)] +
                   fraction *
                       (copper_neutral_cross_sections[
                            species_offset + static_cast<std::size_t>(index) + 1] -
                        copper_neutral_cross_sections[
                            species_offset + static_cast<std::size_t>(index)]);
        };

    const NeutralProjectile* projectile = nullptr;
    for (std::size_t index = 0;
         index < copper_neutral_projectile_count; ++index) {
        if (copper_neutral_projectiles[index].pdg_id == secondary.pdg_id) {
            projectile = copper_neutral_projectiles + index;
            break;
        }
    }
    if (projectile == nullptr || projectile->interaction_count == 0U) {
        return result;
    }

    auto x_mm = reaction_x_mm;
    auto y_mm = reaction_y_mm;
    auto z_mm = reaction_z_mm;
    auto accumulated_optical_depth = 0.0F;
    auto interaction_optical_depth =
        -sycl::log(sycl::fmax(
            rng::uniform01(random_seed, random_stream, 0, 0),
            1.0e-12F));
    std::uint64_t step = 0;
    constexpr std::uint64_t maximum_steps = 100000;
    constexpr std::uint32_t maximum_interactions = 8;
    std::uint32_t interaction_count = 0;
    while (z_mm < collimator_exit_z_mm - 1.0e-6F &&
           step < maximum_steps) {
        const auto axial_step_mm = sycl::fmin(
            copper_max_step_mm, collimator_exit_z_mm - z_mm);
        const auto path_step_mm = axial_step_mm / direction.z;
        const auto midpoint_x =
            x_mm + 0.5F * path_step_mm * direction.x;
        const auto midpoint_y =
            y_mm + 0.5F * path_step_mm * direction.y;
        const auto in_copper = minibeam_point_in_copper(
            midpoint_x - slit_offset_x_mm,
            midpoint_y - slit_offset_y_mm,
            cosine_angle, sine_angle,
            collimator_radius_mm, slit_count, slit_width_mm,
            slit_pitch_mm, slit_half_length_mm);
        if (in_copper) {
            const auto macroscopic_total_per_mm =
                macroscopic_cross_section(energy_MeV);
            const auto step_optical_depth =
                macroscopic_total_per_mm * path_step_mm;
            if (accumulated_optical_depth + step_optical_depth >=
                interaction_optical_depth) {
                if (interaction_count >= maximum_interactions ||
                    macroscopic_total_per_mm <= 0.0F) {
                    return result;
                }
                const auto interaction_path_mm = sycl::clamp(
                    (interaction_optical_depth - accumulated_optical_depth) /
                        macroscopic_total_per_mm,
                    0.0F, path_step_mm);
                x_mm += interaction_path_mm * direction.x;
                y_mm += interaction_path_mm * direction.y;
                z_mm += interaction_path_mm * direction.z;

                std::uint32_t lower = 0;
                std::uint32_t upper = projectile->interaction_count;
                while (lower < upper) {
                    const auto middle = lower + (upper - lower) / 2U;
                    if (copper_neutral_interactions[
                            projectile->interaction_offset + middle]
                            .incident_energy_MeV < energy_MeV) {
                        lower = middle + 1U;
                    } else {
                        upper = middle;
                    }
                }
                auto nearest = lower;
                if (nearest >= projectile->interaction_count) {
                    nearest = projectile->interaction_count - 1U;
                } else if (nearest > 0U) {
                    const auto lower_delta = sycl::fabs(
                        copper_neutral_interactions[
                            projectile->interaction_offset + nearest - 1U]
                                .incident_energy_MeV -
                        energy_MeV);
                    const auto upper_delta = sycl::fabs(
                        copper_neutral_interactions[
                            projectile->interaction_offset + nearest]
                                .incident_energy_MeV -
                        energy_MeV);
                    if (lower_delta <= upper_delta) {
                        --nearest;
                    }
                }
                constexpr std::uint32_t sampling_window = 8;
                const auto window_begin =
                    nearest > sampling_window / 2U
                        ? nearest - sampling_window / 2U
                        : 0U;
                const auto window_count = sycl::min(
                    sampling_window,
                    projectile->interaction_count - window_begin);
                const auto selected_in_window = sycl::min(
                    static_cast<std::uint32_t>(
                        rng::uniform01(
                            random_seed, random_stream,
                            interaction_count + 1U, 1U) *
                        window_count),
                    window_count - 1U);
                const auto interaction =
                    copper_neutral_interactions[
                        projectile->interaction_offset + window_begin +
                        selected_in_window];
                if (interaction.continuation_energy_MeV <= 0.0F) {
                    return result;
                }
                const auto energy_scale =
                    interaction.incident_energy_MeV > 0.0F
                        ? energy_MeV / interaction.incident_energy_MeV
                        : 1.0F;
                energy_MeV =
                    interaction.continuation_energy_MeV * energy_scale;
                direction = rotate_local_direction(
                    interaction.continuation_direction_x,
                    interaction.continuation_direction_y,
                    interaction.continuation_direction_z,
                    direction);
                ++interaction_count;
                if (energy_MeV <= energy_cutoff_MeV ||
                    direction.z <= 1.0e-8F) {
                    return result;
                }
                accumulated_optical_depth = 0.0F;
                interaction_optical_depth =
                    -sycl::log(sycl::fmax(
                        rng::uniform01(
                            random_seed, random_stream,
                            interaction_count, 2U),
                        1.0e-12F));
                ++step;
                continue;
            }
            accumulated_optical_depth += step_optical_depth;
        }
        x_mm += path_step_mm * direction.x;
        y_mm += path_step_mm * direction.y;
        z_mm += axial_step_mm;
        ++step;
    }
    if (step == maximum_steps) {
        return result;
    }

    const auto distance_to_water = -z_mm / direction.z;
    result.position_x_mm = x_mm + distance_to_water * direction.x;
    result.position_y_mm = y_mm + distance_to_water * direction.y;
    result.kinetic_energy_MeV = energy_MeV;
    result.direction = direction;
    result.alive = true;
    return result;
}

inline constexpr std::size_t minibeam_charged_species_count = 9;

inline constexpr std::size_t minibeam_charged_species_category(
    const int atomic_number, const int mass_number) noexcept {
    if (atomic_number == 6) return 0;
    if (atomic_number == 5) return 1;
    if (atomic_number == 4) return 2;
    if (atomic_number == 3) return 3;
    if (atomic_number == 2) return 4;
    if (atomic_number == 1 && mass_number == 1) return 5;
    if (atomic_number == 1 && mass_number == 2) return 6;
    if (atomic_number == 1 && mass_number == 3) return 7;
    return 8;
}

inline float minibeam_species_stopping_power(
    const float* carbon_table,
    const float* species_ratio_table,
    const std::uint8_t* species_present,
    const float kinetic_energy_MeV,
    const int atomic_number,
    const int mass_number,
    const float minimum_table_energy,
    const float inverse_table_step,
    const std::size_t table_size) noexcept {
    if (carbon_table == nullptr || atomic_number <= 0 ||
        mass_number <= 0 || table_size < 2) {
        return 0.0F;
    }
    const auto energy_MeV_per_u =
        kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto floating_index =
        (energy_MeV_per_u - minimum_table_energy) *
        inverse_table_step;
    auto index = static_cast<int>(sycl::floor(floating_index));
    index = sycl::max(
        0, sycl::min(index, static_cast<int>(table_size) - 2));
    const auto fraction = sycl::clamp(
        floating_index - static_cast<float>(index), 0.0F, 1.0F);
    const auto carbon_stopping =
        carbon_table[index] +
        fraction * (carbon_table[index + 1] - carbon_table[index]);
    const auto species =
        static_cast<std::size_t>(atomic_number) *
            IonStoppingPowerTables::mass_stride +
        static_cast<std::size_t>(mass_number);
    if (species_ratio_table != nullptr &&
        species_present != nullptr &&
        atomic_number <
            static_cast<int>(
                IonStoppingPowerTables::atomic_number_slots) &&
        mass_number <
            static_cast<int>(IonStoppingPowerTables::mass_stride) &&
        species_present[species] != 0) {
        const auto base = species * table_size;
        const auto ratio =
            species_ratio_table[base + static_cast<std::size_t>(index)] +
            fraction *
                (species_ratio_table[
                     base + static_cast<std::size_t>(index) + 1] -
                 species_ratio_table[
                     base + static_cast<std::size_t>(index)]);
        return carbon_stopping * ratio;
    }
    const auto charge = static_cast<float>(atomic_number);
    return carbon_stopping * charge * charge / 36.0F;
}

inline MinibeamChargedSurvivor transport_minibeam_charged_product(
    const ReactionSecondary secondary,
    const float reaction_energy_scale,
    const Direction3F parent_direction,
    const float reaction_x_mm,
    const float reaction_y_mm,
    const float reaction_z_mm,
    const float collimator_exit_z_mm,
    const float cosine_angle,
    const float sine_angle,
    const float slit_offset_x_mm,
    const float slit_offset_y_mm,
    const float collimator_radius_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm,
    const float copper_max_step_mm,
    const float copper_density_g_per_cm3,
    const float copper_radiation_length_g_per_cm2,
    const float copper_mcs_scale,
    const float* copper_stopping_power,
    const float* air_stopping_power,
    const float* copper_species_stopping_ratio,
    const std::uint8_t* copper_species_stopping_present,
    const float* copper_species_cross_section,
    const std::uint8_t* copper_species_cross_section_present,
    const std::size_t copper_cross_section_grid_size,
    const float copper_cross_section_minimum_energy,
    const float copper_cross_section_inverse_step,
    const float minimum_table_energy,
    const float inverse_table_step,
    const std::size_t table_size,
    const float energy_cutoff_MeV,
    const std::uint64_t random_seed,
    const std::uint64_t random_stream) noexcept {
    MinibeamChargedSurvivor result{};
    if (secondary.atomic_number <= 0 || secondary.mass_number <= 0) {
        return result;
    }
    auto energy_MeV =
        secondary.kinetic_energy_MeV * reaction_energy_scale;
    auto x_mm = reaction_x_mm;
    auto y_mm = reaction_y_mm;
    auto z_mm = reaction_z_mm;
    auto direction = rotate_local_direction(
        secondary.direction_x, secondary.direction_y,
        secondary.direction_z, parent_direction);
    if (energy_MeV <= energy_cutoff_MeV ||
        direction.z <= 1.0e-8F) {
        return result;
    }

    auto copper_segment_path_mm = 0.0F;
    std::uint64_t step = 0;
    constexpr std::uint64_t maximum_steps = 100000;
    while (z_mm < collimator_exit_z_mm - 1.0e-6F &&
           step < maximum_steps) {
        if (direction.z <= 1.0e-8F ||
            energy_MeV <= energy_cutoff_MeV) {
            return result;
        }
        const auto axial_step_mm = sycl::fmin(
            copper_max_step_mm, collimator_exit_z_mm - z_mm);
        const auto path_step_mm = axial_step_mm / direction.z;
        const auto midpoint_x =
            x_mm + 0.5F * path_step_mm * direction.x;
        const auto midpoint_y =
            y_mm + 0.5F * path_step_mm * direction.y;
        const auto in_copper = minibeam_point_in_copper(
            midpoint_x - slit_offset_x_mm,
            midpoint_y - slit_offset_y_mm,
            cosine_angle, sine_angle,
            collimator_radius_mm, slit_count, slit_width_mm,
            slit_pitch_mm, slit_half_length_mm);
        x_mm += path_step_mm * direction.x;
        y_mm += path_step_mm * direction.y;
        z_mm += axial_step_mm;
        if (!in_copper) {
            copper_segment_path_mm = 0.0F;
            ++step;
            continue;
        }

        const auto stopping_power = minibeam_species_stopping_power(
            copper_stopping_power, copper_species_stopping_ratio,
            copper_species_stopping_present, energy_MeV,
            secondary.atomic_number, secondary.mass_number,
            minimum_table_energy, inverse_table_step, table_size);
        const auto mean_loss_MeV = stopping_power * path_step_mm;
        if (mean_loss_MeV >= energy_MeV - energy_cutoff_MeV) {
            return result;
        }
        const auto scattering_energy_MeV =
            energy_MeV - 0.5F * mean_loss_MeV;
        energy_MeV -= mean_loss_MeV;
        const auto species =
            static_cast<std::size_t>(secondary.atomic_number) *
                IonCrossSectionTables::mass_stride +
            static_cast<std::size_t>(secondary.mass_number);
        if (copper_species_cross_section != nullptr &&
            copper_species_cross_section_present != nullptr &&
            copper_cross_section_grid_size >= 2 &&
            secondary.atomic_number <
                static_cast<int>(
                    IonCrossSectionTables::atomic_number_slots) &&
            secondary.mass_number <
                static_cast<int>(
                    IonCrossSectionTables::mass_stride) &&
            copper_species_cross_section_present[species] != 0) {
            const auto energy_MeV_per_u =
                scattering_energy_MeV /
                static_cast<float>(secondary.mass_number);
            const auto floating_xs_index =
                (energy_MeV_per_u -
                 copper_cross_section_minimum_energy) *
                copper_cross_section_inverse_step;
            auto xs_index =
                static_cast<int>(sycl::floor(floating_xs_index));
            xs_index = sycl::max(
                0, sycl::min(
                       xs_index,
                       static_cast<int>(
                           copper_cross_section_grid_size) -
                           2));
            const auto xs_fraction = sycl::clamp(
                floating_xs_index -
                    static_cast<float>(xs_index),
                0.0F, 1.0F);
            const auto base =
                species * copper_cross_section_grid_size;
            const auto macroscopic_xs =
                copper_species_cross_section[
                    base + static_cast<std::size_t>(xs_index)] +
                xs_fraction *
                    (copper_species_cross_section[
                         base + static_cast<std::size_t>(xs_index) + 1] -
                     copper_species_cross_section[
                         base + static_cast<std::size_t>(xs_index)]);
            const auto interaction_probability =
                1.0F - sycl::exp(-macroscopic_xs * path_step_mm);
            if (rng::uniform01(
                    random_seed, random_stream, step, 2) <
                interaction_probability) {
                return result;
            }
        }
        const auto previous_path_mm = copper_segment_path_mm;
        copper_segment_path_mm += path_step_mm;
        const auto total_rms = highland_projected_rms_angle_device(
            scattering_energy_MeV, secondary.atomic_number,
            secondary.mass_number, copper_segment_path_mm,
            copper_density_g_per_cm3,
            copper_radiation_length_g_per_cm2);
        const auto previous_rms =
            highland_projected_rms_angle_device(
                scattering_energy_MeV, secondary.atomic_number,
                secondary.mass_number, previous_path_mm,
                copper_density_g_per_cm3,
                copper_radiation_length_g_per_cm2);
        const auto incremental_rms = copper_mcs_scale *
            sycl::sqrt(sycl::fmax(
                0.0F, total_rms * total_rms -
                          previous_rms * previous_rms));
        direction = scatter_direction(
            direction, incremental_rms, random_seed, random_stream,
            step, 0);
        ++step;
    }
    if (step == maximum_steps || direction.z <= 1.0e-8F) {
        return result;
    }

    const auto air_path_mm = -z_mm / direction.z;
    if (air_path_mm < 0.0F) {
        return result;
    }
    const auto air_stopping = minibeam_species_stopping_power(
        air_stopping_power, nullptr, nullptr, energy_MeV,
        secondary.atomic_number,
        secondary.mass_number, minimum_table_energy,
        inverse_table_step, table_size);
    const auto air_loss_MeV = air_stopping * air_path_mm;
    if (air_loss_MeV >= energy_MeV - energy_cutoff_MeV) {
        return result;
    }
    energy_MeV -= air_loss_MeV;
    x_mm += air_path_mm * direction.x;
    y_mm += air_path_mm * direction.y;
    result.position_x_mm = x_mm;
    result.position_y_mm = y_mm;
    result.kinetic_energy_MeV = energy_MeV;
    result.direction = direction;
    result.alive = true;
    return result;
}


#include "detail/sycl_cascade_select.inc"
#include "detail/sycl_cascade_host_lut.inc"


inline CascadeSelectionResult select_binned_energy_cascade_interaction(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u,
    const float u01,
    const bool allow_nearest_fallback) noexcept {
    if (count == 0U) {
        return no_cascade_energy_coverage(
            interactions, offset, count, energy_MeV_per_u);
    }
    const auto target_energy_bin =
        cascade_condition_energy_bin(energy_MeV_per_u);
    // Keep parent-energy conditioning tight: ±5 × 2 MeV/u = ±10 MeV/u.
    constexpr std::uint32_t max_energy_radius = 5U;
    for (std::uint32_t radius = 0; radius <= max_energy_radius; ++radius) {
        std::uint32_t range_begin[2]{};
        std::uint32_t range_count[2]{};
        std::uint32_t range_total = 0;
        std::uint32_t total_candidates = 0;
        for (std::uint32_t side = 0; side < 2U; ++side) {
            if (radius == 0U && side == 1U) {
                continue;
            }
            if (side == 0U && radius > target_energy_bin) {
                continue;
            }
            const auto energy_bin =
                side == 0U ? target_energy_bin - radius
                           : target_energy_bin + radius;
            const auto begin = cascade_condition_cell_lower_bound(
                interactions, offset, count, energy_bin, 0U);
            const auto end = cascade_condition_cell_lower_bound(
                interactions, offset, count, energy_bin + 1U, 0U);
            if (end > begin) {
                range_begin[range_total] = begin;
                range_count[range_total] = end - begin;
                total_candidates += end - begin;
                ++range_total;
            }
        }
        if (total_candidates == 0U) {
            continue;
        }
        auto pick = sycl::min(
            static_cast<std::uint32_t>(
                u01 * static_cast<float>(total_candidates)),
            total_candidates - 1U);
        for (std::uint32_t range = 0; range < range_total; ++range) {
            if (pick < range_count[range]) {
                return make_cascade_selection_result(
                    interactions, offset, range_begin[range] + pick,
                    energy_MeV_per_u,
                    radius == 0U ? CascadeSelectionStatus::exact_cell
                                 : CascadeSelectionStatus::expanded_window);
            }
            pick -= range_count[range];
        }
    }
    // Never fall back to energy-unconditioned random full-table sampling.
    if (!allow_nearest_fallback) {
        return no_cascade_energy_coverage(
            interactions, offset, count, energy_MeV_per_u);
    }
    return make_cascade_selection_result(
        interactions, offset,
        nearest_cascade_interaction(interactions, offset, count,
                                    energy_MeV_per_u),
        energy_MeV_per_u, CascadeSelectionStatus::nearest_fallback);
}

inline CascadeSelectionResult select_cascade_interaction_conditioned(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u,
    const float depth_mm,
    const float u01,
    const bool binned_depth_layout,
    const bool condition_on_reference_depth,
    const bool allow_nearest_fallback) noexcept {
    if (!binned_depth_layout) {
        return select_legacy_cascade_interaction_energy_conditioned(
            interactions, offset, count, energy_MeV_per_u, u01,
            allow_nearest_fallback);
    }
    if (condition_on_reference_depth) {
        return select_depth_conditioned_cascade_interaction(
            interactions, offset, count, energy_MeV_per_u, depth_mm, u01,
            allow_nearest_fallback);
    }
    return select_binned_energy_cascade_interaction(
        interactions, offset, count, energy_MeV_per_u, u01,
        allow_nearest_fallback);
}

#include "detail/sycl_score_device.inc"

// Smoothly limit a minibeam-only low-energy MCS correction to the distal
// transport regime.  A smoothstep avoids an angular-distribution discontinuity
// at the configured transition energy.
inline float minibeam_water_low_energy_mcs_scale(
    const float kinetic_energy_MeV,
    const int mass_number,
    const float transition_MeVu,
    const float scale_at_zero) noexcept {
    if (mass_number <= 0 || transition_MeVu <= 0.0F ||
        scale_at_zero == 1.0F) {
        return 1.0F;
    }
    const auto energy_MeVu =
        sycl::fmax(0.0F, kinetic_energy_MeV /
                             static_cast<float>(mass_number));
    const auto x = sycl::clamp(energy_MeVu / transition_MeVu, 0.0F, 1.0F);
    const auto smooth = x * x * (3.0F - 2.0F * x);
    return scale_at_zero + (1.0F - scale_at_zero) * smooth;
}

// Select one voxel for an aggregated continuous-loss deposit. With probability
// delayed/total, move the entire aggregate by a 2-D Gaussian displacement.
// This is an unbiased estimator of splitting every step into local and lateral
// delta-electron dose, without multiplying the dominant voxel atomic traffic.
inline std::size_t electronic_voxel_target(
    const std::size_t production_voxel,
    const float total_MeV,
    const float delayed_MeV,
    const float longitudinal_mfp_mm,
    const float lateral_sigma_mm,
    const float depth_bin_width_mm,
    const std::size_t depth_bin_count,
    const float voxel_size_x_mm,
    const float voxel_size_y_mm,
    const std::size_t voxel_bins_x,
    const std::size_t voxel_bins_y,
    const float choice_uniform,
    const float exponential_uniform,
    const float gaussian_uniform_1,
    const float gaussian_uniform_2) noexcept {
    if (total_MeV <= 0.0F || delayed_MeV <= 0.0F ||
        voxel_bins_x == 0 || voxel_bins_y == 0 ||
        depth_bin_count == 0 || depth_bin_width_mm <= 0.0F ||
        voxel_size_x_mm <= 0.0F ||
        voxel_size_y_mm <= 0.0F ||
        choice_uniform >= sycl::fmin(1.0F, delayed_MeV / total_MeV)) {
        return production_voxel;
    }
    const auto plane_size = voxel_bins_x * voxel_bins_y;
    const auto voxel_in_plane = production_voxel % plane_size;
    const auto original_x = static_cast<int>(voxel_in_plane % voxel_bins_x);
    const auto original_y = static_cast<int>(voxel_in_plane / voxel_bins_x);
    const auto original_z = production_voxel / plane_size;
    auto target_z = original_z;
    if (longitudinal_mfp_mm > 1.0e-3F) {
        const auto flight_mm =
            -longitudinal_mfp_mm *
            sycl::log(sycl::fmax(exponential_uniform, 1.0e-12F));
        target_z += static_cast<std::size_t>(
            sycl::floor(flight_mm / depth_bin_width_mm));
        if (target_z >= depth_bin_count) {
            return static_cast<std::size_t>(-1);
        }
    }
    auto dx_mm = 0.0F;
    auto dy_mm = 0.0F;
    if (lateral_sigma_mm > 0.0F) {
        constexpr float two_pi = 6.2831853071795864769F;
        const auto radius_standard_normal = sycl::sqrt(
            -2.0F * sycl::log(sycl::fmax(gaussian_uniform_1, 1.0e-12F)));
        dx_mm = lateral_sigma_mm * radius_standard_normal *
                sycl::cos(two_pi * gaussian_uniform_2);
        dy_mm = lateral_sigma_mm * radius_standard_normal *
                sycl::sin(two_pi * gaussian_uniform_2);
    }
    const auto offset_x =
        static_cast<int>(sycl::floor(dx_mm / voxel_size_x_mm + 0.5F));
    const auto offset_y =
        static_cast<int>(sycl::floor(dy_mm / voxel_size_y_mm + 0.5F));
    const auto target_x = original_x + offset_x;
    const auto target_y = original_y + offset_y;
    if (target_x < 0 || target_x >= static_cast<int>(voxel_bins_x) ||
        target_y < 0 || target_y >= static_cast<int>(voxel_bins_y)) {
        return static_cast<std::size_t>(-1);
    }
    return target_z * plane_size +
           static_cast<std::size_t>(target_y) * voxel_bins_x +
           static_cast<std::size_t>(target_x);
}


// Local nuclear residual / unsupported-product heat at the interaction point
// (heavy residual nucleus thermalization approximation, matches G4 local deposit).
inline void deposit_local_heat_device(
    const float residual_MeV,
    const float position_x_mm,
    const float position_y_mm,
    const float position_z_mm,
    const float direction_x,
    const float direction_y,
    const float direction_z,
    const float depth_bin_width_mm,
    const std::size_t number_of_bins,
    const bool enable_voxel_scoring,
    const float voxel_min_x_mm,
    const float voxel_min_y_mm,
    const float voxel_size_x_mm,
    const float voxel_size_y_mm,
    const std::size_t voxel_bins_x,
    const std::size_t voxel_bins_y,
    const std::size_t voxel_plane_size,
    DoseAtomicT* dose_device,
    DoseAtomicT* fragment_dose_device,
    const std::size_t species_index,
    DoseAtomicT* voxel_dose_device,
    const bool enable_charged_origin_voxel_scoring,
    DoseAtomicT* charged_origin_voxel_dose_device,
    const std::size_t charged_origin_voxel_offset) noexcept {
    if (residual_MeV <= 0.0F) {
        return;
    }
    auto rbin = direction_z < 0.0F
                    ? static_cast<int>(sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                    : static_cast<int>(sycl::floor(position_z_mm / depth_bin_width_mm));
    rbin = sycl::max(0, sycl::min(rbin, static_cast<int>(number_of_bins) - 1));
    const auto amount = static_cast<DoseAtomicT>(residual_MeV);
    if (dose_device != nullptr) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_r(dose_device[static_cast<std::size_t>(rbin)]);
        atomic_r.fetch_add(amount);
    }
    if (fragment_dose_device != nullptr) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_f(fragment_dose_device[species_index * number_of_bins +
                                          static_cast<std::size_t>(rbin)]);
        atomic_f.fetch_add(amount);
    }
    if (enable_voxel_scoring && voxel_dose_device != nullptr) {
        const auto x_coordinate = (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
        const auto y_coordinate = (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
        auto voxel_x = direction_x < 0.0F ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(x_coordinate));
        auto voxel_y = direction_y < 0.0F ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(y_coordinate));
        voxel_x = sycl::max(0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
        voxel_y = sycl::max(0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
        const auto vidx = static_cast<std::size_t>(rbin) * voxel_plane_size +
                          static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                          static_cast<std::size_t>(voxel_x);
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_v(voxel_dose_device[vidx]);
        atomic_v.fetch_add(amount);
        if (enable_charged_origin_voxel_scoring &&
            charged_origin_voxel_dose_device != nullptr) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cat(
                    charged_origin_voxel_dose_device[charged_origin_voxel_offset + vidx]);
            atomic_cat.fetch_add(amount);
        }
    }
}


inline void score_secondary_uniform_z_segment_device(
    const DoseAtomicT amount_MeV,
    const float start_z_mm,
    const float direction_z,
    const float path_step_mm,
    const bool is_neutral_lineage,
    const std::size_t species_index,
    const std::size_t neutral_origin,
    const float depth_bin_width_mm,
    const std::size_t number_of_bins,
    const bool enable_voxel_scoring,
    const bool enable_charged_origin_voxel_scoring,
    const std::size_t transverse_voxel_index,
    const std::size_t voxel_plane_size,
    const std::size_t charged_origin_voxel_offset,
    const std::size_t neutral_origin_voxel_offset,
    DoseAtomicT* aggregate_secondary_dose_device,
    DoseAtomicT* fragment_dose_device,
    DoseAtomicT* voxel_dose_device,
    DoseAtomicT* charged_origin_voxel_dose_device,
    DoseAtomicT* neutral_origin_dose_device,
    DoseAtomicT* neutral_origin_voxel_dose_device) noexcept {
    if (amount_MeV <= DoseAtomicT{0} || path_step_mm <= 0.0F) {
        return;
    }
    const auto end_z_mm = start_z_mm + direction_z * path_step_mm;
    const auto delta_z_mm = end_z_mm - start_z_mm;
    if (sycl::fabs(delta_z_mm) < 1.0e-7F) {
        auto bin = static_cast<int>(
            sycl::floor(start_z_mm / depth_bin_width_mm));
        bin = sycl::max(
            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
        const auto voxel_index =
            static_cast<std::size_t>(bin) * voxel_plane_size +
            transverse_voxel_index;
        score_secondary_dose_device(
            amount_MeV, is_neutral_lineage, species_index, neutral_origin,
            bin, number_of_bins, enable_voxel_scoring,
            enable_charged_origin_voxel_scoring, voxel_index,
            charged_origin_voxel_offset, neutral_origin_voxel_offset,
            aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
            charged_origin_voxel_dose_device, neutral_origin_dose_device,
            neutral_origin_voxel_dose_device);
        return;
    }

    const auto low_z_mm = sycl::fmin(start_z_mm, end_z_mm);
    const auto high_z_mm = sycl::fmax(start_z_mm, end_z_mm);
    auto first_bin = static_cast<int>(
        sycl::floor(low_z_mm / depth_bin_width_mm));
    auto last_bin = static_cast<int>(
        sycl::floor(
            sycl::nextafter(high_z_mm, low_z_mm) / depth_bin_width_mm));
    first_bin = sycl::max(
        0, sycl::min(first_bin, static_cast<int>(number_of_bins) - 1));
    last_bin = sycl::max(
        0, sycl::min(last_bin, static_cast<int>(number_of_bins) - 1));
    const auto inverse_delta_z = 1.0F / sycl::fabs(delta_z_mm);
    auto scored_MeV = DoseAtomicT{0};
    for (auto bin = first_bin; bin <= last_bin; ++bin) {
        const auto bin_low =
            static_cast<float>(bin) * depth_bin_width_mm;
        const auto bin_high = bin_low + depth_bin_width_mm;
        const auto overlap_z = sycl::fmax(
            0.0F,
            sycl::fmin(high_z_mm, bin_high) -
                sycl::fmax(low_z_mm, bin_low));
        auto bin_amount = static_cast<DoseAtomicT>(
            amount_MeV * static_cast<DoseAtomicT>(
                             overlap_z * inverse_delta_z));
        if (bin == last_bin) {
            bin_amount = amount_MeV - scored_MeV;
        }
        scored_MeV += bin_amount;
        const auto voxel_index =
            static_cast<std::size_t>(bin) * voxel_plane_size +
            transverse_voxel_index;
        score_secondary_dose_device(
            bin_amount, is_neutral_lineage, species_index, neutral_origin,
            bin, number_of_bins, enable_voxel_scoring,
            enable_charged_origin_voxel_scoring, voxel_index,
            charged_origin_voxel_offset, neutral_origin_voxel_offset,
            aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
            charged_origin_voxel_dose_device, neutral_origin_dose_device,
            neutral_origin_voxel_dose_device);
    }
}

}  // namespace

TransportResult transport_sycl_minibeam(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages,
                               const CascadePackageTable* cascade_packages,
                               const NeutralPackageTable* neutral_packages,
                               SyclTransportContext* context,
                               const CrossSectionTable* elastic_cross_section,
                               const ElasticPackageTable* elastic_packages) {
    (void)elastic_cross_section;
    (void)elastic_packages;
    if (config.enable_primary_elastic_interactions) {
        throw std::invalid_argument(
            "primary elastic interactions are supported only by the legacy SYCL path");
    }
    config.validate();
    if (!is_uniform_grid(stopping_power.energies())) {
        throw std::invalid_argument("The current SYCL backend requires a uniform stopping-power grid");
    }
    if (!is_uniform_grid(cross_section.energies())) {
        throw std::invalid_argument("The current SYCL backend requires a uniform cross-section grid");
    }
    if (config.enable_secondary_generation && !config.enable_primary_attenuation) {
        throw std::invalid_argument(
            "Secondary generation requires enable_primary_attenuation=true");
    }
    if (config.enable_secondary_generation && reaction_packages == nullptr) {
        throw std::invalid_argument("Secondary generation requires a reaction package table");
    }
    if (config.enable_fragment_cascade && cascade_packages == nullptr) {
        throw std::invalid_argument("Fragment cascade requires a cascade package table");
    }
    if (config.cascade_condition_on_reference_depth &&
        (cascade_packages == nullptr || cascade_packages->interactions().empty() ||
         !std::isfinite(cascade_packages->interactions().front().depth_mm))) {
        throw std::invalid_argument(
            "cascade_condition_on_reference_depth requires a v3 cascade package");
    }
    if (config.enable_neutral_transport && neutral_packages == nullptr) {
        throw std::invalid_argument("Neutral transport requires a neutral package table");
    }

    if (context != nullptr && context->impl_->device_name != device_name) {
        throw std::invalid_argument(
            "SyclTransportContext device does not match transport device");
    }
    auto queue = context != nullptr ? context->impl_->queue : make_sycl_queue(device_name);
    const auto reuse_immutable_buffers = context != nullptr;
    const auto start = std::chrono::steady_clock::now();
    auto primary_kernel_seconds = 0.0;
    auto secondary_kernel_seconds = 0.0;
    auto neutral_kernel_seconds = 0.0;
    auto charged_after_neutral_kernel_seconds = 0.0;
    const auto table_size = stopping_power.values().size();
    const auto cross_section_table_size = cross_section.values().size();
    std::optional<CascadePackageTable> ct_lung_cascade_xs_packages;
    std::optional<CascadePackageTable> ct_bone_cascade_xs_packages;
    if (config.enable_ct_grid && config.enable_fragment_cascade) {
        if (!config.ct_lung_cascade_cross_section_package_file.empty()) {
            ct_lung_cascade_xs_packages =
                CascadePackageTable::from_binary(
                    config.ct_lung_cascade_cross_section_package_file);
        }
        if (!config.ct_bone_cascade_cross_section_package_file.empty()) {
            ct_bone_cascade_xs_packages =
                CascadePackageTable::from_binary(
                    config.ct_bone_cascade_cross_section_package_file);
        }
    }
    const auto use_ct_lung_cascade_xs =
        ct_lung_cascade_xs_packages.has_value();
    const auto use_ct_bone_cascade_xs =
        ct_bone_cascade_xs_packages.has_value();
    const auto cascade_xs_material_count =
        use_ct_lung_cascade_xs || use_ct_bone_cascade_xs
            ? std::size_t{3}
            : std::size_t{1};
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_histories = config.number_of_histories;
    const auto primary_spot_count = config.primary_spot_batch.size();
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto voxel_scorer_clamps_transport =
        config.voxel_scorer_clamps_transport;
    const auto enable_charged_origin_voxel_scoring =
        config.enable_charged_origin_voxel_scoring;
    const auto number_of_voxels =
        enable_voxel_scoring ? config.number_of_voxels() : std::size_t{0};
    const auto voxel_plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    // Default: origin-centered scorer (homogeneous / water phantoms).
    // Extent is always [min, min + n * size); do not use max = -min (wrong for
    // odd bin counts when later rebased onto a CT origin).
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    const auto enable_secondary_generation = config.enable_secondary_generation;
    const auto enable_secondary_transport = config.enable_secondary_transport;
    const auto enable_neutral_transport = config.enable_neutral_transport;
    const auto enable_secondary_energy_sorting =
        config.enable_secondary_energy_sorting && enable_secondary_transport;
    const auto enable_fragment_cascade = config.enable_fragment_cascade;
    const auto enable_fragment_species_scoring =
        config.enable_fragment_species_scoring && enable_secondary_transport;
    const auto enable_minibeam_copper_em =
        config.enable_minibeam &&
        config.minibeam_transport_mode == "copper_em";
    const auto enable_minibeam_copper_nuclear_attenuation =
        enable_minibeam_copper_em &&
        config.minibeam_copper_enable_nuclear_attenuation;
    const auto enable_minibeam_copper_reaction_products =
        enable_minibeam_copper_nuclear_attenuation &&
        config.minibeam_copper_enable_reaction_products;
    std::optional<ReactionPackageTable> minibeam_copper_reaction_packages;
    if (enable_minibeam_copper_reaction_products) {
        minibeam_copper_reaction_packages =
            ReactionPackageTable::from_binary(
                config.minibeam_copper_reaction_package_file);
    }
    std::optional<NeutralPackageTable> minibeam_copper_neutral_packages;
    if (enable_minibeam_copper_reaction_products &&
        enable_neutral_transport) {
        minibeam_copper_neutral_packages =
            NeutralPackageTable::from_binary(
                config.minibeam_copper_neutral_package_file);
    }
    const auto minibeam_copper_neutral_projectile_count =
        minibeam_copper_neutral_packages.has_value()
            ? minibeam_copper_neutral_packages->projectiles().size()
            : std::size_t{0};
    std::vector<float> minibeam_copper_sp_host;
    std::vector<float> minibeam_air_sp_host;
    std::vector<float> minibeam_copper_xs_host;
    std::vector<float> minibeam_copper_ion_sp_ratio_host;
    std::vector<std::uint8_t>
        minibeam_copper_ion_sp_present_host;
    std::vector<float> minibeam_copper_ion_xs_host;
    std::vector<std::uint8_t>
        minibeam_copper_ion_xs_present_host;
    std::size_t minibeam_copper_ion_xs_grid_size = 0;
    float minibeam_copper_ion_xs_minimum_energy = 0.0F;
    float minibeam_copper_ion_xs_inverse_step = 0.0F;
    std::vector<float> minibeam_copper_neutral_xs_host;
    std::size_t minibeam_copper_neutral_xs_grid_size = 0;
    float minibeam_copper_neutral_xs_minimum_log_energy = 0.0F;
    float minibeam_copper_neutral_xs_inverse_log_step = 0.0F;
    if (enable_minibeam_copper_em) {
        const auto load_matching_table =
            [&](const std::filesystem::path& path,
                const char* material) {
                const auto material_sp = StoppingPowerTable::from_csv(path);
                if (material_sp.values().size() != table_size ||
                    !is_uniform_grid(material_sp.energies())) {
                    throw std::invalid_argument(
                        std::string("Minibeam ") + material +
                        " stopping-power table must match the water grid");
                }
                std::vector<float> result(table_size);
                for (std::size_t index = 0; index < table_size; ++index) {
                    if (std::abs(material_sp.energies()[index] -
                                 stopping_power.energies()[index]) >
                        1.0e-9) {
                        throw std::invalid_argument(
                            std::string("Minibeam ") + material +
                            " stopping-power energies differ from water grid");
                    }
                    result[index] =
                        static_cast<float>(material_sp.values()[index]);
                }
                return result;
            };
        minibeam_copper_sp_host = load_matching_table(
            config.minibeam_copper_stopping_power_file, "Copper");
        minibeam_air_sp_host = load_matching_table(
            config.minibeam_air_stopping_power_file, "Air");
        if (enable_minibeam_copper_nuclear_attenuation) {
            const auto copper_xs = CrossSectionTable::from_csv(
                config.minibeam_copper_cross_section_file);
            if (copper_xs.values().size() != cross_section_table_size ||
                !is_uniform_grid(copper_xs.energies())) {
                throw std::invalid_argument(
                    "Minibeam Copper XS table must match the water XS grid");
            }
            minibeam_copper_xs_host.resize(cross_section_table_size);
            for (std::size_t index = 0; index < cross_section_table_size;
                 ++index) {
                if (std::abs(copper_xs.energies()[index] -
                             cross_section.energies()[index]) >
                    1.0e-9) {
                    throw std::invalid_argument(
                        "Minibeam Copper XS energies differ from water grid");
                }
                minibeam_copper_xs_host[index] =
                    static_cast<float>(copper_xs.values()[index]);
            }
        }
        if (enable_minibeam_copper_reaction_products) {
            const auto copper_carbon =
                StoppingPowerTable::from_csv(
                    config.minibeam_copper_stopping_power_file);
            const auto ion_stopping =
                IonStoppingPowerTables::from_csv(
                    config.minibeam_copper_ion_stopping_power_file,
                    copper_carbon);
            minibeam_copper_ion_sp_ratio_host =
                ion_stopping.ratios_to_carbon();
            minibeam_copper_ion_sp_present_host =
                ion_stopping.species_present();
            const auto ion_cross_sections =
                IonCrossSectionTables::from_csv(
                    config.minibeam_copper_ion_cross_section_file);
            minibeam_copper_ion_xs_host =
                ion_cross_sections.values();
            minibeam_copper_ion_xs_present_host =
                ion_cross_sections.species_present();
            minibeam_copper_ion_xs_grid_size =
                ion_cross_sections.energy_grid_size();
            minibeam_copper_ion_xs_minimum_energy =
                ion_cross_sections.minimum_energy_MeVu();
            minibeam_copper_ion_xs_inverse_step =
                1.0F / ion_cross_sections.energy_step_MeVu();
            if (enable_neutral_transport) {
                const auto neutral_cross_sections =
                    NeutralCrossSectionTables::from_csv(
                        config.minibeam_copper_neutral_cross_section_file);
                minibeam_copper_neutral_xs_host =
                    neutral_cross_sections.values();
                minibeam_copper_neutral_xs_grid_size =
                    neutral_cross_sections.energy_grid_size();
                minibeam_copper_neutral_xs_minimum_log_energy =
                    neutral_cross_sections.minimum_log_energy();
                minibeam_copper_neutral_xs_inverse_log_step =
                    1.0F / neutral_cross_sections.log_energy_step();
            }
        }
    }
    const auto enable_let_scoring = config.enable_let_scoring;
    const auto enable_depth_let_scoring =
        enable_let_scoring &&
        (!config.let_output_file.empty() ||
         !config.fragment_species_let_output_file.empty() ||
         !config.light_isotope_let_output_file.empty());
    const auto enable_voxel_let_scoring =
        enable_let_scoring && enable_voxel_scoring &&
        !config.let_voxel_mhd_output_file.empty();
    const auto enable_species_let_scoring =
        enable_let_scoring && !config.fragment_species_let_output_file.empty();
    const auto enable_light_isotope_let_scoring =
        enable_let_scoring && !config.light_isotope_let_output_file.empty();
    const auto enable_birth_spectrum =
        enable_secondary_generation &&
        !config.fragment_birth_spectrum_output_file.empty();
    std::vector<float> let_delta_fraction_host;
    if (enable_let_scoring &&
        !config.let_delta_electron_fraction_file.empty()) {
        const auto delta_fraction = StoppingPowerTable::from_csv(
            config.let_delta_electron_fraction_file);
        if (delta_fraction.values().size() != table_size ||
            delta_fraction.energies().size() !=
                stopping_power.energies().size()) {
            throw std::invalid_argument(
                "LET delta-electron fraction table must use the stopping-power grid");
        }
        let_delta_fraction_host.resize(table_size);
        for (std::size_t index = 0; index < table_size; ++index) {
            if (std::abs(delta_fraction.energies()[index] -
                         stopping_power.energies()[index]) > 1.0e-9) {
                throw std::invalid_argument(
                    "LET delta-electron fraction energies differ from stopping-power grid");
            }
            const auto value = delta_fraction.values()[index];
            if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
                throw std::invalid_argument(
                    "LET delta-electron fractions must be finite and in [0,1)");
            }
            let_delta_fraction_host[index] = static_cast<float>(value);
        }
    }
    const auto use_let_delta_fraction_table =
        !let_delta_fraction_host.empty();
    const auto use_particle_specific_stopping_power =
        config.use_particle_specific_stopping_power && enable_secondary_transport;
    std::vector<float> particle_sp_ratio_host;
    std::vector<float> particle_delta_fraction_host;
    std::vector<std::uint8_t> particle_species_present_host;
    if (use_particle_specific_stopping_power) {
        const auto ion_tables = IonStoppingPowerTables::from_csv(
            config.particle_stopping_power_file, stopping_power);
        particle_sp_ratio_host = ion_tables.ratios_to_carbon();
        particle_delta_fraction_host =
            ion_tables.delta_electron_fractions();
        particle_species_present_host = ion_tables.species_present();
        std::cout << "Particle-specific stopping power: enabled ("
                  << std::count(particle_species_present_host.begin(),
                                particle_species_present_host.end(),
                                std::uint8_t{1})
                  << " isotope tables; missing isotopes fall back to C-12 scaling)\n";
    }
    const auto neutral_allow_continuation =
        enable_neutral_transport && config.neutral_transport_mode == "full";
    const auto automatic_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() / 16
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * 16;
    auto secondary_queue_capacity =
        config.secondary_queue_capacity == 0 ? automatic_queue_capacity
                                             : config.secondary_queue_capacity;
    // Production SOBP measurements use about 1.8 neutral slots/history for
    // first-interaction mode and 2.4 for two generations. Preserve explicit
    // headroom without allocating and clearing 32 slots/history for every spot.
    const std::size_t automatic_neutral_slots_per_history =
        neutral_allow_continuation ? 8U : 4U;
    const auto automatic_neutral_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() /
                                  automatic_neutral_slots_per_history
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * automatic_neutral_slots_per_history;
    auto neutral_queue_capacity =
        config.neutral_queue_capacity == 0 ? automatic_neutral_queue_capacity
                                           : config.neutral_queue_capacity;
    if (enable_secondary_generation &&
        secondary_queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Secondary queue capacity exceeds the uint32 runtime limit");
    }
    if (enable_neutral_transport &&
        neutral_queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Neutral queue capacity exceeds the uint32 runtime limit");
    }

    const auto& device = queue.get_device();
    if constexpr (!k_dose_atomic_fp32) {
        if (!device.has(sycl::aspect::fp64) || !device.has(sycl::aspect::atomic64)) {
            throw std::runtime_error(
                "The FP64 SYCL dose scorer requires fp64 and atomic64 device aspects "
                "(rebuild with -DCARBON_DOSE_FP32=ON for float atomics)");
        }
    }
    if constexpr (!k_let_atomic_fp32) {
      if (enable_let_scoring &&
        (!device.has(sycl::aspect::fp64) ||
         !device.has(sycl::aspect::atomic64))) {
        throw std::runtime_error(
            "The FP64 LET_d scorer requires fp64 and atomic64 device aspects");
      }
    }

    // CUDA (especially under WSL2) cannot host multi-minute single kernels: the
    // WDDM/TDR path and shared GPU stack freeze the whole distro, and USM often
    // reserves a huge host VA. Clamp queues early so estimates/allocations stay
    // modest. Intel Level Zero keeps the original larger defaults.
    const bool is_cuda_backend =
        device.get_backend() == sycl::backend::ext_oneapi_cuda;
    if (is_cuda_backend) {
        // Hard ceiling only for pathological sizes; real limit is the device
        // memory budget below. SOBP 10M + cascade needs ~3–4 slots/history
        // (~30–40M) when launched as one batched plan.
        constexpr std::size_t kCudaMaxSecondarySlots = 64ULL * 1024ULL * 1024ULL;
        // Explicit high-statistics runs may request a larger neutral queue;
        // the device-memory budget below remains the final allocation guard.
        constexpr std::size_t kCudaMaxNeutralSlots = 96ULL * 1024ULL * 1024ULL;
        const auto cuda_auto_secondary =
            std::max(number_of_histories * 4U, std::size_t{8192});
        if (config.secondary_queue_capacity == 0) {
            secondary_queue_capacity =
                std::min(secondary_queue_capacity, cuda_auto_secondary);
        }
        secondary_queue_capacity =
            std::min(secondary_queue_capacity, kCudaMaxSecondarySlots);
        if (config.neutral_queue_capacity == 0) {
            neutral_queue_capacity =
                std::min(neutral_queue_capacity,
                         std::max(number_of_histories * 2U, std::size_t{4096}));
        }
        neutral_queue_capacity =
            std::min(neutral_queue_capacity, kCudaMaxNeutralSlots);
        std::cout << "CUDA backend detected: WSL-safe queue caps "
                  << "secondary_queue_capacity=" << secondary_queue_capacity
                  << " neutral_queue_capacity=" << neutral_queue_capacity << '\n'
                  << std::flush;
    }

    // Soft device-memory budget: keep estimated USM under max_device_memory_fraction
    // of global memory (default 80%). Shrink secondary/neutral queues first.
    const auto device_global_bytes =
        device.get_info<sycl::info::device::global_mem_size>();
    if (device_global_bytes == 0) {
        throw std::runtime_error(
            "SYCL device reported global_mem_size=0; refusing to allocate");
    }
    // Soft clamp only for extreme YAML values; large SOBP needs >35% for queues.
    auto memory_fraction = config.max_device_memory_fraction;
    if (is_cuda_backend && memory_fraction > 0.82) {
        memory_fraction = 0.82;
        std::cout << "CUDA backend: clamping max_device_memory_fraction to 0.82\n"
                  << std::flush;
    }
    const auto memory_budget_bytes = static_cast<std::size_t>(
        static_cast<double>(device_global_bytes) *
        std::min(1.0, std::max(0.05, memory_fraction)));

    const auto estimate_device_bytes = [&](std::size_t sec_cap,
                                          std::size_t neu_cap) -> std::size_t {
        std::size_t bytes = 0;
        bytes += table_size * sizeof(float);
        bytes += cross_section_table_size * sizeof(float);
        if (use_let_delta_fraction_table) {
            bytes += table_size * sizeof(float);
        }
        if (use_particle_specific_stopping_power) {
            bytes += particle_sp_ratio_host.size() * sizeof(float);
            bytes += particle_delta_fraction_host.size() * sizeof(float);
            bytes += particle_species_present_host.size() * sizeof(std::uint8_t);
        }
        bytes += number_of_bins * sizeof(DoseAtomicT);
        if (enable_depth_let_scoring) {
            bytes += 4 * number_of_bins * sizeof(LetAtomicT);
        }
        if (enable_species_let_scoring) {
            bytes += 2 * charged_origin_category_count * number_of_bins *
                     sizeof(LetAtomicT);
        }
        if (enable_light_isotope_let_scoring) {
            bytes += 2 * light_isotope_category_count * number_of_bins *
                     sizeof(LetAtomicT);
        }
        if (enable_voxel_let_scoring) {
            bytes += 4 * number_of_voxels * sizeof(LetAtomicT);
        }
        if (enable_birth_spectrum) {
            bytes += light_isotope_category_count * birth_generation_bin_count *
                     (sizeof(std::uint64_t) + sizeof(double));
            bytes += birth_hist_plane_size(birth_mevu_bin_count) * sizeof(std::uint64_t);
            bytes += birth_hist_plane_size(number_of_bins) * sizeof(std::uint64_t);
            bytes += birth_hist_plane_size(birth_cos_bin_count) * sizeof(std::uint64_t);
            bytes +=
                birth_hist_plane_size(birth_parent_mevu_bin_count) * sizeof(std::uint64_t);
            bytes +=
                birth_hist_plane_size(birth_parent_z_bin_count) * sizeof(std::uint64_t);
            bytes += birth_joint_plane_size() * sizeof(std::uint64_t);
        }
        bytes += number_of_histories * (3 * sizeof(float) + sizeof(std::uint32_t));
        bytes += primary_spot_count * sizeof(PrimarySpotBatchEntry);
        if (enable_voxel_scoring) {
            bytes += number_of_voxels * sizeof(DoseAtomicT);
        }
        if (enable_charged_origin_voxel_scoring) {
            bytes += charged_origin_category_count * number_of_voxels * sizeof(DoseAtomicT);
        }
        if (enable_secondary_generation && reaction_packages != nullptr) {
            bytes += reaction_packages->energy_bins().size() * sizeof(ReactionEnergyBin);
            bytes += reaction_packages->reactions().size() * sizeof(ReactionPackage);
            bytes += reaction_packages->secondaries().size() * sizeof(ReactionSecondary);
            bytes += sec_cap * sizeof(SecondaryParticle3D);
            bytes += 2 * sizeof(std::uint64_t);
            bytes += number_of_histories * sizeof(SecondaryGenerationSummary);
            if (enable_secondary_transport) {
                bytes += number_of_bins * sizeof(DoseAtomicT);
                if (enable_fragment_species_scoring) {
                    bytes += fragment_species_count * number_of_bins * sizeof(DoseAtomicT);
                }
                bytes += sec_cap * (2 * sizeof(float) + sizeof(std::uint32_t));
                if (enable_secondary_energy_sorting) {
                    const auto scratch_capacity =
                        std::min(sec_cap, std::size_t{65536});
                    bytes += scratch_capacity *
                             (sizeof(SecondaryParticle3D) + sizeof(std::uint64_t));
                    bytes += 4 * sizeof(std::uint64_t);
                }
            }
            if (enable_fragment_cascade && cascade_packages != nullptr) {
                bytes += cascade_packages->projectiles().size() * sizeof(CascadeProjectile);
                bytes += cascade_packages->cross_sections().size() *
                         sizeof(CascadeCrossSectionSample);
                bytes += cascade_packages->interactions().size() * sizeof(CascadeInteraction);
                bytes += cascade_packages->products().size() * sizeof(ReactionSecondary);
                bytes += sec_cap * sizeof(CascadeTransportSummary);
            }
        }
        if (enable_neutral_transport && neutral_packages != nullptr) {
            bytes += neutral_packages->projectiles().size() * sizeof(NeutralProjectile);
            bytes += neutral_packages->cross_sections().size() *
                     sizeof(NeutralCrossSectionSample);
            bytes += neutral_packages->interactions().size() * sizeof(NeutralInteraction);
            bytes += neutral_packages->products().size() * sizeof(ReactionSecondary);
            bytes += neu_cap * sizeof(NeutralParticle3D);
            bytes += 2 * sizeof(std::uint64_t);
            bytes += neu_cap * sizeof(NeutralTransportSummary);
            bytes += neutral_origin_category_count * number_of_bins * sizeof(DoseAtomicT);
            if (enable_voxel_scoring) {
                bytes += neutral_origin_category_count * number_of_voxels * sizeof(DoseAtomicT);
            }
        }
        if (config.enable_ct_grid) {
            // Conservative upper bound until grid is loaded below.
            bytes += 64ULL * 1024ULL * 1024ULL;
        }
        // Driver / allocator overhead headroom.
        bytes = bytes + bytes / 8;
        return bytes;
    };

    {
        auto estimated = estimate_device_bytes(secondary_queue_capacity, neutral_queue_capacity);
        if (estimated > memory_budget_bytes) {
            const auto fixed = estimate_device_bytes(0, 0);
            if (fixed >= memory_budget_bytes) {
                throw std::runtime_error(
                    "Device memory budget exceeded by fixed buffers alone (tables/packages/"
                    "histories). Reduce histories, disable voxels, or raise "
                    "max_device_memory_fraction (device=" +
                    std::to_string(device_global_bytes / (1024ULL * 1024ULL)) +
                    " MiB, budget=" +
                    std::to_string(memory_budget_bytes / (1024ULL * 1024ULL)) +
                    " MiB, fixed~" + std::to_string(fixed / (1024ULL * 1024ULL)) + " MiB)");
            }
            const auto variable_budget = memory_budget_bytes - fixed;
            // Per secondary slot: particle + dep/esc/steps + cascade summary.
            std::size_t bytes_per_sec = sizeof(SecondaryParticle3D);
            if (enable_secondary_transport) {
                bytes_per_sec += 2 * sizeof(float) + sizeof(std::uint32_t);
            }
            if (enable_fragment_cascade) {
                bytes_per_sec += sizeof(CascadeTransportSummary);
            }
            std::size_t bytes_per_neu = 0;
            if (enable_neutral_transport) {
                bytes_per_neu =
                    sizeof(NeutralParticle3D) + sizeof(NeutralTransportSummary);
            }
            // Prefer keeping secondary capacity; scale both proportionally.
            const auto total_var =
                secondary_queue_capacity * bytes_per_sec +
                neutral_queue_capacity * bytes_per_neu;
            if (total_var > 0) {
                const auto scale = static_cast<double>(variable_budget) /
                                   static_cast<double>(total_var);
                if (scale < 1.0) {
                    secondary_queue_capacity = static_cast<std::size_t>(
                        std::floor(static_cast<double>(secondary_queue_capacity) * scale));
                    neutral_queue_capacity = static_cast<std::size_t>(
                        std::floor(static_cast<double>(neutral_queue_capacity) * scale));
                }
            }
            // Enforce minimum usable queues.
            if (enable_secondary_generation) {
                secondary_queue_capacity =
                    std::max<std::size_t>(secondary_queue_capacity, number_of_histories);
            }
            if (enable_neutral_transport) {
                neutral_queue_capacity =
                    std::max<std::size_t>(neutral_queue_capacity, number_of_histories);
            }
            estimated = estimate_device_bytes(secondary_queue_capacity, neutral_queue_capacity);
            if (estimated > memory_budget_bytes) {
                throw std::runtime_error(
                    "Unable to fit device buffers under max_device_memory_fraction=" +
                    std::to_string(config.max_device_memory_fraction) + " (estimate " +
                    std::to_string(estimated / (1024ULL * 1024ULL)) + " MiB > budget " +
                    std::to_string(memory_budget_bytes / (1024ULL * 1024ULL)) + " MiB)");
            }
            std::cout << "Device memory budget: scaled queues to fit "
                      << static_cast<int>(config.max_device_memory_fraction * 100.0)
                      << "% of "
                      << (device_global_bytes / (1024ULL * 1024ULL)) << " MiB"
                      << " (estimate " << (estimated / (1024ULL * 1024ULL)) << " MiB;"
                      << " secondary_queue_capacity=" << secondary_queue_capacity
                      << "; neutral_queue_capacity=" << neutral_queue_capacity << ")\n";
        } else {
            std::cout << "Device memory estimate: "
                      << (estimated / (1024ULL * 1024ULL)) << " MiB / budget "
                      << (memory_budget_bytes / (1024ULL * 1024ULL)) << " MiB ("
                      << static_cast<int>(config.max_device_memory_fraction * 100.0)
                      << "% of "
                      << (device_global_bytes / (1024ULL * 1024ULL)) << " MiB); "
                      << "secondary_queue_capacity=" << secondary_queue_capacity
                      << "\n";
        }
    }

    if (reuse_immutable_buffers) {
        context->impl_->ensure_initialized(stopping_power, cross_section, reaction_packages,
                                           cascade_packages, neutral_packages);
    }
    auto* table_device = reuse_immutable_buffers
                             ? context->impl_->table_device
                             : sycl::malloc_device<float>(table_size, queue);
    auto* cross_section_device =
        reuse_immutable_buffers
            ? context->impl_->cross_section_device
            : sycl::malloc_device<float>(cross_section_table_size, queue);
    auto* let_delta_fraction_device =
        use_let_delta_fraction_table
            ? sycl::malloc_device<float>(table_size, queue)
            : nullptr;
    auto* particle_sp_ratio_device =
        use_particle_specific_stopping_power
            ? sycl::malloc_device<float>(particle_sp_ratio_host.size(), queue)
            : nullptr;
    auto* particle_delta_fraction_device =
        use_particle_specific_stopping_power
            ? sycl::malloc_device<float>(particle_delta_fraction_host.size(), queue)
            : nullptr;
    auto* particle_species_present_device =
        use_particle_specific_stopping_power
            ? sycl::malloc_device<std::uint8_t>(
                  particle_species_present_host.size(), queue)
            : nullptr;
    auto* dose_device = sycl::malloc_device<DoseAtomicT>(number_of_bins, queue);
    auto* let_moments_device =
        enable_depth_let_scoring
            ? sycl::malloc_device<LetAtomicT>(4 * number_of_bins, queue)
            : nullptr;
    auto* voxel_let_moments_device =
        enable_voxel_let_scoring
            ? sycl::malloc_device<LetAtomicT>(4 * number_of_voxels, queue)
            : nullptr;
    auto* species_let_moments_device =
        enable_species_let_scoring
            ? sycl::malloc_device<LetAtomicT>(
                  2 * charged_origin_category_count * number_of_bins, queue)
            : nullptr;
    auto* isotope_let_moments_device =
        enable_light_isotope_let_scoring
            ? sycl::malloc_device<LetAtomicT>(
                  2 * light_isotope_category_count * number_of_bins, queue)
            : nullptr;
    const auto birth_gen_size =
        light_isotope_category_count * birth_generation_bin_count;
    const auto birth_mevu_size = birth_hist_plane_size(birth_mevu_bin_count);
    const auto birth_depth_size = birth_hist_plane_size(number_of_bins);
    const auto birth_cos_size = birth_hist_plane_size(birth_cos_bin_count);
    const auto birth_parent_mevu_size =
        birth_hist_plane_size(birth_parent_mevu_bin_count);
    const auto birth_parent_z_size =
        birth_hist_plane_size(birth_parent_z_bin_count);
    const auto birth_joint_size = birth_joint_plane_size();
    auto* birth_counts_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_gen_size, queue)
            : nullptr;
    auto* birth_ke_sum_device =
        enable_birth_spectrum ? sycl::malloc_device<double>(birth_gen_size, queue)
                              : nullptr;
    auto* birth_mevu_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_mevu_size, queue)
            : nullptr;
    auto* birth_depth_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_depth_size, queue)
            : nullptr;
    auto* birth_cos_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_cos_size, queue)
            : nullptr;
    auto* birth_parent_mevu_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_parent_mevu_size, queue)
            : nullptr;
    auto* birth_parent_z_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_parent_z_size, queue)
            : nullptr;
    auto* birth_parent_product_mevu_hist_device =
        enable_birth_spectrum
            ? sycl::malloc_device<std::uint64_t>(birth_joint_size, queue)
            : nullptr;
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? sycl::malloc_device<DoseAtomicT>(number_of_voxels, queue)
                                  : nullptr;
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? sycl::malloc_device<DoseAtomicT>(
                  charged_origin_category_count * number_of_voxels, queue)
            : nullptr;
    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* nuclear_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);
#ifdef CARBON_TRANSPORT_PROFILE
    auto* profile_counters_device =
        sycl::malloc_device<std::uint64_t>(transport_profile_slot_count(), queue);
#else
    std::uint64_t* profile_counters_device = nullptr;
#endif
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            sycl::malloc_device<PrimarySpotBatchEntry>(primary_spot_count, queue);
        if (primary_spots_device != nullptr) {
            queue.copy(config.primary_spot_batch.data(), primary_spots_device,
                       primary_spot_count)
                .wait_and_throw();
        }
    }
    ReactionEnergyBin* reaction_bins_device = nullptr;
    ReactionPackage* reactions_device = nullptr;
    ReactionSecondary* reaction_secondaries_device = nullptr;
    SecondaryParticle3D* secondary_queue_device = nullptr;
    SecondaryParticle3D* secondary_bucket_scratch_device = nullptr;
    std::uint64_t* secondary_bucket_index_device = nullptr;
    std::uint64_t* secondary_bucket_counters_device = nullptr;
    std::uint64_t* secondary_queue_counter_device = nullptr;
    std::uint64_t* secondary_queue_filled_device = nullptr;
    std::uint64_t* secondary_work_counter_device = nullptr;
    SecondaryGenerationSummary* secondary_summaries_device = nullptr;
    DoseAtomicT* aggregate_secondary_dose_device = nullptr;
    DoseAtomicT* fragment_dose_device = nullptr;
    float* secondary_deposited_device = nullptr;
    float* secondary_escaped_device = nullptr;
    std::uint32_t* secondary_steps_device = nullptr;
    CascadeProjectile* cascade_projectiles_device = nullptr;
    CascadeCrossSectionSample* cascade_cross_sections_device = nullptr;
    CascadeInteraction* cascade_interactions_device = nullptr;
    ReactionSecondary* cascade_products_device = nullptr;
    CascadeTransportSummary* cascade_summaries_device = nullptr;
    // Projectile-major XS on water SP energy grid: size n_proj * table_size.
    float* cascade_xs_lut_device = nullptr;
    std::size_t cascade_xs_lut_size = 0;
    NeutralProjectile* neutral_projectiles_device = nullptr;
    NeutralCrossSectionSample* neutral_cross_sections_device = nullptr;
    NeutralInteraction* neutral_interactions_device = nullptr;
    ReactionSecondary* neutral_products_device = nullptr;
    NeutralParticle3D* neutral_queue_device = nullptr;
    std::uint64_t* neutral_queue_counter_device = nullptr;
    std::uint64_t* neutral_queue_filled_device = nullptr;
    NeutralTransportSummary* neutral_summaries_device = nullptr;
    DoseAtomicT* neutral_origin_dose_device = nullptr;
    DoseAtomicT* neutral_origin_voxel_dose_device = nullptr;
    if (enable_secondary_generation) {
        reaction_bins_device =
            reuse_immutable_buffers
                ? context->impl_->reaction_bins_device
                : sycl::malloc_device<ReactionEnergyBin>(
                      reaction_packages->energy_bins().size(), queue);
        reactions_device = reuse_immutable_buffers
                               ? context->impl_->reactions_device
                               : sycl::malloc_device<ReactionPackage>(
                                     reaction_packages->reactions().size(), queue);
        reaction_secondaries_device =
            reuse_immutable_buffers
                ? context->impl_->reaction_secondaries_device
                : sycl::malloc_device<ReactionSecondary>(
                      reaction_packages->secondaries().size(), queue);
        secondary_queue_device =
            sycl::malloc_device<SecondaryParticle3D>(secondary_queue_capacity, queue);
        secondary_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_summaries_device =
            sycl::malloc_device<SecondaryGenerationSummary>(number_of_histories, queue);
        if (enable_secondary_transport) {
            secondary_work_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
            aggregate_secondary_dose_device =
                sycl::malloc_device<DoseAtomicT>(number_of_bins, queue);
            if (enable_secondary_energy_sorting) {
                const auto scratch_capacity =
                    std::min(secondary_queue_capacity, std::size_t{65536});
                secondary_bucket_scratch_device =
                    sycl::malloc_device<SecondaryParticle3D>(scratch_capacity, queue);
                secondary_bucket_index_device =
                    sycl::malloc_device<std::uint64_t>(scratch_capacity, queue);
                secondary_bucket_counters_device =
                    sycl::malloc_device<std::uint64_t>(4, queue);
            }
            if (enable_fragment_species_scoring) {
                fragment_dose_device = sycl::malloc_device<DoseAtomicT>(
                    fragment_species_count * number_of_bins, queue);
            }
            secondary_deposited_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_escaped_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_steps_device =
                sycl::malloc_device<std::uint32_t>(secondary_queue_capacity, queue);
        }
        if (enable_fragment_cascade) {
            cascade_projectiles_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_projectiles_device
                    : sycl::malloc_device<CascadeProjectile>(
                          cascade_packages->projectiles().size(), queue);
            cascade_cross_sections_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_cross_sections_device
                    : sycl::malloc_device<CascadeCrossSectionSample>(
                          cascade_packages->cross_sections().size(), queue);
            cascade_interactions_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_interactions_device
                    : sycl::malloc_device<CascadeInteraction>(
                          cascade_packages->interactions().size(), queue);
            cascade_products_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_products_device
                    : sycl::malloc_device<ReactionSecondary>(
                          cascade_packages->products().size(), queue);
            cascade_summaries_device = sycl::malloc_device<CascadeTransportSummary>(
                secondary_queue_capacity, queue);
            cascade_xs_lut_size =
                cascade_xs_material_count *
                cascade_packages->projectiles().size() * table_size;
            cascade_xs_lut_device =
                sycl::malloc_device<float>(cascade_xs_lut_size, queue);
        }
    }
    if (enable_neutral_transport) {
        neutral_projectiles_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_projectiles_device
                : sycl::malloc_device<NeutralProjectile>(
                      neutral_packages->projectiles().size(), queue);
        neutral_cross_sections_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_cross_sections_device
                : sycl::malloc_device<NeutralCrossSectionSample>(
                      neutral_packages->cross_sections().size(), queue);
        neutral_interactions_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_interactions_device
                : sycl::malloc_device<NeutralInteraction>(
                      neutral_packages->interactions().size(), queue);
        neutral_products_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_products_device
                : sycl::malloc_device<ReactionSecondary>(
                      neutral_packages->products().size(), queue);
        neutral_queue_device =
            sycl::malloc_device<NeutralParticle3D>(neutral_queue_capacity, queue);
        neutral_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_summaries_device =
            sycl::malloc_device<NeutralTransportSummary>(neutral_queue_capacity, queue);
        neutral_origin_dose_device = sycl::malloc_device<DoseAtomicT>(
            neutral_origin_category_count * number_of_bins, queue);
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_device = sycl::malloc_device<DoseAtomicT>(
                neutral_origin_category_count * number_of_voxels, queue);
        }
    }
    const auto free_device = [&queue](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };
    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers) {
            free_device(pointer);
        }
    };

    const auto enable_layered_phantom = config.enable_layered_phantom;
    const auto slab_layer_count =
        enable_layered_phantom
            ? static_cast<std::uint32_t>(config.slab_layers.size())
            : 0U;
    float* slab_z_ends_device = nullptr;
    float* slab_densities_device = nullptr;
    float* slab_radiation_lengths_device = nullptr;
    if (slab_layer_count > 0) {
        slab_z_ends_device = sycl::malloc_device<float>(slab_layer_count, queue);
        slab_densities_device = sycl::malloc_device<float>(slab_layer_count, queue);
        if (!config.slab_radiation_lengths_g_per_cm2.empty()) {
            slab_radiation_lengths_device =
                sycl::malloc_device<float>(slab_layer_count, queue);
        }
        if (slab_z_ends_device == nullptr || slab_densities_device == nullptr ||
            (!config.slab_radiation_lengths_g_per_cm2.empty() &&
             slab_radiation_lengths_device == nullptr)) {
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            free_device(slab_radiation_lengths_device);
            free_immutable_device(table_device);
            free_immutable_device(cross_section_device);
            free_device(let_delta_fraction_device);
            free_device(particle_sp_ratio_device);
            free_device(particle_delta_fraction_device);
            free_device(particle_species_present_device);
            free_device(dose_device);
            free_device(let_moments_device);
            free_device(voxel_let_moments_device);
            free_device(species_let_moments_device);
            free_device(isotope_let_moments_device);
            free_device(birth_counts_device);
            free_device(birth_ke_sum_device);
            free_device(birth_mevu_hist_device);
            free_device(birth_depth_hist_device);
            free_device(birth_cos_hist_device);
            free_device(birth_parent_mevu_hist_device);
            free_device(birth_parent_z_hist_device);
            free_device(birth_parent_product_mevu_hist_device);
            free_device(voxel_dose_device);
            free_device(charged_origin_voxel_dose_device);
            free_device(deposited_device);
            free_device(escaped_device);
            free_device(nuclear_device);
            free_device(steps_device);
            free_device(profile_counters_device);
            throw std::bad_alloc();
        }
        std::vector<float> slab_z_host(slab_layer_count);
        std::vector<float> slab_rho_host(slab_layer_count);
        std::vector<float> slab_radiation_length_host;
        if (slab_radiation_lengths_device != nullptr) {
            slab_radiation_length_host.resize(slab_layer_count);
        }
        for (std::uint32_t index = 0; index < slab_layer_count; ++index) {
            slab_z_host[index] =
                static_cast<float>(config.slab_layers[index].z_end_mm);
            slab_rho_host[index] =
                static_cast<float>(config.slab_layers[index].density_g_per_cm3);
            if (slab_radiation_lengths_device != nullptr) {
                slab_radiation_length_host[index] = static_cast<float>(
                    config.slab_radiation_lengths_g_per_cm2[index]);
            }
        }
        queue.memcpy(slab_z_ends_device, slab_z_host.data(),
                     sizeof(float) * slab_layer_count)
            .wait_and_throw();
        queue.memcpy(slab_densities_device, slab_rho_host.data(),
                     sizeof(float) * slab_layer_count)
            .wait_and_throw();
        if (slab_radiation_lengths_device != nullptr) {
            queue
                .memcpy(slab_radiation_lengths_device,
                        slab_radiation_length_host.data(),
                        sizeof(float) * slab_layer_count)
                .wait_and_throw();
        }
    }

    // Optional absolute per-layer material tables (real bone/lung, not density-scaled water).
    const auto use_material_tables =
        enable_layered_phantom && !config.slab_stopping_power_files.empty();
    const auto material_table_count =
        use_material_tables ? static_cast<std::uint32_t>(config.slab_stopping_power_files.size())
                            : 0U;
    float* material_sp_device = nullptr;
    float* material_xs_device = nullptr;
    if (use_material_tables) {
        if (material_table_count != slab_layer_count) {
            throw std::invalid_argument("Material table count must match slab layer count");
        }
        std::vector<float> material_sp_host(static_cast<std::size_t>(material_table_count) *
                                            table_size);
        std::vector<float> material_xs_host(
            static_cast<std::size_t>(material_table_count) * cross_section_table_size);
        for (std::uint32_t mat = 0; mat < material_table_count; ++mat) {
            const auto sp_table =
                StoppingPowerTable::from_csv(config.slab_stopping_power_files[mat]);
            const auto xs_table =
                CrossSectionTable::from_csv(config.slab_cross_section_files[mat]);
            if (sp_table.values().size() != table_size ||
                !is_uniform_grid(sp_table.energies()) ||
                std::abs(sp_table.energies().front() - stopping_power.energies().front()) >
                    1.0e-9 ||
                std::abs(sp_table.energies()[1] - sp_table.energies()[0] -
                         (stopping_power.energies()[1] - stopping_power.energies()[0])) >
                    1.0e-9) {
                throw std::invalid_argument(
                    "Slab material SP table must match the primary water energy grid");
            }
            if (xs_table.values().size() != cross_section_table_size ||
                !is_uniform_grid(xs_table.energies()) ||
                std::abs(xs_table.energies().front() - cross_section.energies().front()) >
                    1.0e-9) {
                throw std::invalid_argument(
                    "Slab material XS table must match the primary water energy grid");
            }
            for (std::size_t i = 0; i < table_size; ++i) {
                material_sp_host[static_cast<std::size_t>(mat) * table_size + i] =
                    static_cast<float>(sp_table.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                material_xs_host[static_cast<std::size_t>(mat) * cross_section_table_size + i] =
                    static_cast<float>(xs_table.values()[i]);
            }
        }
        material_sp_device = sycl::malloc_device<float>(material_sp_host.size(), queue);
        material_xs_device = sycl::malloc_device<float>(material_xs_host.size(), queue);
        if (material_sp_device == nullptr || material_xs_device == nullptr) {
            free_device(material_sp_device);
            free_device(material_xs_device);
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            free_device(slab_radiation_lengths_device);
            throw std::bad_alloc();
        }
        queue.memcpy(material_sp_device, material_sp_host.data(),
                     sizeof(float) * material_sp_host.size())
            .wait_and_throw();
        queue.memcpy(material_xs_device, material_xs_host.data(),
                     sizeof(float) * material_xs_host.size())
            .wait_and_throw();
    }

    // 7b: single AABB insert (bone strip / cavity) in water background.
    const auto enable_hetero_insert = config.enable_hetero_insert;
    const auto insert_x_min = static_cast<float>(config.hetero_insert.x_min_mm);
    const auto insert_x_max = static_cast<float>(config.hetero_insert.x_max_mm);
    const auto insert_y_min = static_cast<float>(config.hetero_insert.y_min_mm);
    const auto insert_y_max = static_cast<float>(config.hetero_insert.y_max_mm);
    const auto insert_z_min = static_cast<float>(config.hetero_insert.z_min_mm);
    const auto insert_z_max = static_cast<float>(config.hetero_insert.z_max_mm);
    const auto insert_density_g_per_cm3 =
        static_cast<float>(config.hetero_insert.density_g_per_cm3);
    const auto insert_radiation_length_g_per_cm2 =
        static_cast<float>(config.insert_radiation_length_g_per_cm2);
    const auto use_insert_material_tables =
        enable_hetero_insert && !config.insert_stopping_power_file.empty();
    float* insert_sp_device = nullptr;
    float* insert_xs_device = nullptr;
    if (use_insert_material_tables) {
        const auto insert_sp =
            StoppingPowerTable::from_csv(config.insert_stopping_power_file);
        const auto insert_xs =
            CrossSectionTable::from_csv(config.insert_cross_section_file);
        if (insert_sp.values().size() != table_size ||
            !is_uniform_grid(insert_sp.energies()) ||
            std::abs(insert_sp.energies().front() - stopping_power.energies().front()) >
                1.0e-9) {
            throw std::invalid_argument(
                "Insert SP table must match the primary water energy grid");
        }
        if (insert_xs.values().size() != cross_section_table_size ||
            !is_uniform_grid(insert_xs.energies()) ||
            std::abs(insert_xs.energies().front() - cross_section.energies().front()) >
                1.0e-9) {
            throw std::invalid_argument(
                "Insert XS table must match the primary water energy grid");
        }
        std::vector<float> insert_sp_host(table_size);
        std::vector<float> insert_xs_host(cross_section_table_size);
        for (std::size_t i = 0; i < table_size; ++i) {
            insert_sp_host[i] = static_cast<float>(insert_sp.values()[i]);
        }
        for (std::size_t i = 0; i < cross_section_table_size; ++i) {
            insert_xs_host[i] = static_cast<float>(insert_xs.values()[i]);
        }
        insert_sp_device = sycl::malloc_device<float>(table_size, queue);
        insert_xs_device = sycl::malloc_device<float>(cross_section_table_size, queue);
        if (insert_sp_device == nullptr || insert_xs_device == nullptr) {
            free_device(insert_sp_device);
            free_device(insert_xs_device);
            free_device(material_sp_device);
            free_device(material_xs_device);
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            free_device(slab_radiation_lengths_device);
            throw std::bad_alloc();
        }
        queue.memcpy(insert_sp_device, insert_sp_host.data(), sizeof(float) * table_size)
            .wait_and_throw();
        queue
            .memcpy(insert_xs_device, insert_xs_host.data(),
                    sizeof(float) * cross_section_table_size)
            .wait_and_throw();
    }

    // 7c: CT voxel density + material-id field.
    const auto enable_ct_grid = config.enable_ct_grid;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;
    CtGrid ct_grid_host{};
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    // Schneider section mass-SP: (Z/A)_rel and Bragg I [eV] (CCTG v2/v3).
    // Precomputed mass-SP factor LUT: section * table_size + energy_index.
    // Replaces per-step Bethe log evaluation (P2).
    float* ct_mass_sp_factor_lut_device = nullptr;
    float* ct_mass_sp_za_rel_device = nullptr;  // Schneider (Z/A)_rel per section
    float* ct_sp_device = nullptr;  // optional absolute 4-class tables
    float* ct_xs_device = nullptr;
    float* ct_ref_density_device = nullptr;
    std::uint32_t ct_nx = 0;
    std::uint32_t ct_ny = 0;
    std::uint32_t ct_nz = 0;
    std::uint32_t ct_n_mass_factors = 0;
    float ct_origin_x = 0.0F;
    float ct_origin_y = 0.0F;
    float ct_origin_z = 0.0F;
    float ct_spacing_x = 1.0F;
    float ct_spacing_y = 1.0F;
    float ct_spacing_z = 1.0F;
    // Prefer Schneider mass-SP scaling when the grid carries factors (CCTG v2/v3).
    // Material SP and nuclear XS are independent: a v3 mass-SP grid can still
    // use composition-specific 4-class nuclear cross sections.
    auto use_ct_mass_sp = false;
    auto use_ct_density_mass_spr = false;
    auto ct_mass_spr_log_rho_min = 0.0F;
    auto ct_mass_spr_inv_dlog = 0.0F;
    auto ct_density_spr_n_rho = 0U;
    auto use_ct_material_sp = false;
    auto use_ct_material_xs = false;
    auto use_ct_schneider_xs = false;
    std::uint32_t ct_xs_material_count = 4;
    auto ct_material_ids_are_schneider_sections = false;
    if (enable_ct_grid) {
        ct_grid_host = CtGrid::from_config(config);
        ct_nx = ct_grid_host.nx;
        ct_ny = ct_grid_host.ny;
        ct_nz = ct_grid_host.nz;
        ct_origin_x = ct_grid_host.origin_x_mm;
        ct_origin_y = ct_grid_host.origin_y_mm;
        ct_origin_z = ct_grid_host.origin_z_mm;
        ct_spacing_x = ct_grid_host.spacing_x_mm;
        ct_spacing_y = ct_grid_host.spacing_y_mm;
        ct_spacing_z = ct_grid_host.spacing_z_mm;
        if (config.uses_fixed_patient_coordinates()) {
            // TpsSourcePlan applies the opposite shift to every source origin.
            // This lets arbitrary-angle rays traverse a fixed patient CT while
            // retaining the kernel's long-standing z=[0,L] scorer convention.
            ct_origin_z = 0.0F;
        }
        // Align dose scorer xy bins with the CT sample grid so that:
        //   floor((x - origin) / spacing) matches for density, mass and tally.
        // Previously the scorer was forced to a 0-centered box (edge -n*s/2),
        // which is 0.25 mm (X) / 1 mm (Y) off the TPS-90 CT origins and breaks
        // dose-to-medium index pairing used by voxel_masses_kg.
        if (enable_voxel_scoring && ct_nx == voxel_bins_x && ct_ny == voxel_bins_y &&
            std::fabs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
            std::fabs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
            voxel_min_x_mm = ct_origin_x;
            voxel_min_y_mm = ct_origin_y;
            voxel_max_x_mm =
                ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
            voxel_max_y_mm =
                ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
        }
        // CCTG v1 material IDs select the explicit four-class absolute tables.
        // from_binary() supplies unit factors for legacy diagnostics, but those
        // are not Schneider mass-SP metadata and must not shadow the material
        // tables configured for air/lung/water/bone.
        use_ct_mass_sp = ct_grid_host.uses_schneider_mass_sp();
        ct_material_ids_are_schneider_sections =
            ct_grid_host.file_version != CtGrid::version_legacy;
        use_ct_material_sp =
            !use_ct_mass_sp &&
            (!config.ct_air_stopping_power_file.empty() ||
             !config.ct_lung_stopping_power_file.empty() ||
             !config.ct_water_stopping_power_file.empty() ||
             !config.ct_bone_stopping_power_file.empty());
        use_ct_material_xs =
            !config.ct_schneider_cross_section_file.empty() ||
            !config.ct_air_cross_section_file.empty() ||
            !config.ct_lung_cross_section_file.empty() ||
            !config.ct_water_cross_section_file.empty() ||
            !config.ct_bone_cross_section_file.empty();
        const auto ct_count = ct_grid_host.number_of_voxels();
        ct_density_device = sycl::malloc_device<float>(ct_count, queue);
        ct_material_device = sycl::malloc_device<std::uint8_t>(ct_count, queue);
        if (ct_density_device == nullptr || ct_material_device == nullptr) {
            free_device(ct_density_device);
            free_device(ct_material_device);
            throw std::bad_alloc();
        }
        queue
            .memcpy(ct_density_device, ct_grid_host.density_g_per_cm3.data(),
                    sizeof(float) * ct_count)
            .wait_and_throw();
        queue
            .memcpy(ct_material_device, ct_grid_host.material_id.data(),
                    sizeof(std::uint8_t) * ct_count)
            .wait_and_throw();

        if (use_ct_mass_sp) {
            const auto& za_host = !ct_grid_host.mass_sp_za_rel.empty()
                                      ? ct_grid_host.mass_sp_za_rel
                                      : ct_grid_host.mass_sp_factor;
            ct_n_mass_factors = static_cast<std::uint32_t>(za_host.size());
            std::vector<float> I_host = ct_grid_host.mass_sp_I_eV;
            if (I_host.size() != za_host.size()) {
                I_host.assign(za_host.size(), 78.0F);
            }
            // P2: precompute mass-SP factor on the water energy grid so the
            // kernel only does clamp/index/lerp (no log/Bethe on every step).
            // Layout: section-major, size n_sections * table_size.
            // Bake ct_stopping_power_scale into the LUT (avoids a per-step mul).
            const auto sp_scale =
                static_cast<float>(config.ct_stopping_power_scale);
            const auto lut_count =
                static_cast<std::size_t>(ct_n_mass_factors) * table_size;
            std::vector<float> mass_factor_lut(lut_count);
            const auto load_configured_hu_lut = [&]() {
                if (config.ct_hu_stopping_power_lut_file.empty()) {
                    return false;
                }
                // Explicit physics input must never fail open: a missing or
                // malformed LUT would otherwise select a different material
                // model while leaving the run apparently successful.
                mass_factor_lut = load_hu_stopping_power_lut(
                    config.ct_hu_stopping_power_lut_file,
                    ct_n_mass_factors, table_size, sp_scale);
                return true;
            };
            const auto try_density_spr = [&]() {
                if (!config.ct_use_density_mass_spr) {
                    return false;
                }
                const auto resolve = [](const std::filesystem::path& configured,
                                        const char* fallback) {
                    if (!configured.empty() && std::filesystem::exists(configured)) {
                        return configured;
                    }
                    const std::filesystem::path candidate{fallback};
                    return std::filesystem::exists(candidate) ? candidate
                                                              : std::filesystem::path{};
                };
                const auto air_path = resolve(
                    config.ct_air_stopping_power_file,
                    "data/stopping_power_air_geant4_11_3_2.csv");
                const auto lung_path = resolve(
                    config.ct_lung_stopping_power_file,
                    "data/stopping_power_lung_geant4_11_3_2.csv");
                const auto bone_path = resolve(
                    config.ct_bone_stopping_power_file,
                    "data/stopping_power_bone_geant4_11_3_2.csv");
                if (air_path.empty() || lung_path.empty() || bone_path.empty()) {
                    return false;
                }
                const auto air_table = StoppingPowerTable::from_csv(air_path);
                const auto lung_table = StoppingPowerTable::from_csv(lung_path);
                const auto bone_table = StoppingPowerTable::from_csv(bone_path);
                const auto density_lut = build_density_mass_spr_lut(
                    stopping_power, air_table, lung_table, bone_table, sp_scale);
                if (density_lut.n_rho < 2 ||
                    density_lut.factors.size() !=
                        static_cast<std::size_t>(density_lut.n_rho) * table_size) {
                    return false;
                }
                mass_factor_lut = density_lut.factors;
                ct_density_spr_n_rho = density_lut.n_rho;
                ct_mass_spr_log_rho_min = density_lut.log_rho_min;
                ct_mass_spr_inv_dlog = density_lut.inv_dlog;
                use_ct_density_mass_spr = true;
                return true;
            };
            if (!load_configured_hu_lut()) {
                if (!try_density_spr()) {
                    for (std::uint32_t sec = 0; sec < ct_n_mass_factors; ++sec) {
                        const auto za = za_host[sec];
                        const auto I_eV = I_host[sec];
                        const auto base = static_cast<std::size_t>(sec) * table_size;
                        for (std::size_t i = 0; i < table_size; ++i) {
                            mass_factor_lut[base + i] =
                                sp_scale * ct_mass_sp_energy_factor(
                                               za, I_eV,
                                               static_cast<float>(
                                                   stopping_power.energies()[i]));
                        }
                    }
                }
            }
            const auto lut_bytes = mass_factor_lut.size();
            ct_mass_sp_factor_lut_device = sycl::malloc_device<float>(lut_bytes, queue);
            ct_mass_sp_za_rel_device =
                sycl::malloc_device<float>(ct_n_mass_factors, queue);
            if (ct_mass_sp_factor_lut_device == nullptr ||
                ct_mass_sp_za_rel_device == nullptr) {
                free_device(ct_density_device);
                free_device(ct_material_device);
                free_device(ct_mass_sp_factor_lut_device);
                free_device(ct_mass_sp_za_rel_device);
                throw std::bad_alloc();
            }
            queue
                .memcpy(ct_mass_sp_factor_lut_device, mass_factor_lut.data(),
                        sizeof(float) * lut_bytes)
                .wait_and_throw();
            queue
                .memcpy(ct_mass_sp_za_rel_device, za_host.data(),
                        sizeof(float) * ct_n_mass_factors)
                .wait_and_throw();
        }

        // Optional absolute 4-class tables (legacy path when no mass-SP LUT).
        std::vector<float> ct_sp_host(4 * table_size);
        use_ct_schneider_xs =
            ct_material_ids_are_schneider_sections &&
            !config.ct_schneider_cross_section_file.empty();
        std::vector<CrossSectionTable> schneider_xs_tables;
        if (use_ct_schneider_xs) {
            schneider_xs_tables = CrossSectionTable::from_schneider_csv(
                config.ct_schneider_cross_section_file);
            if (schneider_xs_tables.size() < ct_n_mass_factors) {
                throw std::invalid_argument(
                    "Schneider cross-section table has fewer sections than the CT grid");
            }
            ct_xs_material_count =
                static_cast<std::uint32_t>(schneider_xs_tables.size());
        }
        std::vector<float> ct_xs_host(
            static_cast<std::size_t>(ct_xs_material_count) *
            cross_section_table_size);
        // Each empty material path falls back to the water table, whose
        // reference density is 1.0. Only use a material's native density when
        // at least one native table was actually supplied for that class.
        std::vector<float> ct_ref_host(ct_xs_material_count, 1.0F);
        for (std::uint32_t mat = 0; mat < 4; ++mat) {
            for (std::size_t i = 0; i < table_size; ++i) {
                ct_sp_host[mat * table_size + i] =
                    static_cast<float>(stopping_power.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                ct_xs_host[mat * cross_section_table_size + i] =
                    static_cast<float>(cross_section.values()[i]);
            }
        }
        if (use_ct_schneider_xs) {
            for (std::uint32_t section = 0; section < ct_xs_material_count;
                 ++section) {
                const auto& table = schneider_xs_tables[section];
                if (table.values().size() != cross_section_table_size ||
                    table.energies() != cross_section.energies()) {
                    throw std::invalid_argument(
                        "Schneider cross-section tables must match the water grid");
                }
                for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                    ct_xs_host[static_cast<std::size_t>(section) *
                                   cross_section_table_size +
                               i] = static_cast<float>(table.values()[i]);
                }
            }
        }
        if (use_ct_material_sp ||
            (use_ct_material_xs && !use_ct_schneider_xs)) {
            const std::array<std::filesystem::path, 4> sp_paths = {
                config.ct_air_stopping_power_file.empty() ? config.primary_stopping_power_file
                                                          : config.ct_air_stopping_power_file,
                config.ct_lung_stopping_power_file.empty()
                    ? config.primary_stopping_power_file
                    : config.ct_lung_stopping_power_file,
                config.ct_water_stopping_power_file.empty()
                    ? config.primary_stopping_power_file
                    : config.ct_water_stopping_power_file,
                config.ct_bone_stopping_power_file.empty()
                    ? config.primary_stopping_power_file
                    : config.ct_bone_stopping_power_file,
            };
            const std::array<std::filesystem::path, 4> xs_paths = {
                config.ct_air_cross_section_file.empty() ? config.primary_inelastic_cross_section_file
                                                         : config.ct_air_cross_section_file,
                config.ct_lung_cross_section_file.empty()
                    ? config.primary_inelastic_cross_section_file
                    : config.ct_lung_cross_section_file,
                config.ct_water_cross_section_file.empty()
                    ? config.primary_inelastic_cross_section_file
                    : config.ct_water_cross_section_file,
                config.ct_bone_cross_section_file.empty()
                    ? config.primary_inelastic_cross_section_file
                    : config.ct_bone_cross_section_file,
            };
            const std::array<bool, 4> has_native_table = {
                !config.ct_air_stopping_power_file.empty() ||
                    !config.ct_air_cross_section_file.empty(),
                !config.ct_lung_stopping_power_file.empty() ||
                    !config.ct_lung_cross_section_file.empty(),
                !config.ct_water_stopping_power_file.empty() ||
                    !config.ct_water_cross_section_file.empty(),
                !config.ct_bone_stopping_power_file.empty() ||
                    !config.ct_bone_cross_section_file.empty(),
            };
            for (std::uint8_t mat = 0; mat < 4U; ++mat) {
                if (has_native_table[mat]) {
                    ct_ref_host[mat] =
                        ct_material_reference_density_g_per_cm3(mat);
                }
            }
            for (std::uint32_t mat = 0; mat < 4; ++mat) {
                if (use_ct_material_sp) {
                    const auto sp_table = StoppingPowerTable::from_csv(sp_paths[mat]);
                    if (sp_table.values().size() != table_size) {
                        throw std::invalid_argument(
                            "CT material stopping-power tables must match water grid size");
                    }
                    for (std::size_t i = 0; i < table_size; ++i) {
                        ct_sp_host[mat * table_size + i] =
                            static_cast<float>(sp_table.values()[i]);
                    }
                }
                if (use_ct_material_xs) {
                    const auto xs_table = CrossSectionTable::from_csv(xs_paths[mat]);
                    if (xs_table.values().size() != cross_section_table_size) {
                        throw std::invalid_argument(
                            "CT material cross-section tables must match water grid size");
                    }
                    for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                        ct_xs_host[mat * cross_section_table_size + i] =
                            static_cast<float>(xs_table.values()[i]);
                    }
                }
            }
        }
        ct_sp_device = sycl::malloc_device<float>(ct_sp_host.size(), queue);
        ct_xs_device = sycl::malloc_device<float>(ct_xs_host.size(), queue);
        ct_ref_density_device =
            sycl::malloc_device<float>(ct_ref_host.size(), queue);
        if (ct_sp_device == nullptr || ct_xs_device == nullptr ||
            ct_ref_density_device == nullptr) {
            free_device(ct_sp_device);
            free_device(ct_xs_device);
            free_device(ct_ref_density_device);
            free_device(ct_mass_sp_factor_lut_device);
            free_device(ct_mass_sp_za_rel_device);
            free_device(ct_density_device);
            free_device(ct_material_device);
            throw std::bad_alloc();
        }
        queue.memcpy(ct_sp_device, ct_sp_host.data(), sizeof(float) * ct_sp_host.size())
            .wait_and_throw();
        queue.memcpy(ct_xs_device, ct_xs_host.data(), sizeof(float) * ct_xs_host.size())
            .wait_and_throw();
        queue.memcpy(ct_ref_density_device, ct_ref_host.data(),
                     sizeof(float) * ct_ref_host.size())
            .wait_and_throw();
    }
    const auto secondary_allocation_failed =
        enable_secondary_generation &&
        (reaction_bins_device == nullptr || reactions_device == nullptr ||
         reaction_secondaries_device == nullptr || secondary_queue_device == nullptr ||
         secondary_queue_counter_device == nullptr || secondary_queue_filled_device == nullptr ||
         secondary_summaries_device == nullptr ||
         (enable_secondary_transport &&
           (secondary_work_counter_device == nullptr ||
           aggregate_secondary_dose_device == nullptr ||
           (enable_fragment_species_scoring && fragment_dose_device == nullptr) ||
           secondary_deposited_device == nullptr ||
           secondary_escaped_device == nullptr || secondary_steps_device == nullptr ||
           (enable_secondary_energy_sorting &&
            (secondary_bucket_scratch_device == nullptr ||
             secondary_bucket_index_device == nullptr ||
             secondary_bucket_counters_device == nullptr)))) ||
         (enable_fragment_cascade &&
          (cascade_projectiles_device == nullptr || cascade_cross_sections_device == nullptr ||
           cascade_interactions_device == nullptr || cascade_products_device == nullptr ||
           cascade_summaries_device == nullptr || cascade_xs_lut_device == nullptr)));
    const auto neutral_allocation_failed =
        enable_neutral_transport &&
        (neutral_projectiles_device == nullptr || neutral_cross_sections_device == nullptr ||
         neutral_interactions_device == nullptr || neutral_products_device == nullptr ||
         neutral_queue_device == nullptr || neutral_queue_counter_device == nullptr ||
         neutral_queue_filled_device == nullptr || neutral_summaries_device == nullptr ||
         neutral_origin_dose_device == nullptr ||
         (enable_voxel_scoring && neutral_origin_voxel_dose_device == nullptr));
    if (table_device == nullptr || cross_section_device == nullptr || dose_device == nullptr ||
        (use_let_delta_fraction_table && let_delta_fraction_device == nullptr) ||
        (use_particle_specific_stopping_power &&
         (particle_sp_ratio_device == nullptr ||
          particle_delta_fraction_device == nullptr ||
          particle_species_present_device == nullptr)) ||
        (enable_depth_let_scoring && let_moments_device == nullptr) ||
        (enable_voxel_let_scoring &&
         voxel_let_moments_device == nullptr) ||
        (enable_species_let_scoring && species_let_moments_device == nullptr) ||
        (enable_light_isotope_let_scoring &&
         isotope_let_moments_device == nullptr) ||
        (enable_birth_spectrum &&
         (birth_counts_device == nullptr || birth_ke_sum_device == nullptr ||
          birth_mevu_hist_device == nullptr || birth_depth_hist_device == nullptr ||
          birth_cos_hist_device == nullptr ||
          birth_parent_mevu_hist_device == nullptr ||
          birth_parent_z_hist_device == nullptr ||
          birth_parent_product_mevu_hist_device == nullptr)) ||
        (enable_voxel_scoring && voxel_dose_device == nullptr) ||
        (enable_charged_origin_voxel_scoring &&
         charged_origin_voxel_dose_device == nullptr) ||
        deposited_device == nullptr ||
        escaped_device == nullptr || nuclear_device == nullptr || steps_device == nullptr ||
#ifdef CARBON_TRANSPORT_PROFILE
        profile_counters_device == nullptr ||
#endif
        (primary_spot_count > 0 && primary_spots_device == nullptr) ||
        secondary_allocation_failed || neutral_allocation_failed) {
        free_immutable_device(table_device);
        free_immutable_device(cross_section_device);
        free_device(let_delta_fraction_device);
        free_device(particle_sp_ratio_device);
        free_device(particle_delta_fraction_device);
        free_device(particle_species_present_device);
        free_device(dose_device);
        free_device(let_moments_device);
        free_device(voxel_let_moments_device);
        free_device(species_let_moments_device);
        free_device(isotope_let_moments_device);
        free_device(birth_counts_device);
        free_device(birth_ke_sum_device);
        free_device(birth_mevu_hist_device);
        free_device(birth_depth_hist_device);
        free_device(birth_cos_hist_device);
        free_device(birth_parent_mevu_hist_device);
        free_device(birth_parent_z_hist_device);
        free_device(birth_parent_product_mevu_hist_device);
        free_device(voxel_dose_device);
        free_device(charged_origin_voxel_dose_device);
        free_device(deposited_device);
        free_device(escaped_device);
        free_device(nuclear_device);
        free_device(steps_device);
        free_device(profile_counters_device);
        free_device(primary_spots_device);
        free_device(slab_z_ends_device);
        free_device(slab_densities_device);
        free_device(slab_radiation_lengths_device);
        free_device(material_sp_device);
        free_device(material_xs_device);
        free_device(insert_sp_device);
        free_device(insert_xs_device);
        free_device(ct_density_device);
        free_device(ct_material_device);
        free_device(ct_mass_sp_factor_lut_device);
        free_device(ct_mass_sp_za_rel_device);
        free_device(ct_sp_device);
        free_device(ct_xs_device);
        free_device(ct_ref_density_device);
        free_immutable_device(reaction_bins_device);
        free_immutable_device(reactions_device);
        free_immutable_device(reaction_secondaries_device);
        free_device(secondary_queue_device);
        free_device(secondary_bucket_scratch_device);
        free_device(secondary_bucket_index_device);
        free_device(secondary_bucket_counters_device);
        free_device(secondary_queue_counter_device);
        free_device(secondary_queue_filled_device);
        free_device(secondary_work_counter_device);
        free_device(secondary_summaries_device);
        free_device(aggregate_secondary_dose_device);
        free_device(fragment_dose_device);
        free_device(secondary_deposited_device);
        free_device(secondary_escaped_device);
        free_device(secondary_steps_device);
        free_immutable_device(cascade_projectiles_device);
        free_immutable_device(cascade_cross_sections_device);
        free_immutable_device(cascade_interactions_device);
        free_immutable_device(cascade_products_device);
        free_device(cascade_summaries_device);
        free_device(cascade_xs_lut_device);
        free_immutable_device(neutral_projectiles_device);
        free_immutable_device(neutral_cross_sections_device);
        free_immutable_device(neutral_interactions_device);
        free_immutable_device(neutral_products_device);
        free_device(neutral_queue_device);
        free_device(neutral_queue_counter_device);
        free_device(neutral_queue_filled_device);
        free_device(neutral_summaries_device);
        free_device(neutral_origin_dose_device);
        free_device(neutral_origin_voxel_dose_device);
        throw std::runtime_error("SYCL USM device allocation failed");
    }

    if (!reuse_immutable_buffers) {
        std::vector<float> table_host(table_size);
        std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                       table_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(table_host.data(), table_device, table_size);
        std::vector<float> cross_section_host(cross_section_table_size);
        std::transform(cross_section.values().begin(), cross_section.values().end(),
                       cross_section_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(cross_section_host.data(), cross_section_device, cross_section_table_size);
        queue.wait_and_throw();
    }
    if (use_let_delta_fraction_table) {
        queue.copy(let_delta_fraction_host.data(), let_delta_fraction_device,
                   table_size)
            .wait_and_throw();
    }
    if (use_particle_specific_stopping_power) {
        queue.copy(particle_sp_ratio_host.data(), particle_sp_ratio_device,
                   particle_sp_ratio_host.size());
        queue.copy(particle_delta_fraction_host.data(),
                   particle_delta_fraction_device,
                   particle_delta_fraction_host.size());
        queue.copy(particle_species_present_host.data(),
                   particle_species_present_device,
                   particle_species_present_host.size())
            .wait_and_throw();
    }
    if (enable_fragment_cascade && cascade_xs_lut_device != nullptr &&
        cascade_packages != nullptr) {
        auto cascade_xs_lut_host =
            build_cascade_xs_energy_lut(*cascade_packages, stopping_power.energies());
        if (cascade_xs_material_count > 1) {
            const auto water_lut = cascade_xs_lut_host;
            const auto append_material = [&](const auto& material_packages,
                                             const double reference_density) {
                auto material_lut =
                    material_packages
                        ? build_aligned_material_cascade_xs_lut(
                              *cascade_packages, *material_packages,
                              stopping_power.energies(),
                              static_cast<float>(reference_density))
                        : water_lut;
                cascade_xs_lut_host.insert(
                    cascade_xs_lut_host.end(),
                    material_lut.begin(), material_lut.end());
            };
            append_material(
                ct_lung_cascade_xs_packages,
                config.ct_lung_cascade_reference_density_g_per_cm3);
            append_material(
                ct_bone_cascade_xs_packages,
                config.ct_bone_cascade_reference_density_g_per_cm3);
        }
        if (cascade_xs_lut_host.size() != cascade_xs_lut_size) {
            free_device(cascade_xs_lut_device);
            throw std::runtime_error("Cascade XS LUT size mismatch");
        }
        queue.copy(cascade_xs_lut_host.data(), cascade_xs_lut_device, cascade_xs_lut_size)
            .wait_and_throw();
        std::cout << "Cascade XS LUT: " << cascade_packages->projectiles().size()
                  << " projectiles x " << table_size << " energies x "
                  << cascade_xs_material_count
                  << " material blocks on water SP grid\n"
                  << std::flush;
    }
    queue.memset(dose_device, 0, number_of_bins * sizeof(DoseAtomicT));
    if (enable_depth_let_scoring) {
        queue.memset(let_moments_device, 0,
                     4 * number_of_bins * sizeof(LetAtomicT));
    }
    if (enable_voxel_let_scoring) {
        queue.memset(voxel_let_moments_device, 0,
                     4 * number_of_voxels * sizeof(LetAtomicT));
    }
    if (enable_species_let_scoring) {
        queue.memset(
            species_let_moments_device, 0,
            2 * charged_origin_category_count * number_of_bins *
                sizeof(LetAtomicT));
    }
    if (enable_light_isotope_let_scoring) {
        queue.memset(
            isotope_let_moments_device, 0,
            2 * light_isotope_category_count * number_of_bins *
                sizeof(LetAtomicT));
    }
    if (enable_birth_spectrum) {
        queue.memset(birth_counts_device, 0,
                     birth_gen_size * sizeof(std::uint64_t));
        queue.memset(birth_ke_sum_device, 0, birth_gen_size * sizeof(double));
        queue.memset(birth_mevu_hist_device, 0,
                     birth_mevu_size * sizeof(std::uint64_t));
        queue.memset(birth_depth_hist_device, 0,
                     birth_depth_size * sizeof(std::uint64_t));
        queue.memset(birth_cos_hist_device, 0,
                     birth_cos_size * sizeof(std::uint64_t));
        queue.memset(birth_parent_mevu_hist_device, 0,
                     birth_parent_mevu_size * sizeof(std::uint64_t));
        queue.memset(birth_parent_z_hist_device, 0,
                     birth_parent_z_size * sizeof(std::uint64_t));
        queue.memset(birth_parent_product_mevu_hist_device, 0,
                     birth_joint_size * sizeof(std::uint64_t));
    }
#ifdef CARBON_TRANSPORT_PROFILE
    if (profile_counters_device != nullptr) {
        queue.memset(profile_counters_device, 0,
                     transport_profile_slot_count() * sizeof(std::uint64_t));
    }
#endif
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_secondary_generation) {
        if (!reuse_immutable_buffers) {
            queue.copy(reaction_packages->energy_bins().data(), reaction_bins_device,
                       reaction_packages->energy_bins().size());
            queue.copy(reaction_packages->reactions().data(), reactions_device,
                       reaction_packages->reactions().size());
            queue.copy(reaction_packages->secondaries().data(), reaction_secondaries_device,
                       reaction_packages->secondaries().size());
        }
        queue.memset(secondary_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(secondary_queue_filled_device, 0, sizeof(std::uint64_t));
        if (enable_secondary_transport) {
            queue.memset(aggregate_secondary_dose_device, 0,
                         number_of_bins * sizeof(DoseAtomicT));
        }
        if (enable_fragment_species_scoring) {
            queue.memset(fragment_dose_device, 0,
                         fragment_species_count * number_of_bins * sizeof(DoseAtomicT));
        }
        if (enable_fragment_cascade) {
            if (!reuse_immutable_buffers) {
                queue.copy(cascade_packages->projectiles().data(), cascade_projectiles_device,
                           cascade_packages->projectiles().size());
                queue.copy(cascade_packages->cross_sections().data(),
                           cascade_cross_sections_device,
                           cascade_packages->cross_sections().size());
                queue.copy(cascade_packages->interactions().data(), cascade_interactions_device,
                           cascade_packages->interactions().size());
                queue.copy(cascade_packages->products().data(), cascade_products_device,
                           cascade_packages->products().size());
            }
            queue.memset(cascade_summaries_device, 0,
                         secondary_queue_capacity * sizeof(CascadeTransportSummary));
        }
    }
    if (enable_neutral_transport) {
        if (!reuse_immutable_buffers) {
            queue.copy(neutral_packages->projectiles().data(), neutral_projectiles_device,
                       neutral_packages->projectiles().size());
            queue.copy(neutral_packages->cross_sections().data(),
                       neutral_cross_sections_device,
                       neutral_packages->cross_sections().size());
            queue.copy(neutral_packages->interactions().data(), neutral_interactions_device,
                       neutral_packages->interactions().size());
            queue.copy(neutral_packages->products().data(), neutral_products_device,
                       neutral_packages->products().size());
        }
        queue.memset(neutral_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_queue_filled_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_summaries_device, 0,
                     neutral_queue_capacity * sizeof(NeutralTransportSummary));
        queue.memset(neutral_origin_dose_device, 0,
                     neutral_origin_category_count * number_of_bins * sizeof(DoseAtomicT));
        if (enable_voxel_scoring) {
            queue.memset(neutral_origin_voxel_dose_device, 0,
                         neutral_origin_category_count * number_of_voxels * sizeof(DoseAtomicT));
        }
    }

    float* minibeam_copper_sp_device = nullptr;
    float* minibeam_air_sp_device = nullptr;
    float* minibeam_copper_xs_device = nullptr;
    ReactionEnergyBin* minibeam_copper_reaction_bins_device = nullptr;
    ReactionPackage* minibeam_copper_reactions_device = nullptr;
    ReactionSecondary* minibeam_copper_secondaries_device = nullptr;
    float* minibeam_copper_ion_sp_ratio_device = nullptr;
    std::uint8_t* minibeam_copper_ion_sp_present_device = nullptr;
    float* minibeam_copper_ion_xs_device = nullptr;
    std::uint8_t* minibeam_copper_ion_xs_present_device = nullptr;
    float* minibeam_copper_neutral_xs_device = nullptr;
    NeutralProjectile* minibeam_copper_neutral_projectiles_device = nullptr;
    NeutralInteraction* minibeam_copper_neutral_interactions_device = nullptr;
    if (enable_minibeam_copper_em) {
        minibeam_copper_sp_device =
            sycl::malloc_device<float>(table_size, queue);
        minibeam_air_sp_device =
            sycl::malloc_device<float>(table_size, queue);
        if (minibeam_copper_sp_device == nullptr ||
            minibeam_air_sp_device == nullptr) {
            free_device(minibeam_copper_sp_device);
            free_device(minibeam_air_sp_device);
            throw std::runtime_error(
                "SYCL minibeam material stopping-power allocation failed");
        }
        queue.copy(minibeam_copper_sp_host.data(),
                   minibeam_copper_sp_device, table_size);
        queue.copy(minibeam_air_sp_host.data(),
                   minibeam_air_sp_device, table_size)
            .wait_and_throw();
        if (enable_minibeam_copper_nuclear_attenuation) {
            minibeam_copper_xs_device =
                sycl::malloc_device<float>(cross_section_table_size, queue);
            if (minibeam_copper_xs_device == nullptr) {
                free_device(minibeam_copper_sp_device);
                free_device(minibeam_air_sp_device);
                throw std::runtime_error(
                    "SYCL minibeam Copper XS allocation failed");
            }
            queue.copy(minibeam_copper_xs_host.data(),
                       minibeam_copper_xs_device,
                       cross_section_table_size)
                .wait_and_throw();
        }
        if (enable_minibeam_copper_reaction_products) {
            const auto& packages = *minibeam_copper_reaction_packages;
            minibeam_copper_reaction_bins_device =
                sycl::malloc_device<ReactionEnergyBin>(
                    packages.energy_bins().size(), queue);
            minibeam_copper_reactions_device =
                sycl::malloc_device<ReactionPackage>(
                    packages.reactions().size(), queue);
            minibeam_copper_secondaries_device =
                sycl::malloc_device<ReactionSecondary>(
                    packages.secondaries().size(), queue);
            minibeam_copper_ion_sp_ratio_device =
                sycl::malloc_device<float>(
                    minibeam_copper_ion_sp_ratio_host.size(), queue);
            minibeam_copper_ion_sp_present_device =
                sycl::malloc_device<std::uint8_t>(
                    minibeam_copper_ion_sp_present_host.size(), queue);
            minibeam_copper_ion_xs_device =
                sycl::malloc_device<float>(
                    minibeam_copper_ion_xs_host.size(), queue);
            minibeam_copper_ion_xs_present_device =
                sycl::malloc_device<std::uint8_t>(
                    minibeam_copper_ion_xs_present_host.size(), queue);
            if (enable_neutral_transport) {
                minibeam_copper_neutral_xs_device =
                    sycl::malloc_device<float>(
                        minibeam_copper_neutral_xs_host.size(), queue);
                minibeam_copper_neutral_projectiles_device =
                    sycl::malloc_device<NeutralProjectile>(
                        minibeam_copper_neutral_packages->projectiles().size(),
                        queue);
                minibeam_copper_neutral_interactions_device =
                    sycl::malloc_device<NeutralInteraction>(
                        minibeam_copper_neutral_packages->interactions().size(),
                        queue);
            }
            if (minibeam_copper_reaction_bins_device == nullptr ||
                minibeam_copper_reactions_device == nullptr ||
                minibeam_copper_secondaries_device == nullptr ||
                minibeam_copper_ion_sp_ratio_device == nullptr ||
                minibeam_copper_ion_sp_present_device == nullptr ||
                minibeam_copper_ion_xs_device == nullptr ||
                minibeam_copper_ion_xs_present_device == nullptr ||
                (enable_neutral_transport &&
                 (minibeam_copper_neutral_xs_device == nullptr ||
                  minibeam_copper_neutral_projectiles_device == nullptr ||
                  minibeam_copper_neutral_interactions_device == nullptr))) {
                free_device(minibeam_copper_reaction_bins_device);
                free_device(minibeam_copper_reactions_device);
                free_device(minibeam_copper_secondaries_device);
                free_device(minibeam_copper_ion_sp_ratio_device);
                free_device(minibeam_copper_ion_sp_present_device);
                free_device(minibeam_copper_ion_xs_device);
                free_device(minibeam_copper_ion_xs_present_device);
                free_device(minibeam_copper_neutral_xs_device);
                free_device(minibeam_copper_neutral_projectiles_device);
                free_device(minibeam_copper_neutral_interactions_device);
                throw std::runtime_error(
                    "SYCL minibeam Copper reaction-package allocation failed");
            }
            queue.copy(
                packages.energy_bins().data(),
                minibeam_copper_reaction_bins_device,
                packages.energy_bins().size());
            queue.copy(
                packages.reactions().data(),
                minibeam_copper_reactions_device,
                packages.reactions().size());
            queue.copy(
                packages.secondaries().data(),
                minibeam_copper_secondaries_device,
                packages.secondaries().size());
            queue.copy(
                minibeam_copper_ion_sp_ratio_host.data(),
                minibeam_copper_ion_sp_ratio_device,
                minibeam_copper_ion_sp_ratio_host.size());
            queue.copy(
                minibeam_copper_ion_sp_present_host.data(),
                minibeam_copper_ion_sp_present_device,
                minibeam_copper_ion_sp_present_host.size());
            queue.copy(
                minibeam_copper_ion_xs_host.data(),
                minibeam_copper_ion_xs_device,
                minibeam_copper_ion_xs_host.size());
            queue.copy(
                minibeam_copper_ion_xs_present_host.data(),
                minibeam_copper_ion_xs_present_device,
                minibeam_copper_ion_xs_present_host.size());
            if (enable_neutral_transport) {
                queue.copy(
                    minibeam_copper_neutral_xs_host.data(),
                    minibeam_copper_neutral_xs_device,
                    minibeam_copper_neutral_xs_host.size());
                queue.copy(
                    minibeam_copper_neutral_packages->projectiles().data(),
                    minibeam_copper_neutral_projectiles_device,
                    minibeam_copper_neutral_packages->projectiles().size());
                queue.copy(
                    minibeam_copper_neutral_packages->interactions().data(),
                    minibeam_copper_neutral_interactions_device,
                    minibeam_copper_neutral_packages->interactions().size());
            }
            queue.wait_and_throw();
        }
    }
    constexpr std::size_t minibeam_diagnostic_count_size =
        8 + minibeam_charged_species_count +
        3 * MinibeamDiagnostics::slit_count +
        MinibeamDiagnostics::touched_energy_bin_count +
        3 * MinibeamDiagnostics::fragment_energy_bin_count;
    constexpr std::size_t minibeam_diagnostic_moment_size =
        15 + minibeam_charged_species_count;
    auto* minibeam_diagnostic_counts_device =
        config.enable_minibeam && config.enable_minibeam_diagnostics
            ? sycl::malloc_device<std::uint64_t>(
                  minibeam_diagnostic_count_size, queue)
            : nullptr;
    auto* minibeam_diagnostic_moments_device =
        config.enable_minibeam
            ? sycl::malloc_device<double>(
                  minibeam_diagnostic_moment_size, queue)
            : nullptr;
    if (config.enable_minibeam &&
        (minibeam_diagnostic_moments_device == nullptr ||
         (config.enable_minibeam_diagnostics &&
          minibeam_diagnostic_counts_device == nullptr))) {
        free_device(minibeam_diagnostic_counts_device);
        free_device(minibeam_diagnostic_moments_device);
        free_device(minibeam_copper_sp_device);
        free_device(minibeam_air_sp_device);
        free_device(minibeam_copper_xs_device);
        free_device(minibeam_copper_reaction_bins_device);
        free_device(minibeam_copper_reactions_device);
        free_device(minibeam_copper_secondaries_device);
        free_device(minibeam_copper_ion_sp_ratio_device);
        free_device(minibeam_copper_ion_sp_present_device);
        free_device(minibeam_copper_ion_xs_device);
        free_device(minibeam_copper_ion_xs_present_device);
        throw std::runtime_error(
            "SYCL minibeam diagnostic allocation failed");
    }
    if (config.enable_minibeam) {
        if (config.enable_minibeam_diagnostics) {
            queue.memset(minibeam_diagnostic_counts_device, 0,
                         minibeam_diagnostic_count_size *
                             sizeof(std::uint64_t));
        }
        queue.memset(minibeam_diagnostic_moments_device, 0,
                     minibeam_diagnostic_moment_size * sizeof(double))
            .wait_and_throw();
    }

    // GPU backends (CUDA / Level Zero / OpenCL GPU / --device gpu|cuda|...) use larger
    // work-groups; SYCL CPU and other selectors stay at 128. CUDA stays at 128:
    // the transport kernels are register-heavy (FP64 atomics + cascade) and 256
    // often hurts occupancy while making each submit longer under WSL.
    const bool prefer_large_workgroup =
        !is_cuda_backend &&
        (device.is_gpu() || device_name == "gpu" || device_name == "cuda" ||
         device_name == "nvidia" || device_name == "level_zero" || device_name == "intel" ||
         device_name == "arc" || device_name == "opencl");
    const std::size_t local_size = prefer_large_workgroup ? 256U : 128U;
    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        if (is_cuda_backend) {
            // Prefer large primary chunks for 100k–10M plans.
            history_chunk = number_of_histories > 1'000'000 ? 16384 : 4096;
        } else if (device.is_gpu()) {
            history_chunk = 8192;
        } else {
            history_chunk = number_of_histories;
        }
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);
    if (is_cuda_backend) {
        history_chunk = std::min(history_chunk, std::size_t{65536});
    }
    std::size_t secondary_batch = config.secondary_batch_size;
    if (secondary_batch == 0) {
        // CUDA secondary transport is dominated by long, divergent tracks.  A
        // 64k launch provides enough resident warps to hide that latency even
        // for sub-million-history jobs (100k CT-plan A/B: 8k=40.9 s,
        // 32k=12.0 s, 64k=8.1 s with identical histories/steps).
        secondary_batch = is_cuda_backend
                              ? 65536
                              : std::numeric_limits<std::size_t>::max() / 4;
    }
    if (is_cuda_backend) {
        secondary_batch = std::min(secondary_batch, std::size_t{65536});
    }
    if (enable_secondary_energy_sorting) {
        secondary_batch = std::min(
            secondary_batch,
            std::min(secondary_queue_capacity, std::size_t{65536}));
    }
    secondary_batch = std::max<std::size_t>(local_size, secondary_batch);
    const auto secondary_persistent_workers =
        config.secondary_persistent_workers == 0
            ? std::size_t{0}
            : std::max(
                  local_size,
                  std::min(config.secondary_persistent_workers,
                           secondary_batch));
    const auto secondary_fp32_energy_residual =
        config.secondary_fp32_energy_residual;
    const auto robust_boundary_nudge = config.robust_boundary_nudge;

    // Throttle progress I/O: 10M plans previously printed ~100k+ lines/sec of
    // secondary batch logs, which dominated wall time on WSL.
    const auto progress_log_every_histories =
        std::max<std::size_t>(history_chunk, number_of_histories / 50);
    const auto progress_log_every_secondary =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(secondary_batch),
                                250'000ULL);

    std::cout << "Launch plan: histories=" << number_of_histories
              << " history_chunk=" << history_chunk
              << " secondary_batch=" << secondary_batch
              << " secondary_persistent_workers="
              << secondary_persistent_workers
              << " local_size=" << local_size
              << " secondary_energy_sorting="
              << (enable_secondary_energy_sorting ? "on" : "off")
              << (is_cuda_backend ? " [CUDA]" : "")
              << (k_dose_atomic_fp32 ? " dose_atomic=fp32" : " dose_atomic=fp64")
              << (k_let_atomic_fp32 ? " let_atomic=fp32" : " let_atomic=fp64")
              << '\n'
              << std::flush;
    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);
    const auto secondary_local_deposit_cutoff_MeV = static_cast<float>(
        config.effective_secondary_local_deposit_cutoff_MeV());
    // 0 = transport all supported charged Z; else Z >= min deposit as local heat.
    const auto secondary_heavy_local_deposit_z_min =
        config.secondary_heavy_local_deposit_z_min;
    const auto secondary_condensed_step_mm =
        static_cast<float>(config.secondary_condensed_step_mm);
    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_step_stable_straggling =
        enable_energy_straggling && config.enable_step_stable_straggling;
    const auto straggling_sampling_length_mm =
        static_cast<float>(config.straggling_sampling_length_mm);
    const auto enable_secondary_energy_straggling =
        enable_energy_straggling &&
        config.enable_secondary_energy_straggling;
    const auto enable_secondary_condensed_history =
        secondary_condensed_step_mm > 0.0F &&
        !enable_let_scoring && !config.enable_ct_grid &&
        !config.enable_layered_phantom &&
        !config.enable_hetero_insert;
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto straggling_sampler = config.straggling_sampler_id();
    std::array<float, max_straggling_scale_points> straggling_scale_energies{};
    std::array<float, max_straggling_scale_points> straggling_scale_values{};
    std::array<float, max_straggling_scale_points> primary_xs_correction_energies{};
    std::array<float, max_straggling_scale_points> primary_xs_correction_scales{};
    const auto straggling_scale_point_count =
        config.straggling_scale_energies_MeVu.size();
    for (std::size_t index = 0; index < straggling_scale_point_count; ++index) {
        straggling_scale_energies[index] = static_cast<float>(
            config.straggling_scale_energies_MeVu[index]);
        straggling_scale_values[index] = static_cast<float>(
            config.straggling_scale_values[index]);
    }
    const auto primary_xs_correction_point_count =
        config.primary_inelastic_xs_correction_energies_MeVu.size();
    for (std::size_t index = 0; index < primary_xs_correction_point_count; ++index) {
        primary_xs_correction_energies[index] = static_cast<float>(
            config.primary_inelastic_xs_correction_energies_MeVu[index]);
        primary_xs_correction_scales[index] = static_cast<float>(
            config.primary_inelastic_xs_correction_scales[index]);
    }
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    const auto enable_tps_source = config.uses_fixed_patient_coordinates();
    const auto enable_minibeam = config.enable_minibeam;
    const auto enable_minibeam_diagnostics =
        config.enable_minibeam_diagnostics;
    const auto minibeam_absorbing_geometry =
        enable_minibeam &&
        config.minibeam_transport_mode == "absorbing_geometry";
    constexpr float degrees_to_radians = 0.01745329251994329577F;
    const auto minibeam_angle_radians =
        static_cast<float>(config.minibeam_collimator_angle_deg) *
        degrees_to_radians;
    const auto minibeam_cosine_angle =
        static_cast<float>(std::cos(minibeam_angle_radians));
    const auto minibeam_sine_angle =
        static_cast<float>(std::sin(minibeam_angle_radians));
    const auto minibeam_radius_mm =
        static_cast<float>(config.minibeam_radius_mm);
    const auto minibeam_thickness_mm =
        static_cast<float>(config.minibeam_thickness_mm);
    const auto minibeam_exit_to_phantom_mm =
        static_cast<float>(config.minibeam_exit_to_phantom_mm);
    const auto minibeam_slit_count = config.minibeam_slit_count;
    const auto minibeam_slit_width_mm =
        static_cast<float>(config.minibeam_slit_width_mm);
    const auto minibeam_slit_pitch_mm =
        static_cast<float>(config.minibeam_slit_pitch_mm);
    const auto minibeam_slit_half_length_mm =
        static_cast<float>(config.minibeam_slit_half_length_mm);
    const auto minibeam_slit_offset_mm =
        static_cast<float>(config.minibeam_slit_offset_mm);
    const auto minibeam_slit_offset_x_mm =
        minibeam_cosine_angle * minibeam_slit_offset_mm;
    const auto minibeam_slit_offset_y_mm =
        minibeam_sine_angle * minibeam_slit_offset_mm;
    const auto minibeam_copper_density_g_per_cm3 =
        static_cast<float>(config.minibeam_copper_density_g_per_cm3);
    const auto minibeam_copper_radiation_length_g_per_cm2 =
        static_cast<float>(
            config.minibeam_copper_radiation_length_g_per_cm2);
    const auto minibeam_copper_max_step_mm =
        static_cast<float>(config.minibeam_copper_max_step_mm);
    const auto minibeam_copper_enable_mcs =
        config.minibeam_copper_enable_mcs;
    const auto minibeam_copper_enable_energy_straggling =
        config.minibeam_copper_enable_energy_straggling;
    const auto minibeam_copper_straggling_scale =
        static_cast<float>(config.minibeam_copper_straggling_scale);
    const auto minibeam_copper_mcs_scale =
        static_cast<float>(config.minibeam_copper_mcs_scale);
    const auto minibeam_copper_survivor_energy_loss_scale =
        static_cast<float>(
            config.minibeam_copper_survivor_energy_loss_scale);
    constexpr std::size_t max_minibeam_copper_survivor_calibration_points =
        16;
    std::array<
        float, max_minibeam_copper_survivor_calibration_points>
        minibeam_copper_survivor_calibration_energies{};
    std::array<
        float, max_minibeam_copper_survivor_calibration_points>
        minibeam_copper_survivor_calibration_scales{};
    const auto minibeam_copper_survivor_calibration_count =
        config.minibeam_copper_survivor_energy_loss_energies_MeVu.size();
    for (std::size_t index = 0;
         index < minibeam_copper_survivor_calibration_count; ++index) {
        minibeam_copper_survivor_calibration_energies[index] =
            static_cast<float>(
                config
                    .minibeam_copper_survivor_energy_loss_energies_MeVu[
                        index]);
        minibeam_copper_survivor_calibration_scales[index] =
            static_cast<float>(
                config.minibeam_copper_survivor_energy_loss_scales[
                    index]);
    }
    const auto minibeam_water_primary_stopping_power_scale =
        static_cast<float>(
            config.minibeam_water_primary_stopping_power_scale);
    const auto minibeam_water_low_energy_mcs_transition_MeVu =
        static_cast<float>(
            config.minibeam_water_low_energy_mcs_transition_MeVu);
    const auto minibeam_water_primary_low_energy_mcs_scale =
        static_cast<float>(
            config.minibeam_water_primary_low_energy_mcs_scale);
    const auto minibeam_water_fragment_low_energy_mcs_scale =
        static_cast<float>(
            config.minibeam_water_fragment_low_energy_mcs_scale);
    const auto minibeam_water_touched_primary_surface_boost =
        static_cast<float>(
            config.minibeam_water_touched_primary_surface_boost);
    const auto minibeam_water_touched_primary_surface_sigma_mm =
        static_cast<float>(
            config.minibeam_water_touched_primary_surface_sigma_mm);
    const auto minibeam_water_touched_primary_deficit =
        static_cast<float>(
            config.minibeam_water_touched_primary_deficit);
    const auto minibeam_water_touched_primary_deficit_center_mm =
        static_cast<float>(
            config
                .minibeam_water_touched_primary_deficit_center_mm);
    const auto minibeam_water_touched_primary_deficit_sigma_mm =
        static_cast<float>(
            config
                .minibeam_water_touched_primary_deficit_sigma_mm);
    const auto random_seed = config.random_seed;
    const auto enable_primary_attenuation = config.enable_primary_attenuation;
    const auto primary_inelastic_xs_scale =
        static_cast<float>(config.primary_inelastic_xs_scale);
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime =
        static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime =
        static_cast<float>(config.emittance_sigma_y_prime);
    const auto source_origin_x_mm = static_cast<float>(config.source_origin_x_mm);
    const auto source_origin_y_mm = static_cast<float>(config.source_origin_y_mm);
    const auto source_origin_z_mm = static_cast<float>(config.source_origin_z_mm);
    const auto beam_ux_x = static_cast<float>(config.beam_ux_x);
    const auto beam_ux_y = static_cast<float>(config.beam_ux_y);
    const auto beam_ux_z = static_cast<float>(config.beam_ux_z);
    const auto beam_uy_x = static_cast<float>(config.beam_uy_x);
    const auto beam_uy_y = static_cast<float>(config.beam_uy_y);
    const auto beam_uy_z = static_cast<float>(config.beam_uy_z);
    const auto beam_uz_x = static_cast<float>(config.beam_uz_x);
    const auto beam_uz_y = static_cast<float>(config.beam_uz_y);
    const auto beam_uz_z = static_cast<float>(config.beam_uz_z);
    const auto emittance_correlation_x =
        static_cast<float>(config.emittance_correlation_x);
    const auto emittance_correlation_y =
        static_cast<float>(config.emittance_correlation_y);
    const auto neutral_local_kerma_fraction =
        static_cast<float>(config.neutral_local_kerma_fraction);
    const auto neutral_kerma_high_energy_scale =
        static_cast<float>(config.neutral_kerma_high_energy_scale);
    const auto neutral_kerma_mean_free_path_mm =
        static_cast<float>(config.neutral_kerma_mean_free_path_mm);
    const auto electronic_buildup_fraction =
        static_cast<float>(config.electronic_buildup_fraction);
    const auto electronic_buildup_mfp_mm =
        static_cast<float>(config.electronic_buildup_mfp_mm);
    const auto minibeam_electronic_buildup_primary_only =
        config.minibeam_electronic_buildup_primary_only;
    const auto electronic_buildup_lateral_sigma_mm =
        static_cast<float>(config.electronic_buildup_lateral_sigma_mm);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_charge = static_cast<float>(primary_atomic_number);
    const auto primary_charge_power = static_cast<float>(
        config.primary_ion().charge_power);
    const auto use_explicit_primary_rest_mass =
        config.primary_rest_mass_MeV > 0.0;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);
    const auto minimum_cross_section_energy =
        static_cast<float>(cross_section.energies().front());
    const auto inverse_cross_section_step =
        1.0f / static_cast<float>(cross_section.energies()[1] - cross_section.energies()[0]);
    const auto minibeam_copper_reaction_bin_count =
        enable_minibeam_copper_reaction_products
            ? static_cast<std::uint32_t>(
                  minibeam_copper_reaction_packages->energy_bins().size())
            : 0U;
    const auto minibeam_copper_reaction_minimum_energy =
        enable_minibeam_copper_reaction_products
            ? minibeam_copper_reaction_packages->minimum_energy_MeV_per_u()
            : 0.0F;
    const auto minibeam_copper_inverse_reaction_bin_width =
        enable_minibeam_copper_reaction_products
            ? 1.0F /
                  minibeam_copper_reaction_packages
                      ->energy_bin_width_MeV_per_u()
            : 0.0F;
    const auto reaction_energy_bin_count =
        enable_secondary_generation
            ? static_cast<std::uint32_t>(reaction_packages->energy_bins().size())
            : 0U;
    const auto minimum_reaction_energy =
        enable_secondary_generation ? reaction_packages->minimum_energy_MeV_per_u() : 0.0F;
    const auto inverse_reaction_energy_bin_width =
        enable_secondary_generation
            ? 1.0F / reaction_packages->energy_bin_width_MeV_per_u()
            : 0.0F;
    const auto secondary_queue_capacity_u32 =
        static_cast<std::uint32_t>(secondary_queue_capacity);
    const auto neutral_queue_capacity_u32 =
        static_cast<std::uint32_t>(neutral_queue_capacity);
    const auto cascade_projectile_count = enable_fragment_cascade
                                              ? cascade_packages->projectiles().size()
                                              : std::size_t{0};
    const auto cascade_depth_conditioned_layout =
        enable_fragment_cascade &&
        !cascade_packages->interactions().empty() &&
        std::isfinite(cascade_packages->interactions().front().depth_mm);
    const auto cascade_condition_on_reference_depth =
        config.cascade_condition_on_reference_depth;
    const auto cascade_allow_nearest_fallback =
        config.cascade_selection_policy ==
        CascadeSelectionPolicy::legacy_nearest;
    const auto neutral_projectile_count = enable_neutral_transport
                                              ? neutral_packages->projectiles().size()
                                              : std::size_t{0};
    const auto maximum_cascade_generations = config.maximum_cascade_generations;
    const auto maximum_neutral_generations = config.maximum_neutral_generations;

    for (std::size_t hist_offset = 0; hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        auto kernel_event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) {
            const auto lane = item.get_global_linear_id();
            if (lane >= chunk_count) {
                return;
            }
            const auto global_history = hist_offset + lane;

            const PrimarySpotBatchEntry* spot = nullptr;
            std::uint64_t rng_history = global_history;
            if (primary_spot_count > 0) {
                std::size_t lower = 0;
                std::size_t upper = primary_spot_count;
                while (lower + 1 < upper) {
                    const auto middle = lower + (upper - lower) / 2;
                    if (global_history < primary_spots_device[middle].history_begin) {
                        upper = middle;
                    } else {
                        lower = middle;
                    }
                }
                spot = primary_spots_device + lower;
                rng_history = global_history - spot->history_begin;
            }
            const auto spot_seed = spot != nullptr ? spot->random_seed : random_seed;
            const auto spot_initial_energy_MeV =
                spot != nullptr ? spot->floats[0] : initial_energy_MeV;
            const auto spot_energy_spread =
                spot != nullptr ? spot->floats[1] : beam_energy_spread;

            auto energy_MeV = spot_initial_energy_MeV;
            if (spot_energy_spread > 0.0F) {
                // TOPAS BeamEnergySpread: Gaussian RMS = spread * mean total KE.
                const auto u0 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 40), 1.0e-12F);
                const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 41);
                constexpr float two_pi = 6.2831853071795864769F;
                const auto gauss =
                    sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                energy_MeV =
                    spot_initial_energy_MeV * (1.0F + spot_energy_spread * gauss);
                if (energy_MeV < energy_cutoff_MeV) {
                    energy_MeV = energy_cutoff_MeV;
                }
            }
            const auto sampled_initial_energy_MeV = energy_MeV;
            const auto history_primary_inelastic_xs_scale =
                interpolate_straggling_scale(
                    energy_MeV * inverse_mass_number,
                    primary_xs_correction_energies,
                    primary_xs_correction_scales,
                    primary_xs_correction_point_count,
                    primary_inelastic_xs_scale);
            // Local beam frame (defaults: origin 0, +z beam). Emittance is local x/y.
            auto local_x_mm = 0.0F;
            auto local_y_mm = 0.0F;
            auto local_dx = 0.0F;
            auto local_dy = 0.0F;
            auto local_dz = 1.0F;
            if (enable_flat_source) {
                local_x_mm = flat_source_half_width_x_mm *
                             (2.0F * rng::uniform01(spot_seed, rng_history, 0, 34) - 1.0F);
                local_y_mm = flat_source_half_width_y_mm *
                             (2.0F * rng::uniform01(spot_seed, rng_history, 0, 35) - 1.0F);
            } else if (enable_emittance_source) {
                // TOPAS BiGaussian: sample (x,x') and (y,y') from bivariate normals.
                // x' = dx/dz (unitless, rad-like). Independent axes with correlations.
                const auto u0 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 30), 1.0e-12F);
                const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 31);
                const auto u2 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 32), 1.0e-12F);
                const auto u3 = rng::uniform01(spot_seed, rng_history, 0, 33);
                constexpr float two_pi = 6.2831853071795864769F;
                const auto g0 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                const auto g1 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::sin(two_pi * u1);
                const auto g2 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                const auto g3 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::sin(two_pi * u3);
                const auto sigma_x = spot != nullptr ? spot->floats[2]
                                                     : emittance_sigma_x_mm;
                const auto sigma_y = spot != nullptr ? spot->floats[3]
                                                     : emittance_sigma_y_mm;
                const auto sigma_x_prime =
                    spot != nullptr ? spot->floats[4]
                                    : emittance_sigma_x_prime;
                const auto sigma_y_prime =
                    spot != nullptr ? spot->floats[5]
                                    : emittance_sigma_y_prime;
                local_x_mm = sigma_x * g0;
                local_y_mm = sigma_y * g2;
                const auto rho_x = sycl::clamp(
                    spot != nullptr ? spot->floats[6]
                                    : emittance_correlation_x,
                    -0.9999F, 0.9999F);
                const auto rho_y = sycl::clamp(
                    spot != nullptr ? spot->floats[7]
                                    : emittance_correlation_y,
                    -0.9999F, 0.9999F);
                const auto x_prime =
                    sigma_x_prime *
                    (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                const auto y_prime =
                    sigma_y_prime *
                    (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                // Paraxial unit direction from slopes (dx/dz, dy/dz).
                const auto inv_norm =
                    sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                local_dx = x_prime * inv_norm;
                local_dy = y_prime * inv_norm;
                local_dz = inv_norm;
            }
            // World = origin + ux*x + uy*y ; dir = ux*dx + uy*dy + uz*dz
            const auto origin_x =
                spot != nullptr ? spot->floats[8] : source_origin_x_mm;
            const auto origin_y =
                spot != nullptr ? spot->floats[9] : source_origin_y_mm;
            const auto origin_z =
                spot != nullptr ? spot->floats[10] : source_origin_z_mm;
            const auto ux_x = spot != nullptr ? spot->floats[11] : beam_ux_x;
            const auto ux_y = spot != nullptr ? spot->floats[12] : beam_ux_y;
            const auto ux_z = spot != nullptr ? spot->floats[13] : beam_ux_z;
            const auto uy_x = spot != nullptr ? spot->floats[14] : beam_uy_x;
            const auto uy_y = spot != nullptr ? spot->floats[15] : beam_uy_y;
            const auto uy_z = spot != nullptr ? spot->floats[16] : beam_uy_z;
            const auto uz_x = spot != nullptr ? spot->floats[17] : beam_uz_x;
            const auto uz_y = spot != nullptr ? spot->floats[18] : beam_uz_y;
            const auto uz_z = spot != nullptr ? spot->floats[19] : beam_uz_z;
            auto position_x_mm =
                origin_x + ux_x * local_x_mm + uy_x * local_y_mm;
            auto position_y_mm =
                origin_y + ux_y * local_x_mm + uy_y * local_y_mm;
            auto position_z_mm =
                origin_z + ux_z * local_x_mm + uy_z * local_y_mm;
            auto direction_x =
                ux_x * local_dx + uy_x * local_dy + uz_x * local_dz;
            auto direction_y =
                ux_y * local_dx + uy_y * local_dy + uz_y * local_dz;
            auto direction_z =
                ux_z * local_dx + uy_z * local_dy + uz_z * local_dz;
            {
                const auto inv_n = sycl::rsqrt(sycl::fmax(
                    1.0e-20F,
                    direction_x * direction_x + direction_y * direction_y +
                        direction_z * direction_z));
                direction_x *= inv_n;
                direction_y *= inv_n;
                direction_z *= inv_n;
            }
            const auto add_minibeam_count =
                [&](const std::size_t index,
                    const std::uint64_t count = 1ULL) {
                    if (!enable_minibeam_diagnostics ||
                        minibeam_diagnostic_counts_device == nullptr) {
                        return;
                    }
                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        value(minibeam_diagnostic_counts_device[index]);
                    value.fetch_add(count);
                };
            const auto add_minibeam_moment =
                [&](const std::size_t index, const double value_to_add) {
                    if ((!enable_minibeam_diagnostics && index != 10) ||
                        minibeam_diagnostic_moments_device == nullptr) {
                        return;
                    }
                    sycl::atomic_ref<
                        double, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        value(minibeam_diagnostic_moments_device[index]);
                    value.fetch_add(value_to_add);
                };
            const auto record_beamline_removed =
                [&](const float energy_reaching_water_MeV) {
                    add_minibeam_moment(
                        10, static_cast<double>(
                                sycl::fmax(
                                    0.0F,
                                    sampled_initial_energy_MeV -
                                        energy_reaching_water_MeV)));
                };
            const auto minibeam_stopping_power =
                [&](const float* material_table,
                    const float kinetic_energy_MeV) {
                    auto floating_index =
                        (kinetic_energy_MeV * inverse_mass_number -
                         minimum_table_energy) *
                        inverse_table_step;
                    auto index =
                        static_cast<int>(sycl::floor(floating_index));
                    index = sycl::max(
                        0, sycl::min(index,
                                     static_cast<int>(table_size) - 2));
                    const auto fraction = sycl::clamp(
                        floating_index - static_cast<float>(index),
                        0.0F, 1.0F);
                    return material_table[index] +
                           fraction *
                               (material_table[index + 1] -
                                material_table[index]);
                };
            auto minibeam_direct = false;
            if (enable_minibeam) {
                add_minibeam_count(0);
                minibeam_direct = minibeam_straight_through_air_slit(
                    position_x_mm - minibeam_slit_offset_x_mm,
                    position_y_mm - minibeam_slit_offset_y_mm,
                    position_z_mm, direction_x,
                    direction_y, direction_z, minibeam_cosine_angle,
                    minibeam_sine_angle, minibeam_radius_mm,
                    minibeam_thickness_mm, minibeam_exit_to_phantom_mm,
                    minibeam_slit_count, minibeam_slit_width_mm,
                    minibeam_slit_pitch_mm, minibeam_slit_half_length_mm);
                if (minibeam_direct) {
                    add_minibeam_count(1);
                }
            }
            if (minibeam_absorbing_geometry &&
                !minibeam_direct) {
                record_beamline_removed(0.0F);
                return;
            }
            if (enable_minibeam_copper_em) {
                if (direction_z <= 1.0e-8F) {
                    record_beamline_removed(0.0F);
                    return;
                }
                const auto collimator_entrance_z =
                    -(minibeam_exit_to_phantom_mm +
                      minibeam_thickness_mm);
                const auto collimator_exit_z =
                    -minibeam_exit_to_phantom_mm;
                const auto entrance_distance =
                    (collimator_entrance_z - position_z_mm) / direction_z;
                if (entrance_distance < 0.0F) {
                    record_beamline_removed(0.0F);
                    return;
                }
                const auto entrance_air_loss =
                    minibeam_stopping_power(
                        minibeam_air_sp_device, energy_MeV) *
                    entrance_distance;
                if (entrance_air_loss >=
                    energy_MeV - energy_cutoff_MeV) {
                    record_beamline_removed(0.0F);
                    return;
                }
                energy_MeV -= entrance_air_loss;
                const auto energy_at_collimator_entrance_MeV =
                    energy_MeV;
                position_x_mm += entrance_distance * direction_x;
                position_y_mm += entrance_distance * direction_y;
                position_z_mm = collimator_entrance_z;
                const auto entrance_slit_u_mm =
                    minibeam_cosine_angle * position_x_mm +
                    minibeam_sine_angle * position_y_mm -
                    minibeam_slit_offset_mm;
                const auto entrance_nearest_slit = nearest_minibeam_slit(
                    entrance_slit_u_mm, minibeam_slit_pitch_mm);
                const auto entrance_slit_array_index =
                    entrance_nearest_slit + minibeam_slit_count / 2;
                if (entrance_slit_array_index >= 0 &&
                    entrance_slit_array_index <
                        static_cast<int>(MinibeamDiagnostics::slit_count)) {
                    add_minibeam_count(
                        8 + minibeam_charged_species_count +
                        MinibeamDiagnostics::slit_count +
                        static_cast<std::size_t>(
                            entrance_slit_array_index));
                    if (minibeam_direct) {
                        add_minibeam_count(
                            8 + minibeam_charged_species_count +
                            2 * MinibeamDiagnostics::slit_count +
                            static_cast<std::size_t>(
                                entrance_slit_array_index));
                    }
                }

                auto copper_segment_path_mm = 0.0F;
                auto copper_touched = false;
                std::uint64_t beamline_step = 0;
                constexpr std::uint64_t maximum_beamline_steps = 100000;
                while (position_z_mm < collimator_exit_z - 1.0e-6F &&
                       beamline_step < maximum_beamline_steps) {
                    if (direction_z <= 1.0e-8F ||
                        energy_MeV <= energy_cutoff_MeV) {
                        record_beamline_removed(0.0F);
                        return;
                    }
                    const auto axial_step_mm = sycl::fmin(
                        minibeam_copper_max_step_mm,
                        collimator_exit_z - position_z_mm);
                    const auto path_step_mm =
                        axial_step_mm / direction_z;
                    const auto midpoint_x =
                        position_x_mm + 0.5F * path_step_mm * direction_x;
                    const auto midpoint_y =
                        position_y_mm + 0.5F * path_step_mm * direction_y;
                    const auto in_copper = minibeam_point_in_copper(
                        midpoint_x - minibeam_slit_offset_x_mm,
                        midpoint_y - minibeam_slit_offset_y_mm,
                        minibeam_cosine_angle,
                        minibeam_sine_angle, minibeam_radius_mm,
                        minibeam_slit_count, minibeam_slit_width_mm,
                        minibeam_slit_pitch_mm,
                        minibeam_slit_half_length_mm);

                    position_x_mm += path_step_mm * direction_x;
                    position_y_mm += path_step_mm * direction_y;
                    position_z_mm += axial_step_mm;
                    if (!in_copper) {
                        copper_segment_path_mm = 0.0F;
                        ++beamline_step;
                        continue;
                    }
                    if (!copper_touched) {
                        copper_touched = true;
                        add_minibeam_count(2);
                    }

                    const auto copper_stopping_power =
                        minibeam_stopping_power(
                            minibeam_copper_sp_device, energy_MeV);
                    const auto mean_loss_MeV =
                        copper_stopping_power * path_step_mm;
                    if (mean_loss_MeV >=
                        energy_MeV - energy_cutoff_MeV) {
                        record_beamline_removed(0.0F);
                        return;
                    }
                    auto sampled_loss_MeV = mean_loss_MeV;
                    if (minibeam_copper_enable_energy_straggling) {
                        const auto uniform1 = sycl::fmax(
                            rng::uniform01(
                                spot_seed, rng_history, beamline_step, 40),
                            1.0e-12F);
                        const auto uniform2 = rng::uniform01(
                            spot_seed, rng_history, beamline_step, 41);
                        const auto extra_uniform = rng::uniform01(
                            spot_seed, rng_history, beamline_step, 42);
                        constexpr float two_pi = 6.2831853071795864769F;
                        const auto gaussian =
                            sycl::sqrt(-2.0F * sycl::log(uniform1)) *
                            sycl::cos(two_pi * uniform2);
                        constexpr float nucleon_mass_MeV = 931.49410242F;
                        const auto energy_MeV_per_u =
                            energy_MeV * inverse_mass_number;
                        const auto gamma =
                            1.0F + energy_MeV_per_u / nucleon_mass_MeV;
                        const auto beta_squared = sycl::fmax(
                            0.0F, 1.0F - 1.0F / (gamma * gamma));
                        const auto beta = sycl::sqrt(beta_squared);
                        const auto effective_charge =
                            primary_charge *
                            (1.0F -
                             sycl::exp(
                                 -125.0F * beta * primary_charge_power));
                        constexpr float bethe_K_MeV_cm2_per_g = 0.307075F;
                        constexpr float electron_mass_MeV = 0.51099895F;
                        constexpr float water_Z_over_A = 0.55509F;
                        // Natural Copper: Z/A = 29/63.546.  Express it
                        // relative to water to share the same Bohr convention
                        // as the downstream transport.
                        constexpr float copper_Z_over_A_rel_water =
                            (29.0F / 63.546F) / water_Z_over_A;
                        const auto variance_MeV2 =
                            bethe_K_MeV_cm2_per_g * electron_mass_MeV *
                            effective_charge * effective_charge *
                            water_Z_over_A * copper_Z_over_A_rel_water *
                            minibeam_copper_density_g_per_cm3 *
                            (path_step_mm / 10.0F);
                        const auto sigma_MeV =
                            minibeam_copper_straggling_scale *
                            sycl::sqrt(
                                sycl::fmax(0.0F, variance_MeV2));
                        sampled_loss_MeV = sample_condensed_energy_loss(
                            mean_loss_MeV, sigma_MeV, gaussian, extra_uniform,
                            energy_MeV - energy_cutoff_MeV, straggling_sampler);
                        if (sampled_loss_MeV >=
                            energy_MeV - energy_cutoff_MeV) {
                            record_beamline_removed(0.0F);
                            return;
                        }
                    }
                    const auto scattering_energy_MeV =
                        energy_MeV - 0.5F * sampled_loss_MeV;
                    energy_MeV -= sampled_loss_MeV;

                    if (enable_minibeam_copper_nuclear_attenuation) {
                        const auto energy_MeV_per_u =
                            scattering_energy_MeV * inverse_mass_number;
                        const auto floating_index =
                            (energy_MeV_per_u -
                             minimum_cross_section_energy) *
                            inverse_cross_section_step;
                        auto index =
                            static_cast<int>(sycl::floor(floating_index));
                        index = sycl::max(
                            0, sycl::min(
                                   index,
                                   static_cast<int>(
                                       cross_section_table_size) -
                                       2));
                        const auto fraction = sycl::clamp(
                            floating_index - static_cast<float>(index),
                            0.0F, 1.0F);
                        const auto macroscopic_xs =
                            minibeam_copper_xs_device[index] +
                            fraction *
                                (minibeam_copper_xs_device[index + 1] -
                                 minibeam_copper_xs_device[index]);
                        const auto interaction_probability =
                            1.0F -
                            sycl::exp(-macroscopic_xs * path_step_mm);
                        if (rng::uniform01(
                                spot_seed, rng_history, beamline_step,
                                60) < interaction_probability) {
                            add_minibeam_count(4);
                            if (!enable_minibeam_copper_reaction_products) {
                                record_beamline_removed(0.0F);
                                return;
                            }

                            auto reaction_bin_index =
                                static_cast<int>(sycl::floor(
                                    (energy_MeV_per_u -
                                     minibeam_copper_reaction_minimum_energy) *
                                    minibeam_copper_inverse_reaction_bin_width));
                            reaction_bin_index = sycl::max(
                                0, sycl::min(
                                       reaction_bin_index,
                                       static_cast<int>(
                                           minibeam_copper_reaction_bin_count) -
                                           1));
                            const auto reaction_bin =
                                minibeam_copper_reaction_bins_device[
                                    reaction_bin_index];
                            const auto package_uniform = rng::uniform01(
                                spot_seed, rng_history, beamline_step, 61);
                            const auto package_in_bin = sycl::min(
                                static_cast<std::uint32_t>(
                                    package_uniform *
                                    reaction_bin.reaction_count),
                                reaction_bin.reaction_count - 1U);
                            const auto reaction =
                                minibeam_copper_reactions_device[
                                    reaction_bin.reaction_offset +
                                    package_in_bin];
                            const auto reaction_energy_scale =
                                reaction.incident_energy_MeV_per_u > 0.0F
                                    ? energy_MeV_per_u /
                                          reaction.incident_energy_MeV_per_u
                                    : 1.0F;
                            add_minibeam_count(
                                5, reaction.secondary_count);

                            const auto parent_direction =
                                Direction3F{direction_x, direction_y,
                                            direction_z};
                            const auto charged_survivor =
                                [&](const ReactionSecondary secondary,
                                    const std::uint32_t secondary_index) {
                                    return transport_minibeam_charged_product(
                                        secondary, reaction_energy_scale,
                                        parent_direction, position_x_mm,
                                        position_y_mm, position_z_mm,
                                        collimator_exit_z,
                                        minibeam_cosine_angle,
                                        minibeam_sine_angle,
                                        minibeam_slit_offset_x_mm,
                                        minibeam_slit_offset_y_mm,
                                        minibeam_radius_mm,
                                        minibeam_slit_count,
                                        minibeam_slit_width_mm,
                                        minibeam_slit_pitch_mm,
                                        minibeam_slit_half_length_mm,
                                        minibeam_copper_max_step_mm,
                                        minibeam_copper_density_g_per_cm3,
                                        minibeam_copper_radiation_length_g_per_cm2,
                                        minibeam_copper_mcs_scale,
                                        minibeam_copper_sp_device,
                                        minibeam_air_sp_device,
                                        minibeam_copper_ion_sp_ratio_device,
                                        minibeam_copper_ion_sp_present_device,
                                        minibeam_copper_ion_xs_device,
                                        minibeam_copper_ion_xs_present_device,
                                        minibeam_copper_ion_xs_grid_size,
                                        minibeam_copper_ion_xs_minimum_energy,
                                        minibeam_copper_ion_xs_inverse_step,
                                        minimum_table_energy,
                                        inverse_table_step, table_size,
                                        energy_cutoff_MeV, spot_seed,
                                        rng::child_stream(
                                            global_history,
                                            rng::branch_tag(
                                                rng::branch_role_primary_charged,
                                                secondary_index)));
                                };
                            const auto neutral_survivor =
                                [&](const ReactionSecondary secondary,
                                    const std::uint32_t secondary_index) {
                                    return transport_minibeam_neutral_product(
                                        secondary, reaction_energy_scale,
                                        parent_direction, position_x_mm,
                                        position_y_mm, position_z_mm,
                                        collimator_exit_z,
                                        minibeam_cosine_angle,
                                        minibeam_sine_angle,
                                        minibeam_slit_offset_x_mm,
                                        minibeam_slit_offset_y_mm,
                                        minibeam_radius_mm,
                                        minibeam_slit_count,
                                        minibeam_slit_width_mm,
                                        minibeam_slit_pitch_mm,
                                        minibeam_slit_half_length_mm,
                                        minibeam_copper_max_step_mm,
                                        minibeam_copper_neutral_xs_device,
                                        minibeam_copper_neutral_xs_grid_size,
                                        minibeam_copper_neutral_xs_minimum_log_energy,
                                        minibeam_copper_neutral_xs_inverse_log_step,
                                        minibeam_copper_neutral_projectiles_device,
                                        minibeam_copper_neutral_projectile_count,
                                        minibeam_copper_neutral_interactions_device,
                                        energy_cutoff_MeV, spot_seed,
                                        rng::child_stream(
                                            global_history,
                                            rng::branch_tag(
                                                rng::branch_role_primary_neutral,
                                                secondary_index)));
                                };

                            std::uint32_t charged_count = 0;
                            std::uint32_t neutral_count = 0;
                            auto charged_energy_MeV = 0.0F;
                            auto neutral_energy_MeV = 0.0F;
                            for (std::uint32_t secondary_index = 0;
                                 secondary_index <
                                 reaction.secondary_count;
                                 ++secondary_index) {
                                const auto secondary =
                                    minibeam_copper_secondaries_device[
                                        reaction.secondary_offset +
                                        secondary_index];
                                const auto is_neutral =
                                    secondary.pdg_id == 22 ||
                                    secondary.pdg_id == 2112;
                                if (is_neutral) {
                                    if (!enable_neutral_transport) {
                                        continue;
                                    }
                                    const auto survivor =
                                        neutral_survivor(
                                            secondary, secondary_index);
                                    if (survivor.alive) {
                                        ++neutral_count;
                                        neutral_energy_MeV +=
                                            survivor.kinetic_energy_MeV;
                                    }
                                } else if (
                                    secondary.atomic_number > 0 &&
                                    secondary.mass_number > 0) {
                                    const auto survivor =
                                        charged_survivor(
                                            secondary,
                                            secondary_index);
                                    if (survivor.alive) {
                                        ++charged_count;
                                        charged_energy_MeV +=
                                            survivor.kinetic_energy_MeV;
                                    }
                                }
                            }

                            auto queued_charged_count =
                                std::uint32_t{0};
                            auto queued_neutral_count =
                                std::uint32_t{0};
                            auto queued_charged_energy_MeV = 0.0F;
                            auto queued_neutral_energy_MeV = 0.0F;
                            if (charged_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    counter(
                                        *secondary_queue_counter_device);
                                const auto queue_offset =
                                    counter.fetch_add(charged_count);
                                const auto package_fits =
                                    queue_offset <=
                                        secondary_queue_capacity_u32 &&
                                    charged_count <=
                                        secondary_queue_capacity_u32 -
                                            queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index <
                                         reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary =
                                            minibeam_copper_secondaries_device[
                                                reaction.secondary_offset +
                                                secondary_index];
                                        if (secondary.atomic_number <= 0 ||
                                            secondary.mass_number <= 0) {
                                            continue;
                                        }
                                        const auto survivor =
                                            charged_survivor(
                                                secondary,
                                                secondary_index);
                                        if (!survivor.alive) {
                                            continue;
                                        }
                                        const auto is_primary_continuation =
                                            secondary.pdg_id ==
                                                -1'000'060'120 &&
                                            secondary.atomic_number == 6 &&
                                            secondary.mass_number == 12;
                                        const auto origin_category =
                                            is_primary_continuation
                                                ? primary_charged_origin_category
                                                : charged_dose_category(
                                                      secondary.atomic_number,
                                                      secondary.mass_number);
                                        const auto output_pdg_id =
                                            secondary.pdg_id < 0
                                                ? -secondary.pdg_id
                                                : secondary.pdg_id;
                                        secondary_queue_device[
                                            output_index++] =
                                            SecondaryParticle3D{
                                                survivor.position_x_mm,
                                                survivor.position_y_mm,
                                                0.0F,
                                                survivor
                                                    .kinetic_energy_MeV,
                                                survivor.direction.x,
                                                survivor.direction.y,
                                                survivor.direction.z,
                                                output_pdg_id,
                                                secondary.atomic_number,
                                                secondary.mass_number,
                                                static_cast<std::uint8_t>(
                                                    origin_category),
                                                0,
                                                charged_lineage,
                                                rng::child_stream(
                                                    global_history,
                                                    rng::branch_tag(
                                                        rng::branch_role_primary_charged,
                                                        secondary_index)),
                                            };
                                        const auto diagnostic_category =
                                            minibeam_charged_species_category(
                                                secondary.atomic_number,
                                                secondary.mass_number);
                                        add_minibeam_count(
                                            8 + diagnostic_category);
                                        add_minibeam_moment(
                                            13 + diagnostic_category,
                                            survivor.kinetic_energy_MeV);
                                        auto fragment_histogram = -1;
                                        if (secondary.atomic_number == 1 &&
                                            secondary.mass_number == 2) {
                                            fragment_histogram = 0;
                                        } else if (
                                            secondary.atomic_number == 1 &&
                                            secondary.mass_number == 3) {
                                            fragment_histogram = 1;
                                        } else if (
                                            secondary.atomic_number == 2) {
                                            fragment_histogram = 2;
                                        }
                                        if (fragment_histogram >= 0) {
                                            const auto energy_bin = sycl::min(
                                                static_cast<std::size_t>(
                                                    survivor.kinetic_energy_MeV /
                                                    50.0F),
                                                MinibeamDiagnostics::
                                                        fragment_energy_bin_count -
                                                    1);
                                            add_minibeam_count(
                                                8 +
                                                minibeam_charged_species_count +
                                                3 *
                                                    MinibeamDiagnostics::
                                                        slit_count +
                                                MinibeamDiagnostics::
                                                    touched_energy_bin_count +
                                                static_cast<std::size_t>(
                                                    fragment_histogram) *
                                                    MinibeamDiagnostics::
                                                        fragment_energy_bin_count +
                                                energy_bin);
                                        }
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled(
                                            *secondary_queue_filled_device);
                                    filled.fetch_add(charged_count);
                                    queued_charged_count =
                                        charged_count;
                                    queued_charged_energy_MeV =
                                        charged_energy_MeV;
                                }
                            }
                            if (neutral_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    counter(
                                        *neutral_queue_counter_device);
                                const auto queue_offset =
                                    counter.fetch_add(neutral_count);
                                const auto package_fits =
                                    queue_offset <=
                                        neutral_queue_capacity_u32 &&
                                    neutral_count <=
                                        neutral_queue_capacity_u32 -
                                            queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index <
                                         reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary =
                                            minibeam_copper_secondaries_device[
                                                reaction.secondary_offset +
                                                secondary_index];
                                        if (secondary.pdg_id != 22 &&
                                            secondary.pdg_id != 2112) {
                                            continue;
                                        }
                                        const auto survivor =
                                            neutral_survivor(
                                                secondary,
                                                secondary_index);
                                        if (!survivor.alive) {
                                            continue;
                                        }
                                        const auto lineage =
                                            neutral_lineage_from_pdg(
                                                secondary.pdg_id);
                                        neutral_queue_device[
                                            output_index++] =
                                            NeutralParticle3D{
                                                survivor.position_x_mm,
                                                survivor.position_y_mm,
                                                0.0F,
                                                survivor
                                                    .kinetic_energy_MeV,
                                                survivor.direction.x,
                                                survivor.direction.y,
                                                survivor.direction.z,
                                                secondary.pdg_id,
                                                static_cast<std::uint8_t>(
                                                    neutral_origin_category_from_lineage(
                                                        lineage)),
                                                0, 0,
                                                rng::child_stream(
                                                    global_history,
                                                    rng::branch_tag(
                                                        rng::branch_role_primary_neutral,
                                                        secondary_index)),
                                            };
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled(
                                            *neutral_queue_filled_device);
                                    filled.fetch_add(neutral_count);
                                    queued_neutral_count =
                                        neutral_count;
                                    queued_neutral_energy_MeV =
                                        neutral_energy_MeV;
                                }
                            }
                            add_minibeam_count(
                                6, queued_charged_count);
                            add_minibeam_count(
                                7, queued_neutral_count);
                            add_minibeam_moment(
                                11, queued_charged_energy_MeV);
                            add_minibeam_moment(
                                12, queued_neutral_energy_MeV);
                            record_beamline_removed(
                                queued_charged_energy_MeV +
                                queued_neutral_energy_MeV);
                            return;
                        }
                    }

                    if (minibeam_copper_enable_mcs) {
                        const auto previous_segment_path =
                            copper_segment_path_mm;
                        copper_segment_path_mm += path_step_mm;
                        const auto total_rms = use_explicit_primary_rest_mass
                            ? highland_projected_rms_angle_with_mass_device(
                                  scattering_energy_MeV,
                                  primary_atomic_number,
                                  primary_rest_mass_MeV,
                                  copper_segment_path_mm,
                                  minibeam_copper_density_g_per_cm3,
                                  minibeam_copper_radiation_length_g_per_cm2)
                            : highland_projected_rms_angle_device(
                                  scattering_energy_MeV,
                                  primary_atomic_number,
                                  primary_mass_number,
                                  copper_segment_path_mm,
                                  minibeam_copper_density_g_per_cm3,
                                  minibeam_copper_radiation_length_g_per_cm2);
                        const auto previous_rms = use_explicit_primary_rest_mass
                            ? highland_projected_rms_angle_with_mass_device(
                                  scattering_energy_MeV,
                                  primary_atomic_number,
                                  primary_rest_mass_MeV,
                                  previous_segment_path,
                                  minibeam_copper_density_g_per_cm3,
                                  minibeam_copper_radiation_length_g_per_cm2)
                            : highland_projected_rms_angle_device(
                                  scattering_energy_MeV,
                                  primary_atomic_number,
                                  primary_mass_number,
                                  previous_segment_path,
                                  minibeam_copper_density_g_per_cm3,
                                  minibeam_copper_radiation_length_g_per_cm2);
                        const auto incremental_variance = sycl::fmax(
                            0.0F, total_rms * total_rms -
                                      previous_rms * previous_rms);
                        const auto scattered = scatter_direction(
                            Direction3F{direction_x, direction_y,
                                        direction_z},
                            minibeam_copper_mcs_scale *
                                sycl::sqrt(incremental_variance),
                            spot_seed,
                            rng_history, beamline_step, 50);
                        direction_x = scattered.x;
                        direction_y = scattered.y;
                        direction_z = scattered.z;
                    }
                    ++beamline_step;
                }
                if (beamline_step == maximum_beamline_steps) {
                    record_beamline_removed(0.0F);
                    return;
                }
                if (direction_z <= 1.0e-8F) {
                    record_beamline_removed(0.0F);
                    return;
                }
                if (copper_touched) {
                    auto survivor_energy_loss_scale =
                        minibeam_copper_survivor_energy_loss_scale;
                    if (minibeam_copper_survivor_calibration_count > 0) {
                        const auto entrance_energy_MeV_per_u =
                            energy_at_collimator_entrance_MeV *
                            inverse_mass_number;
                        survivor_energy_loss_scale =
                            minibeam_copper_survivor_calibration_scales[0];
                        if (entrance_energy_MeV_per_u >=
                            minibeam_copper_survivor_calibration_energies[
                                minibeam_copper_survivor_calibration_count -
                                1]) {
                            survivor_energy_loss_scale =
                                minibeam_copper_survivor_calibration_scales[
                                    minibeam_copper_survivor_calibration_count -
                                    1];
                        } else {
                            for (std::size_t calibration_index = 1;
                                 calibration_index <
                                 minibeam_copper_survivor_calibration_count;
                                 ++calibration_index) {
                                const auto upper_energy =
                                    minibeam_copper_survivor_calibration_energies[
                                        calibration_index];
                                if (entrance_energy_MeV_per_u <=
                                    upper_energy) {
                                    const auto lower_energy =
                                        minibeam_copper_survivor_calibration_energies[
                                            calibration_index - 1];
                                    const auto fraction = sycl::clamp(
                                        (entrance_energy_MeV_per_u -
                                         lower_energy) /
                                            (upper_energy - lower_energy),
                                        0.0F, 1.0F);
                                    const auto lower_scale =
                                        minibeam_copper_survivor_calibration_scales[
                                            calibration_index - 1];
                                    survivor_energy_loss_scale =
                                        lower_scale +
                                        fraction *
                                            (minibeam_copper_survivor_calibration_scales[
                                                 calibration_index] -
                                             lower_scale);
                                    break;
                                }
                            }
                        }
                    }
                    const auto accumulated_copper_loss_MeV =
                        sycl::fmax(
                            0.0F,
                            energy_at_collimator_entrance_MeV -
                                energy_MeV);
                    energy_MeV =
                        energy_at_collimator_entrance_MeV -
                        survivor_energy_loss_scale *
                            accumulated_copper_loss_MeV;
                    if (energy_MeV <= energy_cutoff_MeV) {
                        record_beamline_removed(0.0F);
                        return;
                    }
                }
                const auto exit_air_distance =
                    -position_z_mm / direction_z;
                const auto exit_air_loss =
                    minibeam_stopping_power(
                        minibeam_air_sp_device, energy_MeV) *
                    exit_air_distance;
                if (exit_air_loss >=
                    energy_MeV - energy_cutoff_MeV) {
                    record_beamline_removed(0.0F);
                    return;
                }
                energy_MeV -= exit_air_loss;
            }
            if (enable_tps_source) {
                // Clinical TPS sources live at SAD outside the transport volume.
                // Move through vacuum to the first intersection with the scorer
                // AABB. This supports cardinal gantry directions without changing
                // the legacy CT/TOPAS z=0 projection below.
                auto t_enter = 0.0F;
                auto t_exit = 1.0e30F;
                auto hit = true;
                auto intersect_slab = [&](const float position, const float direction,
                                          const float lower, const float upper) {
                    if (sycl::fabs(direction) < 1.0e-8F) {
                        if (position < lower || position >= upper) {
                            hit = false;
                        }
                        return;
                    }
                    auto first = (lower - position) / direction;
                    auto second = (upper - position) / direction;
                    if (first > second) {
                        const auto temporary = first;
                        first = second;
                        second = temporary;
                    }
                    t_enter = sycl::fmax(t_enter, first);
                    t_exit = sycl::fmin(t_exit, second);
                    if (t_exit < t_enter) {
                        hit = false;
                    }
                };
                if (enable_voxel_scoring) {
                    intersect_slab(position_x_mm, direction_x,
                                   voxel_min_x_mm, voxel_max_x_mm);
                    intersect_slab(position_y_mm, direction_y,
                                   voxel_min_y_mm, voxel_max_y_mm);
                }
                intersect_slab(position_z_mm, direction_z, 0.0F, phantom_length_mm);
                if (hit && t_exit >= t_enter) {
                    const auto entry = t_enter + 1.0e-4F;
                    position_x_mm += entry * direction_x;
                    position_y_mm += entry * direction_y;
                    position_z_mm += entry * direction_z;
                }
            } else {
                // TPS-90 legacy entrance sampling: project every sampled point
                // onto the CT entrance plane.  Even an exactly cardinal pose has
                // O(1e-16) transverse-basis z components from sin/cos.  Leaving
                // those values untouched puts roughly half of a centred Gaussian
                // infinitesimally below z=0, where the escape test rejects it.
                if (sycl::fabs(direction_z) > 1.0e-8F) {
                    const auto t_plane = -position_z_mm / direction_z;
                    position_x_mm += t_plane * direction_x;
                    position_y_mm += t_plane * direction_y;
                    position_z_mm = 0.0F;
                }
            }
            if (enable_minibeam) {
                add_minibeam_count(3);
                add_minibeam_moment(
                    13 + minibeam_charged_species_count +
                        (minibeam_direct ? 0 : 1),
                    static_cast<double>(energy_MeV));
                if (!minibeam_direct) {
                    const auto energy_bin = sycl::min(
                        static_cast<std::size_t>(
                            energy_MeV / 200.0F),
                        MinibeamDiagnostics::
                                touched_energy_bin_count -
                            1);
                    add_minibeam_count(
                        8 + minibeam_charged_species_count +
                        3 * MinibeamDiagnostics::slit_count +
                        energy_bin);
                }
                const auto slit_u_mm =
                    minibeam_cosine_angle * position_x_mm +
                    minibeam_sine_angle * position_y_mm -
                    minibeam_slit_offset_mm;
                const auto nearest_slit = nearest_minibeam_slit(
                    slit_u_mm, minibeam_slit_pitch_mm);
                const auto slit_array_index =
                    nearest_slit + minibeam_slit_count / 2;
                if (slit_array_index >= 0 &&
                    slit_array_index <
                        static_cast<int>(MinibeamDiagnostics::slit_count)) {
                    add_minibeam_count(
                        8 + minibeam_charged_species_count +
                        static_cast<std::size_t>(slit_array_index));
                }
                record_beamline_removed(energy_MeV);
                const auto energy = static_cast<double>(energy_MeV);
                const auto x = static_cast<double>(position_x_mm);
                const auto y = static_cast<double>(position_y_mm);
                const auto dx = static_cast<double>(direction_x);
                const auto dy = static_cast<double>(direction_y);
                add_minibeam_moment(0, energy);
                add_minibeam_moment(1, energy * energy);
                add_minibeam_moment(2, x);
                add_minibeam_moment(3, x * x);
                add_minibeam_moment(4, y);
                add_minibeam_moment(5, y * y);
                add_minibeam_moment(6, dx);
                add_minibeam_moment(7, dx * dx);
                add_minibeam_moment(8, dy);
                add_minibeam_moment(9, dy * dy);
            }
            auto history_deposited_MeV = 0.0f;
            auto history_nuclear_MeV = 0.0f;
            SecondaryGenerationSummary secondary_summary{};
            std::uint32_t steps = 0;
            StepStableStragglingState<float> stable_straggling;
            stable_straggling.initialize(straggling_sampling_length_mm);
            double pending_primary_depth_MeV = 0.0;
            int pending_primary_bin = 0;
            double pending_primary_voxel_MeV = 0.0;
            double pending_primary_voxel_delayed_MeV = 0.0;
            float pending_primary_electronic_mfp_mm = electronic_buildup_mfp_mm;
            std::size_t pending_primary_voxel = 0;
            auto last_primary_stopping_power_MeV_per_mm = 0.0F;
            auto last_primary_density_g_per_cm3 = 0.0F;
            auto last_primary_let_delta_fraction = 0.0F;
            const auto attenuation_uniform = sycl::fmax(
                rng::uniform01(spot_seed, rng_history, 0, 9), 1.0e-12F);
            auto remaining_interaction_lengths = -sycl::log(attenuation_uniform);
            constexpr std::uint32_t max_primary_steps = 2'000'000U;
            while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                const auto escaped_z =
                    position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                const auto escaped_xy =
                    enable_voxel_scoring &&
                    (position_x_mm < voxel_min_x_mm || position_x_mm >= voxel_max_x_mm ||
                     position_y_mm < voxel_min_y_mm || position_y_mm >= voxel_max_y_mm);
                if (escaped_z || escaped_xy) {
                    break;
                }

                const auto absolute_direction_x = sycl::fabs(direction_x);
                const auto absolute_direction_y = sycl::fabs(direction_y);
                const auto absolute_direction_z = sycl::fabs(direction_z);
                auto bin = direction_z < 0.0F
                               ? static_cast<int>(
                                     sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                               : static_cast<int>(
                                     sycl::floor(position_z_mm / depth_bin_width_mm));
                bin = sycl::max(
                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                if (enable_voxel_scoring) {
                    const auto x_coordinate =
                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                    const auto y_coordinate =
                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                    voxel_x = direction_x < 0.0F
                                  ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                  : static_cast<int>(sycl::floor(x_coordinate));
                    voxel_y = direction_y < 0.0F
                                  ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                  : static_cast<int>(sycl::floor(y_coordinate));
                    voxel_x = sycl::max(
                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                    voxel_y = sycl::max(
                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                }
                const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                         static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                         static_cast<std::size_t>(voxel_x);
                const auto energy_MeVu = energy_MeV * inverse_mass_number;
                auto floating_index = (energy_MeVu - minimum_table_energy) * inverse_table_step;
                auto index = static_cast<int>(sycl::floor(floating_index));
                index = sycl::max(0, sycl::min(index, static_cast<int>(table_size) - 2));
                const auto fraction = sycl::clamp(floating_index - static_cast<float>(index),
                                                  0.0f, 1.0f);
                const auto in_insert =
                    enable_hetero_insert &&
                    inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                         insert_x_min, insert_x_max, insert_y_min,
                                         insert_y_max, insert_z_min, insert_z_max);
                auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                    position_z_mm, slab_z_ends_device, slab_densities_device,
                    slab_layer_count, water_density_g_per_cm3);
                if (in_insert) {
                    local_density_g_per_cm3 = insert_density_g_per_cm3;
                }
                std::uint8_t ct_material = 2;
                auto in_ct = false;
                if (enable_ct_grid) {
                    float ct_rho = water_density_g_per_cm3;
                    in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                      ct_origin_x, ct_origin_y, ct_origin_z, ct_spacing_x,
                                      ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz,
                                      ct_density_device, ct_material_device, ct_rho,
                                      ct_material);
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_ct_samples);
                    if (in_ct) {
                        local_density_g_per_cm3 = ct_rho;
                    }
                }
                const auto layer_for_material =
                    slab_layer_count > 0
                        ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                           slab_layer_count)
                        : 0U;
                float stopping_power_MeV_per_mm = 0.0F;
                if (enable_ct_grid) {
                    const auto water_sp =
                        table_device[index] +
                        fraction * (table_device[index + 1] - table_device[index]);
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_sp_table_lookups);
                    if (use_ct_mass_sp && in_ct && ct_mass_sp_factor_lut_device != nullptr &&
                        ct_n_mass_factors > 0) {
                        // Precomputed Schneider mass-SP factor LUT × density (P2).
                        const auto mass_factor = ct_lookup_mass_sp_factor(
                            ct_mass_sp_factor_lut_device,
                            use_ct_density_mass_spr ? ct_density_spr_n_rho
                                                    : ct_n_mass_factors,
                            table_size, use_ct_density_mass_spr,
                            ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                            static_cast<std::uint32_t>(ct_material),
                            local_density_g_per_cm3,
                            static_cast<std::size_t>(index), fraction,
                            [](float x) { return sycl::log(x); });
                        stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                            water_sp, local_density_g_per_cm3, mass_factor);
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::primary_mass_sp_lookups);
                    } else if (use_ct_material_sp && in_ct && ct_sp_device != nullptr &&
                               ct_ref_density_device != nullptr) {
                        // Legacy absolute material SP × (local ρ / ρ_ref).
                        const auto mat = static_cast<std::uint32_t>(ct_material_class(
                            ct_material, ct_material_ids_are_schneider_sections));
                        const auto base = mat * table_size;
                        const auto sp_abs =
                            ct_sp_device[base + static_cast<std::size_t>(index)] +
                            fraction *
                                (ct_sp_device[base + static_cast<std::size_t>(index) + 1] -
                                 ct_sp_device[base + static_cast<std::size_t>(index)]);
                        const auto ref_rho =
                            sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                        stopping_power_MeV_per_mm =
                            sp_abs *
                            (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) / ref_rho);
                    } else {
                        // Water-equivalent: water SP × local density.
                        stopping_power_MeV_per_mm =
                            water_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                    }
                } else if (in_insert && use_insert_material_tables) {
                    stopping_power_MeV_per_mm =
                        insert_sp_device[static_cast<std::size_t>(index)] +
                        fraction *
                            (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                             insert_sp_device[static_cast<std::size_t>(index)]);
                } else if (material_table_count > 0) {
                    const auto base = static_cast<std::size_t>(layer_for_material) * table_size;
                    stopping_power_MeV_per_mm =
                        material_sp_device[base + static_cast<std::size_t>(index)] +
                        fraction *
                            (material_sp_device[base + static_cast<std::size_t>(index) + 1] -
                             material_sp_device[base + static_cast<std::size_t>(index)]);
                } else {
                    const auto table_stopping_power_MeV_per_mm =
                        table_device[index] +
                        fraction * (table_device[index + 1] - table_device[index]);
                    // Density-only hetero: scale water SP by local ρ (insert or slab).
                    const auto scale_density =
                        (slab_layer_count > 0 || in_insert) ? local_density_g_per_cm3
                                                            : 1.0F;
                    stopping_power_MeV_per_mm =
                        (slab_layer_count > 0 || in_insert)
                            ? table_stopping_power_MeV_per_mm * scale_density
                            : table_stopping_power_MeV_per_mm;
                }
                if (enable_minibeam && !enable_ct_grid &&
                    !enable_layered_phantom && !enable_hetero_insert &&
                    minibeam_water_primary_stopping_power_scale != 1.0F) {
                    stopping_power_MeV_per_mm *=
                        minibeam_water_primary_stopping_power_scale;
                }
                // The Copper-touched C-12 response can carry an optional smooth
                // entrance correction. It modifies stopping, not scored dose:
                // retained kinetic energy continues downstream and history
                // energy remains conserved. Both amplitudes default to zero.
                if (enable_minibeam && !minibeam_direct &&
                    (minibeam_water_touched_primary_surface_boost > 0.0F ||
                     minibeam_water_touched_primary_deficit > 0.0F)) {
                    const auto surface =
                        minibeam_water_touched_primary_surface_boost *
                        sycl::exp(
                                    -0.5F *
                                    (position_z_mm /
                                     minibeam_water_touched_primary_surface_sigma_mm) *
                                    (position_z_mm /
                                     minibeam_water_touched_primary_surface_sigma_mm));
                    const auto centred =
                        (position_z_mm -
                         minibeam_water_touched_primary_deficit_center_mm) /
                        minibeam_water_touched_primary_deficit_sigma_mm;
                    const auto deficit =
                        minibeam_water_touched_primary_deficit *
                        sycl::exp(
                                    -0.5F * centred * centred);
                    stopping_power_MeV_per_mm *=
                        sycl::clamp(
                            1.0F + surface - deficit,
                            0.5F, 1.5F);
                }

                auto step_mm = sycl::fmin(
                    maximum_step_mm,
                    maximum_relative_energy_loss * energy_MeV /
                        sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                if (absolute_direction_z >= 1.0e-6F) {
                    const auto boundary_z_mm =
                        direction_z < 0.0F
                            ? static_cast<float>(bin) * depth_bin_width_mm
                            : static_cast<float>(bin + 1) * depth_bin_width_mm;
                    const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                    if (dz_step > 0.0F) {
                        step_mm = sycl::fmin(step_mm, dz_step);
                    }
                }
                auto stable_straggling_interface_limited = false;
                if (slab_layer_count > 0) {
                    const auto slab_step = distance_to_slab_interface_mm(
                        position_z_mm, direction_z, slab_z_ends_device, slab_layer_count,
                        phantom_length_mm);
                    stable_straggling_interface_limited = slab_step < step_mm;
                    step_mm = sycl::fmin(step_mm, slab_step);
                }
                if (enable_hetero_insert) {
                    const auto insert_step = distance_to_insert_interface_mm(
                        position_x_mm, position_y_mm, position_z_mm, direction_x,
                        direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                        insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                    stable_straggling_interface_limited =
                        stable_straggling_interface_limited || insert_step < step_mm;
                    step_mm = sycl::fmin(step_mm, insert_step);
                }
                if (enable_ct_grid && in_ct) {
                    const auto step_before_ct_clamp = step_mm;
                    // Face clamp only when density/material changes along the step
                    // (unless ct_skip_homogeneous_face_clamp is false).
                    CtClampPath clamp_path = CtClampPath::three_axis;
                    step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                        step_mm, position_x_mm, position_y_mm, position_z_mm,
                        direction_x, direction_y, direction_z, ct_origin_x,
                        ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                        ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                        ct_material_device, local_density_g_per_cm3, ct_material,
                        ct_skip_homogeneous_face_clamp,
#ifdef CARBON_TRANSPORT_PROFILE
                        &clamp_path
#else
                        nullptr
#endif
                    );
                    profile_face(profile_counters_device, true, clamp_path);
                    stable_straggling_interface_limited =
                        stable_straggling_interface_limited || step_mm < step_before_ct_clamp;
                }
                if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                    absolute_direction_x >= 1.0e-6F) {
                    const auto boundary_x_mm =
                        voxel_min_x_mm +
                        static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                            voxel_size_x_mm;
                    step_mm = sycl::fmin(
                        step_mm, (boundary_x_mm - position_x_mm) / direction_x);
                }
                if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                    absolute_direction_y >= 1.0e-6F) {
                    const auto boundary_y_mm =
                        voxel_min_y_mm +
                        static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                            voxel_size_y_mm;
                    step_mm = sycl::fmin(
                        step_mm, (boundary_y_mm - position_y_mm) / direction_y);
                }
                // Accept CT micro-steps (DDA already avoids zero-length face clamps).
                // Only snap/nudge when the step is non-positive or non-CT thrash.
                const auto min_step_accept =
                    (enable_ct_grid && in_ct) ? 1.0e-8F : 1.0e-6F;
                if (step_mm <= min_step_accept) {
                    auto snapped_to_boundary = false;
                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if ((boundary_z_mm - position_z_mm) / direction_z <= 1.0e-6F) {
                            position_z_mm = nudge_past_axis_boundary(
                                boundary_z_mm, direction_z,
                                robust_boundary_nudge);
                            snapped_to_boundary = true;
                        }
                    }
                    if (slab_layer_count > 0 && absolute_direction_z >= 1.0e-6F) {
                        const auto layer = slab_layer_index(
                            position_z_mm, slab_z_ends_device, slab_layer_count);
                        const auto interface_z =
                            direction_z > 0.0F
                                ? slab_z_ends_device[layer]
                                : (layer == 0U ? 0.0F : slab_z_ends_device[layer - 1U]);
                        if (sycl::fabs((interface_z - position_z_mm) / direction_z) <=
                            1.0e-6F) {
                            position_z_mm = nudge_past_axis_boundary(
                                interface_z, direction_z,
                                robust_boundary_nudge);
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max,
                            insert_y_min, insert_y_max, insert_z_min, insert_z_max,
                            phantom_length_mm);
                        if (insert_step <= 1.0e-6F) {
                            // Nudge along the ray to leave/enter the insert AABB.
                            constexpr float nudge = 1.0e-4F;
                            position_x_mm += direction_x * nudge;
                            position_y_mm += direction_y * nudge;
                            position_z_mm += direction_z * nudge;
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_ct_grid && in_ct) {
                        // Particle sitting on a CT face with vanishing step: nudge past.
                        constexpr float nudge = 1.0e-4F;
                        position_x_mm += direction_x * nudge;
                        position_y_mm += direction_y * nudge;
                        position_z_mm += direction_z * nudge;
                        snapped_to_boundary = true;
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(
                                voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        if ((boundary_x_mm - position_x_mm) / direction_x <= 1.0e-6F) {
                            position_x_mm = nudge_past_axis_boundary(
                                boundary_x_mm, direction_x,
                                robust_boundary_nudge);
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(
                                voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        if ((boundary_y_mm - position_y_mm) / direction_y <= 1.0e-6F) {
                            position_y_mm = nudge_past_axis_boundary(
                                boundary_y_mm, direction_y,
                                robust_boundary_nudge);
                            snapped_to_boundary = true;
                        }
                    }
                    if (snapped_to_boundary) {
                        ++steps;
                        profile_add(
                            profile_counters_device,
                            TransportProfileSlot::primary_boundary_nudge_continues);
                        continue;
                    }
                    break;
                }

                if (enable_energy_straggling && enable_step_stable_straggling) {
                    stable_straggling.prepare_step(
                        step_mm, (phantom_length_mm - position_z_mm) /
                                      sycl::fmax(direction_z, 1.0e-6F));
                }
                const auto mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);
                if (enable_energy_straggling) {
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_straggling);
                    const auto random_index = enable_step_stable_straggling
                        ? stable_straggling.block_index
                        : static_cast<std::uint64_t>(steps);
                    const auto uniform1 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, random_index, 0), 1.0e-12f);
                    const auto uniform2 = rng::uniform01(spot_seed, rng_history, random_index, 1);
                    constexpr float two_pi = 6.2831853071795864769f;
                    const auto extra_uniform =
                        rng::uniform01(spot_seed, rng_history, random_index, 2);
                    const auto gaussian = sycl::sqrt(-2.0f * sycl::log(uniform1)) *
                                          sycl::cos(two_pi * uniform2);

                    constexpr float nucleon_mass_MeV = 931.49410242f;
                    const auto gamma = 1.0f + energy_MeVu / nucleon_mass_MeV;
                    const auto beta_squared =
                        sycl::fmax(0.0f, 1.0f - 1.0f / (gamma * gamma));
                    const auto beta = sycl::sqrt(beta_squared);
                    const auto effective_charge =
                        primary_charge *
                        (1.0f - sycl::exp(-125.0f * beta * primary_charge_power));
                    // Condensed total-loss variance: Schneider Z/A when present.
                    auto za_rel = 1.0F;
                    if (enable_ct_grid && in_ct) {
                        if (use_ct_mass_sp && ct_mass_sp_za_rel_device != nullptr &&
                            ct_n_mass_factors > 0) {
                            const auto sec = static_cast<std::uint32_t>(ct_material);
                            const auto fi = sec < ct_n_mass_factors
                                                ? sec
                                                : (ct_n_mass_factors - 1U);
                            za_rel = sycl::clamp(ct_mass_sp_za_rel_device[fi], 0.5F, 1.5F);
                        } else {
                            const auto mat = ct_material_class(
                                ct_material, ct_material_ids_are_schneider_sections);
                            // Class defaults: air/lung/water ~1, bone ~0.93.
                            za_rel = mat == 3U ? 0.93F : 1.0F;
                        }
                    }
                    const auto variance_MeV2 = use_explicit_primary_rest_mass
                        ? condensed_total_loss_variance_with_mass_MeV2_device(
                              energy_MeV, primary_rest_mass_MeV,
                              effective_charge, step_mm,
                              local_density_g_per_cm3, za_rel)
                        : condensed_total_loss_variance_MeV2_device(
                              energy_MeVu, primary_mass_number,
                              effective_charge, step_mm,
                              local_density_g_per_cm3, za_rel);
                    const auto local_straggling_scale =
                        interpolate_straggling_scale(
                            energy_MeVu, straggling_scale_energies,
                            straggling_scale_values,
                            straggling_scale_point_count, straggling_scale);
                    const auto sigma_MeV =
                        local_straggling_scale * sycl::sqrt(sycl::fmax(0.0f, variance_MeV2));
                    if (enable_step_stable_straggling) {
                        if (!stable_straggling.block_active) {
                            stable_straggling.begin_block(
                                mean_loss_MeV, variance_MeV2, step_mm,
                                local_straggling_scale, gaussian, energy_MeV,
                                extra_uniform, straggling_sampler);
                        }
                        deposited_MeV = stable_straggling.consume_loss(step_mm, energy_MeV);
                        if (stable_straggling_interface_limited) {
                            stable_straggling.reset_block();
                        }
                    } else {
                        deposited_MeV = sample_condensed_energy_loss(
                            mean_loss_MeV, sigma_MeV, gaussian, extra_uniform,
                            energy_MeV, straggling_sampler);
                    }
                }
                // Electronic build-up: local (1-f)*dE + short-range delta f*dE along +z.
                // Equilibrium dose still ≈ full unrestricted SP; surface suppressed.
                const auto e_frac = electronic_buildup_fraction_at_energy(
                    energy_MeVu, electronic_buildup_fraction);
                const auto delayed_MeV = deposited_MeV * e_frac;
                const auto local_MeV = deposited_MeV - delayed_MeV;
                const auto electronic_mfp_mm = electronic_buildup_mfp_at_energy(
                    energy_MeVu, electronic_buildup_mfp_mm);
                pending_primary_electronic_mfp_mm = electronic_mfp_mm;
                if (enable_let_scoring) {
                    const auto let_delta_fraction =
                        use_let_delta_fraction_table
                            ? let_delta_fraction_device[index] +
                                  fraction *
                                      (let_delta_fraction_device[index + 1] -
                                       let_delta_fraction_device[index])
                            : e_frac;
                    last_primary_stopping_power_MeV_per_mm =
                        stopping_power_MeV_per_mm;
                    last_primary_density_g_per_cm3 =
                        sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                    last_primary_let_delta_fraction = let_delta_fraction;
                    score_letd_moments_device(
                        let_moments_device, number_of_bins,
                        static_cast<std::size_t>(bin),
                        species_let_moments_device,
                        charged_origin_category_count, 0,
                        nullptr, 0, 0,
                        deposited_MeV * (1.0F - let_delta_fraction),
                        deposited_MeV,
                        stopping_power_MeV_per_mm,
                        sycl::fmax(local_density_g_per_cm3, 1.0e-6F), true);
                    score_letd_moments_device(
                        voxel_let_moments_device, number_of_voxels, voxel_index,
                        nullptr, 0, 0,
                        nullptr, 0, 0,
                        deposited_MeV * (1.0F - let_delta_fraction),
                        deposited_MeV, stopping_power_MeV_per_mm,
                        sycl::fmax(local_density_g_per_cm3, 1.0e-6F), true);
                }
                if (pending_primary_depth_MeV > 0.0 && pending_primary_bin != bin) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_dose_depth_atomics);
                    pending_primary_depth_MeV = 0.0;
                }
                if (pending_primary_depth_MeV == 0.0) {
                    pending_primary_bin = bin;
                }
                pending_primary_depth_MeV += static_cast<double>(local_MeV);
                if (delayed_MeV > 0.0F) {
                    const auto z_mid = position_z_mm + 0.5F * direction_z * step_mm;
                    score_exponential_depth(
                        delayed_MeV, z_mid, depth_bin_width_mm, phantom_length_mm,
                        static_cast<std::uint32_t>(number_of_bins),
                        electronic_mfp_mm, dose_device);
                }
                if (enable_voxel_scoring) {
                    // Aggregate consecutive deposits in one voxel before the
                    // expensive global FP64 atomic update.
                    if (pending_primary_voxel_MeV > 0.0 &&
                        pending_primary_voxel != voxel_index) {
                        std::size_t target_voxel = pending_primary_voxel;
                        if (pending_primary_voxel_delayed_MeV > 0.0) {
                            target_voxel = electronic_voxel_target(
                                target_voxel,
                                static_cast<float>(pending_primary_voxel_MeV),
                                static_cast<float>(pending_primary_voxel_delayed_MeV),
                                electronic_mfp_mm,
                                electronic_buildup_lateral_sigma_mm,
                                depth_bin_width_mm, number_of_bins,
                                voxel_size_x_mm, voxel_size_y_mm,
                                voxel_bins_x, voxel_bins_y,
                                rng::uniform01(spot_seed, rng_history, steps, 50),
                                rng::uniform01(spot_seed, rng_history, steps, 53),
                                rng::uniform01(spot_seed, rng_history, steps, 51),
                                rng::uniform01(spot_seed, rng_history, steps, 52));
                        }
                        if (target_voxel != static_cast<std::size_t>(-1)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_voxel_dose(voxel_dose_device[target_voxel]);
                            atomic_voxel_dose.fetch_add(
                                static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::primary_dose_voxel_atomics);
                            if (enable_charged_origin_voxel_scoring) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_category_dose(
                                        charged_origin_voxel_dose_device[target_voxel]);
                                atomic_category_dose.fetch_add(
                                    static_cast<DoseAtomicT>(
                                        pending_primary_voxel_MeV));
                            }
                        }
                        pending_primary_voxel_MeV = 0.0;
                        pending_primary_voxel_delayed_MeV = 0.0;
                    }
                    if (pending_primary_voxel_MeV == 0.0) {
                        pending_primary_voxel = voxel_index;
                    }
                    pending_primary_voxel_MeV += static_cast<double>(deposited_MeV);
                    pending_primary_voxel_delayed_MeV +=
                        static_cast<double>(delayed_MeV);
                }
                history_deposited_MeV += deposited_MeV;
                const auto scattering_energy_MeV =
                    energy_MeV - 0.5F * deposited_MeV;
                energy_MeV -= deposited_MeV;
                position_x_mm += direction_x * step_mm;
                position_y_mm += direction_y * step_mm;
                position_z_mm += direction_z * step_mm;
                if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
                    profile_add(profile_counters_device, TransportProfileSlot::primary_mcs);
                    auto mcs_radiation_length =
                        static_cast<float>(water_radiation_length_g_per_cm2);
                    if (enable_ct_material_mcs && in_ct) {
                        mcs_radiation_length = static_cast<float>(
                            ct_material_radiation_length_g_per_cm2(
                                ct_material_class(
                                    ct_material,
                                    ct_material_ids_are_schneider_sections)));
                    } else if (in_insert) {
                        mcs_radiation_length =
                            insert_radiation_length_g_per_cm2;
                    } else if (slab_radiation_lengths_device != nullptr) {
                        mcs_radiation_length =
                            slab_radiation_lengths_device[layer_for_material];
                    }
                    auto projected_rms_angle_rad = use_explicit_primary_rest_mass
                        ? highland_projected_rms_angle_with_mass_device(
                              scattering_energy_MeV,
                              primary_atomic_number,
                              primary_rest_mass_MeV, step_mm,
                              local_density_g_per_cm3,
                              mcs_radiation_length)
                        : highland_projected_rms_angle_device(
                              scattering_energy_MeV,
                              primary_atomic_number,
                              primary_mass_number, step_mm,
                              local_density_g_per_cm3,
                              mcs_radiation_length);
                    // The low-energy correction compensates the step-wise
                    // Highland approximation and is a projectile correction,
                    // not a water-only material correction.  Keep the local
                    // CT density/radiation length above, then apply the same
                    // validated smooth low-energy scale in explicit CT
                    // materials as in homogeneous water.
                    if (enable_minibeam) {
                        projected_rms_angle_rad *=
                            minibeam_water_low_energy_mcs_scale(
                                scattering_energy_MeV,
                                primary_mass_number,
                                minibeam_water_low_energy_mcs_transition_MeVu,
                                minibeam_water_primary_low_energy_mcs_scale);
                    }
                    const auto scattered = scatter_direction(
                        Direction3F{direction_x, direction_y, direction_z},
                        projected_rms_angle_rad, spot_seed, rng_history, steps, 4);
                    direction_x = scattered.x;
                    direction_y = scattered.y;
                    direction_z = scattered.z;
                }
                if (enable_primary_attenuation && energy_MeV > energy_cutoff_MeV) {
                    const auto post_step_energy_MeVu = energy_MeV * inverse_mass_number;
                    const auto attenuation_energy_MeVu =
                        (energy_MeV + 0.5F * deposited_MeV) * inverse_mass_number;
                    auto cross_section_floating_index =
                        (attenuation_energy_MeVu - minimum_cross_section_energy) *
                        inverse_cross_section_step;
                    auto cross_section_index =
                        static_cast<int>(sycl::floor(cross_section_floating_index));
                    cross_section_index = sycl::max(
                        0, sycl::min(cross_section_index,
                                     static_cast<int>(cross_section_table_size) - 2));
                    const auto cross_section_fraction = sycl::clamp(
                        cross_section_floating_index - static_cast<float>(cross_section_index),
                        0.0f, 1.0f);
                    float macroscopic_cross_section_per_mm = 0.0F;
                    if (enable_ct_grid) {
                        if (use_ct_material_xs && in_ct && ct_xs_device != nullptr &&
                            ct_ref_density_device != nullptr) {
                            const auto mat = static_cast<std::uint32_t>(
                                ct_cross_section_material_index(
                                    ct_material,
                                    ct_material_ids_are_schneider_sections,
                                    use_ct_schneider_xs));
                            const auto base = mat * cross_section_table_size;
                            const auto xs_abs =
                                ct_xs_device[base +
                                             static_cast<std::size_t>(cross_section_index)] +
                                cross_section_fraction *
                                    (ct_xs_device[base + static_cast<std::size_t>(
                                                              cross_section_index) +
                                                  1] -
                                     ct_xs_device[base + static_cast<std::size_t>(
                                                              cross_section_index)]);
                            const auto ref_rho =
                                sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                            macroscopic_cross_section_per_mm =
                                xs_abs *
                                (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) / ref_rho);
                        } else {
                            const auto water_xs =
                                cross_section_device[cross_section_index] +
                                cross_section_fraction *
                                    (cross_section_device[cross_section_index + 1] -
                                     cross_section_device[cross_section_index]);
                            macroscopic_cross_section_per_mm =
                                water_xs * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        }
                    } else if (in_insert && use_insert_material_tables) {
                        macroscopic_cross_section_per_mm =
                            insert_xs_device[static_cast<std::size_t>(cross_section_index)] +
                            cross_section_fraction *
                                (insert_xs_device[static_cast<std::size_t>(
                                                      cross_section_index) +
                                                  1] -
                                 insert_xs_device[static_cast<std::size_t>(
                                     cross_section_index)]);
                    } else if (material_table_count > 0) {
                        const auto base = static_cast<std::size_t>(layer_for_material) *
                                          cross_section_table_size;
                        macroscopic_cross_section_per_mm =
                            material_xs_device[base + static_cast<std::size_t>(
                                                          cross_section_index)] +
                            cross_section_fraction *
                                (material_xs_device[base + static_cast<std::size_t>(
                                                                cross_section_index) +
                                                    1] -
                                 material_xs_device[base + static_cast<std::size_t>(
                                                                cross_section_index)]);
                    } else {
                        macroscopic_cross_section_per_mm =
                            cross_section_device[cross_section_index] +
                            cross_section_fraction *
                                (cross_section_device[cross_section_index + 1] -
                                 cross_section_device[cross_section_index]);
                        if (slab_layer_count > 0 || in_insert) {
                            macroscopic_cross_section_per_mm *= local_density_g_per_cm3;
                        }
                    }
                    remaining_interaction_lengths -=
                        history_primary_inelastic_xs_scale *
                        macroscopic_cross_section_per_mm * step_mm;
                    if (remaining_interaction_lengths <= 0.0F) {
                        history_nuclear_MeV = energy_MeV;
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::primary_nuclear);
                        if (enable_secondary_generation) {
                            auto reaction_bin_index = static_cast<int>(sycl::floor(
                                (post_step_energy_MeVu - minimum_reaction_energy) *
                                inverse_reaction_energy_bin_width));
                            reaction_bin_index = sycl::max(
                                0, sycl::min(reaction_bin_index,
                                             static_cast<int>(reaction_energy_bin_count) - 1));
                            const auto reaction_bin = reaction_bins_device[reaction_bin_index];
                            const auto package_uniform =
                                rng::uniform01(spot_seed, rng_history, steps, 3);
                            const auto package_in_bin = sycl::min(
                                static_cast<std::uint32_t>(
                                    package_uniform * reaction_bin.reaction_count),
                                reaction_bin.reaction_count - 1U);
                            const auto reaction =
                                reactions_device[reaction_bin.reaction_offset + package_in_bin];
                            // Reaction packages are grouped in 1 MeV/u bins. Preserve
                            // their correlated final state, but scale product kinetic
                            // energies to the actual post-step projectile energy rather
                            // than silently using the sampled event's nearby energy.
                            const auto reaction_energy_scale =
                                reaction.incident_energy_MeV_per_u > 0.0F
                                    ? post_step_energy_MeVu /
                                          reaction.incident_energy_MeV_per_u
                                    : 1.0F;
                            secondary_summary.direct_count = reaction.secondary_count;

                            std::uint32_t queueable_count = 0;
                            auto queueable_energy_MeV = 0.0F;
                            std::uint32_t neutral_queueable_count = 0;
                            auto neutral_queueable_energy_MeV = 0.0F;
                            for (std::uint32_t secondary_index = 0;
                                 secondary_index < reaction.secondary_count;
                                 ++secondary_index) {
                                const auto secondary = reaction_secondaries_device[
                                    reaction.secondary_offset + secondary_index];
                                const auto is_neutral = secondary.pdg_id == 22 ||
                                                        secondary.pdg_id == 2112;
                                const auto is_supported = secondary.atomic_number > 0 &&
                                                          secondary.mass_number > 0;
                                const auto scaled_secondary_energy_MeV =
                                    secondary.kinetic_energy_MeV *
                                    reaction_energy_scale;
                                if (is_neutral) {
                                    if (enable_neutral_transport) {
                                        ++neutral_queueable_count;
                                        neutral_queueable_energy_MeV +=
                                            scaled_secondary_energy_MeV;
                                    } else {
                                        const auto kerma_frac = neutral_kerma_fraction_at_energy(
                                            energy_MeVu, neutral_local_kerma_fraction,
                                            neutral_kerma_high_energy_scale);
                                        const auto kerma_MeV =
                                            scaled_secondary_energy_MeV * kerma_frac;
                                        const auto residual_MeV =
                                            scaled_secondary_energy_MeV - kerma_MeV;
                                        secondary_summary.neutral_energy_MeV += residual_MeV;
                                        if (kerma_MeV > 0.0F) {
                                            // Interim n/γ kerma into "other" fragment channel,
                                            // distributed along +z; renormalize into phantom.
                                            constexpr std::size_t other_species = 6;
                                            score_exponential_depth(
                                                kerma_MeV, position_z_mm, depth_bin_width_mm,
                                                phantom_length_mm,
                                                static_cast<std::uint32_t>(number_of_bins),
                                                neutral_kerma_mean_free_path_mm,
                                                aggregate_secondary_dose_device, true);
                                            if (fragment_dose_device != nullptr) {
                                                score_exponential_depth(
                                                    kerma_MeV, position_z_mm,
                                                    depth_bin_width_mm, phantom_length_mm,
                                                    static_cast<std::uint32_t>(number_of_bins),
                                                    neutral_kerma_mean_free_path_mm,
                                                    fragment_dose_device +
                                                        other_species * number_of_bins,
                                                    true);
                                            }
                                            if (enable_voxel_scoring) {
                                                score_exponential_voxel_depth(
                                                    kerma_MeV, position_z_mm,
                                                    depth_bin_width_mm, phantom_length_mm,
                                                    static_cast<std::uint32_t>(number_of_bins),
                                                    neutral_kerma_mean_free_path_mm,
                                                    voxel_plane_size, voxel_index,
                                                    voxel_dose_device, true);
                                            }
                                        }
                                    }
                                } else if (is_supported) {
                                    const auto heavy_local =
                                        secondary_heavy_local_deposit_z_min > 0 &&
                                        secondary.atomic_number >=
                                            secondary_heavy_local_deposit_z_min;
                                    if (heavy_local) {
                                        // Ablation: Li+ kinetic energy → local heat
                                        // (not charged secondary transport).
                                        deposit_local_heat_device(
                                            scaled_secondary_energy_MeV, position_x_mm,
                                            position_y_mm, position_z_mm, direction_x,
                                            direction_y, direction_z, depth_bin_width_mm,
                                            number_of_bins, enable_voxel_scoring,
                                            voxel_min_x_mm, voxel_min_y_mm, voxel_size_x_mm,
                                            voxel_size_y_mm, voxel_bins_x, voxel_bins_y,
                                            voxel_plane_size, dose_device, nullptr, 0,
                                            voxel_dose_device,
                                            enable_charged_origin_voxel_scoring,
                                            charged_origin_voxel_dose_device, 0);
                                    } else {
                                        ++queueable_count;
                                        queueable_energy_MeV +=
                                            scaled_secondary_energy_MeV;
                                        // Birth spectrum before queue fit so overflow
                                        // does not bias production diagnostics.
                                        const auto child_direction =
                                            rotate_local_direction(
                                                secondary.direction_x,
                                                secondary.direction_y,
                                                secondary.direction_z,
                                                Direction3F{direction_x, direction_y,
                                                            direction_z});
                                        score_fragment_birth_device(
                                            birth_counts_device, birth_ke_sum_device,
                                            birth_mevu_hist_device, birth_depth_hist_device,
                                            number_of_bins, depth_bin_width_mm,
                                            birth_cos_hist_device,
                                            birth_parent_mevu_hist_device,
                                            birth_parent_z_hist_device,
                                            birth_parent_product_mevu_hist_device,
                                            secondary.atomic_number, secondary.mass_number,
                                            scaled_secondary_energy_MeV, position_z_mm,
                                            child_direction.z, /*generation=*/0,
                                            /*parent_Z=*/primary_atomic_number,
                                            /*parent_A=*/primary_mass_number, energy_MeV);
                                    }
                                }
                            }

                            if (queueable_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    queue_counter(*secondary_queue_counter_device);
                                const auto queue_offset =
                                    queue_counter.fetch_add(queueable_count);
                                const auto package_fits =
                                    queue_offset <= secondary_queue_capacity_u32 &&
                                    queueable_count <=
                                        secondary_queue_capacity_u32 - queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index < reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary = reaction_secondaries_device[
                                            reaction.secondary_offset + secondary_index];
                                        const auto is_supported =
                                            secondary.atomic_number > 0 &&
                                            secondary.mass_number > 0;
                                        const auto heavy_local =
                                            secondary_heavy_local_deposit_z_min > 0 &&
                                            secondary.atomic_number >=
                                                secondary_heavy_local_deposit_z_min;
                                        if (is_supported && !heavy_local) {
                                            const auto origin_category =
                                                charged_dose_category(
                                                    secondary.atomic_number,
                                                    secondary.mass_number);
                                            const auto child_direction = rotate_local_direction(
                                                secondary.direction_x,
                                                secondary.direction_y,
                                                secondary.direction_z,
                                                Direction3F{direction_x, direction_y, direction_z});
                                            secondary_queue_device[output_index++] =
                                                SecondaryParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    secondary.kinetic_energy_MeV *
                                                        reaction_energy_scale,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    secondary.pdg_id,
                                                    secondary.atomic_number,
                                                    secondary.mass_number,
                                                    origin_category,
                                                    0,
                                                    charged_lineage,
                                                    rng::child_stream(
                                                        global_history,
                                                        rng::branch_tag(
                                                            rng::branch_role_primary_charged,
                                                            secondary_index)),
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled_counter(*secondary_queue_filled_device);
                                    filled_counter.fetch_add(queueable_count);
                                    secondary_summary.queued_count = queueable_count;
                                    secondary_summary.queued_energy_MeV =
                                        queueable_energy_MeV;
                                } else {
                                    // Queue full: fold overflow charged KE into residual
                                    // local heat; keep energy diagnostic for tests/budget.
                                    secondary_summary.overflow_count = queueable_count;
                                    secondary_summary.overflow_energy_MeV =
                                        queueable_energy_MeV;
                                }
                            }
                            if (neutral_queueable_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    neutral_counter(*neutral_queue_counter_device);
                                const auto neutral_offset =
                                    neutral_counter.fetch_add(neutral_queueable_count);
                                const auto neutral_fits =
                                    neutral_offset <= neutral_queue_capacity_u32 &&
                                    neutral_queueable_count <=
                                        neutral_queue_capacity_u32 - neutral_offset;
                                if (neutral_fits) {
                                    auto output_index = neutral_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index < reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary = reaction_secondaries_device[
                                            reaction.secondary_offset + secondary_index];
                                        if (secondary.pdg_id == 22 ||
                                            secondary.pdg_id == 2112) {
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    secondary.direction_x,
                                                    secondary.direction_y,
                                                    secondary.direction_z,
                                                    Direction3F{direction_x, direction_y,
                                                                direction_z});
                                            const auto lineage =
                                                neutral_lineage_from_pdg(secondary.pdg_id);
                                            neutral_queue_device[output_index++] =
                                                NeutralParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    secondary.kinetic_energy_MeV *
                                                        reaction_energy_scale,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    secondary.pdg_id,
                                                    static_cast<std::uint8_t>(
                                                        neutral_origin_category_from_lineage(
                                                            lineage)),
                                                    0,
                                                    0,
                                                    rng::child_stream(
                                                        global_history,
                                                        rng::branch_tag(
                                                            rng::branch_role_primary_neutral,
                                                            secondary_index)),
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled_counter(*neutral_queue_filled_device);
                                    filled_counter.fetch_add(neutral_queueable_count);
                                    secondary_summary.queued_neutral_count =
                                        neutral_queueable_count;
                                    secondary_summary.queued_neutral_energy_MeV =
                                        neutral_queueable_energy_MeV;
                                } else {
                                    secondary_summary.neutral_queue_overflow_count =
                                        neutral_queueable_count;
                                    secondary_summary.neutral_queue_overflow_energy_MeV =
                                        neutral_queueable_energy_MeV;
                                    secondary_summary.neutral_energy_MeV +=
                                        neutral_queueable_energy_MeV;
                                }
                            }
                            const auto residual_nuclear_MeV =
                                reaction.local_deposit_MeV * reaction_energy_scale;
                            deposit_local_heat_device(
                                residual_nuclear_MeV, position_x_mm, position_y_mm,
                                position_z_mm, direction_x, direction_y, direction_z,
                                depth_bin_width_mm, number_of_bins, enable_voxel_scoring,
                                voxel_min_x_mm, voxel_min_y_mm, voxel_size_x_mm,
                                voxel_size_y_mm, voxel_bins_x, voxel_bins_y,
                                voxel_plane_size, dose_device, nullptr, 0,
                                voxel_dose_device, enable_charged_origin_voxel_scoring,
                                charged_origin_voxel_dose_device, 0);
                        }
                        energy_MeV = 0.0f;
                    }
                }
                ++steps;
                profile_add(profile_counters_device, TransportProfileSlot::primary_steps);
            }

            const auto stopped_inside =
                energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                position_z_mm < phantom_length_mm &&
                (!enable_voxel_scoring ||
                 (position_x_mm >= voxel_min_x_mm && position_x_mm < voxel_max_x_mm &&
                  position_y_mm >= voxel_min_y_mm && position_y_mm < voxel_max_y_mm));
            if (stopped_inside) {
                auto bin = direction_z < 0.0F
                               ? static_cast<int>(
                                     sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                               : static_cast<int>(
                                     sycl::floor(position_z_mm / depth_bin_width_mm));
                bin = sycl::max(
                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                if (enable_let_scoring &&
                    last_primary_stopping_power_MeV_per_mm > 0.0F) {
                    const auto local_cutoff_MeV =
                        energy_MeV * (1.0F - last_primary_let_delta_fraction);
                    score_letd_moments_device(
                        let_moments_device, number_of_bins,
                        static_cast<std::size_t>(bin), species_let_moments_device,
                        charged_origin_category_count, 0, nullptr, 0, 0,
                        local_cutoff_MeV, energy_MeV,
                        last_primary_stopping_power_MeV_per_mm,
                        last_primary_density_g_per_cm3, true);
                }
                sycl::atomic_ref<DoseAtomicT,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[bin]);
                atomic_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                if (enable_voxel_scoring) {
                    const auto x_coordinate =
                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                    const auto y_coordinate =
                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                    auto voxel_x = direction_x < 0.0F
                                       ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                       : static_cast<int>(sycl::floor(x_coordinate));
                    auto voxel_y = direction_y < 0.0F
                                       ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                       : static_cast<int>(sycl::floor(y_coordinate));
                    voxel_x = sycl::max(
                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                    voxel_y = sycl::max(
                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                    const auto voxel_index =
                        static_cast<std::size_t>(bin) * voxel_plane_size +
                        static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                        static_cast<std::size_t>(voxel_x);
                    if (enable_let_scoring &&
                        last_primary_stopping_power_MeV_per_mm > 0.0F) {
                        score_letd_moments_device(
                            voxel_let_moments_device, number_of_voxels,
                            voxel_index, nullptr, 0, 0, nullptr, 0, 0,
                            energy_MeV *
                                (1.0F - last_primary_let_delta_fraction),
                            energy_MeV,
                            last_primary_stopping_power_MeV_per_mm,
                            last_primary_density_g_per_cm3, true);
                    }
                    sycl::atomic_ref<DoseAtomicT,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_voxel_dose(voxel_dose_device[voxel_index]);
                    atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                    if (enable_charged_origin_voxel_scoring) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_category_dose(
                                charged_origin_voxel_dose_device[voxel_index]);
                        atomic_category_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                    }
                }
                history_deposited_MeV += energy_MeV;
                energy_MeV = 0.0F;
            }
            if (pending_primary_depth_MeV > 0.0) {
                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[pending_primary_bin]);
                atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                profile_add(profile_counters_device,
                            TransportProfileSlot::primary_dose_depth_atomics);
            }
            if (enable_voxel_scoring && pending_primary_voxel_MeV > 0.0) {
                std::size_t target_voxel = pending_primary_voxel;
                if (pending_primary_voxel_delayed_MeV > 0.0) {
                    target_voxel = electronic_voxel_target(
                        target_voxel,
                        static_cast<float>(pending_primary_voxel_MeV),
                        static_cast<float>(pending_primary_voxel_delayed_MeV),
                        pending_primary_electronic_mfp_mm,
                        electronic_buildup_lateral_sigma_mm,
                        depth_bin_width_mm, number_of_bins,
                        voxel_size_x_mm, voxel_size_y_mm,
                        voxel_bins_x, voxel_bins_y,
                        rng::uniform01(spot_seed, rng_history, steps, 50),
                        rng::uniform01(spot_seed, rng_history, steps, 53),
                        rng::uniform01(spot_seed, rng_history, steps, 51),
                        rng::uniform01(spot_seed, rng_history, steps, 52));
                }
                if (target_voxel != static_cast<std::size_t>(-1)) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_voxel_dose(voxel_dose_device[target_voxel]);
                    atomic_voxel_dose.fetch_add(
                        static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_dose_voxel_atomics);
                    if (enable_charged_origin_voxel_scoring) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_category_dose(
                                charged_origin_voxel_dose_device[target_voxel]);
                        atomic_category_dose.fetch_add(
                            static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                    }
                }
            }
            deposited_device[global_history] = history_deposited_MeV;
            escaped_device[global_history] = energy_MeV;
            nuclear_device[global_history] = history_nuclear_MeV;
            steps_device[global_history] = steps;
            if (enable_secondary_generation) {
                secondary_summaries_device[global_history] = secondary_summary;
            }
        });
        kernel_event.wait_and_throw();
        primary_kernel_seconds += event_duration_seconds(kernel_event);
        const auto primary_done = hist_offset + chunk_count;
        if (primary_done == number_of_histories ||
            hist_offset == 0 ||
            (progress_log_every_histories > 0 &&
             primary_done / progress_log_every_histories !=
                 hist_offset / progress_log_every_histories)) {
            std::cout << "  primary: " << primary_done << '/' << number_of_histories
                      << " histories\n"
                      << std::flush;
        }
    }

    std::uint64_t transported_queue_count = 0;
    if (enable_secondary_transport) {
        std::uint64_t generation_begin = 0;
        std::uint64_t generation_end = 0;
        queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
        std::uint32_t secondary_generation_index = 0;
        while (generation_begin < generation_end) {
            // Capture the generation limit before cascade products extend the queue.
            const auto generation_limit = generation_end;
            std::uint64_t batch_begin = generation_begin;
            std::uint64_t last_logged_secondary = generation_begin;
            std::cout << "  secondary gen " << secondary_generation_index << ": "
                      << generation_limit - generation_begin << " particles ["
                      << generation_begin << ',' << generation_limit << ")\n"
                      << std::flush;
            while (batch_begin < generation_limit) {
            const auto batch_end = std::min<std::uint64_t>(
                batch_begin + static_cast<std::uint64_t>(secondary_batch),
                generation_limit);
            const auto batch_size = batch_end - batch_begin;
            const auto secondary_sort_global_size =
                ((batch_size + local_size - 1) / local_size) * local_size;
            auto* secondary_input_device = secondary_queue_device;
            std::uint64_t* secondary_input_index_device = nullptr;
            if (enable_secondary_energy_sorting) {
                queue.memset(secondary_bucket_counters_device, 0,
                             4 * sizeof(std::uint64_t));
                auto count_event = queue.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{secondary_sort_global_size},
                                      sycl::range<1>{local_size}},
                    [=](sycl::nd_item<1> item) {
                        const auto lane = item.get_global_linear_id();
                        if (lane >= batch_size) {
                            return;
                        }
                        const auto& particle =
                            secondary_queue_device[batch_begin + lane];
                        const auto energy_MeVu =
                            particle.kinetic_energy_MeV /
                            sycl::fmax(1.0F, static_cast<float>(particle.mass_number));
                        const auto bucket =
                            energy_MeVu < 2.0F
                                ? 0U
                                : (energy_MeVu < 10.0F
                                       ? 1U
                                       : (energy_MeVu < 50.0F ? 2U : 3U));
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            counter(secondary_bucket_counters_device[bucket]);
                        counter.fetch_add(1);
                    });
                count_event.wait_and_throw();
                std::array<std::uint64_t, 4> bucket_counts{};
                queue.copy(secondary_bucket_counters_device,
                           bucket_counts.data(), bucket_counts.size())
                    .wait_and_throw();
                std::array<std::uint64_t, 4> bucket_offsets{
                    0,
                    bucket_counts[0],
                    bucket_counts[0] + bucket_counts[1],
                    bucket_counts[0] + bucket_counts[1] + bucket_counts[2]};
                queue.copy(bucket_offsets.data(), secondary_bucket_counters_device,
                           bucket_offsets.size())
                    .wait_and_throw();
                auto scatter_event = queue.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{secondary_sort_global_size},
                                      sycl::range<1>{local_size}},
                    [=](sycl::nd_item<1> item) {
                        const auto lane = item.get_global_linear_id();
                        if (lane >= batch_size) {
                            return;
                        }
                        const auto particle_index = batch_begin + lane;
                        const auto particle = secondary_queue_device[particle_index];
                        const auto energy_MeVu =
                            particle.kinetic_energy_MeV /
                            sycl::fmax(1.0F, static_cast<float>(particle.mass_number));
                        const auto bucket =
                            energy_MeVu < 2.0F
                                ? 0U
                                : (energy_MeVu < 10.0F
                                       ? 1U
                                       : (energy_MeVu < 50.0F ? 2U : 3U));
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            counter(secondary_bucket_counters_device[bucket]);
                        const auto output_index = counter.fetch_add(1);
                        secondary_bucket_scratch_device[output_index] = particle;
                        secondary_bucket_index_device[output_index] = particle_index;
                    });
                scatter_event.wait_and_throw();
                const auto sort_seconds = event_duration_seconds(count_event) +
                                          event_duration_seconds(scatter_event);
                secondary_kernel_seconds += sort_seconds;
                secondary_input_device = secondary_bucket_scratch_device;
                secondary_input_index_device = secondary_bucket_index_device;
            }
            const auto persistent_workers_enabled =
                secondary_persistent_workers > 0;
            const auto worker_count = persistent_workers_enabled
                                          ? std::min<std::uint64_t>(
                                                batch_size,
                                                secondary_persistent_workers)
                                          : batch_size;
            const auto secondary_global_size =
                ((worker_count + local_size - 1) / local_size) * local_size;
            if (persistent_workers_enabled) {
                queue.memset(secondary_work_counter_device, 0,
                             sizeof(std::uint64_t))
                    .wait_and_throw();
            }
            auto secondary_kernel_event = queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{secondary_global_size},
                              sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                for (;;) {
                const auto lane =
                    persistent_workers_enabled
                        ? sycl::atomic_ref<
                              std::uint64_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(
                              *secondary_work_counter_device)
                              .fetch_add(1)
                        : item.get_global_linear_id();
                if (lane >= batch_size) {
                    return;
                }
                const auto particle_index = enable_secondary_energy_sorting
                                                ? secondary_input_index_device[lane]
                                                : batch_begin + lane;
                auto deposited_MeV = 0.0F;
                auto escaped_MeV = 0.0F;
                std::uint32_t steps = 0;
                CascadeTransportSummary cascade_summary{};
                {
                    const auto particle = enable_secondary_energy_sorting
                                              ? secondary_input_device[lane]
                                              : secondary_input_device[particle_index];
                    // Philox stream is particle identity, not atomic queue slot.
                    const auto rng_stream = particle.rng_stream;
                    auto energy_MeV = particle.kinetic_energy_MeV;
                    auto position_x_mm = particle.position_x_mm;
                    auto position_y_mm = particle.position_y_mm;
                    auto position_z_mm = particle.position_z_mm;
                    auto direction_x = particle.direction_x;
                    auto direction_y = particle.direction_y;
                    auto direction_z = sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                    const auto atomic_number = static_cast<int>(particle.atomic_number);
                    const auto mass_number = static_cast<int>(particle.mass_number);
                    const auto inverse_mass_number_for_particle =
                        1.0F / static_cast<float>(mass_number);
                    constexpr std::size_t particle_mass_stride = 32;
                    const auto particle_table_species =
                        static_cast<std::size_t>(atomic_number) *
                            particle_mass_stride +
                        static_cast<std::size_t>(mass_number);
                    const auto has_particle_specific_table =
                        use_particle_specific_stopping_power &&
                        atomic_number > 0 && atomic_number < 10 &&
                        mass_number > 0 && mass_number < 32 &&
                        particle_species_present_device[
                            particle_table_species] != 0;
                    const auto particle_table_base =
                        particle_table_species * table_size;
                    const auto cascade_projectile_index_for_particle =
                        enable_fragment_cascade &&
                                particle.generation < maximum_cascade_generations
                            ? cascade_projectile_index(
                                  cascade_projectiles_device, cascade_projectile_count,
                                  atomic_number, mass_number)
                            : -1;
                    CascadeProjectile cascade_projectile{};
                    if (cascade_projectile_index_for_particle >= 0) {
                        cascade_projectile = cascade_projectiles_device[
                            cascade_projectile_index_for_particle];
                    }
                    const auto charge = static_cast<float>(atomic_number);
                    const auto charge_power = sycl::pow(charge, -2.0F / 3.0F);
                    const auto is_neutral_lineage =
                        particle.reserved == neutron_lineage ||
                        particle.reserved == gamma_lineage;
                    const auto neutral_origin =
                        neutral_origin_category_from_lineage(particle.reserved);
                    const auto species_index =
                        sycl::min(static_cast<std::size_t>(particle.origin_category),
                                  fragment_species_count - 1);
                    const auto charged_origin_voxel_offset =
                        (species_index + 1) * number_of_voxels;
                    const auto neutral_origin_voxel_offset =
                        neutral_origin * number_of_voxels;
                    auto pending_dose_MeV = DoseAtomicT{0};
                    auto pending_bin = 0;
                    std::size_t pending_voxel_index = 0;
                    // Continuous losses can be smaller than one FP32 ULP at
                    // multi-GeV fragment energies. Keep the unrepresented
                    // remainder so scored energy is eventually removed from
                    // the transported particle instead of being counted twice
                    // in the escaped-energy balance.
                    auto energy_loss_residual_MeV = 0.0F;
                    // Primary already caps steps; secondary previously did not, so a
                    // voxel-boundary nudge thrash could run for minutes on CUDA.
                    constexpr std::uint32_t max_secondary_steps = 500'000U;
                    auto last_stopping_power_MeV_per_mm = 0.0F;
                    auto last_density_g_per_cm3 =
                        static_cast<float>(water_density_g_per_cm3);

                    while (energy_MeV > secondary_local_deposit_cutoff_MeV &&
                           steps < max_secondary_steps) {
                        const auto escaped_z =
                            position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                        const auto escaped_xy =
                            enable_voxel_scoring &&
                            (position_x_mm < voxel_min_x_mm || position_x_mm >= voxel_max_x_mm ||
                             position_y_mm < voxel_min_y_mm || position_y_mm >= voxel_max_y_mm);
                        if (escaped_z || escaped_xy) {
                            break;
                        }
                        const auto absolute_direction_x = sycl::fabs(direction_x);
                        const auto absolute_direction_y = sycl::fabs(direction_y);
                        const auto absolute_direction_z = sycl::fabs(direction_z);

                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_z_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                        auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                        auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                        if (enable_voxel_scoring) {
                            const auto x_coordinate =
                                (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                            const auto y_coordinate =
                                (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                            voxel_x = direction_x < 0.0F
                                          ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(x_coordinate));
                            voxel_y = direction_y < 0.0F
                                          ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(y_coordinate));
                            voxel_x = sycl::max(
                                0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                            voxel_y = sycl::max(
                                0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                        }
                        const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                                 static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                                 static_cast<std::size_t>(voxel_x);

                        if (absolute_direction_x < 1.0e-6F &&
                            absolute_direction_y < 1.0e-6F && absolute_direction_z < 1.0e-6F) {
                            score_secondary_dose_device(
                                pending_dose_MeV, is_neutral_lineage, species_index,
                                neutral_origin, pending_bin, number_of_bins,
                                enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring,
                                pending_voxel_index, charged_origin_voxel_offset,
                                neutral_origin_voxel_offset, aggregate_secondary_dose_device, fragment_dose_device,
                                voxel_dose_device, charged_origin_voxel_dose_device,
                                neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            pending_dose_MeV = DoseAtomicT{0};
                            score_secondary_dose_device(
                                energy_MeV, is_neutral_lineage, species_index, neutral_origin,
                                bin, number_of_bins, enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring, voxel_index,
                                charged_origin_voxel_offset, neutral_origin_voxel_offset,
                                aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
                                charged_origin_voxel_dose_device, neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            deposited_MeV += energy_MeV;
                            energy_MeV = 0.0F;
                            break;
                        }

                        const auto energy_MeVu =
                            energy_MeV * inverse_mass_number_for_particle;
                        auto floating_index =
                            (energy_MeVu - minimum_table_energy) * inverse_table_step;
                        auto index = static_cast<int>(sycl::floor(floating_index));
                        index = sycl::max(
                            0, sycl::min(index, static_cast<int>(table_size) - 2));
                        const auto fraction = sycl::clamp(
                            floating_index - static_cast<float>(index), 0.0F, 1.0F);
                        const auto carbon_stopping_power_MeV_per_mm =
                            table_device[index] +
                            fraction * (table_device[index + 1] - table_device[index]);

                        auto charge_ratio = 1.0F;
                        if (!has_particle_specific_table) {
                            constexpr float nucleon_mass_MeV = 931.49410242F;
                            const auto gamma =
                                1.0F + energy_MeVu / nucleon_mass_MeV;
                            const auto beta_squared =
                                sycl::fmax(0.0F, 1.0F - 1.0F / (gamma * gamma));
                            const auto beta = sycl::sqrt(beta_squared);
                            const auto effective_charge =
                                charge *
                                (1.0F -
                                 sycl::exp(-125.0F * beta * charge_power));
                            const auto reference_effective_charge =
                                primary_charge *
                                (1.0F -
                                 sycl::exp(-125.0F * beta *
                                           primary_charge_power));
                            charge_ratio =
                                effective_charge / reference_effective_charge;
                        }
                        const auto in_insert =
                            enable_hetero_insert &&
                            inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                                 insert_x_min, insert_x_max, insert_y_min,
                                                 insert_y_max, insert_z_min, insert_z_max);
                        auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                            position_z_mm, slab_z_ends_device, slab_densities_device,
                            slab_layer_count, water_density_g_per_cm3);
                        if (in_insert) {
                            local_density_g_per_cm3 = insert_density_g_per_cm3;
                        }
                        std::uint8_t ct_material = 2;
                        auto in_ct = false;
                        if (enable_ct_grid) {
                            float ct_rho = water_density_g_per_cm3;
                            in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                              ct_origin_x, ct_origin_y, ct_origin_z,
                                              ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                              ct_nx, ct_ny, ct_nz, ct_density_device,
                                              ct_material_device, ct_rho, ct_material);
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_ct_samples);
                            if (in_ct) {
                                local_density_g_per_cm3 = ct_rho;
                            }
                        }
                        const auto layer_for_material =
                            slab_layer_count > 0
                                ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                                   slab_layer_count)
                                : 0U;
                        float carbon_sp_local = carbon_stopping_power_MeV_per_mm;
                        if (enable_ct_grid) {
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_sp_table_lookups);
                            if (use_ct_mass_sp && in_ct &&
                                ct_mass_sp_factor_lut_device != nullptr &&
                                ct_n_mass_factors > 0) {
                                const auto mass_factor = ct_lookup_mass_sp_factor(
                                    ct_mass_sp_factor_lut_device,
                                    use_ct_density_mass_spr ? ct_density_spr_n_rho
                                                            : ct_n_mass_factors,
                                    table_size, use_ct_density_mass_spr,
                                    ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                    static_cast<std::uint32_t>(ct_material),
                                    local_density_g_per_cm3,
                                    static_cast<std::size_t>(index), fraction,
                                    [](float x) { return sycl::log(x); });
                                carbon_sp_local = ct_mass_scaled_stopping_power(
                                    carbon_stopping_power_MeV_per_mm,
                                    local_density_g_per_cm3, mass_factor);
                                profile_add(profile_counters_device,
                                            TransportProfileSlot::secondary_mass_sp_lookups);
                            } else if (use_ct_material_sp && in_ct &&
                                       ct_sp_device != nullptr &&
                                       ct_ref_density_device != nullptr) {
                                const auto mat = static_cast<std::uint32_t>(ct_material_class(
                                    ct_material, ct_material_ids_are_schneider_sections));
                                const auto base = mat * table_size;
                                const auto sp_abs =
                                    ct_sp_device[base + static_cast<std::size_t>(index)] +
                                    fraction *
                                        (ct_sp_device[base +
                                                     static_cast<std::size_t>(index) + 1] -
                                         ct_sp_device[base +
                                                     static_cast<std::size_t>(index)]);
                                const auto ref_rho =
                                    sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                                carbon_sp_local =
                                    sp_abs *
                                    (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) /
                                     ref_rho);
                            } else {
                                carbon_sp_local = carbon_stopping_power_MeV_per_mm *
                                                  sycl::fmax(local_density_g_per_cm3,
                                                             1.0e-6F);
                            }
                        } else if (in_insert && use_insert_material_tables) {
                            carbon_sp_local =
                                insert_sp_device[static_cast<std::size_t>(index)] +
                                fraction *
                                    (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                                     insert_sp_device[static_cast<std::size_t>(index)]);
                        } else if (material_table_count > 0) {
                            const auto base =
                                static_cast<std::size_t>(layer_for_material) * table_size;
                            carbon_sp_local =
                                material_sp_device[base + static_cast<std::size_t>(index)] +
                                fraction *
                                    (material_sp_device[base +
                                                       static_cast<std::size_t>(index) + 1] -
                                     material_sp_device[base +
                                                       static_cast<std::size_t>(index)]);
                        }
                        auto stopping_power_MeV_per_mm =
                            carbon_sp_local * charge_ratio * charge_ratio;
                        if (has_particle_specific_table) {
                            const auto exact_ratio =
                                particle_sp_ratio_device[
                                    particle_table_base +
                                    static_cast<std::size_t>(index)] +
                                fraction *
                                    (particle_sp_ratio_device[
                                         particle_table_base +
                                         static_cast<std::size_t>(index) + 1] -
                                     particle_sp_ratio_device[
                                         particle_table_base +
                                         static_cast<std::size_t>(index)]);
                            stopping_power_MeV_per_mm =
                                carbon_sp_local * exact_ratio;
                        }
                        if (enable_minibeam && !enable_ct_grid &&
                            !enable_layered_phantom &&
                            !enable_hetero_insert &&
                            species_index ==
                                primary_charged_origin_category &&
                            atomic_number == 6 && mass_number == 12) {
                            stopping_power_MeV_per_mm *=
                                minibeam_water_primary_stopping_power_scale;
                        }
                        last_stopping_power_MeV_per_mm =
                            stopping_power_MeV_per_mm;
                        last_density_g_per_cm3 =
                            local_density_g_per_cm3;
                        // Density scale only for density-only slab/insert (not absolute
                        // material tables, not CT which already scaled carbon_sp_local).
                        if ((slab_layer_count > 0 || in_insert) && !enable_ct_grid &&
                            !(in_insert && use_insert_material_tables) &&
                            material_table_count == 0) {
                            stopping_power_MeV_per_mm *= local_density_g_per_cm3;
                        }
                        auto path_step_mm = sycl::fmin(
                            enable_secondary_condensed_history
                                ? secondary_condensed_step_mm
                                : maximum_step_mm,
                            maximum_relative_energy_loss * energy_MeV /
                                sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                        const auto energy_limited_step_mm = path_step_mm;
                        const auto boundary_z_mm =
                            enable_secondary_condensed_history
                                ? (direction_z < 0.0F ? 0.0F
                                                     : phantom_length_mm)
                                : (direction_z < 0.0F
                                       ? static_cast<float>(bin) *
                                             depth_bin_width_mm
                                       : static_cast<float>(bin + 1) *
                                             depth_bin_width_mm);
                        const auto distance_to_boundary_mm =
                            direction_z < 0.0F ? position_z_mm - boundary_z_mm
                                               : boundary_z_mm - position_z_mm;
                        path_step_mm = sycl::fmin(
                            path_step_mm, distance_to_boundary_mm / absolute_direction_z);
                        if (slab_layer_count > 0) {
                            path_step_mm = sycl::fmin(
                                path_step_mm,
                                distance_to_slab_interface_mm(
                                    position_z_mm, direction_z, slab_z_ends_device,
                                    slab_layer_count, phantom_length_mm));
                        }
                        if (enable_hetero_insert) {
                            path_step_mm = sycl::fmin(
                                path_step_mm,
                                distance_to_insert_interface_mm(
                                    position_x_mm, position_y_mm, position_z_mm, direction_x,
                                    direction_y, direction_z, insert_x_min, insert_x_max,
                                    insert_y_min, insert_y_max, insert_z_min, insert_z_max,
                                    phantom_length_mm));
                        }
                        if (enable_ct_grid && in_ct) {
                            CtClampPath clamp_path = CtClampPath::three_axis;
                            path_step_mm = clamp_step_to_ct_faces_if_needed(
                                path_step_mm, position_x_mm, position_y_mm, position_z_mm,
                                direction_x, direction_y, direction_z, ct_origin_x,
                                ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                                ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                                ct_material_device, local_density_g_per_cm3, ct_material,
                                ct_skip_homogeneous_face_clamp,
#ifdef CARBON_TRANSPORT_PROFILE
                                &clamp_path
#else
                                nullptr
#endif
                            );
                            profile_face(profile_counters_device, false, clamp_path);
                        }
                        if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                            absolute_direction_x >= 1.0e-6F) {
                            const auto boundary_x_mm =
                                voxel_min_x_mm +
                                static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                    voxel_size_x_mm;
                            const auto distance_to_boundary_x_mm =
                                (boundary_x_mm - position_x_mm) / direction_x;
                            path_step_mm = sycl::fmin(path_step_mm, distance_to_boundary_x_mm);
                        }
                        if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                            absolute_direction_y >= 1.0e-6F) {
                            const auto boundary_y_mm =
                                voxel_min_y_mm +
                                static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                    voxel_size_y_mm;
                            const auto distance_to_boundary_y_mm =
                                (boundary_y_mm - position_y_mm) / direction_y;
                            path_step_mm = sycl::fmin(path_step_mm, distance_to_boundary_y_mm);
                        }
                        const auto min_step_accept =
                            (enable_ct_grid && in_ct) ? 1.0e-8F : 1.0e-6F;
                        if (path_step_mm <= min_step_accept) {
                            auto snapped_to_boundary = false;
                            if (absolute_direction_z >= 1.0e-6F &&
                                distance_to_boundary_mm / absolute_direction_z <= 1.0e-6F) {
                                position_z_mm = nudge_past_axis_boundary(
                                    boundary_z_mm, direction_z,
                                robust_boundary_nudge);
                                snapped_to_boundary = true;
                            }
                            if (slab_layer_count > 0 && absolute_direction_z >= 1.0e-6F) {
                                const auto layer = slab_layer_index(
                                    position_z_mm, slab_z_ends_device, slab_layer_count);
                                const auto interface_z =
                                    direction_z > 0.0F
                                        ? slab_z_ends_device[layer]
                                        : (layer == 0U ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                if (sycl::fabs((interface_z - position_z_mm) / direction_z) <=
                                    1.0e-6F) {
                                    position_z_mm = nudge_past_axis_boundary(
                                        interface_z, direction_z,
                                robust_boundary_nudge);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                                absolute_direction_x >= 1.0e-6F) {
                                const auto boundary_x_mm =
                                    voxel_min_x_mm +
                                    static_cast<float>(
                                        voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                        voxel_size_x_mm;
                                if ((boundary_x_mm - position_x_mm) / direction_x <= 1.0e-6F) {
                                    position_x_mm = nudge_past_axis_boundary(
                                        boundary_x_mm, direction_x,
                                robust_boundary_nudge);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                                absolute_direction_y >= 1.0e-6F) {
                                const auto boundary_y_mm =
                                    voxel_min_y_mm +
                                    static_cast<float>(
                                        voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                        voxel_size_y_mm;
                                if ((boundary_y_mm - position_y_mm) / direction_y <= 1.0e-6F) {
                                    position_y_mm = nudge_past_axis_boundary(
                                        boundary_y_mm, direction_y,
                                robust_boundary_nudge);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_ct_grid && in_ct) {
                                constexpr float nudge = 1.0e-4F;
                                position_x_mm += direction_x * nudge;
                                position_y_mm += direction_y * nudge;
                                position_z_mm += direction_z * nudge;
                                snapped_to_boundary = true;
                            }
                            if (snapped_to_boundary) {
                                ++steps;
                                profile_add(
                                    profile_counters_device,
                                    TransportProfileSlot::
                                        secondary_boundary_nudge_continues);
                                continue;
                            }
                        }
                        // A positive face-limited step can still be smaller than one
                        // FP32 ULP at the current coordinates.  CUDA then accepts the
                        // step but changes neither position nor energy, so the same
                        // face is selected until max_secondary_steps is reached.
                        // Advance by the same 1e-4 mm tolerance used by CT DDA when it
                        // classifies a point as lying on a face.  A single nextafter
                        // is not sufficient on CUDA because the new point can remain
                        // inside that face tolerance and select another zero-state
                        // step.  The nextafter fallback guarantees progress for a
                        // very small direction component whose nudge still rounds
                        // back to the original coordinate.
                        const auto proposed_position_x_mm =
                            position_x_mm + direction_x * path_step_mm;
                        const auto proposed_position_y_mm =
                            position_y_mm + direction_y * path_step_mm;
                        const auto proposed_position_z_mm =
                            position_z_mm + direction_z * path_step_mm;
                        const auto boundary_limited =
                            path_step_mm < energy_limited_step_mm;
                        if (boundary_limited &&
                            proposed_position_x_mm == position_x_mm &&
                            proposed_position_y_mm == position_y_mm &&
                            proposed_position_z_mm == position_z_mm) {
                            constexpr float nudge_mm = 1.0e-4F;
                            position_x_mm = advance_representable(
                                position_x_mm, direction_x, nudge_mm);
                            position_y_mm = advance_representable(
                                position_y_mm, direction_y, nudge_mm);
                            position_z_mm = advance_representable(
                                position_z_mm, direction_z, nudge_mm);
                            ++steps;
                            profile_add(
                                profile_counters_device,
                                TransportProfileSlot::secondary_forced_progress_nudges);
                            continue;
                        }
                        const auto mean_step_loss_MeV =
                            stopping_power_MeV_per_mm * path_step_mm;
                        auto step_deposited_MeV =
                            sycl::fmin(mean_step_loss_MeV, energy_MeV);
                        if (enable_secondary_energy_straggling) {
                            const auto uniform1 = sycl::fmax(
                                rng::uniform01(random_seed, rng_stream, steps, 4),
                                1.0e-12F);
                            const auto uniform2 =
                                rng::uniform01(random_seed, rng_stream, steps, 5);
                            const auto extra_uniform =
                                rng::uniform01(random_seed, rng_stream, steps, 6);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gaussian =
                                sycl::sqrt(-2.0F * sycl::log(uniform1)) *
                                sycl::cos(two_pi * uniform2);
                            constexpr float nucleon_mass_MeV = 931.49410242F;
                            const auto gamma =
                                1.0F + energy_MeVu / nucleon_mass_MeV;
                            const auto beta_squared = sycl::fmax(
                                0.0F, 1.0F - 1.0F / (gamma * gamma));
                            const auto beta = sycl::sqrt(beta_squared);
                            const auto fragment_effective_charge =
                                charge *
                                (1.0F -
                                 sycl::exp(-125.0F * beta * charge_power));
                            auto za_rel = 1.0F;
                            if (enable_ct_grid && in_ct) {
                                if (use_ct_mass_sp &&
                                    ct_mass_sp_za_rel_device != nullptr &&
                                    ct_n_mass_factors > 0) {
                                    const auto section =
                                        static_cast<std::uint32_t>(ct_material);
                                    const auto factor_index =
                                        section < ct_n_mass_factors
                                            ? section
                                            : (ct_n_mass_factors - 1U);
                                    za_rel = sycl::clamp(
                                        ct_mass_sp_za_rel_device[factor_index],
                                        0.5F, 1.5F);
                                } else {
                                    const auto material = ct_material_class(
                                        ct_material,
                                        ct_material_ids_are_schneider_sections);
                                    za_rel = material == 3U ? 0.93F : 1.0F;
                                }
                            }
                            const auto variance_MeV2 =
                                condensed_total_loss_variance_MeV2_device(
                                    energy_MeVu, mass_number,
                                    fragment_effective_charge, path_step_mm,
                                    local_density_g_per_cm3, za_rel);
                            const auto local_straggling_scale =
                                interpolate_straggling_scale(
                                    energy_MeVu, straggling_scale_energies,
                                    straggling_scale_values,
                                    straggling_scale_point_count, straggling_scale);
                            const auto sigma_MeV =
                                local_straggling_scale *
                                sycl::sqrt(sycl::fmax(0.0F, variance_MeV2));
                            step_deposited_MeV = sample_condensed_energy_loss(
                                mean_step_loss_MeV, sigma_MeV, gaussian,
                                extra_uniform, energy_MeV, straggling_sampler);
                        }
                        const auto sec_e_frac =
                            enable_minibeam &&
                                    minibeam_electronic_buildup_primary_only
                                ? 0.0F
                                : electronic_buildup_fraction_at_energy(
                                      energy_MeVu,
                                      electronic_buildup_fraction);
                        const auto sec_delayed = step_deposited_MeV * sec_e_frac;
                        const auto sec_local = step_deposited_MeV - sec_delayed;
                        if (enable_let_scoring) {
                            auto let_delta_fraction =
                                use_let_delta_fraction_table
                                    ? let_delta_fraction_device[index] +
                                          fraction *
                                              (let_delta_fraction_device[index + 1] -
                                               let_delta_fraction_device[index])
                                    : sec_e_frac;
                            if (has_particle_specific_table) {
                                let_delta_fraction =
                                    particle_delta_fraction_device[
                                        particle_table_base +
                                        static_cast<std::size_t>(index)] +
                                    fraction *
                                        (particle_delta_fraction_device[
                                             particle_table_base +
                                             static_cast<std::size_t>(index) + 1] -
                                         particle_delta_fraction_device[
                                             particle_table_base +
                                             static_cast<std::size_t>(index)]);
                            }
                            score_letd_moments_device(
                                let_moments_device, number_of_bins,
                                static_cast<std::size_t>(bin),
                                species_let_moments_device,
                                charged_origin_category_count,
                                atomic_number >= 1 && atomic_number <= 6
                                    ? static_cast<std::size_t>(7 - atomic_number)
                                    : std::size_t{7},
                                isotope_let_moments_device,
                                light_isotope_category_count,
                                light_isotope_category(atomic_number, mass_number),
                                step_deposited_MeV *
                                    (1.0F - let_delta_fraction),
                                step_deposited_MeV, stopping_power_MeV_per_mm,
                                sycl::fmax(local_density_g_per_cm3, 1.0e-6F),
                                false);
                            score_letd_moments_device(
                                voxel_let_moments_device, number_of_voxels,
                                voxel_index,
                                nullptr, 0, 0,
                                nullptr, 0, 0,
                                step_deposited_MeV *
                                    (1.0F - let_delta_fraction),
                                step_deposited_MeV, stopping_power_MeV_per_mm,
                                sycl::fmax(local_density_g_per_cm3, 1.0e-6F),
                                false);
                        }
                        if (!enable_secondary_condensed_history &&
                            pending_dose_MeV > 0.0 &&
                            (pending_bin != bin ||
                             pending_voxel_index != voxel_index)) {
                            score_secondary_dose_device(
                                pending_dose_MeV, is_neutral_lineage, species_index,
                                neutral_origin, pending_bin, number_of_bins,
                                enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring,
                                pending_voxel_index, charged_origin_voxel_offset,
                                neutral_origin_voxel_offset, aggregate_secondary_dose_device, fragment_dose_device,
                                voxel_dose_device, charged_origin_voxel_dose_device,
                                neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_dose_atomics);
                            pending_dose_MeV = DoseAtomicT{0};
                        }
                        if (enable_secondary_condensed_history) {
                            score_secondary_uniform_z_segment_device(
                                static_cast<DoseAtomicT>(sec_local),
                                position_z_mm, direction_z, path_step_mm,
                                is_neutral_lineage, species_index,
                                neutral_origin, depth_bin_width_mm,
                                number_of_bins, enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring,
                                voxel_index % voxel_plane_size,
                                voxel_plane_size,
                                charged_origin_voxel_offset,
                                neutral_origin_voxel_offset,
                                aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
                                charged_origin_voxel_dose_device,
                                neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                        } else {
                            if (pending_dose_MeV == 0.0) {
                                pending_bin = bin;
                                pending_voxel_index = voxel_index;
                            }
                            pending_dose_MeV +=
                                static_cast<DoseAtomicT>(sec_local);
                        }
                        if (sec_delayed > 0.0F && !is_neutral_lineage) {
                            const auto z_mid =
                                position_z_mm + 0.5F * direction_z * path_step_mm;
                            const auto buildup_mfp_mm = electronic_buildup_mfp_at_energy(
                                energy_MeVu, electronic_buildup_mfp_mm);
                            score_exponential_depth(
                                sec_delayed, z_mid, depth_bin_width_mm,
                                phantom_length_mm,
                                static_cast<std::uint32_t>(number_of_bins),
                                buildup_mfp_mm, aggregate_secondary_dose_device);
                            if (fragment_dose_device != nullptr) {
                                score_exponential_depth(
                                    sec_delayed, z_mid, depth_bin_width_mm,
                                    phantom_length_mm,
                                    static_cast<std::uint32_t>(number_of_bins),
                                    buildup_mfp_mm,
                                    fragment_dose_device +
                                        species_index * number_of_bins);
                            }
                        }
                        deposited_MeV += step_deposited_MeV;
                        const auto scattering_energy_MeV =
                            energy_MeV - 0.5F * step_deposited_MeV;
                        const auto previous_energy_MeV = energy_MeV;
                        const auto previous_position_x_mm = position_x_mm;
                        const auto previous_position_y_mm = position_y_mm;
                        const auto previous_position_z_mm = position_z_mm;
                        if (secondary_fp32_energy_residual) {
                            const auto requested_energy_loss_MeV =
                                step_deposited_MeV +
                                energy_loss_residual_MeV;
                            const auto updated_energy_MeV =
                                sycl::fmax(
                                    0.0F,
                                    energy_MeV -
                                        requested_energy_loss_MeV);
                            const auto represented_energy_loss_MeV =
                                energy_MeV - updated_energy_MeV;
                            energy_MeV = updated_energy_MeV;
                            energy_loss_residual_MeV =
                                sycl::fmax(
                                    0.0F,
                                    requested_energy_loss_MeV -
                                        represented_energy_loss_MeV);
                        } else {
                            energy_MeV = sycl::fmax(
                                0.0F, energy_MeV - step_deposited_MeV);
                            energy_loss_residual_MeV = 0.0F;
                        }
                        position_x_mm += direction_x * path_step_mm;
                        position_y_mm += direction_y * path_step_mm;
                        position_z_mm += direction_z * path_step_mm;
                        if (energy_MeV == previous_energy_MeV) {
                            profile_add(
                                profile_counters_device,
                                TransportProfileSlot::secondary_energy_nonprogress_steps);
                        }
                        const auto x_nonprogress =
                            position_x_mm == previous_position_x_mm;
                        const auto y_nonprogress =
                            position_y_mm == previous_position_y_mm;
                        const auto z_nonprogress =
                            position_z_mm == previous_position_z_mm;
                        if (x_nonprogress && y_nonprogress && z_nonprogress) {
                            profile_add(
                                profile_counters_device,
                                TransportProfileSlot::secondary_position_nonprogress_steps);
                        }
                        if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_mcs);
                            auto mcs_radiation_length =
                                static_cast<float>(water_radiation_length_g_per_cm2);
                            if (enable_ct_material_mcs && in_ct) {
                                mcs_radiation_length = static_cast<float>(
                                    ct_material_radiation_length_g_per_cm2(
                                        ct_material_class(
                                            ct_material,
                                            ct_material_ids_are_schneider_sections)));
                            } else if (in_insert) {
                                mcs_radiation_length =
                                    insert_radiation_length_g_per_cm2;
                            } else if (slab_radiation_lengths_device != nullptr) {
                                mcs_radiation_length =
                                    slab_radiation_lengths_device[
                                        layer_for_material];
                            }
                            auto projected_rms_angle_rad =
                                highland_projected_rms_angle_device(
                                    scattering_energy_MeV, atomic_number, mass_number,
                                    path_step_mm, local_density_g_per_cm3,
                                    mcs_radiation_length);
                            if (enable_minibeam) {
                                const auto is_primary_continuation =
                                    species_index ==
                                    primary_charged_origin_category;
                                projected_rms_angle_rad *=
                                    minibeam_water_low_energy_mcs_scale(
                                        scattering_energy_MeV, mass_number,
                                        minibeam_water_low_energy_mcs_transition_MeVu,
                                        is_primary_continuation
                                            ? minibeam_water_primary_low_energy_mcs_scale
                                            : minibeam_water_fragment_low_energy_mcs_scale);
                            }
                            const auto scattered = scatter_direction(
                                Direction3F{direction_x, direction_y, direction_z},
                                projected_rms_angle_rad, random_seed, rng_stream,
                                steps, 10);
                            direction_x = scattered.x;
                            direction_y = scattered.y;
                            direction_z = scattered.z;
                        }

                        if (cascade_projectile_index_for_particle >= 0 &&
                            energy_MeV > energy_cutoff_MeV &&
                            cascade_xs_lut_device != nullptr) {
                                const auto projectile = cascade_projectile;
                                const auto current_energy_MeVu =
                                    energy_MeV * inverse_mass_number_for_particle;
                                // O(1) LUT on water SP energy grid (same index/fraction
                                // as the stopping-power table).
                                const auto ct_cascade_material =
                                    in_ct
                                        ? ct_material_class(
                                              ct_material,
                                              ct_material_ids_are_schneider_sections)
                                        : std::uint8_t{2};
                                const auto cascade_xs_material_index =
                                    ct_cascade_material == 1U &&
                                            use_ct_lung_cascade_xs
                                        ? std::size_t{1}
                                        : (ct_cascade_material == 3U &&
                                                   use_ct_bone_cascade_xs
                                               ? std::size_t{2}
                                               : std::size_t{0});
                                const auto lut_base =
                                    (cascade_xs_material_index *
                                         cascade_projectile_count +
                                     static_cast<std::size_t>(
                                         cascade_projectile_index_for_particle)) *
                                        table_size +
                                    static_cast<std::size_t>(index);
                                float macroscopic_cross_section_per_mm =
                                    cascade_xs_lut_device[lut_base] +
                                    fraction * (cascade_xs_lut_device[lut_base + 1U] -
                                                cascade_xs_lut_device[lut_base]);
                                // Density scale for layered phantom or CT (master).
                                if (slab_layer_count > 0 || enable_ct_grid) {
                                    macroscopic_cross_section_per_mm *=
                                        local_density_g_per_cm3;
                                }
                                // CT material mass-XS ratio (C-12 table) scales fragment
                                // cascade rate vs water×ρ — bone/lung composition effect.
                                if (cascade_xs_material_index == 0U &&
                                    use_ct_material_xs && in_ct && ct_xs_device != nullptr &&
                                    ct_ref_density_device != nullptr &&
                                    cross_section_device != nullptr) {
                                    auto xs_fi =
                                        (current_energy_MeVu - minimum_cross_section_energy) *
                                        inverse_cross_section_step;
                                    auto xs_i = static_cast<int>(sycl::floor(xs_fi));
                                    xs_i = sycl::max(
                                        0, sycl::min(xs_i,
                                                     static_cast<int>(cross_section_table_size) -
                                                         2));
                                    const auto xs_f = sycl::clamp(
                                        xs_fi - static_cast<float>(xs_i), 0.0F, 1.0F);
                                    const auto water_xs =
                                        cross_section_device[xs_i] +
                                        xs_f * (cross_section_device[xs_i + 1] -
                                                cross_section_device[xs_i]);
                                    const auto mat = static_cast<std::uint32_t>(
                                        ct_cross_section_material_index(
                                            ct_material,
                                            ct_material_ids_are_schneider_sections,
                                            use_ct_schneider_xs));
                                    const auto base =
                                        mat * cross_section_table_size +
                                        static_cast<std::size_t>(xs_i);
                                    const auto mat_xs =
                                        ct_xs_device[base] +
                                        xs_f * (ct_xs_device[base + 1U] - ct_xs_device[base]);
                                    const auto ref_rho =
                                        sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                                    const auto mass_ratio =
                                        (mat_xs / ref_rho) /
                                        sycl::fmax(water_xs, 1.0e-12F);
                                    macroscopic_cross_section_per_mm *=
                                        sycl::clamp(mass_ratio, 0.25F, 4.0F);
                                }
                                const auto interaction_probability =
                                    1.0F - sycl::exp(-macroscopic_cross_section_per_mm *
                                                     path_step_mm);
                                if (rng::uniform01(random_seed, rng_stream, steps, 8) <
                                    interaction_probability) {
                                    const auto package_uniform =
                                        rng::uniform01(random_seed, rng_stream, steps, 9);
                                    const auto selected =
                                        select_cascade_interaction_conditioned(
                                            cascade_interactions_device,
                                            projectile.interaction_offset,
                                            projectile.interaction_count,
                                            current_energy_MeVu, position_z_mm,
                                            package_uniform,
                                            cascade_depth_conditioned_layout,
                                            cascade_condition_on_reference_depth,
                                            cascade_allow_nearest_fallback);
                                    if (selected.status ==
                                        CascadeSelectionStatus::exact_cell) {
                                        ++cascade_summary.selection_exact_count;
                                    } else if (selected.status ==
                                               CascadeSelectionStatus::expanded_window) {
                                        ++cascade_summary.selection_expanded_count;
                                    } else if (selected.status ==
                                               CascadeSelectionStatus::nearest_fallback) {
                                        ++cascade_summary.selection_nearest_count;
                                    } else {
                                        ++cascade_summary.selection_no_coverage_count;
                                    }
                                    cascade_summary.selection_energy_distance_sum_MeVu +=
                                        selected.energy_distance_MeVu;
                                    cascade_summary.selection_energy_distance_max_MeVu =
                                        sycl::fmax(
                                            cascade_summary
                                                .selection_energy_distance_max_MeVu,
                                            selected.energy_distance_MeVu);
                                    if (selected.status ==
                                        CascadeSelectionStatus::no_energy_coverage) {
                                        cascade_summary.interaction_count = 1;
                                        cascade_summary.incident_energy_MeV = energy_MeV;
                                        profile_add(profile_counters_device,
                                                    TransportProfileSlot::secondary_cascade);
                                        deposit_local_heat_device(
                                            energy_MeV, position_x_mm, position_y_mm,
                                            position_z_mm, direction_x, direction_y,
                                            direction_z, depth_bin_width_mm,
                                            number_of_bins,
                                            enable_voxel_scoring, voxel_min_x_mm,
                                            voxel_min_y_mm, voxel_size_x_mm,
                                            voxel_size_y_mm, voxel_bins_x, voxel_bins_y,
                                            voxel_plane_size,
                                            aggregate_secondary_dose_device,
                                            fragment_dose_device, species_index,
                                            voxel_dose_device,
                                            enable_charged_origin_voxel_scoring,
                                            charged_origin_voxel_dose_device,
                                            charged_origin_voxel_offset);
                                        cascade_summary.residual_local_MeV += energy_MeV;
                                        deposited_MeV += energy_MeV;
                                        energy_MeV = 0.0F;
                                        break;
                                    }
                                    const auto interaction = cascade_interactions_device[
                                        projectile.interaction_offset +
                                        selected.interaction_index];
                                    auto package_product_ke_MeV = 0.0F;
                                    for (std::uint32_t product_index = 0;
                                         product_index < interaction.product_count;
                                         ++product_index) {
                                        package_product_ke_MeV +=
                                            cascade_products_device
                                                [interaction.product_offset + product_index]
                                                    .kinetic_energy_MeV;
                                    }
                                    const auto energy_scale = cascade_event_energy_scale(
                                        current_energy_MeVu,
                                        interaction.incident_energy_MeV_per_u,
                                        package_product_ke_MeV, atomic_number,
                                        mass_number);
                                    cascade_summary.interaction_count = 1;
                                    profile_add(profile_counters_device,
                                                TransportProfileSlot::secondary_cascade);
                                    cascade_summary.direct_count = interaction.product_count;
                                    cascade_summary.incident_energy_MeV = energy_MeV;
                                    std::uint32_t queueable_count = 0;
                                    auto queueable_energy = 0.0F;
                                    std::uint32_t neutral_queueable_count = 0;
                                    auto neutral_queueable_energy = 0.0F;
                                    for (std::uint32_t product_index = 0;
                                         product_index < interaction.product_count;
                                         ++product_index) {
                                        const auto product = cascade_products_device[
                                            interaction.product_offset + product_index];
                                        const auto scaled_energy =
                                            product.kinetic_energy_MeV * energy_scale;
                                        const auto is_neutral =
                                            product.pdg_id == 22 || product.pdg_id == 2112;
                                        const auto supported = product.atomic_number > 0 &&
                                                               product.mass_number > 0;
                                        if (is_neutral) {
                                            if (enable_neutral_transport) {
                                                ++neutral_queueable_count;
                                                neutral_queueable_energy += scaled_energy;
                                            } else {
                                                const auto kerma_frac =
                                                    neutral_kerma_fraction_at_energy(
                                                        energy_MeVu,
                                                        neutral_local_kerma_fraction,
                                                        neutral_kerma_high_energy_scale);
                                                const auto kerma_MeV =
                                                    scaled_energy * kerma_frac;
                                                const auto residual_MeV =
                                                    scaled_energy - kerma_MeV;
                                                cascade_summary.neutral_energy_MeV +=
                                                    residual_MeV;
                                                if (kerma_MeV > 0.0F) {
                                                    constexpr std::size_t other_species = 6;
                                                    score_exponential_depth(
                                                        kerma_MeV, position_z_mm,
                                                        depth_bin_width_mm, phantom_length_mm,
                                                        static_cast<std::uint32_t>(number_of_bins),
                                                        neutral_kerma_mean_free_path_mm,
                                                        aggregate_secondary_dose_device, true);
                                                    if (fragment_dose_device != nullptr) {
                                                        score_exponential_depth(
                                                            kerma_MeV, position_z_mm,
                                                            depth_bin_width_mm,
                                                            phantom_length_mm,
                                                            static_cast<std::uint32_t>(
                                                                number_of_bins),
                                                            neutral_kerma_mean_free_path_mm,
                                                            fragment_dose_device +
                                                                other_species * number_of_bins,
                                                            true);
                                                    }
                                                    if (enable_voxel_scoring) {
                                                        score_exponential_voxel_depth(
                                                            kerma_MeV, position_z_mm,
                                                            depth_bin_width_mm,
                                                            phantom_length_mm,
                                                            static_cast<std::uint32_t>(
                                                                number_of_bins),
                                                            neutral_kerma_mean_free_path_mm,
                                                            voxel_plane_size, voxel_index,
                                                            voxel_dose_device, true);
                                                    }
                                                }
                                            }
                                        } else if (supported) {
                                            const auto heavy_local =
                                                secondary_heavy_local_deposit_z_min > 0 &&
                                                product.atomic_number >=
                                                    secondary_heavy_local_deposit_z_min;
                                            if (heavy_local) {
                                                deposit_local_heat_device(
                                                    scaled_energy, position_x_mm,
                                                    position_y_mm, position_z_mm,
                                                    direction_x, direction_y, direction_z,
                                                    depth_bin_width_mm, number_of_bins,
                                                    enable_voxel_scoring, voxel_min_x_mm,
                                                    voxel_min_y_mm, voxel_size_x_mm,
                                                    voxel_size_y_mm, voxel_bins_x,
                                                    voxel_bins_y, voxel_plane_size,
                                                    dose_device, nullptr, 0, voxel_dose_device,
                                                    enable_charged_origin_voxel_scoring,
                                                    charged_origin_voxel_dose_device, 0);
                                            } else {
                                                ++queueable_count;
                                                queueable_energy += scaled_energy;
                                                const auto child_direction =
                                                    rotate_local_direction(
                                                        product.direction_x,
                                                        product.direction_y,
                                                        product.direction_z,
                                                        Direction3F{direction_x, direction_y,
                                                                    direction_z});
                                                score_fragment_birth_device(
                                                    birth_counts_device, birth_ke_sum_device,
                                                    birth_mevu_hist_device,
                                                    birth_depth_hist_device, number_of_bins,
                                                    depth_bin_width_mm, birth_cos_hist_device,
                                                    birth_parent_mevu_hist_device,
                                                    birth_parent_z_hist_device,
                                                    birth_parent_product_mevu_hist_device,
                                                    product.atomic_number, product.mass_number,
                                                    scaled_energy, position_z_mm,
                                                    child_direction.z,
                                                    static_cast<std::uint8_t>(
                                                        particle.generation + 1),
                                                    particle.atomic_number,
                                                    particle.mass_number, energy_MeV);
                                            }
                                        }
                                        // else: unsupported charged → residual local heat.
                                    }
                                    if (queueable_count > 0) {
                                        sycl::atomic_ref<
                                            std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            queue_counter(*secondary_queue_counter_device);
                                        const auto queue_offset =
                                            queue_counter.fetch_add(queueable_count);
                                        const auto package_fits =
                                            queue_offset <= secondary_queue_capacity_u32 &&
                                            queueable_count <=
                                                secondary_queue_capacity_u32 - queue_offset;
                                        if (package_fits) {
                                            auto output_index = queue_offset;
                                            for (std::uint32_t product_index = 0;
                                                 product_index < interaction.product_count;
                                                 ++product_index) {
                                                const auto product = cascade_products_device[
                                                    interaction.product_offset + product_index];
                                                const auto heavy_local =
                                                    secondary_heavy_local_deposit_z_min > 0 &&
                                                    product.atomic_number >=
                                                        secondary_heavy_local_deposit_z_min;
                                                if (product.atomic_number > 0 &&
                                                    product.mass_number > 0 &&
                                                    !heavy_local) {
                                                    const auto child_direction =
                                                        rotate_local_direction(
                                                            product.direction_x,
                                                            product.direction_y,
                                                            product.direction_z,
                                                            Direction3F{direction_x, direction_y,
                                                                        direction_z});
                                                    secondary_queue_device[output_index++] =
                                                        SecondaryParticle3D{
                                                            position_x_mm,
                                                            position_y_mm,
                                                            position_z_mm,
                                                            product.kinetic_energy_MeV *
                                                                energy_scale,
                                                            child_direction.x,
                                                            child_direction.y,
                                                            child_direction.z,
                                                            product.pdg_id,
                                                            product.atomic_number,
                                                            product.mass_number,
                                                            charged_dose_category(
                                                                product.atomic_number,
                                                                product.mass_number),
                                                            static_cast<std::uint8_t>(
                                                                particle.generation + 1),
                                                            charged_lineage,
                                                            rng::child_stream(
                                                                rng_stream,
                                                                rng::branch_tag(
                                                                    rng::branch_role_cascade_charged,
                                                                    product_index)),
                                                        };
                                                }
                                            }
                                            sycl::atomic_ref<
                                                std::uint64_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                filled_counter(*secondary_queue_filled_device);
                                            filled_counter.fetch_add(queueable_count);
                                            cascade_summary.queued_count = queueable_count;
                                            cascade_summary.queued_energy_MeV = queueable_energy;
                                        } else {
                                            // Overflow charged KE → residual local heat.
                                            cascade_summary.overflow_count = queueable_count;
                                            cascade_summary.overflow_energy_MeV =
                                                queueable_energy;
                                        }
                                    }
                                    if (neutral_queueable_count > 0) {
                                        sycl::atomic_ref<
                                            std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            neutral_counter(*neutral_queue_counter_device);
                                        const auto neutral_offset =
                                            neutral_counter.fetch_add(neutral_queueable_count);
                                        const auto neutral_fits =
                                            neutral_offset <= neutral_queue_capacity_u32 &&
                                            neutral_queueable_count <=
                                                neutral_queue_capacity_u32 - neutral_offset;
                                        if (neutral_fits) {
                                            auto output_index = neutral_offset;
                                            for (std::uint32_t product_index = 0;
                                                 product_index < interaction.product_count;
                                                 ++product_index) {
                                                const auto product = cascade_products_device[
                                                    interaction.product_offset + product_index];
                                                if (product.pdg_id == 22 ||
                                                    product.pdg_id == 2112) {
                                                    const auto child_direction =
                                                        rotate_local_direction(
                                                            product.direction_x,
                                                            product.direction_y,
                                                            product.direction_z,
                                                            Direction3F{direction_x, direction_y,
                                                                        direction_z});
                                                    const auto lineage =
                                                        neutral_lineage_from_pdg(product.pdg_id);
                                                    neutral_queue_device[output_index++] =
                                                        NeutralParticle3D{
                                                            position_x_mm,
                                                            position_y_mm,
                                                            position_z_mm,
                                                            product.kinetic_energy_MeV *
                                                                energy_scale,
                                                            child_direction.x,
                                                            child_direction.y,
                                                            child_direction.z,
                                                            product.pdg_id,
                                                            static_cast<std::uint8_t>(
                                                                neutral_origin_category_from_lineage(
                                                                    lineage)),
                                                            static_cast<std::uint8_t>(
                                                                particle.generation + 1),
                                                            0,
                                                            rng::child_stream(
                                                                rng_stream,
                                                                rng::branch_tag(
                                                                    rng::branch_role_cascade_neutral,
                                                                    product_index)),
                                                        };
                                                }
                                            }
                                            sycl::atomic_ref<
                                                std::uint64_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                filled_counter(*neutral_queue_filled_device);
                                            filled_counter.fetch_add(neutral_queueable_count);
                                            cascade_summary.queued_neutral_count =
                                                neutral_queueable_count;
                                            cascade_summary.queued_neutral_energy_MeV =
                                                neutral_queueable_energy;
                                        } else {
                                            cascade_summary.neutral_queue_overflow_count =
                                                neutral_queueable_count;
                                            cascade_summary.neutral_queue_overflow_energy_MeV =
                                                neutral_queueable_energy;
                                            cascade_summary.neutral_energy_MeV +=
                                                neutral_queueable_energy;
                                        }
                                    }
                                    const auto residual_cascade_MeV =
                                        interaction.local_deposit_MeV * energy_scale;
                                    deposit_local_heat_device(
                                        residual_cascade_MeV, position_x_mm, position_y_mm,
                                        position_z_mm, direction_x, direction_y, direction_z,
                                        depth_bin_width_mm, number_of_bins,
                                        enable_voxel_scoring, voxel_min_x_mm, voxel_min_y_mm,
                                        voxel_size_x_mm, voxel_size_y_mm, voxel_bins_x,
                                        voxel_bins_y, voxel_plane_size,
                                        aggregate_secondary_dose_device,
                                        fragment_dose_device, species_index,
                                        voxel_dose_device,
                                        enable_charged_origin_voxel_scoring,
                                        charged_origin_voxel_dose_device,
                                        charged_origin_voxel_offset);
                                    cascade_summary.residual_local_MeV += residual_cascade_MeV;
                                    // This energy is part of the transported particle's
                                    // deposited-energy balance even when the optional
                                    // fragment depth scorer is disabled.
                                    deposited_MeV += residual_cascade_MeV;
                                    energy_MeV = 0.0F;
                                }
                        }
                        ++steps;
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::secondary_steps);
                    }

                    if (steps == max_secondary_steps) {
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::secondary_step_cap_hits);
                    }
                    score_secondary_dose_device(
                        pending_dose_MeV, is_neutral_lineage, species_index,
                        neutral_origin, pending_bin, number_of_bins,
                        enable_voxel_scoring,
                        enable_charged_origin_voxel_scoring,
                        pending_voxel_index, charged_origin_voxel_offset,
                        neutral_origin_voxel_offset, aggregate_secondary_dose_device, fragment_dose_device,
                        voxel_dose_device, charged_origin_voxel_dose_device,
                        neutral_origin_dose_device,
                        neutral_origin_voxel_dose_device);
                    if (pending_dose_MeV > 0.0) {
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::secondary_dose_atomics);
                    }
                    if (energy_loss_residual_MeV > 0.0F) {
                        energy_MeV = sycl::fmax(
                            0.0F,
                            energy_MeV -
                                energy_loss_residual_MeV);
                        energy_loss_residual_MeV = 0.0F;
                    }
                    {
                        const auto hist_base = static_cast<std::size_t>(
                            TransportProfileSlot::secondary_track_hist_base);
                        const auto bucket = secondary_track_hist_bucket(steps);
                        profile_add(profile_counters_device,
                                    static_cast<TransportProfileSlot>(hist_base + bucket));
                    }
                    const auto stopped_inside =
                        energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                        position_z_mm < phantom_length_mm &&
                        (!enable_voxel_scoring ||
                         (position_x_mm >= voxel_min_x_mm && position_x_mm < voxel_max_x_mm &&
                          position_y_mm >= voxel_min_y_mm && position_y_mm < voxel_max_y_mm)) &&
                        !((direction_z < 0.0F && position_z_mm <= 0.0F) ||
                          (direction_z >= 0.0F && position_z_mm >= phantom_length_mm));
                    if (stopped_inside) {
                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_z_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                        auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                        auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                        if (enable_voxel_scoring) {
                            const auto x_coordinate =
                                (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                            const auto y_coordinate =
                                (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                            voxel_x = direction_x < 0.0F
                                          ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(x_coordinate));
                            voxel_y = direction_y < 0.0F
                                          ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(y_coordinate));
                            voxel_x = sycl::max(
                                0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                            voxel_y = sycl::max(
                                0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                        }
                        const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                                 static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                                 static_cast<std::size_t>(voxel_x);
                        if (enable_let_scoring &&
                            last_stopping_power_MeV_per_mm > 0.0F) {
                            // Continue HadronLET through the locally deposited
                            // sub-cutoff tail. TOPAS transports these ions down to
                            // its much lower production threshold; omitting this
                            // high-dE/dx tail biases fragment LET low.
                            score_letd_moments_device(
                                let_moments_device, number_of_bins,
                                static_cast<std::size_t>(bin),
                                species_let_moments_device,
                                charged_origin_category_count,
                                atomic_number >= 1 && atomic_number <= 6
                                    ? static_cast<std::size_t>(7 - atomic_number)
                                    : std::size_t{7},
                                isotope_let_moments_device,
                                light_isotope_category_count,
                                light_isotope_category(
                                    atomic_number, mass_number),
                                energy_MeV, energy_MeV,
                                last_stopping_power_MeV_per_mm,
                                sycl::fmax(
                                    last_density_g_per_cm3, 1.0e-6F),
                                false);
                            score_letd_moments_device(
                                voxel_let_moments_device, number_of_voxels,
                                voxel_index, nullptr, 0, 0, nullptr, 0, 0,
                                energy_MeV, energy_MeV,
                                last_stopping_power_MeV_per_mm,
                                sycl::fmax(
                                    last_density_g_per_cm3, 1.0e-6F),
                                false);
                        }
                        score_secondary_dose_device(
                            energy_MeV, is_neutral_lineage, species_index, neutral_origin,
                            bin, number_of_bins, enable_voxel_scoring,
                            enable_charged_origin_voxel_scoring, voxel_index,
                            charged_origin_voxel_offset, neutral_origin_voxel_offset,
                            aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
                            charged_origin_voxel_dose_device, neutral_origin_dose_device,
                            neutral_origin_voxel_dose_device);
                        deposited_MeV += energy_MeV;
                        energy_MeV = 0.0F;
                    }
                    escaped_MeV = energy_MeV;
                }
                secondary_deposited_device[particle_index] = deposited_MeV;
                secondary_escaped_device[particle_index] = escaped_MeV;
                secondary_steps_device[particle_index] = steps;
                if (enable_fragment_cascade) {
                    cascade_summaries_device[particle_index] = cascade_summary;
                }
                if (!persistent_workers_enabled) {
                    return;
                }
                }
                });
            secondary_kernel_event.wait_and_throw();
            const auto batch_seconds = event_duration_seconds(secondary_kernel_event);
            secondary_kernel_seconds += batch_seconds;
            if (batch_end == generation_limit ||
                batch_end - last_logged_secondary >= progress_log_every_secondary) {
                const auto pct =
                    generation_limit > generation_begin
                        ? 100.0 * static_cast<double>(batch_end - generation_begin) /
                              static_cast<double>(generation_limit - generation_begin)
                        : 100.0;
                std::cout << "  secondary: " << batch_end << '/' << generation_limit
                          << " (" << pct << "% gen" << secondary_generation_index
                          << ", last batch " << batch_seconds << " s, cum "
                          << secondary_kernel_seconds << " s)\n"
                          << std::flush;
                last_logged_secondary = batch_end;
            }
            batch_begin = batch_end;
            }  // secondary batch within generation
            transported_queue_count = generation_limit;
            if (!enable_fragment_cascade) {
                break;
            }
            generation_begin = generation_limit;
            queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
            generation_end = std::min<std::uint64_t>(generation_end,
                                                     secondary_queue_capacity);
            ++secondary_generation_index;
        }
    }

    std::uint64_t transported_neutral_count = 0;
    std::uint64_t charged_after_neutral_begin = transported_queue_count;
    if (enable_neutral_transport) {
        std::uint64_t neutral_generation_begin = 0;
        std::uint64_t neutral_generation_end = 0;
        queue.copy(neutral_queue_filled_device, &neutral_generation_end, 1).wait_and_throw();
        std::uint32_t neutral_generation = 0;
        while (neutral_generation_begin < neutral_generation_end &&
               neutral_generation < maximum_neutral_generations) {
            const auto generation_size = neutral_generation_end - neutral_generation_begin;
            const auto neutral_global_size =
                ((generation_size + local_size - 1) / local_size) * local_size;
            auto neutral_kernel_event = queue.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{neutral_global_size},
                                  sycl::range<1>{local_size}},
                [=](sycl::nd_item<1> item) {
                    const auto generation_index = item.get_global_linear_id();
                    if (generation_index >= generation_size) {
                        return;
                    }
                    const auto particle_index = neutral_generation_begin + generation_index;
                    NeutralTransportSummary summary{};
                    std::uint32_t steps = 0;
                    if (particle_index < neutral_generation_end) {
                        const auto particle = neutral_queue_device[particle_index];
                        const auto rng_stream = particle.rng_stream;
                        auto energy_MeV = particle.kinetic_energy_MeV;
                        auto position_x_mm = particle.position_x_mm;
                        auto position_y_mm = particle.position_y_mm;
                        auto position_z_mm = particle.position_z_mm;
                        auto direction_x = particle.direction_x;
                        auto direction_y = particle.direction_y;
                        auto direction_z = sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                        const auto origin_category = static_cast<std::size_t>(
                            sycl::min(static_cast<std::uint32_t>(particle.origin_category),
                                      static_cast<std::uint32_t>(
                                          neutral_origin_category_count - 1)));
                        const auto lineage = particle.pdg_id == 22 ? gamma_lineage
                                                                   : neutron_lineage;
                        constexpr std::uint32_t max_neutral_steps = 500'000U;

                        while (energy_MeV > energy_cutoff_MeV && steps < max_neutral_steps) {
                            const auto escaped_z = position_z_mm < 0.0F ||
                                                   position_z_mm >= phantom_length_mm;
                            const auto escaped_xy =
                                enable_voxel_scoring &&
                                (position_x_mm < voxel_min_x_mm ||
                                 position_x_mm >= voxel_max_x_mm ||
                                 position_y_mm < voxel_min_y_mm ||
                                 position_y_mm >= voxel_max_y_mm);
                            if (escaped_z || escaped_xy) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }

                            int projectile_index = -1;
                            for (std::size_t candidate = 0;
                                 candidate < neutral_projectile_count; ++candidate) {
                                if (neutral_projectiles_device[candidate].pdg_id ==
                                    particle.pdg_id) {
                                    projectile_index = static_cast<int>(candidate);
                                    break;
                                }
                            }
                            if (projectile_index < 0) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }
                            const auto projectile =
                                neutral_projectiles_device[projectile_index];
                            const auto upper = neutral_cross_section_lower_bound(
                                neutral_cross_sections_device,
                                projectile.cross_section_offset,
                                projectile.cross_section_count,
                                energy_MeV);
                            float macroscopic_total_per_mm = 0.0F;
                            if (upper == 0) {
                                macroscopic_total_per_mm =
                                    neutral_cross_sections_device
                                        [projectile.cross_section_offset]
                                            .macroscopic_total_per_mm;
                            } else if (upper >= projectile.cross_section_count) {
                                macroscopic_total_per_mm =
                                    neutral_cross_sections_device
                                        [projectile.cross_section_offset +
                                         projectile.cross_section_count - 1]
                                            .macroscopic_total_per_mm;
                            } else {
                                const auto lower_sample = neutral_cross_sections_device
                                    [projectile.cross_section_offset + upper - 1];
                                const auto upper_sample = neutral_cross_sections_device
                                    [projectile.cross_section_offset + upper];
                                const auto interval =
                                    upper_sample.energy_MeV - lower_sample.energy_MeV;
                                const auto xs_fraction =
                                    interval > 0.0F
                                        ? sycl::clamp((energy_MeV - lower_sample.energy_MeV) /
                                                          interval,
                                                      0.0F, 1.0F)
                                        : 0.0F;
                                macroscopic_total_per_mm =
                                    lower_sample.macroscopic_total_per_mm +
                                    xs_fraction *
                                        (upper_sample.macroscopic_total_per_mm -
                                         lower_sample.macroscopic_total_per_mm);
                            }
                            if (macroscopic_total_per_mm <= 0.0F) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }
                            const auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                                position_z_mm, slab_z_ends_device, slab_densities_device,
                                slab_layer_count, water_density_g_per_cm3);
                            if (slab_layer_count > 0) {
                                macroscopic_total_per_mm *= local_density_g_per_cm3;
                            }

                            const auto free_path_mm =
                                -sycl::log(sycl::fmax(
                                    rng::uniform01(random_seed, rng_stream, steps, 20),
                                    1.0e-12F)) /
                                macroscopic_total_per_mm;
                            if (slab_layer_count > 0) {
                                const auto to_interface = distance_to_slab_interface_mm(
                                    position_z_mm, direction_z, slab_z_ends_device,
                                    slab_layer_count, phantom_length_mm);
                                if (free_path_mm > to_interface && to_interface > 0.0F) {
                                    // Cross interface without interaction; re-sample in new layer.
                                    position_x_mm += direction_x * to_interface;
                                    position_y_mm += direction_y * to_interface;
                                    position_z_mm += direction_z * to_interface;
                                    if (sycl::fabs(direction_z) >= 1.0e-6F) {
                                        const auto layer = slab_layer_index(
                                            position_z_mm - direction_z * 1.0e-5F,
                                            slab_z_ends_device, slab_layer_count);
                                        const auto interface_z =
                                            direction_z > 0.0F
                                                ? slab_z_ends_device[layer]
                                                : (layer == 0U
                                                       ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                        position_z_mm = nudge_past_axis_boundary(
                                            interface_z, direction_z,
                                robust_boundary_nudge);
                                    }
                                    ++steps;
                                    continue;
                                }
                            }
                            position_x_mm += direction_x * free_path_mm;
                            position_y_mm += direction_y * free_path_mm;
                            position_z_mm += direction_z * free_path_mm;
                            ++steps;

                            const auto left_z = position_z_mm < 0.0F ||
                                                position_z_mm >= phantom_length_mm;
                            const auto left_xy =
                                enable_voxel_scoring &&
                                (position_x_mm < voxel_min_x_mm ||
                                 position_x_mm >= voxel_max_x_mm ||
                                 position_y_mm < voxel_min_y_mm ||
                                 position_y_mm >= voxel_max_y_mm);
                            if (left_z || left_xy) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }

                            const auto nearest = nearest_neutral_interaction(
                                neutral_interactions_device,
                                projectile.interaction_offset,
                                projectile.interaction_count,
                                energy_MeV);
                            constexpr std::uint32_t sampling_window = 8;
                            const auto window_begin =
                                nearest > sampling_window / 2
                                    ? nearest - sampling_window / 2
                                    : 0U;
                            const auto window_count = sycl::min(
                                sampling_window,
                                projectile.interaction_count - window_begin);
                            const auto selected_in_window = sycl::min(
                                static_cast<std::uint32_t>(
                                    rng::uniform01(random_seed, rng_stream, steps, 21) *
                                    window_count),
                                window_count - 1U);
                            const auto interaction = neutral_interactions_device
                                [projectile.interaction_offset + window_begin +
                                 selected_in_window];
                            const auto energy_scale =
                                interaction.incident_energy_MeV > 0.0F
                                    ? energy_MeV / interaction.incident_energy_MeV
                                    : 1.0F;
                            summary.interaction_count += 1;
                            const auto local_deposit =
                                interaction.local_deposit_MeV * energy_scale;
                            if (local_deposit > 0.0F) {
                                auto bin = direction_z < 0.0F
                                               ? static_cast<int>(sycl::ceil(
                                                     position_z_mm / depth_bin_width_mm)) -
                                                     1
                                               : static_cast<int>(sycl::floor(
                                                     position_z_mm / depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x = direction_x < 0.0F
                                                  ? static_cast<int>(sycl::ceil(x_coordinate)) -
                                                        1
                                                  : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y = direction_y < 0.0F
                                                  ? static_cast<int>(sycl::ceil(y_coordinate)) -
                                                        1
                                                  : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                score_secondary_dose_device(
                                    local_deposit, true, 0, origin_category, bin,
                                    number_of_bins, enable_voxel_scoring, false, voxel_index,
                                    0, origin_category * number_of_voxels, aggregate_secondary_dose_device, fragment_dose_device,
                                    voxel_dose_device, charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                summary.local_deposit_MeV += local_deposit;
                            }

                            std::uint32_t charged_count = 0;
                            auto charged_energy = 0.0F;
                            auto product_energy = 0.0F;
                            auto unsupported_product_energy = 0.0F;
                            for (std::uint32_t product_index = 0;
                                 product_index < interaction.product_count; ++product_index) {
                                const auto product = neutral_products_device
                                    [interaction.product_offset + product_index];
                                const auto scaled =
                                    product.kinetic_energy_MeV * energy_scale;
                                product_energy += scaled;
                                if (product.atomic_number > 0 && product.mass_number > 0 &&
                                    scaled > 0.0F) {
                                    ++charged_count;
                                    charged_energy += scaled;
                                } else if (product.pdg_id != 22 && product.pdg_id != 2112) {
                                    unsupported_product_energy += scaled;
                                }
                            }
                            summary.unsupported_product_energy_MeV +=
                                unsupported_product_energy;
                            if (charged_count > 0) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    queue_counter(*secondary_queue_counter_device);
                                const auto queue_offset = queue_counter.fetch_add(charged_count);
                                const auto package_fits =
                                    queue_offset <= secondary_queue_capacity_u32 &&
                                    charged_count <=
                                        secondary_queue_capacity_u32 - queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t product_index = 0;
                                         product_index < interaction.product_count;
                                         ++product_index) {
                                        const auto product = neutral_products_device
                                            [interaction.product_offset + product_index];
                                        const auto scaled =
                                            product.kinetic_energy_MeV * energy_scale;
                                        if (product.atomic_number > 0 &&
                                            product.mass_number > 0 && scaled > 0.0F) {
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    product.direction_x, product.direction_y,
                                                    product.direction_z,
                                                    Direction3F{direction_x, direction_y,
                                                                direction_z});
                                            secondary_queue_device[output_index++] =
                                                SecondaryParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    scaled,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    product.pdg_id,
                                                    product.atomic_number,
                                                    product.mass_number,
                                                    charged_dose_category(
                                                        product.atomic_number,
                                                        product.mass_number),
                                                    particle.generation,
                                                    lineage,
                                                    rng::child_stream(
                                                        rng_stream,
                                                        rng::branch_tag(
                                                            rng::branch_role_neutral_charged,
                                                            product_index)),
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        filled_counter(*secondary_queue_filled_device);
                                    filled_counter.fetch_add(charged_count);
                                    summary.queued_charged_count += charged_count;
                                    summary.queued_charged_energy_MeV += charged_energy;
                                } else {
                                    summary.charged_overflow_count += charged_count;
                                    summary.charged_overflow_energy_MeV += charged_energy;
                                }
                            }

                            // Nested neutral products stay residual (not re-queued in mode D).
                            for (std::uint32_t product_index = 0;
                                 product_index < interaction.product_count; ++product_index) {
                                const auto product = neutral_products_device
                                    [interaction.product_offset + product_index];
                                if (product.pdg_id == 22 || product.pdg_id == 2112) {
                                    summary.residual_energy_MeV +=
                                        product.kinetic_energy_MeV * energy_scale;
                                }
                            }

                            const auto continuation =
                                interaction.continuation_energy_MeV * energy_scale;
                            summary.package_closure_residual_MeV +=
                                energy_MeV - local_deposit - continuation - product_energy;
                            // Mode D (first_interaction): free path + one package only.
                            // Continuation kinetic energy becomes residual, not re-queued.
                            if (continuation > 0.0F && neutral_allow_continuation &&
                                continuation > energy_cutoff_MeV &&
                                particle.generation + 1 < maximum_neutral_generations) {
                                const auto child_direction = rotate_local_direction(
                                    interaction.continuation_direction_x,
                                    interaction.continuation_direction_y,
                                    interaction.continuation_direction_z,
                                    Direction3F{direction_x, direction_y, direction_z});
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    neutral_counter(*neutral_queue_counter_device);
                                const auto neutral_offset = neutral_counter.fetch_add(1);
                                if (neutral_offset < neutral_queue_capacity_u32) {
                                    neutral_queue_device[neutral_offset] = NeutralParticle3D{
                                        position_x_mm,
                                        position_y_mm,
                                        position_z_mm,
                                        continuation,
                                        child_direction.x,
                                        child_direction.y,
                                        child_direction.z,
                                        particle.pdg_id,
                                        particle.origin_category,
                                        static_cast<std::uint8_t>(particle.generation + 1),
                                        0,
                                        rng::child_stream(
                                            rng_stream,
                                            rng::branch_tag(
                                                rng::branch_role_neutral_continuation, 0U)),
                                    };
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        filled_counter(*neutral_queue_filled_device);
                                    filled_counter.fetch_add(1);
                                    summary.continuation_count += 1;
                                    summary.continuation_energy_MeV += continuation;
                                } else {
                                    summary.neutral_overflow_count += 1;
                                    summary.neutral_overflow_energy_MeV += continuation;
                                }
                            } else if (continuation > 0.0F) {
                                summary.residual_energy_MeV += continuation;
                            }
                            energy_MeV = 0.0F;
                        }
                    }
                    neutral_summaries_device[particle_index] = summary;
                    (void)steps;
                });
            neutral_kernel_event.wait_and_throw();
            neutral_kernel_seconds += event_duration_seconds(neutral_kernel_event);
            transported_neutral_count = neutral_generation_end;
            neutral_generation_begin = neutral_generation_end;
            queue.copy(neutral_queue_filled_device, &neutral_generation_end, 1)
                .wait_and_throw();
            neutral_generation_end =
                std::min<std::uint64_t>(neutral_generation_end, neutral_queue_capacity);
            ++neutral_generation;
        }

        // Transport charged products created by neutral interactions.
        if (enable_secondary_transport) {
            std::uint64_t generation_end = 0;
            queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
            generation_end =
                std::min<std::uint64_t>(generation_end, secondary_queue_capacity);
            if (generation_end > charged_after_neutral_begin) {
                const auto generation_begin = charged_after_neutral_begin;
                for (std::uint64_t batch_begin = generation_begin; batch_begin < generation_end;
                     batch_begin += static_cast<std::uint64_t>(secondary_batch)) {
                const auto batch_end = std::min<std::uint64_t>(
                    batch_begin + static_cast<std::uint64_t>(secondary_batch),
                    generation_end);
                const auto batch_size = batch_end - batch_begin;
                const auto secondary_global_size =
                    ((batch_size + local_size - 1) / local_size) * local_size;
                auto secondary_kernel_event = queue.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{secondary_global_size},
                                      sycl::range<1>{local_size}},
                    [=](sycl::nd_item<1> item) {
                        const auto generation_index = item.get_global_linear_id();
                        if (generation_index >= batch_size) {
                            return;
                        }
                        const auto particle_index = batch_begin + generation_index;
                        auto deposited_MeV = 0.0F;
                        auto escaped_MeV = 0.0F;
                        std::uint32_t steps = 0;
                        if (particle_index < generation_end) {
                            const auto particle = secondary_queue_device[particle_index];
                            auto energy_MeV = particle.kinetic_energy_MeV;
                            auto position_x_mm = particle.position_x_mm;
                            auto position_y_mm = particle.position_y_mm;
                            auto position_z_mm = particle.position_z_mm;
                            auto direction_x = particle.direction_x;
                            auto direction_y = particle.direction_y;
                            auto direction_z =
                                sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                            const auto is_neutral_lineage =
                                particle.reserved == neutron_lineage ||
                                particle.reserved == gamma_lineage;
                            const auto neutral_origin =
                                neutral_origin_category_from_lineage(particle.reserved);
                            const auto species_index = sycl::min(
                                static_cast<std::size_t>(particle.origin_category),
                                fragment_species_count - 1);
                            const auto charged_origin_voxel_offset =
                                (species_index + 1) * number_of_voxels;
                            const auto neutral_origin_voxel_offset =
                                neutral_origin * number_of_voxels;
                            const auto atomic_number =
                                static_cast<int>(particle.atomic_number);
                            const auto mass_number = static_cast<int>(particle.mass_number);
                            const auto inverse_mass_number_for_particle =
                                1.0F / static_cast<float>(mass_number);
                            constexpr std::size_t particle_mass_stride = 32;
                            const auto particle_table_species =
                                static_cast<std::size_t>(atomic_number) *
                                    particle_mass_stride +
                                static_cast<std::size_t>(mass_number);
                            const auto has_particle_specific_table =
                                use_particle_specific_stopping_power &&
                                atomic_number > 0 && atomic_number < 10 &&
                                mass_number > 0 && mass_number < 32 &&
                                particle_species_present_device[
                                    particle_table_species] != 0;
                            const auto particle_table_base =
                                particle_table_species * table_size;
                            const auto charge = static_cast<float>(atomic_number);
                            const auto charge_power =
                                sycl::pow(charge, -2.0F / 3.0F);

                            constexpr std::uint32_t max_secondary_steps = 500'000U;
                            auto last_stopping_power_MeV_per_mm = 0.0F;
                            auto last_density_g_per_cm3 =
                                static_cast<float>(water_density_g_per_cm3);
                            while (energy_MeV > secondary_local_deposit_cutoff_MeV &&
                                   steps < max_secondary_steps) {
                                const auto escaped_z =
                                    position_z_mm < 0.0F ||
                                    position_z_mm >= phantom_length_mm;
                                const auto escaped_xy =
                                    enable_voxel_scoring &&
                                    (position_x_mm < voxel_min_x_mm ||
                                     position_x_mm >= voxel_max_x_mm ||
                                     position_y_mm < voxel_min_y_mm ||
                                     position_y_mm >= voxel_max_y_mm);
                                if (escaped_z || escaped_xy) {
                                    break;
                                }
                                auto bin =
                                    direction_z < 0.0F
                                        ? static_cast<int>(sycl::ceil(position_z_mm /
                                                                      depth_bin_width_mm)) -
                                              1
                                        : static_cast<int>(sycl::floor(position_z_mm /
                                                                       depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x =
                                        direction_x < 0.0F
                                            ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y =
                                        direction_y < 0.0F
                                            ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0,
                                        sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0,
                                        sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                const auto energy_MeVu =
                                    energy_MeV * inverse_mass_number_for_particle;
                                auto floating_index =
                                    (energy_MeVu - minimum_table_energy) * inverse_table_step;
                                auto index = static_cast<int>(sycl::floor(floating_index));
                                index = sycl::max(
                                    0, sycl::min(index, static_cast<int>(table_size) - 2));
                                const auto fraction = sycl::clamp(
                                    floating_index - static_cast<float>(index), 0.0F, 1.0F);
                                const auto carbon_stopping_power_MeV_per_mm =
                                    table_device[index] +
                                    fraction * (table_device[index + 1] - table_device[index]);
                                auto charge_ratio = 1.0F;
                                if (!has_particle_specific_table) {
                                    constexpr float nucleon_mass_MeV =
                                        931.49410242F;
                                    const auto gamma =
                                        1.0F + energy_MeVu / nucleon_mass_MeV;
                                    const auto beta_squared = sycl::fmax(
                                        0.0F,
                                        1.0F - 1.0F / (gamma * gamma));
                                    const auto beta = sycl::sqrt(beta_squared);
                                    const auto effective_charge =
                                        charge *
                                        (1.0F -
                                         sycl::exp(-125.0F * beta *
                                                   charge_power));
                                    const auto reference_effective_charge =
                                        primary_charge *
                                        (1.0F -
                                         sycl::exp(-125.0F * beta *
                                                   primary_charge_power));
                                    charge_ratio =
                                        effective_charge /
                                        reference_effective_charge;
                                }
                                const auto in_insert =
                                    enable_hetero_insert &&
                                    inside_hetero_insert(
                                        position_x_mm, position_y_mm, position_z_mm,
                                        insert_x_min, insert_x_max, insert_y_min,
                                        insert_y_max, insert_z_min, insert_z_max);
                                auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                                    position_z_mm, slab_z_ends_device, slab_densities_device,
                                    slab_layer_count, water_density_g_per_cm3);
                                if (in_insert) {
                                    local_density_g_per_cm3 = insert_density_g_per_cm3;
                                }
                                const auto layer_for_material =
                                    slab_layer_count > 0
                                        ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                                           slab_layer_count)
                                        : 0U;
                                float carbon_sp_local = carbon_stopping_power_MeV_per_mm;
                                if (in_insert && use_insert_material_tables) {
                                    carbon_sp_local =
                                        insert_sp_device[static_cast<std::size_t>(index)] +
                                        fraction *
                                            (insert_sp_device[static_cast<std::size_t>(
                                                                  index) +
                                                              1] -
                                             insert_sp_device[static_cast<std::size_t>(
                                                 index)]);
                                } else if (material_table_count > 0) {
                                    const auto base =
                                        static_cast<std::size_t>(layer_for_material) *
                                        table_size;
                                    carbon_sp_local =
                                        material_sp_device[base +
                                                          static_cast<std::size_t>(index)] +
                                        fraction *
                                            (material_sp_device[base +
                                                               static_cast<std::size_t>(
                                                                   index) +
                                                               1] -
                                             material_sp_device[base +
                                                               static_cast<std::size_t>(
                                                                   index)]);
                                }
                                auto stopping_power_MeV_per_mm =
                                    carbon_sp_local * charge_ratio * charge_ratio;
                                if (has_particle_specific_table) {
                                    const auto exact_ratio =
                                        particle_sp_ratio_device[
                                            particle_table_base +
                                            static_cast<std::size_t>(index)] +
                                        fraction *
                                            (particle_sp_ratio_device[
                                                 particle_table_base +
                                                 static_cast<std::size_t>(index) + 1] -
                                             particle_sp_ratio_device[
                                                 particle_table_base +
                                                 static_cast<std::size_t>(index)]);
                                    stopping_power_MeV_per_mm =
                                        carbon_sp_local * exact_ratio;
                                }
                                last_stopping_power_MeV_per_mm =
                                    stopping_power_MeV_per_mm;
                                last_density_g_per_cm3 =
                                    local_density_g_per_cm3;
                                if ((slab_layer_count > 0 || in_insert) &&
                                    !(in_insert && use_insert_material_tables) &&
                                    material_table_count == 0) {
                                    stopping_power_MeV_per_mm *= local_density_g_per_cm3;
                                }
                                auto path_step_mm = sycl::fmin(
                                    maximum_step_mm,
                                    maximum_relative_energy_loss * energy_MeV /
                                        stopping_power_MeV_per_mm);
                                const auto absolute_direction_z = sycl::fabs(direction_z);
                                const auto boundary_z_mm =
                                    direction_z < 0.0F
                                        ? static_cast<float>(bin) * depth_bin_width_mm
                                        : static_cast<float>(bin + 1) * depth_bin_width_mm;
                                const auto distance_to_boundary_mm =
                                    direction_z < 0.0F ? position_z_mm - boundary_z_mm
                                                       : boundary_z_mm - position_z_mm;
                                if (absolute_direction_z >= 1.0e-6F) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_boundary_mm / absolute_direction_z);
                                }
                                if (slab_layer_count > 0) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_slab_interface_mm(
                                            position_z_mm, direction_z, slab_z_ends_device,
                                            slab_layer_count, phantom_length_mm));
                                }
                                if (enable_hetero_insert) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_insert_interface_mm(
                                            position_x_mm, position_y_mm, position_z_mm,
                                            direction_x, direction_y, direction_z,
                                            insert_x_min, insert_x_max, insert_y_min,
                                            insert_y_max, insert_z_min, insert_z_max,
                                            phantom_length_mm));
                                }
                                if (path_step_mm <= 1.0e-6F) {
                                    if (absolute_direction_z >= 1.0e-6F) {
                                        position_z_mm = nudge_past_axis_boundary(
                                            boundary_z_mm, direction_z,
                                robust_boundary_nudge);
                                    }
                                    if (slab_layer_count > 0 &&
                                        absolute_direction_z >= 1.0e-6F) {
                                        const auto layer = slab_layer_index(
                                            position_z_mm, slab_z_ends_device,
                                            slab_layer_count);
                                        const auto interface_z =
                                            direction_z > 0.0F
                                                ? slab_z_ends_device[layer]
                                                : (layer == 0U
                                                       ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                        position_z_mm = nudge_past_axis_boundary(
                                            interface_z, direction_z,
                                robust_boundary_nudge);
                                    }
                                    ++steps;
                                    continue;
                                }
                                const auto mean_step_loss_MeV =
                                    stopping_power_MeV_per_mm * path_step_mm;
                                auto step_deposited_MeV =
                                    sycl::fmin(mean_step_loss_MeV, energy_MeV);
                                if (enable_secondary_energy_straggling) {
                                    const auto uniform1 = sycl::fmax(
                                        rng::uniform01(
                                            random_seed, particle.rng_stream,
                                            steps, 22),
                                        1.0e-12F);
                                    const auto uniform2 = rng::uniform01(
                                        random_seed, particle.rng_stream,
                                        steps, 23);
                                    const auto extra_uniform = rng::uniform01(
                                        random_seed, particle.rng_stream,
                                        steps, 24);
                                    constexpr float two_pi =
                                        6.2831853071795864769F;
                                    const auto gaussian =
                                        sycl::sqrt(-2.0F * sycl::log(uniform1)) *
                                        sycl::cos(two_pi * uniform2);
                                    constexpr float nucleon_mass_MeV =
                                        931.49410242F;
                                    const auto gamma =
                                        1.0F + energy_MeVu / nucleon_mass_MeV;
                                    const auto beta_squared = sycl::fmax(
                                        0.0F,
                                        1.0F - 1.0F / (gamma * gamma));
                                    const auto beta =
                                        sycl::sqrt(beta_squared);
                                    const auto fragment_effective_charge =
                                        charge *
                                        (1.0F -
                                         sycl::exp(-125.0F * beta *
                                                   charge_power));
                                    const auto variance_MeV2 =
                                        condensed_total_loss_variance_MeV2_device(
                                            energy_MeVu, mass_number,
                                            fragment_effective_charge,
                                            path_step_mm,
                                            local_density_g_per_cm3);
                                    const auto local_straggling_scale =
                                        interpolate_straggling_scale(
                                            energy_MeVu,
                                            straggling_scale_energies,
                                            straggling_scale_values,
                                            straggling_scale_point_count,
                                            straggling_scale);
                                    const auto sigma_MeV =
                                        local_straggling_scale *
                                        sycl::sqrt(sycl::fmax(
                                            0.0F, variance_MeV2));
                                    step_deposited_MeV = sample_condensed_energy_loss(
                                        mean_step_loss_MeV, sigma_MeV, gaussian,
                                        extra_uniform, energy_MeV,
                                        straggling_sampler);
                                }
                                // A thin-segment sampler may return zero loss;
                                // spatial transport still advances below.
                                if (enable_let_scoring) {
                                    const auto sec_e_frac =
                                        electronic_buildup_fraction_at_energy(
                                            energy_MeVu,
                                            electronic_buildup_fraction);
                                    auto let_delta_fraction =
                                        use_let_delta_fraction_table
                                            ? let_delta_fraction_device[index] +
                                                  fraction *
                                                      (let_delta_fraction_device[index + 1] -
                                                       let_delta_fraction_device[index])
                                            : sec_e_frac;
                                    if (has_particle_specific_table) {
                                        let_delta_fraction =
                                            particle_delta_fraction_device[
                                                particle_table_base +
                                                static_cast<std::size_t>(index)] +
                                            fraction *
                                                (particle_delta_fraction_device[
                                                     particle_table_base +
                                                     static_cast<std::size_t>(index) + 1] -
                                                 particle_delta_fraction_device[
                                                     particle_table_base +
                                                     static_cast<std::size_t>(index)]);
                                    }
                                    score_letd_moments_device(
                                        let_moments_device, number_of_bins,
                                        static_cast<std::size_t>(bin),
                                        species_let_moments_device,
                                        charged_origin_category_count,
                                        atomic_number >= 1 && atomic_number <= 6
                                            ? static_cast<std::size_t>(
                                                  7 - atomic_number)
                                            : std::size_t{7},
                                        isotope_let_moments_device,
                                        light_isotope_category_count,
                                        light_isotope_category(
                                            atomic_number, mass_number),
                                        step_deposited_MeV *
                                            (1.0F - let_delta_fraction),
                                        step_deposited_MeV,
                                        stopping_power_MeV_per_mm,
                                        sycl::fmax(local_density_g_per_cm3, 1.0e-6F),
                                        false);
                                    score_letd_moments_device(
                                        voxel_let_moments_device, number_of_voxels,
                                        voxel_index,
                                        nullptr, 0, 0,
                                        nullptr, 0, 0,
                                        step_deposited_MeV *
                                            (1.0F - let_delta_fraction),
                                        step_deposited_MeV,
                                        stopping_power_MeV_per_mm,
                                        sycl::fmax(local_density_g_per_cm3, 1.0e-6F),
                                        false);
                                }
                                score_secondary_dose_device(
                                    step_deposited_MeV, is_neutral_lineage,
                                    species_index, neutral_origin, bin,
                                    number_of_bins, enable_voxel_scoring,
                                    enable_charged_origin_voxel_scoring, voxel_index,
                                    charged_origin_voxel_offset,
                                    neutral_origin_voxel_offset,
                                    aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
                                    charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                deposited_MeV += step_deposited_MeV;
                                energy_MeV -= step_deposited_MeV;
                                position_x_mm += direction_x * path_step_mm;
                                position_y_mm += direction_y * path_step_mm;
                                position_z_mm += direction_z * path_step_mm;
                                ++steps;
                            }
                            if (energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                                position_z_mm < phantom_length_mm) {
                                auto bin =
                                    direction_z < 0.0F
                                        ? static_cast<int>(sycl::ceil(position_z_mm /
                                                                      depth_bin_width_mm)) -
                                              1
                                        : static_cast<int>(sycl::floor(position_z_mm /
                                                                       depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x =
                                        direction_x < 0.0F
                                            ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y =
                                        direction_y < 0.0F
                                            ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0,
                                        sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0,
                                        sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                if (enable_let_scoring &&
                                    last_stopping_power_MeV_per_mm > 0.0F) {
                                    score_letd_moments_device(
                                        let_moments_device, number_of_bins,
                                        static_cast<std::size_t>(bin),
                                        species_let_moments_device,
                                        charged_origin_category_count,
                                        atomic_number >= 1 &&
                                                atomic_number <= 6
                                            ? static_cast<std::size_t>(
                                                  7 - atomic_number)
                                            : std::size_t{7},
                                        isotope_let_moments_device,
                                        light_isotope_category_count,
                                        light_isotope_category(
                                            atomic_number, mass_number),
                                        energy_MeV, energy_MeV,
                                        last_stopping_power_MeV_per_mm,
                                        sycl::fmax(
                                            last_density_g_per_cm3,
                                            1.0e-6F),
                                        false);
                                    score_letd_moments_device(
                                        voxel_let_moments_device,
                                        number_of_voxels, voxel_index,
                                        nullptr, 0, 0, nullptr, 0, 0,
                                        energy_MeV, energy_MeV,
                                        last_stopping_power_MeV_per_mm,
                                        sycl::fmax(
                                            last_density_g_per_cm3,
                                            1.0e-6F),
                                        false);
                                }
                                score_secondary_dose_device(
                                    energy_MeV, is_neutral_lineage, species_index,
                                    neutral_origin, bin, number_of_bins, enable_voxel_scoring,
                                    enable_charged_origin_voxel_scoring, voxel_index,
                                    charged_origin_voxel_offset, neutral_origin_voxel_offset,
                                    aggregate_secondary_dose_device, fragment_dose_device, voxel_dose_device,
                                    charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                deposited_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                            }
                            escaped_MeV = energy_MeV;
                        }
                        secondary_deposited_device[particle_index] = deposited_MeV;
                        secondary_escaped_device[particle_index] = escaped_MeV;
                        secondary_steps_device[particle_index] = steps;
                    });
                secondary_kernel_event.wait_and_throw();
                charged_after_neutral_kernel_seconds +=
                    event_duration_seconds(secondary_kernel_event);
                std::cout << "  charged-after-neutral: [" << batch_begin << ','
                          << batch_end << ")/" << generation_end << '\n'
                          << std::flush;
                }  // batch loop
                transported_queue_count = generation_end;
            }
        }
    }

    // Device scorers may be FP32; promote to double for TransportResult.
    std::vector<DoseAtomicT> dose_atomic_host(number_of_bins);
    std::vector<DoseAtomicT> voxel_dose_atomic_host;
    std::vector<DoseAtomicT> charged_origin_voxel_dose_atomic_host;
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<float> nuclear_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<SecondaryGenerationSummary> secondary_summaries_host;
    std::vector<DoseAtomicT> aggregate_secondary_dose_atomic_host;
    std::vector<DoseAtomicT> fragment_dose_atomic_host;
    std::vector<float> secondary_deposited_host;
    std::vector<float> secondary_escaped_host;
    std::vector<std::uint32_t> secondary_steps_host;
    std::vector<CascadeTransportSummary> cascade_summaries_host;
    std::vector<NeutralTransportSummary> neutral_summaries_host;
    std::vector<DoseAtomicT> neutral_origin_dose_atomic_host;
    std::vector<DoseAtomicT> neutral_origin_voxel_dose_atomic_host;
    std::vector<LetAtomicT> let_moments_host;
    std::vector<LetAtomicT> voxel_let_moments_host;
    std::vector<LetAtomicT> species_let_moments_host;
    std::vector<LetAtomicT> isotope_let_moments_host;
    std::array<std::uint64_t, minibeam_diagnostic_count_size>
        minibeam_diagnostic_counts_host{};
    std::array<double, minibeam_diagnostic_moment_size>
        minibeam_diagnostic_moments_host{};
    if (enable_minibeam) {
        if (enable_minibeam_diagnostics) {
            queue.copy(minibeam_diagnostic_counts_device,
                       minibeam_diagnostic_counts_host.data(),
                       minibeam_diagnostic_counts_host.size());
        }
        queue.copy(minibeam_diagnostic_moments_device,
                   minibeam_diagnostic_moments_host.data(),
                   minibeam_diagnostic_moments_host.size())
            .wait_and_throw();
    }
    queue.copy(dose_device, dose_atomic_host.data(), number_of_bins);
    if (enable_depth_let_scoring) {
        let_moments_host.resize(4 * number_of_bins);
        queue.copy(let_moments_device, let_moments_host.data(),
                   let_moments_host.size())
            .wait_and_throw();
    }
    if (enable_voxel_let_scoring) {
        voxel_let_moments_host.resize(4 * number_of_voxels);
        queue.copy(voxel_let_moments_device,
                   voxel_let_moments_host.data(),
                   voxel_let_moments_host.size())
            .wait_and_throw();
    }
    if (enable_species_let_scoring) {
        species_let_moments_host.resize(
            2 * charged_origin_category_count * number_of_bins);
        queue.copy(species_let_moments_device,
                   species_let_moments_host.data(),
                   species_let_moments_host.size())
            .wait_and_throw();
    }
    if (enable_light_isotope_let_scoring) {
        isotope_let_moments_host.resize(
            2 * light_isotope_category_count * number_of_bins);
        queue.copy(isotope_let_moments_device,
                   isotope_let_moments_host.data(),
                   isotope_let_moments_host.size())
            .wait_and_throw();
    }
    std::vector<std::uint64_t> birth_counts_host;
    std::vector<double> birth_ke_sum_host;
    std::vector<std::uint64_t> birth_mevu_host;
    std::vector<std::uint64_t> birth_depth_host;
    std::vector<std::uint64_t> birth_cos_host;
    std::vector<std::uint64_t> birth_parent_mevu_host;
    std::vector<std::uint64_t> birth_parent_z_host;
    std::vector<std::uint64_t> birth_parent_product_mevu_host;
    if (enable_birth_spectrum) {
        birth_counts_host.resize(birth_gen_size);
        birth_ke_sum_host.resize(birth_gen_size);
        birth_mevu_host.resize(birth_mevu_size);
        birth_depth_host.resize(birth_depth_size);
        birth_cos_host.resize(birth_cos_size);
        birth_parent_mevu_host.resize(birth_parent_mevu_size);
        birth_parent_z_host.resize(birth_parent_z_size);
        birth_parent_product_mevu_host.resize(birth_joint_size);
        queue.copy(birth_counts_device, birth_counts_host.data(), birth_gen_size);
        queue.copy(birth_ke_sum_device, birth_ke_sum_host.data(), birth_gen_size);
        queue.copy(birth_mevu_hist_device, birth_mevu_host.data(), birth_mevu_size);
        queue.copy(birth_depth_hist_device, birth_depth_host.data(), birth_depth_size);
        queue.copy(birth_cos_hist_device, birth_cos_host.data(), birth_cos_size);
        queue.copy(birth_parent_mevu_hist_device, birth_parent_mevu_host.data(),
                   birth_parent_mevu_size);
        queue.copy(birth_parent_z_hist_device, birth_parent_z_host.data(),
                   birth_parent_z_size);
        queue.copy(birth_parent_product_mevu_hist_device,
                   birth_parent_product_mevu_host.data(), birth_joint_size)
            .wait_and_throw();
    }
    if (enable_voxel_scoring) {
        voxel_dose_atomic_host.resize(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_atomic_host.data(), number_of_voxels)
            .wait_and_throw();
    }
    if (enable_charged_origin_voxel_scoring) {
        charged_origin_voxel_dose_atomic_host.resize(
            charged_origin_category_count * number_of_voxels);
        queue.copy(charged_origin_voxel_dose_device,
                   charged_origin_voxel_dose_atomic_host.data(),
                   charged_origin_voxel_dose_atomic_host.size())
            .wait_and_throw();
    }
    const auto to_double_vec = [](const std::vector<DoseAtomicT>& src) {
        return std::vector<double>(src.begin(), src.end());
    };
    std::vector<double> dose_host = to_double_vec(dose_atomic_host);
    std::vector<double> voxel_dose_host = to_double_vec(voxel_dose_atomic_host);
    std::vector<double> charged_origin_voxel_dose_host =
        to_double_vec(charged_origin_voxel_dose_atomic_host);
    std::vector<double> fragment_dose_host;
    std::vector<double> aggregate_secondary_dose_host;
    std::vector<double> neutral_origin_dose_host;
    std::vector<double> neutral_origin_voxel_dose_host;
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(nuclear_device, nuclear_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories).wait_and_throw();
    if (enable_secondary_generation) {
        secondary_summaries_host.resize(number_of_histories);
        queue.copy(secondary_summaries_device, secondary_summaries_host.data(),
                   number_of_histories)
            .wait_and_throw();
    }
    if (enable_secondary_transport) {
        const auto transported_secondary_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(transported_queue_count, secondary_queue_capacity));
        if (enable_fragment_species_scoring) {
            fragment_dose_atomic_host.resize(fragment_species_count * number_of_bins);
        }
        secondary_deposited_host.resize(transported_secondary_count);
        secondary_escaped_host.resize(transported_secondary_count);
        secondary_steps_host.resize(transported_secondary_count);
        aggregate_secondary_dose_atomic_host.resize(number_of_bins);
        queue.copy(aggregate_secondary_dose_device,
                   aggregate_secondary_dose_atomic_host.data(), number_of_bins);
        aggregate_secondary_dose_host = to_double_vec(aggregate_secondary_dose_atomic_host);
        if (enable_fragment_species_scoring) {
            queue.copy(fragment_dose_device, fragment_dose_atomic_host.data(),
                       fragment_dose_atomic_host.size());
            fragment_dose_host = to_double_vec(fragment_dose_atomic_host);
        }
        if (transported_secondary_count > 0) {
            queue.copy(secondary_deposited_device, secondary_deposited_host.data(),
                       transported_secondary_count);
            queue.copy(secondary_escaped_device, secondary_escaped_host.data(),
                       transported_secondary_count);
            queue.copy(secondary_steps_device, secondary_steps_host.data(),
                       transported_secondary_count)
                .wait_and_throw();
        } else {
            queue.wait_and_throw();
        }
        if (enable_fragment_cascade) {
            cascade_summaries_host.resize(transported_secondary_count);
            if (transported_secondary_count > 0) {
                queue.copy(cascade_summaries_device, cascade_summaries_host.data(),
                           transported_secondary_count)
                    .wait_and_throw();
            }
        }
    }
    if (enable_neutral_transport) {
        const auto transported_neutral_summary_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(transported_neutral_count, neutral_queue_capacity));
        neutral_summaries_host.resize(transported_neutral_summary_count);
        neutral_origin_dose_atomic_host.resize(neutral_origin_category_count * number_of_bins);
        if (transported_neutral_summary_count > 0) {
            queue.copy(neutral_summaries_device, neutral_summaries_host.data(),
                       transported_neutral_summary_count);
        }
        queue.copy(neutral_origin_dose_device, neutral_origin_dose_atomic_host.data(),
                   neutral_origin_dose_atomic_host.size())
            .wait_and_throw();
        neutral_origin_dose_host = to_double_vec(neutral_origin_dose_atomic_host);
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_atomic_host.resize(
                neutral_origin_category_count * number_of_voxels);
            queue.copy(neutral_origin_voxel_dose_device,
                       neutral_origin_voxel_dose_atomic_host.data(),
                       neutral_origin_voxel_dose_atomic_host.size())
                .wait_and_throw();
            neutral_origin_voxel_dose_host =
                to_double_vec(neutral_origin_voxel_dose_atomic_host);
        }
    }

#ifdef CARBON_TRANSPORT_PROFILE
    std::array<std::uint64_t, static_cast<std::size_t>(TransportProfileSlot::count)>
        profile_host{};
    auto profile_enabled = false;
    if (profile_counters_device != nullptr) {
        queue
            .copy(profile_counters_device, profile_host.data(), transport_profile_slot_count())
            .wait_and_throw();
        profile_enabled = true;
    }
#else
    const auto profile_enabled = false;
    std::array<std::uint64_t, 1> profile_host{};
#endif

    free_immutable_device(table_device);
    free_immutable_device(cross_section_device);
    free_device(let_delta_fraction_device);
    free_device(particle_sp_ratio_device);
    free_device(particle_delta_fraction_device);
    free_device(particle_species_present_device);
    free_device(dose_device);
    free_device(let_moments_device);
    free_device(voxel_let_moments_device);
    free_device(species_let_moments_device);
    free_device(isotope_let_moments_device);
    free_device(birth_counts_device);
    free_device(birth_ke_sum_device);
    free_device(birth_mevu_hist_device);
    free_device(birth_depth_hist_device);
    free_device(birth_cos_hist_device);
    free_device(birth_parent_mevu_hist_device);
    free_device(birth_parent_z_hist_device);
    free_device(birth_parent_product_mevu_hist_device);
    free_device(voxel_dose_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(deposited_device);
    free_device(escaped_device);
    free_device(nuclear_device);
    free_device(steps_device);
    free_device(profile_counters_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(slab_radiation_lengths_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_factor_lut_device);
    free_device(ct_mass_sp_za_rel_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);
    free_immutable_device(reaction_bins_device);
    free_immutable_device(reactions_device);
    free_immutable_device(reaction_secondaries_device);
    free_device(secondary_queue_device);
    free_device(secondary_bucket_scratch_device);
    free_device(secondary_bucket_index_device);
    free_device(secondary_bucket_counters_device);
    free_device(secondary_queue_counter_device);
    free_device(secondary_queue_filled_device);
    free_device(secondary_work_counter_device);
    free_device(secondary_summaries_device);
    free_device(fragment_dose_device);
    free_device(secondary_deposited_device);
    free_device(secondary_escaped_device);
    free_device(secondary_steps_device);
    free_immutable_device(cascade_projectiles_device);
    free_immutable_device(cascade_cross_sections_device);
    free_immutable_device(cascade_interactions_device);
    free_immutable_device(cascade_products_device);
    free_device(cascade_summaries_device);
    free_device(cascade_xs_lut_device);
    free_immutable_device(neutral_projectiles_device);
    free_immutable_device(neutral_cross_sections_device);
    free_immutable_device(neutral_interactions_device);
    free_immutable_device(neutral_products_device);
    free_device(neutral_queue_device);
    free_device(neutral_queue_counter_device);
    free_device(neutral_queue_filled_device);
    free_device(neutral_summaries_device);
    free_device(neutral_origin_dose_device);
    free_device(neutral_origin_voxel_dose_device);
    free_device(minibeam_copper_sp_device);
    free_device(minibeam_air_sp_device);
    free_device(minibeam_copper_xs_device);
    free_device(minibeam_copper_reaction_bins_device);
    free_device(minibeam_copper_reactions_device);
    free_device(minibeam_copper_secondaries_device);
    free_device(minibeam_copper_ion_sp_ratio_device);
    free_device(minibeam_copper_ion_sp_present_device);
    free_device(minibeam_copper_ion_xs_device);
    free_device(minibeam_copper_ion_xs_present_device);
    free_device(minibeam_copper_neutral_xs_device);
    free_device(minibeam_copper_neutral_projectiles_device);
    free_device(minibeam_copper_neutral_interactions_device);
    free_device(minibeam_diagnostic_counts_device);
    free_device(minibeam_diagnostic_moments_device);

    TransportResult result;
    result.backend = "sycl-" + device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    if (config.uses_moment_matched_straggling()) {
        result.backend += "+moment-matched-straggling";
    }
    if (config.enable_step_stable_straggling) {
        result.backend += "+step-stable-primary-straggling";
    }
    if (enable_secondary_energy_straggling) {
        result.backend += "+secondary-straggling";
    }
#ifdef CARBON_TRANSPORT_PROFILE
    if (profile_enabled) {
        result.profile.enabled = true;
        result.profile.counters = profile_host;
        result.backend += "+profile";
    }
#else
    (void)profile_enabled;
    (void)profile_host;
#endif
    const auto batch_has_energy_spread = std::any_of(
        config.primary_spot_batch.begin(), config.primary_spot_batch.end(),
        [](const auto& spot) { return spot.floats[1] > 0.0F; });
    if (config.beam_energy_spread > 0.0 || batch_has_energy_spread) {
        result.backend += "+espread";
    }
    if (!config.primary_spot_batch.empty()) {
        result.backend += "+spot-batch";
    }
    if (config.enable_tps_source) {
        result.backend += "+tps-source";
    }
    if (enable_minibeam) {
        result.backend += enable_minibeam_copper_em
                              ? "+minibeam-copper-em"
                              : "+minibeam-absorbing";
        if (enable_minibeam_copper_em &&
            minibeam_copper_enable_energy_straggling) {
            result.backend += "+copper-straggling";
        }
        if (enable_minibeam_copper_nuclear_attenuation) {
            result.backend += "+copper-nuclear-attenuation";
        }
        if (enable_minibeam_copper_reaction_products) {
            result.backend += "+copper-reaction-products";
        }
    }
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_ct_grid && enable_ct_material_mcs) {
        result.backend += "+ct-material-mcs";
    }
    if (enable_layered_phantom) {
        result.backend += use_material_tables ? "+layered-material" : "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += use_insert_material_tables ? "+hetero-insert-material"
                                                     : "+hetero-insert";
    }
    if (enable_ct_grid) {
        if (use_ct_mass_sp) {
            result.backend += use_ct_density_mass_spr
                                    ? "+ct-grid-density-spr+ct-dda"
                                    : "+ct-grid-mass-sp-lut+ct-dda";
        } else if (use_ct_material_sp) {
            result.backend += "+ct-grid-material+ct-dda";
        } else {
            result.backend += "+ct-grid+ct-dda";
        }
        if (use_ct_material_xs) {
            result.backend += "+ct-material-xs";
        }
    }
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring";
        if (!voxel_scorer_clamps_transport) {
            result.backend += "+scorer-decoupled";
        }
    }
    if (enable_charged_origin_voxel_scoring) {
        result.backend += "+charged-origin-voxel-scoring";
    }
    if (enable_let_scoring) {
        result.backend += "+letd-scoring";
    }
    if (config.enable_primary_attenuation) {
        result.backend += "+attenuation";
        if (config.enable_primary_inelastic_xs_correction) {
            result.backend += "+primary-xs-table";
        } else if (config.primary_inelastic_xs_scale != 1.0) {
            result.backend += "+primary-xs-scale";
        }
    }
    if (enable_secondary_generation) {
        result.backend += "+secondary-generation";
    }
    if (enable_secondary_transport) {
        result.backend += "+secondary-transport";
        result.backend += use_particle_specific_stopping_power
                              ? "+isotope-stopping-tables"
                              : "+c12-effective-charge-scaling";
        if (enable_secondary_energy_sorting) {
            result.backend += "+energy-sorted";
        }
        if (enable_fragment_species_scoring) {
            result.backend += "+fragment-species-scoring";
        }
    }
    if (enable_fragment_cascade) {
        result.backend += "+fragment-cascade";
        if (cascade_xs_lut_size > 0) {
            result.backend += "+cascade-xs-lut";
        }
    }
    if (enable_neutral_transport) {
        result.backend += neutral_allow_continuation ? "+neutral-transport-full"
                                                     : "+neutral-transport-first-interaction";
    }
    if (enable_minibeam && enable_minibeam_diagnostics) {
        result.minibeam.enabled = true;
        result.minibeam.incident_histories =
            minibeam_diagnostic_counts_host[0];
        result.minibeam.direct_air_slit_histories =
            minibeam_diagnostic_counts_host[1];
        result.minibeam.copper_touched_histories =
            minibeam_diagnostic_counts_host[2];
        result.minibeam.water_entrance_primary =
            minibeam_diagnostic_counts_host[3];
        result.minibeam.copper_nuclear_interactions =
            minibeam_diagnostic_counts_host[4];
        result.minibeam.copper_generated_direct_secondaries =
            minibeam_diagnostic_counts_host[5];
        result.minibeam.copper_charged_survivors =
            minibeam_diagnostic_counts_host[6];
        result.minibeam.copper_neutral_survivors =
            minibeam_diagnostic_counts_host[7];
        result.minibeam.energy_sum_MeV =
            minibeam_diagnostic_moments_host[0];
        result.minibeam.energy_squared_sum_MeV2 =
            minibeam_diagnostic_moments_host[1];
        result.minibeam.x_sum_mm =
            minibeam_diagnostic_moments_host[2];
        result.minibeam.x_squared_sum_mm2 =
            minibeam_diagnostic_moments_host[3];
        result.minibeam.y_sum_mm =
            minibeam_diagnostic_moments_host[4];
        result.minibeam.y_squared_sum_mm2 =
            minibeam_diagnostic_moments_host[5];
        result.minibeam.direction_x_sum =
            minibeam_diagnostic_moments_host[6];
        result.minibeam.direction_x_squared_sum =
            minibeam_diagnostic_moments_host[7];
        result.minibeam.direction_y_sum =
            minibeam_diagnostic_moments_host[8];
        result.minibeam.direction_y_squared_sum =
            minibeam_diagnostic_moments_host[9];
        result.minibeam.beamline_removed_energy_MeV =
            minibeam_diagnostic_moments_host[10];
        result.minibeam.copper_charged_survivor_energy_MeV =
            minibeam_diagnostic_moments_host[11];
        result.minibeam.copper_neutral_survivor_energy_MeV =
            minibeam_diagnostic_moments_host[12];
        for (std::size_t category = 0;
             category < minibeam_charged_species_count; ++category) {
            result.minibeam.copper_charged_survivors_by_species[category] =
                minibeam_diagnostic_counts_host[8 + category];
            result.minibeam
                .copper_charged_survivor_energy_by_species_MeV[category] =
                minibeam_diagnostic_moments_host[13 + category];
        }
        result.minibeam.direct_air_primary_energy_MeV =
            minibeam_diagnostic_moments_host[
                13 + minibeam_charged_species_count];
        result.minibeam.copper_touched_primary_energy_MeV =
            minibeam_diagnostic_moments_host[
                14 + minibeam_charged_species_count];
        for (std::size_t slit = 0;
             slit < MinibeamDiagnostics::slit_count; ++slit) {
            result.minibeam.water_entrance_primary_by_slit[slit] =
                minibeam_diagnostic_counts_host[
                    8 + minibeam_charged_species_count + slit];
            result.minibeam.collimator_entrance_primary_by_slit[slit] =
                minibeam_diagnostic_counts_host[
                    8 + minibeam_charged_species_count +
                    MinibeamDiagnostics::slit_count + slit];
            result.minibeam.direct_air_primary_by_slit[slit] =
                minibeam_diagnostic_counts_host[
                    8 + minibeam_charged_species_count +
                    2 * MinibeamDiagnostics::slit_count + slit];
        }
        for (std::size_t energy_bin = 0;
             energy_bin <
             MinibeamDiagnostics::touched_energy_bin_count;
             ++energy_bin) {
            result.minibeam
                .copper_touched_primary_energy_histogram[energy_bin] =
                minibeam_diagnostic_counts_host[
                    8 + minibeam_charged_species_count +
                    3 * MinibeamDiagnostics::slit_count +
                    energy_bin];
        }
        const auto fragment_histogram_offset =
            8 + minibeam_charged_species_count +
            3 * MinibeamDiagnostics::slit_count +
            MinibeamDiagnostics::touched_energy_bin_count;
        for (std::size_t energy_bin = 0;
             energy_bin < MinibeamDiagnostics::fragment_energy_bin_count;
             ++energy_bin) {
            result.minibeam.copper_deuteron_energy_histogram[energy_bin] =
                minibeam_diagnostic_counts_host[
                    fragment_histogram_offset + energy_bin];
            result.minibeam.copper_triton_energy_histogram[energy_bin] =
                minibeam_diagnostic_counts_host[
                    fragment_histogram_offset +
                    MinibeamDiagnostics::fragment_energy_bin_count +
                    energy_bin];
            result.minibeam.copper_helium_energy_histogram[energy_bin] =
                minibeam_diagnostic_counts_host[
                    fragment_histogram_offset +
                    2 * MinibeamDiagnostics::fragment_energy_bin_count +
                    energy_bin];
        }
    }
    if (enable_minibeam) {
        result.beamline_removed_energy_MeV =
            minibeam_diagnostic_moments_host[10];
    }
    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    if (enable_depth_let_scoring) {
        const auto extract_let_moment = [&](const std::size_t moment) {
            const auto begin = let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_bins);
            return std::vector<double>(
                begin, begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.primary_letd_numerator = extract_let_moment(0);
        result.primary_letd_denominator = extract_let_moment(1);
        result.all_hadron_letd_numerator = extract_let_moment(2);
        result.all_hadron_letd_denominator = extract_let_moment(3);
    }
    if (enable_species_let_scoring) {
        const auto species_moment_size =
            charged_origin_category_count * number_of_bins;
        result.charged_origin_letd_numerator.assign(
            species_let_moments_host.begin(),
            species_let_moments_host.begin() +
                static_cast<std::ptrdiff_t>(species_moment_size));
        result.charged_origin_letd_denominator.assign(
            species_let_moments_host.begin() +
                static_cast<std::ptrdiff_t>(species_moment_size),
            species_let_moments_host.end());
    }
    if (enable_light_isotope_let_scoring) {
        const auto isotope_moment_size =
            light_isotope_category_count * number_of_bins;
        result.light_isotope_letd_numerator.assign(
            isotope_let_moments_host.begin(),
            isotope_let_moments_host.begin() +
                static_cast<std::ptrdiff_t>(isotope_moment_size));
        result.light_isotope_letd_denominator.assign(
            isotope_let_moments_host.begin() +
                static_cast<std::ptrdiff_t>(isotope_moment_size),
            isotope_let_moments_host.end());
    }
    if (enable_voxel_let_scoring) {
            const auto extract_voxel_let_moment = [&](const std::size_t moment) {
                const auto begin = voxel_let_moments_host.begin() +
                                   static_cast<std::ptrdiff_t>(
                                       moment * number_of_voxels);
                return std::vector<double>(
                    begin,
                    begin + static_cast<std::ptrdiff_t>(number_of_voxels));
            };
            result.primary_voxel_letd_numerator =
                extract_voxel_let_moment(0);
            result.primary_voxel_letd_denominator =
                extract_voxel_let_moment(1);
            result.all_hadron_voxel_letd_numerator =
                extract_voxel_let_moment(2);
            result.all_hadron_voxel_letd_denominator =
                extract_voxel_let_moment(3);
    }
    if (enable_birth_spectrum) {
        result.birth_counts_by_generation = std::move(birth_counts_host);
        result.birth_ke_sum_MeV_by_generation = std::move(birth_ke_sum_host);
        result.birth_mevu_hist = std::move(birth_mevu_host);
        result.birth_depth_hist = std::move(birth_depth_host);
        result.birth_cos_hist = std::move(birth_cos_host);
        result.birth_parent_mevu_hist = std::move(birth_parent_mevu_host);
        result.birth_parent_z_hist = std::move(birth_parent_z_host);
        result.birth_parent_product_mevu_hist =
            std::move(birth_parent_product_mevu_host);
    }
    if (config.primary_spot_batch.empty()) {
        result.initial_energy_MeV =
            config.initial_total_energy_MeV() * static_cast<double>(number_of_histories);
    } else {
        result.initial_energy_MeV = 0.0;
        for (const auto& spot : config.primary_spot_batch) {
            result.initial_energy_MeV +=
                static_cast<double>(spot.floats[0]) *
                static_cast<double>(spot.history_end - spot.history_begin);
        }
    }
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);
    result.untracked_nuclear_energy_MeV =
        std::accumulate(nuclear_host.begin(), nuclear_host.end(), 0.0);
    result.nuclear_interactions = static_cast<std::uint64_t>(std::count_if(
        nuclear_host.begin(), nuclear_host.end(), [](float energy) { return energy > 0.0f; }));
    if (enable_secondary_generation) {
        result.sampled_reaction_packages = result.nuclear_interactions;
        for (const auto& summary : secondary_summaries_host) {
            result.generated_direct_secondaries += summary.direct_count;
            result.queued_secondaries += summary.queued_count;
            result.secondary_queue_overflow += summary.overflow_count;
            result.queued_secondary_energy_MeV += summary.queued_energy_MeV;
            result.secondary_queue_overflow_energy_MeV += summary.overflow_energy_MeV;
            result.untransported_neutral_energy_MeV += summary.neutral_energy_MeV;
            result.untransported_unsupported_charged_energy_MeV +=
                summary.unsupported_charged_energy_MeV;
            result.queued_neutrals += summary.queued_neutral_count;
            result.queued_neutral_energy_MeV += summary.queued_neutral_energy_MeV;
            result.neutral_queue_overflow += summary.neutral_queue_overflow_count;
            result.neutral_queue_overflow_energy_MeV +=
                summary.neutral_queue_overflow_energy_MeV;
        }
        result.generated_direct_secondary_energy_MeV =
            result.queued_secondary_energy_MeV +
            result.secondary_queue_overflow_energy_MeV +
            result.untransported_neutral_energy_MeV +
            result.untransported_unsupported_charged_energy_MeV +
            result.queued_neutral_energy_MeV;
        result.nuclear_energy_not_in_direct_secondaries_MeV =
            result.untracked_nuclear_energy_MeV -
            result.generated_direct_secondary_energy_MeV;
    }
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});
    if (enable_secondary_transport) {
        const auto aggregate_secondary_integral_MeV =
            std::accumulate(aggregate_secondary_dose_host.begin(),
                            aggregate_secondary_dose_host.end(), 0.0);
        for (std::size_t bin = 0; bin < number_of_bins; ++bin) {
            result.deposited_energy_MeV[bin] += aggregate_secondary_dose_host[bin];
        }
        if (enable_fragment_species_scoring) {
            const auto extract_species = [&](std::size_t species_index) {
                const auto begin = fragment_dose_host.begin() +
                                   static_cast<std::ptrdiff_t>(species_index * number_of_bins);
                return std::vector<double>(
                    begin, begin + static_cast<std::ptrdiff_t>(number_of_bins));
            };
            result.secondary_carbon_deposited_energy_MeV = extract_species(0);
            result.secondary_boron_deposited_energy_MeV = extract_species(1);
            result.secondary_beryllium_deposited_energy_MeV = extract_species(2);
            result.secondary_lithium_deposited_energy_MeV = extract_species(3);
            result.secondary_helium_deposited_energy_MeV = extract_species(4);
            result.secondary_proton_deposited_energy_MeV = extract_species(5);
            result.secondary_other_charged_deposited_energy_MeV = extract_species(6);
        }
        result.transported_secondaries = transported_queue_count;
        result.secondary_deposited_energy_MeV =
            std::accumulate(secondary_deposited_host.begin(),
                            secondary_deposited_host.end(), 0.0);
        result.secondary_escaped_energy_MeV =
            std::accumulate(secondary_escaped_host.begin(),
                            secondary_escaped_host.end(), 0.0);
        result.secondary_transport_steps =
            std::accumulate(secondary_steps_host.begin(), secondary_steps_host.end(),
                            std::uint64_t{0});
        // The aggregate depth scorer includes charged transport and local secondary
        // heat that is not represented in per-track deposited-energy ledgers.
        const auto secondary_local_scored_MeV = std::max(
            0.0, aggregate_secondary_integral_MeV - result.secondary_deposited_energy_MeV);
        result.total_deposited_energy_MeV +=
            result.secondary_deposited_energy_MeV + secondary_local_scored_MeV;
        // Secondary local heat is already
        // present in the aggregate dose above. Remove the same amount from
        // the unresolved nuclear reservoir so it is not counted twice by the
        // history energy-balance diagnostic.
        result.untracked_nuclear_energy_MeV -= secondary_local_scored_MeV;
        result.escaped_energy_MeV += result.secondary_escaped_energy_MeV;
        result.untracked_nuclear_energy_MeV -= result.queued_secondary_energy_MeV;
        if (enable_fragment_cascade) {
            for (std::size_t index = 0;
                 index < static_cast<std::size_t>(
                             std::min<std::uint64_t>(transported_queue_count,
                                                     cascade_summaries_host.size()));
                 ++index) {
                const auto& summary = cascade_summaries_host[index];
                result.cascade_interactions += summary.interaction_count;
                result.cascade_selection_exact += summary.selection_exact_count;
                result.cascade_selection_expanded +=
                    summary.selection_expanded_count;
                result.cascade_selection_nearest +=
                    summary.selection_nearest_count;
                result.cascade_selection_no_coverage +=
                    summary.selection_no_coverage_count;
                result.cascade_selection_energy_distance_sum_MeVu +=
                    summary.selection_energy_distance_sum_MeVu;
                result.cascade_selection_energy_distance_max_MeVu = std::max(
                    result.cascade_selection_energy_distance_max_MeVu,
                    static_cast<double>(
                        summary.selection_energy_distance_max_MeVu));
                result.generated_cascade_products += summary.direct_count;
                result.queued_cascade_secondaries += summary.queued_count;
                result.cascade_queue_overflow += summary.overflow_count;
                result.queued_cascade_energy_MeV += summary.queued_energy_MeV;
                result.cascade_nuclear_energy_MeV += summary.incident_energy_MeV;
                result.queued_neutrals += summary.queued_neutral_count;
                result.queued_neutral_energy_MeV += summary.queued_neutral_energy_MeV;
                result.neutral_queue_overflow += summary.neutral_queue_overflow_count;
                result.neutral_queue_overflow_energy_MeV +=
                    summary.neutral_queue_overflow_energy_MeV;
                result.untransported_neutral_energy_MeV += summary.neutral_energy_MeV;
                // Residual local heat is scored into fragment dose (→ total_deposited);
                // remove it from untracked so energy balance closes.
                result.untracked_nuclear_energy_MeV -=
                    static_cast<double>(summary.residual_local_MeV);
            }
            result.untracked_nuclear_energy_MeV +=
                result.cascade_nuclear_energy_MeV - result.queued_cascade_energy_MeV;
        }
        result.total_steps += result.secondary_transport_steps;
    }
    if (enable_neutral_transport) {
        const auto extract_neutral = [&](std::size_t origin_index) {
            const auto begin = neutral_origin_dose_host.begin() +
                               static_cast<std::ptrdiff_t>(origin_index * number_of_bins);
            return std::vector<double>(begin,
                                       begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.neutron_origin_deposited_energy_MeV = extract_neutral(0);
        result.gamma_origin_deposited_energy_MeV = extract_neutral(1);
        for (std::size_t bin = 0; bin < number_of_bins; ++bin) {
            result.deposited_energy_MeV[bin] +=
                result.neutron_origin_deposited_energy_MeV[bin] +
                result.gamma_origin_deposited_energy_MeV[bin];
        }
        result.neutral_origin_voxel_deposited_energy_MeV =
            std::move(neutral_origin_voxel_dose_host);
        result.transported_neutrals = transported_neutral_count;
        std::uint64_t continuation_count = 0;
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(
                         std::min<std::uint64_t>(transported_neutral_count,
                                                 neutral_summaries_host.size()));
             ++index) {
            const auto& summary = neutral_summaries_host[index];
            result.neutral_interactions += summary.interaction_count;
            result.neutral_deposited_energy_MeV += summary.local_deposit_MeV;
            result.neutral_escaped_energy_MeV += summary.escaped_energy_MeV;
            result.residual_neutral_energy_MeV += summary.residual_energy_MeV;
            result.charged_from_neutral_energy_MeV += summary.queued_charged_energy_MeV;
            result.neutral_unsupported_product_energy_MeV +=
                summary.unsupported_product_energy_MeV;
            result.neutral_package_closure_residual_MeV +=
                summary.package_closure_residual_MeV;
            result.neutral_queue_overflow += summary.neutral_overflow_count;
            result.neutral_queue_overflow_energy_MeV += summary.neutral_overflow_energy_MeV;
            result.secondary_queue_overflow += summary.charged_overflow_count;
            result.secondary_queue_overflow_energy_MeV += summary.charged_overflow_energy_MeV;
            continuation_count += summary.continuation_count;
            // Charged products are transported in the secondary pass; their deposits are
            // already included in secondary_deposited_energy_MeV.
        }
        result.queued_neutrals += continuation_count;
        // Birth neutral energy leaves untracked once. Residual continuation / nested
        // neutrals (mode D) return to the untracked nuclear residual, not phantom escape.
        result.untracked_nuclear_energy_MeV -= result.queued_neutral_energy_MeV;
        result.untracked_nuclear_energy_MeV += result.residual_neutral_energy_MeV;
        result.untracked_nuclear_energy_MeV +=
            result.neutral_unsupported_product_energy_MeV +
            result.neutral_package_closure_residual_MeV;
        result.untransported_neutral_energy_MeV += result.residual_neutral_energy_MeV;
        result.total_deposited_energy_MeV += result.neutral_deposited_energy_MeV;
        result.escaped_energy_MeV += result.neutral_escaped_energy_MeV;
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    result.neutral_kernel_seconds = neutral_kernel_seconds;
    result.charged_after_neutral_kernel_seconds =
        charged_after_neutral_kernel_seconds;
    return result;
}

}  // namespace carbon

#endif
