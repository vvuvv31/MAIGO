// Shared SYCL transport helpers included by transport_sycl_legacy.cpp and
// transport_sycl.cpp. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon`
//   - for helpers below, typically inside the file-local anonymous namespace
//   - SYCL headers and carbon/* headers already included by the parent .cpp
//
// Keep body text bit-identical to the historical legacy definitions so OFF/ON
// isolation remains numerically stable.

inline std::uint32_t cascade_cross_section_lower_bound(
    const CascadeCrossSectionSample* samples,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (samples[offset + middle].energy_MeV_per_u < energy_MeV_per_u) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline int cascade_projectile_index(const CascadeProjectile* projectiles,
                                    const std::size_t count,
                                    const int atomic_number,
                                    const int mass_number) noexcept {
    std::size_t lower = 0;
    std::size_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        const auto projectile = projectiles[middle];
        const auto precedes_target =
            projectile.atomic_number < atomic_number ||
            (projectile.atomic_number == atomic_number &&
             projectile.mass_number < mass_number);
        if (precedes_target) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    if (lower < count && projectiles[lower].atomic_number == atomic_number &&
        projectiles[lower].mass_number == mass_number) {
        return static_cast<int>(lower);
    }
    return -1;
}

inline std::uint32_t cascade_interaction_lower_bound(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (interactions[offset + middle].incident_energy_MeV_per_u < energy_MeV_per_u) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t nearest_cascade_interaction(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    const auto lower =
        cascade_interaction_lower_bound(interactions, offset, count, energy_MeV_per_u);
    if (lower == 0U) {
        return 0U;
    }
    if (lower >= count) {
        return count - 1U;
    }
    const auto lower_delta = sycl::fabs(
        interactions[offset + lower - 1U].incident_energy_MeV_per_u - energy_MeV_per_u);
    const auto upper_delta = sycl::fabs(
        interactions[offset + lower].incident_energy_MeV_per_u - energy_MeV_per_u);
    return upper_delta < lower_delta ? lower : lower - 1U;
}

inline CascadeSelectionResult make_cascade_selection_result(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t interaction_index,
    const float energy_MeV_per_u,
    const CascadeSelectionStatus status) noexcept {
    const auto source_energy =
        interactions[offset + interaction_index].incident_energy_MeV_per_u;
    return CascadeSelectionResult{
        interaction_index,
        source_energy,
        sycl::fabs(source_energy - energy_MeV_per_u),
        status,
    };
}

inline CascadeSelectionResult no_cascade_energy_coverage(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    if (count != 0U) {
        const auto nearest = nearest_cascade_interaction(
            interactions, offset, count, energy_MeV_per_u);
        const auto source_energy =
            interactions[offset + nearest].incident_energy_MeV_per_u;
        return CascadeSelectionResult{
            count, source_energy, sycl::fabs(source_energy - energy_MeV_per_u),
            CascadeSelectionStatus::no_energy_coverage,
        };
    }
    return CascadeSelectionResult{
        count, 0.0F, 0.0F,
        CascadeSelectionStatus::no_energy_coverage,
    };
}

// Energy-conditioned cascade final-state selection.
// Keep a tight incident-energy band so residual-soft and fragmentation-hard
// topologies are not mixed, then over-scaled. Returns `count` when no event
// lies inside the maximum allowed band (caller should deposit residual heat
// instead of inventing a distant final state).
inline CascadeSelectionResult select_legacy_cascade_interaction_energy_conditioned(
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
    // Prefer a tight band (≤15%). Absolute floor helps sparse low-E species;
    // absolute cap avoids huge high-E windows that re-mix residual/fragmentation
    // topologies. If nothing is nearby, fall back to nearest (still scaled with
    // a narrow clamp) so sparse smoke packages keep cascading.
    constexpr float k_rel_bandwidths[4] = {0.03F, 0.05F, 0.10F, 0.15F};
    for (const auto rel : k_rel_bandwidths) {
        const auto bandwidth_MeVu = sycl::fmin(
            25.0F, sycl::fmax(1.0F, rel * sycl::fmax(energy_MeV_per_u, 1.0F)));
        const auto e_lo = energy_MeV_per_u - bandwidth_MeVu;
        const auto e_hi = energy_MeV_per_u + bandwidth_MeVu;
        auto begin =
            cascade_interaction_lower_bound(interactions, offset, count, e_lo);
        auto end =
            cascade_interaction_lower_bound(interactions, offset, count, e_hi);
        if (end <= begin) {
            continue;
        }
        constexpr std::uint32_t k_max_window = 32U;
        if (end - begin > k_max_window) {
            const auto center = cascade_interaction_lower_bound(
                interactions, offset, count, energy_MeV_per_u);
            const auto half = k_max_window / 2U;
            auto c0 = center > half ? center - half : 0U;
            auto c1 = c0 + k_max_window;
            if (c1 > count) {
                c1 = count;
                c0 = count > k_max_window ? count - k_max_window : 0U;
            }
            begin = sycl::max(begin, c0);
            end = sycl::min(end, c1);
            if (end <= begin) {
                continue;
            }
        }
        const auto window = end - begin;
        const auto pick = sycl::min(
            static_cast<std::uint32_t>(u01 * static_cast<float>(window)),
            window - 1U);
        const auto selected = begin + pick;
        const auto distance = sycl::fabs(
            interactions[offset + selected].incident_energy_MeV_per_u -
            energy_MeV_per_u);
        return make_cascade_selection_result(
            interactions, offset, selected, energy_MeV_per_u,
            distance <= 1.0e-6F ? CascadeSelectionStatus::exact_cell
                               : CascadeSelectionStatus::expanded_window);
    }
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

constexpr float cascade_condition_energy_bin_width_MeVu = 2.0F;
constexpr float cascade_condition_depth_bin_width_mm = 10.0F;

inline std::uint32_t cascade_condition_energy_bin(
    const float energy_MeV_per_u) noexcept {
    return static_cast<std::uint32_t>(
        sycl::fmax(0.0F, sycl::floor(
            energy_MeV_per_u / cascade_condition_energy_bin_width_MeVu)));
}

inline std::uint32_t cascade_condition_depth_bin(
    const float depth_mm) noexcept {
    return static_cast<std::uint32_t>(
        sycl::fmax(0.0F, sycl::floor(
            depth_mm / cascade_condition_depth_bin_width_mm)));
}

inline std::uint32_t cascade_condition_cell_lower_bound(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const std::uint32_t energy_bin,
    const std::uint32_t depth_bin) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        const auto& interaction = interactions[offset + middle];
        const auto candidate_energy_bin =
            cascade_condition_energy_bin(
                interaction.incident_energy_MeV_per_u);
        const auto candidate_depth_bin =
            cascade_condition_depth_bin(interaction.depth_mm);
        if (candidate_energy_bin < energy_bin ||
            (candidate_energy_bin == energy_bin &&
             candidate_depth_bin < depth_bin)) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline CascadeSelectionResult select_depth_conditioned_cascade_interaction(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u,
    const float depth_mm,
    const float u01,
    const bool allow_nearest_fallback) noexcept {
    if (count == 0U) {
        return no_cascade_energy_coverage(
            interactions, offset, count, energy_MeV_per_u);
    }
    const auto target_energy_bin =
        cascade_condition_energy_bin(energy_MeV_per_u);
    const auto target_depth_bin = cascade_condition_depth_bin(depth_mm);
    // Search the exact 2 MeV/u × 10 mm cell first, then expand in energy
    // before depth. This retains complete correlated final states and avoids
    // scanning the full projectile table in the device kernel.
    constexpr std::uint32_t max_energy_radius = 5U;  // ±10 MeV/u parent-energy band
    constexpr std::uint32_t max_depth_radius = 8U;    // 80 mm
    for (std::uint32_t energy_radius = 0;
         energy_radius <= max_energy_radius; ++energy_radius) {
        for (std::uint32_t depth_radius = 0;
             depth_radius <= max_depth_radius; ++depth_radius) {
            std::uint32_t total_candidates = 0;
            std::uint32_t range_begin[4]{};
            std::uint32_t range_count[4]{};
            std::uint32_t range_total = 0;
            for (std::uint32_t energy_side = 0; energy_side < 2U;
                 ++energy_side) {
                if (energy_radius == 0U && energy_side == 1U) {
                    continue;
                }
                if (energy_side == 0U &&
                    energy_radius > target_energy_bin) {
                    continue;
                }
                const auto energy_bin =
                    energy_side == 0U
                        ? target_energy_bin - energy_radius
                        : target_energy_bin + energy_radius;
                for (std::uint32_t depth_side = 0; depth_side < 2U;
                     ++depth_side) {
                    if (depth_radius == 0U && depth_side == 1U) {
                        continue;
                    }
                    if (depth_side == 0U &&
                        depth_radius > target_depth_bin) {
                        continue;
                    }
                    const auto depth_bin =
                        depth_side == 0U
                            ? target_depth_bin - depth_radius
                            : target_depth_bin + depth_radius;
                    const auto begin =
                        cascade_condition_cell_lower_bound(
                            interactions, offset, count, energy_bin,
                            depth_bin);
                    const auto end =
                        cascade_condition_cell_lower_bound(
                            interactions, offset, count, energy_bin,
                            depth_bin + 1U);
                    if (end > begin && range_total < 4U) {
                        range_begin[range_total] = begin;
                        range_count[range_total] = end - begin;
                        total_candidates += end - begin;
                        ++range_total;
                    }
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
                        energy_radius == 0U && depth_radius == 0U
                            ? CascadeSelectionStatus::exact_cell
                            : CascadeSelectionStatus::expanded_window);
                }
                pick -= range_count[range];
            }
        }
    }
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

// Scale correlated final-state KE to the true projectile energy.
// Topology-aware clamps (birth-spectrum gen1 diagnosis):
// - residual-like events (low product KE / projectile KE) must not be
//   hard-stretched into fragmentation-like spectra;
// - light parents (Z<=2) use a tighter band than heavy fragments.
inline float cascade_event_energy_scale(
    const float current_energy_MeVu,
    const float event_energy_MeVu,
    const float package_product_ke_MeV,
    const int projectile_Z,
    const int projectile_A) noexcept {
    if (!(event_energy_MeVu > 0.0F) || !(current_energy_MeVu > 0.0F)) {
        return 1.0F;
    }
    const auto scale = current_energy_MeVu / event_energy_MeVu;
    const auto projectile_ke_MeV =
        event_energy_MeVu * static_cast<float>(sycl::max(projectile_A, 1));
    const auto product_frac =
        package_product_ke_MeV / sycl::fmax(projectile_ke_MeV, 1.0e-6F);
    if (product_frac < 0.35F) {
        return sycl::clamp(scale, 0.97F, 1.03F);
    }
    if (projectile_Z > 0 && projectile_Z <= 2) {
        return sycl::clamp(scale, 0.92F, 1.08F);
    }
    return sycl::clamp(scale, 0.85F, 1.18F);
}

inline std::uint32_t neutral_cross_section_lower_bound(
    const NeutralCrossSectionSample* samples,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (samples[offset + middle].energy_MeV < energy_MeV) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t neutral_interaction_lower_bound(
    const NeutralInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (interactions[offset + middle].incident_energy_MeV < energy_MeV) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t nearest_neutral_interaction(
    const NeutralInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    const auto lower =
        neutral_interaction_lower_bound(interactions, offset, count, energy_MeV);
    if (lower == 0U) {
        return 0U;
    }
    if (lower >= count) {
        return count - 1U;
    }
    const auto lower_delta = sycl::fabs(
        interactions[offset + lower - 1U].incident_energy_MeV - energy_MeV);
    const auto upper_delta =
        sycl::fabs(interactions[offset + lower].incident_energy_MeV - energy_MeV);
    return upper_delta < lower_delta ? lower : lower - 1U;
}
