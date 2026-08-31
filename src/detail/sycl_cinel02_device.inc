inline std::uint32_t cinel02_find_event_device(
    const Cinel02EnergyNode* nodes, const std::uint32_t node_count,
    const std::uint32_t* offsets, const std::uint32_t* indices,
    const Cinel02DeviceInteraction* interactions,
    const std::uint32_t interaction_count, const int projectile_z,
    const int projectile_a, const int target_z, const int target_a,
    const float energy_MeV_per_u, const float tolerance_MeV_per_u,
    const float u01) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    if (nodes == nullptr || offsets == nullptr || indices == nullptr ||
        interactions == nullptr || node_count == 0U || interaction_count == 0U ||
        !sycl::isfinite(energy_MeV_per_u) ||
        !sycl::isfinite(tolerance_MeV_per_u) || tolerance_MeV_per_u < 0.0F ||
        !sycl::isfinite(u01) || u01 < 0.0F || u01 > 1.0F) {
        return invalid;
    }
    const auto lower_energy = energy_MeV_per_u - tolerance_MeV_per_u;
    const auto upper_energy = energy_MeV_per_u + tolerance_MeV_per_u;
    const auto before = [&](const Cinel02EnergyNode& node, const float energy) {
        return node.projectile_z < projectile_z ||
               (node.projectile_z == projectile_z && node.projectile_a < projectile_a) ||
               (node.projectile_z == projectile_z && node.projectile_a == projectile_a &&
                node.target_z < target_z) ||
               (node.projectile_z == projectile_z && node.projectile_a == projectile_a &&
                node.target_z == target_z && node.target_a < target_a) ||
               (node.projectile_z == projectile_z && node.projectile_a == projectile_a &&
                node.target_z == target_z && node.target_a == target_a &&
                node.collision_energy_MeV_per_u < energy);
    };
    auto first = 0U;
    auto last = node_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2U;
        if (before(nodes[middle], lower_energy)) first = middle + 1U;
        else last = middle;
    }
    const auto first_node = first;
    last = node_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2U;
        const auto& node = nodes[middle];
        const auto same_identity =
            node.projectile_z == projectile_z && node.projectile_a == projectile_a &&
            node.target_z == target_z && node.target_a == target_a;
        if (before(node, upper_energy) ||
            (same_identity && node.collision_energy_MeV_per_u == upper_energy)) {
            first = middle + 1U;
        } else {
            last = middle;
        }
    }
    const auto last_node = first;
    if (first_node >= last_node || last_node > node_count) return invalid;
    const auto first_event = offsets[first_node];
    const auto last_event = offsets[last_node];
    if (first_event >= last_event || last_event > interaction_count) return invalid;
    const auto count = last_event - first_event;
    const auto pick = u01 >= 1.0F
                          ? count - 1U
                          : sycl::min(static_cast<std::uint32_t>(u01 * count), count - 1U);
    const auto selected = indices[first_event + pick];
    if (selected >= interaction_count ||
        sycl::fabs(interactions[selected].incident_energy_MeV_per_u -
                   energy_MeV_per_u) > tolerance_MeV_per_u) {
        return invalid;
    }
    return selected;
}

struct Cinel02DeviceRateLookup {
    bool covered{false};
    float value_per_mm{0.0F};
};

struct Cinel02DeviceTargetSelection {
    bool covered{false};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    float total_rate_per_mm{0.0F};
};

inline Cinel02DeviceRateLookup cinel02_rate_lookup_device(
    const Cinel02RateGroup* groups, const std::uint32_t group_count,
    const Cinel02RateSample* samples, const std::uint32_t sample_count,
    const int projectile_z, const int projectile_a,
    const int target_z, const int target_a,
    const float energy_MeV_per_u) noexcept {
    if (groups == nullptr || samples == nullptr || group_count == 0U ||
        sample_count == 0U || !sycl::isfinite(energy_MeV_per_u)) return {};
    const auto before = [&](const Cinel02RateGroup& group) {
        return group.projectile_z < projectile_z ||
               (group.projectile_z == projectile_z && group.projectile_a < projectile_a) ||
               (group.projectile_z == projectile_z && group.projectile_a == projectile_a &&
                group.target_z < target_z) ||
               (group.projectile_z == projectile_z && group.projectile_a == projectile_a &&
                group.target_z == target_z && group.target_a < target_a);
    };
    auto first = 0U;
    auto last = group_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2U;
        if (before(groups[middle])) first = middle + 1U;
        else last = middle;
    }
    if (first >= group_count) return {};
    const auto& group = groups[first];
    if (group.projectile_z != projectile_z || group.projectile_a != projectile_a ||
        group.target_z != target_z || group.target_a != target_a ||
        group.sample_count < 1U || group.sample_offset > sample_count ||
        group.sample_count > sample_count - group.sample_offset) return {};
    const auto* begin = samples + group.sample_offset;
    const auto* end = begin + group.sample_count;
    if (energy_MeV_per_u < begin->energy_MeV_per_u ||
        energy_MeV_per_u > (end - 1)->energy_MeV_per_u) return {};
    if (energy_MeV_per_u == begin->energy_MeV_per_u)
        return {true, begin->macroscopic_cross_section_per_mm};
    if (energy_MeV_per_u == (end - 1)->energy_MeV_per_u)
        return {true, (end - 1)->macroscopic_cross_section_per_mm};
    auto low = 0U;
    auto high = group.sample_count;
    while (low < high) {
        const auto middle = low + (high - low) / 2U;
        if (begin[middle].energy_MeV_per_u <= energy_MeV_per_u) low = middle + 1U;
        else high = middle;
    }
    if (low == 0U || low >= group.sample_count) return {};
    const auto& lower = begin[low - 1U];
    const auto& upper = begin[low];
    const auto span = upper.energy_MeV_per_u - lower.energy_MeV_per_u;
    if (!(span > 0.0F)) return {};
    const auto fraction = (energy_MeV_per_u - lower.energy_MeV_per_u) / span;
    const auto value = lower.macroscopic_cross_section_per_mm +
                       fraction * (upper.macroscopic_cross_section_per_mm -
                                   lower.macroscopic_cross_section_per_mm);
    return sycl::isfinite(value) && value >= 0.0F
               ? Cinel02DeviceRateLookup{true, value}
               : Cinel02DeviceRateLookup{};
}

inline Cinel02DeviceTargetSelection cinel02_select_water_target_device(
    const Cinel02RateGroup* groups, const std::uint32_t group_count,
    const Cinel02RateSample* samples, const std::uint32_t sample_count,
    const int projectile_z, const int projectile_a,
    const float energy_MeV_per_u, const float local_density_g_per_cm3,
    const float reference_density_g_per_cm3, const float u01) noexcept {
    if (!(local_density_g_per_cm3 > 0.0F) ||
        !(reference_density_g_per_cm3 > 0.0F) ||
        !sycl::isfinite(u01) || u01 < 0.0F || u01 > 1.0F) return {};
    const auto hydrogen = cinel02_rate_lookup_device(
        groups, group_count, samples, sample_count,
        projectile_z, projectile_a, 1, 1, energy_MeV_per_u);
    const auto oxygen = cinel02_rate_lookup_device(
        groups, group_count, samples, sample_count,
        projectile_z, projectile_a, 8, 16, energy_MeV_per_u);
    if (!hydrogen.covered || !oxygen.covered) return {};
    const auto density_scale = local_density_g_per_cm3 / reference_density_g_per_cm3;
    const auto h_rate = hydrogen.value_per_mm * density_scale;
    const auto o_rate = oxygen.value_per_mm * density_scale;
    const auto total = h_rate + o_rate;
    if (!(total > 0.0F) || !sycl::isfinite(total)) return {};
    if (u01 < h_rate / total) return {true, 1, 1, total};
    return {true, 8, 16, total};
}
