inline void score_be_isotope_origin_voxel_device(
    DoseAtomicT* dose, const std::size_t category, const std::size_t voxel_count,
    const std::size_t voxel, const DoseAtomicT amount) noexcept {
    if (dose == nullptr || category >= be_isotope_origin_category_count ||
        voxel >= voxel_count || amount <= DoseAtomicT{0}) return;
    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic(dose[category * voxel_count + voxel]);
    atomic.fetch_add(amount);
}

inline void cinel02_diag_add_device(
    std::uint64_t* diagnostics, const std::uint32_t slot,
    const std::uint64_t value) noexcept {
    if (diagnostics == nullptr) return;
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        counter(diagnostics[slot]);
    counter.fetch_add(value);
}

inline void cinel02_diag_increment_device(
    std::uint64_t* diagnostics, const std::uint32_t slot) noexcept {
    if (diagnostics == nullptr) return;
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        counter(diagnostics[slot]);
    counter.fetch_add(1U);
}

// Explicit in-grid/outside-grid deposited-energy split (global MeV sinks).
// Call at every site that credits deposited_device/history ledgers, with
// in_grid_flag mirroring that site's paired voxel-scorer guard exactly.
// Double precision: global accumulators reach 1e10 MeV where float32
// atomics would absorb MeV-scale adds.
inline void grid_deposit_split_device(double* in_grid_deposited,
                                      double* outside_grid_deposited,
                                      bool in_grid_flag,
                                      float value_MeV) noexcept {
    if (!sycl::isfinite(value_MeV) || value_MeV <= 0.0F) return;
    double* target = in_grid_flag ? in_grid_deposited : outside_grid_deposited;
    if (target == nullptr) return;
    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        acc(*target);
    acc.fetch_add(static_cast<double>(value_MeV));
}

inline void cinel02_energy_add_device(
    float* ledger, const std::uint32_t slot, const float value) noexcept {
    if (ledger == nullptr || !sycl::isfinite(value)) return;
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        counter(ledger[slot]);
    counter.fetch_add(value);
}

inline void cinel02_species_energy_add_device(
    float* ledger, const int species, const std::uint32_t metric,
    const float value) noexcept {
    constexpr auto species_count =
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::species_count);
    constexpr auto metric_count =
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::metric_count);
    if (ledger == nullptr || species < 0 || species >= static_cast<int>(species_count) ||
        metric >= metric_count || !sycl::isfinite(value) || value <= 0.0F) return;
    cinel02_energy_add_device(
        ledger, static_cast<std::uint32_t>(species) * metric_count + metric, value);
}

inline void cinel02_record_queued_secondary_birth_device(
    float* ledger, const int z, const int a, const float kinetic_energy_MeV) noexcept {
    cinel02_species_energy_add_device(
        ledger, carbon::get_charged_species_idx(z, a),
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::queued_birth_kinetic),
        kinetic_energy_MeV);
}

inline void cinel02_record_topas_compat_discard_device(
    std::uint64_t* counts, float* kinetic_energy,
    const int z, const int a, const float value_MeV) noexcept {
    const auto species = carbon::get_charged_species_idx(z, a);
    constexpr auto species_count = static_cast<int>(
        Cinel02SpeciesLedgerSchema::species_count);
    if (species < 0 || species >= species_count ||
        !sycl::isfinite(value_MeV)) {
        return;
    }
    // Count every generated unsupported product, including a zero-KE
    // record; the kinetic sink only accepts a positive finite value.
    cinel02_diag_increment_device(
        counts, static_cast<std::uint32_t>(species));
    if (value_MeV > 0.0F) {
        cinel02_energy_add_device(
            kinetic_energy, static_cast<std::uint32_t>(species), value_MeV);
    }
}

inline void cinel02_record_replay_delta_device(
    float* delta_sum_MeV_per_u, float* abs_delta_sum_MeV_per_u,
    std::uint64_t* positive_counts, std::uint64_t* negative_counts,
    std::uint64_t* valid_counts, const int species,
    const float delta_MeV_per_u) noexcept {
    constexpr auto species_count =
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::species_count);
    if (species < 0 || species >= static_cast<int>(species_count) ||
        !sycl::isfinite(delta_MeV_per_u)) {
        return;
    }
    if (valid_counts != nullptr) {
        cinel02_diag_increment_device(
            valid_counts, static_cast<std::uint32_t>(species));
    }
    cinel02_energy_add_device(
        delta_sum_MeV_per_u, static_cast<std::uint32_t>(species),
        delta_MeV_per_u);
    cinel02_energy_add_device(
        abs_delta_sum_MeV_per_u, static_cast<std::uint32_t>(species),
        sycl::fabs(delta_MeV_per_u));
    if (delta_MeV_per_u > 0.0F && positive_counts != nullptr) {
        cinel02_diag_increment_device(
            positive_counts, static_cast<std::uint32_t>(species));
    } else if (delta_MeV_per_u < 0.0F && negative_counts != nullptr) {
        cinel02_diag_increment_device(
            negative_counts, static_cast<std::uint32_t>(species));
    }
}

inline std::uint32_t cinel02_replay_status_cell_index_device(
    const int projectile_z, const int projectile_a, const int target_z,
    const std::uint32_t generation, const float energy_MeV_per_u) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    constexpr auto target_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::target_count);
    constexpr auto generation_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::generation_count);
    constexpr auto energy_bin_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::energy_bin_count);
    constexpr float energy_bin_width_MeV_per_u =
        Cinel02ReplayLedgerSchema::energy_bin_width_MeV_per_u;
    const auto mapped_species =
        carbon::get_charged_species_idx(projectile_z, projectile_a);
    if (mapped_species < 0 ||
        mapped_species >= static_cast<int>(Cinel02ReplayLedgerSchema::species_count)) {
        return invalid;
    }
    const auto target = target_z == 1 ? 0U : (target_z == 8 ? 1U : target_count);
    if (target >= target_count) return invalid;
    const auto generation_bin = sycl::min(generation, generation_count - 1U);
    const auto finite_energy = sycl::isfinite(energy_MeV_per_u)
                                   ? sycl::fmax(energy_MeV_per_u, 0.0F)
                                   : 0.0F;
    const auto energy_bin = sycl::min(
        static_cast<std::uint32_t>(finite_energy / energy_bin_width_MeV_per_u),
        energy_bin_count - 1U);
    return ((static_cast<std::uint32_t>(mapped_species) * target_count + target) *
                generation_count + generation_bin) *
               energy_bin_count +
           energy_bin;
}

inline void cinel02_record_replay_status_device(
    std::uint64_t* counts, float* rate_query_energy_MeV,
    float* replay_query_energy_MeV,
    float* continuous_loss_to_collision_MeV,
    float* delta_MeV_per_u, float* abs_delta_MeV_per_u,
    const int projectile_z, const int projectile_a, const int target_z,
    const std::uint32_t generation, const float rate_energy_MeV,
    const float replay_energy_MeV, const float continuous_loss_MeV,
    const float delta_energy_MeV_per_u,
    const std::uint32_t status) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    constexpr auto status_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::status_count);
    if (status >= status_count) return;
    const auto cell = cinel02_replay_status_cell_index_device(
        projectile_z, projectile_a, target_z, generation,
        replay_energy_MeV / sycl::fmax(1.0F,
            static_cast<float>(sycl::max(1, projectile_a))));
    if (cell == invalid) return;
    const auto slot = cell * status_count + status;
    cinel02_diag_increment_device(counts, slot);
    cinel02_energy_add_device(rate_query_energy_MeV, slot, rate_energy_MeV);
    cinel02_energy_add_device(replay_query_energy_MeV, slot, replay_energy_MeV);
    cinel02_energy_add_device(continuous_loss_to_collision_MeV, slot,
                              continuous_loss_MeV);
    cinel02_energy_add_device(delta_MeV_per_u, slot, delta_energy_MeV_per_u);
    cinel02_energy_add_device(abs_delta_MeV_per_u, slot,
                              sycl::fabs(delta_energy_MeV_per_u));
}

inline std::uint32_t cinel02_exposure_cell_index_device(
    const int projectile_z, const int projectile_a,
    const std::uint32_t generation, const float energy_MeV_per_u) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    constexpr auto species_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::species_count);
    constexpr auto generation_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::generation_count);
    constexpr auto energy_bin_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::energy_bin_count);
    constexpr float energy_bin_width_MeV_per_u =
        Cinel02ExposureLedgerSchema::energy_bin_width_MeV_per_u;
    const auto mapped_species =
        carbon::get_charged_species_idx(projectile_z, projectile_a);
    if (mapped_species < 0 || mapped_species >= static_cast<int>(species_count) ||
        !sycl::isfinite(energy_MeV_per_u)) {
        return invalid;
    }
    const auto generation_bin = sycl::min(generation, generation_count - 1U);
    const auto finite_energy = sycl::fmax(energy_MeV_per_u, 0.0F);
    const auto energy_bin = sycl::min(
        static_cast<std::uint32_t>(finite_energy / energy_bin_width_MeV_per_u),
        energy_bin_count - 1U);
    return (static_cast<std::uint32_t>(mapped_species) * generation_count +
            generation_bin) * energy_bin_count + energy_bin;
}

inline void cinel02_exposure_sum_add_device(
    float* sums, const std::uint32_t cell, const std::uint32_t metric,
    const float value) noexcept {
    constexpr auto metric_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::sum_metric_count);
    constexpr auto cell_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::cell_count);
    if (sums == nullptr || cell >= cell_count || metric >= metric_count ||
        !sycl::isfinite(value) || value == 0.0F) {
        return;
    }
    cinel02_energy_add_device(sums, cell * metric_count + metric, value);
}

inline void cinel02_exposure_count_increment_device(
    std::uint64_t* counts, const std::uint32_t cell, const std::uint32_t metric) noexcept {
    constexpr auto metric_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::count_metric_count);
    constexpr auto cell_count =
        static_cast<std::uint32_t>(Cinel02ExposureLedgerSchema::cell_count);
    if (counts == nullptr || cell >= cell_count || metric >= metric_count) return;
    cinel02_diag_increment_device(counts, cell * metric_count + metric);
}

inline std::uint32_t cinel02_parent_outcome_cell_index_device(
    const int projectile_z, const int projectile_a, const int target_z,
    const std::uint32_t generation, const int parent_status) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    constexpr auto target_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::target_count);
    constexpr auto generation_count =
        static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::generation_count);
    const auto mapped_species =
        carbon::get_charged_species_idx(projectile_z, projectile_a);
    if (mapped_species < 0 ||
        mapped_species >= static_cast<int>(Cinel02ReplayLedgerSchema::species_count)) {
        return invalid;
    }
    const auto target = target_z == 1 ? 0U : (target_z == 8 ? 1U : target_count);
    if (target >= target_count) return invalid;
    const auto generation_bin = sycl::min(generation, generation_count - 1U);
    const auto outcome = parent_status == 0 ? 0U : 1U;
    return ((static_cast<std::uint32_t>(mapped_species) * target_count + target) *
                generation_count + generation_bin) * 2U + outcome;
}

inline void cinel02_record_parent_outcome_device(
    std::uint64_t* counts, float* incident_energy_MeV,
    float* parent_after_energy_MeV, float* local_deposit_MeV,
    float* export_MeV, float* import_MeV,
    const int projectile_z, const int projectile_a, const int target_z,
    const std::uint32_t generation, const int parent_status,
    const float incident_energy, const float parent_after_energy,
    const float local_deposit, const float export_energy,
    const float import_energy) noexcept {
    const auto slot = cinel02_parent_outcome_cell_index_device(
        projectile_z, projectile_a, target_z, generation, parent_status);
    if (slot == std::numeric_limits<std::uint32_t>::max()) return;
    cinel02_diag_increment_device(counts, slot);
    cinel02_energy_add_device(incident_energy_MeV, slot, incident_energy);
    cinel02_energy_add_device(parent_after_energy_MeV, slot, parent_after_energy);
    cinel02_energy_add_device(local_deposit_MeV, slot, local_deposit);
    cinel02_energy_add_device(export_MeV, slot, export_energy);
    cinel02_energy_add_device(import_MeV, slot, import_energy);
}

inline std::uint32_t cinel02_transition_cell_index_device(
    const int parent_z, const int parent_a, const int child_z,
    const int child_a) noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    const auto parent = carbon::get_charged_species_idx(parent_z, parent_a);
    const auto child = carbon::get_charged_species_idx(child_z, child_a);
    if (parent < 0 || child < 0 ||
        parent >= static_cast<int>(Cinel02ReplayLedgerSchema::species_count) ||
        child >= static_cast<int>(Cinel02ReplayLedgerSchema::species_count)) {
        return invalid;
    }
    return static_cast<std::uint32_t>(parent) *
               static_cast<std::uint32_t>(Cinel02ReplayLedgerSchema::species_count) +
           static_cast<std::uint32_t>(child);
}

inline void cinel02_record_transition_device(
    std::uint64_t* counts, float* kinetic_energy_MeV,
    const int parent_z, const int parent_a, const int child_z,
    const int child_a, const float child_energy_MeV) noexcept {
    const auto slot = cinel02_transition_cell_index_device(
        parent_z, parent_a, child_z, child_a);
    if (slot == std::numeric_limits<std::uint32_t>::max()) return;
    cinel02_diag_increment_device(counts, slot);
    cinel02_energy_add_device(kinetic_energy_MeV, slot, child_energy_MeV);
}

inline void cinel02_species_terminal_increment_device(
    std::uint64_t* counters, const int species, const std::uint32_t reason) noexcept {
    constexpr auto species_count =
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::species_count);
    constexpr auto reason_count =
        static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::terminal_reason_count);
    if (counters == nullptr || species < 0 || species >= static_cast<int>(species_count) ||
        reason >= reason_count) return;
    cinel02_diag_increment_device(
        counters, static_cast<std::uint32_t>(species) * reason_count + reason);
}

inline std::uint32_t cinel02_hazard_diag_slot_device(
    const int projectile_z, const int projectile_a, const int target_z,
    const std::uint32_t generation, const float energy_MeV_per_u) noexcept {
    constexpr std::uint32_t offset = 64U;
    constexpr std::uint32_t target_count = 2U;
    constexpr std::uint32_t generation_count = 3U;
    constexpr std::uint32_t energy_bin_count = 8U;
    const auto mapped_species =
        carbon::get_charged_species_idx(projectile_z, projectile_a);
    const auto species = static_cast<std::uint32_t>(
        mapped_species >= 0 ? mapped_species : 17);
    const auto target = target_z == 1 ? 0U : 1U;
    const auto generation_bin = sycl::min(generation, generation_count - 1U);
    const auto finite_energy = sycl::isfinite(energy_MeV_per_u)
                                   ? sycl::fmax(energy_MeV_per_u, 0.0F)
                                   : 0.0F;
    const auto energy_bin = sycl::min(
        static_cast<std::uint32_t>(finite_energy / 50.0F), energy_bin_count - 1U);
    return offset +
           (((species * target_count + target) * generation_count + generation_bin) *
                energy_bin_count +
            energy_bin);
}

inline std::uint32_t cinel02_transition_diag_slot_device(
    const int parent_z, const int child_z, const std::uint32_t generation) noexcept {
    constexpr std::uint32_t offset = 928U;
    constexpr std::uint32_t generation_count = 2U;
    constexpr std::uint32_t parent_count = 6U;
    constexpr std::uint32_t child_bucket_count = 7U;
    if (parent_z < 1 || parent_z > 6) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    const auto child_bucket = static_cast<std::uint32_t>(
        child_z >= 1 && child_z <= 6 ? child_z : 0);
    const auto generation_bin = sycl::min(
        generation, generation_count - 1U);
    return offset + generation_bin * parent_count * child_bucket_count +
           static_cast<std::uint32_t>(parent_z - 1) *
               child_bucket_count + child_bucket;
}

inline std::uint32_t cinel02_parent_outcome_diag_slot_device(
    const int parent_z, const int parent_status,
    const std::uint32_t generation) noexcept {
    constexpr std::uint32_t offset = 1012U;
    constexpr std::uint32_t generation_count = 2U;
    constexpr std::uint32_t parent_count = 6U;
    if (parent_z < 1 || parent_z > 6) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    const auto killed = static_cast<std::uint32_t>(parent_status != 0);
    const auto generation_bin = sycl::min(
        generation, generation_count - 1U);
    return offset + generation_bin * parent_count * 2U +
           static_cast<std::uint32_t>(parent_z - 1) * 2U + killed;
}

inline std::uint32_t cinel02_be_channel_diag_slot_device(
    const int parent_z, const int target_z, const std::uint32_t generation,
    const float energy_MeV_per_u) noexcept {
    constexpr std::uint32_t offset = 1132U;
    if (parent_z < 4 || parent_z > 6) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    const auto generation_bin = sycl::min(generation, 1U);
    const auto target_bin = target_z == 1 ? 0U : 1U;
    const auto energy_bin = sycl::min(
        static_cast<std::uint32_t>(sycl::fmax(energy_MeV_per_u, 0.0F) / 50.0F),
        7U);
    const auto parent_channel = static_cast<std::uint32_t>(parent_z - 4);
    return offset + (((generation_bin * 2U + target_bin) * 8U + energy_bin) * 3U +
                     parent_channel);
}

inline std::uint32_t cinel02_be_incident_diag_slot_device(
    const int target_z, const std::uint32_t generation,
    const float energy_MeV_per_u) noexcept {
    constexpr std::uint32_t offset = 1324U;
    const auto generation_bin = sycl::min(generation, 1U);
    const auto target_bin = target_z == 1 ? 0U : 1U;
    const auto energy_bin = sycl::min(
        static_cast<std::uint32_t>(sycl::fmax(energy_MeV_per_u, 0.0F) / 50.0F),
        7U);
    return offset + ((generation_bin * 2U + target_bin) * 8U + energy_bin);
}

inline std::uint32_t cinel02_be_isotope_birth_diag_slot_device(
    const int parent_z, const int target_z, const std::uint32_t generation,
    const int be_mass_number) noexcept {
    constexpr std::uint32_t offset = 1388U;
    if (parent_z < 4 || parent_z > 6) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    const auto isotope = be_isotope_origin_category(4, be_mass_number);
    if (isotope >= be_isotope_origin_category_count) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    const auto generation_bin = generation == 0U ? 0U : 1U;
    const auto target_bin = target_z == 1 ? 0U : 1U;
    const auto parent_channel = static_cast<std::uint32_t>(parent_z - 4);
    return offset + (((generation_bin * 2U + target_bin) * 3U + parent_channel) * 4U +
                     static_cast<std::uint32_t>(isotope));
}

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

inline Cinel02DeviceRateLookup cinel02_material_rate_lookup_device(
    const Cinel02MaterialRateGroup* groups, const std::uint32_t group_count,
    const Cinel02RateSample* samples, const std::uint32_t sample_count,
    const int material_section, const int projectile_z, const int projectile_a,
    const int target_z, const int target_a,
    const float energy_MeV_per_u) noexcept {
    if (groups == nullptr || samples == nullptr || group_count == 0U ||
        sample_count == 0U || material_section < 0 ||
        !sycl::isfinite(energy_MeV_per_u)) return {};
    const auto before = [&](const Cinel02MaterialRateGroup& group) {
        return group.material_section < material_section ||
               (group.material_section == material_section &&
                (group.projectile_z < projectile_z ||
                 (group.projectile_z == projectile_z && group.projectile_a < projectile_a) ||
                 (group.projectile_z == projectile_z && group.projectile_a == projectile_a &&
                  group.target_z < target_z) ||
                 (group.projectile_z == projectile_z && group.projectile_a == projectile_a &&
                  group.target_z == target_z && group.target_a < target_a)));
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
    if (group.material_section != material_section ||
        group.projectile_z != projectile_z || group.projectile_a != projectile_a ||
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

inline Cinel02DeviceTargetSelection cinel02_select_material_target_device(
    const Cinel02MaterialRateGroup* groups, const std::uint32_t group_count,
    const Cinel02RateSample* samples, const std::uint32_t sample_count,
    const int material_section, const int projectile_z, const int projectile_a,
    const float energy_MeV_per_u, const float local_density_g_per_cm3,
    const float reference_density_g_per_cm3, const float u01) noexcept {
    if (!(local_density_g_per_cm3 > 0.0F) ||
        !(reference_density_g_per_cm3 > 0.0F) ||
        !sycl::isfinite(u01) || u01 < 0.0F || u01 > 1.0F) return {};
    const auto hydrogen = cinel02_material_rate_lookup_device(
        groups, group_count, samples, sample_count, material_section,
        projectile_z, projectile_a, 1, 1, energy_MeV_per_u);
    const auto oxygen = cinel02_material_rate_lookup_device(
        groups, group_count, samples, sample_count, material_section,
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
