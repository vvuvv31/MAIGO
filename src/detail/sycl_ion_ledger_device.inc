
inline void ion_counter_increment_device(
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

inline void ion_energy_add_device(
    float* ledger, const std::uint32_t slot, const float value) noexcept {
    if (ledger == nullptr || !sycl::isfinite(value)) return;
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        counter(ledger[slot]);
    counter.fetch_add(value);
}

inline void ion_species_energy_add_device(
    float* ledger, const int species, const std::uint32_t metric,
    const float value) noexcept {
    constexpr auto species_count =
        static_cast<std::uint32_t>(IonSpeciesLedgerSchema::species_count);
    constexpr auto metric_count =
        static_cast<std::uint32_t>(IonSpeciesLedgerSchema::metric_count);
    if (ledger == nullptr || species < 0 || species >= static_cast<int>(species_count) ||
        metric >= metric_count || !sycl::isfinite(value) || value <= 0.0F) return;
    ion_energy_add_device(
        ledger, static_cast<std::uint32_t>(species) * metric_count + metric, value);
}

inline void ion_record_queued_secondary_birth_device(
    float* ledger, const int z, const int a, const float kinetic_energy_MeV) noexcept {
    ion_species_energy_add_device(
        ledger, carbon::get_charged_species_idx(z, a),
        static_cast<std::uint32_t>(IonSpeciesLedgerSchema::queued_birth_kinetic),
        kinetic_energy_MeV);
}

inline void ion_species_terminal_increment_device(
    std::uint64_t* counters, const int species, const std::uint32_t reason) noexcept {
    constexpr auto species_count =
        static_cast<std::uint32_t>(IonSpeciesLedgerSchema::species_count);
    constexpr auto reason_count =
        static_cast<std::uint32_t>(IonSpeciesLedgerSchema::terminal_reason_count);
    if (counters == nullptr || species < 0 || species >= static_cast<int>(species_count) ||
        reason >= reason_count) return;
    ion_counter_increment_device(
        counters, static_cast<std::uint32_t>(species) * reason_count + reason);
}
