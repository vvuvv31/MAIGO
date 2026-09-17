// Shared SyclTransportContext::Impl definition for dual SYCL TUs.
// Included by transport_sycl_legacy.cpp and transport_sycl.cpp.
// Method definitions are emitted only when CARBON_DEFINE_SYCL_CONTEXT=1.

// Parsed, immutable Schneider/CINEL03 host tables shared across shards in one
// process. Keyed by exact input paths + expected hashes. Device arrays are
// still uploaded per shard by upload_schneider_ct_device_context().
struct SchneiderHostCache {
    bool valid{false};
    std::string signature;
    std::shared_ptr<SchneiderRateTable> primary_rate;
    std::shared_ptr<SchneiderTargetSampler> primary_sampler;
    std::shared_ptr<InelasticPackageV3Table> c12_package;
    std::shared_ptr<Cinel03DeviceTables> c12_device_tables;
    std::shared_ptr<SecondaryRateTable> secondary_rate;
    std::shared_ptr<InelasticPackageV3Table> secondary_cinel03;
    std::shared_ptr<Cinel03DeviceTables> secondary_cinel03_device_tables;
    std::shared_ptr<SchneiderStoppingTable> schneider_stopping;
    std::shared_ptr<std::vector<float>> schneider_stopping_flat;
    std::shared_ptr<SchneiderIonStoppingTable> schneider_ion_stopping;
    std::shared_ptr<std::vector<float>> schneider_ion_stopping_flat;
    std::uint64_t hits{0};
    std::uint64_t misses{0};
};

struct SyclTransportContext::Impl {
    explicit Impl(const std::string& requested_device_name)
        : device_name(requested_device_name), queue(make_sycl_queue(requested_device_name)) {}

    ~Impl() { clear(); }

    void clear() noexcept {
        const auto release = [this](auto*& pointer) {
            if (pointer != nullptr) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };
        release(table_device);
        release(energy_grid_device);
        release(cumulative_range_device);
        release(cross_section_device);
        initialized = false;
    }

    // Persistent, validated, immutable host parse cache shared across shards in
    // one process. Keyed by exact input path + expected hash (never by filename
    // alone). Device arrays stay per-shard and are uploaded by the caller.
    struct CachedUnifiedEm {
        bool valid{false};
        std::string signature;
        std::shared_ptr<const UnifiedEmPackage> package;
        std::shared_ptr<const std::vector<float>> delta_means;
        std::size_t package_node_count{0};
    };
    CachedUnifiedEm unified_em_cache;
    SchneiderHostCache schneider_host_cache;
    std::uint64_t physics_cache_hits{0};
    std::uint64_t physics_cache_misses{0};

    void clear_physics_cache() noexcept {
        unified_em_cache = CachedUnifiedEm{};
        schneider_host_cache = SchneiderHostCache{};
        physics_cache_hits = 0;
        physics_cache_misses = 0;
    }

    // `signature` identifies the physics inputs (paths/hashes). When non-empty
    // and unchanged the cached device tables are reused even if the caller
    // passes a freshly loaded host table object; when it changes the tables are
    // rebuilt. An empty signature keeps the historical address-based check.
    void ensure_initialized(const StoppingPowerTable& stopping_power,
                            const CrossSectionTable& cross_section,
                            const std::string& signature = std::string{}) {
        if (initialized) {
            if (!signature.empty()) {
                if (physics_signature == signature) return;
            } else if (stopping_power_host == &stopping_power &&
                       cross_section_host == &cross_section) {
                return;
            }
            clear();
        }

        physics_signature = signature;
        stopping_power_host = &stopping_power;
        cross_section_host = &cross_section;

        try {
            table_device = sycl::malloc_device<float>(stopping_power.values().size(), queue);
            energy_grid_device = sycl::malloc_device<float>(
                stopping_power.energies().size(), queue);
            cumulative_range_device = sycl::malloc_device<float>(
                stopping_power.cumulative_ranges_mm().size(), queue);
            cross_section_device = sycl::malloc_device<float>(
                stopping_power.energies().size(), queue);

            const auto allocation_failed =
                table_device == nullptr || energy_grid_device == nullptr ||
                cumulative_range_device == nullptr ||
                cross_section_device == nullptr;
            if (allocation_failed) {
                throw std::bad_alloc();
            }

            std::vector<float> table_host(stopping_power.values().size());
            std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                           table_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            const auto resampled_xs =
                resample_cross_section_grid(cross_section, stopping_power.energies());
            std::vector<float> energy_grid_host(stopping_power.energies().size());
            std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                           energy_grid_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            std::vector<float> cumulative_range_host(stopping_power.cumulative_ranges_mm().size());
            std::transform(stopping_power.cumulative_ranges_mm().begin(),
                           stopping_power.cumulative_ranges_mm().end(),
                           cumulative_range_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(table_host.data(), table_device, table_host.size());
            queue.copy(energy_grid_host.data(), energy_grid_device, energy_grid_host.size());
            queue.copy(cumulative_range_host.data(), cumulative_range_device,
                       cumulative_range_host.size());
            queue.copy(resampled_xs.macroscopic_per_mm.data(), cross_section_device,
                       resampled_xs.macroscopic_per_mm.size());
            queue.wait_and_throw();
            initialized = true;
        } catch (...) {
            clear();
            throw;
        }
    }

    std::string device_name;
    sycl::queue queue;
    bool initialized{false};
    std::string physics_signature;
    const StoppingPowerTable* stopping_power_host{nullptr};
    const CrossSectionTable* cross_section_host{nullptr};
    float* table_device{nullptr};
    float* energy_grid_device{nullptr};
    float* cumulative_range_device{nullptr};
    float* cross_section_device{nullptr};
};
