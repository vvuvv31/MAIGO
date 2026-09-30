// SyclTransportContext::Impl for the Unified EM transport TU.
// Method definitions are emitted only when CARBON_DEFINE_SYCL_CONTEXT=1.

// Parsed, immutable Schneider/CINEL03 host tables shared across shards in one
// process. Keyed by exact input paths + expected hashes. Device arrays are
// still uploaded per shard by upload_schneider_ct_device_context().
struct SchneiderHostCache {
    bool valid{false};
    std::string signature;
    std::shared_ptr<SchneiderRateTable> primary_rate;
    std::shared_ptr<SchneiderTargetSampler> primary_sampler;
    std::shared_ptr<InelasticPackageV3Table> primary_package;
    std::shared_ptr<Cinel03DeviceTables> primary_device_tables;
    std::shared_ptr<SecondaryRateTable> secondary_rate;
    std::shared_ptr<InelasticPackageV3Table> secondary_cinel03;
    std::shared_ptr<Cinel03DeviceTables> secondary_cinel03_device_tables;
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
        release(electron_ct.density);
        release(electron_ct.material);
        electron_ct=CachedElectronCt{};
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
    struct CachedElectronCt {
        std::string geometry_key,bank_key;
        std::shared_ptr<const CtGrid> grid;
        std::shared_ptr<MaterialElectronDeviceBank> bank;
        float* density{};
        std::uint8_t* material{};
    };
    CachedElectronCt electron_ct;
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
            cross_section_device = sycl::malloc_device<float>(
                stopping_power.energies().size(), queue);

            const auto allocation_failed = cross_section_device == nullptr;
            if (allocation_failed) {
                throw std::bad_alloc();
            }

            const auto resampled_xs =
                resample_cross_section_grid(cross_section, stopping_power.energies());
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
    float* cross_section_device{nullptr};
};
