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

// One validated immutable package on one session queue. Never stores dose,
// particles, RNG state, failures or audit counters. Only completed copies enter
// the map; destruction and input-key changes release every owned allocation.
struct ResidentPhysicsTables {
    struct Block { void* pointer; std::size_t count, bytes; std::type_index type; };
    sycl::queue queue;
    std::string signature;
    std::map<std::string, Block> blocks;
    std::size_t bytes{};
    ResidentPhysicsTables(sycl::queue q, std::string key)
        : queue(std::move(q)), signature(std::move(key)) {}
    ResidentPhysicsTables(const ResidentPhysicsTables&) = delete;
    ~ResidentPhysicsTables() {
        for (const auto& [name, block] : blocks) {
            try { sycl::free(block.pointer, queue); } catch (...) {}
        }
    }
    bool contains(const std::string& name) const { return blocks.contains(name); }
    template<class T> T* get(const std::string& name) const {
        const auto& block=blocks.at(name);
        if (block.type != std::type_index(typeid(T)))
            throw std::logic_error("Resident physics table type mismatch");
        return static_cast<T*>(block.pointer);
    }
    template<class T> T* upload(const std::string& name, const std::vector<T>& values,
                                detail::DeviceMemoryTracker& tracker) {
        if (auto it=blocks.find(name); it!=blocks.end()) {
            if (it->second.count != values.size())
                throw std::logic_error("Resident physics table size mismatch");
            return get<T>(name);
        }
        if (values.empty()) return nullptr;
        if (values.size() > std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw std::bad_alloc();
        const auto nbytes=values.size()*sizeof(T);
        tracker.account_retained(nbytes);
        auto* pointer=sycl::malloc_device<T>(values.size(),queue);
        if (!pointer) throw std::bad_alloc();
        try {
            queue.copy(values.data(),pointer,values.size()).wait_and_throw();
            blocks.emplace(name,Block{pointer,values.size(),nbytes,std::type_index(typeid(T))});
        } catch (...) { sycl::free(pointer,queue); throw; }
        bytes+=nbytes;
        return pointer;
    }
};

struct SyclTransportContext::Impl {
    explicit Impl(const std::string& requested_device_name)
        : device_name(requested_device_name), queue(make_sycl_queue(requested_device_name)) {}

    ~Impl() { clear(); }

    void clear() noexcept {
        resident_em.reset();resident_urban.reset();resident_air.reset();
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
    // alone). Immutable device arrays may also remain in resident_em;
    // audits, failures and mutable transport state are always per-shard.
    struct CachedUnifiedEm {
        bool valid{false};
        std::string signature;
        std::shared_ptr<const UnifiedEmPackage> package;
        std::shared_ptr<const std::vector<float>> delta_means;
        std::size_t package_node_count{0};
    };
    struct CachedElectronCt {
        std::string geometry_key,bank_key,joint_key,birth_key,photon_key,charged_key,fp32_key;
        double electron_distance_threshold{-1};
        std::vector<float> electron_distance_mm;
        // Box-metric selected region before any BODY edit; reset with the grid.
        std::string electron_region_key;
        std::vector<std::uint8_t> electron_region_box;
        std::string electron_body_key;
        std::vector<std::uint8_t> electron_body_mask;
        std::vector<float> electron_body_distance_mm;
        std::shared_ptr<IndependentElectronFloatDevice> fp32;
        std::shared_ptr<ElectronJointPilotDevice> joint;
        std::shared_ptr<ElectronBirthBankDevice> birth;
        std::shared_ptr<PhotonResponseBankDevice> photon;
        std::shared_ptr<ChargedBankDevice> charged;
        std::shared_ptr<const CtGrid> grid;
        std::shared_ptr<MaterialElectronDeviceBank> bank;
        float* density{};
        std::uint8_t* material{};
    };
    CachedElectronCt electron_ct;
    CachedUnifiedEm unified_em_cache;
    std::shared_ptr<ResidentPhysicsTables> resident_em, resident_urban, resident_air;
    SchneiderHostCache schneider_host_cache;
    ImmutableInputCache immutable_inputs;
    std::uint64_t physics_cache_hits{0};
    std::uint64_t physics_cache_misses{0};

    void clear_physics_cache() noexcept {
        resident_em.reset();resident_urban.reset();resident_air.reset();
        immutable_inputs = ImmutableInputCache{};
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
