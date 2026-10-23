// SyclTransportContext::Impl for the Unified EM transport TU.

// Parsed, immutable Schneider/CINEL03 host tables shared across shards in one
// process. Keyed by source identity, input paths, sizes and modification times.
// A session may also retain the validated immutable device view for CT broadbeam.
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
        schneider_device=CachedSchneiderDevice{};
        resident_beamline_em.reset();resident_em.reset();resident_urban.reset();resident_condensed.reset();resident_air.reset();resident_copper_ions.reset();resident_ct_fields.reset();
        const auto release = [this](auto*& pointer) {
            if (pointer != nullptr) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };
        release(ct_geometry.density);
        release(ct_geometry.material);
        ct_geometry=CachedCtGeometry{};
    }

    // Persistent, validated, immutable host parse cache shared across shards in
    // one process. Keyed by exact input path + size and modification time (never by filename
    // alone). Immutable device arrays may also remain in resident_em;
    // audits, failures and mutable transport state are always per-shard.
    struct CachedUnifiedEm {
        bool valid{false};
        std::string signature;
        std::shared_ptr<const UnifiedEmPackage> package;
    };
    struct CachedCtGeometry {
        std::string geometry_key;
        std::shared_ptr<const CtGrid> grid;
        float* density{};
        std::uint8_t* material{};
    };
    CachedCtGeometry ct_geometry;
    CachedUnifiedEm unified_em_cache;
    std::shared_ptr<ResidentPhysicsTables> resident_em, resident_urban, resident_air, resident_copper_ions, resident_condensed;
    std::shared_ptr<ResidentPhysicsTables> resident_ct_fields, resident_beamline_em;
    // Published only after complete upload and source/mass validation.
    // All particle/scoring/diagnostic buffers remain owned by each shard.
    struct CachedSchneiderDevice {
        std::shared_ptr<ResidentPhysicsTables> tables;
        SchneiderCtDeviceContext view{};
        std::string signature;
        bool valid{false};
    };
    CachedSchneiderDevice schneider_device;
    SchneiderHostCache schneider_host_cache;
    ImmutableInputCache immutable_inputs;

    std::string device_name;
    sycl::queue queue;
};
