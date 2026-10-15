// Shared setup phase, included inside transport_pipeline<Pipeline>.
    using SecondaryResumeState = SecondaryResumeStateT<BroadBeam>;
    ImmutableInputCache local_inputs;
    auto& inputs = context ? context->impl_->immutable_inputs : local_inputs;
    const auto input_hits_before = inputs.hits;
    const auto input_misses_before = inputs.misses;
    RuntimeScope runtime_transport("transport_total");
    RuntimeScope runtime_setup("transport_setup_including_upload");
    config.validate();
    if (device_name != "cuda" && device_name != "intel")
        throw std::invalid_argument("device must be cuda or intel");
    std::cout<<"[transport-profile] requested="<<config.transport_profile
             <<" compiled_kernel="<<(BroadBeam?("broadbeam_fe"):"reference_detailed")<<"\n";
    validate_schneider_ct_startup(config);
    // Parse the large read-only packages on background threads while the
    // setup below proceeds; each consumer takes the same loader's result.
    // Nothing is prefetched when a process-level cache will serve it.
    {
        SchneiderCtDeviceContext uses{};
        uses.mode = config.material_physics_mode;
        uses.unified_water = config.unified_water_nuclear_transport;
        if (uses.uses_cinel03() &&
            !(context != nullptr && context->impl_->schneider_host_cache.valid)) {
            prefetch_cinel03(config.ct_schneider_primary_cinel03_file);
            prefetch_cinel03(config.ct_schneider_secondary_cinel03_file);
        }
        if ((!BroadBeam && config.enable_minibeam) &&
            !inputs.contains<LoadedCinel03>(input_cache_key(config.minibeam_copper_inclxx_file)))
            prefetch_cinel03(config.minibeam_copper_inclxx_file);
        if ( !(context != nullptr && context->impl_->unified_em_cache.valid))
            prefetch_unified_em(unified_em_request(config));
    }

    const auto start = std::chrono::steady_clock::now();
    const auto& table_energies = stopping_power.energies();
    if (table_energies.size() < 2 || !is_uniform_grid(table_energies)) {
        throw std::invalid_argument("Stopping power table must have a uniform energy grid");
    }

    const auto resolved_device_name =
        device_name;

    sycl::queue local_queue = (context != nullptr)
                                  ? context->impl_->queue
                                  : make_sycl_queue(resolved_device_name);
    auto& queue = local_queue;
    const auto device = queue.get_device();
    const auto expected_backend = device_name == "cuda"
        ? sycl::backend::ext_oneapi_cuda : sycl::backend::ext_oneapi_level_zero;
    if (!device.is_gpu() || queue.get_backend() != expected_backend ||
        (device_name == "intel" && device.get_info<sycl::info::device::vendor_id>() != 0x8086))
        throw std::invalid_argument("Transport queue does not match requested GPU backend");

    const auto device_memory_budget_bytes =
        config.device_memory_budget_gib > 0.0
            ? static_cast<std::size_t>(config.device_memory_budget_gib *
                                       1024.0 * 1024.0 * 1024.0)
            : std::size_t{0};
    DeviceMemoryTracker mem_tracker{queue, device_memory_budget_bytes};

    const auto free_device = [&](auto* pointer) {
        mem_tracker.free(pointer);
    };

    const auto number_of_histories = config.number_of_histories;
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_voxels = config.number_of_voxels();

    const bool use_unified_water = config.unified_water_nuclear_transport;

    bool use_schneider_primary_xs = use_unified_water;
    constexpr std::size_t kSchneiderDiagSlots =
        static_cast<std::size_t>(SchneiderDiagSlot::Count);
    constexpr std::size_t kSchneiderFloatSlots =
        static_cast<std::size_t>(SchneiderFloatSlot::Count);
    std::uint64_t* schneider_diag_device = nullptr;
    float* schneider_float_device = nullptr;
    // Bounded per-record logs (Schneider CT only). Capacities are generous
    // for full-shard runs (50k needs ~10^2 entries); overflow is counted,
    // never silently wrapped.
    constexpr std::uint32_t kSchneiderMissLogCap = 1U << 19;   // 524288
    constexpr std::uint32_t kSchneiderTrackLogCap = 1U << 19;  // 524288
    SchneiderMissRecord* schneider_miss_device = nullptr;
    SchneiderUnsupportedTrack* schneider_track_log_device = nullptr;
    std::uint32_t* schneider_miss_count_device = nullptr;
    std::uint32_t* schneider_track_count_device = nullptr;
    constexpr std::size_t kIonSpeciesEnergySlots =
        TransportResult::species_ledger_species_count *
        TransportResult::species_ledger_metric_count;
    float* ion_species_energy_device = nullptr;
    double* grid_deposited_in_device = nullptr;
    double* grid_deposited_out_device = nullptr;
    constexpr std::size_t kIonSpeciesTerminalSlots =
        IonSpeciesLedgerSchema::species_count *
        IonSpeciesLedgerSchema::terminal_reason_count;
    std::uint64_t* ion_species_terminal_device = nullptr;

    // Read-only species ledger is mode-independent: secondary EM transport
    // records per-species deposited/escaped energy unconditionally on device
    // (null-checked increments), so these two buffers remain in the shared
    // CINEL03 framework. No physics effect:
    // ledger writes are isolated atomics into dedicated buffers.
    if (ion_species_energy_device == nullptr) {
        ion_species_energy_device =
            mem_tracker.allocate<float>(kIonSpeciesEnergySlots);
        if (ion_species_energy_device == nullptr) throw std::bad_alloc();
    }
    if (ion_species_terminal_device == nullptr) {
        ion_species_terminal_device =
            mem_tracker.allocate<std::uint64_t>(kIonSpeciesTerminalSlots);
        if (ion_species_terminal_device == nullptr) throw std::bad_alloc();
    }
    queue.fill(ion_species_energy_device, 0.0F,
               kIonSpeciesEnergySlots).wait_and_throw();
    queue.fill(ion_species_terminal_device, std::uint64_t{0},
               kIonSpeciesTerminalSlots).wait_and_throw();

    // Explicit in-grid/outside-grid deposited-energy sinks (global MeV).
    // Every site crediting deposited ledgers splits the same amount here
    // using that site's paired voxel-scorer guard, so total ≈ in + outside
    // and voxel_sum ≈ in_grid hold by construction. Double precision:
    // global accumulators reach 1e10 MeV where float32 atomics would absorb
    // MeV-scale adds.
    grid_deposited_in_device = mem_tracker.allocate<double>(1);
    grid_deposited_out_device = mem_tracker.allocate<double>(1);
    if (grid_deposited_in_device == nullptr ||
        grid_deposited_out_device == nullptr) {
        throw std::bad_alloc();
    }
    queue.fill(grid_deposited_in_device, 0.0, 1).wait_and_throw();
    queue.fill(grid_deposited_out_device, 0.0, 1).wait_and_throw();

    // CT grid
    const auto enable_ct_grid = config.enable_ct_grid;
    float ct_origin_x = 0.0F, ct_origin_y = 0.0F, ct_origin_z = 0.0F;
    float ct_spacing_x = 0.0F, ct_spacing_y = 0.0F, ct_spacing_z = 0.0F;
    std::uint32_t ct_nx = 0, ct_ny = 0, ct_nz = 0;
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    float* ct_xs_device = nullptr;
    bool ct_material_ids_are_schneider_sections = false;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;

    // Water/Schneider charged-particle energy loss has one implementation.
    const float maximum_mean_loss_fraction = BroadBeam ? static_cast<float>(config.broadbeam_maximum_mean_loss_fraction) : .01f;
    UnifiedEmDevice unified_device;
    UnifiedEmMaterial* unified_materials=nullptr;
    UnifiedEmSpecies* unified_species=nullptr;
    unsigned* unified_energy_index=nullptr;
    unsigned* unified_failure_count=nullptr;
    UnifiedEmFailureRecord* unified_failure_records=nullptr;
    UnifiedEmRecord* unified_records=nullptr;
    UnifiedEmNode* unified_nodes=nullptr;
    EmCubicSegmentCandidate<float>* unified_segments=nullptr;
    std::uint64_t* unified_audit=nullptr;

    const bool global_urban=!BroadBeam && config.multiple_scattering_model=="urban_v2";
    const bool uses_urban_package=global_urban;
    const bool retain_physics_device = context != nullptr;
    const auto nuclear_input_key = [](const std::filesystem::path& path) {
        return !path.empty() && std::filesystem::exists(path)
            ? input_cache_key(path) : path.string()+"|missing";
    };
    const std::string schneider_signature =
        std::to_string(config.primary_atomic_number)+":"+std::to_string(config.primary_mass_number)+"|"+
        nuclear_input_key(config.ct_schneider_primary_rate_file)+"|"+
        nuclear_input_key(config.ct_schneider_primary_cinel03_file)+"|"+
        nuclear_input_key(config.ct_schneider_secondary_rate_file)+"|"+
        nuclear_input_key(config.ct_schneider_secondary_cinel03_file)+"|"+
        std::to_string(static_cast<int>(config.material_physics_mode))+"|"+
        std::to_string(config.unified_water_nuclear_transport)+"|"+
        std::to_string(config.enable_secondary_transport)+"|"+
        std::to_string(static_cast<int>(config.run_mode))+"|"+device_name+
        (config.unified_water_nuclear_transport ? "|"+nuclear_input_key(config.unified_water_material_file)+
            "|"+nuclear_input_key(config.ct_schneider_file.empty()
                ? std::filesystem::path("data/materials/schneider/hu_to_material.txt") : config.ct_schneider_file) : "");
    if (context) {
        auto& cached=context->impl_->schneider_device;
        if (!std::is_same_v<Pipeline, pipelines::ProtonBroadbeam> || !enable_ct_grid ||
            config.unified_water_nuclear_transport || cached.signature!=schneider_signature)
            cached=SyclTransportContext::Impl::CachedSchneiderDevice{};
    }

    if (context && retain_physics_device) {
        if (!uses_urban_package) context->impl_->resident_urban.reset();
        if (BroadBeam || !config.enable_minibeam) context->impl_->resident_air.reset();
        if (BroadBeam || !config.enable_minibeam || config.minibeam_copper_straggling_model != "g4_ion_delta")
            context->impl_->resident_copper_ions.reset();
    }
    const auto resident_tables = [&](auto& slot,const std::string& key,const char* name) {
        const bool hit=slot && slot->signature==key;
        if (!hit) slot=std::make_shared<ResidentPhysicsTables>(queue,key);
        mem_tracker.account_retained(slot->bytes);
        std::cout<<"[device-physics-cache] "<<name<<" "<<(hit?"hit":"miss")
                 <<" resident_bytes="<<slot->bytes<<"\n";
        return slot;
    };

    const float urban_water_half_width_mm=static_cast<float>(config.urban_water_half_width_mm);
    UrbanMcsView urban_device{};
    std::vector<UrbanMcsSpecies> urban_species_host;
    std::uint64_t* urban_step_counts=nullptr;
    if(uses_urban_package) {
        const auto& package = inputs.get<UrbanMcsPackage>(input_cache_key(config.urban_mcs_package_file),
            [&] { return UrbanMcsPackage::load(config.urban_mcs_package_file); });
        if (global_urban) {
            const auto index = package.view().projectile(config.primary_atomic_number, config.primary_mass_number);
            if (index < 0) throw std::invalid_argument("Urban package lacks the YAML primary projectile");
            config.require_primary_reference_mass(package.species[index].mass_mev, "Urban package");
            if ((!BroadBeam && config.enable_minibeam) &&
                package.view().material(-2, static_cast<float>(config.minibeam_copper_density_g_per_cm3)) < 0)
                throw std::invalid_argument("Global minibeam Urban requires a G4_Cu (-2) material record");
        }

        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_urban,
            input_cache_key(config.urban_mcs_package_file),"urban") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        urban_device={upload("materials",package.materials),upload("species",package.species),upload("records",package.records),
            upload("nodes",package.nodes),upload("mfps",package.mfps),upload("segments",package.segments),
            static_cast<unsigned>(package.materials.size()),static_cast<unsigned>(package.species.size())};
        urban_species_host=package.species;
        urban_device.ct_density_scaling=config.urban_ct_density_scaling;
        if(config.urban_ct_density_scaling)
            std::cout<<"[urban-v2] CT continuous26 restricted loss/range; native MSC density scaling; additional elastic recoils retain their separate native context\n";
        urban_step_counts=mem_tracker.allocate<std::uint64_t>(2*package.species.size());
        if(!urban_step_counts)throw std::bad_alloc();
        queue.fill(urban_step_counts,std::uint64_t{0},2*package.species.size()).wait_and_throw();
        std::cout<<"[urban-v2] "
                 <<"global all transported charged ions"
                 <<"; materials="<<package.materials.size()
                 <<" species="<<package.species.size()<<"\n";
    }
    UrbanMcsView copper_fragment_urban_device{};
    std::vector<UrbanMcsSpecies> copper_fragment_species_host;
    std::shared_ptr<ResidentPhysicsTables> copper_ion_resident;
    if (!BroadBeam && config.enable_minibeam && config.minibeam_copper_straggling_model == "g4_ion_delta") {
        const auto& package = inputs.get<UrbanMcsPackage>(input_cache_key(config.minibeam_copper_fragment_urban_package_file),
            [&] { return UrbanMcsPackage::load(config.minibeam_copper_fragment_urban_package_file); });
        if (package.materials.size() != 1 || package.view().material(-2,8.96F) < 0)
            throw std::runtime_error("Cu fragment Urban package requires exact G4_Cu only");
        if (retain_physics_device) copper_ion_resident=resident_tables(context->impl_->resident_copper_ions,
            input_cache_key(config.minibeam_copper_fragment_urban_package_file)+":"+
                (config.minibeam_copper_enable_elastic ? input_cache_key(config.minibeam_copper_fragment_elastic_file) : "elastic-disabled"),
            "copper-ions");
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values) {
            if (copper_ion_resident) return copper_ion_resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        copper_fragment_urban_device={upload("materials",package.materials),upload("species",package.species),
            upload("records",package.records),upload("nodes",package.nodes),upload("mfps",package.mfps),
            upload("segments",package.segments),static_cast<unsigned>(package.materials.size()),static_cast<unsigned>(package.species.size())};
        copper_fragment_species_host=package.species;
        std::cout<<"[copper-ion-packages] native_fragment_species="<<package.species.size()<<'\n';
    }
    // Minibeam beamline air always uses global Urban v2 with exact Cu/air boundaries.
    const bool minibeam_air_urban = !BroadBeam && config.enable_minibeam;
    const float air_slit_max_step = static_cast<float>(config.minibeam_air_slit_max_step_mm);
    UrbanMcsView air_urban_device{};
    int air_urban_projectile = -1;
    if (minibeam_air_urban) {
        if (!config.minibeam_copper_exact_material_boundaries)
            throw std::invalid_argument("Minibeam air Urban requires exact material boundaries");
        const auto& package = inputs.get<UrbanMcsPackage>(input_cache_key(config.minibeam_air_urban_package_file),
            [&] { return UrbanMcsPackage::load(config.minibeam_air_urban_package_file); });
        air_urban_projectile = package.view().projectile(config.primary_atomic_number, config.primary_mass_number);
        if (package.materials.size() != 1 || package.materials[0].section != -3 || air_urban_projectile < 0)
            throw std::invalid_argument("Air Urban requires a G4_AIR (-3) bank containing source ZA");
        if (!config.minibeam_air_ionization_rate_file.empty() &&
            (std::abs(package.materials[0].density-0.00120479F)>1e-9F ||
             std::abs(package.materials[0].production_cut_mm-0.05F)>1e-7F))
            throw std::invalid_argument("Air ionization clock and Urban material/cut must match");
        config.require_primary_reference_mass(package.species[air_urban_projectile].mass_mev, "Air Urban package");
        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_air,
            input_cache_key(config.minibeam_air_urban_package_file),"air") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        air_urban_device={upload("materials",package.materials),upload("species",package.species),upload("records",package.records),
            upload("nodes",package.nodes),upload("mfps",package.mfps),upload("segments",package.segments),
            static_cast<unsigned>(package.materials.size()),static_cast<unsigned>(package.species.size())};
        std::cout << "[air-urban] primary beamline G4_AIR\n";
    }
    bool nonion_primary_package_safe=true;
    {
        const std::string unified_em_signature =
            input_cache_key(config.em_package_file)+"|"+
            input_cache_key(unified_em_request(config).delta_file);
        std::shared_ptr<const UnifiedEmPackage> unified_package;
        bool unified_em_cache_hit=false;
        if(context!=nullptr){
            auto& cached=context->impl_->unified_em_cache;
            if(cached.valid && cached.signature==unified_em_signature){
                unified_package=cached.package;
                unified_em_cache_hit=true;
            }
        }
        if(!unified_package){
            // The prefetch result includes package and delta-sidecar validation.
            unified_package=runtime_call("unified_em_package_load_verify",[&]{return take_unified_em(unified_em_request(config));});
            if(context!=nullptr){
                auto& cached=context->impl_->unified_em_cache;
                cached.valid=true;cached.signature=unified_em_signature;
                cached.package=unified_package;
            }
        }
        const auto& package=*unified_package;
        if(package.materials.size()!=26 || package.continuous_descriptors.size()!=468)
            throw std::invalid_argument("GPU transport requires the continuous26-composition EMCNT001 package; archived density-state packages are host diagnostics only");
        bool primary_record_found = false;
        for (const auto& record : package.records) {
            if (record.z == static_cast<unsigned>(config.primary_atomic_number) &&
                record.a == static_cast<unsigned>(config.primary_mass_number)) {
                primary_record_found = true;
                if(record.is_ion!=0)nonion_primary_package_safe=false;
                for(unsigned j=0;j<record.node_count;++j)
                    if(package.nodes[record.node_offset+j].fluctuation==1)nonion_primary_package_safe=false;
                config.require_primary_reference_mass(record.mass, "Unified EM package");
            }
        }
        if (!primary_record_found)
            throw std::invalid_argument("Unified EM package lacks the YAML primary projectile");
        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_em,unified_em_signature,"em") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        std::cout<<"[physics-cache] unified_em="<<(unified_em_cache_hit?"hit":"miss")
                 <<" nodes="<<package.nodes.size()<<"\n";
        RuntimeScope t_em_device("setup_em_device_upload");
        if(uses_urban_package)std::cout<<"[urban-ionization-clock] native proposal clock limits true transport paths; aggregate energy law unchanged\n";
        std::cout<<"[delta-moments] condensed_partition_v1; aggregate Gamma; nodes="
                 <<package.nodes.size()<<"\n";
        unified_failure_count=mem_tracker.allocate<unsigned>(1);
        unified_failure_records=mem_tracker.allocate<UnifiedEmFailureRecord>(16);
        if(!unified_failure_count || !unified_failure_records)throw std::bad_alloc();
        queue.fill(unified_failure_count,0u,1).wait_and_throw();
        if (resident && resident->contains("energy_index"))
            unified_energy_index=resident->template get<std::remove_pointer_t<decltype(unified_energy_index)>>("energy_index");
        else {
            const auto index=build_unified_em_index(package);
            unified_energy_index=upload("energy_index",index);
        }
        unified_materials=upload("materials",package.materials);unified_species=upload("species",package.species);
        unified_records=upload("records",package.records);unified_nodes=upload("nodes",package.nodes);unified_segments=upload("segments",package.segments);
        unified_audit=mem_tracker.allocate<std::uint64_t>(kUnifiedEmAuditCounters*kUnifiedEmAuditShards);if(!unified_audit)throw std::bad_alloc();
        queue.fill(unified_audit,std::uint64_t{0},kUnifiedEmAuditCounters*kUnifiedEmAuditShards).wait_and_throw();

        unified_device={unified_materials,unified_species,unified_records,unified_nodes,unified_segments,unified_energy_index
        };
        if(!package.continuous_descriptors.empty()) {
            unified_device.continuous_descriptors=upload("continuous_descriptors",package.continuous_descriptors);
            unified_device.continuous_scalars=upload("continuous_scalars",package.continuous_scalars);
            unified_device.continuous_ranges=upload("continuous_ranges",package.continuous_ranges);
            unified_device.continuous_lengths=upload("continuous_lengths",package.continuous_lengths);
            unified_device.continuous_degree=package.continuous_degree;
            unified_device.continuous_scalar_count=package.continuous_scalar_count;
            std::cout<<"[em-continuous-density] compositions="<<package.materials.size()
                     <<" density_response_states=0 range_degree="<<package.continuous_degree-1<<"\n";
        }

        t_em_device.finish();
        std::cout<<"[unified-em] all 18 charged ions; water + Schneider continuous density; native particle step parameters plus 1% combined mean-loss guard; local aggregate delta deposition; density cut-onset/patient accuracy validation pending\n";
    }
    // One host-table cache per process for the Schneider nuclear packages.
    // Reset only when
    // the validated inputs or device change.
    SchneiderHostCache* schneider_host_cache = nullptr;
    if (context != nullptr) {
        auto& cache = context->impl_->schneider_host_cache;
        if (!cache.valid || cache.signature != schneider_signature) {
            cache = SchneiderHostCache{};
            cache.signature = schneider_signature;
            cache.valid = true;
        }
        schneider_host_cache = &cache;
    }
    const auto ct_secondary_exact_faces = config.ct_secondary_exact_faces;

    bool retained_ct_arrays=false;

    if (enable_ct_grid) {
        std::shared_ptr<const CtGrid> grid_storage;
        const auto make_geometry = [&]() { return CtGrid::load(config.ct_grid_file); };
        const auto geometry_key=ct_input_key(config.ct_grid_file);
        if(context) {
            auto& cached=context->impl_->ct_geometry;
            const bool hit=cached.grid && cached.geometry_key==geometry_key;
            if(!hit) {
                queue.wait_and_throw();
                if(cached.density)sycl::free(cached.density,queue);
                if(cached.material)sycl::free(cached.material,queue);
                cached={};cached.grid=make_geometry();cached.geometry_key=geometry_key;
            }
            grid_storage=cached.grid;
            std::cout<<"[ct-geometry-cache] "<<(hit?"hit":"miss")
                     <<" medium="<<(enable_ct_grid?"Schneider_CT":"Water_75eV")<<'\n';
        } else grid_storage=make_geometry();
        const auto& grid=*grid_storage;
        if(enable_ct_grid) {
            if (grid.file_version < CtGrid::version_value)
                throw std::invalid_argument("Unified EM CT requires a Schneider grid with section IDs");
            for (const auto section : grid.material_id)
                if (section >= 25)
                    throw std::invalid_argument("Unified EM CT material section must be in [0,24]");
        }
        ct_origin_x = grid.origin_x_mm;
        ct_origin_y = grid.origin_y_mm;
        ct_origin_z = grid.origin_z_mm;
        ct_spacing_x = grid.spacing_x_mm;
        ct_spacing_y = grid.spacing_y_mm;
        ct_spacing_z = grid.spacing_z_mm;
        ct_nx = grid.nx;
        ct_ny = grid.ny;
        ct_nz = grid.nz;
        ct_material_ids_are_schneider_sections =
            enable_ct_grid && grid.file_version >= CtGrid::version_value;

        const auto voxel_count = static_cast<std::size_t>(ct_nx) * ct_ny * ct_nz;
        if(context) {
            auto& cached=context->impl_->ct_geometry;
            if(!cached.density || !cached.material) {
                if(cached.density) {sycl::free(cached.density,queue);cached.density=nullptr;}
                if(cached.material) {sycl::free(cached.material,queue);cached.material=nullptr;}
                cached.density=sycl::malloc_device<float>(voxel_count,queue);
                cached.material=sycl::malloc_device<std::uint8_t>(voxel_count,queue);
                if(!cached.density || !cached.material)throw std::bad_alloc();
                queue.copy(grid.density_g_per_cm3.data(),cached.density,voxel_count);
                queue.copy(grid.material_id.data(),cached.material,voxel_count).wait_and_throw();
            }
            ct_density_device=cached.density;ct_material_device=cached.material;
            retained_ct_arrays=true;
        } else {
        ct_density_device = mem_tracker.allocate<float>(voxel_count);
        ct_material_device = mem_tracker.allocate<std::uint8_t>(voxel_count);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();
        }

        if constexpr((std::is_same_v<Pipeline,pipelines::ProtonBroadbeam> || std::is_same_v<Pipeline,pipelines::CarbonBroadbeam> || std::is_same_v<Pipeline,pipelines::ProtonMinibeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>)) {
            if(enable_ct_grid && !use_unified_water) {
                RuntimeScope fields_setup("setup_ct_exact_material_fields");
                const auto key=geometry_key+"|"+input_cache_key(config.em_package_file)+"|"+
                    input_cache_key(unified_em_request(config).delta_file)+"|"+device_name+((std::is_same_v<Pipeline,pipelines::CarbonBroadbeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>)?"|ct_exact_fields_range_C12_v1":"|ct_exact_fields_range_v2");
                auto resident=context ? resident_tables(context->impl_->resident_ct_fields,key,"ct_exact_fields") : nullptr;
                std::uint16_t* voxel_tuples=nullptr;
                UnifiedEmCtFields* fields=nullptr;
                float* range_values=nullptr;unsigned* range_offsets=nullptr;
                std::vector<UnifiedEmSpecies> range_species(18);
                queue.copy(unified_device.species,range_species.data(),range_species.size()).wait_and_throw();
                int range_ion=-1;
                constexpr int range_z=(std::is_same_v<Pipeline,pipelines::CarbonBroadbeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>)?6:1;
                constexpr int range_a=(std::is_same_v<Pipeline,pipelines::CarbonBroadbeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>)?12:1;
                for(int i=0;i<18;++i)if(range_species[i].z==range_z && range_species[i].a==range_a)range_ion=i;
                if(range_ion<0)throw std::runtime_error("Exact range cache requires projectile package coverage");
                unsigned tuple_count=0;
                if(resident && resident->contains("ready")) {
                    voxel_tuples=resident->template get<std::uint16_t>("voxels");
                    fields=resident->template get<UnifiedEmCtFields>("fields");
                    tuple_count=static_cast<unsigned>(resident->blocks.at("fields").count/18);
                    range_values=resident->template get<float>("range_values");
                    range_offsets=resident->template get<unsigned>("range_offsets");
                } else {
                    std::unordered_map<std::uint64_t,std::uint16_t> directory;
                    std::vector<std::uint16_t> indices(voxel_count);
                    std::vector<float> densities;
                    std::vector<std::uint8_t> sections;
                    bool too_many=false;
                    for(std::size_t i=0;i<voxel_count;++i) {
                        std::uint32_t bits;std::memcpy(&bits,&grid.density_g_per_cm3[i],sizeof bits);
                        const auto id=std::uint64_t(bits)|(std::uint64_t(grid.material_id[i])<<32);
                        auto it=directory.find(id);
                        if(it==directory.end()) {
                            if(directory.size()>=65536){too_many=true;break;}
                            const auto index=static_cast<std::uint16_t>(directory.size());
                            it=directory.emplace(id,index).first;
                            densities.push_back(grid.density_g_per_cm3[i]);sections.push_back(grid.material_id[i]);
                        }
                        indices[i]=it->second;
                    }
                    // Optional memoization can be ineligible without changing the
                    // underlying continuous-density evaluation or physics coverage.
                    if(!too_many) {
                        tuple_count=densities.size();
                        auto upload_fields=[&]<class T>(const char* name,const std::vector<T>& values) {
                            if(resident)return resident->upload(name,values,mem_tracker);
                            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
                            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
                        };
                        voxel_tuples=upload_fields("voxels",indices);
                        auto* rho=upload_fields("densities",densities);
                        auto* section=upload_fields("sections",sections);
                        fields=upload_fields("fields",std::vector<UnifiedEmCtFields>(tuple_count*18));
                        const auto source=unified_device;
                        queue.parallel_for(sycl::range<1>(tuple_count*18),[=](sycl::id<1> id) {
                            const unsigned i=id[0],t=i/18,ion=i%18;
                            const auto state=source.select(section[t],rho[t],ion);
                            UnifiedEmCtFields value;value.valid=state.valid;
                            if(state.valid){const auto p=state.point();value.material=p.material;value.coordinate=p.continuous_coordinate;}
                            fields[i]=value;
                        }).wait_and_throw();
                        std::vector<EmContinuousDescriptor> descriptors(468);
                        queue.copy(source.continuous_descriptors,descriptors.data(),descriptors.size()).wait_and_throw();
                        std::vector<unsigned> offsets(tuple_count+1);
                        for(unsigned t=0;t<tuple_count;++t)offsets[t+1]=offsets[t]+descriptors[(sections[t]+1)*18+range_ion].range_count;
                        range_offsets=upload_fields("range_offsets",offsets);
                        range_values=upload_fields("range_values",std::vector<float>(offsets.back()));
                        queue.parallel_for(sycl::range<1>(tuple_count),[=](sycl::id<1> id) {
                            const unsigned t=id[0];
                            const auto state=source.select(section[t],rho[t],range_ion);
                            if(!state.valid)return;
                            const auto p=state.point();const float x=2*p.continuous_coordinate-1;
                            for(unsigned j=0;j<p.continuous->range_count;++j)
                                range_values[range_offsets[t]+j]=em_chebyshev(p.continuous_range+j*(p.continuous_degree+1)+1,p.continuous_lengths?p.continuous_lengths[j]:p.continuous_degree,x);
                        }).wait_and_throw();
                        std::cout<<"[ct-exact-range-nodes] values="<<offsets.back()<<" bytes="<<offsets.back()*sizeof(float)<<"\n";
                        if(resident)resident->upload("ready",std::vector<unsigned>{1},mem_tracker);
                    }
                }
                unified_device.ct_fields=fields;unified_device.ct_fields_voxels=voxel_tuples;
                unified_device.ct_fields_count=tuple_count;
                unified_device.ct_range_values=range_values;unified_device.ct_range_offsets=range_offsets;unified_device.ct_range_ion=range_ion;
                std::cout<<"[ct-exact-material-fields] tuples="<<tuple_count<<" voxels="<<voxel_count
                         <<" enabled="<<(fields!=nullptr)<<"\n";
            }
        }

        if (enable_ct_grid) {
            use_schneider_primary_xs = config.nuclear_model != "none";
            for (std::size_t i = 0; i < grid.material_id.size(); ++i)
                if (grid.material_id[i] >= 25)
                    throw std::runtime_error("Invalid Schneider CT composition at voxel " + std::to_string(i));
            if (use_schneider_primary_xs) {
                if (schneider_rate_binary_version(config.ct_schneider_primary_rate_file) != 3)
                    throw std::runtime_error("Schneider CT requires masked primary rate v3");
                std::cout << "[schneider-primary-xs] mode=primary-masked-binary-v3\n";
            }
        }
    }

    // Spot batching / TPS source
    const auto primary_spot_count = config.primary_spot_batch.size();
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            mem_tracker.allocate<PrimarySpotBatchEntry>(primary_spot_count);
        queue.copy(config.primary_spot_batch.data(), primary_spots_device, primary_spot_count)
            .wait_and_throw();
    }

    const bool use_all_elastic = !config.all_ion_elastic_file.empty();
    AllIonElasticView all_elastic{};
    const int primary_elastic_projectile = all_elastic.projectile(
        config.primary_atomic_number, config.primary_mass_number);
    if (use_all_elastic && primary_elastic_projectile < 0)
        throw std::invalid_argument("All-ion elastic has no record for the YAML primary projectile");
    ElasticRecoilStoppingView recoil_stopping{};
    float* elastic_energy = nullptr; // local, queued charged, overflow
    std::uint32_t* elastic_audit = nullptr; // primary, secondary, bad query, unsupported recoil
    if(use_all_elastic) {
        const auto& stop = inputs.get<ElasticRecoilStoppingTable>(input_cache_key(config.elastic_recoil_stopping_file),
            [&] { return ElasticRecoilStoppingTable::load(config.elastic_recoil_stopping_file); });
        auto* keys=mem_tracker.allocate<std::int32_t>(stop.keys.size());
        auto* se=mem_tracker.allocate<float>(stop.energies.size());auto* sv=mem_tracker.allocate<float>(stop.values.size());
        if(!keys||!se||!sv)throw std::bad_alloc();
        queue.copy(stop.keys.data(),keys,stop.keys.size());queue.copy(stop.energies.data(),se,stop.energies.size());queue.copy(stop.values.data(),sv,stop.values.size()).wait_and_throw();
        recoil_stopping={keys,se,sv,static_cast<std::uint32_t>(stop.keys.size()/2),static_cast<std::uint32_t>(stop.energies.size())};
        const auto& bank = inputs.get<AllIonElasticTable>(input_cache_key(config.all_ion_elastic_file),
            [&] { return AllIonElasticTable::load(config.all_ion_elastic_file); });
        config.require_primary_reference_mass(bank.masses[primary_elastic_projectile], "Elastic package");
        auto* e=mem_tracker.allocate<float>(bank.energies.size());
        auto* r=mem_tracker.allocate<float>(bank.rates.size());
        auto* a=mem_tracker.allocate<ElasticSample>(bank.samples.size());
        auto* m=mem_tracker.allocate<double>(18);
        elastic_audit=mem_tracker.allocate<std::uint32_t>(12);
        elastic_energy=mem_tracker.allocate<float>(3);
        if(!elastic_energy)throw std::bad_alloc();
        queue.fill(elastic_energy,0.0F,3);
        if(!e||!r||!a||!m||!elastic_audit)throw std::bad_alloc();
        queue.copy(bank.energies.data(),e,bank.energies.size());queue.copy(bank.rates.data(),r,bank.rates.size());
        queue.copy(bank.samples.data(),a,bank.samples.size());queue.copy(bank.masses.data(),m,18);
        queue.fill(elastic_audit,std::uint32_t{0},12).wait_and_throw();
        all_elastic={e,r,a,m,static_cast<std::uint32_t>(bank.energies.size()),bank.nq};
        if constexpr(std::is_same_v<Pipeline,pipelines::CarbonBroadbeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>) {
          if(enable_ct_grid) {
            const auto ne=bank.energies.size();std::vector<float> totals(18*26*ne,0.0F);
            for(std::size_t projectile=0;projectile<18;++projectile)
              for(std::size_t section=0;section<26;++section)
                for(std::size_t node=0;node<ne;++node)
                  for(std::size_t target=0;target<13;++target)
                    totals[(projectile*26+section)*ne+node]+=bank.rates[((projectile*26+section)*13+target)*ne+node];
            auto* device_totals=mem_tracker.allocate<float>(totals.size());
            if(!device_totals)throw std::bad_alloc();
            queue.copy(totals.data(),device_totals,totals.size()).wait_and_throw();
            all_elastic.species_totals=device_totals;
          }
        }
        if constexpr(std::is_same_v<Pipeline,pipelines::ProtonBroadbeam> || std::is_same_v<Pipeline,pipelines::ProtonMinibeam>) {
          if (enable_ct_grid) {
            const auto ne=bank.energies.size();std::vector<float> totals(26*ne,0.0F);
            for(std::size_t section=0;section<26;++section)
                for(std::size_t node=0;node<ne;++node)
                    for(std::size_t target=0;target<13;++target)
                        totals[section*ne+node]+=bank.rates[((primary_elastic_projectile*26+section)*13+target)*ne+node];
            auto* device_totals=mem_tracker.allocate<float>(totals.size());
            if(!device_totals)throw std::bad_alloc();
            queue.copy(totals.data(),device_totals,totals.size()).wait_and_throw();
            all_elastic.primary_totals=device_totals;all_elastic.totals_projectile=primary_elastic_projectile;
          }
        }

    }
    const bool minibeam_copper_transport = !BroadBeam && config.enable_minibeam;
    float* minibeam_copper_sp_energies_device = nullptr;
    float* minibeam_copper_sp_values_device = nullptr;
    std::uint32_t minibeam_copper_sp_count = 0;

    float* air_ion_e_device=nullptr;
    float* air_ion_rate_device=nullptr;
    std::uint32_t air_ion_count=0;
    float air_ion_peak=0;
    float* minibeam_air_sp_energies_device = nullptr;
    float* minibeam_air_sp_values_device = nullptr;
    std::uint32_t minibeam_air_sp_count = 0;
    float* minibeam_copper_xs_energies_device = nullptr;
    float* minibeam_copper_xs_values_device = nullptr;
    std::uint32_t minibeam_copper_xs_count = 0;
    CopperElasticRecord* minibeam_copper_elastic_device = nullptr;
    CopperElasticRecord* copper_fragment_elastic_device = nullptr;
    std::uint32_t* copper_fragment_elastic_offsets = nullptr;
    std::uint32_t minibeam_copper_elastic_count = 0;
    std::uint32_t* minibeam_copper_elastic_brackets_device = nullptr;
    std::uint32_t minibeam_copper_elastic_bracket_bins = 0;
    float minibeam_copper_elastic_bracket_first_energy = 0.0F;
    float minibeam_copper_elastic_bracket_inverse_width = 0.0F;
    float* minibeam_copper_ion_sp_ratios_device = nullptr;
    std::uint8_t* minibeam_copper_ion_sp_present_device = nullptr;
    float* minibeam_copper_ion_xs_device = nullptr;
    std::uint8_t* minibeam_copper_ion_xs_present_device = nullptr;
    std::uint32_t minibeam_copper_ion_xs_grid_size = 0;
    float minibeam_copper_ion_xs_minimum_energy = 0.0F;
    float minibeam_copper_ion_xs_inverse_step = 0.0F;
    Cinel03EnergyNode* minibeam_copper_nodes_device = nullptr;
    std::uint32_t* minibeam_copper_offsets_device = nullptr;
    std::uint32_t* minibeam_copper_indices_device = nullptr;
    Cinel03DeviceInteraction* minibeam_copper_interactions_device = nullptr;
    Cinel03DeviceProduct* minibeam_copper_products_device = nullptr;
    std::uint32_t minibeam_copper_node_count = 0;
    std::uint32_t minibeam_copper_event_count = 0;
    std::uint32_t minibeam_copper_product_count = 0;
    // Host-side float-pair uploader shared by the Copper tables and the
    // mode-independent water-Urban loss-range table below.
    const auto upload_float_pair = [&](const std::vector<double>& energies,
                                       const std::vector<double>& values,
                                       float*& out_energies,
                                       float*& out_values) {
        std::vector<float> e(energies.begin(), energies.end());
        std::vector<float> v(values.begin(), values.end());
        out_energies = mem_tracker.allocate<float>(e.size());
        out_values = mem_tracker.allocate<float>(v.size());
        if (!out_energies || !out_values) throw std::bad_alloc();
        queue.copy(e.data(), out_energies, e.size());
        // Both temporary host vectors must survive their asynchronous copies.
        queue.copy(v.data(), out_values, v.size()).wait_and_throw();
    };
    if (!config.minibeam_air_ionization_rate_file.empty()) {
        int z=0,a=0;double density=0,cut=0,peak=0;std::string material;
        std::ifstream meta(config.minibeam_air_ionization_rate_file);
        std::string line;
        while(std::getline(meta,line)) {
            std::istringstream in(line);std::string marker,key;in>>marker>>key;
            if(marker!="#")continue;
            if(key=="projectile_z")in>>z;else if(key=="projectile_a")in>>a;
            else if(key=="material")in>>material;else if(key=="density_g_cm3")in>>density;
            else if(key=="production_cut_mm")in>>cut;else if(key=="peak_energy_MeV")in>>peak;
        }
        if(z!=config.primary_atomic_number || a!=config.primary_mass_number ||
           material!="G4_AIR" || std::abs(density-0.00120479)>1e-10 ||
           std::abs(cut-0.05)>1e-12 || !(peak>0) || !std::isfinite(peak))
            throw std::runtime_error("Air ionization rate metadata mismatch");
        const auto& rates=inputs.get<CrossSectionTable>(input_cache_key(config.minibeam_air_ionization_rate_file),
            [&]{return CrossSectionTable::from_csv(config.minibeam_air_ionization_rate_file);});
        if(rates.energies().front()*a>config.energy_cutoff_MeV ||
           rates.energies().back()*a<config.initial_energy_MeVu*a)
            throw std::runtime_error("Air ionization rate energy domain does not cover source");
        upload_float_pair(rates.energies(),rates.values(),air_ion_e_device,air_ion_rate_device);
        air_ion_count=static_cast<std::uint32_t>(rates.energies().size());air_ion_peak=static_cast<float>(peak);
        std::cout<<"[air-ionization-clock] native proposal rate; aggregate air energy law unchanged\n";
    }
    if (minibeam_copper_transport) {
        const auto& copper_sp = inputs.get<StoppingPowerTable>(input_cache_key(config.minibeam_copper_stopping_power_file),
            [&] { return StoppingPowerTable::from_csv(config.minibeam_copper_stopping_power_file); });
        const auto& air_sp = inputs.get<StoppingPowerTable>(input_cache_key(config.minibeam_air_stopping_power_file),
            [&] { return StoppingPowerTable::from_csv(config.minibeam_air_stopping_power_file); });
        const auto& copper_xs = inputs.get<CrossSectionTable>(input_cache_key(config.minibeam_copper_cross_section_file),
            [&] { return CrossSectionTable::from_csv(config.minibeam_copper_cross_section_file); });
        const auto& elastic = inputs.get<std::vector<CopperElasticRecord>>(input_cache_key(config.minibeam_copper_elastic_file),
            [&] { return load_copper_elastic_events(config.minibeam_copper_elastic_file); });
        if(config.minibeam_copper_straggling_model=="g4_ion_delta" && config.minibeam_copper_enable_elastic) {
            const auto& bank=inputs.get<CopperFragmentElasticBank>(input_cache_key(config.minibeam_copper_fragment_elastic_file),
                [&]{return load_copper_fragment_elastic(config.minibeam_copper_fragment_elastic_file,copper_fragment_species_host);});
            if(copper_ion_resident) {
                copper_fragment_elastic_device=copper_ion_resident->upload("elastic_records",bank.records,mem_tracker);
                copper_fragment_elastic_offsets=copper_ion_resident->upload("elastic_offsets",bank.offsets,mem_tracker);
            } else {
                copper_fragment_elastic_device=mem_tracker.allocate<CopperElasticRecord>(bank.records.size());
                copper_fragment_elastic_offsets=mem_tracker.allocate<std::uint32_t>(bank.offsets.size());
                if(!copper_fragment_elastic_device||!copper_fragment_elastic_offsets)throw std::bad_alloc();
                queue.copy(bank.records.data(),copper_fragment_elastic_device,bank.records.size());
                queue.copy(bank.offsets.data(),copper_fragment_elastic_offsets,bank.offsets.size()).wait_and_throw();
            }
        }
        std::vector<float> copper_ion_sp_ratios;
        std::vector<std::uint8_t> copper_ion_sp_present;
        std::vector<float> copper_ion_xs;
        std::vector<std::uint8_t> copper_ion_xs_present;
        if (!config.minibeam_copper_ion_stopping_power_file.empty()) {
            const auto& ion_stopping = inputs.get<IonStoppingPowerTables>(
                input_cache_key(config.minibeam_copper_ion_stopping_power_file) + "|base=" +
                input_cache_key(config.minibeam_copper_stopping_power_file), [&] {
                    return IonStoppingPowerTables::from_csv(config.minibeam_copper_ion_stopping_power_file, copper_sp); });
            copper_ion_sp_ratios = ion_stopping.ratios_to_carbon();
            copper_ion_sp_present = ion_stopping.species_present();
        }
        if (!config.minibeam_copper_ion_cross_section_file.empty()) {
            const auto& ion_xs = inputs.get<IonCrossSectionTables>(input_cache_key(config.minibeam_copper_ion_cross_section_file),
            [&] { return IonCrossSectionTables::from_csv(config.minibeam_copper_ion_cross_section_file); });
            copper_ion_xs = ion_xs.values();
            copper_ion_xs_present = ion_xs.species_present();
            minibeam_copper_ion_xs_grid_size =
                static_cast<std::uint32_t>(ion_xs.energy_grid_size());
            minibeam_copper_ion_xs_minimum_energy = ion_xs.minimum_energy_MeVu();
            minibeam_copper_ion_xs_inverse_step = 1.0F / ion_xs.energy_step_MeVu();
        }
        const auto& loaded_cascade = inputs.get<LoadedCinel03>(input_cache_key(config.minibeam_copper_inclxx_file),
            [&] { return take_cinel03(config.minibeam_copper_inclxx_file); });
        const auto& tables = *loaded_cascade.tables;
        upload_float_pair(copper_sp.energies(), copper_sp.values(),
                          minibeam_copper_sp_energies_device,
                          minibeam_copper_sp_values_device);
        minibeam_copper_sp_count = static_cast<std::uint32_t>(copper_sp.energies().size());
        upload_float_pair(air_sp.energies(), air_sp.values(),
                          minibeam_air_sp_energies_device,
                          minibeam_air_sp_values_device);
        minibeam_air_sp_count = static_cast<std::uint32_t>(air_sp.energies().size());
        upload_float_pair(copper_xs.energies(), copper_xs.values(),
                          minibeam_copper_xs_energies_device,
                          minibeam_copper_xs_values_device);
        minibeam_copper_xs_count = static_cast<std::uint32_t>(copper_xs.energies().size());
        minibeam_copper_elastic_device =
            mem_tracker.allocate<CopperElasticRecord>(elastic.size());
        if (!copper_ion_sp_ratios.empty()) {
            minibeam_copper_ion_sp_ratios_device =
                mem_tracker.allocate<float>(copper_ion_sp_ratios.size());
            minibeam_copper_ion_sp_present_device =
                mem_tracker.allocate<std::uint8_t>(copper_ion_sp_present.size());
        }
        if (!copper_ion_xs.empty()) {
            minibeam_copper_ion_xs_device =
                mem_tracker.allocate<float>(copper_ion_xs.size());
            minibeam_copper_ion_xs_present_device =
                mem_tracker.allocate<std::uint8_t>(copper_ion_xs_present.size());
        }
        minibeam_copper_nodes_device =
            mem_tracker.allocate<Cinel03EnergyNode>(tables.energy_nodes.size());
        minibeam_copper_offsets_device =
            mem_tracker.allocate<std::uint32_t>(tables.event_offsets.size());
        minibeam_copper_indices_device =
            mem_tracker.allocate<std::uint32_t>(tables.event_indices.size());
        minibeam_copper_interactions_device =
            mem_tracker.allocate<Cinel03DeviceInteraction>(tables.interactions.size());
        minibeam_copper_products_device =
            mem_tracker.allocate<Cinel03DeviceProduct>(tables.products.size());
        if (!minibeam_copper_elastic_device ||
            (!copper_ion_sp_ratios.empty() &&
             (!minibeam_copper_ion_sp_ratios_device ||
              !minibeam_copper_ion_sp_present_device)) ||
            (!copper_ion_xs.empty() &&
             (!minibeam_copper_ion_xs_device ||
              !minibeam_copper_ion_xs_present_device)) ||
            !minibeam_copper_nodes_device ||
            !minibeam_copper_offsets_device || !minibeam_copper_indices_device ||
            !minibeam_copper_interactions_device || !minibeam_copper_products_device) {
            throw std::bad_alloc();
        }
        queue.copy(elastic.data(), minibeam_copper_elastic_device, elastic.size());
        if (!copper_ion_sp_ratios.empty()) {
            queue.copy(copper_ion_sp_ratios.data(),
                       minibeam_copper_ion_sp_ratios_device,
                       copper_ion_sp_ratios.size());
            queue.copy(copper_ion_sp_present.data(),
                       minibeam_copper_ion_sp_present_device,
                       copper_ion_sp_present.size());
        }
        if (!copper_ion_xs.empty()) {
            queue.copy(copper_ion_xs.data(), minibeam_copper_ion_xs_device,
                       copper_ion_xs.size());
            queue.copy(copper_ion_xs_present.data(),
                       minibeam_copper_ion_xs_present_device,
                       copper_ion_xs_present.size());
        }
        queue.copy(tables.energy_nodes.data(), minibeam_copper_nodes_device,
                   tables.energy_nodes.size());
        queue.copy(tables.event_offsets.data(), minibeam_copper_offsets_device,
                   tables.event_offsets.size());
        queue.copy(tables.event_indices.data(), minibeam_copper_indices_device,
                   tables.event_indices.size());
        queue.copy(tables.interactions.data(), minibeam_copper_interactions_device,
                   tables.interactions.size());
        queue.copy(tables.products.data(), minibeam_copper_products_device,
                   tables.products.size()).wait_and_throw();
        minibeam_copper_elastic_count = static_cast<std::uint32_t>(elastic.size());
        {
            // Search brackets for the per-step nearest-record rate lookup;
            // each is widened by one record and verified again on device.
            constexpr std::uint32_t bins = 8192;
            const double first = elastic.front().energy_MeV_per_u;
            const double width = (static_cast<double>(elastic.back().energy_MeV_per_u) - first) / bins;
            if (elastic.size() > 2 && width > 0.0) {
                std::vector<float> energies(elastic.size());
                for (std::size_t i = 0; i < elastic.size(); ++i) energies[i] = elastic[i].energy_MeV_per_u;
                std::vector<std::uint32_t> brackets(2 * bins);
                for (std::uint32_t bin = 0; bin < bins; ++bin) {
                    const auto low_energy = static_cast<float>(first + width * bin);
                    const auto high_energy = static_cast<float>(first + width * (bin + 1));
                    auto lower = static_cast<std::size_t>(
                        std::upper_bound(energies.begin(), energies.end(), low_energy) - energies.begin());
                    lower = lower >= 2 ? lower - 2 : 0;
                    auto upper = static_cast<std::size_t>(
                        std::upper_bound(energies.begin(), energies.end(), high_energy) - energies.begin());
                    upper = std::min(upper + 1, energies.size() - 1);
                    brackets[2 * bin] = static_cast<std::uint32_t>(lower);
                    brackets[2 * bin + 1] = static_cast<std::uint32_t>(upper);
                }
                minibeam_copper_elastic_brackets_device = mem_tracker.allocate<std::uint32_t>(brackets.size());
                if (!minibeam_copper_elastic_brackets_device) throw std::bad_alloc();
                queue.copy(brackets.data(), minibeam_copper_elastic_brackets_device, brackets.size()).wait_and_throw();
                minibeam_copper_elastic_bracket_bins = bins;
                minibeam_copper_elastic_bracket_first_energy = static_cast<float>(first);
                minibeam_copper_elastic_bracket_inverse_width = static_cast<float>(1.0 / width);
            }
        }
        minibeam_copper_node_count = static_cast<std::uint32_t>(tables.energy_nodes.size());
        minibeam_copper_event_count = static_cast<std::uint32_t>(tables.interactions.size());
        minibeam_copper_product_count = static_cast<std::uint32_t>(tables.products.size());
        std::cout << "[minibeam-copper] stopping=" << minibeam_copper_sp_count
                  << " inelastic-rate=" << minibeam_copper_xs_count
                  << " elastic-events=" << minibeam_copper_elastic_count
                  << " INCLXX-events/products=" << minibeam_copper_event_count
                  << '/' << minibeam_copper_product_count << '\n';
    }
    // Water-Urban loss-range table: independent of the Copper transport mode.
    RuntimeScope t_sch_ctx("setup_schneider_ctx_host_and_device");
    const SchneiderCtDeviceContext schneider_ct_device_ctx = [&] {
        if constexpr (std::is_same_v<Pipeline, pipelines::ProtonBroadbeam>) {
            if (context && enable_ct_grid && !config.unified_water_nuclear_transport) {
                auto& cached=context->impl_->schneider_device;
                const bool hit=cached.valid && cached.signature==schneider_signature;
                auto tables=resident_tables(cached.tables,schneider_signature,"schneider");
                if (!hit) {
                    cached.valid=false;
                    cached.view=upload_schneider_ct_device_context(
                        queue,mem_tracker,config,schneider_host_cache,tables.get());
                    cached.signature=schneider_signature;
                    cached.valid=true;
                }
                return cached.view;
            }
        }
        return upload_schneider_ct_device_context(queue,mem_tracker,config,schneider_host_cache);
    }();
    t_sch_ctx.finish();
    if (context != nullptr && schneider_host_cache != nullptr) {
        std::cout << "[physics-cache] schneider hits=" << schneider_host_cache->hits
                  << " misses=" << schneider_host_cache->misses << "\n";
    }
    if (schneider_ct_device_ctx.uses_cinel03()) {
        schneider_diag_device = mem_tracker.allocate<std::uint64_t>(kSchneiderDiagSlots);
        schneider_float_device = mem_tracker.allocate<float>(kSchneiderFloatSlots);
        schneider_miss_device =
            mem_tracker.allocate<SchneiderMissRecord>(kSchneiderMissLogCap);
        schneider_track_log_device =
            mem_tracker.allocate<SchneiderUnsupportedTrack>(kSchneiderTrackLogCap);
        schneider_miss_count_device = mem_tracker.allocate<std::uint32_t>(2);
        schneider_track_count_device = mem_tracker.allocate<std::uint32_t>(2);
        if (schneider_diag_device == nullptr || schneider_float_device == nullptr ||
            schneider_miss_device == nullptr || schneider_track_log_device == nullptr ||
            schneider_miss_count_device == nullptr || schneider_track_count_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(schneider_diag_device, std::uint64_t{0}, kSchneiderDiagSlots)
            .wait_and_throw();
        queue.fill(schneider_float_device, 0.0F, kSchneiderFloatSlots).wait_and_throw();
        // [0] = written count, [1] = dropped (over-capacity) count.
        queue.fill(schneider_miss_count_device, std::uint32_t{0}, 2).wait_and_throw();
        queue.fill(schneider_track_count_device, std::uint32_t{0}, 2).wait_and_throw();
    }

    // Scorers & Result buffers
    const auto max_secondary_inelastic_generations =
        config.max_secondary_inelastic_generations;
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    // Resolve the uniform-water grid exactly as its allocation/header do.
    // Water Z geometry is explicit; CT dimensions come from the CCTG header.
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_plane_size = voxel_bins_x * voxel_bins_y;
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    if (enable_ct_grid &&
        std::abs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
        std::abs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
        voxel_min_x_mm = ct_origin_x;
        voxel_min_y_mm = ct_origin_y;
        voxel_max_x_mm =
            ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
        voxel_max_y_mm =
            ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    }

    auto* dose_device = mem_tracker.allocate<DepthAtomicT>(number_of_bins);

    auto* voxel_dose_device = enable_voxel_scoring
                                  ? mem_tracker.allocate<DoseAtomicT>(number_of_voxels)
                                  : nullptr;

    const auto enable_inelastic = config.enable_inelastic;
    if (enable_inelastic && !use_schneider_primary_xs) {
        throw std::runtime_error("Nuclear transport requires the validated Schneider/unified-water CINEL03 path");
    }
    const auto enable_secondary_transport = config.enable_secondary_transport;
    auto* deposited_device = mem_tracker.allocate<float>(number_of_histories);
    auto* sampled_incident_device = mem_tracker.allocate<float>(number_of_histories);
    auto* escaped_device = mem_tracker.allocate<float>(number_of_histories);
    auto* steps_device = mem_tracker.allocate<std::uint32_t>(number_of_histories);
    auto* untracked_nuclear_device = mem_tracker.allocate<float>(number_of_histories);
    auto* other_terminal_energy_device = mem_tracker.allocate<float>(number_of_histories);
    auto* cutoff_stopped_energy_device = mem_tracker.allocate<float>(number_of_histories);

    auto* beamline_removed_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_primary_survivor_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_air_loss_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* minibeam_event_counts_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<std::uint64_t>(minibeam_event_counter_count)
        : nullptr;
    const bool enable_minibeam_fragment_miss_joint =
        (!BroadBeam && config.enable_minibeam) &&
        config.minibeam_copper_fragment_cascade_generations > 0;
    constexpr auto minibeam_fragment_miss_joint_cell_count =
        MinibeamDiagnostics::fragment_miss_joint_cell_count;
    auto* minibeam_fragment_miss_joint_counts_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_energy_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_depth_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_remaining_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* schneider_inelastic_device =
        use_schneider_primary_xs ? mem_tracker.allocate<std::uint64_t>(1) : nullptr;
    auto* primary_terminal_counts_device = mem_tracker.allocate<std::uint64_t>(4);

    if (deposited_device == nullptr || sampled_incident_device == nullptr || escaped_device == nullptr || steps_device == nullptr ||
        untracked_nuclear_device == nullptr || other_terminal_energy_device == nullptr ||
        cutoff_stopped_energy_device == nullptr || primary_terminal_counts_device == nullptr ||
        ((!BroadBeam && config.enable_minibeam) && (beamline_removed_device == nullptr ||
         beamline_primary_survivor_device == nullptr ||
         beamline_air_loss_device == nullptr ||
         minibeam_event_counts_device == nullptr ||
         (enable_minibeam_fragment_miss_joint &&
          (minibeam_fragment_miss_joint_counts_device == nullptr ||
           minibeam_fragment_miss_joint_energy_device == nullptr ||
           minibeam_fragment_miss_joint_depth_device == nullptr ||
           minibeam_fragment_miss_joint_remaining_device == nullptr)))) ||
        (use_schneider_primary_xs && schneider_inelastic_device == nullptr)) {
        free_device(deposited_device);
        free_device(sampled_incident_device);
        free_device(escaped_device);
        free_device(steps_device);
        free_device(untracked_nuclear_device);
        free_device(other_terminal_energy_device);
        free_device(cutoff_stopped_energy_device);
        free_device(schneider_inelastic_device);
        free_device(primary_terminal_counts_device);
        free_device(beamline_removed_device);
        free_device(beamline_primary_survivor_device);
        free_device(beamline_air_loss_device);
    free_device(minibeam_event_counts_device);

        free_device(minibeam_fragment_miss_joint_counts_device);
        free_device(minibeam_fragment_miss_joint_energy_device);
        free_device(minibeam_fragment_miss_joint_depth_device);
        free_device(minibeam_fragment_miss_joint_remaining_device);

        throw std::bad_alloc();
    }

    queue.fill(untracked_nuclear_device, 0.0F, number_of_histories);
    queue.fill(other_terminal_energy_device, 0.0F, number_of_histories);
    queue.fill(cutoff_stopped_energy_device, 0.0F, number_of_histories);
    if (beamline_removed_device != nullptr) {
        queue.fill(beamline_removed_device, 0.0F, number_of_histories);
        queue.fill(beamline_primary_survivor_device, 0.0F, number_of_histories);
        queue.fill(beamline_air_loss_device, 0.0F, number_of_histories);
        queue.fill(minibeam_event_counts_device, std::uint64_t{0},
                   minibeam_event_counter_count);

        if (minibeam_fragment_miss_joint_counts_device != nullptr) {
            queue.fill(minibeam_fragment_miss_joint_counts_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_energy_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_depth_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_remaining_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
        }

    }
    if (schneider_inelastic_device != nullptr) {
        queue.fill(schneider_inelastic_device, std::uint64_t{0}, 1);
    }
    queue.fill(primary_terminal_counts_device, std::uint64_t{0}, 4).wait_and_throw();

    const std::size_t max_secondaries = config.secondary_queue_capacity;
    const bool need_secondary_buffers =

        enable_inelastic;
    auto* secondary_queue_device =
        need_secondary_buffers
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* secondary_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    uint32_t* secondary_overflow_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    float* secondary_overflow_energy_device =
        need_secondary_buffers
            ? mem_tracker.allocate<float>(1)
            : nullptr;
    // CT uses the identical upstream Cu/air transport and entry record, but
    // must retain the generic material-aware downstream kernel. Never apply
    // the uniform-water substitutions to patient transport.
    const bool ct_minibeam_split = !BroadBeam && config.enable_minibeam &&
        config.enable_ct_grid;
    // The shared beamline and generic material-aware stage 4 retain the
    // projectile ZA and total kinetic energy in their runtime state. C12 may
    // use this handoff with its own pinned packages and the shared Cu-child queue.
    const bool ct_minibeam_projectile_supported =
        (config.primary_atomic_number == 1 && config.primary_mass_number == 1) ||
        (config.primary_atomic_number == 6 && config.primary_mass_number == 12);
    if(ct_minibeam_split && (config.run_mode!=RunMode::research ||
        config.enable_tps_source || config.enable_tps_coordinate_system ||
        !ct_minibeam_projectile_supported ||
        config.multiple_scattering_model!="urban_v2" ||
        !config.enable_secondary_transport))
        throw std::invalid_argument("CT split minibeam requires research proton/C12 Urban copper transport without TPS/replay");
    // Proton and C12 CT Cu/air charged children are queued after the
    // compact beamline launch instead of inline in the primary kernel.
    const bool queue_copper_primary_children = !BroadBeam && config.run_mode==RunMode::research &&
        config.enable_minibeam && (!config.enable_ct_grid || ct_minibeam_split) && !config.enable_tps_source &&
        !config.enable_tps_coordinate_system &&
        ((config.primary_atomic_number==1 && config.primary_mass_number==1) ||
         (ct_minibeam_split && config.primary_atomic_number==6 && config.primary_mass_number==12)) &&
        config.multiple_scattering_model=="urban_v2" &&
        config.enable_secondary_transport &&
        (config.minibeam_primary_specialization || ct_minibeam_split);
    // Carbon nuclear events can create more charged children than proton events.
    // Retain the configured secondary cap and reject any overflow without truncation.
    const auto copper_primary_child_capacity=std::min<std::size_t>(max_secondaries,
        std::max<std::size_t>(4096,
            (config.primary_atomic_number==6 ? 16 : 2)*number_of_histories));
    auto* copper_primary_child_tasks=queue_copper_primary_children
        ? mem_tracker.allocate<CopperPrimaryChildTask>(copper_primary_child_capacity) : nullptr;
    auto* copper_primary_child_counts=queue_copper_primary_children
        ? mem_tracker.allocate<std::uint32_t>(3) : nullptr;
    if(queue_copper_primary_children && (!copper_primary_child_tasks || !copper_primary_child_counts))throw std::bad_alloc();
    if(copper_primary_child_counts)queue.fill(copper_primary_child_counts,0U,3).wait_and_throw();
    auto* minibeam_copper_cascade_queue_device =
        need_secondary_buffers && (!BroadBeam && config.enable_minibeam) &&
                config.minibeam_copper_fragment_cascade_generations > 0
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* minibeam_copper_cascade_count_device =
        minibeam_copper_cascade_queue_device
            ? mem_tracker.allocate<std::uint32_t>(1)
            : nullptr;

    if (need_secondary_buffers) {
        if (!secondary_queue_device || !secondary_count_device ||
            !secondary_overflow_count_device || !secondary_overflow_energy_device
            || ((!BroadBeam && config.enable_minibeam) &&
                config.minibeam_copper_fragment_cascade_generations > 0 &&
                !minibeam_copper_cascade_count_device)
            )
            throw std::bad_alloc();
        queue.fill(secondary_count_device, 0U, 1);
        queue.fill(secondary_overflow_count_device, 0U, 1);
        queue.fill(secondary_overflow_energy_device, 0.0F, 1);
        if (minibeam_copper_cascade_count_device) {
            queue.fill(minibeam_copper_cascade_count_device, 0U, 1);
        }

        queue.wait_and_throw();
    }

    auto* in_fov_dose_device =
        enable_voxel_scoring ? mem_tracker.allocate<DepthAtomicT>(number_of_bins) : nullptr;
    if (in_fov_dose_device != nullptr) {
        queue.fill(in_fov_dose_device, DepthAtomicT{0}, number_of_bins).wait_and_throw();
    }

    if ((dose_device == nullptr) || (deposited_device == nullptr) || (escaped_device == nullptr) || (steps_device == nullptr) || ((enable_voxel_scoring) && ((voxel_dose_device == nullptr) || (in_fov_dose_device == nullptr)))) {
        throw std::bad_alloc();
    }

    queue.memset(dose_device, 0, number_of_bins * sizeof(DepthAtomicT));
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }

    std::size_t local_size = Pipeline::primary_workgroup(config.enable_ct_grid);
    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        // Preserve the CUDA launch sizes used by the eight presets.
        // Explicit history_chunk_size remains authoritative.
        history_chunk = Pipeline::history_chunk(number_of_histories);
        std::cout << "[history-chunk-size] backend default=" << history_chunk
                  << "\n";
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);

    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto inverse_depth_bin_width_mm = 1.0F / depth_bin_width_mm;
    const auto inverse_voxel_size_x_mm = 1.0F / voxel_size_x_mm;
    const auto inverse_voxel_size_y_mm = 1.0F / voxel_size_y_mm;
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_primary_steps = config.maximum_primary_steps;
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);

    if (device_name == "cuda" && !(!BroadBeam && config.enable_minibeam)) {
        cuda_clock_warmup(queue, mem_tracker);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_secondary_energy_straggling = config.enable_secondary_energy_straggling;

    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto active_water_radiation_length = use_unified_water
        ? schneider_ct_device_ctx.water_radiation_length_g_cm2 : water_radiation_length_g_per_cm2;
    const auto enable_multiple_scattering = BroadBeam || config.enable_multiple_scattering;
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    // Fermi-Eyges MCS applies to every charged ion with species-water
    // parameters; otherwise global Urban v2 handles all charged ions.
    const bool primary_fermi_eyges =
        BroadBeam || config.multiple_scattering_model == "fermi_eyges";
    const bool fe_gaussian_only=config.fermi_eyges_gaussian_only;

    const auto c12_fermi_eyges_max_segment_mm =
        static_cast<float>(config.fermi_eyges_max_segment_mm);
    if (primary_fermi_eyges) {
        std::cout << "[ion-mcs] model=fermi_eyges gaussian_only=" << fe_gaussian_only
                  << " species=all_charged parameters=species_water max_segment_mm="
                  << c12_fermi_eyges_max_segment_mm << '\n';
    }
    const auto enable_tps_source = config.uses_fixed_patient_coordinates();
    const auto random_seed = config.random_seed;
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime = static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime = static_cast<float>(config.emittance_sigma_y_prime);
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
    const auto enable_minibeam = (!BroadBeam && config.enable_minibeam);
    const auto minibeam_angle_rad = static_cast<float>(
        config.minibeam_collimator_angle_deg * 0.01745329251994329577);
    const auto minibeam_cos = std::cos(minibeam_angle_rad);
    const auto minibeam_sin = std::sin(minibeam_angle_rad);
    const auto minibeam_block_radius =
        static_cast<float>(config.minibeam_radius_mm);
    const auto minibeam_block_thickness =
        static_cast<float>(config.minibeam_collimator_thickness_mm);
    const auto minibeam_slit_length =
        static_cast<float>(config.minibeam_slit_length_mm);
    const auto minibeam_slit_thickness =
        static_cast<float>(config.minibeam_slit_thickness_mm);
    const auto minibeam_slit_width = static_cast<float>(config.minibeam_slit_width_mm);
    const auto minibeam_slit_pitch = static_cast<float>(config.minibeam_slit_pitch_mm);
    const auto minibeam_slit_offset = static_cast<float>(config.minibeam_slit_offset_mm);
    const auto minibeam_slit_count = config.minibeam_slit_count;
    const auto minibeam_block_center_z = -static_cast<float>(
        config.minibeam_water_entrance_world_y_mm +
        config.minibeam_collimator_center_to_isocenter_mm);
    const auto minibeam_copper_max_step =
        static_cast<float>(config.minibeam_copper_max_step_mm);
    const auto minibeam_copper_exact_material_boundaries =
        config.minibeam_copper_exact_material_boundaries;
    const auto minibeam_copper_density =
        static_cast<float>(config.minibeam_copper_density_g_per_cm3);
    const auto minibeam_copper_radiation_length =
        static_cast<float>(config.minibeam_copper_radiation_length_g_per_cm2);
    const auto minibeam_copper_mcs_scale =
        static_cast<float>(config.minibeam_copper_mcs_scale);
    const auto minibeam_copper_fragment_mcs_scale =
        static_cast<float>(config.minibeam_copper_fragment_mcs_scale);
    const auto minibeam_copper_fragment_cascade_generations =
        config.minibeam_copper_fragment_cascade_generations;
    const auto minibeam_copper_enable_mcs = config.minibeam_copper_enable_mcs;
    const bool copper_proton_native_fluctuation =
        config.minibeam_copper_straggling_model != "clipped_gaussian";
    const bool copper_ion_native_transport =
        config.minibeam_copper_straggling_model == "g4_ion_delta";
    const bool copper_primary_native_mean = config.minibeam_copper_primary_native_mean_loss;
    const auto minibeam_copper_enable_energy_straggling =
        config.minibeam_copper_enable_energy_straggling;
    const auto minibeam_copper_straggling_scale =
        static_cast<float>(config.minibeam_copper_straggling_scale);
    const auto minibeam_copper_fragment_enable_energy_straggling =
        config.minibeam_copper_fragment_enable_energy_straggling;
    const auto minibeam_copper_fragment_straggling_scale =
        static_cast<float>(config.minibeam_copper_fragment_straggling_scale);
    const auto minibeam_copper_enable_elastic =
        config.minibeam_copper_enable_elastic;
    const auto minibeam_copper_enable_inelastic =
        config.minibeam_copper_enable_nuclear_attenuation;
    const auto emittance_correlation_x = static_cast<float>(config.emittance_correlation_x);

    const auto emittance_correlation_y = static_cast<float>(config.emittance_correlation_y);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.primary_ion().rest_mass_MeV);
    // FE is calibrated per nucleon; copper uses the same FP32 A*amu product
    // as the production beamline model. Package mass checks use nuclear mass.
    const float primary_fe_mass_per_u = config.primary_atomic_number == 1
        ? static_cast<float>(config.primary_ion().rest_mass_MeV)
        : static_cast<float>(ion_nucleon_rest_mass_MeV);
    const float primary_copper_scattering_mass_MeV = primary_mass_number * primary_fe_mass_per_u;

    std::cout << "[input-cache] hits=" << inputs.hits-input_hits_before
              << " misses=" << inputs.misses-input_misses_before << "\n";
    runtime_setup.finish();
    // Fold eligible CT configurations into dedicated ion kernels.
    // Other configurations keep generic stage 6.
    bool broadbeam_ct_primary_specialize=false,broadbeam_ct_secondary_specialize=false;
    // Prove the fixed CT proton configuration before specializing.
    if constexpr (std::is_same_v<Pipeline, pipelines::ProtonBroadbeam>) {
        broadbeam_ct_primary_specialize = enable_ct_grid && !use_unified_water &&
            enable_voxel_scoring && enable_multiple_scattering && enable_inelastic &&
            use_schneider_primary_xs &&
            primary_atomic_number==1 && primary_mass_number==1 && inverse_mass_number==1.0F &&
            nonion_primary_package_safe;
        std::cout << "[proton-ct-local-electron-specialize] enabled="
                  << broadbeam_ct_primary_specialize << "\n";
    }

    if constexpr (std::is_same_v<Pipeline, pipelines::CarbonBroadbeam>) {
        broadbeam_ct_primary_specialize = enable_ct_grid && !use_unified_water &&
            enable_voxel_scoring && enable_multiple_scattering && enable_inelastic &&
            use_schneider_primary_xs &&
            primary_atomic_number==6 && primary_mass_number==12 && inverse_mass_number==1.0F/12.0F;
        broadbeam_ct_secondary_specialize = enable_ct_grid && enable_voxel_scoring &&
            ct_secondary_exact_faces && ct_skip_homogeneous_face_clamp && !use_unified_water;
        std::cout << "[carbon-ct-local-electron-specialize] primary="
                  << broadbeam_ct_primary_specialize << " secondary="
                  << broadbeam_ct_secondary_specialize << "\n";
    }
