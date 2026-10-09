// Shared finalize phase, included inside transport_pipeline<Pipeline, JointResponse>.
    runtime_steps.finish();
    RuntimeScope runtime_post("transport_readback_and_host_finalize");
    std::vector<DepthAtomicT> dose_device_host(number_of_bins);
    queue.copy(dose_device, dose_device_host.data(), number_of_bins).wait_and_throw();
    std::vector<double> dose_host(number_of_bins);
    std::transform(dose_device_host.begin(), dose_device_host.end(), dose_host.begin(),
                   [](DoseAtomicT val) { return static_cast<double>(val); });

    std::vector<double> voxel_dose_host;
    if (enable_voxel_scoring) {
        std::vector<DoseAtomicT> voxel_dose_device_host(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_device_host.data(), number_of_voxels)
            .wait_and_throw();
        voxel_dose_host.resize(number_of_voxels);
        std::transform(voxel_dose_device_host.begin(), voxel_dose_device_host.end(),
                       voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> primary_voxel_track_length_host;
    if (enable_primary_voxel_fluence) {
        std::vector<DoseAtomicT> device_host(number_of_voxels);
        queue.copy(primary_voxel_track_length_device, device_host.data(),
                   number_of_voxels).wait_and_throw();
        primary_voxel_track_length_host.resize(number_of_voxels);
        std::transform(device_host.begin(), device_host.end(),
                       primary_voxel_track_length_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> charged_origin_voxel_dose_host;
    std::vector<double> minibeam_component_voxel_dose_host;
    std::vector<double> minibeam_energy_band_roi_dose_host;
    std::array<double, 6> he4_hazard_audit_host{};
    if (he4_hazard_audit_device)
        queue.copy(he4_hazard_audit_device, he4_hazard_audit_host.data(), 6).wait_and_throw();
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            charged_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(charged_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        charged_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       charged_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    if (enable_minibeam_component_voxel_scoring) {
        const auto value_count =
            minibeam_component_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(minibeam_component_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        minibeam_component_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_component_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    if (enable_minibeam_energy_band_roi_scoring) {
        std::vector<DoseAtomicT> device_host(
            minibeam_energy_band_roi_value_count);
        queue.copy(minibeam_energy_band_roi_dose_device, device_host.data(),
                   minibeam_energy_band_roi_value_count).wait_and_throw();
        minibeam_energy_band_roi_dose_host.resize(
            minibeam_energy_band_roi_value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_energy_band_roi_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::vector<double> minibeam_primary_roi_dose_host;
    if (enable_minibeam_primary_roi_scoring) {
        std::vector<DoseAtomicT> device_host(minibeam_primary_roi_value_count);
        queue.copy(minibeam_primary_roi_dose_device, device_host.data(),
                   minibeam_primary_roi_value_count).wait_and_throw();
        minibeam_primary_roi_dose_host.resize(minibeam_primary_roi_value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_primary_roi_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::array<std::uint64_t, minibeam_spatial_audit_slot_count>
        minibeam_spatial_audit_host{};
    if (enable_minibeam_primary_roi_scoring) {
        queue.copy(minibeam_spatial_audit_device, minibeam_spatial_audit_host.data(),
                   minibeam_spatial_audit_slot_count).wait_and_throw();
    }

    std::vector<double> be_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            be_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(be_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        be_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       be_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::vector<double> he_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            he_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(he_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        he_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       he_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> in_fov_dose_host;
    if (enable_voxel_scoring && in_fov_dose_device != nullptr) {
        std::vector<DepthAtomicT> in_fov_device_host(number_of_bins);
        queue.copy(in_fov_dose_device, in_fov_device_host.data(), number_of_bins).wait_and_throw();
        in_fov_dose_host.resize(number_of_bins);
        std::transform(in_fov_device_host.begin(), in_fov_device_host.end(),
                       in_fov_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::array<std::uint32_t,12> elastic_counts_host{};
    std::array<float,3> elastic_energy_host{};
    if(use_all_elastic) {
        auto& counts=elastic_counts_host;queue.copy(elastic_audit,counts.data(),12);
        queue.copy(elastic_energy,elastic_energy_host.data(),3).wait_and_throw();
        std::cout<<"[all-ion-elastic] primary="<<counts[0]<<" secondary="<<counts[1]
                 <<" bad_queries="<<counts[2]<<" unsupported_recoils="<<counts[3]<<'\n';
        std::cout<<"[elastic-failure-sites] primary_rate="<<counts[4]<<" primary_draw="<<counts[5]<<" recoil_stopping="<<counts[6]<<" secondary_rate="<<counts[7]<<" secondary_draw="<<counts[8]<<" null_post_em="<<counts[9]<<'\n';
        if(counts[2]||counts[3])throw std::runtime_error("All-ion elastic coverage failure: missing bank query or recoil transport; dose rejected");
    }

    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> sampled_incident_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<float> untracked_host(number_of_histories, 0.0F);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(sampled_incident_device, sampled_incident_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories);
    if (untracked_nuclear_device != nullptr) {
        queue.copy(untracked_nuclear_device, untracked_host.data(), number_of_histories);
    }
    std::vector<float> other_terminal_host(number_of_histories, 0.0F);
    if (other_terminal_energy_device != nullptr) {
        queue.copy(other_terminal_energy_device, other_terminal_host.data(), number_of_histories);
    }
    std::vector<float> cutoff_stopped_host(number_of_histories, 0.0F);
    if (cutoff_stopped_energy_device != nullptr) {
        queue.copy(cutoff_stopped_energy_device, cutoff_stopped_host.data(), number_of_histories);
    }
    std::vector<float> beamline_removed_host;
    std::vector<float> beamline_primary_survivor_host;
    std::vector<float> beamline_air_loss_host;
    std::array<std::uint64_t, minibeam_event_counter_count>
        minibeam_event_counts_host{};
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_counts_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_energy_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_depth_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_remaining_host;
    std::vector<MinibeamPhaseSpaceRecord> minibeam_phase_space_host;
    std::vector<CopperHistoryDiagnostic> copper_diagnostic_host;
    std::vector<PrimaryMediumEntryState> copper_capture_entry_host;
    if(copper_capture_entry_device) {
        copper_capture_entry_host.resize(number_of_histories);
        queue.copy(copper_capture_entry_device,copper_capture_entry_host.data(),number_of_histories).wait_and_throw();
    }
    if (copper_diagnostic_device) {
        copper_diagnostic_host.resize(number_of_histories);
        queue.copy(copper_diagnostic_device,copper_diagnostic_host.data(),number_of_histories).wait_and_throw();
    }
    std::vector<MinibeamWaterPrimaryPlaneRecord>
        minibeam_water_primary_plane_records_host;
    std::vector<MinibeamFragmentPhaseSpaceRecord>
        minibeam_fragment_phase_space_host;
    if (beamline_removed_device != nullptr) {
        beamline_removed_host.resize(number_of_histories);
        beamline_primary_survivor_host.resize(number_of_histories);
        beamline_air_loss_host.resize(number_of_histories);
        queue.copy(beamline_removed_device, beamline_removed_host.data(),
                   number_of_histories);
        queue.copy(beamline_primary_survivor_device,
                   beamline_primary_survivor_host.data(), number_of_histories);
        queue.copy(beamline_air_loss_device,
                   beamline_air_loss_host.data(), number_of_histories);
        queue.copy(minibeam_event_counts_device,
                   minibeam_event_counts_host.data(),
                   minibeam_event_counts_host.size());
        queue.wait_and_throw();
        if (copper_proton_native_fluctuation) {
            if(air_ion_count)std::cout<<"AIR_IONIZATION_CLOCK steps="
                <<minibeam_event_counts_host[air_ion_step_slot]<<" proposals="
                <<minibeam_event_counts_host[air_ion_proposal_slot]<<'\n';
            std::cout << (copper_ion_native_transport ? "COPPER_ION_LOSS_QUALITY failures=" : "COPPER_PROTON_LOSS_QUALITY failures=")
                      << minibeam_event_counts_host[copper_proton_loss_failure_slot]
                      << " primary=" << minibeam_event_counts_host[copper_proton_primary_loss_slot]
                      << " fragment=" << minibeam_event_counts_host[copper_proton_fragment_loss_slot] << '\n';
            if (minibeam_event_counts_host[copper_proton_loss_failure_slot])
                throw std::runtime_error("Invalid copper ion fluctuation draw: reject partial dose");
        }

        if (copper_ion_native_transport) {
            std::cout << "COPPER_ION_TRANSPORT_QUALITY failures="
                << minibeam_event_counts_host[copper_ion_transport_failure_slot]
                << " fragment_urban=" << minibeam_event_counts_host[copper_fragment_urban_slot]
                << " fragment_elastic=" << minibeam_event_counts_host[copper_fragment_elastic_slot]
                << " post_loss_null=" << minibeam_event_counts_host[copper_fragment_post_loss_null_slot] << '\n';
            if (minibeam_event_counts_host[copper_ion_transport_failure_slot]) {
                const auto* f=minibeam_event_counts_host.data()+copper_ion_first_failure_slot;
                const auto as_float=[](std::uint64_t bits) {auto u=static_cast<std::uint32_t>(bits);float v;std::memcpy(&v,&u,4);return v;};
                std::cout<<"COPPER_ION_FIRST_FAILURE reason="<<f[1]<<" Z="<<f[2]<<" A="<<f[3]
                    <<" energy="<<as_float(f[4])<<" step="<<f[5]<<" path="<<as_float(f[6])<<" tau="<<as_float(f[7])<<'\n';
                throw std::runtime_error("Invalid Cu ion transport/coverage: reject partial dose");
            }
        }

        if (minibeam_fragment_miss_joint_counts_device != nullptr) {
            minibeam_fragment_miss_joint_counts_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_energy_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_depth_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_remaining_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_counts_device,
                       minibeam_fragment_miss_joint_counts_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_energy_device,
                       minibeam_fragment_miss_joint_energy_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_depth_device,
                       minibeam_fragment_miss_joint_depth_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_remaining_device,
                       minibeam_fragment_miss_joint_remaining_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
        }
        if (minibeam_phase_space_device != nullptr) {
            minibeam_phase_space_host.resize(number_of_histories);
            queue.copy(minibeam_phase_space_device,
                       minibeam_phase_space_host.data(), number_of_histories);
        }
        if (minibeam_water_primary_plane_records_device != nullptr) {
            minibeam_water_primary_plane_records_host.resize(
                number_of_histories * minibeam_water_primary_plane_count);
            queue.copy(
                minibeam_water_primary_plane_records_device,
                minibeam_water_primary_plane_records_host.data(),
                minibeam_water_primary_plane_records_host.size());
        }
        if (minibeam_fragment_phase_space_device != nullptr) {
            std::uint32_t count = 0;
            queue.copy(minibeam_fragment_phase_space_count_device, &count, 1)
                .wait_and_throw();
            count = std::min<std::uint32_t>(
                count, static_cast<std::uint32_t>(max_secondaries));
            minibeam_fragment_phase_space_host.resize(count);
            if (count != 0) {
                queue.copy(minibeam_fragment_phase_space_device,
                           minibeam_fragment_phase_space_host.data(), count);
            }
        }
    }
        if(uses_urban_package) {
            std::vector<std::uint64_t> counts(2*urban_species_host.size());
            queue.copy(urban_step_counts,counts.data(),counts.size()).wait_and_throw();
            for(unsigned i=0;i<urban_species_host.size();++i)
                std::cout<<"[urban-v2-steps] Z="<<urban_species_host[i].z<<" A="<<urban_species_host[i].a
                         <<" primary="<<counts[2*i]<<" secondary="<<counts[2*i+1]<<"\n";
        }
        {
        std::array<std::uint64_t,kUnifiedEmAuditCounters*kUnifiedEmAuditShards> shard_audit{};
        queue.copy(unified_audit,shard_audit.data(),shard_audit.size()).wait_and_throw();
        std::uint64_t audit[8]{};
        for(unsigned shard=0;shard<kUnifiedEmAuditShards;++shard)
            for(unsigned counter=0;counter<kUnifiedEmAuditCounters;++counter)
                audit[counter]+=shard_audit[static_cast<std::size_t>(counter)*kUnifiedEmAuditShards+shard];
        std::cout<<"[unified-em-audit]";for(auto count:audit)std::cout<<" "<<count;std::cout<<"\n";
        if(audit[0]) {
            unsigned count=0;queue.copy(unified_failure_count,&count,1).wait_and_throw();
            std::vector<UnifiedEmFailureRecord> records(std::min(count,16u));
            queue.copy(unified_failure_records,records.data(),records.size()).wait_and_throw();
            for(const auto& f:records)std::cerr<<"[unified-em-failure] reason="<<f.reason<<" section="<<f.section<<" Z="<<f.z<<" A="<<f.a<<" kinetic_MeV="<<f.energy<<" density="<<f.density<<" step_mm="<<f.step<<" mean="<<f.d0<<" delta_mean="<<f.d1<<" delta_variance="<<f.d2<<"\n";
            std::cerr<<"[runtime-rejected-kernel] primary_s="<<primary_kernel_seconds<<" secondary_s="<<secondary_kernel_seconds<<" histories="<<number_of_histories<<"\n";
            throw std::runtime_error("Unified EM missing domain or sampling failure; dose rejected");
        }
        mem_tracker.free(unified_failure_count);mem_tracker.free(unified_failure_records);
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
            std::uint64_t secprof[60]{};queue.copy(sec_step_profile_device,secprof,60).wait_and_throw();
            std::cout<<"[secondary-step-profile] slot steps short depMeV\n";
            for(int sec_si=0;sec_si<20;++sec_si)
                std::cout<<"[secondary-step-profile] "<<sec_si<<" "<<secprof[sec_si]<<" "<<secprof[20+sec_si]<<" "<<(static_cast<double>(secprof[40+sec_si])*1e-6)<<"\n";
        }
        if constexpr(CARBON_EM_EMPTY_BUCKET_AUDIT) {
            std::uint64_t counts[10]{};
            queue.copy(unified_empty_bucket_audit,counts,10).wait_and_throw();
            constexpr const char* names[5]={"node","raw0","raw1","raw2","raw3"};
            for(unsigned kind=0;kind<5;++kind) {
                const auto total=counts[2*kind];
                const auto empty=counts[2*kind+1];
                const double fraction=total?static_cast<double>(empty)/static_cast<double>(total):0.0;
                std::cout<<"[em-empty-bucket-audit] kind="<<names[kind]
                         <<" total="<<total<<" empty="<<empty
                         <<" fraction="<<fraction<<"\n";
            }
        }
        if constexpr(CARBON_EM_INTERVAL_WIDTH_AUDIT) {
            std::uint64_t widths[15]{};
            queue.copy(unified_interval_width_audit,widths,15).wait_and_throw();
            constexpr const char* names[3]={"node","raw0","raw1"};
            for(unsigned kind=0;kind<3;++kind) {
                std::uint64_t total=0;
                for(unsigned w=0;w<5;++w)total+=widths[kind*5+w];
                std::cout<<"[em-interval-width-audit] kind="<<names[kind]<<" total="<<total;
                for(unsigned w=0;w<4;++w)
                    std::cout<<" w"<<w<<"="<<widths[kind*5+w];
                std::cout<<" w>=4="<<widths[kind*5+4]<<"\n";
            }
        }
        if constexpr(CARBON_EM_SEARCH_KEY_AUDIT) {
            std::uint32_t observed=0;
            queue.copy(unified_search_audit_count,&observed,1).wait_and_throw();
            const auto sampled=std::min(observed,kUnifiedSearchAuditCapacity);
            std::vector<UnifiedEmSearchAuditRecord> records(sampled);
            queue.copy(unified_search_audit_records,records.data(),sampled).wait_and_throw();
            const auto group_key=[](const auto& r){
                return std::tuple{r.kind,r.launch,r.warp,r.step,r.ordinal};
            };
            std::sort(records.begin(),records.end(),[&](const auto& a,const auto& b){
                return std::tuple{group_key(a),a.key}<std::tuple{group_key(b),b.key};
            });
            std::uint64_t groups[5]{},lanes[5]{},unique[5]{},shared4[5]{};
            for(std::size_t begin=0;begin<records.size();) {
                std::size_t end=begin+1;
                while(end<records.size() && group_key(records[end])==group_key(records[begin]))++end;
                std::uint64_t u=1;
                for(std::size_t i=begin+1;i<end;++i)if(records[i].key!=records[i-1].key)++u;
                const auto kind=records[begin].kind;
                ++groups[kind];lanes[kind]+=end-begin;unique[kind]+=u;
                if(u<=4)++shared4[kind];
                begin=end;
            }
            constexpr const char* names[5]={"node","raw0","raw1","raw2","raw3"};
            std::cout<<"[em-search-key-audit] observed="<<observed
                     <<" sampled="<<sampled<<"\n";
            for(unsigned kind=0;kind<5;++kind) {
                std::cout<<"[em-search-key-audit] kind="<<names[kind]
                         <<" groups="<<groups[kind]<<" lanes="<<lanes[kind]
                         <<" unique="<<unique[kind]
                         <<" mean_unique="<<(groups[kind]?double(unique[kind])/groups[kind]:0.0)
                         <<" groups_unique_le4="<<shared4[kind]<<"\n";
            }
        }
        mem_tracker.free(delta_mean_device);mem_tracker.free(unified_energy_index);mem_tracker.free(unified_sections);mem_tracker.free(unified_materials);mem_tracker.free(unified_species);mem_tracker.free(unified_records);
        mem_tracker.free(unified_nodes);mem_tracker.free(unified_segments);mem_tracker.free(unified_audit);
        if constexpr(CARBON_EM_EMPTY_BUCKET_AUDIT) mem_tracker.free(unified_empty_bucket_audit);
        if constexpr(CARBON_EM_INTERVAL_WIDTH_AUDIT) mem_tracker.free(unified_interval_width_audit);
        if constexpr(CARBON_EM_SEARCH_KEY_AUDIT) {
            mem_tracker.free(unified_search_audit_records);
            mem_tracker.free(unified_search_audit_count);
        }
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) mem_tracker.free(sec_step_profile_device);
    }
    std::uint64_t schneider_inelastic_host = 0;
    if (schneider_inelastic_device != nullptr) {
        queue.copy(schneider_inelastic_device, &schneider_inelastic_host, 1);
    }
    std::array<std::uint64_t, 4> terminal_counts_host{};
    if (primary_terminal_counts_device != nullptr) {
        queue.copy(primary_terminal_counts_device, terminal_counts_host.data(), 4).wait_and_throw();
    }

    // Only shared species and grid ledgers are backed by device buffers.
    std::array<float, kIonSpeciesEnergySlots> ion_species_energy_host{};
    std::array<std::uint64_t, kIonSpeciesTerminalSlots> ion_species_terminal_host{};
    double grid_deposited_in_host = 0.0;
    double grid_deposited_out_host = 0.0;
    queue.copy(ion_species_energy_device, ion_species_energy_host.data(),
               kIonSpeciesEnergySlots);
    queue.copy(ion_species_terminal_device, ion_species_terminal_host.data(),
               kIonSpeciesTerminalSlots);
    if (grid_deposited_in_device != nullptr)
        queue.copy(grid_deposited_in_device, &grid_deposited_in_host, 1);
    if (grid_deposited_out_device != nullptr)
        queue.copy(grid_deposited_out_device, &grid_deposited_out_host, 1);
    std::array<std::uint64_t, kSchneiderDiagSlots> schneider_diag_host{};
    static_assert(kSchneiderDiagSlots ==
                      static_cast<std::size_t>(SchneiderDiagSlot::Count),
                  "Schneider host mirror must match the device schema");
    std::array<float, 12> schneider_float_host{};
    static_assert(12 == static_cast<std::size_t>(SchneiderFloatSlot::Count),
                  "Schneider float mirror must match the device schema");
    if (schneider_diag_device != nullptr) {
        queue.copy(schneider_diag_device, schneider_diag_host.data(), kSchneiderDiagSlots);
    }
    if (schneider_float_device != nullptr) {
        queue.copy(schneider_float_device, schneider_float_host.data(), kSchneiderFloatSlots);
    }
    std::array<std::uint32_t, 2> schneider_miss_counts_host{0, 0};
    std::array<std::uint32_t, 2> schneider_track_counts_host{0, 0};
    if (schneider_miss_count_device != nullptr) {
        queue.copy(schneider_miss_count_device, schneider_miss_counts_host.data(), 2);
    }
    if (schneider_track_count_device != nullptr) {
        queue.copy(schneider_track_count_device, schneider_track_counts_host.data(), 2);
    }
    uint32_t overflow_count_host = 0;
    float overflow_energy_host = 0.0F;
    if (secondary_overflow_count_device != nullptr) {
        queue.copy(secondary_overflow_count_device, &overflow_count_host, 1);
        queue.copy(secondary_overflow_energy_device, &overflow_energy_host, 1);
    }
    queue.wait_and_throw();

    // Electron worker ledgers are private FP32 sums. Widen only on the host
    // after transport, avoiding a shared FP64 atomic in the full FP32 worker.
    if(float_electron_grid_ledger) {
        std::vector<float> partial(2*material_queue_lanes);
        queue.copy(float_electron_grid_ledger,partial.data(),partial.size()).wait_and_throw();
        for(unsigned lane=0;lane<material_queue_lanes;++lane) {
            grid_deposited_in_host+=partial[2*lane];
            grid_deposited_out_host+=partial[2*lane+1];
        }
    }
    double electron_vr_gain_host=0;
    if(float_electron_vr_ledger) {
        std::vector<float> partial(material_queue_lanes);std::array<std::uint64_t,2> work{};
        queue.copy(float_electron_vr_ledger,partial.data(),partial.size());
        queue.copy(electron_vr_work,work.data(),2).wait_and_throw();
        for(const float x:partial)electron_vr_gain_host+=x;
        if(work[0]||work[1])
            std::cout<<"[electron-variance-reduction-work] roulette_killed="<<work[0]<<" roulette_survived="<<work[1]
                     <<" weight_gain_MeV="<<electron_vr_gain_host<<"\n";
    }

    // Per-record Schneider logs: counts are exact after the fence above.
    // Staged into host vectors here (device buffers are freed below);
    // moved into TransportResult after its declaration.
    std::vector<SchneiderMissRecord> schneider_miss_host;
    std::vector<SchneiderUnsupportedTrack> schneider_track_host;
    if (schneider_miss_device != nullptr && schneider_miss_counts_host[0] > 0) {
        const auto n_miss = std::min<std::uint32_t>(
            schneider_miss_counts_host[0], kSchneiderMissLogCap);
        schneider_miss_host.resize(n_miss);
        queue.copy(schneider_miss_device, schneider_miss_host.data(), n_miss)
            .wait_and_throw();
    }
    if (schneider_track_log_device != nullptr && schneider_track_counts_host[0] > 0) {
        const auto n_trk = std::min<std::uint32_t>(
            schneider_track_counts_host[0], kSchneiderTrackLogCap);
        schneider_track_host.resize(n_trk);
        queue.copy(schneider_track_log_device, schneider_track_host.data(), n_trk)
            .wait_and_throw();
    }

    std::array<std::uint64_t,7> electron_joint_diag_host{};
    std::array<std::uint64_t,3> material_untracked_host{};
    if(material_untracked_device)queue.copy(material_untracked_device,material_untracked_host.data(),3).wait_and_throw();

    if(material_failure_device && material_untracked_host[2]) {
        MaterialPacketFailure f;queue.copy(material_failure_device,&f,1).wait_and_throw();
        const auto& c=f.before.cursor;
        std::cerr<<"[material-packet-failure] history="<<f.history<<" step="<<f.step<<" rng="<<f.rng
                 <<" photon_reason="<<f.photon_reason<<std::setprecision(9)<<" status="<<int(f.after.status)<<" gap="<<int(f.after.gap)<<" section="<<c.section
                 <<" rho="<<c.density_g_cm3<<" physical_rho="<<c.physical_density_g_cm3
                 <<" pdg="<<c.pdg<<" KE="<<c.energy_MeV<<" W="<<f.before.weight_MeV
                 <<" position="<<c.position[0]<<','<<c.position[1]<<','<<c.position[2]<<" direction="<<c.direction[0]<<','<<c.direction[1]<<','<<c.direction[2]
                 <<" after_position="<<f.after.cursor.position[0]<<','<<f.after.cursor.position[1]<<','<<f.after.cursor.position[2]<<'\n';
    }
    if(electron_joint_diag_device)queue.copy(electron_joint_diag_device,electron_joint_diag_host.data(),7).wait_and_throw();
    if(use_electron_fp32) {
        if(electron_joint_diag_host[6]!=electron_tasks_total)
            throw std::runtime_error("Independent FP32 electron queue completion mismatch; batch rejected");
        std::cout<<"[independent-fp32-completion] expected="<<electron_tasks_total
                 <<" completed="<<electron_joint_diag_host[6]<<"\n";
    }
    // Free buffers
    free_device(electron_joint_diag_device);
    free_device(dose_device);
    free_device(primary_terminal_counts_device);
    free_device(in_fov_dose_device);
    free_device(voxel_dose_device);
    free_device(primary_voxel_track_length_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(minibeam_component_voxel_dose_device);
    free_device(minibeam_energy_band_roi_dose_device);
    free_device(minibeam_primary_roi_dose_device);
    free_device(minibeam_spatial_audit_device);
    free_device(minibeam_fixed_region_by_x_bin_device);
    free_device(be_isotope_origin_voxel_dose_device);
    free_device(he4_hazard_audit_device);
    free_device(he_isotope_origin_voxel_dose_device);
    free_device(deposited_device);
    free_device(secondary_queue_device);
    free_device(generic_recoil_reference_sp_device);
    free_device(secondary_count_device);
    free_device(ion_species_energy_device);
    free_device(ion_species_terminal_device);
    free_device(untracked_nuclear_device);
    free_device(other_terminal_energy_device);
    free_device(cutoff_stopped_energy_device);
    free_device(copper_diagnostic_device);
    free_device(copper_capture_entry_device);
    free_device(copper_history_mask_device);
    free_device(minibeam_phase_space_device);
    free_device(minibeam_water_primary_plane_depths_device);
    free_device(minibeam_water_primary_plane_records_device);
    free_device(minibeam_fragment_phase_space_device);
    free_device(minibeam_fragment_phase_space_count_device);
    free_device(minibeam_copper_cascade_queue_device);
    free_device(minibeam_copper_cascade_count_device);
    free_device(beamline_removed_device);
    free_device(beamline_primary_survivor_device);
    free_device(beamline_air_loss_device);
    free_device(minibeam_event_counts_device);

    free_device(minibeam_fragment_miss_joint_counts_device);
    free_device(minibeam_fragment_miss_joint_energy_device);
    free_device(minibeam_fragment_miss_joint_depth_device);
    free_device(minibeam_fragment_miss_joint_remaining_device);
    free_device(minibeam_copper_sp_energies_device);
    free_device(minibeam_copper_sp_values_device);

    free_device(air_ion_e_device);
    free_device(air_ion_rate_device);
    free_device(minibeam_air_sp_energies_device);
    free_device(minibeam_air_sp_values_device);
    free_device(minibeam_copper_xs_energies_device);
    free_device(minibeam_copper_xs_values_device);
    free_device(minibeam_copper_elastic_device);
    if(!copper_ion_resident) {
        free_device(copper_fragment_elastic_device);
        free_device(copper_fragment_elastic_offsets);
    }
    free_device(minibeam_copper_nodes_device);
    free_device(minibeam_copper_offsets_device);
    free_device(minibeam_copper_indices_device);
    free_device(minibeam_copper_interactions_device);
    free_device(minibeam_copper_products_device);
    free_device(secondary_overflow_count_device);
    free_device(secondary_overflow_energy_device);
    free_device(escaped_device);
    free_device(sampled_incident_device);
    free_device(steps_device);
    free_device(primary_spots_device);
    if(!retained_ct_arrays) {
        free_device(ct_density_device);
        free_device(ct_material_device);
    }
    free_device(ct_xs_device);
    free_device(broadbeam_electron_mask);
    free_device(broadbeam_birth_regions);free_device(broadbeam_region_work);
    free_device(schneider_inelastic_device);
    free_device(schneider_diag_device);
    free_device(schneider_float_device);
    free_device(schneider_miss_device);
    free_device(schneider_track_log_device);
    free_device(schneider_miss_count_device);
    free_device(schneider_track_count_device);

    TransportResult result;
    result.fe_sampling_diagnostics_enabled=fe_diagnostics;
    if (fe_counts) {
        queue.copy(fe_counts,result.fe_sampling_counts.data(),8).wait_and_throw();
        free_device(fe_counts);
    }
    result.primary_elastic_interactions=elastic_counts_host[0];
    result.secondary_elastic_interactions=elastic_counts_host[1];
    result.elastic_post_em_null_collisions=elastic_counts_host[9]+elastic_counts_host[10];
    result.elastic_local_deposited_energy_MeV=elastic_energy_host[0];
    result.elastic_queued_charged_energy_MeV=elastic_energy_host[1];
    result.elastic_queue_overflow_energy_MeV=elastic_energy_host[2];
    result.elastic_queue_overflow=elastic_counts_host[11];
    result.schneider_miss_log = std::move(schneider_miss_host);
    result.electron_joint_diagnostics={electron_joint_diag_host[0],electron_joint_diag_host[1],electron_joint_diag_host[2],
        static_cast<double>(electron_joint_diag_host[3])*1e-6,static_cast<double>(electron_joint_diag_host[4])*1e-6,static_cast<double>(electron_joint_diag_host[5])*1e-6,electron_joint_diag_host[6]};
    result.material_electron_photon_untracked_MeV=double(material_untracked_host[0])*1e-6;
    result.material_electron_untracked_MeV=double(material_untracked_host[0]+material_untracked_host[1])*1e-6;
    result.electron_variance_reduction_weight_gain_MeV=electron_vr_gain_host;
    free_device(material_untracked_device);
    free_device(material_failure_device);

    result.schneider_unsupported_tracks = std::move(schneider_track_host);
    result.backend = "sycl-" + resolved_device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
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
    if ((!BroadBeam && config.enable_minibeam)) {
        result.backend += "+minibeam-copper";
    }
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_ct_grid && enable_ct_material_mcs) {
        result.backend += "+ct-material-mcs";
    }
    if (enable_ct_grid) result.backend += "+ct-grid-unified-em+ct-dda";
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring+scorer-decoupled";
    }

    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);
    result.primary_voxel_track_length_mm =
        std::move(primary_voxel_track_length_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    result.minibeam_component_voxel_deposited_energy_MeV =
        std::move(minibeam_component_voxel_dose_host);
    result.minibeam_energy_band_roi_deposited_energy_MeV =
        std::move(minibeam_energy_band_roi_dose_host);
    result.minibeam_primary_roi_values =
        std::move(minibeam_primary_roi_dose_host);
    result.minibeam_spatial_audit_counts = minibeam_spatial_audit_host;
    result.be_isotope_origin_voxel_deposited_energy_MeV =
        std::move(be_isotope_origin_voxel_dose_host);
    result.he_isotope_origin_voxel_deposited_energy_MeV =
        std::move(he_isotope_origin_voxel_dose_host);
    result.helium4_hazard_audit = he4_hazard_audit_host;
    result.in_fov_deposited_energy_MeV = std::move(in_fov_dose_host);
    if (enable_fragment_birth_spectrum) {
        constexpr std::size_t categories = light_isotope_category_count;
        constexpr std::size_t generations = birth_generation_bin_count;
        result.birth_counts_by_generation.assign(categories * generations, 0);
        result.birth_ke_sum_MeV_by_generation.assign(categories * generations, 0.0);
        result.birth_mevu_hist.assign(birth_hist_plane_size(birth_mevu_bin_count), 0);
        result.birth_depth_hist.assign(birth_hist_plane_size(number_of_bins), 0);
        result.birth_cos_hist.assign(birth_hist_plane_size(birth_cos_bin_count), 0);
        result.birth_parent_mevu_hist.assign(
            birth_hist_plane_size(birth_parent_mevu_bin_count), 0);
        result.birth_parent_z_hist.assign(
            birth_hist_plane_size(birth_parent_z_bin_count), 0);
        result.birth_parent_product_mevu_hist.assign(birth_joint_plane_size(), 0);

        const auto parent_mevu_bin = birth_parent_mevu_bin(
            config.initial_total_energy_MeV(), config.primary_mass_number);
        const auto parent_z_bin = birth_parent_z_bin(config.primary_atomic_number);
        for (std::size_t birth_index = 0;
             birth_index < birth_secondaries_host.size(); ++birth_index) {
            const auto& fragment = birth_secondaries_host[birth_index];
            if (fragment.z == 2 && (fragment.a == 3 || fragment.a == 4 || fragment.a == 6)) {
                result.helium_birth_records.push_back({
                    fragment.parent_history, static_cast<unsigned>(fragment.generation),
                    static_cast<unsigned>(fragment.a), fragment.energy_MeV,
                    fragment.pos_x_mm, fragment.pos_y_mm, fragment.pos_z_mm,
                    fragment.dir_x, fragment.dir_y, fragment.dir_z, fragment.weight});
            }
            if (fragment.z == 6 && fragment.a == 12) {
                result.c12_birth_records.push_back({
                    birth_index, fragment.parent_history, fragment.rng_stream,
                    static_cast<unsigned>(fragment.generation),
                    static_cast<unsigned>(fragment.birth_region),
                    fragment.energy_MeV, fragment.pos_x_mm, fragment.pos_y_mm,
                    fragment.pos_z_mm, fragment.dir_x, fragment.dir_y,
                    fragment.dir_z, fragment.weight});
            }
            const auto generation = birth_generation_bin(
                static_cast<std::uint8_t>(fragment.generation));
            const auto category =
                light_isotope_category(fragment.z, fragment.a);
            if (category >= categories || fragment.a <= 0) {
                continue;
            }
            const auto summary_index = category * generations + generation;
            const auto mevu_bin =
                birth_mevu_bin(fragment.energy_MeV, fragment.a);
            const auto depth_bin = std::min(
                number_of_bins - 1,
                static_cast<std::size_t>(std::max(
                    0.0F, fragment.pos_z_mm / depth_bin_width_mm)));
            const auto cos_bin = birth_cos_bin(fragment.dir_z);
            ++result.birth_counts_by_generation[summary_index];
            result.birth_ke_sum_MeV_by_generation[summary_index] +=
                fragment.energy_MeV;
            ++result.birth_mevu_hist[birth_hist_index(
                category, generation, mevu_bin, birth_mevu_bin_count)];
            ++result.birth_depth_hist[birth_hist_index(
                category, generation, depth_bin, number_of_bins)];
            ++result.birth_cos_hist[birth_hist_index(
                category, generation, cos_bin, birth_cos_bin_count)];
            ++result.birth_parent_mevu_hist[birth_hist_index(
                category, generation, parent_mevu_bin, birth_parent_mevu_bin_count)];
            ++result.birth_parent_z_hist[birth_hist_index(
                category, generation, parent_z_bin, birth_parent_z_bin_count)];

            ++result.birth_parent_product_mevu_hist[birth_joint_index(
                category, generation, parent_mevu_bin, mevu_bin)];
        }
    }

    result.initial_energy_MeV =
        std::accumulate(sampled_incident_host.begin(), sampled_incident_host.end(), 0.0);
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);

    result.untracked_nuclear_energy_MeV =
        std::accumulate(untracked_host.begin(), untracked_host.end(), 0.0);
    result.primary_inelastic_terminated_count = terminal_counts_host[0];
    result.primary_escaped_ct_count = terminal_counts_host[1];
    result.primary_stopped_count = terminal_counts_host[2];
    result.primary_other_terminal_count = terminal_counts_host[3];
    result.primary_inelastic_removed_kinetic_MeV = result.untracked_nuclear_energy_MeV;
    result.primary_other_terminal_kinetic_MeV =
        std::accumulate(other_terminal_host.begin(), other_terminal_host.end(), 0.0);
    result.primary_cutoff_stopped_energy_MeV =
        std::accumulate(cutoff_stopped_host.begin(), cutoff_stopped_host.end(), 0.0);
    if ((!BroadBeam && config.enable_minibeam)) {

        result.beamline_removed_energy_MeV = std::accumulate(
            beamline_removed_host.begin(), beamline_removed_host.end(), 0.0);
        result.beamline_removed_energy_MeV += std::accumulate(
            beamline_air_loss_host.begin(), beamline_air_loss_host.end(), 0.0);
        const auto removed = static_cast<std::uint64_t>(std::count_if(
            beamline_removed_host.begin(), beamline_removed_host.end(),
            [](const float value) { return value > 0.0F; }));
        const auto primary_survivors = static_cast<std::uint64_t>(std::count_if(
            beamline_primary_survivor_host.begin(),
            beamline_primary_survivor_host.end(),
            [](const float value) { return value > 0.0F; }));
        std::uint64_t direct_survivors = 0;
        for (std::size_t history = 0; history < number_of_histories; ++history)
            if (beamline_primary_survivor_host[history] > 0.0F &&
                beamline_removed_host[history] == 0.0F) ++direct_survivors;
        result.minibeam.enabled = true;
        result.minibeam.incident_histories = number_of_histories;
        result.minibeam.copper_touched_histories = removed;
        result.minibeam.direct_air_slit_histories = direct_survivors;
        result.minibeam.water_entrance_primary = primary_survivors;
        result.minibeam.copper_nuclear_interactions = minibeam_event_counts_host[0];
        result.minibeam.copper_generated_direct_secondaries =
            minibeam_event_counts_host[1];
        result.minibeam.copper_charged_survivors = minibeam_event_counts_host[2];
        for (std::size_t category = 0; category < 9; ++category) {
            result.minibeam.copper_charged_survivors_by_species[category] =
                minibeam_event_counts_host[3 + category];
            result.minibeam.copper_charged_survivor_energy_by_species_MeV[
                category] = static_cast<double>(
                    minibeam_event_counts_host[12 + category]) / 1000.0;
            result.minibeam.copper_fragment_absorptions_by_species[category] =
                minibeam_event_counts_host[21 + category];
            result.minibeam
                .copper_fragment_absorbed_energy_by_species_MeV[category] =
                static_cast<double>(minibeam_event_counts_host[30 + category]) /
                1000.0;
            result.minibeam
                .copper_fragment_absorption_straight_copper_path_by_species_mm[
                    category] =
                static_cast<double>(minibeam_event_counts_host[39 + category]) /
                1000.0;
        }
        result.minibeam.copper_fragment_cascade_interactions =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_interactions_slot];
        result.minibeam.copper_fragment_cascade_lookup_hits =
            minibeam_event_counts_host[minibeam_fragment_cascade_hits_slot];
        for (std::size_t generation = 0;
             generation < MinibeamDiagnostics::copper_cascade_generation_count;
             ++generation) {
            result.minibeam
                .copper_fragment_cascade_interactions_by_generation[
                    generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_interaction_slot + generation];
            result.minibeam.copper_fragment_cascade_hits_by_generation[
                generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_hit_slot + generation];
            result.minibeam.copper_fragment_cascade_misses_by_generation[
                generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_miss_slot + generation];
        }
        for (std::size_t miss = 0; miss < 6; ++miss) {
            result.minibeam.copper_fragment_cascade_lookup_misses[miss] =
                minibeam_event_counts_host[
                    minibeam_fragment_cascade_miss_slot + miss];
        }
        result.minibeam.copper_fragment_cascade_generated_charged =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_charged_slot];
        result.minibeam.copper_fragment_cascade_generated_neutral =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_neutral_slot];
        result.minibeam.copper_fragment_cascade_generated_unsupported =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_unsupported_slot];
        result.minibeam.copper_fragment_cascade_queue_overflows =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_overflow_slot];
        result.minibeam.copper_fragment_cascade_local_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_cascade_local_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_untracked_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_cascade_untracked_keV_slot]) / 1000.0;
        for (std::size_t category = 0; category < 9; ++category) {
            result.minibeam
                .copper_fragment_ignored_nuclear_optical_depth_by_species[
                    category] =
                static_cast<double>(minibeam_event_counts_host[
                    minibeam_fragment_ignored_tau_micro_slot + category]) /
                1.0e6;
            result.minibeam.copper_fragment_terminal_tracks_by_species[
                category] = minibeam_event_counts_host[
                    minibeam_fragment_terminal_track_slot + category];
            result.minibeam
                .copper_fragment_ignored_reaction_probability_by_species[
                    category] =
                static_cast<double>(minibeam_event_counts_host[
                    minibeam_fragment_ignored_probability_micro_slot +
                    category]) /
                1.0e6;
            result.minibeam
                .copper_fragment_cascade_lookup_misses_by_species[category] =
                minibeam_event_counts_host[
                    minibeam_fragment_miss_species_slot + category];
        }
        for (std::size_t bin = 0;
             bin < MinibeamDiagnostics::fragment_lookup_energy_bin_count;
             ++bin) {
            result.minibeam
                .copper_fragment_cascade_lookup_misses_by_energy[bin] =
                minibeam_event_counts_host[
                minibeam_fragment_miss_energy_slot + bin];
        }
        result.minibeam.copper_fragment_miss_joint_counts =
            std::move(minibeam_fragment_miss_joint_counts_host);
        result.minibeam.copper_fragment_miss_joint_input_energy_keV =
            std::move(minibeam_fragment_miss_joint_energy_host);
        result.minibeam.copper_fragment_miss_joint_collision_depth_um =
            std::move(minibeam_fragment_miss_joint_depth_host);
        result.minibeam.copper_fragment_miss_joint_remaining_copper_um =
            std::move(minibeam_fragment_miss_joint_remaining_host);
        result.minibeam.copper_fragment_cascade_actual_input_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_actual_input_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_selected_input_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_selected_input_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_replay_output_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_replay_output_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_selection_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_selection_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_closure_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_closure_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_mass_energy_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_mass_energy_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_baryon_mismatch =
            minibeam_event_counts_host[
                minibeam_fragment_baryon_mismatch_slot];
        result.minibeam.beamline_removed_energy_MeV =
            result.beamline_removed_energy_MeV;
        for (std::size_t history = 0; history < number_of_histories; ++history) {
            const auto energy = static_cast<double>(
                beamline_primary_survivor_host[history]);
            if (!(energy > 0.0)) continue;
            result.minibeam.energy_sum_MeV += energy;
            result.minibeam.energy_squared_sum_MeV2 += energy * energy;
            if (beamline_removed_host[history] == 0.0F) {
                result.minibeam.direct_air_primary_energy_MeV += energy;
            } else {
                result.minibeam.copper_touched_primary_energy_MeV += energy;
                const auto bin = std::min(
                    static_cast<std::size_t>(energy / 200.0),
                    MinibeamDiagnostics::touched_energy_bin_count - 1);
                ++result.minibeam
                      .copper_touched_primary_energy_histogram[bin];
            }
        }
        if (!minibeam_phase_space_host.empty()) {
            result.minibeam_phase_space_records =
                std::move(minibeam_phase_space_host);
            for (const auto& record : result.minibeam_phase_space_records) {
                if (!record.valid) continue;
                result.minibeam.x_sum_mm += record.x_mm;
                result.minibeam.x_squared_sum_mm2 +=
                    static_cast<double>(record.x_mm) * record.x_mm;
                result.minibeam.y_sum_mm += record.y_mm;
                result.minibeam.y_squared_sum_mm2 +=
                    static_cast<double>(record.y_mm) * record.y_mm;
                result.minibeam.direction_x_sum += record.direction_x;
                result.minibeam.direction_x_squared_sum +=
                    static_cast<double>(record.direction_x) *
                    record.direction_x;
                result.minibeam.direction_y_sum += record.direction_y;
                result.minibeam.direction_y_squared_sum +=
                    static_cast<double>(record.direction_y) *
                    record.direction_y;
            }
        }
        result.minibeam_water_primary_plane_records =
            std::move(minibeam_water_primary_plane_records_host);
        result.minibeam_fragment_phase_space_records =
            std::move(minibeam_fragment_phase_space_host);
    }
    result.secondary_queue_overflow = overflow_count_host;
    result.secondary_queue_overflow_energy_MeV = static_cast<double>(overflow_energy_host);
    // Copy the current shared species/grid measurements.
    for (std::size_t i = 0; i < kIonSpeciesEnergySlots; ++i) {
        result.ion_species_transport_ledger_MeV[i] =
            static_cast<double>(ion_species_energy_host[i]);
    }
    result.in_grid_deposited_energy_MeV = grid_deposited_in_host;
    result.outside_grid_deposited_energy_MeV = grid_deposited_out_host;
    result.ion_species_terminal_reason_counts = ion_species_terminal_host;

    // Aggregate the flat Schneider device schema into named diagnostics.
    {
        auto& sch = result.schneider_diagnostics;
        const auto slot = [&](SchneiderDiagSlot s) -> std::uint64_t {
            return schneider_diag_host[static_cast<std::uint32_t>(s)];
        };
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        sch.primary_rate_queries = slot(SchneiderDiagSlot::PrimaryRateQueries);
        sch.primary_hazards = slot(SchneiderDiagSlot::PrimaryHazards);
        sch.primary_exact_target_hits = slot(SchneiderDiagSlot::PrimaryExactTargetHits);
        sch.primary_missing_projectile = slot(SchneiderDiagSlot::PrimaryMissingProjectile);
        sch.primary_missing_target = slot(SchneiderDiagSlot::PrimaryMissingTarget);
        sch.primary_below_domain = slot(SchneiderDiagSlot::PrimaryBelowDomain);
        sch.primary_above_domain = slot(SchneiderDiagSlot::PrimaryAboveDomain);
        sch.primary_energy_gap_misses = slot(SchneiderDiagSlot::PrimaryEnergyGapMisses);
        sch.primary_empty_nodes = slot(SchneiderDiagSlot::PrimaryEmptyNodes);
        sch.primary_events_replayed = slot(SchneiderDiagSlot::PrimaryEventsReplayed);
        sch.primary_charged_products_born = slot(SchneiderDiagSlot::PrimaryChargedBorn);
        sch.primary_charged_products_queued = slot(SchneiderDiagSlot::PrimaryChargedQueued);
        sch.primary_charged_cutoff_kills = slot(SchneiderDiagSlot::PrimaryChargedCutoffKills);
        sch.primary_be6_kills = slot(SchneiderDiagSlot::PrimaryBe6Kills);
        sch.primary_queue_overflows = slot(SchneiderDiagSlot::PrimaryQueueOverflows);
        sch.secondary_tracks_started = slot(SchneiderDiagSlot::SecondaryTracksStarted);
        sch.secondary_steps = slot(SchneiderDiagSlot::SecondarySteps);
        sch.secondary_fe_c12_steps = slot(SchneiderDiagSlot::SecondaryFeC12Steps);
        sch.secondary_fe_he4_steps = slot(SchneiderDiagSlot::SecondaryFeHe4Steps);
        sch.secondary_fe_pdt_steps = slot(SchneiderDiagSlot::SecondaryFePdtSteps);
        sch.secondary_fe_other_charged_steps =
            slot(SchneiderDiagSlot::SecondaryFeOtherChargedSteps);
        sch.secondary_highland_steps =
            slot(SchneiderDiagSlot::SecondaryHighlandSteps);
        sch.secondary_rate_queries = slot(SchneiderDiagSlot::SecondaryRateQueries);
        sch.secondary_hazards = slot(SchneiderDiagSlot::SecondaryHazards);
        sch.secondary_exact_target_hits = slot(SchneiderDiagSlot::SecondaryExactTargetHits);
        sch.secondary_missing_projectile = slot(SchneiderDiagSlot::SecondaryMissingProjectile);
        sch.secondary_missing_target = slot(SchneiderDiagSlot::SecondaryMissingTarget);
        sch.secondary_below_domain = slot(SchneiderDiagSlot::SecondaryBelowDomain);
        sch.secondary_above_domain = slot(SchneiderDiagSlot::SecondaryAboveDomain);
        sch.secondary_energy_gap_misses = slot(SchneiderDiagSlot::SecondaryEnergyGapMisses);
        sch.secondary_empty_nodes = slot(SchneiderDiagSlot::SecondaryEmptyNodes);
        sch.secondary_events_replayed = slot(SchneiderDiagSlot::SecondaryEventsReplayed);
        sch.secondary_charged_products_born = slot(SchneiderDiagSlot::SecondaryChargedBorn);
        sch.secondary_charged_products_queued = slot(SchneiderDiagSlot::SecondaryChargedQueued);
        sch.secondary_charged_cutoff_kills = slot(SchneiderDiagSlot::SecondaryChargedCutoffKills);
        sch.secondary_be6_kills = slot(SchneiderDiagSlot::SecondaryBe6Kills);
        sch.secondary_queue_overflows = slot(SchneiderDiagSlot::SecondaryQueueOverflows);
        sch.secondary_stopped_before_replay = slot(SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
        sch.secondary_post_em_null_collisions = slot(SchneiderDiagSlot::SecondaryPostEmNullCollisions);
        sch.secondary_post_em_null_energy_MeV = fslot(SchneiderFloatSlot::PostEmNullEnergy);
        sch.primary_post_em_null_collisions = slot(SchneiderDiagSlot::PrimaryPostEmNullCollisions);
        // Exact fixed-point secondary sub-ledger (micro-MeV -> MeV). These
        // fields were historically always zero (never assigned); on the
        // Schneider path they now accumulate every secondary deposit/escape
        // at the same sites as the per-history arrays.
        result.secondary_deposited_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryDepositedMicroMeV)) * 1.0e-6;
        result.secondary_escaped_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryEscapedMicroMeV)) * 1.0e-6;
        sch.be6_topas_compat_kills = slot(SchneiderDiagSlot::Be6TopasCompatKills);
        sch.unsupported_projectile_steps = slot(SchneiderDiagSlot::UnsupportedProjectileSteps);
        sch.unsupported_projectile_tracks = slot(SchneiderDiagSlot::UnsupportedProjectileTracks);
        sch.unsupported_be6_tracks = slot(SchneiderDiagSlot::UnsupportedBe6Tracks);
        sch.unsupported_projectile_birth_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV)) * 1.0e-6;
        // Dropped counts come from the log-count buffers (device-side
        // overflow tallies), not the diag slots (unused for these two).
        sch.miss_log_dropped = schneider_miss_counts_host[1];
        sch.unsupported_log_dropped = schneider_track_counts_host[1];
        sch.unsupported_targets = slot(SchneiderDiagSlot::UnsupportedTargets);
        sch.queue_overflows = slot(SchneiderDiagSlot::QueueOverflows);
        sch.primary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::PrimaryMismatchSum);
        sch.primary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::PrimaryMismatchMax);
        sch.secondary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::SecondaryMismatchSum);
        sch.secondary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::SecondaryMismatchMax);
        sch.lookup_failure_energy_MeV = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        sch.be6_kill_energy_MeV = fslot(SchneiderFloatSlot::Be6KillEnergy);
        sch.neutral_product_kinetic_MeV = fslot(SchneiderFloatSlot::NeutralProductKinetic);
        sch.reaction_q_residual_MeV = fslot(SchneiderFloatSlot::ReactionQResidual);
    }

    // Charged birth energy and nuclear-vertex itemization
    result.energy_ledger.E_transported_secondaries = result.queued_secondary_energy_MeV;
    // Split Schneider nuclear-vertex ledger, accumulated on device from
    // actually executed vertices (informational itemization; the legacy
    // untracked sink behavior above is unchanged this step).
    {
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        result.energy_ledger.E_be6_kill = fslot(SchneiderFloatSlot::Be6KillEnergy);
        result.energy_ledger.E_lookup_failure = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        result.energy_ledger.E_neutral_product_kinetic =
            fslot(SchneiderFloatSlot::NeutralProductKinetic);
        result.energy_ledger.E_reaction_q_residual = fslot(SchneiderFloatSlot::ReactionQResidual);
        result.energy_ledger.E_unsupported_charged =
            fslot(SchneiderFloatSlot::UnsupportedProductEnergy);
        result.energy_ledger.E_out_of_domain = fslot(SchneiderFloatSlot::OutOfDomainEnergy);
        if (schneider_diag_device != nullptr) {
            result.energy_ledger.E_transported_secondaries =
                fslot(SchneiderFloatSlot::SecondaryTransportBirthEnergy);
        }
    }

    if (config.quality_reject_any_queue_overflow && overflow_count_host > 0) {
        throw std::runtime_error("Secondary particle queue overflow detected: discarded " +
                                 std::to_string(overflow_count_host) + " particles (" +
                                 std::to_string(overflow_energy_host) + " MeV). Shard must be split and rerun with fewer particles.");
    }

    result.nuclear_interactions =
        schneider_inelastic_host;
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

    if (!copper_diagnostic_path.empty())
        write_copper_history_diagnostic(copper_diagnostic_path,copper_diagnostic_host,
            beamline_primary_survivor_host,deposited_host,steps_host,copper_history_mask_host);
    if(!copper_capture_path.empty()) {
        const std::filesystem::path directory(copper_capture_path);
        std::filesystem::create_directories(directory);
        write_copper_capture_records(directory/"input.bin",copper_diagnostic_host);
        write_copper_capture_records(directory/"water_primary.bin",copper_capture_entry_host);
        write_copper_capture_records(directory/"removed.bin",beamline_removed_host);
        write_copper_capture_records(directory/"air_loss.bin",beamline_air_loss_host);
        write_copper_capture_records(directory/"water_charged.bin",copper_capture_charged_host);
        std::cout << "[copper-response-capture] source_histories=" << number_of_histories
                  << " charged_at_water=" << copper_capture_charged_host.size()
                  << " entry_bytes=" << sizeof(PrimaryMediumEntryState)
                  << " charged_bytes=" << sizeof(SecondaryParticle)
                  << '\n';
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.electron_kernel_seconds = electron_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;