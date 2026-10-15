// Shared finalize phase, included inside transport_pipeline<Pipeline>.
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

        mem_tracker.free(unified_energy_index);mem_tracker.free(unified_materials);mem_tracker.free(unified_species);mem_tracker.free(unified_records);
        mem_tracker.free(unified_nodes);mem_tracker.free(unified_segments);mem_tracker.free(unified_audit);

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

    free_device(dose_device);
    free_device(primary_terminal_counts_device);
    free_device(in_fov_dose_device);
    free_device(voxel_dose_device);

    free_device(deposited_device);
    free_device(secondary_queue_device);
    free_device(secondary_count_device);
    free_device(ion_species_energy_device);
    free_device(ion_species_terminal_device);
    free_device(untracked_nuclear_device);
    free_device(other_terminal_energy_device);
    free_device(cutoff_stopped_energy_device);

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
    free_device(schneider_inelastic_device);
    free_device(schneider_diag_device);
    free_device(schneider_float_device);
    free_device(schneider_miss_device);
    free_device(schneider_track_log_device);
    free_device(schneider_miss_count_device);
    free_device(schneider_track_count_device);

    TransportResult result;
    if constexpr (std::is_same_v<Pipeline, pipelines::ProtonBroadbeam>)
        result.primary_first_order_delta_enabled=broadbeam_ct_primary_specialize;
    if constexpr (std::is_same_v<Pipeline, pipelines::CarbonBroadbeam>) {
        result.carbon_secondary_first_order_delta_enabled=broadbeam_ct_secondary_specialize;
    }
    if constexpr (BroadBeam) {
        if constexpr (std::is_same_v<Pipeline, pipelines::ProtonBroadbeam>)
            result.ct_primary_first_order_restricted_enabled=broadbeam_ct_primary_specialize;
        result.ct_secondary_first_order_restricted_enabled=broadbeam_ct_secondary_specialize;
    }
    if constexpr(BroadBeam) result.ct_broadbeam_step_cap_enabled=enable_ct_grid;

    result.primary_elastic_interactions=elastic_counts_host[0];
    result.secondary_elastic_interactions=elastic_counts_host[1];
    result.elastic_post_em_null_collisions=elastic_counts_host[9]+elastic_counts_host[10];
    result.elastic_local_deposited_energy_MeV=elastic_energy_host[0];
    result.elastic_queued_charged_energy_MeV=elastic_energy_host[1];
    result.elastic_queue_overflow_energy_MeV=elastic_energy_host[2];
    result.elastic_queue_overflow=elastic_counts_host[11];
    result.schneider_miss_log = std::move(schneider_miss_host);

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
    result.backend += "+fp32-dose";
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring+scorer-decoupled";
    }

    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);

    result.in_fov_deposited_energy_MeV = std::move(in_fov_dose_host);

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

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;
