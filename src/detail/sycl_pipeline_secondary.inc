// Include contract: follows primary inside transport_pipeline<Pipeline>.
// Drain charged-particle generations, including children appended by earlier
// secondary generations. Grouping changes index order, never particle identity;
// checkpoints retain pending dose, nuclear/ionization clocks and RNG counters.
// Generation bounds distinguish already scheduled particles from new births.
    double secondary_kernel_seconds = 0.0;
    const bool segment_secondaries = config.secondary_step_chunking;
    constexpr unsigned secondary_tail_threshold = Pipeline::secondary_tail_threshold;
    // Reorder indices only: RNG streams and parent histories belong to particles.
    const bool group_secondaries = config.secondary_species_grouping &&
        enable_inelastic && enable_secondary_transport;
    std::uint32_t* secondary_order = nullptr;
    std::uint32_t* secondary_group_counts = nullptr;
    std::uint32_t* secondary_group_cursors = nullptr;
    double secondary_group_seconds = 0.0;
    if (group_secondaries && secondary_queue_device) {
        secondary_order = mem_tracker.allocate<std::uint32_t>(max_secondaries);
        secondary_group_counts =
            mem_tracker.allocate<std::uint32_t>(kSecondaryGroupBucketCount);
        secondary_group_cursors =
            mem_tracker.allocate<std::uint32_t>(kSecondaryGroupBucketCount);
        if (!secondary_order || !secondary_group_counts || !secondary_group_cursors)
            throw std::bad_alloc();
    }
    std::cout << "[secondary-schedule] group=" << group_secondaries
              << "; particle state and RNG identities preserved\n";

    if (enable_secondary_transport &&
        secondary_count_device != nullptr && secondary_queue_device != nullptr) {
        uint32_t secondary_count_host = 0;
        queue.copy(secondary_count_device, &secondary_count_host, 1).wait_and_throw();
        if (secondary_count_host > max_secondaries) {
            secondary_count_host = static_cast<uint32_t>(max_secondaries);
        }
        if (secondary_count_host > 0) {
            std::uint32_t generation_begin = 0U;
            std::uint32_t generation_end = secondary_count_host;
            while (generation_begin < generation_end) {
                const auto batch_begin = generation_begin;
                if (group_secondaries) {
                    const auto grouping_start = std::chrono::steady_clock::now();
                    queue.fill(secondary_group_counts, 0u,
                               kSecondaryGroupBucketCount).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const auto frag = secondary_queue_device[generation_begin + id[0]];
                            const unsigned bucket =
                                secondary_group_bucket(frag.z, frag.a, frag.energy_MeV);
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                count(secondary_group_counts[bucket]);
                            count.fetch_add(1);
                        }).wait_and_throw();
                    queue.single_task([=]() {
                        unsigned total = 0;
                        for (unsigned k = 0; k < kSecondaryGroupBucketCount; ++k) {
                            secondary_group_cursors[k] = total;
                            total += secondary_group_counts[k];
                        }
                    }).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const unsigned index = generation_begin + id[0];
                            const auto frag = secondary_queue_device[index];
                            const unsigned bucket =
                                secondary_group_bucket(frag.z, frag.a, frag.energy_MeV);
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                cursor(secondary_group_cursors[bucket]);
                            secondary_order[cursor.fetch_add(1)] = index;
                        }).wait_and_throw();

                    secondary_group_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - grouping_start).count();
                }
                const unsigned resume_capacity = generation_end - generation_begin;
                const unsigned state_capacity = segment_secondaries ? resume_capacity : 0;
                auto* resume_states = mem_tracker.allocate<SecondaryResumeState>(state_capacity);
                auto* resume_ready = mem_tracker.allocate<unsigned>(state_capacity);
                auto* active_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* next_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* keep = mem_tracker.allocate<unsigned>(state_capacity);
                auto* ranks = mem_tracker.allocate<unsigned>(state_capacity);
                auto* block_counts = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                auto* block_offsets = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                auto* active_count = mem_tracker.allocate<unsigned>(
                    segment_secondaries ? 1u : 0u);
                if (segment_secondaries) {
                    if (!resume_states || !resume_ready || !active_order || !next_order ||
                        !keep || !ranks || !block_counts || !block_offsets || !active_count)
                        throw std::runtime_error("Secondary continuation allocation failed; reduce histories per shard");
                    queue.fill(resume_ready, 0u, resume_capacity).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(resume_capacity), [=](sycl::id<1> i) {
                        active_order[i[0]] = i[0];
                    }).wait_and_throw();
                    std::cout << "[segmented-secondary] step_limit=" << Pipeline::secondary_segment_steps << " tail_threshold="
                              << secondary_tail_threshold << " state_bytes="
                              << sizeof(SecondaryResumeState) << " capacity=" << resume_capacity << "\n";
                }
                unsigned resume_active = resume_capacity, segment_rounds = 0;
                unsigned tail_rounds = 0;
                double compact_seconds = 0, generation_kernel_seconds = 0, generation_tail_seconds = 0;
                while (resume_active) {
                const bool finish_tail = !segment_secondaries || resume_active < secondary_tail_threshold;
                const auto launch_secondary_round = [&](auto production_path_tag,
                                                        unsigned active_size) {
                constexpr bool kProductionSecondaryPath =
                    decltype(production_path_tag)::value;
                constexpr bool kCarbonCtPointReuse = kProductionSecondaryPath &&
                    std::is_same_v<Pipeline, pipelines::CarbonBroadbeam>;
                constexpr bool kCachedElasticTotals = kCarbonCtPointReuse ||
                    (std::is_same_v<Pipeline,pipelines::CarbonMinibeam>);
                constexpr unsigned kSliceSegmentSteps = Pipeline::secondary_segment_steps;
                return queue.submit([&](sycl::handler& cgh) {
                const auto secondary_kernel =
                    [=]
                    (sycl::id<1> item_id) {
                        const unsigned active_pos = item_id[0];
                        const unsigned state_idx = segment_secondaries ? active_order[active_pos] : active_pos;
                        const bool resumed = segment_secondaries && resume_ready[state_idx] != 0;
                        if (segment_secondaries) keep[active_pos] = 0;
                        const auto sec_idx = group_secondaries ? secondary_order[state_idx]
                            : generation_begin + state_idx;
                        const auto frag = secondary_queue_device[sec_idx];
                        const int transport_z = [&] {
                            return static_cast<int>(frag.z);
                        }();
                        const int transport_a = [&] {
                            return static_cast<int>(frag.a);
                        }();
                        if (transport_z <= 0 || transport_a <= 0) return;

                        if (frag.energy_MeV <= energy_cutoff_MeV) {
                            const auto ledger_species_idx = carbon::get_charged_species_idx(
                                static_cast<int>(transport_z), static_cast<int>(transport_a));
                            ion_species_energy_add_device(
                                ion_species_energy_device, ledger_species_idx, 5U,
                                frag.energy_MeV);
                            ion_species_terminal_increment_device(
                                ion_species_terminal_device, ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    IonSpeciesLedgerSchema::initial_below_cutoff));
                            const auto bin_z = static_cast<int>(frag.pos_z_mm * inverse_depth_bin_width_mm);
                            if (bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dose(dose_device[bin_z]);
                                atomic_dose.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
                            }
                            if ((kProductionSecondaryPath ? true : enable_voxel_scoring)) {
                                const auto bin_x = static_cast<int>((frag.pos_x_mm - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto bin_y = static_cast<int>((frag.pos_y_mm - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                if (bin_x >= 0 && bin_x < static_cast<int>(voxel_bins_x) &&
                                    bin_y >= 0 && bin_y < static_cast<int>(voxel_bins_y) &&
                                    bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                    const auto cur_voxel = (bin_z * static_cast<int>(voxel_bins_y) + bin_y) * static_cast<int>(voxel_bins_x) + bin_x;
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_vox(voxel_dose_device[cur_voxel]);
                                    atomic_vox.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));

                                    if (in_fov_dose_device != nullptr) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[bin_z]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
                                        ion_species_energy_add_device(
                                            ion_species_energy_device, ledger_species_idx,
                                            6U, frag.energy_MeV);
                                    }
                                }
                            }
                            if (deposited_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(deposited_device[frag.parent_history]);
                                atomic_dep.fetch_add(frag.energy_MeV);
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                    frag.energy_MeV);
                                {
                                    const auto sqx = static_cast<int>(
                                        (frag.pos_x_mm - voxel_min_x_mm) *
                                        inverse_voxel_size_x_mm);
                                    const auto sqy = static_cast<int>(
                                        (frag.pos_y_mm - voxel_min_y_mm) *
                                        inverse_voxel_size_y_mm);
                                    const auto sqz = static_cast<int>(
                                        frag.pos_z_mm * inverse_depth_bin_width_mm);
                                    const bool sq_in_grid =
                                        (kProductionSecondaryPath ? true : enable_voxel_scoring) && sqx >= 0 &&
                                        sqx < static_cast<int>(voxel_bins_x) &&
                                        sqy >= 0 &&
                                        sqy < static_cast<int>(voxel_bins_y) &&
                                        sqz >= 0 &&
                                        sqz < static_cast<int>(number_of_bins);
                                    grid_deposit_split_device(
                                        grid_deposited_in_device,
                                        grid_deposited_out_device, sq_in_grid,
                                        frag.energy_MeV);
                                }
                            }
                            return;
                        }

                        const auto frag_a = static_cast<float>(transport_a);
                        const auto frag_inv_a = 1.0F / frag_a;
                        const auto charged_sp_idx = [&] {
                            return carbon::get_charged_species_idx(transport_z, transport_a);
                        }();
                        const auto ledger_species_idx = charged_sp_idx;
                        const int recoil_sp_idx=((kProductionSecondaryPath && !BroadBeam) ? false : use_all_elastic)?recoil_stopping.projectile(transport_z,transport_a):-1;
                        const bool generic_recoil=
                            charged_sp_idx<0 && recoil_sp_idx>=0;
                        if (charged_sp_idx < 0 && !generic_recoil) {
                            if (untracked_nuclear_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_untracked(
                                        untracked_nuclear_device[frag.parent_history]);
                                atomic_untracked.fetch_add(sycl::fmax(0.0F, frag.energy_MeV));
                            }
                            return;
                        }
                        bool sec_terminal_recorded = false;
                        bool unified_secondary_escaped_ct = false;
                        ContinuousSpeciesTrackTally continuous_species_tally;
                        // Broadbeam common-step ledgers reduced per segment; flushed
                        // before suspend/terminal handling (resume state unchanged).
                        float sec_dep_pending_MeV = 0.0F;
                        std::uint64_t sec_diag_dep_pending_uMeV = 0;
                        double sec_grid_in_pending_MeV = 0.0, sec_grid_out_pending_MeV = 0.0;
                        double he4_audit[4]{};
                        float sec_e = frag.energy_MeV;
                        float sec_x = frag.pos_x_mm;
                        float sec_y = frag.pos_y_mm;
                        float sec_z = frag.pos_z_mm;
                        float sec_dx = frag.dir_x;
                        float sec_dy = frag.dir_y;
                        float sec_dz = frag.dir_z;

                        int pending_sec_bin = -1;
                        int pending_sec_voxel = -1;
                        float pending_sec_depth_MeV = 0.0F;
                        float pending_sec_voxel_MeV = 0.0F;

                        const bool unified_secondary = !generic_recoil;
                        const int unified_secondary_species=unified_secondary
                            ? ([&] {
                                return unified_device.species_index(
                                    transport_z,transport_a);
                              }())
                            : -1;
                        UnifiedEmClock unified_secondary_clock;
                        std::uint64_t unified_secondary_counter=0;
                        // Diagnostic species profile: 20 slots x {steps,
                        // short-range steps, deposited uMeV}. Zero production
                        // impact (compiled out when profiling is off).

                        UnifiedEmState unified_secondary_state;
                        auto unified_secondary_uniform=[&](){return (rng::random_u32(2026,frag.rng_stream,unified_secondary_counter++,121)>>8)*0x1p-24f;};
                        auto unified_secondary_count=[&](int index,std::uint64_t count=1){
                            if(count==0)return; // zero-increment diagnostics must not touch globals
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(unified_audit[static_cast<std::size_t>(index)*kUnifiedEmAuditShards+active_pos%kUnifiedEmAuditShards]);a.fetch_add(count);
                        };
                        float nuclear_tau_remaining = 0.0F;
                        bool nuclear_tau_active = false;
                        std::uint32_t nuclear_tau_rng_step = 0;
                        uint32_t sec_steps = 0;
                        const uint32_t kSecondaryMaxSteps = 30000U;
                        // Fix C4: one fMinimal track per secondary (persists
                        // across macro steps like the primary state; the old
                        // per-macro-step rebuild pinned tlimit at 1e10, so
                        // secondaries were never MSC-limited at low E).
                        UrbanV2TrackState sec_urban_persist{};
                        bool sec_urban_born = true;
                        bool sec_global_urban_boundary=true;
                        const bool sec_global_urban=!BroadBeam && global_urban;
                        const int sec_global_urban_ion=sec_global_urban?
                            urban_device.projectile(transport_z,transport_a):-1;
                        std::uint64_t local_sec_rate_queries = 0;
                        std::uint64_t local_sec_steps = 0;
                        const int schneider_reg_idx =
                            enable_inelastic
                                ? secondary_projectile_lut_index_device(
                                      schneider_ct_device_ctx.sec_proj_keys,
                                      schneider_ct_device_ctx.sec_num_projectiles,
                                      transport_z, transport_a)
                                : -1;
                        if(!resumed){
                        // A track that reaches the stepping loop has actually
                        // started charged transport: count it and itemize its
                        // birth kinetic energy (not a config-switch inference).
                        schneider_diag_increment_device(
                            schneider_diag_device,
                            SchneiderDiagSlot::SecondaryTracksStarted);
                        schneider_float_add_device(
                            schneider_float_device,
                            SchneiderFloatSlot::SecondaryTransportBirthEnergy,
                            sycl::fmax(0.0F, frag.energy_MeV));
                        // Per-track unsupported-projectile accounting (once per
                        // track, not per step): generation-eligible tracks whose
                        // (Z/A) has no secondary rate-table registry entry.
                        // The per-step evaluation counter below
                        // (UnsupportedProjectileSteps) is diagnostic only and
                        // must NOT drive coverage gates.
                        // v3: registry lookup over the uploaded bundle-ordered
                        // keys (any (Z,A) in the bundle is supported).

                        if (enable_inelastic &&
                            schneider_ct_device_ctx.uses_cinel03() &&
                            frag.generation < max_secondary_inelastic_generations &&
                            schneider_reg_idx < 0) {
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::UnsupportedProjectileTracks);
                            if (transport_z == 4 && transport_a == 6) {
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::UnsupportedBe6Tracks);
                            }
                            schneider_energy_add_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV,
                                sycl::fmax(0.0F, frag.energy_MeV));
                            // Per-(Z/A) census record for the coverage audit.
                            schneider_log_unsupported_track_device(
                                schneider_track_log_device, schneider_track_count_device,
                                kSchneiderTrackLogCap,
                                transport_z, transport_a, frag.generation,
                                sycl::fmax(0.0F, frag.energy_MeV),
                                frag.pos_x_mm, frag.pos_y_mm, frag.pos_z_mm);
                        }
                        }
                        if(resumed){
                            const auto saved=resume_states[state_idx];
                            sec_terminal_recorded=saved.sec_terminal_recorded;
                            unified_secondary_escaped_ct=saved.unified_secondary_escaped_ct;
                            continuous_species_tally=saved.continuous_species_tally;
                            sec_e=saved.sec_e;
                            sec_x=saved.sec_x;
                            sec_y=saved.sec_y;
                            sec_z=saved.sec_z;
                            sec_dx=saved.sec_dx;
                            sec_dy=saved.sec_dy;
                            sec_dz=saved.sec_dz;
                            pending_sec_depth_MeV=saved.pending_sec_depth_MeV;
                            pending_sec_voxel_MeV=saved.pending_sec_voxel_MeV;
                            pending_sec_bin=saved.pending_sec_bin;
                            pending_sec_voxel=saved.pending_sec_voxel;
                            unified_secondary_counter=saved.unified_secondary_counter;
                            unified_secondary_clock=saved.urban_ionization_clock;
                            unified_secondary_state=saved.unified_secondary_state;
                            unified_secondary_state.tables=&unified_device;
                            nuclear_tau_remaining=saved.nuclear_tau_remaining;
                            nuclear_tau_active=saved.nuclear_tau_active;
                            nuclear_tau_rng_step=saved.nuclear_tau_rng_step;
                            sec_steps=saved.sec_steps;
                            if constexpr (!BroadBeam) {
                            sec_urban_persist.tlimit_mm =
                                saved.urban.sec_urban_tlimit_mm;
                            sec_urban_born = saved.urban.sec_urban_born;
                            if(sec_global_urban)sec_urban_persist.first_step=saved.urban.sec_urban_first_step;
                            sec_global_urban_boundary=saved.urban.sec_global_urban_boundary;
                            }
                            local_sec_rate_queries=saved.local_sec_rate_queries;
                            local_sec_steps=saved.local_sec_steps;

                                for(int j=0;j<4;++j)he4_audit[j]=saved.he4_audit[j];

                        }
                        unsigned segment_steps=0;bool segment_paused=false;
                        while (sec_e > energy_cutoff_MeV && sec_z >= 0.0F && sec_z < phantom_length_mm &&
                               sec_steps < kSecondaryMaxSteps) {
                            if(!finish_tail && segment_steps>=kSliceSegmentSteps){segment_paused=true;break;}
                            ++segment_steps;
                            if(sec_global_urban && !(kProductionSecondaryPath ? true : enable_ct_grid) &&
                               urban_global_outside_water({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},
                                   urban_water_half_width_mm,phantom_length_mm)) {
                                // This escape flag also suppresses terminal deposition at an exact water face.
                                unified_secondary_escaped_ct=true;break;
                            }
                            // A surface belongs to the cell entered by the
                            // track. floor(z/dz) already has the right forward
                            // convention; one representable step upstream gives
                            // the corresponding convention for backward tracks.
                            const auto directed_sec_z = sec_dz < -1.0e-6F
                                ? sycl::nextafter(
                                      sec_z,
                                      -std::numeric_limits<float>::infinity())
                                : sec_z;
                            const auto bin_z = static_cast<int>(sycl::floor(
                                directed_sec_z * inverse_depth_bin_width_mm));

                            if (bin_z < 0 || bin_z >= static_cast<int>(number_of_bins)) {
                                if(sec_global_urban)unified_secondary_escaped_ct=true;
                                break;
                            }

                            const auto sec_e_u = sec_e * frag_inv_a;
                            float sec_local_density_g_per_cm3 = water_density_g_per_cm3;
                            std::uint8_t sec_ct_material = 2U;
                            bool sec_in_ct = false;
                            if ((kProductionSecondaryPath ? true : enable_ct_grid)) {
                                sec_in_ct = ct_sample(
                                    sec_x, sec_y, sec_z, ct_origin_x, ct_origin_y,
                                    ct_origin_z, ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                    ct_nx, ct_ny, ct_nz, ct_density_device,
                                    ct_material_device, sec_local_density_g_per_cm3,
                                    sec_ct_material, sec_dx, sec_dy, sec_dz);
                            }
                            // The unified package has no exterior material. Leaving the
                            // CT volume is a charged-particle escape, not a lookup error.
                            if ((unified_secondary || sec_global_urban) && (kProductionSecondaryPath ? true : enable_ct_grid) && !sec_in_ct) {
                                unified_secondary_escaped_ct = true;
                                break;
                            }
                            auto recoil_linear_sp = [&](float energy_mevu) {
                                const float value = recoil_stopping.stopping(
                                    recoil_sp_idx, sec_in_ct ? sec_ct_material : 25, energy_mevu);
                                if (!(value > 0.0F)) {
                                    for (const int slot : {2, 6})
                                        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                            elastic_audit[slot]).fetch_add(1);
                                    return -1.0F;
                                }
                                return value * sec_local_density_g_per_cm3;
                            };
                            float recoil_sp_at_start = 0.0F;
                            if (generic_recoil) {
                                recoil_sp_at_start = recoil_linear_sp(sec_e_u);
                                if (!(recoil_sp_at_start > 1.0e-6F)) break;
                            }

                            float sec_step_mm = maximum_step_mm;
                            UnifiedEmPointStep unified_secondary_pre;
                            UnifiedEmPoint unified_secondary_point;
                            float unified_secondary_rate=0,unified_secondary_distance=std::numeric_limits<float>::infinity();
                            if(unified_secondary){
                                const int section=(kProductionSecondaryPath ? true : enable_ct_grid)?(sec_in_ct?int(sec_ct_material):-2):-1;
                                if(!unified_secondary_state.valid || unified_secondary_state.section!=section ||
                                   unified_secondary_state.density!=sec_local_density_g_per_cm3)
                                    unified_secondary_state=unified_device.select(section,sec_local_density_g_per_cm3,unified_secondary_species);
                                if(!unified_secondary_state.covers(sec_e)){
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,3,section,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,0);
                                    unified_secondary_count(0);break;}
                                if constexpr(kCarbonCtPointReuse) {
                                    // Same-step view only; never retained across continuation.
                                    unified_secondary_point=unified_secondary_state.point();
                                    const auto point_pre=unified_secondary_point.template prepare<
                                        true, true, true>(sec_e);
                                    unified_secondary_pre=point_pre;
                                } else unified_secondary_pre=unified_secondary_state.prepare(sec_e);
                                if(sec_global_urban) {
                                    unified_secondary_rate=unified_secondary_clock.update(unified_secondary_state,sec_e,
                                        unified_secondary_pre.factor);
                                    if(!unified_secondary_clock.clock.active)
                                        unified_secondary_clock.clock.arm(-sycl::log(sycl::fmax(unified_secondary_uniform(),0x1p-24f)));
                                    unified_secondary_distance=unified_secondary_clock.clock.distance(unified_secondary_rate);
                                }
                                sec_step_mm=sycl::fmin(unified_secondary_state.step(unified_secondary_pre),unified_secondary_distance);
                                // Bound combined mean loss, not the old restricted range alone.
                                const float delta_sp=unified_secondary_pre.delta_stopping;
                                if(delta_sp>0) sec_step_mm=sycl::fmin(sec_step_mm,maximum_mean_loss_fraction*sec_e/sycl::fmax(1e-12f,delta_sp+unified_secondary_pre.stopping));
                            }

                            if constexpr(BroadBeam) if((kProductionSecondaryPath ? true : enable_ct_grid) && !sec_global_urban)
                                sec_step_mm=sycl::fmin(sec_step_mm,maximum_step_mm);

                            bool depth_boundary_limited = false;
                            if (!sec_global_urban && sec_dz > 1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z + 1) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            } else if (!sec_global_urban && sec_dz < -1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            }
                            CtFaceClampResult sec_face_clamp{sec_step_mm,false,0};
                            if (!sec_global_urban && (kProductionSecondaryPath ? true : enable_ct_grid) && sec_in_ct && (kProductionSecondaryPath ? true : ct_secondary_exact_faces)) {
                                sec_face_clamp = clamp_step_to_ct_faces_exact(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, ct_origin_x, ct_origin_y, ct_origin_z,
                                    ct_spacing_x, ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz);
                                sec_step_mm = sec_face_clamp.step_mm;
                            } else if (!sec_global_urban && (kProductionSecondaryPath ? true : enable_ct_grid) && sec_in_ct) {
                                sec_step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, ct_origin_x, ct_origin_y, ct_origin_z,
                                    ct_spacing_x, ct_spacing_y, ct_spacing_z, ct_nx, ct_ny,
                                    ct_nz, ct_density_device, ct_material_device,
                                    sec_local_density_g_per_cm3, sec_ct_material,
                                    (kProductionSecondaryPath ? true : ct_skip_homogeneous_face_clamp));
                            }
                            // Do not enlarge a real face distance to the legacy
                            // minimum step: that would cross the material again.
                            if (!unified_secondary && !depth_boundary_limited &&
                                (!(kProductionSecondaryPath ? true : ct_secondary_exact_faces) || !sec_face_clamp.hit_face))
                                sec_step_mm = sycl::fmax(sec_step_mm, 1.0e-5F);
                            // Generic EM-only recoils take CSDA steps without
                            // fluctuations: a 5% relative-loss cap keeps
                            // stopping variation second-order while cutting
                            // the 0.5%-capped micro-step count ~10x.
                            if(generic_recoil)sec_step_mm=sycl::fmin(sec_step_mm,0.05F*sec_e/recoil_sp_at_start);
                            UrbanGlobalProposal sec_urban_proposal{};
                            CorrelatedScatteringStep sec_urban_scattering{};
                            if(sec_global_urban) {
                                const auto origin=Direction3F{ct_origin_x,ct_origin_y,ct_origin_z};
                                const auto spacing=Direction3F{ct_spacing_x,ct_spacing_y,ct_spacing_z};
                                const auto cell=sec_in_ct?urban_voxel_cell({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},origin,
                                    {origin.x+ct_nx*spacing.x,origin.y+ct_ny*spacing.y,origin.z+ct_nz*spacing.z},spacing,
                                    ct_nx,ct_ny,ct_nz):
                                    urban_global_water_cell({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},
                                        {voxel_min_x_mm,voxel_min_y_mm,0},
                                        {voxel_min_x_mm+voxel_bins_x/inverse_voxel_size_x_mm,
                                         voxel_min_y_mm+voxel_bins_y/inverse_voxel_size_y_mm,phantom_length_mm},
                                        {1/inverse_voxel_size_x_mm,1/inverse_voxel_size_y_mm,depth_bin_width_mm},
                                        voxel_bins_x,voxel_bins_y,number_of_bins,urban_water_half_width_mm);
                                const auto view=urban_device;
                                const int material=view.material(sec_in_ct?int(sec_ct_material):-1,sec_local_density_g_per_cm3);
                                if(sec_in_ct && unified_secondary)unified_secondary_point=unified_secondary_state.point();
                                sec_urban_proposal=urban_global_prepare(view,material,sec_global_urban_ion,sec_e,
                                    sycl::fmin(sec_step_mm,maximum_step_mm),
                                    {sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},cell,
                                    sec_global_urban_boundary,sec_urban_persist,2026,frag.rng_stream,sec_steps,
                                    sec_local_density_g_per_cm3,
                                    sec_in_ct && unified_secondary?&unified_secondary_point:nullptr,
                                    unified_secondary?unified_secondary_pre.factor:0.f,generic_recoil);
                                if(!sec_urban_proposal.valid) {
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,100,
                                        sec_in_ct?int(sec_ct_material):-1,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);break;
                                }
                                sec_step_mm=sec_urban_proposal.true_path;
                            }
                            bool secondary_inelastic = false;
                            bool secondary_elastic = false;
                            // A nuclear hazard is not necessarily a replayed
                            // event. Keep this separate so lookup misses and
                            // invalid records follow null-collision transport
                            // semantics and still receive normal MCS.
                            bool secondary_replay_succeeded = false;
                            std::int16_t secondary_target_z = 0;
                            // v3: set when the post-EM sampler finds no
                            // in-domain channel (lookup skipped, never queried).
                            bool skip_schneider_lookup = false;
                            // Stash for the miss log: macro total rate at
                            // sampling (density x mass, density exactly once),
                            // sampled collision distance, and section.
                            float macro_inelastic = 0.0F;
                            float macro_elastic = 0.0F;
                            float schneider_hazard_total_rate = 0.0F;
                            float schneider_hazard_step_mm = 0.0F;
                            std::uint8_t schneider_hazard_section = 255;
                            if (enable_inelastic &&
                                schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : use_unified_water)) &&
                                frag.generation < max_secondary_inelastic_generations) {
                                // Bundle-ordered registry LUT; no hardcoded isotope fallback.
                                const int proj_idx =
                                    secondary_projectile_lut_index_device(
                                              schneider_ct_device_ctx.sec_proj_keys,
                                              schneider_ct_device_ctx.sec_num_projectiles,
                                              transport_z, transport_a);
                                if (proj_idx < 0) {
                                    // Unsupported secondary projectile: explicit
                                    // per-STEP evaluation counter, never a silent
                                    // zero-rate step. (Generation-ineligible tracks
                                    // skip nuclear evaluation entirely: stage-C
                                    // transport without secondary reactions.)
                                    // Coverage gates use the per-TRACK counter
                                    // recorded at track start, not this value.
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::UnsupportedProjectileSteps);
                                } else if (schneider_ct_device_ctx.sec_partial_rates != nullptr) {
                                    const std::size_t section_id = (kProductionSecondaryPath ? false : use_unified_water) ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    ++local_sec_rate_queries;
                                    // v3: single masked computation drives hazard
                                    // AND target sampling (same partials, same
                                    // mask), without an older total/sampler alternative.
                                    {
                                        // v3: hazard from the masked total at the
                                        // step-start energy E_h. Target sampling
                                        // is deferred to the lookup site (post-EM
                                        // collision energy E_c) so the mask and
                                        // the package query share one energy.
                                        // A post-EM empty draw remains a classified null collision.
                                        const auto sec_masked = schneider_ct_device_ctx.secondary_rates(
                                            proj_idx, section_id, sec_e_u);
                                        const float sec_macro_xs =
                                            sec_local_density_g_per_cm3 * sec_masked.total;
                                        schneider_hazard_total_rate = sec_macro_xs;
                                        macro_inelastic = sec_macro_xs;
                                    }
                                }
                            }

                            // Elastic contributes to the same nuclear clock, even when
                            // the secondary inelastic generation limit is reached.
                            if(enable_inelastic &&
                               ((kProductionSecondaryPath && !BroadBeam) ? false : use_all_elastic) && !generic_recoil && (sec_in_ct || (kProductionSecondaryPath ? false : use_unified_water))) {
                                const int p=all_elastic.projectile(transport_z,transport_a);
                                const float rate=all_elastic.template species_total_rate<kCachedElasticTotals>(p,(kProductionSecondaryPath ? false : use_unified_water)?25:sec_ct_material,sec_e_u);
                                if(rate<0) {
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[7]).fetch_add(1);
                                    break;
                                }
                                macro_elastic = rate * sec_local_density_g_per_cm3;
                            }
                            // Same shared optical-depth sampler as primary transport.
                            // Retain the remaining depth across steps, material boundaries
                            // and scheduler slices; only a nuclear candidate resets it.
                            const float macro_nuclear = macro_inelastic + macro_elastic;
                            // The same shared optical-depth rule as for primaries:
                            // locate a combined-hazard candidate, then select one
                            // channel. The generation cap disables new inelastic
                            // branches only; existing charged tracks still move.
                            if (macro_nuclear > 0.0F) {
                                float u_nuclear = 1.0F;
                                if (!nuclear_tau_active) {
                                    u_nuclear = rng::uniform01(
                                        2026, frag.rng_stream, nuclear_tau_rng_step++, 13);
                                }
                                const bool collision = consume_schneider_optical_depth_segment(
                                    nuclear_tau_remaining, nuclear_tau_active,
                                    sec_step_mm, macro_nuclear, u_nuclear);
                                if (collision) {
                                    secondary_elastic = macro_elastic > 0.0F &&
                                        rng::uniform01(2026, frag.rng_stream, sec_steps, 70) *
                                            macro_nuclear < macro_elastic;
                                    secondary_inelastic = !secondary_elastic;
                                    if (secondary_inelastic) {
                                        // The inelastic audit/acceptance denominator excludes elastic.
                                        schneider_hazard_total_rate = macro_inelastic;
                                        schneider_hazard_step_mm = sec_step_mm;
                                        schneider_hazard_section = (kProductionSecondaryPath ? false : use_unified_water)
                                            ? 255U : sec_ct_material;
                                    }
                                }
                            }

                            if(sec_global_urban) {
                                if(unified_secondary) {
                                    const bool ionization_proposal=sec_step_mm>=unified_secondary_distance;
                                    unified_secondary_clock.clock.consume(sec_step_mm,unified_secondary_rate,ionization_proposal);
                                }
                                sec_urban_scattering=urban_global_finalize(sec_urban_proposal,sec_step_mm,2026,frag.rng_stream,sec_steps);
                                const int reason=urban_step_field_reason(sec_urban_scattering);
                                if(reason) {
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,110+reason,
                                        sec_in_ct?int(sec_ct_material):-1,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);break;
                                }
                                // MSC AlongStep direction precedes hadronic PostStep replay.
                                sec_dx=sec_urban_scattering.direction.x;sec_dy=sec_urban_scattering.direction.y;sec_dz=sec_urban_scattering.direction.z;
                                sec_urban_persist=sec_urban_proposal.state;
                                sec_global_urban_boundary=sec_urban_scattering.boundary_crossed;
                                urban_global_count(urban_step_counts,sec_global_urban_ion,true);
                            }
                            if(secondary_inelastic)schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryHazards);

                            float dE = 0.0F;
                            if (generic_recoil) {
                                const auto mid_e_u = sycl::fmax(
                                    recoil_stopping.energies[0],
                                    (sec_e - 0.5F * recoil_sp_at_start * sec_step_mm) * frag_inv_a);
                                const auto stopping=recoil_linear_sp(mid_e_u);
                                if (!(stopping > 0.0F)) break;
                                // Use the same isotope/material/unit-density
                                // recoil package as the step-start limiter.
                                dE = sycl::fmin(stopping * sec_step_mm, sec_e);
                            }

                            // CINEL02 packages are captured at hadronic PostStepDoIt:
                            // advance the continuous EM state to the collision point
                            // before selecting and replaying the final state.
                            const auto collision_input_dx = sec_dx;
                            const auto collision_input_dy = sec_dy;
                            const auto collision_input_dz = sec_dz;
                            if(sec_global_urban && sec_urban_scattering.range_terminal)dE=sec_e;
                            if(unified_secondary && !(sec_global_urban && sec_urban_scattering.range_terminal)){
                                auto draw=[&]() {
                                    if constexpr(kCarbonCtPointReuse)
                                        return unified_em_explicit_loss<false,true>(unified_secondary_state,
                                            sec_e,sec_step_mm,enable_secondary_energy_straggling,
                                            unified_secondary_pre,unified_secondary_uniform,&unified_secondary_point);
                                    else return unified_em_loss(unified_secondary_state,sec_e,sec_step_mm,
                                        enable_secondary_energy_straggling,unified_secondary_pre,unified_secondary_uniform);
                                }();
                                if(!draw.valid){
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,40+static_cast<int>(draw.failure_stage),unified_secondary_state.section,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);unified_secondary_count(7);break;}
                                dE=draw.loss;unified_secondary_count(2);unified_secondary_count(3,draw.proposed);unified_secondary_count(4,draw.accepted);
                                unified_secondary_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_secondary_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                            }
                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);
                             {

                            }
                            auto post_em_x = sec_x + collision_input_dx * sec_step_mm;
                            auto post_em_y = sec_y + collision_input_dy * sec_step_mm;
                            auto post_em_z = sec_z + collision_input_dz * sec_step_mm;
                            if(sec_global_urban) {
                                const float g=sec_urban_scattering.final_geom_path_mm;
                                auto end=urban_voxel_snap_endpoint(sec_urban_proposal.cell,g,
                                    {sec_x+sec_urban_proposal.direction.x*g+sec_urban_scattering.displacement_mm.x,
                                     sec_y+sec_urban_proposal.direction.y*g+sec_urban_scattering.displacement_mm.y,
                                     sec_z+sec_urban_proposal.direction.z*g+sec_urban_scattering.displacement_mm.z});
                                post_em_x=end.x;post_em_y=end.y;post_em_z=end.z;
                            }
                            if (!sec_global_urban && (kProductionSecondaryPath ? true : ct_secondary_exact_faces) && sec_face_clamp.hit_face && !secondary_inelastic && !secondary_elastic) {
                                const auto endpoint=ct_finish_exact_face_step(sec_face_clamp,sec_step_mm,
                                    {sec_x,sec_y,sec_z},{collision_input_dx,collision_input_dy,collision_input_dz},
                                    {ct_origin_x,ct_origin_y,ct_origin_z},{ct_spacing_x,ct_spacing_y,ct_spacing_z});
                                post_em_x=endpoint[0];post_em_y=endpoint[1];post_em_z=endpoint[2];
                            }

                            // Independent continuous optical-depth audit.  The
                            // runtime sampler uses the step-start rates above;
                            // this diagnostic evaluates the same H/O rates at
                            // the start, midpoint and post-EM endpoint and
                            // integrates them with Simpson's rule over the
                            // actual (possibly collision-truncated) segment.
                            // It never feeds back into target selection or
                            // collision sampling.
                            const auto collision_bin = sycl::max(
                                0, sycl::min(
                                       static_cast<int>(post_em_z *
                                           inverse_depth_bin_width_mm),
                                       static_cast<int>(number_of_bins) - 1));
                            // Exact-face transport keeps continuous loss inside
                            // the SOURCE voxel. Do not mix source x/y with the
                            // endpoint z, or credit this loss at a nuclear vertex.
                            // Commit once here before a replay can update sec_x/y/z
                            // or break; the legacy dE commits below are suppressed.
                            const bool source_voxel_continuous_loss =
                                (sec_global_urban || ((kProductionSecondaryPath ? true : ct_secondary_exact_faces) && (kProductionSecondaryPath ? true : enable_ct_grid) && sec_in_ct)) && (kProductionSecondaryPath ? true : enable_voxel_scoring);
                            int continuous_step_voxel = -1;
                            if (source_voxel_continuous_loss) {
                                const auto source_dir=sec_urban_proposal.direction;
                                const int nx=voxel_bins_x;
                                const int ny=voxel_bins_y;
                                const int nz=number_of_bins;
                                const float xmin=voxel_min_x_mm;
                                const float ymin=voxel_min_y_mm;
                                const float vx=1/inverse_voxel_size_x_mm;
                                const float vy=1/inverse_voxel_size_y_mm;
                                const int sx=sec_global_urban ? urban_voxel_axis(sec_x,source_dir.x,xmin,xmin+nx*vx,vx,nx).index :
                                    static_cast<int>(sycl::floor((sec_x-xmin)/vx));
                                const int sy=sec_global_urban ? urban_voxel_axis(sec_y,source_dir.y,ymin,ymin+ny*vy,vy,ny).index :
                                    static_cast<int>(sycl::floor((sec_y-ymin)/vy));
                                const int sz=sec_global_urban ? urban_voxel_axis(sec_z,source_dir.z,0,
                                    phantom_length_mm,depth_bin_width_mm,nz).index : bin_z;
                                if (sx>=0 && sy>=0 && sz>=0 && sx<nx && sy<ny && sz<nz) {
                                    continuous_step_voxel=(sz*ny+sy)*nx+sx;
                                    sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,sycl::access::address_space::global_space>
                                        a(voxel_dose_device[continuous_step_voxel]);
                                    a.fetch_add(static_cast<DoseAtomicT>(dE));

                                }
                            }
                            if (secondary_inelastic) {
                                sec_e = post_em_e;
                                // v3: sample the target HERE at the post-EM
                                // collision energy E_c (sec_e just updated),
                                // so the replay-status record below and the
                                // package query share one energy with the mask.
                                if (schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : use_unified_water)) &&
                                    sec_e > energy_cutoff_MeV &&
                                    frag.generation <
                                        max_secondary_inelastic_generations) {
                                    const float sec_e_c = sec_e * frag_inv_a;
                                    const int proj_idx_c =
                                        secondary_projectile_lut_index_device(
                                            schneider_ct_device_ctx.sec_proj_keys,
                                            schneider_ct_device_ctx.sec_num_projectiles,
                                            transport_z, transport_a);
                                    const std::size_t section_c = (kProductionSecondaryPath ? false : use_unified_water) ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    const auto sec_masked_c = schneider_ct_device_ctx.secondary_rates(
                                        proj_idx_c, section_c, sec_e_c);
                                    // Same tag-14 draw as the legacy site (same
                                    // step counter: sec_steps increments after
                                    // the lookup); only the energy moves E_h ->
                                    // E_c. Empty draw (slowing left every
                                    // channel domain): counted, never queried.
                                    const float u_target_c = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 14);
                                    const int sampled_z_c =
                                        sample_masked_secondary_target_device(
                                            sec_masked_c.partials, u_target_c);
                                    if (sampled_z_c <= 0) {
                                        // Post-EM null collision (declared research
                                        // approximation): no UnsupportedTargets,
                                        // no StoppedBeforeReplay. The track keeps
                                        // its post-EM energy/position, continues
                                        // transport, loses no energy, deposits
                                        // nothing locally, issues no lookup.
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::SecondaryPostEmNullCollisions);
                                        schneider_float_add_device(
                                            schneider_float_device,
                                            SchneiderFloatSlot::PostEmNullEnergy,
                                            sycl::fmax(0.0F, sec_e));
                                        secondary_inelastic = false;
                                        skip_schneider_lookup = true;
                                    } else {
                                        secondary_target_z =
                                            static_cast<std::int16_t>(sampled_z_c);
                                    }
                                }
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                                // Record the hazard before attempting package
                                // replay so isotope/target/generation misses
                                // remain distinguishable from valid events.
                                // v3 sampler-empty skips the record exactly
                                // like the v1 hazard-site empty draw (which
                                // never reaches this block).
                                if (!skip_schneider_lookup) {
                                }  // !skip_schneider_lookup (v3 sampler-empty records nothing)
                            }

                            if (bin_z != pending_sec_bin) {
                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dose(dose_device[pending_sec_bin]);
                                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
                                    pending_sec_depth_MeV = 0.0F;
                                }
                                pending_sec_bin = bin_z;
                            }
                            pending_sec_depth_MeV += dE;
                            continuous_species_tally.add_all(dE);

                            if ((kProductionSecondaryPath ? true : enable_voxel_scoring) || secondary_inelastic) {
                                int cur_voxel = -1;
                                const auto score_bin_x = static_cast<int>(
                                    (sec_x - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto score_bin_y = static_cast<int>(
                                    (sec_y - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                if (score_bin_x >= 0 && score_bin_x < static_cast<int>(voxel_bins_x) && score_bin_y >= 0 && score_bin_y < static_cast<int>(voxel_bins_y)) {
                                    // Continuous EM loss belongs to the source
                                    // step voxel.  `collision_bin` is the
                                    // post-EM endpoint/nuclear-vertex bin and
                                    // mixing it with source x/y creates depth
                                    // spikes when a step ends on a z boundary.
                                    cur_voxel = (bin_z * static_cast<int>(voxel_bins_y) + score_bin_y) * static_cast<int>(voxel_bins_x) + score_bin_x;
                                }
                                if (cur_voxel != pending_sec_voxel || secondary_inelastic) {
                                    if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_vox(voxel_dose_device[pending_sec_voxel]);
                                        atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));

                                    }
                                    pending_sec_voxel_MeV = 0.0F;
                                    pending_sec_voxel = cur_voxel;
                            if (secondary_inelastic && !(sec_e > energy_cutoff_MeV) &&
                                schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : use_unified_water)) &&
                                frag.generation < max_secondary_inelastic_generations) {
                                // Sampled Schneider collision whose post-EM
                                // energy is already below cutoff: continuous
                                // stopping owns the energy, no replay attempted.
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
                            }
                            if (secondary_inelastic && sec_e > energy_cutoff_MeV) {
                                // v3 sampling already happened at the post-EM
                                // update above; skip the query only when the
                                // sampler found no in-domain channel.
                                if (schneider_ct_device_ctx.uses_cinel03() && !skip_schneider_lookup) {
                                    // Exact (Z/A, target_Z) channel with the
                                    // secondary's own sampled target: never the
                                    // primary target, never water/O fallback.
                                    // Tag 15 drives the bracket choice, tag 16
                                    // the intra-node event pick.
                                    const float sec_u_bracket = rng::uniform01(2026, frag.rng_stream, sec_steps, 15);
                                    const float sec_u_event = rng::uniform01(2026, frag.rng_stream, sec_steps, 16);
                                    const auto sec_lookup = cinel03_lookup_event_device(
                                        schneider_ct_device_ctx.sec_energy_nodes,
                                        schneider_ct_device_ctx.sec_node_count,
                                        schneider_ct_device_ctx.sec_event_offsets,
                                        schneider_ct_device_ctx.sec_event_indices,
                                        schneider_ct_device_ctx.sec_total_events,
                                        transport_z, transport_a, secondary_target_z,
                                        sec_e * frag_inv_a,
                                        sec_u_bracket, sec_u_event);
                                    schneider_record_lookup_device(
                                        schneider_diag_device, schneider_float_device,
                                        false, sec_lookup, sec_e);

                                    if (sec_lookup.status == Cinel03LookupStatus::Hit) {
                                        const std::uint32_t event_idx = sec_lookup.event_index;
                                        secondary_replay_succeeded = true;
                                        ion_species_energy_add_device(
                                            ion_species_energy_device, ledger_species_idx, 8U, sec_e);
                                        // Use the collision vertex, not the source
                                        // voxel or legacy pre-x/post-z mixed index.
                                        // These are subsets, never extra energy sinks.
                                        if ((kProductionSecondaryPath ? true : enable_voxel_scoring) && replay_vertex_in_scoring_box(
                                                {post_em_x,post_em_y,post_em_z},
                                                {voxel_min_x_mm,voxel_min_y_mm,0.0F},
                                                {voxel_max_x_mm,voxel_max_y_mm,
                                                 static_cast<float>(number_of_bins)*depth_bin_width_mm})) {
                                            ion_species_energy_add_device(
                                                ion_species_energy_device, ledger_species_idx,
                                                IonSpeciesLedgerSchema::cinel03_replay_input_fov, sec_e);
                                            ion_species_energy_add_device(
                                                ion_species_energy_device, ledger_species_idx,
                                                IonSpeciesLedgerSchema::cinel03_replay_step_dE_fov, dE);
                                        }
                                        const auto& event = schneider_ct_device_ctx.sec_interactions[event_idx];
                                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                                            2026, frag.rng_stream, sec_steps, cinel03_event_azimuth_dimension);
                                        const float event_cos = sycl::cos(event_phi);
                                        const float event_sin = sycl::sin(event_phi);
                                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);
                                        pending_sec_depth_MeV += local_deposit;
                                        if ((kProductionSecondaryPath ? true : enable_voxel_scoring) && cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += local_deposit;
                                        }
                                        // The break below skips the common-path
                                        // energy-band scorer; local nuclear
                                        // deposit must be recorded here or the
                                        // ROI diagnostic will not close.

                                        if (deposited_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(deposited_device[frag.parent_history]);
                                            // Ledger keeps local + step dE here; the
                                            // common path is skipped by the break
                                            // below, so this is the single dE credit.
                                            atomic_dep.fetch_add(local_deposit + dE);
                                            schneider_energy_add_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                local_deposit + dE);
                                            if (source_voxel_continuous_loss) {
                                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                                    cur_voxel>=0,local_deposit);
                                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                                    continuous_step_voxel>=0,dE);
                                            } else grid_deposit_split_device(
                                                grid_deposited_in_device,grid_deposited_out_device,
                                                (kProductionSecondaryPath ? true : enable_voxel_scoring) && cur_voxel>=0,local_deposit+dE);
                                        }
                                        // The break below skips the common-path
                                        // voxel dE commit: commit it here so the
                                        // collision step reaches history ledger,
                                        // depth, voxel and species scorers.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            (kProductionSecondaryPath ? true : enable_voxel_scoring) && !source_voxel_continuous_loss, cur_voxel, dE);

                                        float sec_charged_accounted_MeV = 0.0F;
                                        float sec_neutral_accounted_MeV = 0.0F;
                                        float sec_unsupported_accounted_MeV = 0.0F;

                                        const std::uint32_t prod_offset = event.product_offset;
                                        const std::uint32_t prod_count = event.direct_product_count;
                                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                                            if (prod_offset + ip >= schneider_ct_device_ctx.sec_total_products) break;
                                            const auto& product = schneider_ct_device_ctx.sec_products[prod_offset + ip];

                                            if (product.role == 2) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_unsupported_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                                continue;
                                            }
                                            if (product.z <= 0 || product.a <= 0) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_neutral_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                                continue;
                                            }
                                            if (product.role != 0) {
                                                continue;
                                            }

                                            // TopasCompatKill for Be6 (Z=4, A=6):
                                            // independent counter + energy, never queued.
                                            if (product.z == 4 && product.a == 6) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::SecondaryBe6Kills);
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::Be6TopasCompatKills);
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::Be6KillEnergy, ke);
                                                continue;
                                            }
                                            schneider_diag_increment_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryChargedBorn);

                                            if (product.kinetic_energy_MeV >
                                                energy_cutoff_MeV) {
                                                // Nuclear-generation limits never turn a
                                                // charged product into local dose.  The child
                                                // is queued for EM transport; its generation
                                                // guard suppresses any further nuclear hazard.
                                                const auto local_direction = rotate_cinel03_event_azimuth(
                                                    product.local_direction_x, product.local_direction_y,
                                                    product.local_direction_z, event_cos, event_sin);
                                                const auto child_direction = rotate_local_direction(
                                                    local_direction.x,
                                                    local_direction.y,
                                                    local_direction.z,
                                                    Direction3F{collision_input_dx, collision_input_dy, collision_input_dz});

                                                if (secondary_queue_device != nullptr) {
                                                    auto count_ref = sycl::atomic_ref<
                                                        uint32_t, sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>(
                                                        *secondary_count_device);
                                                    const auto output = count_ref.fetch_add(1U);
                                                    if (output < max_secondaries) {
                                                        SecondaryParticle child{};
                                                        child.z = product.z;
                                                        child.a = product.a;
                                                        child.energy_MeV = product.kinetic_energy_MeV;
                                                        child.pos_x_mm = post_em_x;
                                                        child.pos_y_mm = post_em_y;
                                                        child.pos_z_mm = post_em_z;
                                                        child.dir_x = child_direction.x;
                                                        child.dir_y = child_direction.y;
                                                        child.dir_z = child_direction.z;
                                                        child.weight = 1.0F;
                                                        child.generation = static_cast<std::uint16_t>(frag.generation + 1U);
                                                        child.parent_history = frag.parent_history;
                                                        child.rng_stream = rng::event_product_stream(
                                                            frag.rng_stream, sec_steps,
                                                            rng::branch_role_cascade_charged, ip);
                                                        secondary_queue_device[output] = child;
                                                        sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::SecondaryChargedQueued);
                                                        if (ion_species_energy_device != nullptr) {
                                                            ion_record_queued_secondary_birth_device(
                                                                ion_species_energy_device, child.z, child.a,
                                                                child.energy_MeV);
                                                            const auto child_species_idx = carbon::get_charged_species_idx(child.z, child.a);
                                                            if (child_species_idx < 18) {
                                                                ion_species_energy_add_device(
                                                                    ion_species_energy_device, child_species_idx, 10U, child.energy_MeV);
                                                            }
                                                        }
                                                    } else {
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::SecondaryQueueOverflows);
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::QueueOverflows);
                                                        if (secondary_overflow_count_device != nullptr) {
                                                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ov(*secondary_overflow_count_device);
                                                            atomic_ov.fetch_add(1U);
                                                        }
                                                        if (secondary_overflow_energy_device != nullptr) {
                                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ove(*secondary_overflow_energy_device);
                                                            atomic_ove.fetch_add(product.kinetic_energy_MeV);
                                                        }
                                                    }
                                                }
                                            } else {
                                                pending_sec_depth_MeV += product.kinetic_energy_MeV;
                                                if ((kProductionSecondaryPath ? true : enable_voxel_scoring) && cur_voxel >= 0) {
                                                    pending_sec_voxel_MeV += product.kinetic_energy_MeV;
                                                }

                                                if (deposited_device != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_dep(deposited_device[frag.parent_history]);
                                                    atomic_dep.fetch_add(product.kinetic_energy_MeV);
                                                    schneider_energy_add_device(
                                                        schneider_diag_device,
                                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                        product.kinetic_energy_MeV);
                                                    grid_deposit_split_device(
                                                        grid_deposited_in_device,
                                                        grid_deposited_out_device,
                                                        (kProductionSecondaryPath ? true : enable_voxel_scoring) && cur_voxel >= 0,
                                                        product.kinetic_energy_MeV);
                                                }
                                                sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::SecondaryChargedCutoffKills);
                                            }
                                        }

                                        schneider_float_add_device(
                                            schneider_float_device, SchneiderFloatSlot::ReactionQResidual,
                                            sycl::fmax(0.0F, sec_e - local_deposit -
                                                                  sec_charged_accounted_MeV -
                                                                  sec_neutral_accounted_MeV -
                                                                  sec_unsupported_accounted_MeV));
                                        // NO-DOUBLE-COUNT RULE (same as primary
                                        // vertex): legacy untracked sink already
                                        // contains neutral + unsupported + Q;
                                        // split slots are informational only.
                                        const float sec_untracked_MeV = sycl::fmax(0.0F, sec_e - local_deposit - sec_charged_accounted_MeV);
                                        if (sec_untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(untracked_nuclear_device[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_untracked_MeV);
                                        }
                                        sec_e = 0.0F;
                                        break;
                                    } else {
                                        // CINEL03 miss: fail closed + per-miss log.
                                        schneider_log_miss_device(
                                            schneider_miss_device, schneider_miss_count_device,
                                            kSchneiderMissLogCap, false,
                                            transport_z, transport_a, secondary_target_z,
                                            schneider_hazard_section,
                                            frag.generation > 255 ? 255
                                                                  : static_cast<std::uint8_t>(frag.generation),
                                            sec_lookup, sec_e * frag_inv_a, dE,
                                            schneider_hazard_total_rate,
                                            schneider_hazard_step_mm, sec_e,
                                            frag.energy_MeV);
                                        if (untracked_nuclear_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(untracked_nuclear_device[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_e);
                                        }
                                        // cur_voxel is out of scope on the miss
                                        // path: recompute the collision voxel
                                        // with the same formula for both the
                                        // split and the commit below.
                                        int miss_voxel = -1;
                                        {
                                            const auto miss_sbx = static_cast<int>(
                                                (sec_x - voxel_min_x_mm) *
                                                inverse_voxel_size_x_mm);
                                            const auto miss_sby = static_cast<int>(
                                                (sec_y - voxel_min_y_mm) *
                                                inverse_voxel_size_y_mm);
                                            if (miss_sbx >= 0 &&
                                                miss_sbx < static_cast<int>(voxel_bins_x) &&
                                                miss_sby >= 0 &&
                                                miss_sby < static_cast<int>(voxel_bins_y)) {
                                                miss_voxel =
                                                    (collision_bin *
                                                     static_cast<int>(voxel_bins_y) +
                                                     miss_sby) *
                                                        static_cast<int>(voxel_bins_x) +
                                                    miss_sbx;
                                            }
                                        }
                                        if (deposited_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(deposited_device[frag.parent_history]);
                                            atomic_dep.fetch_add(dE);
                                            schneider_energy_add_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                            grid_deposit_split_device(
                                                grid_deposited_in_device,
                                                grid_deposited_out_device,
                                                (kProductionSecondaryPath ? true : enable_voxel_scoring) && (source_voxel_continuous_loss ? continuous_step_voxel>=0 : miss_voxel>=0),
                                                dE);
                                        }
                                        // Same bypass as the replay-hit path:
                                        // the break below skips the common-path
                                        // voxel dE commit.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            (kProductionSecondaryPath ? true : enable_voxel_scoring) && !source_voxel_continuous_loss, miss_voxel, dE);
                                        sec_e = 0.0F;
                                        break;
                                    }
                                }
                            } else if (secondary_inelastic) {
                                // The post-EM collision energy is already at
                                // or below the transport cutoff, so no package
                                // event is eligible.  Keep the candidate in
                                // the status partition as post_em_below_cutoff.
                            }
                                    pending_sec_voxel = cur_voxel;
                                }
                                if (cur_voxel >= 0) {
                                    if (!source_voxel_continuous_loss) {
                                        pending_sec_voxel_MeV += dE;

                                    }
                                    if (in_fov_dose_device != nullptr && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[pending_sec_bin]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(dE));
                                        continuous_species_tally.add_fov(dE);
                                    }
                                }
                            }

                            if (!secondary_inelastic) sec_e = post_em_e;
                            // cur_voxel is out of scope on the common path:
                            // recompute the step voxel with the same formula.
                            int com_voxel = -1;
                            {
                                const auto com_sbx = static_cast<int>(
                                    (sec_x - voxel_min_x_mm) *
                                    inverse_voxel_size_x_mm);
                                const auto com_sby = static_cast<int>(
                                    (sec_y - voxel_min_y_mm) *
                                    inverse_voxel_size_y_mm);
                                if (com_sbx >= 0 &&
                                    com_sbx < static_cast<int>(voxel_bins_x) &&
                                    com_sby >= 0 &&
                                    com_sby < static_cast<int>(voxel_bins_y)) {
                                    com_voxel =
                                        (bin_z *
                                         static_cast<int>(voxel_bins_y) +
                                         com_sby) *
                                            static_cast<int>(voxel_bins_x) +
                                        com_sbx;
                                }
                            }
                            if (deposited_device != nullptr) {
                                if constexpr (BroadBeam) {
                                    // Same per-step values; integer microMeV truncation is
                                    // applied per step so the diagnostic total is unchanged.
                                    sec_dep_pending_MeV += dE;
                                    if (dE > 0.0F)
                                        sec_diag_dep_pending_uMeV += static_cast<std::uint64_t>(static_cast<double>(dE) * 1.0e6);
                                    if (sycl::isfinite(dE) && dE > 0.0F) {
                                        if (source_voxel_continuous_loss ? continuous_step_voxel>=0 : com_voxel>=0) sec_grid_in_pending_MeV += dE;
                                        else sec_grid_out_pending_MeV += dE;
                                    }
                                } else {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(deposited_device[frag.parent_history]);
                                atomic_dep.fetch_add(dE);
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                grid_deposit_split_device(
                                    grid_deposited_in_device,
                                    grid_deposited_out_device, source_voxel_continuous_loss ? continuous_step_voxel>=0 : com_voxel>=0,
                                    dE);
                                }
                            }
                            if (!secondary_inelastic) {
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                            }

                            if (!sec_global_urban && ion_should_apply_secondary_mcs(
                                    secondary_inelastic, secondary_replay_succeeded,
                                    enable_multiple_scattering, sec_e,
                                    energy_cutoff_MeV)) {
                                // Secondary MCS material selection shares the
                                // primary helper: Schneider CT voxels use the
                                // section X0, everywhere else falls back to
                                // water. Density is the local CT density
                                // (== water outside CT), never double-counted.
                                const auto sec_radiation_length_g_per_cm2 =
                                    static_cast<float>(
                                        select_transport_radiation_length_g_per_cm2(
                                            ct_material_ids_are_schneider_sections,
                                            sec_in_ct,
                                            static_cast<unsigned>(sec_ct_material),
                                            enable_ct_material_mcs,
                                            active_water_radiation_length));
                                // Outside global Urban, every charged secondary uses FE.
                                {
                                    const auto route_slot =
                                        transport_z == 6 && transport_a == 12
                                            ? SchneiderDiagSlot::SecondaryFeC12Steps
                                        : transport_z == 2 && transport_a == 4
                                            ? SchneiderDiagSlot::SecondaryFeHe4Steps
                                        : transport_z == 1 && transport_a >= 1 &&
                                                  transport_a <= 3
                                            ? SchneiderDiagSlot::SecondaryFePdtSteps
                                            : SchneiderDiagSlot::SecondaryFeOtherChargedSteps;
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        route_slot);
                                    auto observation_path_mm = -1.0F;

                                    const auto correlated =
                                        ion_fermi_eyges_transport_step(
                                            Direction3F{collision_input_dx,
                                                        collision_input_dy,
                                                        collision_input_dz},
                                            sec_e + dE,
                                            static_cast<int>(transport_z),
                                            static_cast<int>(transport_a),
                                            dE, sec_step_mm,
                                            c12_fermi_eyges_max_segment_mm,
                                            sec_local_density_g_per_cm3,
                                            sec_radiation_length_g_per_cm2,
                                            2026, frag.rng_stream,
                                            static_cast<std::uint64_t>(sec_steps),
                                            100U, observation_path_mm,
                                            static_cast<float>(nucleon_rest_mass_MeV), fe_gaussian_only);

                                    sec_x += correlated.displacement_mm.x;
                                    sec_y += correlated.displacement_mm.y;
                                    sec_z += correlated.displacement_mm.z;
                                    sec_dx = correlated.direction.x;
                                    sec_dy = correlated.direction.y;
                                    sec_dz = correlated.direction.z;
                                }
                            }
                            // A hazard sampled at step start can leave a supported rate
                            // interval after EM loss. Treat the zero-rate endpoint as a
                            // null collision; retain post-EM energy, position and MCS.
                            if(secondary_elastic && sec_e>energy_cutoff_MeV &&
                               all_elastic.template species_total_rate<kCachedElasticTotals>(all_elastic.projectile(transport_z,transport_a),(kProductionSecondaryPath ? false : use_unified_water)?25:sec_ct_material,sec_e*frag_inv_a)==0) {
                                secondary_elastic=false;
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[9]).fetch_add(1);
                            }
                            if(secondary_elastic && sec_e>energy_cutoff_MeV) {
                                const int p=all_elastic.projectile(transport_z,transport_a);
                                const auto draw=all_elastic.draw(p,(kProductionSecondaryPath ? false : use_unified_water)?25:sec_ct_material,
                                    sec_e,sec_dx,sec_dy,sec_dz,
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,71),rng::uniform01(2026,frag.rng_stream,sec_steps,72),
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,73),rng::uniform01(2026,frag.rng_stream,sec_steps,74));
                                if(!draw.valid){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[8]).fetch_add(1);break;
                                }
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[1]).fetch_add(1);
                                const auto scat=draw.outcome;sec_e=scat.projectile_ke_MeV;
                                sec_dx=scat.proj_dir_x;sec_dy=scat.proj_dir_y;sec_dz=scat.proj_dir_z;
                                const float recoil=scat.recoil_ke_MeV;
                                const int child_species=carbon::get_charged_species_idx(draw.target_z,draw.target_a);
                                if(recoil>energy_cutoff_MeV && (child_species>=0||recoil_stopping.projectile(draw.target_z,draw.target_a)>=0)) {
                                    schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryChargedBorn);
                                    const auto output=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_count_device).fetch_add(1);
                                    if(output<max_secondaries){
                                        SecondaryParticle child{};child.z=draw.target_z;child.a=draw.target_a;child.energy_MeV=recoil;
                                        child.pos_x_mm=sec_x;child.pos_y_mm=sec_y;child.pos_z_mm=sec_z;
                                        child.dir_x=scat.recoil_dir_x;child.dir_y=scat.recoil_dir_y;child.dir_z=scat.recoil_dir_z;
                                        child.weight=1;child.generation=child_species>=0?frag.generation:max_secondary_inelastic_generations;child.parent_history=frag.parent_history;
                                        child.rng_stream=rng::event_product_stream(frag.rng_stream,sec_steps,rng::branch_role_cascade_charged,0xE1U);
                                        secondary_queue_device[output]=child;
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[1]).fetch_add(recoil);
                                        schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryChargedQueued);
                                        ion_record_queued_secondary_birth_device(ion_species_energy_device,child.z,child.a,recoil);
                                        ion_species_energy_add_device(ion_species_energy_device,child_species,10U,recoil);
                                    }else{
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_count_device).fetch_add(1);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_energy_device).fetch_add(recoil);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[2]).fetch_add(recoil);
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[11]).fetch_add(1);
                                    }
                                    ion_species_energy_add_device(ion_species_energy_device,ledger_species_idx,8U,recoil);
                                }else if(recoil>energy_cutoff_MeV){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[3]).fetch_add(1);
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(untracked_nuclear_device[frag.parent_history]).fetch_add(recoil);
                                    ion_species_energy_add_device(ion_species_energy_device,ledger_species_idx,8U,recoil);
                                }else if(recoil>0){
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[0]).fetch_add(recoil);
                                    const int vx=static_cast<int>(sycl::floor((sec_x-voxel_min_x_mm)/voxel_size_x_mm));
                                    const int vy=static_cast<int>(sycl::floor((sec_y-voxel_min_y_mm)/voxel_size_y_mm));
                                    const int vz=static_cast<int>(sycl::floor(sec_z*inverse_depth_bin_width_mm));
                                    const bool inside=(kProductionSecondaryPath ? true : enable_voxel_scoring)&&vx>=0&&vy>=0&&vz>=0&&vx<static_cast<int>(voxel_bins_x)&&vy<static_cast<int>(voxel_bins_y)&&vz<static_cast<int>(number_of_bins);
                                    if(inside){const int v=(vz*voxel_bins_y+vy)*voxel_bins_x+vx;
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(voxel_dose_device[v]).fetch_add(static_cast<DoseAtomicT>(recoil));

                                    }
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(deposited_device[frag.parent_history]).fetch_add(recoil);
                                    grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,inside,recoil);
                                    schneider_energy_add_device(schneider_diag_device,SchneiderDiagSlot::SecondaryDepositedMicroMeV,recoil);
                                    ion_species_energy_add_device(ion_species_energy_device,ledger_species_idx,3U,recoil);
                                    if(inside)ion_species_energy_add_device(ion_species_energy_device,ledger_species_idx,4U,recoil);
                                }
                            }

                            ++sec_steps;
                            ++local_sec_steps;

                        }

                        if constexpr (BroadBeam) {
                            if (sec_dep_pending_MeV != 0.0F && deposited_device != nullptr)
                                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    deposited_device[frag.parent_history]).fetch_add(sec_dep_pending_MeV);
                            schneider_diag_add_device(schneider_diag_device,
                                SchneiderDiagSlot::SecondaryDepositedMicroMeV, sec_diag_dep_pending_uMeV);
                            if (sec_grid_in_pending_MeV != 0.0 && grid_deposited_in_device != nullptr)
                                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *grid_deposited_in_device).fetch_add(sec_grid_in_pending_MeV);
                            if (sec_grid_out_pending_MeV != 0.0 && grid_deposited_out_device != nullptr)
                                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *grid_deposited_out_device).fetch_add(sec_grid_out_pending_MeV);
                            sec_dep_pending_MeV = 0.0F;sec_diag_dep_pending_uMeV = 0;
                            sec_grid_in_pending_MeV = sec_grid_out_pending_MeV = 0.0;
                        }
                        if(segment_paused){
                            SecondaryResumeState saved;
                            saved.sec_terminal_recorded=sec_terminal_recorded;
                            saved.unified_secondary_escaped_ct=unified_secondary_escaped_ct;
                            saved.continuous_species_tally=continuous_species_tally;
                            saved.sec_e=sec_e;
                            saved.sec_x=sec_x;
                            saved.sec_y=sec_y;
                            saved.sec_z=sec_z;
                            saved.sec_dx=sec_dx;
                            saved.sec_dy=sec_dy;
                            saved.sec_dz=sec_dz;
                            saved.pending_sec_depth_MeV=pending_sec_depth_MeV;
                            saved.pending_sec_voxel_MeV=pending_sec_voxel_MeV;
                            saved.pending_sec_bin=pending_sec_bin;
                            saved.pending_sec_voxel=pending_sec_voxel;
                            saved.unified_secondary_counter=unified_secondary_counter;
                            saved.urban_ionization_clock=unified_secondary_clock;
                            saved.unified_secondary_state=unified_secondary_state;
                            saved.nuclear_tau_remaining=nuclear_tau_remaining;
                            saved.nuclear_tau_active=nuclear_tau_active;
                            saved.nuclear_tau_rng_step=nuclear_tau_rng_step;
                            saved.sec_steps=sec_steps;
                            if constexpr (!BroadBeam) {
                            saved.urban.sec_urban_tlimit_mm =
                                sec_urban_persist.tlimit_mm;
                            saved.urban.sec_urban_born = sec_urban_born;
                            saved.urban.sec_urban_first_step=sec_urban_persist.first_step;
                            saved.urban.sec_global_urban_boundary=sec_global_urban_boundary;
                            }
                            saved.local_sec_rate_queries=local_sec_rate_queries;
                            saved.local_sec_steps=local_sec_steps;

                                for(int j=0;j<4;++j)saved.he4_audit[j]=he4_audit[j];

                            resume_states[state_idx]=saved;resume_ready[state_idx]=1;keep[active_pos]=1;
                            return; // Suspend: no terminal scoring or audit flush.
                        }

                        // Small per-step deposits must not contend directly on
                         {

                        }
                        // Small per-step deposits must not contend directly on
                        // one global species scalar. Preserve original scoring
                        // guards and reduce per track; all replay breaks arrive
                        // here too. This changes diagnostics only, not dose.
                        ion_species_energy_add_device(
                            ion_species_energy_device, ledger_species_idx, 1U,
                            continuous_species_tally.all_MeV);
                        ion_species_energy_add_device(
                            ion_species_energy_device, ledger_species_idx, 2U,
                            continuous_species_tally.fov_MeV);
                        schneider_diag_add_device(schneider_diag_device,
                                                  SchneiderDiagSlot::SecondaryRateQueries,
                                                  local_sec_rate_queries);
                        schneider_diag_add_device(schneider_diag_device,
                                                  SchneiderDiagSlot::SecondarySteps,
                                                  local_sec_steps);

                        const bool step_limited =
                            sec_e > energy_cutoff_MeV && sec_steps >= kSecondaryMaxSteps;
                        if (sec_e > 0.0F) {
                            bool sec_terminal_scored = false;
                            if (step_limited) {
                                ion_species_energy_add_device(
                                    ion_species_energy_device, ledger_species_idx,  9U, sec_e);
                                ion_species_terminal_increment_device(
                                    ion_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(IonSpeciesLedgerSchema::step_limit));
                                sec_terminal_recorded = true;
                                if (escaped_device != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_esc(escaped_device[frag.parent_history]);
                                    atomic_esc.fetch_add(sec_e);
                                    schneider_energy_add_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
                                }
                            } else if (!unified_secondary_escaped_ct && sec_z >= 0.0F && sec_z < phantom_length_mm) {
                                ion_species_energy_add_device(
                                    ion_species_energy_device, ledger_species_idx, 5U, sec_e);
                                ion_species_terminal_increment_device(
                                    ion_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(IonSpeciesLedgerSchema::terminal_deposit));
                                sec_terminal_recorded = true;
                                const auto bin_x = static_cast<int>((sec_x - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto bin_y = static_cast<int>((sec_y - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                const auto bin_z = static_cast<int>(sec_z * inverse_depth_bin_width_mm);
                                if (bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                    if (bin_z != pending_sec_bin) {
                                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 &&
                                            pending_sec_bin < static_cast<int>(number_of_bins)) {
                                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dose(dose_device[pending_sec_bin]);
                                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
                                            pending_sec_depth_MeV = 0.0F;
                                        }
                                        pending_sec_bin = bin_z;
                                    }
                                    pending_sec_depth_MeV += sec_e;
                                    if ((kProductionSecondaryPath ? true : enable_voxel_scoring)) {
                                        int cur_voxel = -1;
                                        if (bin_x >= 0 && bin_x < static_cast<int>(voxel_bins_x) &&
                                            bin_y >= 0 && bin_y < static_cast<int>(voxel_bins_y)) {
                                            cur_voxel = (bin_z * static_cast<int>(voxel_bins_y) + bin_y) *
                                                            static_cast<int>(voxel_bins_x) +
                                                        bin_x;
                                        }
                                        if (cur_voxel != pending_sec_voxel) {
                                            if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 &&
                                                pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_vox(voxel_dose_device[pending_sec_voxel]);
                                                atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));

                                            }
                                            pending_sec_voxel_MeV = 0.0F;
                                            pending_sec_voxel = cur_voxel;
                                        }
                                        if (cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += sec_e;

                                            sec_terminal_scored = true;
                                            if (in_fov_dose_device != nullptr) {
                                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_in_fov(in_fov_dose_device[bin_z]);
                                                atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(sec_e));
                                                ion_species_energy_add_device(
                                                    ion_species_energy_device, ledger_species_idx,
                                                    6U, sec_e);
                                            }
                                        }
                                    }
                                }
                                if (deposited_device != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dep(deposited_device[frag.parent_history]);
                                    atomic_dep.fetch_add(sec_e);
                                    schneider_energy_add_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV, sec_e);
                                    grid_deposit_split_device(
                                        grid_deposited_in_device,
                                        grid_deposited_out_device,
                                        sec_terminal_scored, sec_e);
                                }
                            } else {
                                ion_species_energy_add_device(
                                    ion_species_energy_device, ledger_species_idx, 7U, sec_e);
                                ion_species_terminal_increment_device(
                                    ion_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(IonSpeciesLedgerSchema::boundary_escape));
                                sec_terminal_recorded = true;
                                if (escaped_device != nullptr) {
                                // Escaped phantom boundaries
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_esc(escaped_device[frag.parent_history]);
                                atomic_esc.fetch_add(sec_e);
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
                                }
                            }
                        }

                        if (!sec_terminal_recorded) {
                            ion_species_terminal_increment_device(
                                ion_species_terminal_device, ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    IonSpeciesLedgerSchema::continuous_stop));
                        }

                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_sec_bin]);
                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_sec_depth_MeV));
                        }
                        if ((kProductionSecondaryPath ? true : enable_voxel_scoring) && pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_vox(voxel_dose_device[pending_sec_voxel]);
                            atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));

                        }
                                        };
                cgh.parallel_for<SecondaryTransportKernel<
                                                                 Pipeline, kProductionSecondaryPath,
                                                                 BroadBeam>>(
                    sycl::range<1>(active_size), secondary_kernel);
            });
                };
                auto round_event = [&]() {
                    if constexpr(BroadBeam &&
                        (std::is_same_v<Pipeline, pipelines::CarbonBroadbeam>)) {
                        if(broadbeam_ct_secondary_specialize)
                            return launch_secondary_round(std::true_type{},resume_active);
                    }
                    return launch_secondary_round(std::false_type{},resume_active);
                }();
                round_event.wait_and_throw();
                const double round_seconds = event_duration_seconds(round_event);
                secondary_kernel_seconds += round_seconds;
                generation_kernel_seconds += round_seconds;
                if (finish_tail) {
                    generation_tail_seconds += round_seconds;
                    ++tail_rounds;
                }

                ++segment_rounds;
                if (finish_tail) break; // No suspended tracks; avoid an empty compaction.
                const auto compact_start=std::chrono::steady_clock::now();
                const unsigned resume_groups=(resume_active+255)/256;
                queue.parallel_for(sycl::nd_range<1>(resume_groups*256,256),[=](sycl::nd_item<1> it){
                    const auto i=it.get_global_linear_id();const unsigned flag=i<resume_active?keep[i]:0;
                    const auto rank=sycl::exclusive_scan_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    const auto count=sycl::reduce_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    if(i<resume_active)ranks[i]=rank;
                    if(it.get_local_linear_id()==0)block_counts[it.get_group_linear_id()]=count;
                });

                    queue.single_task([=](){unsigned sum=0;for(unsigned i=0;i<resume_groups;++i){block_offsets[i]=sum;sum+=block_counts[i];}*active_count=sum;});
                    queue.parallel_for(sycl::range<1>(resume_active),[=](sycl::id<1> id){const auto i=id[0];if(keep[i])next_order[block_offsets[i/256]+ranks[i]]=active_order[i];});
                    queue.copy(active_count,&resume_active,1).wait_and_throw();

                std::swap(active_order,next_order);
                compact_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-compact_start).count();
                }
                if (segment_secondaries) std::cout<<"[segmented-secondary] rounds="<<segment_rounds
                    <<" tail_rounds="<<tail_rounds<<" tail_s="<<generation_tail_seconds
                    <<" kernel_s="<<generation_kernel_seconds
                    <<" compaction_s="<<compact_seconds
                    <<" begin="<<batch_begin<<" end="<<generation_end<<"\n";
                mem_tracker.free(resume_states);
                for(auto* ptr:{resume_ready,active_order,next_order,keep,ranks,block_counts,block_offsets,active_count})mem_tracker.free(ptr);
                const auto batch_end = generation_end;
                generation_begin = batch_end;
                queue.copy(secondary_count_device, &secondary_count_host, 1)
                    .wait_and_throw();
                generation_end = sycl::min(
                    secondary_count_host, static_cast<std::uint32_t>(max_secondaries));
                std::cout << "[progress] secondary batch completed: ["
                          << batch_begin << ", " << batch_end << ") particles; queued="
                          << generation_end << std::endl;
            }
            std::cout << "[secondary-schedule] grouping_seconds="
                      << secondary_group_seconds << "\n";

        }
    }
