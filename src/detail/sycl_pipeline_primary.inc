// Shared primary phase, included inside transport_pipeline<Pipeline, JointResponse>.
    RuntimeScope runtime_steps("transport_loop_including_scoring_and_queue_transfers");
    double primary_kernel_seconds = 0.0;

    // Opt-in homogeneous-water Urban minibeam specialization. Eligibility
    // proves every compile-time substitution below against runtime settings.
    const bool use_minibeam_primary_path = config.minibeam_primary_specialization;
    if (use_minibeam_primary_path && !(
        (!BroadBeam && config.enable_minibeam) && global_urban && use_unified_water &&
        !enable_ct_grid))
        throw std::invalid_argument("Minibeam primary specialization requires uniform-water Urban minibeam");
    std::cout << "[minibeam-primary-specialization] " << use_minibeam_primary_path << "\n";
    std::cout << "[ct-minibeam-split] " << ct_minibeam_split << "\n";
    const bool primary_minibeam_enabled = (!BroadBeam && config.enable_minibeam);
    // Package models are the authority. Do not infer their behavior from Z/A alone.
    const bool nonion_primary = BroadBeam && JointResponse==0 &&
        config.primary_atomic_number==1 && config.primary_mass_number==1 && nonion_primary_package_safe;
    std::cout<<"[nonion-primary] enabled="<<nonion_primary<<"\n";

    // compact: a beamline launch writes one medium-entry record per history and
    // partitions survivors to the front; the medium launch transports only those.
    const bool compact_minibeam_primary = !BroadBeam && config.enable_minibeam;
    if (compact_minibeam_primary && !(global_urban && (enable_ct_grid || use_unified_water)))
        throw std::invalid_argument("Compact primary requires Urban water or CT minibeam");
    // Exact CT profile: every constant substituted in stage 11 is checked
    // here. The original material-aware stage 4 remains the fallback.
    const bool ct_queued_primary_eligible = ct_minibeam_split &&
        global_urban == true &&
        enable_ct_grid == true &&
        use_unified_water == false &&
        enable_voxel_scoring == true &&
        enable_multiple_scattering == true &&
        enable_inelastic == true &&
        use_schneider_primary_xs == true &&
        use_material_electron == true &&
        use_material_ct == true &&
        defer_material_packets == true &&
        primary_atomic_number == 1 &&
        primary_mass_number == 1 &&
        inverse_mass_number == 1.0F;
    const bool ct_queued_primary_specialize = ct_queued_primary_eligible;
    std::cout << "[ct-queued-primary-specialize] enabled=" << ct_queued_primary_specialize << "\n";
    auto* primary_medium_entry = compact_minibeam_primary
        ? mem_tracker.allocate<PrimaryMediumEntryState>(history_chunk) : nullptr;
    auto* primary_entry_counts = compact_minibeam_primary
        ? mem_tracker.allocate<std::uint64_t>(2) : nullptr;
    if (compact_minibeam_primary && (!primary_medium_entry || !primary_entry_counts)) throw std::bad_alloc();
    // Proton medium-entry records are ordered by entry energy before transport.
    const unsigned water_energy_buckets = compact_minibeam_primary &&
        (use_minibeam_primary_path || ct_minibeam_split) &&
        config.primary_atomic_number==1 && config.primary_mass_number==1 ? 16U : 0U;
    auto* water_energy_order = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(history_chunk) : nullptr;
    auto* water_energy_counts = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(water_energy_buckets+1) : nullptr;
    auto* water_energy_offsets = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(water_energy_buckets+1) : nullptr;
    if(water_energy_buckets && (!water_energy_order || !water_energy_counts || !water_energy_offsets))throw std::bad_alloc();
    std::vector<sycl::event> water_order_events;
    std::cout << "[minibeam-candidates] water_energy_buckets=" << water_energy_buckets << '\n';
    // Beamline lane order: a straight-ray prediction from the sampled flat
    // source groups copper-bound histories (~600 Cu steps) apart from open-slit
    // ones (a few air steps), so fewer warps idle on a single long lane. Only
    // the lane->history map changes; every history keeps its index, RNG
    // stream and output slots.
    const bool order_beamline_lanes = compact_minibeam_primary && enable_minibeam && enable_flat_source &&
        !enable_emittance_source && primary_spot_count == 0;
    auto* beamline_order = order_beamline_lanes
        ? mem_tracker.allocate<std::uint32_t>(history_chunk) : nullptr;
    auto* beamline_order_counts = order_beamline_lanes
        ? mem_tracker.allocate<std::uint32_t>(2) : nullptr;
    if (order_beamline_lanes && (!beamline_order || !beamline_order_counts)) throw std::bad_alloc();
    std::cout << "[beamline-order] enabled=" << order_beamline_lanes << "\n";
    std::cout << "[primary-compact] enabled=" << compact_minibeam_primary
              << " entry_bytes=" << sizeof(PrimaryMediumEntryState) << "\n";
    std::vector<sycl::event> electron_events;
    std::uint64_t electron_tasks_total=0, electron_queue_peak=0;
    // Exact host certificate, never a tolerance-based grid approximation.
    const bool primary_shared_ct_cell = enable_ct_grid &&
        ct_nx==voxel_bins_x && ct_ny==voxel_bins_y && ct_nz==number_of_bins &&
        ct_spacing_x==voxel_size_x_mm && ct_spacing_y==voxel_size_y_mm &&
        ct_spacing_z==depth_bin_width_mm && ct_origin_x==voxel_min_x_mm &&
        ct_origin_y==voxel_min_y_mm && ct_origin_z==0.0F;
    std::vector<sycl::event> primary_events;
    primary_events.reserve((compact_minibeam_primary ? 2 : 1) *
        ((number_of_histories + history_chunk - 1) / history_chunk));
    for (std::size_t hist_offset = 0;
         hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        if (defer_material_packets)queue.fill(material_task_counts,std::uint64_t{0},2);
        const auto launch_primary_chunk = [&](auto uniform_water_path_tag, auto stage_tag) {
            constexpr int kKernelStage = decltype(stage_tag)::value;
            constexpr bool kQueueCopperChildren = kKernelStage == 9;
            constexpr bool kCtQueuedPrimaryPath = kKernelStage == 11;
            // Stage 12 is broadbeam stage 6 with host-proven CT-shared-electron
            // configuration folded at compile time (no physics difference).
            constexpr bool kMinibeamCtCache = JointResponse==0 &&
                (std::is_same_v<Pipeline,pipelines::ProtonMinibeam> || std::is_same_v<Pipeline,pipelines::CarbonMinibeam>);
            constexpr bool kSharedCtCell = kKernelStage == 16 || kKernelStage == 18;
            constexpr bool kLocalProtonCtPath = kKernelStage == 15 || kKernelStage == 16;
            constexpr bool kLocalCarbonCtPath = kKernelStage == 17 || kKernelStage == 18;
            constexpr bool kLocalElectronCtPath = kLocalProtonCtPath || kLocalCarbonCtPath;
            constexpr bool kPrimaryElasticTotals = kLocalProtonCtPath ||
                (kMinibeamCtCache && std::is_same_v<Pipeline,pipelines::ProtonMinibeam>);
            constexpr bool kSpeciesElasticTotals = kLocalCarbonCtPath ||
                (kMinibeamCtCache && std::is_same_v<Pipeline,pipelines::CarbonMinibeam>);
            constexpr bool kBroadBeamCtPath = kKernelStage == 12 || kKernelStage == 13 || kKernelStage == 14 || kLocalElectronCtPath;
            // 13/14 additionally fold the host-checked projectile (proton / C12).
            constexpr int kFoldZ = (kKernelStage == 13 || kLocalProtonCtPath) ? 1 : (kKernelStage == 14 || kLocalCarbonCtPath) ? 6 : 0;
            constexpr int kFoldA = (kKernelStage == 13 || kLocalProtonCtPath) ? 1 : (kKernelStage == 14 || kLocalCarbonCtPath) ? 12 : 0;
            static_assert(!kBroadBeamCtPath || (BroadBeam && (kLocalElectronCtPath ? JointResponse==0 : JointResponse==3)));
            constexpr int kPrimaryStage = kCtQueuedPrimaryPath ? 4 : kKernelStage == 9 ? 3 : kBroadBeamCtPath ? 6 : kKernelStage;
            constexpr bool kUniformWaterPrimaryPath = decltype(uniform_water_path_tag)::value;
            // Host eligibility requires global Urban for every split beamline.
            // Downstream CT material handling must not select upstream physics.
            constexpr bool kUrbanBeamlinePath = kUniformWaterPrimaryPath || kPrimaryStage == 3;
            const auto ct_runtime_global_urban = global_urban;
            const auto ct_runtime_enable_ct_grid = enable_ct_grid;
            const auto ct_runtime_use_unified_water = use_unified_water;
            const auto ct_runtime_enable_voxel_scoring = enable_voxel_scoring;
            const auto ct_runtime_enable_multiple_scattering = enable_multiple_scattering;
            const auto ct_runtime_enable_inelastic = enable_inelastic;
            const auto ct_runtime_use_schneider_primary_xs = use_schneider_primary_xs;
            const auto ct_runtime_use_material_electron = use_material_electron;
            const auto ct_runtime_defer_material_packets = defer_material_packets;
            const auto ct_runtime_primary_atomic_number = primary_atomic_number;
            const auto ct_runtime_primary_mass_number = primary_mass_number;
            const auto ct_runtime_inverse_mass_number = inverse_mass_number;


            return queue.parallel_for<PrimaryTransportKernel<
                Pipeline, kUniformWaterPrimaryPath, kKernelStage, BroadBeam, JointResponse>>(
            sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                // Constexpr-selected values remove unused physics branches;
                // the host eligibility test proves their runtime equivalence.
                const auto global_urban = kCtQueuedPrimaryPath ? true : (BroadBeam ? false : ct_runtime_global_urban);
                const auto enable_ct_grid = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_ct_grid;
                const auto use_unified_water = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? false : ct_runtime_use_unified_water;
                const auto enable_voxel_scoring = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_voxel_scoring;
                const auto enable_multiple_scattering = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_multiple_scattering;
                const auto enable_inelastic = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_inelastic;
                const auto use_schneider_primary_xs = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_use_schneider_primary_xs;
                const auto use_material_electron = kLocalElectronCtPath ? false : (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_use_material_electron;
                const auto defer_material_packets = kLocalElectronCtPath ? false : (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_defer_material_packets;
                const auto primary_atomic_number = kCtQueuedPrimaryPath ? 1 : (kFoldZ ? kFoldZ : ct_runtime_primary_atomic_number);
                const auto primary_mass_number = kCtQueuedPrimaryPath ? 1 : (kFoldA ? kFoldA : ct_runtime_primary_mass_number);
                const auto inverse_mass_number = kCtQueuedPrimaryPath ? 1.0F : (kFoldA ? 1.0F / static_cast<float>(kFoldA) : ct_runtime_inverse_mass_number);
                const bool enable_minibeam = BroadBeam ? false : primary_minibeam_enabled;


                const auto lane = item.get_global_linear_id();
                if (lane >= chunk_count) {
                    return;
                }
                auto global_history = hist_offset + lane;
                if constexpr (kPrimaryStage == 3) {
                    if (beamline_order != nullptr) global_history = hist_offset + beamline_order[lane];
                }

                PrimaryMediumEntryState entry{};
#include "detail/sycl_primary_beamline.inc"
                if constexpr (kPrimaryStage == 3) {
                    // Partition all histories, never discard a terminated track.
                    // The two ends cannot overlap: active + inactive <= chunk_count.
                    const bool active = entry.energy_MeV > energy_cutoff_MeV;
                    const auto rank = sycl::atomic_ref<std::uint64_t,
                        sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                            primary_entry_counts[active ? 0 : 1]).fetch_add(1U);
                    primary_medium_entry[active ? rank : chunk_count - 1 - rank] = entry;
                } else {
                    if constexpr (kPrimaryStage == 4) {
                        entry = primary_medium_entry[water_energy_order ? water_energy_order[lane] : lane];
                        global_history = entry.global_history;
                    }
                const auto rng_history = entry.rng_history;
                const auto spot_seed = entry.spot_seed;
                const auto spot_initial_energy_MeV = entry.spot_initial_energy_MeV;
                auto energy_MeV = entry.energy_MeV;
                auto position_x_mm = entry.position_x_mm;
                auto position_y_mm = entry.position_y_mm;
                auto position_z_mm = entry.position_z_mm;
                auto direction_x = entry.direction_x;
                auto direction_y = entry.direction_y;
                auto direction_z = entry.direction_z;
                auto history_deposited_MeV = 0.0F;
                // Broadbeam: the grid in/out energy ledgers are reduced per
                // history instead of one single-address FP64 atomic per step.
                double grid_in_pending_MeV = 0.0, grid_out_pending_MeV = 0.0;
                auto grid_split = [&](bool in_grid, float value_MeV) {
                    if constexpr (BroadBeam) {
                        if (sycl::isfinite(value_MeV) && value_MeV > 0.0F) {
                            if (in_grid) grid_in_pending_MeV += value_MeV;
                            else grid_out_pending_MeV += value_MeV;
                        }
                    } else grid_deposit_split_device(grid_deposited_in_device, grid_deposited_out_device, in_grid, value_MeV);
                };
                MaterialBirthTask grouped_task{};
                std::uint32_t grouped_count=0;
                double grouped_weight=0;
                // Separate stream: never consume transport or selected replay RNG.
                std::uint64_t group_draw=0;
                auto flush_electron_group=[&]() {
                    if(!grouped_count)return;
                    grouped_task.weight=grouped_weight;
                    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                        sycl::access::address_space::global_space> produced(material_task_counts[0]);
                    const auto slot=produced.fetch_add(1);
                    if(slot<material_task_capacity)material_tasks[slot]=grouped_task;
                    grouped_count=0;grouped_weight=0;
                };

                std::uint64_t local_schneider_rate_queries = 0;
                std::uint32_t steps = 0;
                // Research-only water Urban state (one fMinimal track per
                // primary history; first water segment starts at a boundary).
                UrbanV2TrackState global_urban_state{};
                bool global_urban_boundary=true;
                const int global_urban_ion=(kUniformWaterPrimaryPath || global_urban)?urban_device.projectile(primary_atomic_number,primary_mass_number):-1;
                bool water_urban_escaped_geometry = false;

                double pending_primary_depth_MeV = 0.0;
                int pending_primary_bin = 0;
                double pending_primary_voxel_MeV = 0.0;
                std::size_t pending_primary_voxel = 0;

                float nuclear_tau_remaining = 0.0F;
                HadronicRateCache primary_hadronic_cache;
                bool nuclear_tau_active = false;
                std::uint32_t nuclear_tau_rng_step = 0;
                bool primary_inelastic_occurred = false;
                bool unified_primary_escaped_ct = false;
                UnifiedEmClock unified_primary_clock;
                UnifiedEmState unified_primary_state;

                std::uint64_t unified_primary_counter=0;
                const int unified_primary_species=unified_device.species_index(primary_atomic_number,primary_mass_number);

                std::uint32_t interaction_section = 0;
                float interaction_density = 0.0F;

                const std::uint32_t max_primary_steps = maximum_primary_steps;
                while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    const auto escaped_z =
                        position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                    if (escaped_z) break;
                    if((kUniformWaterPrimaryPath || global_urban) && !((!kUniformWaterPrimaryPath && enable_ct_grid)) &&
                       urban_global_outside_water({position_x_mm,position_y_mm,position_z_mm},
                           {direction_x,direction_y,direction_z},urban_water_half_width_mm,phantom_length_mm)) {
                        water_urban_escaped_geometry=true;break;
                    }

                    const auto absolute_direction_z = sycl::fabs(direction_z);

                    auto bin = direction_z < 0.0F
                                   ? static_cast<int>(
                                         sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                   : static_cast<int>(
                                         sycl::floor(position_z_mm / depth_bin_width_mm));
                    bool ct_score_cell_valid = false;
                    if constexpr (kSharedCtCell)
                        ct_score_cell_valid = bin >= 0 && bin < static_cast<int>(number_of_bins);
                    bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));

                    auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                    auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                    if ((enable_voxel_scoring)) {
                        const auto x_coordinate =
                            (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                        const auto y_coordinate =
                            (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                        voxel_x = direction_x < 0.0F
                                      ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(x_coordinate));
                        voxel_y = direction_y < 0.0F
                                      ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(y_coordinate));
                        if constexpr (kSharedCtCell)
                            ct_score_cell_valid = ct_score_cell_valid &&
                                voxel_x >= 0 && voxel_x < static_cast<int>(voxel_bins_x) &&
                                voxel_y >= 0 && voxel_y < static_cast<int>(voxel_bins_y);
                        voxel_x = sycl::max(
                            0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                        voxel_y = sycl::max(
                            0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                    }

                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const auto scored=urban_voxel_cell({position_x_mm,position_y_mm,position_z_mm},
                            {direction_x,direction_y,direction_z},{voxel_min_x_mm,voxel_min_y_mm,0},
                            {voxel_max_x_mm,voxel_max_y_mm,phantom_length_mm},
                            {voxel_size_x_mm,voxel_size_y_mm,depth_bin_width_mm},voxel_bins_x,voxel_bins_y,number_of_bins);
                        if(scored.valid){voxel_x=scored.ix;voxel_y=scored.iy;bin=scored.iz;}
                    }
                    const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                             static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                             static_cast<std::size_t>(voxel_x);


                    const auto energy_MeVu = energy_MeV * inverse_mass_number;
                    auto local_density_g_per_cm3 = water_density_g_per_cm3;
                    std::uint8_t ct_material = 2;
                    auto in_ct = false;
                    std::size_t ct_material_index=0;
                    if (((!kUniformWaterPrimaryPath && enable_ct_grid))) {
                        float ct_rho = water_density_g_per_cm3;
                        if constexpr (kSharedCtCell) {
                            // Use the directed, unclamped scoring-cell validity.
                            // The clamped scorer index is read only when inside.
                            in_ct = ct_score_cell_valid;
                            if (in_ct) {
                                ct_rho = ct_density_device[voxel_index];
                                ct_material = ct_material_device[voxel_index];
                            }
                        } else {
                        in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                          ct_origin_x, ct_origin_y, ct_origin_z, ct_spacing_x,
                                          ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz,
                                          ct_density_device, ct_material_device, ct_rho,
                                          ct_material, direction_x, direction_y, direction_z, kMinibeamCtCache ? &ct_material_index : nullptr);
                        }
                        if (in_ct) {
                            local_density_g_per_cm3 = ct_rho;
                        }
                    }

                    // Mirror secondary CT escape: the EM package has no exterior material.
                    if(((!kUniformWaterPrimaryPath && enable_ct_grid)) && !in_ct) {
                        unified_primary_escaped_ct=true;
                        break;
                    }

                    float step_mm = 0.0F;
                    UnifiedEmStep unified_primary_pre;
                    UnifiedEmPoint unified_primary_point;
                    float unified_primary_rate=0,unified_primary_distance=std::numeric_limits<float>::infinity();
                    auto unified_uniform=[&](){return (rng::random_u32(spot_seed,rng_history,unified_primary_counter++,120)>>8)*0x1p-24f;};
                    auto unified_count=[&](int index,std::uint64_t count=1){
                        if(count==0)return; // zero-increment diagnostics must not touch globals
                        {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(unified_audit[static_cast<std::size_t>(index)*kUnifiedEmAuditShards+lane%kUnifiedEmAuditShards]);a.fetch_add(count);
                            }
                    };
                    {
                        const int section=((!kUniformWaterPrimaryPath && enable_ct_grid))?(in_ct?int(ct_material):-2):-1;
                        if(!unified_primary_state.valid || unified_primary_state.section!=section ||
                           unified_primary_state.density!=local_density_g_per_cm3)
                        {
                            if constexpr(kSharedCtCell && (kLocalProtonCtPath || kLocalCarbonCtPath))
                                unified_primary_state=unified_device.select_ct(section,local_density_g_per_cm3,unified_primary_species,voxel_index);
                            else if constexpr(kMinibeamCtCache)
                                unified_primary_state=unified_device.select_ct(section,local_density_g_per_cm3,unified_primary_species,ct_material_index);
                            else unified_primary_state=unified_device.select(section,local_density_g_per_cm3,unified_primary_species);
                        }
                        if(!unified_primary_state.covers(energy_MeV)){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,1,section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,0);
                            unified_count(0);break;}
                        if(unified_primary_clock.last_section!=unified_primary_state.section || unified_primary_clock.last_density!=unified_primary_state.density)primary_hadronic_cache.valid=false;
                        if constexpr(kKernelStage==13 || kKernelStage==14 || kLocalElectronCtPath){
                            // Reuse the material view through this step's loss
                            // draw, without adding cross-step/resume state.
                            unified_primary_point=unified_primary_state.lo();
                            // Carbon CT primary retains within-step loss slopes:
                            // omitting them produces a coherent beam-direction
                            // dose bias in low-density lung. Proton broadbeam
                            // keeps its separately scoped step-start approximation.
                            const auto point_pre=unified_primary_point.template prepare<kLocalElectronCtPath,kLocalProtonCtPath,kLocalProtonCtPath>(energy_MeV);
                            unified_primary_pre={point_pre,point_pre};
                        }else unified_primary_pre=unified_primary_state.prepare(energy_MeV);
                        // Match ionIoni proposal segmentation while retaining aggregate loss.
                        // Debit only the true path accepted after geometry/nuclear competition.
                        if((kUniformWaterPrimaryPath || global_urban)) {
                            unified_primary_rate=unified_primary_clock.update(unified_primary_state,energy_MeV,
                                unified_primary_pre.lo.factor);
                            if(!unified_primary_clock.clock.active)
                                unified_primary_clock.clock.arm(-sycl::log(sycl::fmax(unified_uniform(),0x1p-24f)));
                            unified_primary_distance=unified_primary_clock.clock.distance(unified_primary_rate);
                        } else {
                            unified_primary_clock.last_section=unified_primary_state.section;
                            unified_primary_clock.last_density=unified_primary_state.density;
                        }
                        step_mm=sycl::fmin(unified_primary_state.step(unified_primary_pre),unified_primary_distance);
                        // Bound combined mean loss, not the old restricted range alone.
                        const float delta_sp=unified_primary_state.mix(unified_primary_pre.lo.delta_stopping,unified_primary_pre.hi.delta_stopping);
                        if(delta_sp>0) step_mm=sycl::fmin(step_mm,maximum_mean_loss_fraction*energy_MeV/sycl::fmax(1e-12f,delta_sp+unified_primary_state.mix(unified_primary_pre.lo.stopping,unified_primary_pre.hi.stopping)));
                    }

                    // Non-Urban CT steps must honor the configured physical
                    // cap before face/collision competition. Urban already
                    // receives the same cap in its proposal below.
                    if constexpr(BroadBeam) if (enable_ct_grid && !(kUniformWaterPrimaryPath || global_urban))
                        step_mm=sycl::fmin(step_mm,maximum_step_mm);

                    if (!(kUniformWaterPrimaryPath || global_urban) && absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                        if (dz_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dz_step);
                        }
                    }
                    CtFaceClampResult ct_clamp_res{step_mm, false, 0};
                    if (!(kUniformWaterPrimaryPath || global_urban) && ((!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct) {
                        ct_clamp_res = clamp_step_to_ct_faces_exact(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz);
                        step_mm = ct_clamp_res.step_mm;
                    }

                    UrbanGlobalProposal global_urban_proposal{};
                    CorrelatedScatteringStep global_urban_scattering{};
                    int global_urban_material=-1;
                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const auto cell=in_ct?urban_voxel_cell(
                            {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},
                            {ct_origin_x,ct_origin_y,ct_origin_z},
                            {ct_origin_x+ct_nx*ct_spacing_x,ct_origin_y+ct_ny*ct_spacing_y,ct_origin_z+ct_nz*ct_spacing_z},
                            {ct_spacing_x,ct_spacing_y,ct_spacing_z},ct_nx,ct_ny,ct_nz):
                            urban_global_water_cell({position_x_mm,position_y_mm,position_z_mm},
                                {direction_x,direction_y,direction_z},{voxel_min_x_mm,voxel_min_y_mm,0},
                                {voxel_max_x_mm,voxel_max_y_mm,phantom_length_mm},
                                {voxel_size_x_mm,voxel_size_y_mm,depth_bin_width_mm},voxel_bins_x,voxel_bins_y,number_of_bins,
                                urban_water_half_width_mm);
                        global_urban_material=urban_device.material(in_ct?int(ct_material):-1,local_density_g_per_cm3);
                        // Lifetime includes proposal, nuclear shortening and
                        // final Urban sampling below; never stored in a queue.
                        if(in_ct)unified_primary_point=unified_primary_state.lo();
                        global_urban_proposal=urban_global_prepare(urban_device,global_urban_material,global_urban_ion,
                            energy_MeV,sycl::fmin(step_mm,maximum_step_mm),
                            {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},
                            cell,global_urban_boundary,global_urban_state,spot_seed,rng_history,steps,
                            local_density_g_per_cm3,
                            in_ct?&unified_primary_point:nullptr,unified_primary_pre.lo.factor);
                        if(!global_urban_proposal.valid) {
                            record_unified_em_failure(unified_failure_count,unified_failure_records,70,
                                in_ct?int(ct_material):-1,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,step_mm);
                            unified_count(0);break;
                        }
                        step_mm=global_urban_proposal.true_path;
                    }
                    float accepted_loss_path_mm = step_mm;
                    bool urban_range_limited = false;
                    bool inelastic_this_step = false;
                    // Elastic candidates are resolved only by the configured CT/water bank.
                    bool ct_elastic_this_step = false;
                    if ((enable_inelastic) &&
                        energy_MeV > energy_cutoff_MeV) {
                        const auto cur_e_u = energy_MeV * inverse_mass_number;
                        if ((use_schneider_primary_xs)) {
                            if (in_ct || ((kUniformWaterPrimaryPath || use_unified_water))) {
                                const std::uint32_t section_id = ((kUniformWaterPrimaryPath || use_unified_water)) ? 255U : static_cast<std::uint32_t>(
                                    sycl::min(static_cast<std::uint32_t>(ct_material), 24U));
                                // v3: hazard from the masked rate-binary partials
                                // (single source with the target sampler); v1
                                // keeps the CSV XS table EXACTLY.
                                const float mass_rate = schneider_ct_device_ctx.primary_rates(
                                    section_id, cur_e_u).total;
                                // Density enters exactly once, in the total
                                // hazard; target fractions from the sampler CDF
                                // are density-independent.
                                // All-ion elastic hazard: same optical-depth
                                // competition as inelastic; the branch below
                                // splits them exclusively, so no double counting.
                                float macro_el_ct = 0.0F;
                                const bool elastic_armed = use_all_elastic;
                                if (elastic_armed) {
                                    const float rate=all_elastic.template transport_total_rate<kPrimaryElasticTotals,kSpeciesElasticTotals>(primary_elastic_projectile,((kUniformWaterPrimaryPath || use_unified_water))?25:section_id,cur_e_u);
                                    if(rate<0) {
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[4]).fetch_add(1);
                                        break;
                                    }
                                    macro_el_ct=local_density_g_per_cm3*rate;
                                }
                                float macro_inelastic = local_density_g_per_cm3 * mass_rate;
                                {
                                // Cache only the process whose post-step acceptance uses
                                // this rate. Elastic retains its own local hazard and must
                                // never enter the inelastic acceptance denominator.
                                macro_inelastic = (enable_inelastic)
                                    ? primary_hadronic_cache.update(cur_e_u, macro_inelastic)
                                    : 0.0F;
                                }
                                const float macro_tot = macro_inelastic + macro_el_ct;
                                ++local_schneider_rate_queries;

                                float u_nuc = 1.0F;
                                if (!nuclear_tau_active) {
                                    u_nuc = rng::uniform01(
                                        spot_seed, rng_history, nuclear_tau_rng_step++, 8);
                                }
                                const bool collision = consume_schneider_optical_depth_segment(
                                    nuclear_tau_remaining, nuclear_tau_active,
                                    accepted_loss_path_mm, macro_tot, u_nuc);
                                bool elastic_branch = false;
                                if (collision && elastic_armed && macro_tot > 0.0F) {
                                    const float u_br_el = rng::uniform01(
                                        spot_seed, rng_history, steps, 9);
                                    elastic_branch = (u_br_el * macro_tot < macro_el_ct);
                                }
                                if (elastic_branch) {
                                    ct_elastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                } else if (collision && (enable_inelastic)) {
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryHazards);
                                    if (schneider_inelastic_device != nullptr) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            nuc_ref(*schneider_inelastic_device);
                                        nuc_ref.fetch_add(1U);
                                    }

                                    inelastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                }
                            }
                        }
                    }

                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const bool ionization_proposal=accepted_loss_path_mm>=unified_primary_distance;
                        unified_primary_clock.clock.consume(accepted_loss_path_mm,unified_primary_rate,ionization_proposal);
                        global_urban_scattering=urban_global_finalize(global_urban_proposal,accepted_loss_path_mm,
                            spot_seed,rng_history,steps);
                        const int reason=urban_step_field_reason(global_urban_scattering);
                        if(reason) {
                            record_unified_em_failure(unified_failure_count,unified_failure_records,80+reason,
                                in_ct?int(ct_material):-1,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,accepted_loss_path_mm);
                            unified_count(0);break;
                        }
                        step_mm=global_urban_scattering.final_geom_path_mm;
                        urban_range_limited=global_urban_scattering.range_terminal;
                    }
                    // Diagnostic only: CT face clamping above guarantees the
                    // segment stays in its starting CT voxel. Do not score a
                    // laterally escaped segment into a clamped edge voxel.

                    float deposited_MeV = 0.0F;
                    if (urban_range_limited) deposited_MeV = energy_MeV;
                    float birth_expected_loss_MeV=urban_range_limited?energy_MeV:0.f;
                    if(!urban_range_limited){
                        auto draw=[&](){
                            if constexpr(kKernelStage==13 || kKernelStage==14 || kLocalElectronCtPath)return unified_em_explicit_loss<kLocalProtonCtPath,true,kLocalElectronCtPath>(
                                unified_primary_state,energy_MeV,accepted_loss_path_mm,enable_energy_straggling,
                                unified_primary_pre,unified_uniform,&unified_primary_point);
                            else return unified_em_loss<kPrimaryStage==7>(
                                unified_primary_state, energy_MeV, accepted_loss_path_mm,
                                enable_energy_straggling, unified_primary_pre,
                                unified_uniform);
                        }();
                        if(!draw.valid){
                            record_unified_em_failure(
                                unified_failure_count, unified_failure_records,
                                20 + static_cast<int>(draw.failure_stage),
                                unified_primary_state.section,
                                primary_atomic_number, primary_mass_number,
                                energy_MeV, local_density_g_per_cm3,
                                accepted_loss_path_mm, draw.mean,
                                draw.delta_mean, draw.delta_variance);
                            unified_count(0);break;}
                        deposited_MeV=draw.loss;unified_count(1);unified_count(3,draw.proposed);unified_count(4,draw.accepted);
                        unified_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                        birth_expected_loss_MeV=draw.mean+draw.delta_mean;
                    }
                    if (enable_minibeam) {
                        deposited_MeV = sycl::fmin(
                            energy_MeV,
                            deposited_MeV);
                    }
                    auto local_voxel_deposit_MeV = deposited_MeV;
                    auto same_voxel_electron_packet_MeV = 0.0F;
                    auto delta_tail_escaped_scorer_MeV = 0.0F;
                    // Forward-redistributed energy leaves the source depth bin, so the
                    // 1-D depth scorers must not credit it at the source bin (the 3-D
                    // march deposits below credit the destination bins instead).
                    auto forward_shifted_MeV = 0.0F;
                    auto transverse_relocated_MeV = 0.0F;
                    auto transverse_escaped_MeV = 0.0F;
                    auto water_physical_escape_MeV=0.0F;
                    float response_scored_separately_MeV=0.0F;
                    float material_untracked_MeV=0;
                    // Only the primary-loop EM loss exists here. Nuclear recoil
                    // and interaction deposits are scored later and must never
                    // be folded into this ionization response packet weight.
                    if(use_material_electron && (!enable_ct_grid || in_ct) && deposited_MeV>0) {
                        auto count=[&](int slot,std::uint64_t value) {
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);a.fetch_add(value);
                        };
                        auto untracked=[&](double weight,bool photon) {
                            material_untracked_MeV+=static_cast<float>(weight);
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(material_untracked_device[photon?0:1]);
                            a.fetch_add(static_cast<std::uint64_t>(weight*1e6+0.5));
                        };
                        // A separate deterministic packet stream does not advance the
                        // carbon transport RNG. Rebinding consumes this stream only.
                        std::uint64_t packet_rng=spot_seed ^ ((rng_history+1)*0x9e3779b97f4a7c15ULL) ^
                            (static_cast<std::uint64_t>(steps)<<32) ^ 0x6a09e667f3bcc909ULL;
                        auto uniform=[&]() -> double {return float(electron_fp32::random_draw(packet_rng));};
                        count(0,1);
                        // Preserve the existing RNG draw even though its value is unused.
                        static_cast<void>(uniform());
                        (void)uniform(); // Preserve the independent birth CDF draw and RNG order.
                        const auto birth_cdf_bits=packet_rng;
                        const double along=uniform();
                        const auto float_geometry=electron_fp32::ElectronCtGeometry{{ct_nx,ct_ny,ct_nz},
                            {float(ct_origin_x),float(ct_origin_y),float(ct_origin_z)},
                            {float(ct_spacing_x),float(ct_spacing_y),float(ct_spacing_z)},ct_density_device,ct_material_device};
                        const auto birth_position=[&]() -> std::array<double,3> {
                            const auto p=electron_fp32::electron_ct_birth_position(float_geometry,
                                {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},step_mm,float(along));
                            return {p[0],p[1],p[2]};
                        }();
                        const auto birth_material=[&]() -> ElectronCtPointMaterial {
                            const auto p=electron_fp32::electron_ct_point_material(float_geometry,
                                {electron_fp32::Coordinate(birth_position[0]),electron_fp32::Coordinate(birth_position[1]),electron_fp32::Coordinate(birth_position[2])},
                                {direction_x,direction_y,direction_z});return {p.valid,p.density,p.section};
                        }();
                        electron_fp32::ElectronBirthPartition birth;
                        float birth_parent_MeVu=energy_MeVu,birth_cut_MeV=0.f;
                        bool selected_birth=true;
                        if constexpr(BroadBeam && JointResponse==3) {
                            if(birth_material.valid) {
                                const int bx=static_cast<int>(sycl::floor((birth_position[0]-ct_origin_x)/ct_spacing_x));
                                const int by=static_cast<int>(sycl::floor((birth_position[1]-ct_origin_y)/ct_spacing_y));
                                const int bz=static_cast<int>(sycl::floor((birth_position[2]-ct_origin_z)/ct_spacing_z));
                                if(bx>=0&&by>=0&&bz>=0&&bx<int(ct_nx)&&by<int(ct_ny)&&bz<int(ct_nz)) {
                                    selected_birth=broadbeam_electron_mask[(std::size_t(bz)*ct_ny+by)*ct_nx+bx];
                                    // Unselected ionization remains local by explicit research policy.
                                    if(!selected_birth)birth.valid=true;
                                }
                            }
                        }
                        if(birth_material.valid && selected_birth) {
                            if(cut_birth26) {
                                const auto composition=cut_birth26[birth_material.section];
                                // Aggregate expectation is not a realizable
                                // energy jump when it exceeds the parent's KE.
                                const float parent=sycl::fmax(0.f,energy_MeV-float(along)*sycl::fmin(energy_MeV,birth_expected_loss_MeV));
                                // Electron section25 is Water_75eV; ion EM uses section-1.
                                const auto state=unified_device.select(birth_material.section==25?-1:birth_material.section,birth_material.density,unified_primary_species);
                                if(state.valid&&birth_material.density>=composition.density_low&&
                                   birth_material.density<=composition.density_high) {
                                    birth_parent_MeVu=parent*inverse_mass_number;birth_cut_MeV=state.cut();
                                    const float mass=primary_mass_number==1?938.272013f:11174.863388f;
                                    const bool bragg=primary_mass_number==1&&parent<=2.f;
                                    if(bragg)birth_cut_MeV=sycl::fmax(birth_cut_MeV,.00025f*mass/938.272013f);
                                    const bool support=electron_birth_maximum(birth_parent_MeVu,primary_mass_number)>birth_cut_MeV;
                                    birth.node=birth_material.section;
                                    if(!support) {
                                        // Native delta production is exactly
                                        // zero below the cut, including below
                                        // the ion table's energy minimum.
                                        birth.valid=true;birth.fraction=0.f;
                                    } else if(state.covers(parent)) {
                                        float restricted,delta;
                                        if constexpr(kKernelStage==13 || kKernelStage==14 || kLocalElectronCtPath)
                                            state.lo().birth_stopping(parent,restricted,delta);
                                        else {
                                            const auto point=state.prepare(parent);
                                            restricted=state.mix(point.lo.stopping,point.hi.stopping);
                                            delta=state.mix(point.lo.delta_stopping,point.hi.delta_stopping);
                                        }
                                        const float total=restricted+delta;
                                        birth.valid=sycl::isfinite(total)&&total>0.f&&restricted>=0.f&&delta>=0.f;
                                        birth.fraction=delta/total;
                                    }
                                }
                            }
                        }
                        const auto ti=birth.node;
                        if(!birth.valid) {
                            if(cut_birth26){
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> diagnostic(material_untracked_device[2]);
                                if(diagnostic.fetch_add(1)==0){
                                    MaterialPacketFailure failure{};failure.history=global_history;failure.step=steps;failure.rng=packet_rng;
                                    auto& c=failure.before.cursor;c.section=birth_material.section;c.density_g_cm3=birth_material.density;
                                    c.physical_density_g_cm3=birth_material.density;c.energy_MeV=birth_parent_MeVu*primary_mass_number;
                                    c.position={electron_fp32::Coordinate(birth_position[0]),electron_fp32::Coordinate(birth_position[1]),electron_fp32::Coordinate(birth_position[2])};
                                    failure.before.weight_MeV=deposited_MeV;failure.after.status=ElectronPacketStatus::invalid;
                                    *material_failure_device=failure;
                                }
                            }
                            count(1,1);untracked(deposited_MeV,false);
                            local_voxel_deposit_MeV=0;forward_shifted_MeV+=deposited_MeV;
                        } else if(birth.fraction>0 &&
                                  deposited_MeV*static_cast<float>(birth.fraction)>0.0F) {
                            // FP32 loss*yield can underflow to zero. A zero-weight
                            // packet carries no energy and must not enter transport.
                            // Response packets partition the already sampled total loss
                            // by the pinned energy-weighted birth fraction, as in water.
                            // Do not combine a binned birth law with individual aggregate
                            // delta draws: their cut-onset support need not coincide.
                            // No energy is added and the transport loss/RNG is unchanged.
                            const float weight=deposited_MeV*static_cast<float>(birth.fraction);
                            local_voxel_deposit_MeV-=weight;forward_shifted_MeV+=weight;

                                MaterialBirthTask task{QueuedElectronBirth{birth_parent_MeVu,birth_cdf_bits,birth.node,birth_cut_MeV},birth_position,{direction_x,direction_y,direction_z},
                                    birth_material.density,packet_rng,global_history,static_cast<std::uint32_t>(steps),
                                    static_cast<std::uint32_t>(ti),weight};
                                grouped_weight=float(grouped_weight)+weight;
                                bool select=grouped_count==0;
                                if(!select) {
                                    const auto block=carbon::rng::philox4x32_10(
                                        {static_cast<std::uint32_t>(rng_history),static_cast<std::uint32_t>(rng_history>>32),
                                         static_cast<std::uint32_t>(group_draw),static_cast<std::uint32_t>(group_draw>>32)},
                                        static_cast<std::uint32_t>(spot_seed)^0xe1ec7a91U,
                                        static_cast<std::uint32_t>(spot_seed>>32)^0x47525031U);
                                    ++group_draw;
                                    const std::uint64_t bits=(std::uint64_t(block[0])<<32)|block[1];
                                    select=electron_fp32::Draw{bits}<weight/float(grouped_weight);
                                }
                                if(select)grouped_task=task;
                                if(++grouped_count==material_group_size)flush_electron_group();
                                // Pending W is excluded from deposited/grid energy until
                                // the task terminates. No untracked sink is recorded here.
                                material_untracked_MeV+=weight;
                        }
                    }

                    if (bin != pending_primary_bin) {
                        if (pending_primary_depth_MeV > 0.0) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_primary_bin]);
                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
                            pending_primary_depth_MeV = 0.0;
                        }

                        pending_primary_bin = bin;
                    }
                    if ((enable_voxel_scoring) && voxel_index != pending_primary_voxel) {
                        if (pending_primary_voxel_MeV > 0.0 && pending_primary_voxel >= 0 &&
                            pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                            atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));

                        }
                        pending_primary_voxel_MeV = 0.0;

                        pending_primary_voxel = voxel_index;
                    }

                    // Keep the legacy unrestricted-depth escape convention, but
                    // remove relocated energy now tallied at its destination.
                    pending_primary_depth_MeV += deposited_MeV - forward_shifted_MeV -
                        transverse_relocated_MeV;
                    pending_primary_depth_MeV += same_voxel_electron_packet_MeV;
                    if ((enable_voxel_scoring) && voxel_index >= 0) {
                        pending_primary_voxel_MeV += local_voxel_deposit_MeV;
                        pending_primary_voxel_MeV += same_voxel_electron_packet_MeV;

                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(
                                deposited_MeV - forward_shifted_MeV -
                                transverse_relocated_MeV - transverse_escaped_MeV)+
                                static_cast<DepthAtomicT>(same_voxel_electron_packet_MeV));
                        }
                    }
                    history_deposited_MeV += deposited_MeV-water_physical_escape_MeV-material_untracked_MeV;
                    grid_split(
                        (enable_voxel_scoring) && voxel_index >= 0,
                        deposited_MeV - delta_tail_escaped_scorer_MeV-water_physical_escape_MeV-
                            material_untracked_MeV-response_scored_separately_MeV);
                    grid_split(false, delta_tail_escaped_scorer_MeV);

                    const float seg_dir_x = direction_x;
                    const float seg_dir_y = direction_y;
                    const float seg_dir_z = direction_z;
                    Direction3F water_scattering_displacement{
                        0.0F, 0.0F, 0.0F};

                    if ((enable_multiple_scattering)) {
                        auto radiation_length_g_per_cm2 =
                            static_cast<float>(active_water_radiation_length);
                        if (((!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct && (enable_ct_material_mcs)) {
                            radiation_length_g_per_cm2 = static_cast<float>(
                                select_transport_radiation_length_g_per_cm2(
                                    true, in_ct, ct_material,
                                    (enable_ct_material_mcs),
                                    water_radiation_length_g_per_cm2));
                        }
                        // Generic primary FE path. Yields to the minibeam water
                        // branch below when a non-legacy minibeam primary
                        // model is selected (FE-tail or Urban v2); otherwise
                        // the minibeam branch would be dead code whenever the
                        // generic fermi_eyges model is on.
                        if((kUniformWaterPrimaryPath || global_urban)) {
                            direction_x=global_urban_scattering.direction.x;
                            direction_y=global_urban_scattering.direction.y;
                            direction_z=global_urban_scattering.direction.z;
                            water_scattering_displacement=global_urban_scattering.displacement_mm;
                            global_urban_state=global_urban_proposal.state;
                            urban_global_count(urban_step_counts,global_urban_ion,false);
                        } else if (primary_fermi_eyges
                        ) {
                            auto observation_path_mm = -1.0F;

                            const auto correlated =
                                ion_fermi_eyges_transport_step(
                                    Direction3F{direction_x, direction_y,
                                                direction_z},
                                    energy_MeV, primary_atomic_number,
                                    primary_mass_number, deposited_MeV, step_mm,
                                    c12_fermi_eyges_max_segment_mm,
                                    local_density_g_per_cm3,
                                    radiation_length_g_per_cm2, spot_seed,
                                    rng_history,
                                    static_cast<std::uint64_t>(steps), 40U,
                                    observation_path_mm, primary_fe_mass_per_u, fe_gaussian_only);

                            direction_x = correlated.direction.x;
                            direction_y = correlated.direction.y;
                            direction_z = correlated.direction.z;
                            water_scattering_displacement =
                                correlated.displacement_mm;
                        }
                    }

                    position_x_mm += seg_dir_x * step_mm +
                        water_scattering_displacement.x;
                    position_y_mm += seg_dir_y * step_mm +
                        water_scattering_displacement.y;
                    position_z_mm += seg_dir_z * step_mm +
                        water_scattering_displacement.z;
                    if((kUniformWaterPrimaryPath || global_urban)) {
                        global_urban_boundary=global_urban_scattering.boundary_crossed;
                        const auto p=urban_voxel_snap_endpoint(global_urban_proposal.cell,step_mm,
                            {position_x_mm,position_y_mm,position_z_mm});
                        position_x_mm=p.x;position_y_mm=p.y;position_z_mm=p.z;
                    }

                    if (!(kUniformWaterPrimaryPath || global_urban) && ((!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct && ct_clamp_res.hit_face && !inelastic_this_step && !ct_elastic_this_step) {
                        if ((ct_clamp_res.axis_mask & 1) != 0 && sycl::fabs(seg_dir_x) > 1.0e-6F) {
                            const float fx = (position_x_mm - ct_origin_x) / ct_spacing_x;
                            const int face_x = static_cast<int>(sycl::round(fx));
                            const float b_x = ct_origin_x + static_cast<float>(face_x) * ct_spacing_x;
                            position_x_mm = sycl::nextafter(b_x, seg_dir_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 2) != 0 && sycl::fabs(seg_dir_y) > 1.0e-6F) {
                            const float fy = (position_y_mm - ct_origin_y) / ct_spacing_y;
                            const int face_y = static_cast<int>(sycl::round(fy));
                            const float b_y = ct_origin_y + static_cast<float>(face_y) * ct_spacing_y;
                            position_y_mm = sycl::nextafter(b_y, seg_dir_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 4) != 0 && sycl::fabs(seg_dir_z) > 1.0e-6F) {
                            const float fz = (position_z_mm - ct_origin_z) / ct_spacing_z;
                            const int face_z = static_cast<int>(sycl::round(fz));
                            const float b_z = ct_origin_z + static_cast<float>(face_z) * ct_spacing_z;
                            position_z_mm = sycl::nextafter(b_z, seg_dir_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    if (!(kUniformWaterPrimaryPath || global_urban) && absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if (sycl::fabs(boundary_z_mm - position_z_mm) <= 1.0e-5F) {
                            position_z_mm = sycl::nextafter(
                                boundary_z_mm,
                                direction_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    energy_MeV -= deposited_MeV;

                    if(defer_material_packets && (ct_elastic_this_step || inelastic_this_step))flush_electron_group();

                    // Elastic endpoint: the all-ion bank supplies target-specific TOPAS
                    // samples. Supported recoils are queued; below-cutoff energy is scored locally.
                    if((use_all_elastic) && ct_elastic_this_step && energy_MeV>energy_cutoff_MeV &&
                       all_elastic.template transport_total_rate<kPrimaryElasticTotals,kSpeciesElasticTotals>(primary_elastic_projectile,((kUniformWaterPrimaryPath || use_unified_water))?25:interaction_section,energy_MeV*inverse_mass_number)==0) {
                        ct_elastic_this_step=false;
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[10]).fetch_add(1);
                    }
                    if (use_all_elastic && ct_elastic_this_step &&
                        energy_MeV > energy_cutoff_MeV) {
                        const float u_tgt = rng::uniform01(
                            spot_seed, rng_history, steps, 12);
                        const auto draw = all_elastic.draw(primary_elastic_projectile,
                            ((kUniformWaterPrimaryPath || use_unified_water))?25:interaction_section,energy_MeV,direction_x,direction_y,direction_z,
                            u_tgt,rng::uniform01(spot_seed,rng_history,steps,70),
                            rng::uniform01(spot_seed,rng_history,steps,71),rng::uniform01(spot_seed,rng_history,steps,11));
                        if(!draw.valid) {
                            sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[5]).fetch_add(1);
                            break;
                        }
                        const int target_z=draw.target_z;
                        const int target_idx=carbon::elastic_target_index_from_z(target_z);
                        const int target_a=draw.target_a;
                        if (target_idx >= 0) {
                            const auto scat=draw.outcome;
                            sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[0]).fetch_add(1);
                            energy_MeV = scat.projectile_ke_MeV;
                            direction_x = scat.proj_dir_x;
                            direction_y = scat.proj_dir_y;
                            direction_z = scat.proj_dir_z;
                            if ((carbon::get_charged_species_idx(target_z,target_a)>=0 ||
                                 recoil_stopping.projectile(target_z,target_a)>=0) && enable_secondary_transport &&
                                scat.recoil_ke_MeV > energy_cutoff_MeV &&
                                secondary_queue_device != nullptr) {
                                // Elastic recoil born accounting: mirrors the
                                // inelastic born/queued pair so the born
                                // conservation gate stays exact.
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedBorn);
                                auto count_ref = sycl::atomic_ref<
                                    uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *secondary_count_device);
                                const auto base_idx = count_ref.fetch_add(1U);
                                if (base_idx < max_secondaries) {
                                    SecondaryParticle proton{};
                                    proton.z = target_z;
                                    proton.a = target_a;
                                    if(carbon::get_charged_species_idx(target_z,target_a)<0)proton.generation=max_secondary_inelastic_generations;
                                    proton.energy_MeV = scat.recoil_ke_MeV;
                                    proton.pos_x_mm = position_x_mm;
                                    proton.pos_y_mm = position_y_mm;
                                    proton.pos_z_mm = position_z_mm;
                                    proton.dir_x = scat.recoil_dir_x;
                                    proton.dir_y = scat.recoil_dir_y;
                                    proton.dir_z = scat.recoil_dir_z;
                                    proton.weight = 1.0F;
                                    proton.parent_history = global_history;
                                    proton.rng_stream = rng::child_stream(
                                        rng::primary_descendant_stream(spot_seed, rng_history), rng::branch_tag(
                                            rng::branch_role_primary_charged, steps));
                                    secondary_queue_device[base_idx] = proton;
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[1]).fetch_add(scat.recoil_ke_MeV);
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryChargedQueued);
                                    if (ion_species_energy_device != nullptr) {
                                        ion_record_queued_secondary_birth_device(
                                            ion_species_energy_device, proton.z, proton.a,
                                            proton.energy_MeV);
                                    }
                                } else if (secondary_overflow_count_device != nullptr) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_ov(*secondary_overflow_count_device);
                                    atomic_ov.fetch_add(1U);
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[2]).fetch_add(scat.recoil_ke_MeV);sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[11]).fetch_add(1);
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_energy_device).fetch_add(scat.recoil_ke_MeV);
                                }
                            } else if(scat.recoil_ke_MeV>energy_cutoff_MeV) {
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[3]).fetch_add(1);
                                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(untracked_nuclear_device[global_history]).fetch_add(scat.recoil_ke_MeV);
                            } else if (scat.recoil_ke_MeV > 0.0F) {
                                if ((enable_voxel_scoring) && voxel_index >= 0 &&
                                    voxel_index < number_of_voxels) {
                                    pending_primary_voxel_MeV += scat.recoil_ke_MeV;
                                }
                                history_deposited_MeV += scat.recoil_ke_MeV;
                                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[0]).fetch_add(scat.recoil_ke_MeV);
                                pending_primary_depth_MeV += scat.recoil_ke_MeV;
                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                    (enable_voxel_scoring)&&voxel_index>=0,scat.recoil_ke_MeV);
                            }
                        }
                    }

                    if ((use_schneider_primary_xs) && (in_ct || ((kUniformWaterPrimaryPath || use_unified_water))) && inelastic_this_step) {

                        // Production Schneider CT primary inelastic collision.
                        // Exact (C12, target_Z) channel, bounded energy domain,
                        // stochastic bracketing. Tag 15 drives the bracket
                        // choice, tag 16 the intra-node event pick.
                        const float cur_primary_e_u = energy_MeV * inverse_mass_number;
                        const float u_target = rng::uniform01(spot_seed, rng_history, steps, 14);
                        // v3: masked draw from the same partials as the hazard.
                        // A non-positive result (masked hazard fired but
                        // slowing emptied every channel before the collision
                        // point) is a primary post-EM null collision: no
                        // package query, track continues, never
                        // UnsupportedTargets.
                        int target_z = 0;
                        {
                            const auto masked = schneider_ct_device_ctx.primary_rates(
                                interaction_section, cur_primary_e_u);
                            target_z = sample_masked_schneider_target_device(
                                masked.partials, u_target);
                            {
                            // Dedicated dimension 44; do not reuse source, loss,
                            // target, event or electron-response variates.
                            const bool accepted = primary_hadronic_cache.accept(
                                interaction_density * masked.total,
                                rng::uniform01(spot_seed, rng_history, steps, 44));
                            // Candidate-only: rejected proposal reuses the
                            // existing energy-preserving post-EM null route.
                            if (!accepted) target_z = 0;
                            }
                        }
                        if (target_z <= 0) {
                            // Null collision: retain the surviving track and complete this EM step.
                            inelastic_this_step = false;
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::PrimaryPostEmNullCollisions);
                            schneider_float_add_device(
                                schneider_float_device,
                                SchneiderFloatSlot::PostEmNullEnergy,
                                sycl::fmax(0.0F, energy_MeV));
                        } else {

                        const float u_bracket = rng::uniform01(spot_seed, rng_history, steps, 15);
                        const float u_event = rng::uniform01(spot_seed, rng_history, steps, 16);
                        const auto primary_lookup = cinel03_lookup_event_device(
                            schneider_ct_device_ctx.primary_energy_nodes,
                            schneider_ct_device_ctx.primary_node_count,
                            schneider_ct_device_ctx.primary_event_offsets,
                            schneider_ct_device_ctx.primary_event_indices,
                            schneider_ct_device_ctx.primary_total_events,
                            primary_atomic_number, primary_mass_number, target_z,
                            cur_primary_e_u,
                            u_bracket, u_event);
                        schneider_record_lookup_device(schneider_diag_device,
                                                       schneider_float_device,
                                                       true, primary_lookup,
                                                       energy_MeV);

                        if (primary_lookup.status != Cinel03LookupStatus::Hit) {
                            // Per-miss log: recompute the macro total exactly
                            // as at hazard time (density x mass, once).
                            // Per-miss macro total recomputed exactly as at
                            // hazard time: v3 uses the masked binary total
                            // (no CSV exists for v3), v1 the CSV XS table.
                            const float primary_miss_macro = interaction_density *
                                schneider_ct_device_ctx.primary_rates(interaction_section, cur_primary_e_u).total;
                            schneider_log_miss_device(
                                schneider_miss_device, schneider_miss_count_device,
                                kSchneiderMissLogCap, true,
                                primary_atomic_number, primary_mass_number, target_z,
                                interaction_section > 255 ? 255
                                                          : static_cast<std::uint8_t>(interaction_section),
                                0, primary_lookup, cur_primary_e_u, deposited_MeV,
                                primary_miss_macro, 0.0F, energy_MeV,
                                spot_initial_energy_MeV);
                            if (untracked_nuclear_device != nullptr) {
                                untracked_nuclear_device[global_history] = energy_MeV;
                            }
                            energy_MeV = 0.0F;
                            break;
                        }

                        // A hazard proposal (including a post-EM null collision)
                        // is not an inelastic reaction. Mark only an event hit.
                        primary_inelastic_occurred = true;
                        const auto& event = schneider_ct_device_ctx.primary_interactions[primary_lookup.event_index];
                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);

                        pending_primary_depth_MeV += local_deposit;
                        if ((enable_voxel_scoring) && voxel_index >= 0) {
                            pending_primary_voxel_MeV += local_deposit;
                        }
                        history_deposited_MeV += local_deposit;
                        grid_deposit_split_device(
                            grid_deposited_in_device, grid_deposited_out_device,
                            (enable_voxel_scoring) && voxel_index >= 0, local_deposit);

                        float charged_accounted_MeV = 0.0F;
                        float neutral_accounted_MeV = 0.0F;
                        float unsupported_accounted_MeV = 0.0F;

                        const std::uint32_t prod_offset = event.product_offset;
                        const std::uint32_t prod_count = event.direct_product_count;

                        // Finite event-library azimuths are not a preferred
                        // laboratory direction. Rotate the whole event once.
                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                            spot_seed, rng_history, steps, cinel03_event_azimuth_dimension);
                        const float event_cos = sycl::cos(event_phi);
                        const float event_sin = sycl::sin(event_phi);

                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                            if (prod_offset + ip >= schneider_ct_device_ctx.primary_total_products) break;
                            const auto& product = schneider_ct_device_ctx.primary_products[prod_offset + ip];

                            if (product.role == 2) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                unsupported_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                continue;
                            }
                            if (product.z <= 0 || product.a <= 0) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                neutral_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                continue;
                            }
                            if (product.role != 0) {
                                continue;
                            }
                            // Be6 keeps the frozen TopasCompatKill policy:
                            // explicit counter + energy, never queued.
                            if (product.z == 4 && product.a == 6) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryBe6Kills);
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
                                SchneiderDiagSlot::PrimaryChargedBorn);

                            if (enable_secondary_transport && product.kinetic_energy_MeV > energy_cutoff_MeV) {
                                const auto local_direction = rotate_cinel03_event_azimuth(
                                    product.local_direction_x, product.local_direction_y,
                                    product.local_direction_z, event_cos, event_sin);
                                const auto child_direction = rotate_local_direction(
                                    local_direction.x,
                                    local_direction.y,
                                    local_direction.z,
                                    Direction3F{direction_x, direction_y, direction_z});

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
                                        child.pos_x_mm = position_x_mm;
                                        child.pos_y_mm = position_y_mm;
                                        child.pos_z_mm = position_z_mm;
                                        child.dir_x = child_direction.x;
                                        child.dir_y = child_direction.y;
                                        child.dir_z = child_direction.z;
                                        child.weight = 1.0F;
                                        child.parent_history = global_history;
                                        child.rng_stream = rng::event_product_stream(
                                            rng::primary_descendant_stream(spot_seed, rng_history),
                                            steps, rng::branch_role_primary_charged, ip);
                                        secondary_queue_device[output] = child;
                                        charged_accounted_MeV += product.kinetic_energy_MeV;
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryChargedQueued);
                                        if (ion_species_energy_device != nullptr) {
                                            ion_record_queued_secondary_birth_device(
                                                ion_species_energy_device, child.z, child.a,
                                                child.energy_MeV);
                                        }
                                    } else {
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryQueueOverflows);
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
                                pending_primary_depth_MeV += product.kinetic_energy_MeV;
                                if ((enable_voxel_scoring) && voxel_index >= 0) {
                                    pending_primary_voxel_MeV += product.kinetic_energy_MeV;
                                }

                                history_deposited_MeV += product.kinetic_energy_MeV;
                                grid_deposit_split_device(
                                    grid_deposited_in_device,
                                    grid_deposited_out_device,
                                    (enable_voxel_scoring) && voxel_index >= 0,
                                    product.kinetic_energy_MeV);
                                charged_accounted_MeV += product.kinetic_energy_MeV;
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedCutoffKills);
                            }
                        }

                        schneider_float_add_device(
                            schneider_float_device, SchneiderFloatSlot::ReactionQResidual,
                            sycl::fmax(0.0F, energy_MeV - local_deposit -
                                                  charged_accounted_MeV -
                                                  neutral_accounted_MeV -
                                                  unsupported_accounted_MeV));
                        // NO-DOUBLE-COUNT RULE: the legacy untracked sink below
                        // already contains neutral + unsupported + Q-residual
                        // energy (E - local - charged). The split float slots
                        // above are INFORMATIONAL ONLY and must never be added
                        // into the global closure alongside untracked; the
                        // closure in run_quality uses the legacy sink alone.
                        // A unit test pins this (split fields leave the
                        // residual bitwise unchanged).
                        const float untracked_MeV = sycl::fmax(0.0F, energy_MeV - local_deposit - charged_accounted_MeV);
                        if (untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_untracked(untracked_nuclear_device[global_history]);
                            atomic_untracked.fetch_add(untracked_MeV);
                        }

                        if (primary_terminal_counts_device != nullptr) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                term_ref(primary_terminal_counts_device[0]);
                            term_ref.fetch_add(1U);
                        }
                        energy_MeV = 0.0F;
                        break;
                        }  // else of the v3 masked-sampler empty-draw guard
                    }

                    ++steps;
                }

                const bool inside_phantom = !unified_primary_escaped_ct &&
                    !water_urban_escaped_geometry &&
                    (position_z_mm >= 0.0F && position_z_mm < phantom_length_mm);

                if (energy_MeV > 0.0F && energy_MeV <= energy_cutoff_MeV && inside_phantom) {
                    const auto cutoff_energy_MeV = energy_MeV;
                    pending_primary_depth_MeV += cutoff_energy_MeV;
                    if ((enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                        pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                        pending_primary_voxel_MeV += cutoff_energy_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(cutoff_energy_MeV));
                        }
                    }
                    history_deposited_MeV += cutoff_energy_MeV;
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device,
                        (enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                            pending_primary_voxel <
                                static_cast<std::size_t>(number_of_voxels),
                        cutoff_energy_MeV);
                    if (cutoff_stopped_energy_device != nullptr) {
                        cutoff_stopped_energy_device[global_history] = cutoff_energy_MeV;
                    }

                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[2]); // [2] = stopped_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                    energy_MeV = 0.0F;
                } else if (energy_MeV > 0.0F && !inside_phantom) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[1]); // [1] = escaped_ct_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                } else if (!primary_inelastic_occurred) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[3]); // [3] = other_terminal
                        term_ref.fetch_add(1U);
                    }
                    if (energy_MeV > 0.0F) {
                        if (other_terminal_energy_device != nullptr) {
                            other_terminal_energy_device[global_history] = energy_MeV;
                        }
                        energy_MeV = 0.0F;
                    }
                }

                if (pending_primary_depth_MeV > 0.0) {
                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
                }

                if ((enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                    pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                    if (pending_primary_voxel_MeV > 0.0) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                        atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));

                    }

                }

                schneider_diag_add_device(schneider_diag_device,
                                              SchneiderDiagSlot::PrimaryRateQueries,
                                              local_schneider_rate_queries);
                if(defer_material_packets)flush_electron_group();
                if constexpr (BroadBeam) {
                    if (grid_in_pending_MeV != 0.0 && grid_deposited_in_device != nullptr)
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*grid_deposited_in_device).fetch_add(grid_in_pending_MeV);
                    if (grid_out_pending_MeV != 0.0 && grid_deposited_out_device != nullptr)
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*grid_deposited_out_device).fetch_add(grid_out_pending_MeV);
                }
                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV;
                steps_device[global_history] = steps;
                } // medium stage (or broadbeam transport)
            });
        };
        const auto dispatch_primary_chunk = [&]() {
            if constexpr(BroadBeam) {
                if constexpr (std::is_same_v<Pipeline, pipelines::ProtonBroadbeam> && JointResponse==0) {
                    if (broadbeam_ct_primary_specialize && primary_shared_ct_cell)
                        return launch_primary_chunk(std::false_type{}, std::integral_constant<int,16>{});
                }
                if constexpr (std::is_same_v<Pipeline, pipelines::CarbonBroadbeam> && JointResponse==0) {
                    if (broadbeam_ct_primary_specialize && primary_shared_ct_cell)
                        return launch_primary_chunk(std::false_type{}, std::integral_constant<int,18>{});
                }
                return Pipeline::template launch_primary<JointResponse>(
                    broadbeam_ct_primary_specialize, nonion_primary,
                    primary_atomic_number, primary_mass_number, launch_primary_chunk);
            } else {
                // Compact minibeam: beamline launch (3, or 9 with queued Cu
                // children) followed by the medium launch (4, or CT stage 11).
                {
                    const auto launch_minibeam_stage = [&](auto stage) {
                        // The Cu/air stages share one compiled kernel for water
                        // and CT. Only downstream transport selects a material
                        // policy; never apply uniform-water substitutions to CT.
                        if constexpr (decltype(stage)::value == 4) {
                            if (!use_minibeam_primary_path) {
                                if (ct_queued_primary_specialize)
                                    return launch_primary_chunk(std::false_type{}, std::integral_constant<int,11>{});
                                return launch_primary_chunk(std::false_type{}, stage);
                            }
                        }
                        return launch_primary_chunk(std::true_type{}, stage);
                    };
                    if (beamline_order != nullptr) {
                        queue.fill(beamline_order_counts, 0U, 2);
                        queue.parallel_for(sycl::range<1>(chunk_count), [=](sycl::id<1> id) {
                            const auto lane = static_cast<std::uint32_t>(id[0]);
                            const std::uint64_t history = hist_offset + lane;
                            const auto local_x = flat_source_half_width_x_mm *
                                (2.0F * rng::uniform01(random_seed, history, 0, 34) - 1.0F);
                            const auto local_y = flat_source_half_width_y_mm *
                                (2.0F * rng::uniform01(random_seed, history, 0, 35) - 1.0F);
                            const auto direction = compose_tps_direction(
                                0.0F, 0.0F, 1.0F, beam_ux_x, beam_ux_y, beam_ux_z,
                                beam_uy_x, beam_uy_y, beam_uy_z, beam_uz_x, beam_uz_y, beam_uz_z);
                            const bool copper = minibeam_ray_hits_cylindrical_copper(
                                source_origin_x_mm + beam_ux_x * local_x + beam_uy_x * local_y,
                                source_origin_y_mm + beam_ux_y * local_x + beam_uy_y * local_y,
                                source_origin_z_mm + beam_ux_z * local_x + beam_uy_z * local_y,
                                direction.x, direction.y, direction.z,
                                minibeam_cos, minibeam_sin, minibeam_block_radius,
                                minibeam_block_center_z, minibeam_block_thickness,
                                minibeam_slit_count, minibeam_slit_width, minibeam_slit_pitch,
                                0.5F * minibeam_slit_length, minibeam_slit_thickness,
                                minibeam_slit_offset);
                            bool expensive=copper;

                            const auto rank = sycl::atomic_ref<std::uint32_t,
                                sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                    beamline_order_counts[expensive ? 0 : 1]).fetch_add(1U);
                            beamline_order[expensive ? rank : chunk_count - 1 - rank] = lane;
                        });
                    }
                    {
                        queue.fill(primary_entry_counts, std::uint64_t{0}, 2);
                        primary_events.push_back(queue_copper_primary_children
                            ? launch_minibeam_stage(std::integral_constant<int,9>{})
                            : launch_minibeam_stage(std::integral_constant<int,3>{}));
                        if (water_energy_order) {
                            queue.fill(water_energy_counts,0U,water_energy_buckets+1);
                            auto bucket=[=](float energy) {
                                if(energy<=energy_cutoff_MeV)return water_energy_buckets;
                                return sycl::min(water_energy_buckets-1,
                                    static_cast<unsigned>(sycl::fmax(0.F,energy)/initial_energy_MeV*water_energy_buckets));
                            };
                            water_order_events.push_back(queue.parallel_for(sycl::range<1>(chunk_count),[=](sycl::id<1> index){
                                const auto b=bucket(primary_medium_entry[index[0]].energy_MeV);
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(water_energy_counts[b]).fetch_add(1U);
                            }));
                            water_order_events.push_back(queue.single_task([=]{
                                unsigned offset=0;
                                for(unsigned b=0;b<=water_energy_buckets;++b){
                                    water_energy_offsets[b]=offset;offset+=water_energy_counts[b];water_energy_counts[b]=0;
                                }
                            }));
                            water_order_events.push_back(queue.parallel_for(sycl::range<1>(chunk_count),[=](sycl::id<1> index){
                                const auto b=bucket(primary_medium_entry[index[0]].energy_MeV);
                                const auto rank=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(water_energy_counts[b]).fetch_add(1U);
                                water_energy_order[water_energy_offsets[b]+rank]=index[0];
                            }));
                        }
                        return launch_minibeam_stage(std::integral_constant<int, 4>{});
                    }
                }
            }
        };
        auto kernel_event = dispatch_primary_chunk();
        primary_events.push_back(kernel_event);

        if (defer_material_packets) {
            std::uint64_t count_host=0;
            queue.copy(material_task_counts,&count_host,1).wait_and_throw();
            if(count_host>material_task_capacity)
                throw std::runtime_error("Electron response task queue overflow; batch rejected, no tasks discarded as accepted dose");
            electron_tasks_total+=count_host;
            electron_queue_peak=std::max<std::uint64_t>(electron_queue_peak,count_host);
            if(count_host) {
                const std::uint32_t lanes=static_cast<std::uint32_t>(std::min<std::uint64_t>(count_host,material_queue_lanes));
                if(electron_order) {
                    queue.fill(electron_order_counts,0U,electron_order_bins);
                    electron_events.push_back(queue.parallel_for(sycl::range<1>(count_host),[=](sycl::id<1> index){
                        namespace f=electron_fp32;
                        const auto t=material_tasks[index[0]];
                        unsigned bucket=32;
                        if(t.birth.source<kElectronMaterialCount) {
                            const auto e=sample_cut_electron_birth_seeded(cut_birth26,t.birth.source,
                                electron_projectile_parameters,float(t.birth.energy),t.birth.cut,t.birth.cdf_bits).energy;
                            const unsigned eb=(e>.001f)+(e>.003f)+(e>.01f)+(e>.03f)+(e>.1f)+(e>.3f)+(e>.65f);
                            const unsigned db=(t.density>.01)+(t.density>.1)+(t.density>.5);
                            bucket=db*8+eb;
                        }
                        electron_order_keys[index[0]]=static_cast<unsigned char>(bucket);
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(electron_order_counts[bucket]).fetch_add(1U);
                    }));
                    electron_events.push_back(queue.single_task([=]{
                        std::uint32_t offset=0;
                        for(unsigned b=0;b<electron_order_bins;++b){
                            electron_order_offsets[b]=offset;offset+=electron_order_counts[b];electron_order_counts[b]=0;
                        }
                    }));
                    electron_events.push_back(queue.parallel_for(sycl::range<1>(count_host),[=](sycl::id<1> index){
                        const auto b=electron_order_keys[index[0]];
                        const auto rank=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(electron_order_counts[b]).fetch_add(1U);
                        electron_order[electron_order_offsets[b]+rank]=static_cast<std::uint32_t>(index[0]);
                    }));
                }

                {
#include "detail/sycl_independent_electron_fp32.inc"
                }

            }
        }
    }

    if(joint_pilot_work) {
        std::array<std::uint64_t,9> work{};queue.copy(joint_pilot_work,work.data(),9).wait_and_throw();
        sycl::free(joint_pilot_work,queue);
        const char* names[]={"eligible","started","advances","fallback","below","deposited","escaped","photon_gap","invalid"};
        std::cout<<"[electron-joint-pilot-work]";
        for(unsigned k=0;k<9;++k)std::cout<<" "<<names[k]<<"="<<work[k];
        std::cout<<"\n";
    }
    if(independent_charged_work) {
        std::array<std::uint64_t,8> work{};queue.copy(independent_charged_work,work.data(),8).wait_and_throw();
        const char* names[]={"calls","elastic","hard_proposals","hard_collisions","faces","stopped","coverage_missing","invalid"};
        std::cout<<"[independent-charged-work]";for(unsigned k=0;k<8;++k)std::cout<<" "<<names[k]<<"="<<work[k];std::cout<<"\n";
    }
    if(independent_photon_work) {
        std::array<std::uint64_t,8> work{};queue.copy(independent_photon_work,work.data(),8).wait_and_throw();
        const char* names[]={"calls","interactions","voxel_crossings","electron","deposited","escaped","coverage_missing","invalid"};
        std::cout<<"[independent-photon-work]";
        for(unsigned k=0;k<8;++k)std::cout<<" "<<names[k]<<"="<<work[k];
        std::cout<<"\n";
    }
    std::array<std::uint64_t,3> electron_work{};
    if (defer_material_packets) {
        queue.copy(material_queue_work,electron_work.data(),3).wait_and_throw();
        free_device(material_queue_work);
        queue.wait_and_throw();free_device(material_tasks);free_device(material_task_counts);
        free_device(electron_order);free_device(electron_order_keys);free_device(electron_order_counts);free_device(electron_order_offsets);
    }

    double electron_kernel_seconds=0;
    for(const auto& event:electron_events)electron_kernel_seconds+=event_duration_seconds(event);
    if(defer_material_packets)
        std::cout<<"[electron-queue-time] seconds="<<electron_kernel_seconds
                 <<" tasks="<<electron_tasks_total<<" peak_tasks="<<electron_queue_peak
                 <<" advances="<<electron_work[0]<<" source_restarts="<<electron_work[1]
                 <<" boundary_restarts="<<electron_work[2]<<"\n";
    // The queue is in order, so all primary chunks retain exactly the previous
    // execution order.  Waiting once on the final event avoids a host/device
    // round trip and a flushed progress line between every chunk.
    if (!primary_events.empty()) {
        primary_events.back().wait_and_throw();
        double split_beamline_seconds = 0.0, split_water_seconds = 0.0;
        for (std::size_t i=0; i<primary_events.size(); ++i) {
            const auto seconds = event_duration_seconds(primary_events[i]);
            primary_kernel_seconds += seconds;
            if (compact_minibeam_primary) {
                if (i%2 == 0) split_beamline_seconds += seconds;
                else split_water_seconds += seconds;
            }
        }
        if (compact_minibeam_primary)
            std::cout << "[primary-split-time] beamline_s=" << split_beamline_seconds
                      << " water_s=" << split_water_seconds << "\n";
        std::cout << "[progress] primary batches completed: "
                  << number_of_histories << "/" << number_of_histories
                  << " histories (" << primary_events.size() << " launches)\n";
    }

    if (primary_medium_entry) mem_tracker.free(primary_medium_entry);
    if (primary_entry_counts) mem_tracker.free(primary_entry_counts);
    double water_order_seconds=0;
    for(const auto& event:water_order_events)water_order_seconds+=event_duration_seconds(event);
    if(water_energy_order)std::cout << "[water-energy-order] seconds=" << water_order_seconds << '\n';
    if(water_energy_order)mem_tracker.free(water_energy_order);
    if(water_energy_counts)mem_tracker.free(water_energy_counts);
    if(water_energy_offsets)mem_tracker.free(water_energy_offsets);
    if (beamline_order) mem_tracker.free(beamline_order);
    if (beamline_order_counts) mem_tracker.free(beamline_order_counts);

    double copper_primary_child_seconds=0;
    if(copper_primary_child_tasks) {
        std::uint32_t count=0;
        queue.copy(copper_primary_child_counts,&count,1).wait_and_throw();
        if(count>copper_primary_child_capacity)
            throw std::runtime_error("Cu primary-child queue overflow; batch rejected without truncation");
        if(count) {
            const auto lanes=std::min<std::size_t>(count,17408);
            auto event=queue.parallel_for(sycl::nd_range<1>{sycl::range<1>{((lanes+127)/128)*128},sycl::range<1>{128}},
                [=](sycl::nd_item<1>) {
                    for(;;) {
                        const auto slot=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                            (copper_primary_child_counts[1]).fetch_add(1U);
                        if(slot>=count)break;
                        const auto task=copper_primary_child_tasks[slot];
                        const auto& product=minibeam_copper_products_device[task.flat_product_index];
                        const auto product_index=task.product_index;
                        const auto global_history=task.global_history;
                        const auto rng_history=task.rng_history;
                        const auto spot_seed=task.spot_seed;
                        const auto beamline_step=task.beamline_step;
                        const auto position_x_mm=task.x,position_y_mm=task.y,position_z_mm=task.z;
                        const auto direction_x=task.dx,direction_y=task.dy,direction_z=task.dz;
                        const auto event_cos=task.event_cos,event_sin=task.event_sin;
                        const auto block_exit_z=minibeam_block_center_z+.5F*minibeam_block_thickness;
                        // Started tasks are all finished when this event completes,
                        // including the original stop/backward early-continue paths.
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                            (copper_primary_child_counts[2]).fetch_add(1U);
                        float queued_energy=0;
#include "detail/sycl_copper_primary_child.inc"
                        if(queued_energy>0)
                            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(beamline_removed_device[global_history]).fetch_sub(queued_energy);
                    }
                });
            event.wait_and_throw();copper_primary_child_seconds=event_duration_seconds(event);
        }
        std::uint32_t completed=0;
        queue.copy(copper_primary_child_counts+2,&completed,1).wait_and_throw();
        if(completed!=count)throw std::runtime_error("Cu primary-child queue drain mismatch");
        std::cout << "[copper-primary-child-queue] seconds=" << copper_primary_child_seconds
                  << " tasks=" << count << " completed=" << completed << " overflow=0\n";
        free_device(copper_primary_child_tasks);free_device(copper_primary_child_counts);
    }
    // Process-faithful fragment+Cu products are transported through the
    // remaining collimator in generation batches. Copper generation lives in
    // this dedicated queue and never consumes the downstream water budget.
    if (minibeam_copper_cascade_count_device != nullptr) {
        std::uint32_t copper_cascade_count = 0;
        queue.copy(minibeam_copper_cascade_count_device,
                   &copper_cascade_count, 1).wait_and_throw();
        copper_cascade_count = std::min<std::uint32_t>(
            copper_cascade_count, static_cast<std::uint32_t>(max_secondaries));
        std::uint32_t copper_generation_begin = 0;
        std::uint32_t copper_generation_end = copper_cascade_count;
        while (copper_generation_begin < copper_generation_end) {
            const auto generation_begin = copper_generation_begin;
            const auto generation_size = copper_generation_end - generation_begin;
            queue.parallel_for(sycl::range<1>(generation_size),
                [=](sycl::id<1> item) {
                    const auto cascade =
                        minibeam_copper_cascade_queue_device[
                            generation_begin + item[0]];
                    const auto species_category =
                        cascade.z == 6 ? 0U : cascade.z == 5 ? 1U :
                        cascade.z == 4 ? 2U : cascade.z == 3 ? 3U :
                        cascade.z == 2 ? 4U :
                        (cascade.z == 1 && cascade.a == 1) ? 5U :
                        (cascade.z == 1 && cascade.a == 2) ? 6U :
                        (cascade.z == 1 && cascade.a == 3) ? 7U : 8U;
                    auto energy = cascade.energy_MeV;
                    auto x = cascade.pos_x_mm;
                    auto y = cascade.pos_y_mm;
                    auto z = cascade.pos_z_mm;
                    Direction3F direction{
                        cascade.dir_x, cascade.dir_y, cascade.dir_z};
                    auto copper_segment_path = 0.0F;
                    auto ignored_nuclear_tau = 0.0F;
                    auto nuclear_tau_remaining = 0.0F;
                    auto nuclear_tau_active = false;
                    UrbanV2TrackState fragment_urban_state{};
                    bool fragment_at_boundary = true;
                    auto collided = false;
                    std::uint32_t step = 0;
                    const auto block_exit_z = minibeam_block_center_z +
                        0.5F * minibeam_block_thickness;
                    const auto terminal_copper_generation =
                        cascade.generation >=
                            minibeam_copper_fragment_cascade_generations;
                    const auto reaction_generation_index = sycl::min(
                        2U, static_cast<std::uint32_t>(cascade.generation));
                    while (energy > energy_cutoff_MeV &&
                           direction.z > 1.0e-8F &&
                           z < block_exit_z - 1.0e-6F &&
                           step < 100000U) {
                        const auto nominal_path = sycl::fmin(
                            minibeam_copper_max_step,
                            (block_exit_z - z) / direction.z);
                        const auto boundary_path =
                            minibeam_copper_exact_material_boundaries
                                ? minibeam_path_to_material_boundary(
                                      x, y, direction.x, direction.y,
                                      minibeam_cos, minibeam_sin,
                                      minibeam_block_radius,
                                      minibeam_slit_count,
                                      minibeam_slit_width,
                                      minibeam_slit_pitch,
                                      0.5F * minibeam_slit_length,
                                      minibeam_slit_offset,
                                      nominal_path)
                                : nominal_path;
                        const auto mid_x = x + 0.5F * boundary_path * direction.x;
                        const auto mid_y = y + 0.5F * boundary_path * direction.y;
                        const auto in_copper = minibeam_point_in_copper(
                            mid_x, mid_y, minibeam_cos, minibeam_sin,
                            minibeam_block_radius, minibeam_slit_count,
                            minibeam_slit_width, minibeam_slit_pitch,
                            0.5F * minibeam_slit_length,
                            minibeam_slit_offset);
                        auto inelastic_rate = 0.0F;
                        auto collision_in_step = false;
                        auto path = boundary_path;
                        if (in_copper && !copper_ion_native_transport) {
                            inelastic_rate = minibeam_copper_ion_inelastic_rate(
                                minibeam_copper_ion_xs_device,
                                minibeam_copper_ion_xs_present_device,
                                minibeam_copper_ion_xs_grid_size,
                                minibeam_copper_ion_xs_minimum_energy,
                                minibeam_copper_ion_xs_inverse_step,
                                energy, cascade.z, cascade.a);
                            if (terminal_copper_generation) {
                                ignored_nuclear_tau +=
                                    inelastic_rate * boundary_path;
                            } else if (inelastic_rate > 0.0F) {
                                if (!nuclear_tau_active) {
                                    const auto optical_uniform = sycl::fmax(
                                        1.0e-7F,
                                        rng::uniform01(random_seed,
                                                       cascade.rng_stream,
                                                       step, 2));
                                    nuclear_tau_remaining =
                                        -sycl::log(optical_uniform);
                                    nuclear_tau_active = true;
                                }
                                collision_in_step =
                                    nuclear_tau_remaining <=
                                        inelastic_rate * boundary_path;
                                if (collision_in_step) {
                                    path = nuclear_tau_remaining / inelastic_rate;
                                }
                            }
                        }
                        if (copper_ion_native_transport) {
                            auto& f_energy = energy; auto& f_x = x; auto& f_y = y; auto& f_z = z;
                            auto& f_direction = direction; auto& f_path = path;
                            auto& f_rate = inelastic_rate; auto& f_collision = collision_in_step;
                            auto& f_tau = nuclear_tau_remaining; auto& f_tau_active = nuclear_tau_active;
                            auto& f_urban_state = fragment_urban_state; auto& f_at_boundary = fragment_at_boundary;
                            const auto f_charge = cascade.z; const auto f_a = cascade.a;
                            const auto f_seed = random_seed; const auto f_stream = cascade.rng_stream; const auto f_step = step;
                            const auto f_in_copper = in_copper; const auto f_terminal = terminal_copper_generation;
                            auto& f_ignored_tau = ignored_nuclear_tau;
#include "detail/sycl_copper_ion_fragment_step.inc"
                        } else {
                        x += path * direction.x;
                        y += path * direction.y;
                        z += path * direction.z;
                        }
                        if (in_copper) {
                          if (!copper_ion_native_transport) {
                            const auto stopping = minibeam_copper_ion_stopping(
                                minibeam_copper_sp_energies_device,
                                minibeam_copper_sp_values_device,
                                minibeam_copper_sp_count,
                                minibeam_copper_ion_sp_ratios_device,
                                minibeam_copper_ion_sp_present_device,
                                energy, cascade.z, cascade.a, primary_atomic_number);
                            const auto predicted_loss = stopping * path;
                            const auto midpoint_energy = sycl::fmax(
                                0.0F, energy - 0.5F * predicted_loss);
                            const auto midpoint_stopping =
                                minibeam_copper_ion_stopping(
                                    minibeam_copper_sp_energies_device,
                                    minibeam_copper_sp_values_device,
                                    minibeam_copper_sp_count,
                                    minibeam_copper_ion_sp_ratios_device,
                                    minibeam_copper_ion_sp_present_device,
                                    midpoint_energy, cascade.z, cascade.a, primary_atomic_number);
                            const auto mean_loss = midpoint_stopping * path;
                            auto loss = mean_loss;
                            if (minibeam_copper_fragment_enable_energy_straggling) {
                                constexpr float copper_z_over_a_rel_water =
                                    (29.0F / 63.546F) / 0.55509F;
                                const auto energy_u = midpoint_energy /
                                    static_cast<float>(cascade.a);
                                const auto effective_charge =
                                    ion_effective_charge_device(cascade.z, energy_u);
                                const auto variance =
                                    condensed_total_loss_variance_MeV2_device(
                                        energy_u, cascade.a, effective_charge,
                                        path, minibeam_copper_density,
                                        copper_z_over_a_rel_water);
                                const auto gaussian_u0 = sycl::fmax(
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 10),
                                    1.0e-12F);
                                const auto gaussian_u1 = rng::uniform01(
                                    random_seed, cascade.rng_stream, step, 11);
                                const auto gaussian = sycl::sqrt(
                                    -2.0F * sycl::log(gaussian_u0)) *
                                    sycl::cos(6.2831853071795864769F * gaussian_u1);
                                loss = sycl::fmax(
                                    0.0F,
                                    mean_loss +
                                        minibeam_copper_fragment_straggling_scale *
                                            sycl::sqrt(sycl::fmax(0.0F, variance)) *
                                            gaussian);
                            }
                            if (minibeam_copper_fragment_enable_energy_straggling) {
                                if (copper_proton_native_fluctuation && cascade.z == 1 && cascade.a == 1) {
                                    std::uint64_t draw_index = 0;
                                    auto uniform = [&]() { return (rng::random_u32(
                                        random_seed, cascade.rng_stream, (std::uint64_t(step) << 32) | draw_index++, 124) >> 8) * 0x1p-24f; };
                                    const auto draw = copper_proton_total_fluctuation(
                                        energy, 938.272013F, mean_loss, path, minibeam_copper_density, uniform);
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                        minibeam_event_counts_device[draw.valid ? copper_proton_fragment_loss_slot : copper_proton_loss_failure_slot]).fetch_add(1U);
                                    if (!draw.valid) break; // host rejects the entire run
                                    loss = draw.loss;
                                }
                            }
                            if (loss >= energy - energy_cutoff_MeV) {
                                energy = 0.0F;
                                break;
                            }
                            const auto scatter_energy = energy - 0.5F * loss;
                            energy -= loss;
                            if (minibeam_copper_enable_mcs) {
                                const auto previous_path = copper_segment_path;
                                copper_segment_path += path;
                                const auto total_rms =
                                    highland_projected_rms_angle_device(
                                        scatter_energy, cascade.z, cascade.a,
                                        copper_segment_path,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length);
                                const auto previous_rms =
                                    highland_projected_rms_angle_device(
                                        scatter_energy, cascade.z, cascade.a,
                                        previous_path,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length);
                                direction = scatter_direction(
                                    direction,
                                    minibeam_copper_fragment_mcs_scale *
                                        sycl::sqrt(sycl::fmax(
                                            0.0F, total_rms * total_rms -
                                                      previous_rms * previous_rms)),
                                    random_seed, cascade.rng_stream, step, 0);
                            }
                            if (!terminal_copper_generation &&
                                !collision_in_step && inelastic_rate > 0.0F &&
                                nuclear_tau_active) {
                                nuclear_tau_remaining -= inelastic_rate * path;
                            }
                          }
                            if (collision_in_step) {
                                collided = true;
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    reaction_count(minibeam_event_counts_device[
                                        minibeam_fragment_cascade_interactions_slot]);
                                reaction_count.fetch_add(1U);
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    generation_reaction_count(
                                        minibeam_event_counts_device[
                                            minibeam_fragment_generation_interaction_slot +
                                            reaction_generation_index]);
                                generation_reaction_count.fetch_add(1U);
                                const auto lookup = cinel03_lookup_event_device(
                                    minibeam_copper_nodes_device,
                                    minibeam_copper_node_count,
                                    minibeam_copper_offsets_device,
                                    minibeam_copper_indices_device,
                                    minibeam_copper_event_count,
                                    cascade.z, cascade.a, 29,
                                    energy / static_cast<float>(cascade.a),
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 20),
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 21));
                                if (lookup.status == Cinel03LookupStatus::Hit) {
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        hit_count(minibeam_event_counts_device[
                                            minibeam_fragment_cascade_hits_slot]);
                                    hit_count.fetch_add(1U);
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generation_hit_count(
                                            minibeam_event_counts_device[
                                                minibeam_fragment_generation_hit_slot +
                                                reaction_generation_index]);
                                    generation_hit_count.fetch_add(1U);
                                    const auto& event =
                                        minibeam_copper_interactions_device[
                                            lookup.event_index];
                                    const auto phi = 6.2831853071795864769F *
                                        rng::uniform01(random_seed,
                                                       cascade.rng_stream,
                                                       step, 22);
                                    const auto event_cos = sycl::cos(phi);
                                    const auto event_sin = sycl::sin(phi);
                                    auto accounted_charged = 0.0F;
                                    auto replay_product_energy = 0.0F;
                                    auto replay_product_rest_mass = 0.0F;
                                    std::int32_t replay_product_a = 0;
                                    for (std::uint32_t product_index = 0;
                                         product_index < event.direct_product_count;
                                         ++product_index) {
                                        const auto flat_index =
                                            event.product_offset + product_index;
                                        if (flat_index >=
                                            minibeam_copper_product_count) break;
                                        const auto& product =
                                            minibeam_copper_products_device[flat_index];
                                        replay_product_energy += sycl::fmax(
                                            0.0F, product.kinetic_energy_MeV);
                                        replay_product_rest_mass += sycl::fmax(
                                            0.0F, product.rest_mass);
                                        replay_product_a += sycl::max(
                                            0, static_cast<int>(product.a));
                                        const bool supported_charged =
                                            (product.role == 0 || product.role == 1) &&
                                            product.z > 0 && product.a > 0;
                                        const auto count_slot = supported_charged
                                            ? minibeam_fragment_cascade_charged_slot
                                            : (product.z <= 0
                                                   ? minibeam_fragment_cascade_neutral_slot
                                                   : minibeam_fragment_cascade_unsupported_slot);
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            product_count(minibeam_event_counts_device[
                                                count_slot]);
                                        product_count.fetch_add(1U);
                                        if (!supported_charged ||
                                            product.kinetic_energy_MeV <=
                                                energy_cutoff_MeV) continue;
                                        accounted_charged +=
                                            product.kinetic_energy_MeV;
                                        const auto local =
                                            rotate_cinel03_event_azimuth(
                                                product.local_direction_x,
                                                product.local_direction_y,
                                                product.local_direction_z,
                                                event_cos, event_sin);
                                        const auto child_direction =
                                            rotate_local_direction(
                                                local.x, local.y, local.z,
                                                direction);
                                        sycl::atomic_ref<
                                            std::uint32_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            queue_count(
                                                *minibeam_copper_cascade_count_device);
                                        const auto output = queue_count.fetch_add(1U);
                                        if (output < max_secondaries) {
                                            SecondaryParticle child{};
                                            child.z = product.z;
                                            child.a = product.a;
                                            child.energy_MeV =
                                                product.kinetic_energy_MeV;
                                            child.pos_x_mm = x;
                                            child.pos_y_mm = y;
                                            child.pos_z_mm = z;
                                            child.dir_x = child_direction.x;
                                            child.dir_y = child_direction.y;
                                            child.dir_z = child_direction.z;
                                            child.weight = 1.0F;
                                            child.parent_history =
                                                cascade.parent_history;
                                            child.rng_stream =
                                                rng::event_product_stream(
                                                    cascade.rng_stream, step,
                                                    rng::branch_role_cascade_charged,
                                                    product_index);
                                            child.generation = static_cast<std::uint16_t>(
                                                cascade.generation + 1U);
                                            child.birth_region =
                                                minibeam_birth_region_copper;
                                            minibeam_copper_cascade_queue_device[
                                                output] = child;
                                        } else {
                                            sycl::atomic_ref<
                                                std::uint64_t,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                overflow_count(
                                                    minibeam_event_counts_device[
                                                        minibeam_fragment_cascade_overflow_slot]);
                                            overflow_count.fetch_add(1U);
                                        }
                                    }
                                    const auto add_fixed =
                                        [&](const std::size_t slot,
                                            const float value) {
                                            sycl::atomic_ref<
                                                std::uint64_t,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                counter(minibeam_event_counts_device[slot]);
                                            counter.fetch_add(
                                                static_cast<std::uint64_t>(
                                                    sycl::fmax(0.0F, value) *
                                                        1000.0F + 0.5F));
                                        };
                                    add_fixed(
                                        minibeam_fragment_cascade_local_keV_slot,
                                        event.process_local_deposit_MeV);
                                    add_fixed(
                                        minibeam_fragment_cascade_untracked_keV_slot,
                                        sycl::fmax(
                                            0.0F,
                                            energy - event.process_local_deposit_MeV -
                                                accounted_charged));
                                    const auto selected_input =
                                        lookup.selected_energy_MeV_per_u *
                                        static_cast<float>(cascade.a);
                                    const auto replay_output = sycl::fmax(
                                        0.0F, event.parent_energy_MeV) +
                                        sycl::fmax(
                                            0.0F,
                                            event.process_local_deposit_MeV) +
                                        replay_product_energy;
                                    add_fixed(
                                        minibeam_fragment_actual_input_keV_slot,
                                        energy);
                                    add_fixed(
                                        minibeam_fragment_selected_input_keV_slot,
                                        selected_input);
                                    add_fixed(
                                        minibeam_fragment_replay_output_keV_slot,
                                        replay_output);
                                    add_fixed(
                                        minibeam_fragment_selection_mismatch_keV_slot,
                                        sycl::fabs(energy - selected_input));
                                    add_fixed(
                                        minibeam_fragment_closure_mismatch_keV_slot,
                                        sycl::fabs(selected_input - replay_output));
                                    const bool parent_survives =
                                        event.parent_energy_MeV > 0.0F;
                                    const auto copper_target_mass =
                                        event.target_a == 63
                                            ? 58603.7301743F
                                            : (event.target_a == 65
                                                   ? 60465.0342192F
                                                   : sycl::fmax(
                                                         0.0F,
                                                         static_cast<float>(
                                                             event.target_a) *
                                                                 931.49410242F -
                                                             29.0F * 0.51099895F));
                                    add_fixed(
                                        minibeam_fragment_mass_energy_mismatch_keV_slot,
                                        sycl::fabs(
                                            selected_input + event.parent_rest_mass +
                                                copper_target_mass -
                                            (replay_output +
                                             replay_product_rest_mass +
                                             (parent_survives
                                                  ? event.parent_rest_mass
                                                  : 0.0F))));
                                    const auto input_a =
                                        static_cast<int>(cascade.a) +
                                        static_cast<int>(event.target_a);
                                    const auto output_a = replay_product_a +
                                        (parent_survives
                                             ? static_cast<int>(event.parent_a)
                                             : 0);
                                    if (input_a != output_a) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            baryon_miss(minibeam_event_counts_device[
                                                minibeam_fragment_baryon_mismatch_slot]);
                                        baryon_miss.fetch_add(1U);
                                    }
                                } else {
                                    const auto miss_index =
                                        static_cast<std::size_t>(lookup.status) - 1U;
                                    if (miss_index < 6U) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                        miss_count(minibeam_event_counts_device[
                                            minibeam_fragment_cascade_miss_slot +
                                            miss_index]);
                                        miss_count.fetch_add(1U);
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generation_miss_count(
                                            minibeam_event_counts_device[
                                                minibeam_fragment_generation_miss_slot +
                                                reaction_generation_index]);
                                    generation_miss_count.fetch_add(1U);
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        species_miss(minibeam_event_counts_device[
                                            minibeam_fragment_miss_species_slot +
                                            species_category]);
                                    species_miss.fetch_add(1U);
                                    const auto miss_energy_bin = sycl::min(
                                        15U,
                                        static_cast<std::uint32_t>(
                                            sycl::fmax(
                                                0.0F,
                                                energy /
                                                    static_cast<float>(cascade.a)) /
                                            25.0F));
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        energy_miss(minibeam_event_counts_device[
                                            minibeam_fragment_miss_energy_slot +
                                            miss_energy_bin]);
                                    energy_miss.fetch_add(1U);
                                    if (minibeam_fragment_miss_joint_counts_device != nullptr &&
                                        miss_index < MinibeamDiagnostics::fragment_miss_reason_count) {
                                        const auto za_index =
                                            cascade.z >= 0 && cascade.z <= 6 &&
                                                    cascade.a >= 0 && cascade.a <= 12
                                                ? static_cast<std::size_t>(cascade.z) * 13U +
                                                      static_cast<std::size_t>(cascade.a)
                                                : MinibeamDiagnostics::fragment_miss_za_count - 1U;
                                        const auto energy_u = energy /
                                            static_cast<float>(cascade.a);
                                        const auto joint_energy_bin = sycl::min(
                                            127U,
                                            static_cast<std::uint32_t>(
                                                sycl::fmax(0.0F, energy_u) / 5.0F));
                                        const auto joint_index =
                                            (((static_cast<std::size_t>(
                                                   reaction_generation_index) *
                                                   MinibeamDiagnostics::fragment_miss_za_count +
                                               za_index) *
                                                  MinibeamDiagnostics::fragment_miss_reason_count +
                                              miss_index) *
                                                 MinibeamDiagnostics::fragment_miss_joint_energy_bin_count) +
                                            joint_energy_bin;
                                        const auto block_entry_z =
                                            minibeam_block_center_z -
                                            0.5F * minibeam_block_thickness;
                                        const auto collision_depth_um =
                                            static_cast<std::uint64_t>(sycl::fmax(
                                                0.0F, z - block_entry_z) * 1000.0F + 0.5F);
                                        const auto path_to_plane = direction.z > 1.0e-8F
                                            ? sycl::fmax(0.0F,
                                                  (block_exit_z - z) / direction.z)
                                            : 0.0F;
                                        const auto remaining_um =
                                            static_cast<std::uint64_t>(
                                                minibeam_straight_copper_path_to_plane(
                                                    x, y, direction.x, direction.y,
                                                    minibeam_cos, minibeam_sin,
                                                    minibeam_block_radius,
                                                    minibeam_slit_count,
                                                    minibeam_slit_width,
                                                    minibeam_slit_pitch,
                                                    0.5F * minibeam_slit_length,
                                                    minibeam_slit_offset,
                                                    path_to_plane) *
                                                    1000.0F +
                                                0.5F);
                                        const auto energy_keV =
                                            static_cast<std::uint64_t>(
                                                sycl::fmax(0.0F, energy) * 1000.0F +
                                                0.5F);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_counts_device[joint_index])
                                            .fetch_add(1U);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_energy_device[joint_index])
                                            .fetch_add(energy_keV);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_depth_device[joint_index])
                                            .fetch_add(collision_depth_um);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_remaining_device[joint_index])
                                            .fetch_add(remaining_um);
                                    }
                                }
                                energy = 0.0F;
                                break;
                            }
                        } else {
                            copper_segment_path = 0.0F;
                        }
                        ++step;
                    }
                    if (terminal_copper_generation) {
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            terminal_tracks(minibeam_event_counts_device[
                                minibeam_fragment_terminal_track_slot +
                                species_category]);
                        terminal_tracks.fetch_add(1U);
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ignored_tau(minibeam_event_counts_device[
                                minibeam_fragment_ignored_tau_micro_slot +
                                species_category]);
                        ignored_tau.fetch_add(static_cast<std::uint64_t>(
                            sycl::fmax(0.0F, ignored_nuclear_tau) * 1.0e6F +
                            0.5F));
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ignored_probability(minibeam_event_counts_device[
                                minibeam_fragment_ignored_probability_micro_slot +
                                species_category]);
                        ignored_probability.fetch_add(
                            static_cast<std::uint64_t>(
                                (1.0F - sycl::exp(-sycl::fmax(
                                            0.0F, ignored_nuclear_tau))) *
                                    1.0e6F +
                                0.5F));
                    }
                    if (collided) return;
                    if (energy <= energy_cutoff_MeV ||
                        direction.z <= 1.0e-8F || step == 100000U) return;
                    const auto to_water = -z / direction.z;
                    if (to_water < 0.0F) return;
                    x += to_water * direction.x;
                    y += to_water * direction.y;

                    sycl::atomic_ref<
                        std::uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        secondary_count(*secondary_count_device);
                    const auto output = secondary_count.fetch_add(1U);
                    if (output >= max_secondaries) {
                        sycl::atomic_ref<
                            std::uint32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            overflow(*secondary_overflow_count_device);
                        overflow.fetch_add(1U);
                        sycl::atomic_ref<
                            float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            overflow_energy(*secondary_overflow_energy_device);
                        overflow_energy.fetch_add(energy);
                        return;
                    }
                    SecondaryParticle water_child = cascade;
                    water_child.energy_MeV = energy;
                    water_child.pos_x_mm = x;
                    water_child.pos_y_mm = y;
                    water_child.pos_z_mm = 0.0F;
                    water_child.dir_x = direction.x;
                    water_child.dir_y = direction.y;
                    water_child.dir_z = direction.z;
                    // Copper and water use independent generation budgets.
                    // The dedicated Copper queue already enforced its cap;
                    // every survivor starts water at generation zero.
                    water_child.generation = 0U;
                    secondary_queue_device[output] = water_child;

                    // The primary kernel initially booked the complete
                    // terminal fragment energy as beamline-removed.  Restore
                    // exactly the kinetic energy that now reaches water.
                    sycl::atomic_ref<
                        float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        removed_energy(
                            beamline_removed_device[cascade.parent_history]);
                    removed_energy.fetch_sub(energy);

                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        survivor_count(minibeam_event_counts_device[2]);
                    survivor_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        species_count(
                            minibeam_event_counts_device[3U + species_category]);
                    species_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        species_energy(minibeam_event_counts_device[
                            12U + species_category]);
                    species_energy.fetch_add(static_cast<std::uint64_t>(
                        sycl::fmax(0.0F, energy) * 1000.0F + 0.5F));

                }).wait_and_throw();
            copper_generation_begin = copper_generation_end;
            std::uint32_t next_end = 0;
            queue.copy(minibeam_copper_cascade_count_device, &next_end, 1)
                .wait_and_throw();
            copper_generation_end = std::min<std::uint32_t>(
                next_end, static_cast<std::uint32_t>(max_secondaries));
        }
        std::cout << "[minibeam-copper-cascade] generation_limit="
                  << minibeam_copper_fragment_cascade_generations
                  << " initial/final-queued=" << copper_cascade_count << '/'
                  << copper_generation_end << '\n';
    }
