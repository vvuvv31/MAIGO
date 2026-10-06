// Independent FP32 baseline. No reference transport helper is called here.
auto float_event=queue.parallel_for(sycl::range<1>(lanes),[=](sycl::id<1> lane) {
    namespace f=electron_fp32;
    const f::ElectronCtGeometry geometry{{ct_nx,ct_ny,ct_nz},
        {f::Coordinate(ct_origin_x),f::Coordinate(ct_origin_y),f::Coordinate(ct_origin_z)},
        {f::Coordinate(ct_spacing_x),f::Coordinate(ct_spacing_y),f::Coordinate(ct_spacing_z)},ct_density_device,ct_material_device};
    auto increment=[](std::uint64_t* p,std::uint64_t value){
        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p).fetch_add(value);
    };
    auto tally=[&](unsigned k,std::uint64_t n){increment(electron_joint_diag_device+k,n);};
    auto add=[](auto* p,float value){using T=std::remove_pointer_t<decltype(p)>;
        sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p).fetch_add(static_cast<T>(value));
    };
    float grid_inside=0.f,grid_outside=0.f;
    // Two hottest single-address counters (calls, elastic) reduced per worker.
    std::uint32_t lane_charged_calls=0,lane_charged_elastic=0;
    // Low-density variance reduction: lane-private signed roulette weight and
    // expected-value partial deposits, merged per scoring voxel before atomics.
    float vr_gain=0.f,task_inside=0.f,task_outside=0.f,pend_value=0.f;
    std::uint64_t pend_target=0;int pend_dz=-1;bool killed=false;
    auto flush_expected=[&](){
        if(pend_value!=0.f){
            add(voxel_dose_device+pend_target,pend_value);
            if(pend_dz>=0){add(dose_device+pend_dz,pend_value);if(in_fov_dose_device)add(in_fov_dose_device+pend_dz,pend_value);}
            pend_value=0.f;
        }
    };
    auto score_expected=[&](const f::Position& p,float value){
        const int ix=static_cast<int>(sycl::floor((p[0]-f::Coordinate(voxel_min_x_mm))/f::Coordinate(voxel_size_x_mm)));
        const int iy=static_cast<int>(sycl::floor((p[1]-f::Coordinate(voxel_min_y_mm))/f::Coordinate(voxel_size_y_mm)));
        const int iz=static_cast<int>(sycl::floor(p[2]/f::Coordinate(voxel_size_z_mm)));
        if(!(ix>=0&&iy>=0&&iz>=0&&ix<int(voxel_bins_x)&&iy<int(voxel_bins_y)&&iz<int(voxel_bins_z))){task_outside+=value;return;}
        task_inside+=value;
        const auto target=(std::uint64_t(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;
        int dz=static_cast<int>(sycl::floor(p[2]/f::Coordinate(depth_bin_width_mm)));
        if(dz<0||dz>=int(number_of_bins))dz=-1;
        if(pend_value==0.f||target!=pend_target||dz!=pend_dz){flush_expected();pend_target=target;pend_dz=dz;}
        pend_value+=value;
    };
    MaterialBirthTask task{};f::ElectronEnergyPacket packet{},previous{};
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
    namespace segment_diagnostic=electron_segment_diagnostics;
    const f::ElectronCtGeometry segment_scoring{{unsigned(voxel_bins_x),unsigned(voxel_bins_y),unsigned(voxel_bins_z)},
        {voxel_min_x_mm,voxel_min_y_mm,0},{voxel_size_x_mm,voxel_size_y_mm,voxel_size_z_mm},nullptr,nullptr};
    std::uint64_t segment_charged_calls=0;
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
    namespace guard=electron_range_guard_research;
    const f::ElectronCtGeometry scoring{{unsigned(voxel_bins_x),unsigned(voxel_bins_y),unsigned(voxel_bins_z)},
        {voxel_min_x_mm,voxel_min_y_mm,0},{voxel_size_x_mm,voxel_size_y_mm,voxel_size_z_mm},nullptr,nullptr};
    std::uint64_t guard_target=UINT64_MAX;
    unsigned guard_charged=0,guard_photons=0,guard_joint=0,guard_total_charged=0;
#endif
#ifdef CARBON_ELECTRON_VOXEL_CACHE
    f::voxel_cache_research::Cache voxel_cache;
#endif
    std::uint64_t packet_rng=0,previous_rng=0;unsigned packet_steps=0;bool need_task=true;
    unsigned birth_region=6;
    std::uint64_t region_counts[12]{};
    auto uniform=[&](){return f::random_draw(packet_rng);};
    auto joint=[&](){std::uint64_t counts[9]{};auto jt=[&](int k,std::uint64_t n){counts[k]+=n;
        if constexpr(region_diagnostics){if(k==0)region_counts[6]+=n;if(k==2)region_counts[7]+=n;}};
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
        if(guard_target!=UINT64_MAX)++guard_joint;
#endif
        packet=f::transport_joint_independent(packet,float_joint,geometry,uniform,jt,
            score_expected,electron_expected_density,electron_expected_min_energy);
        for(unsigned k=0;k<9;++k)if(counts[k])increment(joint_pilot_work+k,counts[k]);
    };
    for(;;){
        if(need_task){
            const auto slot=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(material_task_counts[1]).fetch_add(1);
            if(slot>=count_host)break;
            if constexpr(BroadBeam)task=material_tasks[electron_order?electron_order[slot]:slot];
            else task=material_tasks[slot];
            packet_rng=task.rng;packet=f::ElectronEnergyPacket{};
            if constexpr(region_diagnostics) {
                for(auto& c:region_counts)c=0;
                birth_region=6;
                const int x=int(sycl::floor((task.position[0]-ct_origin_x)/ct_spacing_x));
                const int y=int(sycl::floor((task.position[1]-ct_origin_y)/ct_spacing_y));
                const int z=int(sycl::floor((task.position[2]-ct_origin_z)/ct_spacing_z));
                if(x>=0&&y>=0&&z>=0&&x<int(ct_nx)&&y<int(ct_ny)&&z<int(ct_nz))
                    birth_region=broadbeam_birth_regions[(std::size_t(z)*ct_ny+y)*ct_nx+x];
                region_counts[0]=1;region_counts[10]=static_cast<std::uint64_t>(float(task.weight)*1e6f+.5f);
            }
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
            guard_target=UINT64_MAX;guard_charged=guard_photons=guard_joint=guard_total_charged=0;
#endif
#ifdef CARBON_ELECTRON_VOXEL_CACHE
            voxel_cache.valid=false;
#endif
            packet.weight_MeV=float(task.weight);packet.material_table_index=task.table;
            if(cut_birth26||(float_birth.nodes&&task.birth.source<float_birth.count)){
                const auto node=cut_birth26?f::ElectronBirthNodeView{}:float_birth.nodes[task.birth.source];
                const auto physical=cut_birth26?sample_cut_electron_birth_seeded(cut_birth26,task.birth.source,
                    electron_projectile_parameters,float(task.birth.energy),task.birth.cut,task.birth.cdf_bits):ElectronCutBirthState{};
                const auto sample=cut_birth26?f::ElectronBirthStateSample{{},physical.energy,physical.cosine}:
                    f::sample_independent_electron_birth(node,float(task.birth.energy),f::Draw{task.birth.cdf_bits});
                const auto parent=f::joint_math::unit({task.direction[0],task.direction[1],task.direction[2]});
                const auto direction=f::independent_electron_direction(parent,sample.cosine,float(uniform()));
                packet.cursor.position={f::Coordinate(task.position[0]),f::Coordinate(task.position[1]),f::Coordinate(task.position[2])};
                packet.cursor.direction={direction.x,direction.y,direction.z};packet.cursor.energy_MeV=sample.energy;
                packet.cursor.section=cut_birth26?int(task.birth.source):node.section;
                packet.cursor.density_g_cm3=cut_birth26?float(task.density):node.density;
                packet.cursor.valid=(!cut_birth26||physical.valid)&&sample.energy>0&&std::isfinite(f::joint_math::dot(direction,direction));
            }
            packet.cursor.physical_density_g_cm3=float(task.density);
            packet.status=packet.cursor.valid?ElectronPacketStatus::active:ElectronPacketStatus::invalid;
            task_inside=task_outside=0.f;killed=false;
            if constexpr(BroadBeam) {
                const bool tissue_roulette=float(task.density)>electron_tissue_density;
                if(packet.status==ElectronPacketStatus::active &&
                   (float(task.density)<=electron_roulette_density || (tissue_roulette && electron_tissue_survival<1.f))){
                    // Deterministic per-task roulette from a hash of the task seed;
                    // the transport RNG stream of survivors is untouched.
                    std::uint64_t z=task.rng+0x9e3779b97f4a7c15ULL;
                    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;z^=z>>31;
                    if(z<(tissue_roulette?electron_tissue_bits:electron_roulette_bits)){
                        const float w=packet.weight_MeV/(tissue_roulette?electron_tissue_survival:electron_roulette_survival);
                        vr_gain+=w-packet.weight_MeV;packet.weight_MeV=w;increment(electron_vr_work+1,1);
                    }else{vr_gain-=packet.weight_MeV;killed=true;increment(electron_vr_work,1);}
                }
            }
            if(!killed)joint();
            previous=packet;previous_rng=packet_rng;packet_steps=0;need_task=false;
        }
        if(!killed&&packet.status==ElectronPacketStatus::active&&packet_steps<200000u){
            previous=packet;previous_rng=packet_rng;
            if(packet.cursor.pdg==22){
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
                if(guard_target!=UINT64_MAX)++guard_photons;
#endif
                const auto r=f::transport_independent_photon(float_photon,geometry,packet.cursor.position,
                    packet.cursor.direction,packet.cursor.energy_MeV,uniform);
                if constexpr(region_diagnostics){++region_counts[8];region_counts[9]+=r.interactions;}
                const std::uint64_t counts[8]={1,r.interactions,r.voxel_crossings,
                    r.status==f::IndependentPhotonStatus::electron,r.status==f::IndependentPhotonStatus::deposited,
                    r.status==f::IndependentPhotonStatus::escaped,r.status==f::IndependentPhotonStatus::coverage_missing,
                    r.status==f::IndependentPhotonStatus::invalid};
                for(unsigned k=0;k<8;++k)if(counts[k])increment(independent_photon_work+k,counts[k]);
                packet.advances+=r.interactions;packet.cursor.position=r.position;packet.cursor.direction=r.direction;packet.cursor.energy_MeV=r.energy;
                if(r.status==f::IndependentPhotonStatus::deposited){packet.deposit_position=r.position;packet.status=ElectronPacketStatus::deposited;}
                else if(r.status==f::IndependentPhotonStatus::escaped)packet.status=ElectronPacketStatus::escaped;
                else if(r.status==f::IndependentPhotonStatus::electron){
                    const auto node=float_photon.nodes[r.node];packet.cursor=f::ElectronContinuationCursor{};
                    packet.cursor.valid=true;packet.cursor.position=r.position;packet.cursor.direction=r.direction;packet.cursor.energy_MeV=r.energy;
                    packet.cursor.section=node.section;packet.cursor.density_g_cm3=node.density;packet.cursor.physical_density_g_cm3=r.physical_density;
                    ++packet.electron_children;joint();
                }else packet.status=ElectronPacketStatus::invalid;
            }else{
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
                ++segment_charged_calls;
                if(segment_diagnostic::selected(task.rng,packet_steps)){
                    const auto selected=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(float_segment_counts[1]).fetch_add(1);
                    if(selected<segment_diagnostic::capacity){
                        auto sample=segment_diagnostic::observe(packet,float_charged,geometry,segment_scoring);
                        sample.history=task.history;sample.birth_rng=task.rng;sample.birth_step=task.step;sample.electron_step=packet_steps;
                        float_segment_samples[selected]=sample;
                    }
                }
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
                if(guard_target==UINT64_MAX)guard_target=guard::eligible(packet,float_charged,geometry,scoring);
                ++guard_total_charged;if(guard_target!=UINT64_MAX)++guard_charged;
#endif
                std::uint64_t counts[8]{};auto ct=[&](int k,std::uint64_t n){
                    if constexpr(region_diagnostics){if(k<5)region_counts[1+k]+=n;}
#ifdef CARBON_ELECTRON_STEP_DIAGNOSTICS
                    if(k>=8){increment(float_step_diagnostics+(lane[0]%64)*1536+(k-8),n);return;}
#endif
                    counts[k]+=n;
                };
#ifdef CARBON_ELECTRON_VOXEL_CACHE
                packet=f::voxel_cache_research::transport_packet(packet,float_charged,geometry,voxel_cache,uniform,ct,
                    score_expected,electron_expected_density,electron_expected_min_energy);
#else
                packet=f::transport_independent_charged_packet(packet,float_charged,geometry,uniform,ct);
#endif
                lane_charged_calls+=static_cast<std::uint32_t>(counts[0]);lane_charged_elastic+=static_cast<std::uint32_t>(counts[1]);
                for(unsigned k=2;k<8;++k)if(counts[k])increment(independent_charged_work+k,counts[k]);
                if(packet.status==ElectronPacketStatus::active&&packet.cursor.pdg==11&&packet_steps%32==31)joint();
            }
            ++packet_steps;
        }
        if(!killed&&packet.status==ElectronPacketStatus::active&&packet_steps<200000u)continue;
        need_task=true;
        if constexpr(region_diagnostics) {
            region_counts[11]=1;
            for(unsigned k=0;k<12;++k)if(region_counts[k])increment(broadbeam_region_work+12*birth_region+k,region_counts[k]);
        }
        const std::uint64_t work[3]={packet.advances,packet.source_restarts,packet.boundary_restarts};
        for(unsigned k=0;k<3;++k)increment(material_queue_work+k,work[k]);
        tally(6,1);flush_expected();
        // Remaining statistical weight; equals task.weight without variance reduction.
        const float weight=packet.weight_MeV;
        float inside_total=task_inside,outside_total=task_outside;
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
        increment(float_range_counts+8,guard_total_charged);increment(float_range_counts+9,1);
        if(guard_target!=UINT64_MAX){
            increment(float_range_counts,1);increment(float_range_counts+1,guard_charged);
            increment(float_range_counts+2,guard_photons);increment(float_range_counts+3,guard_joint);
            if(packet.status==ElectronPacketStatus::deposited){
                const auto actual=guard::scoring_index(scoring,packet.deposit_position);
                if(actual==guard_target)increment(float_range_counts+6,1);
                else{increment(float_range_counts+4,1);add(float_range_delta+guard_target,weight);
                    if(actual!=UINT64_MAX)add(float_range_delta+actual,-weight);}
            }else if(packet.status==ElectronPacketStatus::escaped){increment(float_range_counts+5,1);add(float_range_delta+guard_target,weight);}
            else increment(float_range_counts+7,1);
        }
#endif
        if(killed){}
        else if(packet.status==ElectronPacketStatus::deposited){
            const auto p=packet.deposit_position;
            const int ix=static_cast<int>(sycl::floor((p[0]-f::Coordinate(voxel_min_x_mm))/f::Coordinate(voxel_size_x_mm)));
            const int iy=static_cast<int>(sycl::floor((p[1]-f::Coordinate(voxel_min_y_mm))/f::Coordinate(voxel_size_y_mm)));
            const int iz=static_cast<int>(sycl::floor(p[2]/f::Coordinate(voxel_size_z_mm)));
            const bool inside=ix>=0&&iy>=0&&iz>=0&&ix<int(voxel_bins_x)&&iy<int(voxel_bins_y)&&iz<int(voxel_bins_z);
            if(inside)inside_total+=weight;else outside_total+=weight;
            if(inside){const auto target=(std::size_t(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;add(voxel_dose_device+target,weight);
                const int dz=static_cast<int>(sycl::floor(p[2]/f::Coordinate(depth_bin_width_mm)));
                if(dz>=0&&dz<int(number_of_bins)){add(dose_device+dz,weight);if(in_fov_dose_device)add(in_fov_dose_device+dz,weight);}
            }
        }else if(packet.status==ElectronPacketStatus::escaped){add(escaped_device+task.history,weight);tally(4,static_cast<std::uint64_t>(weight*1e6f));}
        else{
            increment(material_untracked_device+1,static_cast<std::uint64_t>(weight*1e6f+.5f));tally(2,1);
            const auto first=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(material_untracked_device[2]).fetch_add(1);
            if(first==0){
                MaterialPacketFailure diagnostic;diagnostic.history=task.history;diagnostic.step=task.step;
                diagnostic.rng=previous_rng;diagnostic.before=previous;diagnostic.after=packet;*material_failure_device=diagnostic;
            }
        }
        // Single ledger update per task: expected-value partial deposits plus
        // any final point deposit. Identical to the original without VR.
        if(inside_total+outside_total!=0.f)add(deposited_device+task.history,inside_total+outside_total);
        grid_inside+=inside_total;grid_outside+=outside_total;
        if(inside_total!=0.f)tally(3,static_cast<std::uint64_t>(inside_total*1e6f));
        if(outside_total!=0.f)tally(4,static_cast<std::uint64_t>(outside_total*1e6f));
    }
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
    increment(float_segment_counts,segment_charged_calls);
#endif
    if(lane_charged_calls)increment(independent_charged_work,lane_charged_calls);
    if(lane_charged_elastic)increment(independent_charged_work+1,lane_charged_elastic);
    float_electron_grid_ledger[2*lane[0]]+=grid_inside;
    float_electron_grid_ledger[2*lane[0]+1]+=grid_outside;
    if(vr_gain!=0.f)float_electron_vr_ledger[lane[0]]+=vr_gain;
});
electron_events.push_back(float_event);
