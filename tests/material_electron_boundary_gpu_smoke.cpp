// One real-table interface check. Not a dose/physics acceptance benchmark.
#include "carbon/material_electron_device.hpp"
#include "carbon/material_electron_replay.hpp"
#include "carbon/electron_continuation_transport.hpp"
#include "carbon/electron_energy_lineage.hpp"
#include "carbon/electron_packet_transport.hpp"
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=5) {std::cerr<<"usage: smoke index index_sha256 source_tag target_tag\n";return 2;}
    try {
        sycl::queue queue{sycl::gpu_selector_v};
        if(queue.get_device().get_info<sycl::info::device::vendor>().find("NVIDIA")==std::string::npos)
            throw std::runtime_error("This smoke requires the local NVIDIA GPU");
        const auto index=carbon::MaterialElectronResponseIndex::load(argv[1],argv[2]);
        const auto found=std::find_if(index.files.begin(),index.files.end(),
            [&](const auto& f){return f.path.parent_path().filename()==argv[3];});
        if(found==index.files.end() || found->section<0)throw std::runtime_error("Missing CT table");
        const auto target=std::find_if(index.files.begin(),index.files.end(),
            [&](const auto& f){return f.path.parent_path().filename()==argv[4];});
        if(target==index.files.end() || target->section<0 || target==found)throw std::runtime_error("Missing/distinct target CT table required");
        carbon::MaterialElectronDeviceBank bank(queue);
        bank.load(index,{{found->section,found->density_g_cm3},{target->section,target->density_g_cm3}},std::size_t(8)*1024*1024*1024);
        if(bank.size()!=2)throw std::runtime_error("Smoke requires two exact-density tables");
        const unsigned source_index=found<target?0:1,target_index=1-source_index;
        const auto* tables=bank.views();
        constexpr std::size_t n=256;
        struct Result {unsigned status,continuation_status,advance_status,steps,children,recrossings;double residual,advance_residual,deposit;bool scanned;};
        auto* output=sycl::malloc_shared<Result>(n,queue);
        auto* rho=sycl::malloc_shared<float>(2,queue);
        auto* section=sycl::malloc_shared<std::uint8_t>(2,queue);
        if(!output || !rho || !section)throw std::bad_alloc();
        rho[0]=static_cast<float>(found->density_g_cm3);rho[1]=static_cast<float>(target->density_g_cm3);
        section[0]=static_cast<std::uint8_t>(found->section);section[1]=static_cast<std::uint8_t>(target->section);
        // Source 0.1 micron before the z interface: exercise real ancestry
        // crossing while keeping the physical path itself unmodified.
        carbon::ElectronCtGeometry geometry{{1,1,2},{-100,-100,-100},{200,200,100},rho,section};
        carbon::MaterialElectronFrame frame;frame.origin={0,0,-0.0001};
        queue.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id) {
            const auto i=id[0];
            const auto table=tables[source_index];
            const double u=(double(i)+0.5)/n;
            const auto draw=carbon::sample_material_electron_response(table.section,
                table.density_g_cm3,400,0.5,u,0.5,tables,2);
            const auto request=carbon::locate_material_electron_boundary(draw,table,geometry,frame);
            unsigned status=0;
            if(request.path_status==carbon::MaterialElectronPathVisit::completed)status=1;
            else if(request.path_status==carbon::MaterialElectronPathVisit::interrupted &&
                    request.replay_step.status==carbon::ElectronReplayStatus::ready)
                status=request.boundary.status==carbon::ElectronCtBoundaryStatus::material_change?2:3;
            unsigned continuation_status=8;
            unsigned advance_status=6;double advance_residual=0,deposit=0;
            unsigned steps=0,children=0,recrossings=0;
            bool scanned=false;
            if(status==2) {
                std::uint64_t rng=1234567+i;
                auto uniform=[&]() {
                    rng=rng*6364136223846793005ULL+1442695040888963407ULL;
                    return double(rng>>11)*0x1.0p-53;
                };
                const auto continued=carbon::sample_candidate_material_boundary_continuation(request,tables[target_index],uniform);
                continuation_status=static_cast<unsigned>(continued.sampled.status);
                scanned=continued.sampled.exact_bucket_scan;
                if(continued.sampled.status==carbon::ElectronCrossingStatus::ready) {
                    auto cursor=carbon::bind_electron_boundary_continuation(request,continued,tables[target_index],uniform());
                    unsigned current_table=target_index;
                    const double initial_energy=cursor.energy_MeV;
                    double children_energy=0;
                    // A short actual continuation, not a whole-family dose run.
                    // Remaining parent/child energy is retained, never killed
                    // or deposited when this diagnostic's observation ends.
                    for(unsigned j=0;j<16;++j) {
                        const auto advance=carbon::advance_electron_continuation(cursor,tables[current_table],geometry);
                        advance_status=static_cast<unsigned>(advance.status);
                        if(advance.status==carbon::ElectronAdvanceStatus::invalid)break;
                        ++steps;deposit+=advance.deposited_MeV;
                        advance_residual=std::max(advance_residual,std::abs(advance.energy_residual_MeV));
                        double child_energy=0;
                        for(auto c=advance.children_begin;c<advance.children_end;++c) {
                            const auto child=carbon::electron_continuation_child(cursor,advance,tables[current_table],c);
                            if(!child.valid) {advance_status=0;break;}
                            ++children;child_energy+=child.energy_MeV;
                        }
                        if(std::abs(child_energy-advance.child_energy_MeV)>1e-8)advance_status=0;
                        children_energy+=child_energy;
                        advance_residual=std::max(advance_residual,std::abs(initial_energy-deposit-children_energy-advance.next.energy_MeV));
                        if(!advance_status)break;
                        if(advance.status==carbon::ElectronAdvanceStatus::material_boundary) {
                            unsigned destination=2;
                            for(unsigned t=0;t<2;++t)
                                if(tables[t].section==advance.boundary.to_section &&
                                   static_cast<float>(tables[t].density_g_cm3)==advance.boundary.to_density)destination=t;
                            if(destination==2) {advance_status=0;break;}
                            const auto sampled=carbon::sample_material_electron_continuation(tables[destination],11,advance.next.energy_MeV,uniform);
                            const auto rebound=carbon::bind_electron_continuation(sampled,tables[destination],advance.end,advance.next.direction,uniform());
                            if(!rebound.valid) {advance_status=0;break;}
                            ++recrossings;cursor=rebound;current_table=destination;
                            continue;
                        }
                        if(advance.status!=carbon::ElectronAdvanceStatus::step_complete)break;
                        cursor=advance.next;
                    }
                }
            }
            output[i]={status,continuation_status,advance_status,steps,children,recrossings,request.replay_step.energy_residual_MeV,advance_residual,deposit,scanned};
        }).wait_and_throw();
        std::size_t counts[4]{},continuations[9]{},advances[7]{},scanned=0;
        double residual=0,advance_residual=0,deposit=0;
        std::uint64_t steps=0,children=0,recrossings=0;
        for(std::size_t i=0;i<n;++i) {
            ++counts[output[i].status];++continuations[output[i].continuation_status];
            scanned+=output[i].scanned;
            ++advances[output[i].advance_status];deposit+=output[i].deposit;
            steps+=output[i].steps;children+=output[i].children;
            recrossings+=output[i].recrossings;
            advance_residual=std::max(advance_residual,std::abs(output[i].advance_residual));
            residual=std::max(residual,std::abs(output[i].residual));
        }
        sycl::free(output,queue);
        // ---- Second diagnostic (small, bounded): birth CDF -> bind -> advance
        // -> choose lineage on the actual source table, ~256 packets x 16
        // transitions max. Packet weight W=1 is scored only for a selected
        // deposit; otherwise W is preserved in explicit terminal buckets.
        // Electron children continue via the selected cursor. Boundaries use
        // the existing exact-density sampler when resident, else boundary
        // pending without claiming full-transport closure. Conservation:
        // VALID terminal packets (buckets 2..7) == n2 with
        // birth_invalid==0 AND lineage_invalid==0; invalid is reported
        // separately and fails. boundary_pending stays a diagnostic
        // unresolved bucket, never a full-transport PASS.
        constexpr std::size_t n2=256;
        struct LineageResult {unsigned outcome;double physical_deposit;unsigned steps;unsigned parent_selected;unsigned electron_child_selected;unsigned boundary_rebinds;};
        auto* lin_output=sycl::malloc_shared<LineageResult>(n2,queue);
        if(!lin_output)throw std::bad_alloc();
        queue.parallel_for(sycl::range<1>(n2),[=](sycl::id<1> id) {
            const auto i=id[0];
            std::uint64_t rng=987654321ULL+i*2654435761ULL;
            auto uniform=[&]() {
                rng=rng*6364136223846793005ULL+1442695040888963407ULL;
                return double(rng>>11)*0x1.0p-53;
            };
            const auto source_table=tables[source_index];
            const double u_birth=(double(i)+0.5)/double(n2);
            unsigned outcome=0;double physical_deposit=0;unsigned steps=0;
            unsigned parent_selected=0,electron_child_selected=0,boundary_rebinds=0;
            const auto draw=carbon::sample_electron_birth(400.0,u_birth,
                source_table.birth_channels,source_table.channel_count,
                source_table.birth_samples,source_table.birth_sample_count);
            carbon::ElectronContinuationCursor cursor;
            unsigned current_table=source_index;
            if(draw.valid) {
                const std::array<double,3> world{0,0,-0.0001};
                std::array<double,3> carbon_dir{0,0,1};
                cursor=carbon::bind_electron_birth(draw,tables[current_table],world,carbon_dir,uniform());
            }
            if(!draw.valid || !cursor.valid) {
                outcome=0; // birth_invalid
            } else {
                outcome=6; // observation_limit unless terminated earlier
                for(unsigned j=0;j<16;++j) {
                    const auto advance=carbon::advance_electron_continuation(
                        cursor,tables[current_table],geometry);
                    if(advance.status==carbon::ElectronAdvanceStatus::invalid) {outcome=1;break;}
                    const auto lin=carbon::choose_electron_energy_lineage(
                        cursor,advance,tables[current_table],uniform(),uniform());
                    ++steps;
                    if(lin.status==carbon::ElectronLineageStatus::invalid) {outcome=1;break;}
                    if(lin.status==carbon::ElectronLineageStatus::deposited) {
                        outcome=2;physical_deposit=lin.physical_selected_deposit_MeV;break;
                    } else if(lin.status==carbon::ElectronLineageStatus::escaped) {
                        outcome=3;break;
                    } else if(lin.status==carbon::ElectronLineageStatus::photon_pending) {
                        outcome=4;break;
                    } else if(lin.status==carbon::ElectronLineageStatus::source_exhausted) {
                        outcome=5;break;
                    } else if(lin.status==carbon::ElectronLineageStatus::parent_continuation ||
                              lin.status==carbon::ElectronLineageStatus::electron_child) {
                        if(lin.status==carbon::ElectronLineageStatus::parent_continuation)++parent_selected;
                        else ++electron_child_selected;
                        cursor=lin.cursor; // selected child actually continues
                        if(j+1>=16) {outcome=6;break;}
                        continue;
                    } else if(lin.status==carbon::ElectronLineageStatus::material_boundary) {
                        unsigned destination=2;
                        for(unsigned t=0;t<2;++t)
                            if(tables[t].section==advance.boundary.to_section &&
                               static_cast<float>(tables[t].density_g_cm3)==advance.boundary.to_density)destination=t;
                        if(destination==2) {outcome=7;break;}
                        const auto sampled=carbon::sample_material_electron_continuation(
                            tables[destination],11,lin.cursor.energy_MeV,uniform);
                        if(sampled.status!=carbon::ElectronCrossingStatus::ready) {outcome=7;break;}
                        const auto rebound=carbon::bind_electron_continuation(
                            sampled,tables[destination],lin.cursor.position,lin.cursor.direction,uniform());
                        if(!rebound.valid) {outcome=7;break;}
                        ++boundary_rebinds;
                        cursor=rebound;current_table=destination;
                        if(j+1>=16) {outcome=6;break;}
                        continue;
                    } else {outcome=1;break;}
                }
            }
            lin_output[i]={outcome,physical_deposit,steps,parent_selected,electron_child_selected,boundary_rebinds};
        }).wait_and_throw();
        std::size_t lin_counts[8]{};
        double lin_physical=0;std::uint64_t lin_steps=0;
        std::uint64_t lin_parent=0,lin_echild=0,lin_rebind=0;
        for(std::size_t i=0;i<n2;++i) {
            if(lin_output[i].outcome<8)++lin_counts[lin_output[i].outcome];
            lin_physical+=lin_output[i].physical_deposit;
            lin_steps+=lin_output[i].steps;
            lin_parent+=lin_output[i].parent_selected;
            lin_echild+=lin_output[i].electron_child_selected;
            lin_rebind+=lin_output[i].boundary_rebinds;
        }
        sycl::free(lin_output,queue);
        // Bounded actual packet walk: selected photons are advanced, finite
        // electron sources restart. Incomplete coverage and observation caps
        // retain W and are reported separately from deposit/patient escape.
        auto* packets=sycl::malloc_shared<carbon::ElectronEnergyPacket>(n2,queue);
        if(!packets)throw std::bad_alloc();
        queue.parallel_for(sycl::range<1>(n2),[=](sycl::id<1> id) {
            const auto i=id[0];std::uint64_t rng=38293823+i*2654435761ULL;
            auto uniform=[&]() {rng=rng*6364136223846793005ULL+1442695040888963407ULL;return double(rng>>11)*0x1.0p-53;};
            const auto t=tables[source_index];
            const auto birth=carbon::sample_electron_birth(400,(double(i)+0.5)/n2,
                t.birth_channels,t.channel_count,t.birth_samples,t.birth_sample_count);
            carbon::ElectronEnergyPacket p;
            p.cursor=carbon::bind_electron_birth(birth,t,{0,0,-0.0001},{0,0,1},uniform());
            p.weight_MeV=1;
            p.status=p.cursor.valid?carbon::ElectronPacketStatus::active:carbon::ElectronPacketStatus::invalid;
            for(unsigned j=0;j<512 && p.status==carbon::ElectronPacketStatus::active;++j)
                p=carbon::transport_electron_packet_step(p,tables,2,geometry,uniform);
            packets[i]=p;
        }).wait_and_throw();
        std::uint64_t packet_counts[5]{},gap_counts[4]{},photon_children=0,source_restarts=0,packet_steps=0;
        double packet_weight=0;
        for(std::size_t i=0;i<n2;++i) {
            const auto p=packets[i];++packet_counts[static_cast<unsigned>(p.status)];
            ++gap_counts[static_cast<unsigned>(p.gap)];packet_weight+=p.weight_MeV;
            photon_children+=p.photon_children;source_restarts+=p.source_restarts;packet_steps+=p.advances;
        }
        const bool packets_ok=packet_counts[4]==0 && packet_weight==double(n2);
        std::cout<<"packet_walk active="<<packet_counts[0]<<" deposited="<<packet_counts[1]
                 <<" escaped="<<packet_counts[2]<<" coverage_missing="<<packet_counts[3]
                 <<" invalid="<<packet_counts[4]<<" weight="<<packet_weight
                 <<" photon_children="<<photon_children<<" source_restarts="<<source_restarts
                 <<" steps="<<packet_steps<<" photon_gaps="<<gap_counts[3]
                 <<" electron_gaps="<<gap_counts[2]<<" material_gaps="<<gap_counts[1]<<'\n';
        sycl::free(packets,queue);
        // Select a small deterministic set of real photon births explicitly:
        // carbon-birth sampling need not exercise rare bremsstrahlung branches.
        constexpr unsigned photon_n=32;
        auto* photons=sycl::malloc_shared<carbon::ElectronEnergyPacket>(photon_n,queue);
        auto* photon_started=sycl::malloc_shared<unsigned>(1,queue);
        if(!photons || !photon_started)throw std::bad_alloc();
        queue.single_task([=]() {
            *photon_started=0;const auto t=tables[source_index];
            for(std::size_t s=0;s<t.state_source_count && *photon_started<photon_n;++s) {
                const auto src=t.state_sources[s];
                for(std::uint64_t row=0;row<src.bytes/carbon::kElectronStateV4RecordBytes && *photon_started<photon_n;++row) {
                    const auto step=carbon::read_electron_continuation_step(src.data,src.bytes,row);
                    if(!step.valid || step.pdg!=22 || step.step!=1 || step.pre_energy_MeV<=0)continue;
                    carbon::ElectronEnergyPacket p;p.status=carbon::ElectronPacketStatus::active;p.weight_MeV=1;
                    auto& c=p.cursor;c.valid=true;c.source=s;c.row=row;c.pdg=22;c.energy_MeV=step.pre_energy_MeV;
                    c.section=t.section;c.density_g_cm3=t.density_g_cm3;c.position={0,0,0};c.direction=step.pre_direction;
                    for(unsigned k=0;k<3;++k)c.raw_to_world.origin[k]=-step.pre_position[k];
                    photons[(*photon_started)++]=p;
                }
            }
        }).wait_and_throw();
        const auto photon_count=*photon_started;
        const carbon::ElectronCtGeometry photon_geometry{{1,1,1},{-1000,-1000,-1000},{2000,2000,2000},rho,section};
        queue.parallel_for(sycl::range<1>(photon_count),[=](sycl::id<1> id) {
            auto p=photons[id[0]];std::uint64_t rng=432323+id[0];
            auto uniform=[&]() {rng=rng*6364136223846793005ULL+1442695040888963407ULL;return double(rng>>11)*0x1.0p-53;};
            for(unsigned j=0;j<64 && p.status==carbon::ElectronPacketStatus::active;++j)
                p=carbon::transport_electron_packet_step(p,tables,2,photon_geometry,uniform);
            photons[id[0]]=p;
        }).wait_and_throw();
        unsigned photon_states[5]{};std::uint64_t photon_electrons=0;
        for(unsigned i=0;i<photon_count;++i) {
            ++photon_states[static_cast<unsigned>(photons[i].status)];photon_electrons+=photons[i].electron_children;
        }
        const bool photons_ok=photon_count==photon_n && photon_states[4]==0;
        std::cout<<"real_photon_walk started="<<photon_count<<" active="<<photon_states[0]
                 <<" deposited="<<photon_states[1]<<" escaped="<<photon_states[2]
                 <<" coverage_missing="<<photon_states[3]<<" invalid="<<photon_states[4]
                 <<" electron_children="<<photon_electrons<<'\n';
        sycl::free(photons,queue);sycl::free(photon_started,queue);sycl::free(rho,queue);sycl::free(section,queue);
        std::cout<<"table="<<argv[3]<<" resident_bytes="<<bank.allocated_bytes()
                 <<" queries="<<n<<" invalid="<<counts[0]<<" contained="<<counts[1]
                 <<" interfaces="<<counts[2]<<" escaped="<<counts[3]
                 <<" max_step_residual_MeV="<<residual<<'\n';
        std::cout<<"continuation_status_counts=";
        for(auto count:continuations)std::cout<<count<<' ';
        std::cout<<"(ready invalid domain empty rejected exhausted unsupported unused not_crossing) exact_bucket_scans="<<scanned<<'\n';
        std::cout<<"advance_status_counts=";
        for(auto count:advances)std::cout<<count<<' ';
        std::cout<<"(invalid step_complete boundary escaped stopped exhausted not_advanced) max_residual_MeV="
                 <<advance_residual<<" deposited_MeV="<<deposit<<" steps="<<steps<<" child_cursors="<<children
                 <<" repeated_interfaces="<<recrossings<<'\n';
        std::cout<<"lineage_diagnostic_queries="<<n2
                 <<" birth_invalid="<<lin_counts[0]<<" lineage_invalid="<<lin_counts[1]
                 <<" deposited_packets="<<lin_counts[2]<<" escaped_packets="<<lin_counts[3]
                 <<" photon_pending="<<lin_counts[4]<<" source_exhausted="<<lin_counts[5]
                 <<" observation_limit="<<lin_counts[6]<<" boundary_pending="<<lin_counts[7]
                 <<" lineage_steps="<<lin_steps
                 <<" parent_selected="<<lin_parent<<" electron_child_selected="<<lin_echild
                 <<" boundary_rebinds="<<lin_rebind
                 <<" physical_selected_deposit_MeV="<<lin_physical<<'\n';
        // This deliberately fixed electron-interface pilot requires coverage
        // of EVERY interface request, not just a nonzero number of successes.
        const bool first_ok=!(counts[0] || !counts[2] || continuations[0]!=counts[2] || advances[0] ||
            !advances[1] || steps<=counts[2] || !(deposit>0) || advance_residual>1e-8);
        // Second diagnostic: strict conservation on VALID terminals only
        // (no forced rare-event counts). birth_invalid and lineage_invalid
        // must both be zero; valid buckets 2..7 must sum to n2. Invalid is
        // reported separately and fails. boundary_pending stays diagnostic
        // unresolved, never a full-transport PASS.
        std::size_t lin_valid_total=0;
        for(std::size_t b=2;b<8;++b)lin_valid_total+=lin_counts[b];
        const bool lin_ok=(lin_counts[0]==0 && lin_counts[1]==0 && lin_valid_total==n2);
        if(lin_counts[0]!=0 || lin_counts[1]!=0)std::cerr<<"lineage invalid packets: birth_invalid="<<lin_counts[0]<<" lineage_invalid="<<lin_counts[1]<<"\n";
        if(lin_valid_total!=n2)std::cerr<<"lineage packet conservation failed: valid_total="<<lin_valid_total<<" expected="<<n2<<"\n";
        if(lin_counts[7]!=0)std::cerr<<"lineage boundary_pending unresolved: diagnostic only, not full-transport PASS\n";
        return (first_ok && lin_ok && packets_ok && photons_ok) ? 0 : 1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
