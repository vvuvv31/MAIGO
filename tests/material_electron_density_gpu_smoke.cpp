// Real raw genealogy, synthetic non-node densities. Local GPU only.
#include "carbon/material_electron_device.hpp"
#include "carbon/electron_packet_transport.hpp"
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=3 && argc!=4)return 2;
    if(argc==4 && std::string(argv[3])!="--hot-metadata" && std::string(argv[3])!="--device-indices")return 2;
    const bool device_indices=argc==4 && std::string(argv[3])=="--device-indices";
    try {
        sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};
        if(q.get_device().get_info<sycl::info::device::vendor>().find("NVIDIA")==std::string::npos)
            throw std::runtime_error("Local NVIDIA GPU required");
        auto* ownership_ok=sycl::malloc_shared<int>(1,q);
        if(!ownership_ok)throw std::bad_alloc();
        *ownership_ok=0;
        q.single_task([=]() {
            float r[]={0.39880782F,0.37407282F};std::uint8_t s[]={1,1};
            const carbon::ElectronCtGeometry g{{1,2,1},{-0.5,-151.75,0},{1,155.5,100},r,s};
            const auto m=carbon::electron_ct_point_material(g,{0,3.7499963928537983,63},{0,1,0});
            *ownership_ok=m.valid && m.density==r[0] &&
                carbon::electron_ct_point_material(g,{0,3.75,63},{0,-1,0}).density==r[0] &&
                carbon::electron_ct_point_material(g,{0,3.75,63},{0,1,0}).density==r[1];
        }).wait_and_throw();
        if(!*ownership_ok)throw std::runtime_error("Patient birth ownership regression failed");
        sycl::free(ownership_ok,q);
        std::cout<<"PASS GPU patient birth face ownership"<<std::endl;
        const auto index=carbon::MaterialElectronResponseIndex::load(argv[1],argv[2]);
        auto midpoint=[&](int section) {
            auto a=std::find_if(index.files.begin(),index.files.end(),[&](const auto& f){return f.section==section;});
            if(a==index.files.end() || a+1==index.files.end() || (a+1)->section!=section)
                throw std::runtime_error("Missing density brackets");
            const float r=static_cast<float>((a->density_g_cm3+(a+1)->density_g_cm3)/2);
            if(!(r>a->density_g_cm3 && r<(a+1)->density_g_cm3))throw std::runtime_error("Not a strict midpoint");
            return r;
        };
        const float tissue=midpoint(8),lung=midpoint(1);
        carbon::MaterialElectronDeviceBank bank(q,carbon::MaterialElectronMemory::host_mapped,
            argc==4?std::size_t(device_indices?4096:64)*1024*1024:0,device_indices);
        bank.load(index,{{8,tissue},{1,lung}},std::size_t(16)*1024*1024*1024,true);
        std::cout<<"Hot metadata device bytes: "<<bank.hot_device_bytes()<<std::endl;
        if(argc==4 && (!bank.hot_device_bytes() ||
           sycl::get_pointer_type(bank.views(),q.get_context())!=sycl::usm::alloc::device))
            throw std::runtime_error("Hot descriptor was not placed on device");
        auto* rho=sycl::malloc_shared<float>(2,q);
        auto* sections=sycl::malloc_shared<std::uint8_t>(2,q);
        constexpr unsigned n=64;
        struct Result {unsigned status,gap;std::uint64_t crossings,steps;double weight;};
        auto* result=sycl::malloc_shared<Result>(n,q);
        if(!rho || !sections || !result)throw std::bad_alloc();
        const auto* tables=bank.views();const auto count=bank.size();
        unsigned total_crossings=0;
        for(unsigned reverse=0;reverse<2;++reverse) {
            rho[0]=reverse?lung:tissue;rho[1]=reverse?tissue:lung;
            sections[0]=reverse?1:8;sections[1]=reverse?8:1;
            const carbon::ElectronCtGeometry geometry{{1,1,2},{-100,-100,-100},{200,200,100},rho,sections};
            auto event=q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id) {
                std::uint64_t rng=20260909+id[0]+reverse*1000;
                auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return double(rng>>11)*0x1.0p-53;};
                const double mix=uniform(),birth_u=uniform();
                const auto born=carbon::sample_material_electron_birth(sections[0],rho[0],200,mix,birth_u,tables,count);
                carbon::ElectronEnergyPacket p;p.weight_MeV=1;
                if(born.birth.valid && born.birth.fraction>0) {
                    p.cursor=carbon::bind_electron_birth(born.birth,tables[born.table],{0,0,-0.0001},{0,0,1},uniform());
                    p.cursor.physical_density_g_cm3=rho[0];
                    p.status=p.cursor.valid?carbon::ElectronPacketStatus::active:carbon::ElectronPacketStatus::invalid;
                    for(unsigned step=0;step<4096 && p.status==carbon::ElectronPacketStatus::active;++step)
                        p=carbon::transport_electron_packet_step(p,tables,count,geometry,uniform);
                }
                result[id[0]]={unsigned(p.status),unsigned(p.gap),p.boundary_restarts,p.advances,p.weight_MeV};
            });
            event.wait_and_throw();
            const double kernel_ms=1e-6*(event.get_profiling_info<sycl::info::event_profiling::command_end>()-
                event.get_profiling_info<sycl::info::event_profiling::command_start>());
            std::cout<<"kernel_ms reverse="<<reverse<<" value="<<kernel_ms<<std::endl;
            unsigned invalid=0,active=0,gaps=0,photons=0,crossings=0,steps=0;double weight=0;
            for(unsigned i=0;i<n;++i) {
                const auto r=result[i];weight+=r.weight;crossings+=r.crossings;steps+=r.steps;
                std::cout<<"packet_result "<<reverse<<' '<<i<<' '<<r.status<<' '<<r.gap<<' '
                         <<r.crossings<<' '<<r.steps<<' '<<std::hexfloat<<r.weight<<std::defaultfloat<<'\n';
                invalid+=r.status==unsigned(carbon::ElectronPacketStatus::invalid);
                active+=r.status==unsigned(carbon::ElectronPacketStatus::active);
                if(r.status==unsigned(carbon::ElectronPacketStatus::coverage_missing)) {
                    if(r.gap==unsigned(carbon::ElectronPacketGap::photon_continuation))++photons;else ++gaps;
                }
            }
            std::cout<<"reverse="<<reverse<<" rho="<<rho[0]<<" -> "<<rho[1]<<" packets="<<n
                     <<" crossings="<<crossings<<" steps="<<steps<<" invalid="<<invalid<<" active="<<active
                     <<" gaps="<<gaps<<" photon_retained="<<photons<<" weight="<<weight<<std::endl;
            if(!crossings || invalid || active || gaps || weight!=n)throw std::runtime_error("Density interface gate failed");
            total_crossings+=crossings;
        }
        sycl::free(result,q);sycl::free(rho,q);sycl::free(sections,q);
        std::cout<<"PASS non-node bidirectional interface crossings="<<total_crossings<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
