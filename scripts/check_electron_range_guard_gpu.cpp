// Observational only: keep the full charged trajectory and RNG, record what an
// early scoring-voxel deposit would have done. No particle is actually killed.
#include "carbon/electron_fp32_device.hpp"
#include "carbon/electron_voxel_cache.hpp"
#include <iostream>
#include <iomanip>
namespace f=carbon::electron_fp32;
namespace c=carbon::electron_fp32::voxel_cache_research;
struct Pair {unsigned low,high;float density;};
struct Case {Pair pair;float energy,spacing;};
struct Record {unsigned steps{},status{},invalid{};unsigned first[3]{UINT32_MAX,UINT32_MAX,UINT32_MAX};unsigned moved[3]{},photon[3]{},escaped[3]{};};
struct Total {std::uint64_t steps{},invalid{},deposited{},photon{},escaped{};std::uint64_t eligible[3]{},saved_steps[3]{},moved[3]{},photon_after[3]{},escaped_after[3]{};};
int main(int argc,char** argv){try{
    if(argc!=11)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha samples output.json");
    const unsigned samples=std::stoul(argv[9]);if(!samples||samples>65536)throw std::runtime_error("samples");
    sycl::queue queue{sycl::gpu_selector_v};
    carbon::IndependentElectronFloatDevice data(queue,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);
    const auto bank=data.charged;
    std::vector<Pair> pairs;
    for(unsigned i=0;i<data.charged_nodes.size();++i)pairs.push_back({i,i,data.charged_nodes[i].density});
    for(int section=0;section<25;++section){
        std::vector<unsigned> ids;for(unsigned i=0;i<data.charged_nodes.size();++i)if(data.charged_nodes[i].section==section)ids.push_back(i);
        std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return data.charged_nodes[a].density<data.charged_nodes[b].density;});
        for(unsigned i=1;i<ids.size();++i)pairs.push_back({ids[i-1],ids[i],.5f*(data.charged_nodes[ids[i-1]].density+data.charged_nodes[ids[i]].density)});
    }
    std::vector<Case> cases;
    for(const auto& pair:pairs)for(float e:{.0002f,.001f,.003f,.01f,.03f,.1f,.3f,.65f})for(float s:{.02f,.5f})cases.push_back({pair,e,s});
    const unsigned nc=cases.size();const std::size_t count=std::size_t(nc)*samples;
    auto* spec=sycl::malloc_device<Case>(nc,queue);queue.copy(cases.data(),spec,nc).wait_and_throw();
    auto* records=sycl::malloc_shared<Record>(count,queue);if(!records||!spec)throw std::bad_alloc();
    queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
        const auto i=id[0];const auto cs=spec[i/samples];const auto pair=cs.pair;
        std::uint64_t rng=0x32abcdef97651ULL+i;rng=(rng^(rng>>30))*0xbf58476d1ce4e5b9ULL;rng=(rng^(rng>>27))*0x94d049bb133111ebULL;rng^=rng>>31;
        auto uniform=[&](){return f::random_draw(rng);};
        float rho[27];std::uint8_t section[27];for(unsigned k=0;k<27;++k){rho[k]=pair.density;section[k]=bank.nodes[pair.low].section;}
        f::ElectronCtGeometry g{{3,3,3},{-169.1015f,-2.03f,300.f},{cs.spacing,cs.spacing,cs.spacing},rho,section};
        f::ElectronEnergyPacket packet;packet.status=carbon::ElectronPacketStatus::active;packet.cursor.valid=true;packet.weight_MeV=1;
        for(unsigned k=0;k<3;++k)packet.cursor.position[k]=g.origin[k]+cs.spacing*(1+float(uniform()));
        auto dir=f::joint_math::unit({2*float(uniform())-1,2*float(uniform())-1,2*float(uniform())-1});
        packet.cursor.direction={dir.x,dir.y,dir.z};packet.cursor.energy_MeV=cs.energy;packet.cursor.section=section[0];
        packet.cursor.density_g_cm3=packet.cursor.physical_density_g_cm3=pair.density;
        const auto model=f::charged_density_model(bank.nodes+pair.low,bank.nodes+pair.high,pair.density);
        c::Cache cache;Record r;std::uint64_t targets[3]{};
        constexpr float factor[]={1,1.5f,2};
        for(unsigned step=0;step<200000;++step){
            if(r.first[2]==UINT32_MAX){
                const auto face=f::next_charged_face(g,packet.cursor.position,packet.cursor.direction);
                if(face.valid&&!face.escaped&&packet.cursor.energy_MeV>model.lowest_energy){
                    const auto parameters=f::charged_parameters_certified(model,packet.cursor.energy_MeV);
                    float safety=INFINITY,rounding=0;auto cell=face.voxel;
                    for(unsigned k=0;k<3;++k){
                        const auto coordinate=cell%g.bins[k];cell/=g.bins[k];
                        const float low=g.origin[k]+coordinate*g.spacing[k],high=g.origin[k]+(coordinate+1)*g.spacing[k];
                        safety=std::min(safety,std::min(packet.cursor.position[k]-low,high-packet.cursor.position[k]));
                        rounding=std::max(rounding,4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(packet.cursor.position[k])));
                    }
                    for(unsigned k=0;k<3;++k)if(r.first[k]==UINT32_MAX&&parameters.valid&&safety>factor[k]*parameters.range+rounding){r.first[k]=step;targets[k]=face.voxel;}
                }
            }
            auto tally=[](int,std::uint64_t){};
            packet=c::transport_packet(packet,bank,g,cache,uniform,tally);++r.steps;
            if(packet.status!=carbon::ElectronPacketStatus::active||packet.cursor.pdg==22)break;
        }
        const bool deposited=packet.status==carbon::ElectronPacketStatus::deposited;
        const bool photon=packet.status==carbon::ElectronPacketStatus::active&&packet.cursor.pdg==22;
        const bool escaped=packet.status==carbon::ElectronPacketStatus::escaped;
        r.status=deposited?0:photon?1:escaped?2:3;r.invalid=r.status==3;
        std::uint64_t voxel=UINT64_MAX;
        if(deposited){
            // Match the production scorer's represented floor operation,
            // rather than using direction-dependent transport ownership.
            int xyz[3];bool inside=true;for(unsigned k=0;k<3;++k){xyz[k]=int(std::floor((packet.deposit_position[k]-g.origin[k])/g.spacing[k]));inside=inside&&xyz[k]>=0&&xyz[k]<3;}
            if(inside)voxel=(std::uint64_t(xyz[2])*3+xyz[1])*3+xyz[0];
        }
        for(unsigned k=0;k<3;++k)if(r.first[k]!=UINT32_MAX){r.moved[k]=deposited&&voxel!=targets[k];r.photon[k]=photon;r.escaped[k]=escaped;}
        records[i]=r;
    }).wait_and_throw();
    std::vector<Total> totals(nc);std::uint64_t invalid=0;
    for(std::size_t i=0;i<count;++i){const auto r=records[i];auto& t=totals[i/samples];t.steps+=r.steps;t.invalid+=r.invalid;invalid+=r.invalid;t.deposited+=r.status==0;t.photon+=r.status==1;t.escaped+=r.status==2;
        for(unsigned k=0;k<3;++k)if(r.first[k]!=UINT32_MAX){++t.eligible[k];t.saved_steps[k]+=r.steps-r.first[k];t.moved[k]+=r.moved[k];t.photon_after[k]+=r.photon[k];t.escaped_after[k]+=r.escaped[k];}
    }
    std::ofstream out(argv[10]);out<<std::setprecision(17)<<"{\"scope\":\"Observational homogeneous charged-only packet trajectories; photon handoff conservatively counts as unresolved, no joint/photon continuation or CT-dose acceptance\",\"samples_per_case\":"<<samples<<",\"density_cases\":"<<pairs.size()<<",\"cases\":"<<nc<<",\"packets\":"<<count<<",\"invalid\":"<<invalid<<",\"results\":[\n";
    for(unsigned i=0;i<nc;++i){const auto cs=cases[i];const auto t=totals[i];if(i)out<<",\n";out<<"{\"node_low\":"<<cs.pair.low<<",\"node_high\":"<<cs.pair.high<<",\"density\":"<<cs.pair.density<<",\"energy_MeV\":"<<cs.energy<<",\"spacing_mm\":"<<cs.spacing<<",\"steps\":"<<t.steps<<",\"deposited\":"<<t.deposited<<",\"photon\":"<<t.photon<<",\"escaped\":"<<t.escaped<<",\"invalid\":"<<t.invalid<<",\"guards\":[";
        for(unsigned k=0;k<3;++k){if(k)out<<',';out<<"{\"factor\":"<<1+.5*k<<",\"eligible\":"<<t.eligible[k]<<",\"saved_charged_steps\":"<<t.saved_steps[k]<<",\"different_deposit_voxel\":"<<t.moved[k]<<",\"photon_handoff\":"<<t.photon_after[k]<<",\"escaped_after_guard\":"<<t.escaped_after[k]<<'}';}out<<"]}";
    }
    out<<"]}\n";if(!out)throw std::runtime_error("output write");
    std::cout<<"cases="<<nc<<" packets="<<count<<" invalid="<<invalid<<'\n';
    sycl::free(records,queue);sycl::free(spec,queue);return invalid?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
