#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_charged_step.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
struct Record {unsigned status{},steps{},elastic{},hard{},faces{};double length{},closure{};};
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: index");
    const std::filesystem::path path=argv[1];
    std::ifstream index(path);std::string magic;unsigned nodes;index>>magic>>nodes;
    if(magic!="MAIECB02"||!nodes||nodes>83)throw std::runtime_error("Index format");
    sycl::queue q{sycl::gpu_selector_v};std::uint64_t counts[7]{},elastic=0,hard=0,steps=0,faces=0,total=0,bytes=0;double maxLength=0,maxClosure=0;
    constexpr unsigned n=2048;
    for(unsigned node=0;node<nodes;++node){
        int section;double density;std::string file;index>>section>>density>>file;if(!index)throw std::runtime_error("Index row");
        carbon::ChargedResponseData data(path.parent_path()/file);if(data.section!=section||data.density!=density)throw std::runtime_error("Index identity");bytes+=data.bytes;
        auto view=data.view();std::vector<void*> allocations;
        auto upload=[&]<class T>(const T* source,std::size_t size){auto* ptr=sycl::malloc_shared<T>(size,q);if(!ptr)throw std::bad_alloc();std::copy(source,source+size,ptr);allocations.push_back(ptr);return ptr;};
        view.parameters=upload(view.parameters,view.parameter_count);
        std::array<double,2> majorant{};
        for(unsigned p=0;p<2;++p){for(const auto& piece:data.rate[p])for(double x:piece.value)majorant[p]=std::max(majorant[p],6*std::abs(x));view.rate[p]=upload(view.rate[p],view.rate_count[p]);}
        view.edges=upload(view.edges,view.bins+1);view.channels=upload(view.channels,2*view.bins);view.outcomes=upload(view.outcomes,view.outcome_count);view.elastic.rows=upload(view.elastic.rows,31*38);
        auto* results=sycl::malloc_shared<Record>(n,q);auto* score=sycl::malloc_shared<float>(7,q);if(!results||!score)throw std::bad_alloc();for(unsigned k=0;k<7;++k)score[k]=0;
        q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            const unsigned i=id[0];std::uint64_t rng=0x91234abcdefULL+i*747796405ULL+node*31337ULL;
            auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
            float densities[3]={float(view.density),float(view.density),float(view.density)};std::uint8_t sections[3]={std::uint8_t(view.section),std::uint8_t(view.section),std::uint8_t(view.section)};
            const carbon::ElectronCtGeometry g{{3,1,1},{-.03,-100,-100},{.02,200,200},densities,sections};
            std::array<double,3> pos{0,0,0},dir{1,0,0};
            double energy=std::exp(std::log(1e-6)+uniform()*std::log(.65/1e-6));
            Record r;r.status=6;
            for(unsigned j=0;j<200000;++j){
                const auto result=carbon::transport_charged_step(view,g,pos,dir,energy,majorant,uniform);
                ++r.steps;r.elastic+=result.elastic;r.hard+=result.hard_collision;r.faces+=result.face;
                r.length=std::max(r.length,result.length);
                if(result.status==carbon::ChargedStepStatus::continued||result.status==carbon::ChargedStepStatus::photon){
                    if(!std::isfinite(result.energy)||result.energy<=0||result.energy>energy){r.status=0;break;}
                    if(!result.hard_collision)r.closure=std::max(r.closure,std::abs(result.energy+result.physical_deposit-energy));
                }
                if(result.status!=carbon::ChargedStepStatus::continued){r.status=unsigned(result.status);break;}
                pos=result.position;dir=result.direction;energy=result.energy;
            }
            results[i]=r;
            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[r.status]);sum.fetch_add(1.f);
        }).wait_and_throw();
        unsigned tally[7]{};for(unsigned i=0;i<n;++i){const auto r=results[i];++tally[r.status];steps+=r.steps;elastic+=r.elastic;hard+=r.hard;faces+=r.faces;maxLength=std::max(maxLength,r.length);maxClosure=std::max(maxClosure,r.closure);}
        for(unsigned k=0;k<7;++k){if(score[k]!=tally[k])throw std::runtime_error("FP32 packet weight mismatch");counts[k]+=tally[k];}
        total+=n;if(tally[0]||tally[5]||tally[6])std::cerr<<"node "<<node<<" invalid "<<tally[0]<<" gaps "<<tally[5]<<" caps "<<tally[6]<<'\n';
        for(void* ptr:allocations)sycl::free(ptr,q);sycl::free(results,q);sycl::free(score,q);
    }
    std::string extra;if(index>>extra)throw std::runtime_error("Trailing index fields");
    const bool pass=counts[0]==0&&counts[1]==0&&counts[5]==0&&counts[6]==0&&maxLength<=.25&&maxClosure<1e-12&&counts[2]+counts[3]+counts[4]==total;
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"nodes\":"<<nodes<<",\"bytes\":"<<bytes<<",\"packets\":"<<total<<",\"invalid\":"<<counts[0]<<",\"deposited\":"<<counts[2]<<",\"photon_handoff\":"<<counts[3]<<",\"escaped\":"<<counts[4]<<",\"coverage_missing\":"<<counts[5]<<",\"step_caps\":"<<counts[6]<<",\"steps\":"<<steps<<",\"elastic\":"<<elastic<<",\"hard\":"<<hard<<",\"faces\":"<<faces<<",\"maximum_step_mm\":"<<maxLength<<",\"maximum_soft_energy_residual_MeV\":"<<maxClosure<<",\"scorer\":\"FP32\",\"scope\":\"Standalone exact density nodes; photon handoff is retained energy, not deposition. No native interface dose or CT acceptance.\"}\n";return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
