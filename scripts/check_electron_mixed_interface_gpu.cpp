#include "carbon/electron_fp32_device.hpp"
namespace f=carbon::electron_fp32;
#include <iostream>
#include <iomanip>
struct Case {unsigned left,right;float energy;f::Position position;std::array<float,3> direction;};
struct Result {unsigned status{},steps{},crossings{},returns{},photons{};std::array<double,3> position{};};
int main(int argc,char** argv){try{
    if(argc!=14)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha cases samples mean-loss-fraction output translation_mm");
    const unsigned n=std::stoul(argv[10]);const float limit=std::stof(argv[11]);const f::Coordinate shift=std::stod(argv[13]);if(!n||n>10000000)throw std::runtime_error("sample count");
    std::ifstream cases(argv[9]);unsigned nc;cases>>nc;if(nc!=12)throw std::runtime_error("expected12 fixtures");
    std::vector<Case> spec(nc);for(auto& c:spec){cases>>c.left>>c.right>>c.energy;for(auto& x:c.position)cases>>x;for(auto& x:c.direction)cases>>x;}if(!cases)throw std::runtime_error("case input");for(auto& c:spec)c.position[2]+=shift;
    sycl::queue q{sycl::gpu_selector_v};carbon::IndependentElectronFloatDevice data(q,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);
    const auto bank=data.charged;const auto pb=data.photon;
    std::ofstream output(argv[12],std::ios::binary);if(!output)throw std::runtime_error("output open");
    auto* results=sycl::malloc_shared<Result>(n,q);auto* score=sycl::malloc_shared<float>(4,q);if(!results||!score)throw std::bad_alloc();
    bool pass=true;
    for(unsigned test=0;test<nc;++test){const auto c=spec[test];if(c.left>=bank.count||c.right>=bank.count)throw std::runtime_error("case node");
        for(unsigned k=0;k<4;++k)score[k]=0;
        q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            const unsigned i=id[0];std::uint64_t rng=0x124abcdef56ULL+i*0x9e3779b97f4a7c15ULL+test*77771ULL;
            auto uniform=[&](){return f::random_draw(rng);};
            float densities[2]={float(bank.nodes[c.left].density),float(bank.nodes[c.right].density)};std::uint8_t sections[2]={std::uint8_t(bank.nodes[c.left].section),std::uint8_t(bank.nodes[c.right].section)};
            const f::ElectronCtGeometry g{{1,1,2},{-20,-20,shift-20},{40,40,20},densities,sections};
            auto pos=c.position;auto dir=c.direction;float energy=c.energy;Result out;out.status=3;unsigned pdg=11;bool crossed=false;
            for(unsigned step=0;step<200000;++step){++out.steps;
                const auto face=f::next_charged_face(g,pos,dir);if(!face.valid)break;
                if(face.escaped){out.status=2;break;}
                const unsigned node=face.voxel==0?c.left:c.right;
                if(pdg==22){
                    ++out.photons;const auto r=f::transport_independent_photon(pb,g,pos,dir,energy,uniform);pos=r.position;dir=r.direction;energy=r.energy;
                    if(r.status==f::IndependentPhotonStatus::deposited){out.status=pos[2]<shift?0:1;break;}
                    if(r.status==f::IndependentPhotonStatus::escaped){out.status=2;break;}
                    if(r.status!=f::IndependentPhotonStatus::electron)break;pdg=11;continue;
                }
                const auto r=f::transport_charged_step(bank.nodes[node],g,pos,dir,energy,bank.majorants[node],uniform,.05f,limit);
                pos=r.position;dir=r.direction;energy=r.energy;
                if(r.face&&pos[2]==shift){++out.crossings;if(crossed)++out.returns;crossed=true;}
                if(r.status==f::ChargedStepStatus::deposited){out.status=pos[2]<shift?0:1;break;}
                if(r.status==f::ChargedStepStatus::escaped){out.status=2;break;}
                if(r.status==f::ChargedStepStatus::photon){pdg=22;continue;}
                if(r.status!=f::ChargedStepStatus::continued)break;
            }
            out.position={double(pos[0]),double(pos[1]),double(pos[2]-shift)};results[i]=out;sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[out.status]);sum.fetch_add(1.f);
        }).wait_and_throw();
        std::uint64_t counts[4]{},steps=0,cross=0,returns=0,photon=0;for(unsigned i=0;i<n;++i){const auto r=results[i];++counts[r.status];steps+=r.steps;cross+=r.crossings;returns+=r.returns;photon+=r.photons;}
        for(unsigned k=0;k<4;++k)if(score[k]!=counts[k])throw std::runtime_error("FP32 tally mismatch");
        pass=pass&&!counts[3];output.write(reinterpret_cast<const char*>(results),sizeof(Result)*n);
        std::cout<<std::setprecision(17)<<"{\"case\":"<<test<<",\"packets\":"<<n<<",\"left\":"<<counts[0]<<",\"right\":"<<counts[1]<<",\"escaped\":"<<counts[2]<<",\"invalid\":"<<counts[3]<<",\"steps\":"<<steps<<",\"crossings\":"<<cross<<",\"returns\":"<<returns<<",\"photon_calls\":"<<photon<<",\"record_bytes\":"<<sizeof(Result)<<"}"<<std::endl;
    }
    if(!output)throw std::runtime_error("output write");sycl::free(results,q);sycl::free(score,q);return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
