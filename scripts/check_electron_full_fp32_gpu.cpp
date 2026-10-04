#include "carbon/electron_fp32_device.hpp"
#include <iostream>
struct Pair {unsigned low,high;float density;};
struct Record {unsigned status,steps;};
int main(int argc,char** argv){try{
 if(argc!=9)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha");sycl::queue q{sycl::gpu_selector_v};carbon::IndependentElectronFloatDevice data(q,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);const auto bank=data.charged;
 std::vector<Pair> pairs;
 for(unsigned sec=0;sec<25;++sec){std::vector<unsigned> ids;for(unsigned i=0;i<data.charged_nodes.size();++i)if(data.charged_nodes[i].section==int(sec))ids.push_back(i);std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return data.charged_nodes[a].density<data.charged_nodes[b].density;});for(unsigned j=1;j<ids.size();++j)pairs.push_back({ids[j-1],ids[j],float(.5*(data.charged_nodes[ids[j-1]].density+data.charged_nodes[ids[j]].density))});}
 auto* spec=sycl::malloc_shared<Pair>(pairs.size(),q);std::copy(pairs.begin(),pairs.end(),spec);const unsigned n=pairs.size()*2048;auto* result=sycl::malloc_shared<Record>(n,q);auto* score=sycl::malloc_shared<float>(7,q);for(unsigned j=0;j<7;++j)score[j]=0;
 q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){const unsigned i=id[0];auto pair=spec[i%57];const auto* low=bank.nodes+pair.low;const auto* high=bank.nodes+pair.high;auto model=carbon::electron_fp32::charged_density_model(low,high,pair.density);auto bracket=carbon::electron_fp32::charged_density_bracket(bank,low->section,pair.density);
  std::uint64_t rng=0x324765abcdeULL+i*0x9e3779b97f4a7c15ULL;auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
  float densities[3]={pair.density,pair.density,pair.density};std::uint8_t sections[3]={std::uint8_t(low->section),std::uint8_t(low->section),std::uint8_t(low->section)};carbon::electron_fp32::ElectronCtGeometry g{{3,1,1},{-.03f,-100,-100},{.02f,200,200},densities,sections};
  std::array<float,3> pos{0,0,0},dir{1,0,0};float energy=std::exp(std::log(1e-6f)+float(uniform())*std::log(.65f/1e-6f));Record r{6,0};
  for(unsigned j=0;j<200000;++j){++r.steps;auto step=carbon::electron_fp32::transport_charged_step(model,g,pos,dir,energy,carbon::electron_fp32::charged_density_majorant(bank,bracket,pair.density),uniform);r.status=unsigned(step.status);if(step.status!=carbon::electron_fp32::ChargedStepStatus::continued)break;pos=step.position;dir=step.direction;energy=step.energy;}
  result[i]=r;sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[r.status]);sum.fetch_add(1.f);
 }).wait_and_throw();std::uint64_t counts[7]{},steps=0;for(unsigned i=0;i<n;++i){++counts[result[i].status];steps+=result[i].steps;}bool pass=!counts[0]&&!counts[1]&&!counts[5]&&!counts[6];for(unsigned i=0;i<7;++i)pass=pass&&score[i]==counts[i];
 std::cout<<"{\"passed\":"<<(pass?"true":"false")<<",\"pairs\":"<<pairs.size()<<",\"packets\":"<<n<<",\"steps\":"<<steps<<",\"invalid\":"<<counts[0]<<",\"coverage_missing\":"<<counts[5]<<",\"step_caps_or_active\":"<<counts[1]+counts[6]<<",\"deposited\":"<<counts[2]<<",\"photon_handoff\":"<<counts[3]<<",\"escaped\":"<<counts[4]<<",\"scorer\":\"FP32\"}\n";return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
