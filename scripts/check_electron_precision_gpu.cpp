#include "carbon/electron_charged_bank_device.hpp"
#include "carbon/electron_charged_step.hpp"
#include <iostream>
struct Pair {unsigned low,high;float density;};
struct Record {unsigned status,steps;};
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: index sha precision");sycl::queue q{sycl::gpu_selector_v};carbon::ChargedBankDevice data(q,argv[1],argv[2]);const auto bank=data.view;
 std::vector<Pair> pairs;
 for(unsigned sec=0;sec<25;++sec){std::vector<unsigned> ids;for(unsigned i=0;i<data.host_nodes.size();++i)if(data.host_nodes[i].section==int(sec))ids.push_back(i);std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return data.host_nodes[a].density<data.host_nodes[b].density;});for(unsigned j=1;j<ids.size();++j)pairs.push_back({ids[j-1],ids[j],float(.5*(data.host_nodes[ids[j-1]].density+data.host_nodes[ids[j]].density))});}
 auto* spec=sycl::malloc_shared<Pair>(pairs.size(),q);std::copy(pairs.begin(),pairs.end(),spec);const unsigned n=pairs.size()*2048;auto* result=sycl::malloc_shared<Record>(n,q);auto* score=sycl::malloc_shared<float>(7,q);for(unsigned j=0;j<7;++j)score[j]=0;
 auto run=[&](auto tag){ constexpr unsigned precision=decltype(tag)::value;return q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){const unsigned i=id[0];auto pair=spec[i%57];const auto* low=bank.nodes+pair.low;const auto* high=bank.nodes+pair.high;auto model=carbon::charged_density_model(low,high,pair.density);auto bracket=carbon::charged_density_bracket(bank,low->section,pair.density);
  std::uint64_t rng=0x324765abcdeULL+i*0x9e3779b97f4a7c15ULL;auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
  float densities[3]={pair.density,pair.density,pair.density};std::uint8_t sections[3]={std::uint8_t(low->section),std::uint8_t(low->section),std::uint8_t(low->section)};carbon::ElectronCtGeometry g{{3,1,1},{-.03,-100,-100},{.02,200,200},densities,sections};
  std::array<double,3> pos{0,0,0},dir{1,0,0};double energy=std::exp(std::log(1e-6)+uniform()*std::log(.65/1e-6));Record r{6,0};
  for(unsigned j=0;j<200000;++j){++r.steps;auto step=carbon::transport_charged_step<precision>(model,g,pos,dir,energy,carbon::charged_density_majorant(bank,bracket,pair.density),uniform);r.status=unsigned(step.status);if(step.status!=carbon::ChargedStepStatus::continued)break;pos=step.position;dir=step.direction;energy=step.energy;}
  result[i]=r;sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[r.status]);sum.fetch_add(1.f);
 });};
 switch(std::stoi(argv[3])){case 0:run(std::integral_constant<unsigned,0>{}).wait_and_throw();break;
 case 1:run(std::integral_constant<unsigned,1>{}).wait_and_throw();break;
 case 2:run(std::integral_constant<unsigned,2>{}).wait_and_throw();break;
 case 3:run(std::integral_constant<unsigned,3>{}).wait_and_throw();break;default:throw std::runtime_error("precision");}
 std::uint64_t counts[7]{},steps=0;for(unsigned i=0;i<n;++i){++counts[result[i].status];steps+=result[i].steps;}bool pass=!counts[0]&&!counts[1]&&!counts[5]&&!counts[6];for(unsigned i=0;i<7;++i)pass=pass&&score[i]==counts[i];
 std::cout<<"{\"passed\":"<<(pass?"true":"false")<<",\"pairs\":"<<pairs.size()<<",\"packets\":"<<n<<",\"steps\":"<<steps<<",\"invalid\":"<<counts[0]<<",\"coverage_missing\":"<<counts[5]<<",\"step_caps_or_active\":"<<counts[1]+counts[6]<<",\"deposited\":"<<counts[2]<<",\"photon_handoff\":"<<counts[3]<<",\"escaped\":"<<counts[4]<<",\"scorer\":\"FP32\"}\n";return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
