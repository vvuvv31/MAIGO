#include "carbon/electron_fp32_device.hpp"
#include "carbon/electron_elastic_batch.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
namespace f=carbon::electron_fp32;
struct Case {unsigned left,right;float energy;f::Position position;std::array<float,3> direction;float density[2]{};};
struct Record {float position[3];unsigned status,steps,batched,elastic,photons,guard;};
static_assert(sizeof(Record)==36);
int main(int argc,char** argv){try{
 if(argc!=15)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha cases samples loss_fraction output translation_mm scoring_spacing_mm");
 const unsigned n=std::stoul(argv[10]);const float fraction=std::stof(argv[11]),shift=std::stof(argv[13]),spacing=std::stof(argv[14]);
 if(!n||n>10000000||fraction<=0||fraction>.01f||spacing<=0||spacing>.5f)throw std::runtime_error("fixture domain");
 std::ifstream in(argv[9]);unsigned nc;in>>nc;std::vector<Case> cases(nc);
 std::string line;std::getline(in,line);
 for(auto& c:cases){
  if(!std::getline(in,line))throw std::runtime_error("case line");std::istringstream row(line);
  row>>c.left>>c.right>>c.energy;for(auto& x:c.position)row>>x;for(auto& x:c.direction)row>>x;
  if(!row)throw std::runtime_error("case fields");
  if(row>>c.density[0]){if(!(row>>c.density[1])||!(c.density[0]>0&&c.density[1]>0))throw std::runtime_error("case densities");}
  c.position[2]+=shift;
 }
 if(!in||!nc||nc>2048)throw std::runtime_error("cases");
 sycl::queue queue(sycl::gpu_selector_v,sycl::property::queue::enable_profiling{});
 carbon::IndependentElectronFloatDevice data(queue,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);const auto bank=data.charged;const auto photons=data.photon;
 auto* result=sycl::malloc_shared<Record>(n,queue);auto* tally=sycl::malloc_shared<float>(4,queue);if(!result||!tally)throw std::bad_alloc();
 for(unsigned arm=0;arm<2;++arm){
  std::ofstream output(std::string(argv[12])+(arm?".candidate.bin":".reference.bin"),std::ios::binary);if(!output)throw std::runtime_error("output open");
  for(unsigned test=0;test<nc;++test){const auto c=cases[test];if(c.left>=bank.count||c.right>=bank.count)throw std::runtime_error("node");
   for(unsigned k=0;k<4;++k)tally[k]=0;
   auto launch=[&]<bool candidate>(){
    sycl::event event=queue.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
     std::uint64_t rng=0x458abcd1209876ULL+id[0]+test*0x9e3779b97f4a7c15ULL+(candidate?0x6a09e667f3bcc909ULL:0);
     rng=(rng^(rng>>30))*0xbf58476d1ce4e5b9ULL;rng=(rng^(rng>>27))*0x94d049bb133111ebULL;rng^=rng>>31;
     auto uniform=[&](){return f::random_draw(rng);};
     float densities[2]={c.density[0]>0?c.density[0]:bank.nodes[c.left].density,c.density[1]>0?c.density[1]:bank.nodes[c.right].density};std::uint8_t sections[2]={std::uint8_t(bank.nodes[c.left].section),std::uint8_t(bank.nodes[c.right].section)};
     const f::ElectronCtGeometry geometry{{1,1,2},{-20,-20,shift-20},{40,40,20},densities,sections};
     const unsigned bins=unsigned(std::round(40/spacing));
     const f::ElectronCtGeometry scoring{{bins,bins,bins},{-20,-20,shift-20},{spacing,spacing,spacing},nullptr,nullptr};
     auto position=c.position;auto direction=c.direction;float energy=c.energy;unsigned pdg=11;Record out{};out.status=3;
     for(unsigned step=0;step<200000;++step){++out.steps;
      if(pdg==22){
       ++out.photons;const auto r=f::transport_independent_photon(photons,geometry,position,direction,energy,uniform);
       position=r.position;direction=r.direction;energy=r.energy;
       if(r.status==f::IndependentPhotonStatus::deposited){out.status=position[2]<shift?0:1;break;}
       if(r.status==f::IndependentPhotonStatus::escaped){out.status=2;break;}
       if(r.status!=f::IndependentPhotonStatus::electron)break;pdg=11;continue;
      }
      const auto face=f::next_charged_face(geometry,position,direction);
      if(!face.valid)break;if(face.escaped){out.status=2;break;}
      const auto bracket=f::charged_density_bracket(bank,face.section,face.density);if(!bracket.valid)break;
      const auto model=f::charged_density_model(bank.nodes+bracket.low,bank.nodes+bracket.high,face.density);
      const auto majorants=f::charged_density_majorant(bank,bracket,face.density);f::ChargedStep r;
      if constexpr(candidate){
       const auto proposed=carbon::electron_elastic_batch_research::transport(model,geometry,scoring,position,direction,energy,majorants,uniform,fraction);
       r=proposed.step;out.batched+=proposed.batched;out.elastic+=proposed.elastic_events;out.guard+=proposed.guard_violation;
      }else{r=f::transport_charged_step(model,geometry,position,direction,energy,majorants,uniform,.05f,fraction);out.elastic+=r.elastic;}
      position=r.position;direction=r.direction;energy=r.energy;
      if(r.status==f::ChargedStepStatus::deposited){out.status=position[2]<shift?0:1;break;}
      if(r.status==f::ChargedStepStatus::escaped){out.status=2;break;}
      if(r.status==f::ChargedStepStatus::photon){pdg=22;continue;}
      if(r.status!=f::ChargedStepStatus::continued)break;
     }
     out.position[0]=position[0];out.position[1]=position[1];out.position[2]=position[2]-shift;result[id[0]]=out;
     sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(tally[out.status]).fetch_add(1.f);
    });event.wait_and_throw();
   };
   if(arm)launch.template operator()<true>();else launch.template operator()<false>();
   std::uint64_t counts[4]{},steps=0,batches=0,elastic=0,photon=0,guard=0;
   for(unsigned i=0;i<n;++i){const auto r=result[i];++counts[r.status];steps+=r.steps;batches+=r.batched;elastic+=r.elastic;photon+=r.photons;guard+=r.guard;}
   for(unsigned k=0;k<4;++k)if(tally[k]!=counts[k])throw std::runtime_error("FP32 tally closure");
   output.write(reinterpret_cast<const char*>(result),n*sizeof(Record));
   std::cout<<"{\"arm\":"<<arm<<",\"case\":"<<test<<",\"packets\":"<<n<<",\"left\":"<<counts[0]<<",\"right\":"<<counts[1]<<",\"escaped\":"<<counts[2]<<",\"invalid\":"<<counts[3]<<",\"steps\":"<<steps<<",\"batched_calls\":"<<batches<<",\"elastic_events\":"<<elastic<<",\"photon_calls\":"<<photon<<",\"guard_violations\":"<<guard<<"}"<<std::endl;
   if(counts[3]||guard)throw std::runtime_error("invalid or guard violation");
  }
  if(!output)throw std::runtime_error("write");
 }
 sycl::free(result,queue);sycl::free(tally,queue);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
