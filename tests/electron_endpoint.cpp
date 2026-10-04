#include "carbon/electron_endpoint_builder.hpp"
#include <random>
#include <iostream>
using namespace carbon;
void need(bool b){if(!b)throw std::runtime_error("Endpoint test failed");}
int main(){std::mt19937_64 rng(10603);std::size_t checked=0;
 for(unsigned n:{1,2,3,63,512,10007}){
  std::vector<ElectronCompactRecord> src(n);std::vector<std::uint32_t> next(n,UINT32_MAX);
  auto real=[&](){return std::bit_cast<double>(rng());};
  for(unsigned i=0;i<n;++i){auto& r=src[i];for(unsigned a=0;a<3;++a){r.pre_position[a]=real();r.post_position[a]=real();r.pre_direction[a]=real();r.post_direction[a]=real();}r.pre_energy_MeV=real();r.post_energy_MeV=real();r.deposited_MeV=real();r.length_mm=real();r.step=i;}
  for(unsigned i=0;i<n;++i)if(i%3!=0){unsigned j=(i+7)%n;next[i]=j;if(i%5!=0){src[i].post_position=src[j].pre_position;src[i].post_direction=src[j].pre_direction;src[i].post_energy_MeV=src[j].pre_energy_MeV;}}
  ElectronCompactTrack t{};t.pdg=11;ElectronCompactMedium m{};auto packed=ElectronEndpointBuilder::build(src,next);
  ElectronLegacyCompactStateView a{src.data(),&t,&m,n,1,1};ElectronCompactStateView b{packed.records.data(),&t,&m,n,1,1,packed.endpoints.data(),packed.endpoints.size()};
  for(unsigned i=0;i<n;++i){need(electron_state_fields_equal(a.read(i),b.read(i)));++checked;}
  need(packed.endpoints.size()==n-packed.shared);need(!b.read(n).valid);
  packed.records[0].post=n+packed.endpoints.size();need(!b.read(0).valid);
 }
 {std::vector<ElectronCompactRecord> src(2);auto packed=ElectronEndpointBuilder::build(src,{1,0});need(packed.endpoints.empty());ElectronCompactTrack t{};ElectronCompactMedium m{};ElectronCompactStateView v{packed.records.data(),&t,&m,2,1,1,nullptr,0};need(v.read(0).valid&&v.read(1).valid);}
 // Signed zero and NaN payloads are distinct, even if ordinary comparisons hide them.
 std::vector<ElectronCompactRecord> s(2);s[0].post_energy_MeV=-0.;s[1].pre_energy_MeV=0.;auto b=ElectronEndpointBuilder::build(s,{1,UINT32_MAX});need(b.shared==0);
 s[0].post_energy_MeV=s[1].pre_energy_MeV=std::bit_cast<double>(0x7ff8000000000042ULL);b=ElectronEndpointBuilder::build(s,{1,UINT32_MAX});need(b.shared==1);
 s[1].pre_energy_MeV=std::bit_cast<double>(0x7ff8000000000043ULL);b=ElectronEndpointBuilder::build(s,{1,UINT32_MAX});need(b.shared==0);
 for(unsigned mode=0;mode<3;++mode){bool rejected=false;try{if(mode==0)ElectronEndpointBuilder::build(s,{2,UINT32_MAX});if(mode==1)ElectronEndpointBuilder::build(s,{1});if(mode==2){s[0].reserved=1;ElectronEndpointBuilder::build(s,{1,UINT32_MAX});}}catch(const std::runtime_error&){rejected=true;}need(rejected);}
 std::cout<<"PASS endpoint bitwise fields="<<checked<<" nonadjacent/signed-zero/NaN/invalid checks\n";
}
