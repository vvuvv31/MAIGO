#include "carbon/electron_fp32_device.hpp"
#include <iostream>
struct Result{unsigned status{},steps{};};
int main(int argc,char** argv){try{
 if(argc!=9)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha");
 sycl::queue q{sycl::gpu_selector_v};carbon::IndependentElectronFloatDevice data(q,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);
 const auto b=data.birth;const auto j=data.joint;const auto c=data.charged;const auto ph=data.photon;
 const unsigned n=b.count*4096;auto* out=sycl::malloc_shared<Result>(n,q);
 q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){namespace f=carbon::electron_fp32;using S=carbon::ElectronPacketStatus;
  const unsigned i=id[0];const auto node=b.nodes[i%b.count];std::uint64_t rng=0x324765abcdeULL+i*0x9e3779b97f4a7c15ULL;
  auto u=[&](){return f::random_draw(rng);};auto tally=[](int,std::uint64_t){};
  float rho[3]={node.density,node.density,node.density};std::uint8_t sec[3]={std::uint8_t(node.section),std::uint8_t(node.section),std::uint8_t(node.section)};
  f::ElectronCtGeometry g{{3,1,1},{-169.1015f,-100,-100},{.5f,200,200},rho,sec};
  const float parent=std::exp(std::log(.1f)+float(u())*std::log(250.f/.1f));const auto* channel=f::electron_birth_channel(node,parent);
  if(channel && channel->fraction==0 && channel->count==0){out[i]={5,0};return;}
  const auto born=f::sample_independent_electron_birth(node,parent,u());
  const auto d=f::independent_electron_direction({0,0,1},born.cosine,float(u()));
  f::ElectronEnergyPacket p;p.weight_MeV=1;p.cursor.valid=born.energy>0;p.status=p.cursor.valid?S::active:S::invalid;
  p.cursor.position={g.origin[0]+.5f+float(u())*.5f,0,0};p.cursor.direction={d.x,d.y,d.z};p.cursor.energy_MeV=born.energy;
  p.cursor.section=node.section;p.cursor.density_g_cm3=node.density;p.cursor.physical_density_g_cm3=node.density;
  p=f::transport_joint_independent(p,j,g,u,tally);unsigned steps=0;
  while(p.status==S::active&&steps<200000){++steps;
   if(p.cursor.pdg==22){auto r=f::transport_independent_photon(ph,g,p.cursor.position,p.cursor.direction,p.cursor.energy_MeV,u);
    p.cursor.position=r.position;p.cursor.direction=r.direction;p.cursor.energy_MeV=r.energy;
    if(r.status==f::IndependentPhotonStatus::deposited)p.status=S::deposited;
    else if(r.status==f::IndependentPhotonStatus::escaped)p.status=S::escaped;
    else if(r.status==f::IndependentPhotonStatus::electron){p.cursor.pdg=11;p.cursor.section=ph.nodes[r.node].section;p.cursor.density_g_cm3=ph.nodes[r.node].density;p.cursor.physical_density_g_cm3=r.physical_density;p=f::transport_joint_independent(p,j,g,u,tally);}
    else p.status=S::invalid;
   }else {p=f::transport_independent_charged_packet(p,c,g,u,tally);if(p.status==S::active&&p.cursor.pdg==11&&steps%32==0)p=f::transport_joint_independent(p,j,g,u,tally);}
  }
  out[i]={unsigned(p.status),steps};
 }).wait_and_throw();std::uint64_t counts[6]{},steps=0;for(unsigned i=0;i<n;++i){++counts[out[i].status];steps+=out[i].steps;}
 bool ok=!counts[0]&&!counts[3]&&!counts[4];std::cout<<"{\"passed\":"<<(ok?"true":"false")<<",\"packets\":"<<n<<",\"steps\":"<<steps<<",\"active\":"<<counts[0]<<",\"deposited\":"<<counts[1]<<",\"escaped\":"<<counts[2]<<",\"coverage_missing\":"<<counts[3]<<",\"invalid\":"<<counts[4]<<",\"zero_yield_no_packet\":"<<counts[5]<<",\"runtime_bytes\":"<<data.bytes<<",\"source_bytes\":"<<data.source_bytes<<"}\n";sycl::free(out,q);return ok?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
