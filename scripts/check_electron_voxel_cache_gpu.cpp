#include "carbon/electron_fp32_device.hpp"
#include "carbon/electron_voxel_cache.hpp"
#include <iostream>
struct Pair {unsigned low,high;float density;};
struct Record {std::uint64_t steps,hits,material,refresh;unsigned mismatch,invalid,cap;};
int main(int argc,char** argv){try{
 if(argc!=9)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha");
 namespace f=carbon::electron_fp32;namespace c=carbon::electron_fp32::voxel_cache_research;
 sycl::queue queue{sycl::gpu_selector_v};carbon::IndependentElectronFloatDevice data(queue,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);const auto bank=data.charged;
 std::vector<Pair> pairs;for(unsigned i=0;i<data.charged_nodes.size();++i)pairs.push_back({i,i,data.charged_nodes[i].density});
 for(unsigned sec=0;sec<25;++sec){std::vector<unsigned> ids;for(unsigned i=0;i<data.charged_nodes.size();++i)if(data.charged_nodes[i].section==int(sec))ids.push_back(i);
  std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return data.charged_nodes[a].density<data.charged_nodes[b].density;});
  for(unsigned i=1;i<ids.size();++i)pairs.push_back({ids[i-1],ids[i],.5f*(data.charged_nodes[ids[i-1]].density+data.charged_nodes[ids[i]].density)});
 }
 const unsigned np=pairs.size(),count=np*2048;auto* spec=sycl::malloc_shared<Pair>(np,queue);std::copy(pairs.begin(),pairs.end(),spec);
 auto* records=sycl::malloc_shared<Record>(count,queue);
 queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
  const unsigned i=id[0];const auto pair=spec[i%np];
  std::uint64_t ra=0x9725634abcdefULL+i;ra=(ra^(ra>>30))*0xbf58476d1ce4e5b9ULL;ra=(ra^(ra>>27))*0x94d049bb133111ebULL;ra^=ra>>31;
  auto ua=[&](){return f::random_draw(ra);};float rho[27];std::uint8_t sec[27];
  for(unsigned j=0;j<27;++j){const auto node=spec[(i%np+(j==13?0:(j%3)*17))%np];rho[j]=node.density;sec[j]=bank.nodes[node.low].section;}
  const float spacing=(i/np/4)%2?.5f:.02f;f::ElectronCtGeometry g{{3,3,3},{-169.1015f,-2.03f,300.f},{spacing,spacing,spacing},rho,sec};
  f::ElectronEnergyPacket a;a.status=carbon::ElectronPacketStatus::active;a.cursor.valid=true;a.weight_MeV=1;
  for(unsigned k=0;k<3;++k)a.cursor.position[k]=g.origin[k]+spacing*(1+float(ua()));
  const unsigned location=(i/np)%4;const float face=g.origin[0]+spacing;
  if(location==1)a.cursor.position[0]=face;
  if(location==2)a.cursor.position[0]=std::nextafter(face,-INFINITY);
  if(location==3)a.cursor.position[0]=std::nextafter(face,INFINITY);
  auto d=f::joint_math::unit({2*float(ua())-1,2*float(ua())-1,2*float(ua())-1});a.cursor.direction={d.x,d.y,d.z};
  a.cursor.energy_MeV=std::exp(std::log(1e-6f)+float(ua())*std::log(.65f/1e-6f));a.cursor.section=bank.nodes[pair.low].section;a.cursor.physical_density_g_cm3=pair.density;a.cursor.density_g_cm3=pair.density;
  auto b=a;std::uint64_t rb=ra;auto ub=[&](){return f::random_draw(rb);};c::Cache cache;Record result{};
  auto equal=[](float x,float y){return std::bit_cast<std::uint32_t>(x)==std::bit_cast<std::uint32_t>(y);};
  for(unsigned step=0;step<200000;++step){
   auto probe=cache;const auto exact=f::next_charged_face(g,a.cursor.position,a.cursor.direction);
   const auto proposed=c::cached_face(g,b.cursor.position,b.cursor.direction,probe);
   if(probe.face_hits!=cache.face_hits&&(!exact.valid||exact.escaped||exact.voxel!=proposed.voxel||exact.section!=proposed.section||!equal(exact.density,proposed.density)||!(exact.distance>.05f))){result.mismatch=1;break;}
   std::uint64_t ca[8]{},cb[8]{};auto ta=[&](int k,std::uint64_t n){ca[k]+=n;};auto tb=[&](int k,std::uint64_t n){cb[k]+=n;};
   a=f::transport_independent_charged_packet(a,bank,g,ua,ta);b=c::transport_packet(b,bank,g,cache,ub,tb);++result.steps;
   bool ok=ra==rb&&a.status==b.status&&a.cursor.pdg==b.cursor.pdg&&a.cursor.section==b.cursor.section&&a.advances==b.advances&&a.photon_children==b.photon_children&&a.cursor.row==b.cursor.row&&a.cursor.valid==b.cursor.valid;
   ok=ok&&equal(a.cursor.energy_MeV,b.cursor.energy_MeV)&&equal(a.cursor.density_g_cm3,b.cursor.density_g_cm3)&&equal(a.cursor.physical_density_g_cm3,b.cursor.physical_density_g_cm3);
   for(unsigned k=0;k<3;++k)ok=ok&&equal(a.cursor.position[k],b.cursor.position[k])&&equal(a.cursor.direction[k],b.cursor.direction[k])&&equal(a.deposit_position[k],b.deposit_position[k]);
   for(unsigned k=0;k<8;++k)ok=ok&&ca[k]==cb[k];
   if(!ok){result.mismatch=2;break;}
   if(a.status!=carbon::ElectronPacketStatus::active||a.cursor.pdg==22)break;
   // Controlled relocations stand in for joint handoffs, exercising stale
   // voxel-cache rejection without changing either arm's random sequence.
   if(step%32==31&&(i/np)%3==0){
    const unsigned cell=(step/32)%3;for(unsigned k=0;k<3;++k)a.cursor.position[k]=b.cursor.position[k]=g.origin[k]+spacing*(cell+.5f);
   }
  }
  result.hits=cache.face_hits;result.material=cache.material_hits;result.refresh=cache.refreshes;
  result.invalid=a.status==carbon::ElectronPacketStatus::invalid||a.status==carbon::ElectronPacketStatus::coverage_missing;
  result.cap=result.steps==200000;records[i]=result;
 }).wait_and_throw();
 std::uint64_t steps=0,hits=0,material=0,refresh=0,mismatch=0,invalid=0,cap=0;
 for(unsigned i=0;i<count;++i){const auto r=records[i];steps+=r.steps;hits+=r.hits;material+=r.material;refresh+=r.refresh;mismatch+=r.mismatch!=0;invalid+=r.invalid;cap+=r.cap;if(r.mismatch&&mismatch<=5)std::cerr<<"mismatch="<<i<<" kind="<<r.mismatch<<" step="<<r.steps<<'\n';}
 bool ok=!mismatch&&!invalid&&!cap&&hits&&material;
 std::cout<<"{\"passed\":"<<(ok?"true":"false")<<",\"density_cases\":"<<np<<",\"packets\":"<<count<<",\"steps\":"<<steps<<",\"face_hits\":"<<hits<<",\"material_hits\":"<<material<<",\"refreshes\":"<<refresh<<",\"packet_rng_audit_mismatches\":"<<mismatch<<",\"invalid\":"<<invalid<<",\"caps\":"<<cap<<"}\n";
 sycl::free(records,queue);sycl::free(spec,queue);return ok?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
