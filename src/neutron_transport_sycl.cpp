#include "carbon/fe_moments.hpp"
#include "carbon/neutron_transport.hpp"
#include "carbon/neutron_package.hpp"
#include "carbon/transport.hpp"
#include "carbon/minibeam_collimator.hpp"
#include "carbon/all_ion_elastic.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/water_electron_response.hpp"
#include "carbon/urban_mcs_package.hpp"
#include "carbon/host_prefetch.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>

namespace carbon {
namespace {
#include "detail/sycl_device_math.inc"
#include "detail/sycl_urban_global.inc"
struct V {float x,y,z;};
V advance(V p,V d,float t){return {p.x+d.x*t,p.y+d.y*t,p.z+d.z*t};}
V unit(V p){float s=sycl::rsqrt(p.x*p.x+p.y*p.y+p.z*p.z);return {p.x*s,p.y*s,p.z*s};}
V rotate(V p,V axis,float c,float s){
 p={c*p.x-s*p.y,s*p.x+c*p.y,p.z};
 V u=sycl::fabs(axis.z)<.9f?unit(V{-axis.y,axis.x,0}):unit(V{axis.z,0,-axis.x});
 V v={axis.y*u.z-axis.z*u.y,axis.z*u.x-axis.x*u.z,axis.x*u.y-axis.y*u.x};
 return unit(V{u.x*p.x+v.x*p.y+axis.x*p.z,u.y*p.x+v.y*p.y+axis.y*p.z,u.z*p.x+v.z*p.y+axis.z*p.z});
}
float uniform(std::uint64_t& state){state^=state>>12;state^=state<<25;state^=state>>27;return (float((state*2685821657736338717ULL)>>41)+.5f)/8388608.f;}
struct Geometry {
 float water,depth,world,cu0,cu1,radius,ca,sa,width,pitch,half_length,offset;
 int slits;bool exact_planes;
 int material(V p)const{
  if(sycl::fabs(p.x)>=world||sycl::fabs(p.y)>=world||sycl::fabs(p.z)>=world)return -1;
  if(p.z>=0&&p.z<depth&&sycl::fabs(p.x)<water&&sycl::fabs(p.y)<water)return 0;
  if(p.z>=cu0&&p.z<cu1&&minibeam_point_in_copper(p.x,p.y,ca,sa,radius,slits,width,pitch,half_length,offset))return 1;
  return 2;
 }
 float boundary(V p,V d)const{
  float best=1.e20f;
  auto plane=[&](float pos,float dir,float face){if(dir!=0){float t=(face-pos)/dir;if(t>(exact_planes?0.f:1.e-6f)&&t<best)best=t;}};
  for(float a:{-world,world,-water,water}){plane(p.x,d.x,a);plane(p.y,d.y,a);}
  for(float a:{-world,world,0.f,depth,cu0,cu1})plane(p.z,d.z,a);
  float aa=d.x*d.x+d.y*d.y,bb=p.x*d.x+p.y*d.y,cc=p.x*p.x+p.y*p.y-radius*radius;
  if(aa>0){float disc=bb*bb-aa*cc;if(disc>=0){float r=sycl::sqrt(disc);float a=(-bb-r)/aa,b=(-bb+r)/aa;if(a>1.e-6f)best=sycl::fmin(best,a);if(b>1.e-6f)best=sycl::fmin(best,b);}}
  float u=ca*p.x+sa*p.y-offset,v=-sa*p.x+ca*p.y,du=ca*d.x+sa*d.y,dv=-sa*d.x+ca*d.y;
  for(int i=-slits/2;i<=slits/2;++i){plane(u,du,i*pitch-width*.5f);plane(u,du,i*pitch+width*.5f);}
  plane(v,dv,-half_length);plane(v,dv,half_length);return best;
 }
 V cross_boundary(V p,V d,float distance)const{
  V q=advance(p,d,distance);
  if(exact_planes){
   // Reuse the exact plane intersection expression that selected the step.
   // p + d*t can retain cancellation residue (especially at z=0), so a
   // nextafter alone does not guarantee that the selected face was crossed.
   auto snap=[&](float pos,float dir,float face,float& end){
    if(dir!=0&&(face-pos)/dir==distance)end=face;
   };
   for(float a:{-world,world,-water,water}){snap(p.x,d.x,a,q.x);snap(p.y,d.y,a,q.y);}
   for(float a:{-world,world,0.f,depth,cu0,cu1})snap(p.z,d.z,a,q.z);
  }
  return across(q,d);
 }
 V across(V p,V d)const{
  // Keep a zero-face nudge normal: subnormal nextafter values may be
  // flushed by GPU arithmetic and fail the following material comparison.
  if(exact_planes){
   constexpr float tiny=std::numeric_limits<float>::min();
   if(p.x==0&&d.x!=0)p.x=sycl::copysign(tiny,d.x);
   if(p.y==0&&d.y!=0)p.y=sycl::copysign(tiny,d.y);
   if(p.z==0&&d.z!=0)p.z=sycl::copysign(tiny,d.z);
  }
  // One representable step across a face, including backward crossings.
  if(d.x!=0)p.x=sycl::nextafter(p.x,d.x>0?INFINITY:-INFINITY);
  if(d.y!=0)p.y=sycl::nextafter(p.y,d.y>0?INFINITY:-INFINITY);
  if(d.z!=0)p.z=sycl::nextafter(p.z,d.z>0?INFINITY:-INFINITY);
  return p;
 }
};
struct Span {unsigned start{},count{};};
struct StopRecord {int material,z,a;unsigned start,count;};
struct StopPoint {float energy,value;};
struct StopSpecies {int z,a;unsigned head[3];};
struct RootLedger {float energy[NEnergyCount]{};std::uint32_t counts[NCountSize]{};};
struct Track {V p,d;float e;};
struct Allocation {
 sycl::queue& q;std::vector<void*> ptrs;
 ~Allocation(){for(auto* p:ptrs)sycl::free(p,q);}
 template<class T>T* alloc(std::size_t n){auto* p=sycl::malloc_device<T>(std::max(std::size_t{1},n),q);if(!p)throw std::bad_alloc();ptrs.push_back(p);return p;}
 template<class T>T* upload(const std::vector<T>& v){auto* p=alloc<T>(v.size());if(!v.empty())q.copy(v.data(),p,v.size()).wait_and_throw();return p;}
};
void read_stopping(const std::filesystem::path& path,int material,std::vector<StopRecord>& records,std::vector<StopPoint>& points,bool proton=false){
 std::ifstream in(path);if(!in)throw std::runtime_error("Missing neutron recoil stopping table: "+path.string());
 std::map<std::pair<int,int>,std::vector<StopPoint>> rows;std::string line;
 while(std::getline(in,line)){
  if(line.empty()||line[0]=='#'||!std::isdigit(static_cast<unsigned char>(line[0])))continue;
  std::replace(line.begin(),line.end(),',',' ');std::istringstream stream(line);int z=1,a=1;float e,s,restricted=0,nuclear=0;
  if(!proton)stream>>z>>a;
  if(!(stream>>e>>s))throw std::runtime_error("Bad neutron recoil stopping row");
  if(!proton)stream>>restricted>>nuclear;
  if(!(e>0&&s>=0&&nuclear>=0&&std::isfinite(e+s+nuclear)))throw std::runtime_error("Invalid neutron recoil stopping value");
  rows[{z,a}].push_back({e,s+nuclear});
 }
 for(auto& [za,v]:rows){
  if(v.size()<2)throw std::runtime_error("Short neutron recoil stopping table");
  for(unsigned i=1;i<v.size();++i)if(v[i].energy<=v[i-1].energy)throw std::runtime_error("Unsorted recoil stopping table");
  records.push_back({material,za.first,za.second,unsigned(points.size()),unsigned(v.size())});points.insert(points.end(),v.begin(),v.end());
 }
}
}

namespace {
// Per-lane audit of queued recoils: each track is tallied in a fresh float
// ledger, as a neutron birth was, then added in double.
struct LaneLedger {double energy[NEnergyCount]{};std::uint64_t counts[NCountSize]{};};
struct ChargedTask {Track t;float weight;std::uint64_t rng_stream;std::uint32_t ordinal;int z,a;};
struct ChargedState {
 Track t;float weight;std::uint64_t rng_stream,urban_seed;int z,a,urban_ion;StopSpecies tables;
 UrbanV2TrackState urban_state;bool urban_boundary;unsigned step;
};
// Recoil transport shared by the inline and queued paths. step_once() is one
// iteration of the original per-track loop (same geometry, stopping, Urban
// counters and deposits); alive()/finish() are its loop test and epilogue.
struct ChargedStepper {
 Geometry geometry;
 const StopSpecies* stop_species;unsigned species_count;
 const unsigned* stop_next;const StopRecord* stops;const StopPoint* stopping;unsigned ns,original_stops;
 UrbanMcsView urban;int urban_water;bool use_water_urban;
 unsigned nx,ny,nz;float sx,sy,sz;std::size_t nv;
 float* depth;float* components;float* dose;std::uint64_t seed;
 void deposit(RootLedger& out,float weight,V pos,int material,float energy,unsigned component)const{
   if(!(energy>0))return;
   energy*=weight;
   if(material!=0){out.energy[NDepositOutside]+=energy;return;}
   out.energy[NDepositWater]+=energy;
   int iz=int(sycl::floor(pos.z/sz));
   if(iz>=0&&iz<int(nz))sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(depth[iz]).fetch_add(energy);
   int ix=int(sycl::floor(pos.x/sx+nx*.5f)),iy=int(sycl::floor(pos.y/sy+ny*.5f));
   if(nv&&ix>=0&&ix<int(nx)&&iy>=0&&iy<int(ny)&&iz>=0&&iz<int(nz)){
    out.energy[NGridDeposit]+=energy;
    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(components[component*nv+(std::size_t(iz)*ny+iy)*nx+ix]).fetch_add(energy);
    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(dose[(std::size_t(iz)*ny+iy)*nx+ix]).fetch_add(energy);
   }
  }
 ChargedState begin(const ChargedTask& task)const{
  ChargedState c{};c.t=task.t;c.weight=task.weight;c.rng_stream=task.rng_stream;c.z=task.z;c.a=task.a;
  c.urban_boundary=true;c.step=0;
  c.urban_ion=use_water_urban?urban.projectile(task.z,task.a):-1;
  c.urban_seed=seed^(0xD2B74407B1CE6E93ULL*task.ordinal);
  const int z=task.z,a=task.a;
  unsigned left=0,right=species_count;
  while(left<right){unsigned mid=left+(right-left)/2;auto key=stop_species[mid];
   if(key.z<z||(key.z==z&&key.a<a))left=mid+1;else right=mid;
  }
  c.tables=StopSpecies{z,a,{ns,ns,ns}};
  if(left<species_count&&stop_species[left].z==z&&stop_species[left].a==a)c.tables=stop_species[left];
  return c;
 }
 static bool alive(const ChargedState& c){return c.step<20000&&c.t.e>0;}
 void finish(const ChargedState& c,RootLedger& out)const{
  if(c.t.e>0){out.energy[NStepLimit]+=c.weight*c.t.e;++out.counts[NLimitCount];}
 }
 // Returns false when the original loop body returned (track finished).
 bool step_once(ChargedState& c,RootLedger& out)const{
  auto& t=c.t;const float weight=c.weight;const int z=c.z,a=c.a;const unsigned step=c.step;
  auto& urban_state=c.urban_state;auto& urban_boundary=c.urban_boundary;
    int mat=geometry.material(t.p);if(mat<0){out.energy[NEscape]+=weight*t.e;return false;}
    if(t.e<1.e-6f){deposit(out,weight,t.p,mat,t.e,z==1&&a==1?0:1);return false;}
    float eu=t.e/a,S=-1;
    for(unsigned j=c.tables.head[mat];j<ns;j=stop_next[j]){auto s=stops[j];
     auto pts=stopping+s.start;if(eu>pts[s.count-1].energy)continue;
     if(j>=original_stops)++out.counts[NRecoilExtensionSteps];
     if(eu<pts[0].energy){S=pts[0].value*sycl::sqrt(eu/pts[0].energy);break;}
     unsigned lo=0,hi=s.count-1;while(hi-lo>1){unsigned mid=(lo+hi)/2;if(eu<pts[mid].energy)hi=mid;else lo=mid;}
     float f=(eu-pts[lo].energy)/(pts[hi].energy-pts[lo].energy);S=pts[lo].value+f*(pts[hi].value-pts[lo].value);break;
    }
    if(!(S>0)&&t.e>0){out.energy[NUnsupported]+=weight*t.e;out.energy[NMissingIon]+=weight*t.e;++out.counts[NUnsupportedCount];return false;}
    float edge=geometry.boundary(t.p,t.d);
    const float spatial_step=mat==2?10.f:sycl::fmin(.05f,.5f*sycl::fmin(sx,sz));
    float h=sycl::fmin(edge,sycl::fmin(spatial_step,.05f*t.e/S));
    if(use_water_urban&&mat==0){
     auto context=urban.at(urban_water,c.urban_ion,t.e);
     if(context.valid()&&std::isfinite(context.lambda(t.e))&&context.lambda(t.e)>0){
      const auto cell=urban_global_water_cell({t.p.x,t.p.y,t.p.z},{t.d.x,t.d.y,t.d.z},
       {-.5f*nx*sx,-.5f*ny*sy,0},{nx*.5f*sx,ny*.5f*sy,geometry.depth},
       {sx,sy,sz},nx,ny,nz,geometry.water);
      auto proposal=urban_global_prepare(urban,urban_water,c.urban_ion,t.e,h,
       {t.p.x,t.p.y,t.p.z},{t.d.x,t.d.y,t.d.z},cell,urban_boundary,urban_state,
       c.urban_seed,c.rng_stream,step);
      auto scatter=urban_global_finalize(proposal,proposal.true_path,c.urban_seed,c.rng_stream,step);
      if(!proposal.valid||!scatter.proposal_valid||scatter.numeric_guard_tripped||!scatter.energy_prediction_valid||
         !std::isfinite(scatter.direction.x+scatter.direction.y+scatter.direction.z)){
       ++out.counts[NWaterUrbanFailures];out.energy[NStepLimit]+=weight*t.e;return false;
      }
      ++out.counts[NWaterUrbanSteps];urban_state=proposal.state;urban_boundary=scatter.boundary_crossed;
      const float g=scatter.final_geom_path_mm;
      float loss=sycl::fmin(t.e,scatter.final_true_path_mm*S);
      deposit(out,weight,advance(t.p,t.d,g*.5f),mat,loss,z==1&&a==1?0:1);t.e-=loss;
      auto end=urban_voxel_snap_endpoint(cell,g,{t.p.x+t.d.x*g+scatter.displacement_mm.x,
       t.p.y+t.d.y*g+scatter.displacement_mm.y,t.p.z+t.d.z*g+scatter.displacement_mm.z});
      t.p={end.x,end.y,end.z};
      // Cross physical water faces using the incident direction; scorer
      // replica faces remain exactly snapped for the outgoing cell lookup.
      if(t.p.z<=0||t.p.z>=geometry.depth||sycl::fabs(t.p.x)>=geometry.water||sycl::fabs(t.p.y)>=geometry.water)
       t.p=geometry.across(t.p,t.d);
      t.d={scatter.direction.x,scatter.direction.y,scatter.direction.z};++c.step;return true;
     }
     ++out.counts[NWaterUrbanUncovered];
    }
    urban_boundary=true;
    float loss=sycl::fmin(t.e,h*S);deposit(out,weight,advance(t.p,t.d,h*.5f),mat,loss,z==1&&a==1?0:1);t.e-=loss;
    t.p=h==edge?geometry.cross_boundary(t.p,t.d,h):advance(t.p,t.d,h);
  ++c.step;return true;
 }
 void run(const ChargedTask& task,RootLedger& out)const{
  auto c=begin(task);
  while(alive(c))if(!step_once(c,out))return;
  finish(c,out);
 }
};
}

namespace {
struct NeutronStopTables {std::vector<StopRecord> records;std::vector<StopPoint> points;unsigned original{};};
// Host recoil stopping chains; identical to the former inline construction.
NeutronStopTables load_neutron_stop_tables(const TransportConfig& config){
 std::vector<StopRecord> stop_records;std::vector<StopPoint> stop_points;
 // Prefer the dedicated low-energy recoil table for water wherever covered.
 if(!config.elastic_recoil_stopping_file.empty()){
  auto t=ElasticRecoilStoppingTable::load(config.elastic_recoil_stopping_file,config.elastic_recoil_stopping_sha256);
  for(unsigned p=0;p<t.keys.size()/2;++p){
   stop_records.push_back({0,t.keys[2*p],t.keys[2*p+1],unsigned(stop_points.size()),unsigned(t.energies.size())});
   for(unsigned i=0;i<t.energies.size();++i)stop_points.push_back({t.energies[i],t.values[(p*26)*t.energies.size()+i]});
  }
 }
 read_stopping(config.particle_stopping_power_file,0,stop_records,stop_points);
 read_stopping(config.minibeam_copper_ion_stopping_power_file,1,stop_records,stop_points);
 read_stopping(config.minibeam_air_stopping_power_file,2,stop_records,stop_points,true);
 const unsigned original_stops=stop_records.size();
 if(!config.neutron_recoil_stopping_manifest.empty()){
  const auto manifest=config.neutron_recoil_stopping_manifest;
  std::ifstream file(manifest);if(!file)throw std::runtime_error("Missing neutron recoil manifest");
  const std::string text((std::istreambuf_iterator<char>(file)),{});
  auto root=minjson::Parser(text).parse();
  if(root.at("schema").str!="maigo.neutron.recoil-stopping.v1")throw std::runtime_error("Invalid neutron recoil schema");
  const auto& materials=root.at("materials").arr;
  const char* names[]={"Water_75eV","G4_Cu","G4_AIR"};
  if(materials.size()!=3)throw std::runtime_error("Incomplete neutron recoil materials");
  for(unsigned m=0;m<3;++m){
   const auto& entry=materials[m];std::filesystem::path relative=entry.at("file").str;
   if(entry.at("material").str!=names[m]||relative.empty()||relative.is_absolute())throw std::runtime_error("Invalid neutron recoil material/file");
   for(const auto& part:relative)if(part=="..")throw std::runtime_error("Neutron recoil path traversal");
   auto path=manifest.parent_path()/relative;
   if(compute_file_sha256_hex(path)!=entry.at("sha256").str)throw std::runtime_error("Neutron recoil SHA256 mismatch");
   if(m==0&&entry.at("excitation_eV").number!=75)throw std::runtime_error("Neutron recoil water must use 75 eV");
   read_stopping(path,m,stop_records,stop_points);
  }
 }
 return {std::move(stop_records),std::move(stop_points),original_stops};
}
std::string neutron_stop_key(const TransportConfig& config){
 return config.elastic_recoil_stopping_file.string()+"|"+config.elastic_recoil_stopping_sha256+"|"+
  config.particle_stopping_power_file.string()+"|"+config.minibeam_copper_ion_stopping_power_file.string()+"|"+
  config.minibeam_air_stopping_power_file.string()+"|"+config.neutron_recoil_stopping_manifest.string();
}
HostPrefetch<NeutronPackage>& neutron_package_prefetch(){static HostPrefetch<NeutronPackage> registry;return registry;}
HostPrefetch<NeutronStopTables>& neutron_stop_prefetch(){static HostPrefetch<NeutronStopTables> registry;return registry;}
}

// Parse the neutron bank and recoil stopping tables while primaries run.
void prefetch_neutron_inputs(const TransportConfig& config){
 const auto package_file=config.neutron_package_file;
 neutron_package_prefetch().start(package_file.string(),[package_file]{return NeutronPackage::load(package_file,false);});
 neutron_stop_prefetch().start(neutron_stop_key(config),[config]{return load_neutron_stop_tables(config);});
}

void transport_neutrons_experimental(sycl::queue& queue,const TransportConfig& config,TransportResult& result){
 std::cout<<"[neutron-experimental] draft rates/events; correlated energy rescaling; CSDA recoils, optional water Urban; photons/electrons unresolved\n";
 const auto package=neutron_package_prefetch().take(config.neutron_package_file.string(),[&]{return NeutronPackage::load(config.neutron_package_file,false);});
 const auto nb=result.neutron_birth_records.size();if(!nb)return;
 Allocation memory{queue,{}};
 auto rates=memory.upload(package.macroscopic_rates);auto events=memory.upload(package.events);auto products=memory.upload(package.products);
 auto births=memory.upload(result.neutron_birth_records);
 const bool capture_protons=!config.neutron_proton_birth_output_file.empty();
 const std::size_t proton_capacity=capture_protons?config.neutron_proton_birth_capacity:0;
 auto proton_births=memory.alloc<NeutronBirthRecord>(proton_capacity);
 auto proton_count=memory.alloc<std::uint64_t>(1);queue.fill(proton_count,std::uint64_t{0},1).wait_and_throw();
 std::array<Span,12> spans{};std::vector<NeutronRateRecord> ordered;
 for(unsigned m=0;m<3;++m)for(unsigned p=0;p<4;++p){auto& span=spans[m*4+p];span.start=ordered.size();for(auto r:package.rates)if(r.material==m&&r.subtype==111+10*p)ordered.push_back(r);span.count=ordered.size()-span.start;}
 // Event indices continue referring to the original bank; ordered records only select events.
 auto nodes=memory.upload(ordered);std::array<Span,12> fine{};
 for(unsigned i=0;i<package.macroscopic_rates.size();++i){auto r=package.macroscopic_rates[i];auto& s=fine[r.material*4+(r.subtype-111)/10];if(!s.count)s.start=i;++s.count;}
 auto stop_tables=neutron_stop_prefetch().take(neutron_stop_key(config),[&]{return load_neutron_stop_tables(config);});
 auto& stop_records=stop_tables.records;auto& stop_points=stop_tables.points;
 const unsigned original_stops=stop_tables.original;
 // Index candidate chains in original order: older tables retain priority,
 // including fallback when their upper energy limit is exceeded.
 const unsigned ns=stop_records.size();
 std::map<std::pair<int,int>,std::array<unsigned,3>> heads;
 std::vector<unsigned> next_stop(ns,ns);
 for(unsigned j=ns;j-- > 0;){
  const auto& s=stop_records[j];
  auto entry=heads.emplace(std::make_pair(s.z,s.a),std::array<unsigned,3>{ns,ns,ns}).first;
  next_stop[j]=entry->second[s.material];entry->second[s.material]=j;
 }
 std::vector<StopSpecies> species;
 for(const auto& [za,h]:heads)species.push_back({za.first,za.second,{h[0],h[1],h[2]}});
 auto stop_species=memory.upload(species);const unsigned species_count=species.size();
 auto stop_next=memory.upload(next_stop);
 auto stops=memory.upload(stop_records);auto stopping=memory.upload(stop_points);
 UrbanMcsView urban{};int urban_water=-1;
 const bool use_water_urban=config.neutron_water_recoil_urban;
 if(use_water_urban){
  auto package=UrbanMcsPackage::load(config.urban_mcs_package_file,config.urban_mcs_package_sha256);
  urban_water=package.view().material(-1,float(config.water_density_g_per_cm3));
  if(urban_water<0)throw std::runtime_error("Neutron recoil Urban package lacks water");
  urban={memory.upload(package.materials),memory.upload(package.species),memory.upload(package.records),
   memory.upload(package.nodes),memory.upload(package.mfps),memory.upload(package.segments),
   unsigned(package.materials.size()),unsigned(package.species.size())};
 }
 const unsigned nx=config.voxel_bins_x,ny=config.voxel_bins_y,nz=result.deposited_energy_MeV.size();
 const float sx=config.voxel_size_x_mm,sy=config.voxel_size_y_mm,sz=config.scorer_spacing_z_mm();
 const std::size_t nv=config.enable_voxel_scoring?std::size_t(nx)*ny*nz:0;
 auto components=memory.alloc<float>(3*nv);
 queue.fill(components,0.f,3*nv).wait_and_throw();
 auto dose=memory.alloc<float>(nv);auto depth=memory.alloc<float>(nz);auto ledger=memory.alloc<RootLedger>(nb);
 queue.fill(dose,0.f,nv);queue.fill(depth,0.f,nz).wait_and_throw();
 float centre=-(config.minibeam_water_entrance_world_y_mm+config.minibeam_collimator_center_to_isocenter_mm);
 float angle=config.minibeam_collimator_angle_deg*0.017453292519943295;
 Geometry geometry{float(config.urban_water_half_width_mm),float(config.phantom_length_mm),float(config.neutron_world_half_width_mm),centre-float(config.minibeam_collimator_thickness_mm)*.5f,centre+float(config.minibeam_collimator_thickness_mm)*.5f,float(config.minibeam_radius_mm),std::cos(angle),std::sin(angle),float(config.minibeam_slit_width_mm),float(config.minibeam_slit_pitch_mm),float(config.minibeam_slit_length_mm)*.5f,float(config.minibeam_slit_offset_mm),config.minibeam_slit_count,config.neutron_exact_plane_boundaries};
 const auto seed=config.random_seed;const unsigned max_collisions=config.neutron_max_collisions;
 const bool exact_elastic=config.neutron_elastic_two_body;
 const ChargedStepper charged_stepper{geometry,stop_species,species_count,stop_next,stops,stopping,ns,original_stops,
  urban,urban_water,use_water_urban,nx,ny,nz,sx,sy,sz,nv,depth,components,dose,seed};
 // Recoil tracks are transported by a second kernel whose lanes pull tracks
 // from a queue, so one long recoil chain no longer idles its whole warp.
 // Per-track arithmetic is unchanged; CARBON_NEUTRON_CHARGED_QUEUE=0 keeps
 // every recoil inline in its neutron lane.
 const bool queue_charged=[]{
  const char* value=std::getenv("CARBON_NEUTRON_CHARGED_QUEUE");
  if(value&&std::string_view(value)!="0"&&std::string_view(value)!="1")
   throw std::invalid_argument("CARBON_NEUTRON_CHARGED_QUEUE must be 0 or 1");
  return !(value&&std::string_view(value)=="0");
 }();
 const std::size_t charged_capacity=queue_charged?std::max<std::size_t>(std::size_t{1}<<20,8*nb):0;
 auto charged_tasks=memory.alloc<ChargedTask>(charged_capacity);
 auto charged_count=memory.alloc<std::uint64_t>(1);queue.fill(charged_count,std::uint64_t{0},1).wait_and_throw();
 const std::size_t compute_units=std::max<std::size_t>(1,queue.get_device().get_info<sycl::info::device::max_compute_units>());
 auto next_birth=memory.alloc<std::uint64_t>(1);queue.fill(next_birth,std::uint64_t{0},1).wait_and_throw();
 queue.parallel_for(sycl::range<1>(std::min<std::size_t>(nb,compute_units*2048)),[=](sycl::id<1>){
  // Persistent lane: births are pulled one at a time; each birth keeps its
  // own RNG stream, neutron stack order and ledger slot.
  RootLedger out{};NeutronBirthRecord b{};float weight=0;std::uint64_t random=1;
  std::uint64_t birth=0;bool have_birth=false,neutron_active=false;
  Track stack[64];unsigned count=0;Track t{};float tau=0;unsigned flights=0;
  int rate_mat=-1;float rate_energy=-1.f,rate_total=0,rate_rs[4]{};unsigned rate_unresolved=0;bool rate_covered=false;
  const auto start_birth=[&]{
   out=RootLedger{};b=births[birth];weight=b.weight;
   random=b.rng_stream^(seed+0x9e3779b97f4a7c15ULL);if(!random)random=1;
   out.counts[NBirths]=1;out.energy[NInput]=b.kinetic_energy_MeV*weight;
   out.energy[b.material_section==-1?NInputWater:NInputCopper]=out.energy[NInput];
  };
  auto deposit=[&](V pos,int material,float energy,unsigned component){
   charged_stepper.deposit(out,weight,pos,material,energy,component);
  };
  auto charged=[&](Track t,int z,int a){
   ++out.counts[NCharged];
   if(capture_protons&&z==1&&a==1&&t.e>0){
    auto slot=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
     sycl::access::address_space::global_space>(*proton_count).fetch_add(1);
    if(slot<proton_capacity){
     NeutronBirthRecord r{};r.history=b.history;r.rng_stream=b.rng_stream;
     r.collision=out.counts[NCollisions];r.product_index=out.counts[NCharged];
     r.material_section=-1-geometry.material(t.p);r.parent_z=0;r.parent_a=1;
     r.kinetic_energy_MeV=t.e;r.x_mm=t.p.x;r.y_mm=t.p.y;r.z_mm=t.p.z;
     r.direction_x=t.d.x;r.direction_y=t.d.y;r.direction_z=t.d.z;r.weight=weight;
     proton_births[slot]=r;
    }
   }
   const ChargedTask task{t,weight,b.rng_stream,out.counts[NCharged],z,a};
   if(charged_capacity){
    const auto slot=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
     sycl::access::address_space::global_space>(*charged_count).fetch_add(1);
    if(slot<charged_capacity){charged_tasks[slot]=task;return;}
   }
   charged_stepper.run(task,out);
  };
  // One call is one iteration of the original per-neutron flight loop:
  // false where that loop broke (neutron finished), true where it continued.
  auto fly=[&]()->bool{
    if(out.counts[NCollisions]>=max_collisions||++flights>max_collisions*8){out.energy[NStepLimit]+=weight*t.e;++out.counts[NLimitCount];return false;}
    int mat=geometry.material(t.p);if(mat<0){out.energy[NEscape]+=weight*t.e;return false;}
    // Rates depend only on (material, energy); a boundary crossing keeps the
    // energy, so repeated flights reuse the same lookups and tallies.
    if(mat!=rate_mat||t.e!=rate_energy){
     rate_mat=mat;rate_energy=t.e;rate_total=0;rate_covered=true;rate_unresolved=0;
     for(unsigned p=0;p<4;++p){auto s=fine[mat*4+p];auto grid=rates+s.start;
      if(t.e<grid[0].energy||t.e>grid[s.count-1].energy){rate_covered=false;break;}
      unsigned lo=0,hi=s.count-1;while(hi-lo>1){unsigned mid=(lo+hi)/2;if(t.e<grid[mid].energy)hi=mid;else lo=mid;}
      float f=sycl::log(t.e/grid[lo].energy)/sycl::log(grid[hi].energy/grid[lo].energy);
      rate_rs[p]=sycl::fmax(0.f,grid[lo].rate+f*(grid[hi].rate-grid[lo].rate));rate_total+=rate_rs[p];
      if(grid[lo].interval_status>=4)++rate_unresolved;
     }
    }
    out.counts[NUnresolvedIntervals]+=rate_unresolved;
    const float* rs=rate_rs;const float total=rate_total;
    if(!rate_covered){out.energy[NBelowGrid]+=weight*t.e;return false;}
    float edge=geometry.boundary(t.p,t.d),flight=total>0?tau/total:INFINITY;
    if(flight>=edge){
     const auto unsnapped=geometry.across(advance(t.p,t.d,edge),t.d);
     const auto snapped=geometry.cross_boundary(t.p,t.d,edge);
     if(geometry.material(unsnapped)!=geometry.material(snapped))++out.counts[NBoundaryMaterialCorrections];
     t.p=snapped;tau=sycl::fmax(0.f,tau-total*edge);return true;
    }
    t.p=advance(t.p,t.d,flight);unsigned process=0;float pick=uniform(random)*total;
    for(;process<3&&pick>=rs[process];++process)pick-=rs[process];
    ++out.counts[NCollisions];++out.counts[NElastic+process];
    auto span=spans[mat*4+process];auto g=nodes+span.start;
    if(t.e<g[0].energy||t.e>g[span.count-1].energy){out.energy[NBelowGrid]+=weight*t.e;return false;}
    unsigned lo=0,hi=span.count-1;while(hi-lo>1){unsigned mid=(lo+hi)/2;if(t.e<g[mid].energy)hi=mid;else lo=mid;}
    float f=sycl::log(t.e/g[lo].energy)/sycl::log(g[hi].energy/g[lo].energy);
    unsigned selected=uniform(random)<f?hi:lo;
    if(!g[selected].event_count)selected=selected==lo?hi:lo;
    auto node=g[selected];if(!node.event_count){out.energy[NUnsupported]+=weight*t.e;++out.counts[NUnsupportedCount];return false;}
    unsigned ei=node.event_offset+sycl::min(unsigned(uniform(random)*node.event_count),node.event_count-1);
    auto event=events[ei];float sum=node.energy+event.kinetic_gain,target=t.e+event.kinetic_gain;
    // At >=100 keV, use the sampled recoil transfer to recover the native
    // CM angle for an exclusive stationary-target elastic event. Rebuild both
    // four-momenta at the actual energy. Thermal/multi-product events retain
    // the native-bank branch; their target motion is not discarded.
    if(exact_elastic&&process==0&&t.e>=.1f&&node.energy>=.1f&&event.product_count==1&&
       event.continuation>0&&event.deposit==0&&sycl::fabs(event.kinetic_gain)<1.e-5f*node.energy){
     const auto recoil=products[event.product_offset];
     if(recoil.z==event.target_z&&recoil.a==event.target_a&&recoil.mass>0){
      constexpr double mn=939.56536; // pinned Geant4 11.3.2 CLHEP mass
      const double E=node.energy,M=recoil.mass;
      const double pcm2=E*(E+2*mn)*M*M/((mn+M)*(mn+M)+2*M*E);
      const double cosine=1-double(recoil.energy)*M/pcm2;
      if(cosine>=-1.00001&&cosine<=1.00001){
       const auto elastic=elastic_two_body(t.e,t.d.x,t.d.y,t.d.z,std::max(-1.,std::min(1.,cosine)),uniform(random),mn,M);
       ++out.counts[NTwoBodyElastic];
       charged(Track{t.p,{elastic.recoil_dir_x,elastic.recoil_dir_y,elastic.recoil_dir_z},elastic.recoil_ke_MeV},recoil.z,recoil.a);
       t.e=elastic.projectile_ke_MeV;t.d={elastic.proj_dir_x,elastic.proj_dir_y,elastic.proj_dir_z};
       tau=-sycl::log(uniform(random));return true;
      }
     }
    }
    if(!(sum>0&&target>=0)){out.energy[NUnsupported]+=weight*t.e;++out.counts[NUnsupportedCount];return false;}
    float scale=target/sum;if(node.energy!=t.e)++out.counts[NScaledEvents];out.energy[NKineticGain]+=weight*event.kinetic_gain;
    float phi=6.28318530718f*uniform(random),c=sycl::cos(phi),s=sycl::sin(phi);
    deposit(t.p,mat,event.deposit*scale,2);
    for(unsigned j=0;j<event.product_count;++j){auto p=products[event.product_offset+j];float energy=p.energy*scale;if(!(energy>0))continue;
     Track child{t.p,rotate(V{p.dx,p.dy,p.dz},t.d,c,s),energy};
     if(p.pdg==2112){if(count<64)stack[count++]=child;else{out.energy[NQueueOverflow]+=weight*energy;++out.counts[NOverflowCount];}}
     else if(p.z>0&&p.a>=p.z)charged(child,p.z,p.a);
     else{out.energy[NUnsupported]+=weight*energy;++out.counts[NUnsupportedCount];if(p.pdg==22)out.energy[NPhoton]+=weight*energy;if(p.pdg==11||p.pdg==-11)out.energy[NElectron]+=weight*energy;}
    }
    t.e=event.continuation*scale;if(t.e>0)t.d=rotate(V{event.dx,event.dy,event.dz},t.d,c,s);
    tau=-sycl::log(uniform(random));
    return true;
  };
  for(;;){
   if(!neutron_active){
    if(count==0){
     if(have_birth)ledger[birth]=out;
     birth=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
      sycl::access::address_space::global_space>(*next_birth).fetch_add(1);
     if(birth>=nb)break;
     have_birth=true;start_birth();
     stack[0]={{b.x_mm,b.y_mm,b.z_mm},unit(V{b.direction_x,b.direction_y,b.direction_z}),b.kinetic_energy_MeV};count=1;
    }
    t=stack[--count];tau=-sycl::log(uniform(random));
    flights=0;neutron_active=true;
   }
   if(!(t.e>0)){neutron_active=false;continue;}
   if(!fly())neutron_active=false;
  }
 }).wait_and_throw();
 std::vector<LaneLedger> lane_ledgers;
 if(charged_capacity){
  std::uint64_t requested=0;queue.copy(charged_count,&requested,1).wait_and_throw();
  const std::uint64_t tasks=std::min<std::uint64_t>(requested,charged_capacity);
  std::cout<<"[neutron-charged-queue] queued="<<tasks<<" inline_overflow="<<requested-tasks<<" capacity="<<charged_capacity<<'\n';
  if(tasks){
   const std::size_t lanes=std::min<std::size_t>(tasks,compute_units*2048);
   auto lane_ledger=memory.alloc<LaneLedger>(lanes);
   auto next=memory.alloc<std::uint64_t>(1);queue.fill(next,std::uint64_t{0},1).wait_and_throw();
   queue.parallel_for(sycl::range<1>(lanes),[=](sycl::id<1> id){
    LaneLedger total{};RootLedger out{};ChargedState c{};bool active=false;
    const auto close_track=[&]{
     for(unsigned k=0;k<NEnergyCount;++k)total.energy[k]+=out.energy[k];
     for(unsigned k=0;k<NCountSize;++k)total.counts[k]+=out.counts[k];
     out=RootLedger{};active=false;
    };
    // Flat loop: a lane that finishes a track takes the next one while the
    // other lanes of its warp keep stepping theirs.
    for(;;){
     if(!active){
      const auto i=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
       sycl::access::address_space::global_space>(*next).fetch_add(1);
      if(i>=tasks)break;
      c=charged_stepper.begin(charged_tasks[i]);active=true;
     }
     if(!ChargedStepper::alive(c)){charged_stepper.finish(c,out);close_track();continue;}
     if(!charged_stepper.step_once(c,out))close_track();
    }
    lane_ledger[id[0]]=total;
   }).wait_and_throw();
   lane_ledgers.resize(lanes);queue.copy(lane_ledger,lane_ledgers.data(),lanes).wait_and_throw();
  }
 }
 if(capture_protons){
  std::uint64_t n=0;queue.copy(proton_count,&n,1).wait_and_throw();
  if(n>proton_capacity)throw std::runtime_error("Neutron proton birth capture overflow");
  auto& records=result.neutron_proton_birth_records;records.resize(n);
  if(n)queue.copy(proton_births,records.data(),n).wait_and_throw();
  std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){
   if(a.history!=b.history)return a.history<b.history;
   if(a.rng_stream!=b.rng_stream)return a.rng_stream<b.rng_stream;
   return a.product_index<b.product_index;
  });
  std::cout<<"[neutron-proton-births] captured="<<n<<'\n';
 }
 std::vector<RootLedger> ledgers(nb);std::vector<float> dh(nz),vh(nv);
 queue.copy(ledger,ledgers.data(),nb);queue.copy(depth,dh.data(),nz);if(nv)queue.copy(dose,vh.data(),nv);queue.wait_and_throw();
 for(unsigned c=0;c<3;++c){
  auto& values=result.neutron_component_voxel_MeV[c];values.resize(nv);
  if(nv)queue.copy(components+c*nv,values.data(),nv).wait_and_throw();
 }
 auto& audit=result.neutron_experimental;
 for(const auto& l:ledgers){for(unsigned i=0;i<NEnergyCount;++i)audit.energy[i]+=l.energy[i];for(unsigned i=0;i<NCountSize;++i)audit.counts[i]+=l.counts[i];}
 for(const auto& l:lane_ledgers){for(unsigned i=0;i<NEnergyCount;++i)audit.energy[i]+=l.energy[i];for(unsigned i=0;i<NCountSize;++i)audit.counts[i]+=l.counts[i];}
 if(audit.counts[NWaterUrbanFailures])throw std::runtime_error("Neutron water recoil Urban numerical failure");
 result.neutron_origin_deposited_energy_MeV.assign(dh.begin(),dh.end());
 result.neutron_voxel_deposited_energy_MeV.assign(vh.begin(),vh.end());
 for(unsigned i=0;i<nz;++i){result.deposited_energy_MeV[i]+=dh[i];}
 for(std::size_t i=0;i<nv;++i)result.voxel_deposited_energy_MeV[i]+=vh[i];
 if(result.in_fov_deposited_energy_MeV.size()==nz&&nv)
  for(unsigned z=0;z<nz;++z)for(std::size_t i=std::size_t(z)*nx*ny;i<std::size_t(z+1)*nx*ny;++i)result.in_fov_deposited_energy_MeV[z]+=vh[i];
 auto& e=audit.energy;
 result.total_deposited_energy_MeV+=e[NDepositWater];
 result.in_grid_deposited_energy_MeV+=e[NGridDeposit];result.outside_grid_deposited_energy_MeV+=e[NDepositWater]-e[NGridDeposit];
 result.beamline_removed_energy_MeV+=e[NDepositOutside]-e[NInputCopper];
 result.untracked_nuclear_energy_MeV+=e[NUnsupported]+e[NBelowGrid]+e[NStepLimit]+e[NQueueOverflow]-e[NInputWater];
 result.escaped_energy_MeV+=e[NEscape];
 result.energy_ledger.E_continuous_ionizing=result.total_deposited_energy_MeV;
 std::cout<<"[neutron-experimental] births="<<nb<<" collisions="<<audit.counts[NCollisions]<<" water_MeV="<<e[NDepositWater]<<" unsupported_MeV="<<e[NUnsupported]<<" limit_MeV="<<e[NStepLimit]<<" Q_MeV="<<e[NKineticGain]<<'\n';
}
}
