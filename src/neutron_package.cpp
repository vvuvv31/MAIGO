#include "carbon/neutron_package.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
namespace carbon {
namespace {
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
bool direction(float x,float y,float z){return std::isfinite(x+y+z)&&std::abs(x*x+y*y+z*z-1.f)<2e-5f;}
}
NeutronPackage NeutronPackage::load(const std::filesystem::path& manifest,bool runtime) {
 require(std::endian::native==std::endian::little,"Neutron package requires little-endian host");
 require(std::filesystem::file_size(manifest)<16*1024*1024,"Neutron manifest too large");
 std::ifstream in(manifest);std::string text((std::istreambuf_iterator<char>(in)),{});
 auto root=minjson::Parser(text).parse();
 const bool separate_rates=root.at("schema").str=="maigo.neutron.native-collisions.v2";
 require(separate_rates||root.at("schema").str=="maigo.neutron.native-collisions.v1","Unsupported neutron manifest schema");
 const auto& models=root.at("models");
 require(models.type==minjson::Value::Type::Array&&!models.arr.empty(),"Missing neutron model registry");
 for(const auto& model:models.arr)require(model.type==minjson::Value::Type::String&&!model.str.empty(),"Invalid neutron model name");
 const auto& validated=root.at("runtime_validated");
 require(validated.type==minjson::Value::Type::Boolean,"Invalid neutron validation field");
 require(!runtime || validated.boolean,"Neutron package is native extraction only; runtime validation incomplete");
 const std::filesystem::path relative=root.at("binary").str;
 require(!relative.empty()&&!relative.is_absolute()&&relative.filename()==relative,"Invalid neutron binary filename");
 auto path=manifest.parent_path()/relative;const auto& pin=root.at("binary_sha256").str;
 require(pin.size()==64&&compute_file_sha256_hex(path)==pin,"Neutron binary SHA256 mismatch");
 std::ifstream stream(path,std::ios::binary);char magic[8];std::array<std::uint32_t,7> h{};
 stream.read(magic,8);stream.read(reinterpret_cast<char*>(h.data()),28);
 const auto [version,flags,nm,nr,ne,np,reserved]=h;
 require(stream&&std::memcmp(magic,"NEUTV001",8)==0&&version==1&&reserved==0&&flags<=1,"Invalid neutron header");
 require((flags==1)==validated.boolean,"Neutron manifest/header validation disagreement");
 require(nm==3&&nr>0&&ne>0&&np>0,"Empty neutron package");
 const std::uint64_t bytes=36+std::uint64_t(nm)*16+std::uint64_t(nr)*28+std::uint64_t(ne)*52+std::uint64_t(np)*36;
 require(bytes==std::filesystem::file_size(path)&&bytes<4ULL*1024*1024*1024,"Invalid neutron package size");
 NeutronPackage result;result.runtime_validated=validated.boolean;result.materials.resize(nm);result.rates.resize(nr);result.events.resize(ne);result.products.resize(np);
 auto read=[&](auto& a){stream.read(reinterpret_cast<char*>(a.data()),static_cast<std::streamsize>(a.size()*sizeof(a[0])));};
 read(result.materials);read(result.rates);read(result.events);read(result.products);require(bool(stream),"Truncated neutron package");
 std::set<int> sections;
 for(auto m:result.materials){require(m.section>=-3&&m.section<=-1&&sections.insert(m.section).second&&m.density>0&&m.temperature>0&&m.excitation>0&&std::isfinite(m.density+m.temperature+m.excitation),"Invalid neutron material");if(m.section==-1)require(m.density==1&&m.excitation==75,"Neutron water must be Water_75eV");}
 std::map<std::pair<unsigned,unsigned>,float> last_energy;
 std::map<std::pair<unsigned,unsigned>,float> node_energy;
 std::vector<bool> covered(ne,false);std::uint64_t next_product=0;
 for(std::size_t i=0;i<result.rates.size();++i){auto r=result.rates[i];require(r.material<nm&&(r.subtype==111||r.subtype==121||r.subtype==131||r.subtype==141)&&r.energy>0&&r.rate>=0&&std::isfinite(r.energy+r.rate)&&std::uint64_t(r.event_offset)+r.event_count<=ne,"Invalid neutron rate");require((r.rate==0&&r.event_count==0)||(r.rate>0&&r.event_count>0),"Missing neutron final-state support");auto key=std::make_pair(r.material,r.subtype);if(last_energy.count(key))require(r.energy>last_energy[key],"Unsorted neutron rate energies");last_energy[key]=r.energy;auto nk=std::make_pair(r.material,r.node);if(node_energy.count(nk))require(node_energy[nk]==r.energy,"Neutron node energy disagreement");node_energy[nk]=r.energy;
  for(std::uint64_t j=r.event_offset;j<std::uint64_t(r.event_offset)+r.event_count;++j){require(!covered[j]&&result.events[j].rate==i,"Neutron event coverage mismatch");covered[j]=true;}
 }
 for(std::size_t i=0;i<result.events.size();++i){auto e=result.events[i];require(e.model>=0&&std::size_t(e.model)<models.arr.size()&&e.target_z>0&&e.target_a>=e.target_z&&e.status>=0&&e.status<=5,"Invalid neutron event identity");require(covered[i]&&e.rate<nr&&e.product_offset==next_product&&std::uint64_t(e.product_offset)+e.product_count<=np,"Invalid neutron event range");next_product+=e.product_count;require(e.continuation>=0&&e.deposit>=0&&std::isfinite(e.continuation+e.deposit+e.kinetic_gain)&&(!e.continuation||direction(e.dx,e.dy,e.dz)),"Invalid neutron event kinematics");double sum=e.continuation+double(e.deposit);
  for(std::uint64_t j=e.product_offset;j<std::uint64_t(e.product_offset)+e.product_count;++j){auto p=result.products[j];require(p.mass>=0&&p.energy>=0&&std::isfinite(p.mass+p.energy+p.time)&&direction(p.dx,p.dy,p.dz),"Invalid neutron product");sum+=p.energy;}
  double energy=result.rates[e.rate].energy;require(std::abs(sum-energy-e.kinetic_gain)<2e-5*std::max(1.,energy),"Neutron serialized kinetic bookkeeping mismatch");
 }
 require(next_product==np,"Unreferenced neutron products");
 if(separate_rates){
  const auto& meta=root.at("macroscopic");
  const std::filesystem::path filename=meta.at("binary").str;
  require(!filename.empty()&&!filename.is_absolute()&&filename.filename()==filename,"Invalid neutron macroscopic filename");
  const auto rate_path=manifest.parent_path()/filename;
  require(meta.at("binary_sha256").str.size()==64&&compute_file_sha256_hex(rate_path)==meta.at("binary_sha256").str,"Neutron macroscopic checksum mismatch");
  std::ifstream rate_stream(rate_path,std::ios::binary);
  char rate_magic[8];std::array<std::uint32_t,4> rh{};
  rate_stream.read(rate_magic,8);rate_stream.read(reinterpret_cast<char*>(rh.data()),16);
  require(rate_stream&&std::memcmp(rate_magic,"NEUTR001",8)==0&&rh[0]==1&&rh[1]==0&&rh[2]>0&&rh[3]==0,"Invalid neutron macroscopic header");
  const std::uint64_t rate_bytes=24+std::uint64_t(rh[2])*28;
  require(rate_bytes==std::filesystem::file_size(rate_path)&&rate_bytes<1024ULL*1024*1024,"Invalid neutron macroscopic size");
  require(meta.at("count").number==rh[2]&&meta.at("record_bytes").number==28,"Neutron macroscopic manifest count mismatch");
  result.macroscopic_rates.resize(rh[2]);
  rate_stream.read(reinterpret_cast<char*>(result.macroscopic_rates.data()),rate_bytes-24);
  require(bool(rate_stream),"Truncated neutron macroscopic table");
  std::map<std::pair<unsigned,unsigned>,std::vector<NeutronMacroscopicRecord>> grids;
  std::pair<unsigned,unsigned> previous{};bool first=true;
  for(const auto& r:result.macroscopic_rates){
   require(r.material<nm&&(r.subtype==111||r.subtype==121||r.subtype==131||r.subtype==141)&&r.queries>1&&r.interval_status<=5&&r.energy>0&&r.rate>=0&&r.sem>=0&&std::isfinite(r.energy)&&std::isfinite(r.rate)&&std::isfinite(r.sem),"Invalid neutron macroscopic record");
   const auto key=std::make_pair(r.material,r.subtype);
   require(first||key>=previous,"Unsorted neutron macroscopic processes");first=false;previous=key;
   auto& grid=grids[key];
   require(grid.empty()||(r.energy>grid.back().energy&&grid.back().interval_status!=0),"Unsorted or prematurely terminated neutron grid");
   if(!grid.empty()&&grid.back().interval_status==3)require(std::nextafter(grid.back().energy,INFINITY)==r.energy,"Invalid adjacent FP32 interval");
   grid.push_back(r);
  }
  require(grids.size()==12,"Incomplete neutron macroscopic process coverage");
  for(unsigned m=0;m<nm;++m){
   const auto& elastic=grids.at({m,111});
   for(unsigned subtype:{111u,121u,131u,141u}){
    const auto& grid=grids.at({m,subtype});
    require(grid.size()>=2&&grid.size()==elastic.size()&&grid.back().interval_status==0,"Incomplete neutron macroscopic grid");
    for(std::size_t i=0;i<grid.size();++i)require(grid[i].energy==elastic[i].energy,"Neutron process energy grid mismatch");
    for(const auto& r:result.rates)if(r.material==m&&r.subtype==subtype)require(r.energy>=grid.front().energy&&r.energy<=grid.back().energy,"Neutron rate grid does not cover event bank");
   }
  }
  // v2 remains an audit format until event interpolation and descendants exist.
  require(!runtime&&!result.runtime_validated,"Independent neutron grids are not runtime validated");
 }
 return result;
}
}
