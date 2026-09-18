#include "carbon/all_ion_elastic.hpp"
#include "carbon/sha256.hpp"
#include <fstream>
#include <cstring>
#include <stdexcept>
namespace carbon {
AllIonElasticTable AllIonElasticTable::load(const std::filesystem::path& path,const std::string& pin){
 if(pin.size()!=64||!file_sha256_matches(path,pin))throw std::runtime_error("Elastic bank SHA mismatch");
 std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);
 std::uint32_t h[6];f.read(reinterpret_cast<char*>(h),sizeof(h));
 if(!f||std::memcmp(magic,"ELBANK01",8)||h[0]!=1||h[1]!=18||h[2]!=26||h[3]!=13||h[4]<2||h[4]>10000||h[5]<32||h[5]>65536)
  throw std::runtime_error("Invalid elastic bank schema");
 const std::uint64_t ne=h[4],nq=h[5],nr=18*26*13*ne,ns=18*13*ne*nq;
 if(std::filesystem::file_size(path)!=32+18*8+ne*4+nr*4+ns*12)throw std::runtime_error("Elastic bank size mismatch");
 AllIonElasticTable t;t.nq=nq;t.energies.resize(ne);t.rates.resize(nr);t.samples.resize(ns);
 f.read(reinterpret_cast<char*>(t.masses.data()),18*8);f.read(reinterpret_cast<char*>(t.energies.data()),ne*4);
 f.read(reinterpret_cast<char*>(t.rates.data()),nr*4);f.read(reinterpret_cast<char*>(t.samples.data()),ns*12);
 if(!f)throw std::runtime_error("Truncated elastic bank");
 for(double m:t.masses)if(!std::isfinite(m)||m<=0)throw std::runtime_error("Bad elastic mass");
 for(unsigned i=0;i<ne;++i)if(!std::isfinite(t.energies[i])||t.energies[i]<=0||(i&&t.energies[i]<=t.energies[i-1]))throw std::runtime_error("Bad elastic grid");
 for(float r:t.rates)if(!std::isfinite(r)||r<0)throw std::runtime_error("Bad elastic rate");
 for(auto s:t.samples)if(!std::isfinite(s.transfer_fraction)||s.transfer_fraction<0||s.transfer_fraction>1||!std::isfinite(s.target_mass_MeV)||s.target_mass_MeV<=0||s.target_a>250)throw std::runtime_error("Bad elastic sample");
 // Every node with an active material rate must contain valid target samples.
 for(int p=0;p<18;++p)for(int j=0;j<13;++j)for(unsigned e=0;e<ne;++e){bool positive=false;
  for(int s=0;s<26;++s)positive|=t.rates[((p*26+s)*13+j)*ne+e]>0;
  if(positive)for(unsigned q=0;q<nq;++q)if(t.samples[((p*13+j)*ne+e)*nq+q].target_a==0)throw std::runtime_error("Missing active elastic channel");}
 return t;
}
}
namespace carbon {
ElasticRecoilStoppingTable ElasticRecoilStoppingTable::load(const std::filesystem::path& path,const std::string& pin){
 if(pin.size()!=64||!file_sha256_matches(path,pin))throw std::runtime_error("Recoil stopping SHA mismatch");
 std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);std::uint32_t h[3];f.read(reinterpret_cast<char*>(h),12);
 if(!f||std::memcmp(magic,"ELRSP001",8)||h[0]==0||h[0]>200||h[1]!=26||h[2]<2||h[2]>10000)throw std::runtime_error("Invalid recoil stopping schema");
 const std::uint64_t np=h[0],ne=h[2],nv=np*26*ne;
 if(std::filesystem::file_size(path)!=20+np*8+ne*4+nv*4)throw std::runtime_error("Recoil stopping size mismatch");
 ElasticRecoilStoppingTable t;t.keys.resize(np*2);t.energies.resize(ne);t.values.resize(nv);
 f.read(reinterpret_cast<char*>(t.keys.data()),np*8);f.read(reinterpret_cast<char*>(t.energies.data()),ne*4);f.read(reinterpret_cast<char*>(t.values.data()),nv*4);
 if(!f)throw std::runtime_error("Truncated recoil stopping");
 for(unsigned i=0;i<np;++i){if(t.keys[2*i]<=0||t.keys[2*i+1]<t.keys[2*i]||t.keys[2*i+1]>250)throw std::runtime_error("Invalid recoil isotope");
  for(unsigned j=0;j<i;++j)if(t.keys[2*i]==t.keys[2*j]&&t.keys[2*i+1]==t.keys[2*j+1])throw std::runtime_error("Duplicate recoil isotope");}
 for(unsigned i=0;i<ne;++i)if(!std::isfinite(t.energies[i])||t.energies[i]<=0||(i&&t.energies[i]<=t.energies[i-1]))throw std::runtime_error("Bad recoil grid");
 for(float v:t.values)if(!std::isfinite(v)||v<=0)throw std::runtime_error("Bad recoil stopping value");
 return t;
}
}
