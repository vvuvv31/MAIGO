#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>
#include "carbon/nuclear_elastic_scattering.hpp"
namespace carbon {
inline constexpr std::array<std::array<int,2>,18> elastic_projectiles{{
 {1,1},{1,2},{1,3},{2,3},{2,4},{2,6},{3,6},{3,7},{4,7},{4,9},{4,10},
 {5,8},{5,10},{5,11},{6,10},{6,11},{6,12},{4,6}}};
struct ElasticSample { float transfer_fraction{},target_mass_MeV{};std::uint32_t target_a{}; };
struct ElasticDraw { NuclearElasticOutcome outcome{}; int target_z{},target_a{}; bool valid{}; };
struct AllIonElasticView {
 const float* energies{};const float* rates{};const ElasticSample* samples{};
 const double* masses{};std::uint32_t ne{},nq{};
 int projectile(int z,int a) const noexcept {
  for(int i=0;i<18;++i)if(elastic_projectiles[i][0]==z&&elastic_projectiles[i][1]==a)return i;
  return -1;
 }
 int cell(float e) const noexcept {
  if(!energies||!std::isfinite(e)||e<energies[0]||e>energies[ne-1])return -1;
  unsigned lo=0,hi=ne-1;while(hi-lo>1){unsigned mid=(lo+hi)/2;if(e<energies[mid])hi=mid;else lo=mid;}
  return static_cast<int>(lo);
 }
 float rate(int p,int section,float e,int target=-1) const noexcept {
  int i=cell(e);if(p<0||p>=18||section<0||section>=26||i<0)return -1;
  float w=(e-energies[i])/(energies[i+1]-energies[i]),sum=0;
  for(int t=0;t<13;++t){if(target>=0&&t!=target)continue;
   auto b=((p*26+section)*13+t)*ne+i;sum+=rates[b]*(1-w)+rates[b+1]*w;}
  return sum;
 }
 ElasticDraw draw(int p,int section,float T,float dx,float dy,float dz,
                  float ut,float un,float us,float uphi) const noexcept {
  ElasticDraw out{};if(p<0||p>=18)return out;
  float e=T/elastic_projectiles[p][1];int i=cell(e);float total=rate(p,section,e);
  if(i<0||!(total>0))return out;
  float acc=0;int t=-1,last=-1;
  for(int j=0;j<13;++j){float v=rate(p,section,e,j);if(v>0)last=j;acc+=v;if(ut*total<acc){t=j;break;}}
  if(t<0)t=last;if(t<0)return out;
  float w=(e-energies[i])/(energies[i+1]-energies[i]);auto b=((p*26+section)*13+t)*ne+i;
  float lower=(1-w)*rates[b],upper=w*rates[b+1];int node=i+(un*(lower+upper)>=lower);
  auto q=static_cast<unsigned>(us*nq);if(q>=nq)q=nq-1;
  const auto record=samples[((p*13+t)*ne+node)*nq+q];
  if(record.target_a==0||!(record.target_mass_MeV>0))return out;
  out.outcome=elastic_two_body(T,dx,dy,dz,1.0-2.0*record.transfer_fraction,uphi,masses[p],record.target_mass_MeV);
  out.target_z=kSchneiderCanonicalZ[t];out.target_a=record.target_a;out.valid=true;return out;
 }
};
struct AllIonElasticTable {
 std::vector<float> energies,rates;std::vector<ElasticSample> samples;
 std::array<double,18> masses{};std::uint32_t nq{};
 static AllIonElasticTable load(const std::filesystem::path&,const std::string& sha256);
 AllIonElasticView view()const{return {energies.data(),rates.data(),samples.data(),masses.data(),static_cast<std::uint32_t>(energies.size()),nq};}
};
}
namespace carbon {
struct ElasticRecoilStoppingView {
 const std::int32_t* keys{};const float* energies{};const float* values{};
 std::uint32_t np{},ne{};
 int projectile(int z,int a)const noexcept{for(unsigned i=0;i<np;++i)if(keys[2*i]==z&&keys[2*i+1]==a)return i;return -1;}
 float stopping(int p,int section,float e)const noexcept{
  if(p<0||p>=static_cast<int>(np)||section<0||section>=26||!std::isfinite(e)||e<energies[0]||e>energies[ne-1])return -1;
  unsigned lo=0,hi=ne-1;while(hi-lo>1){unsigned m=(lo+hi)/2;if(e<energies[m])hi=m;else lo=m;}
  const float w=(e-energies[lo])/(energies[hi]-energies[lo]);auto b=(p*26+section)*ne;
  return values[b+lo]*(1-w)+values[b+hi]*w;
 }
};
struct ElasticRecoilStoppingTable {
 std::vector<std::int32_t> keys;std::vector<float> energies,values;
 static ElasticRecoilStoppingTable load(const std::filesystem::path&,const std::string&);
};
}
