#include "carbon/unified_em_view.hpp"
#include <iostream>
#include <array>
using namespace carbon;
struct Uniform {uint64_t state=202610091234;unsigned calls=0;float operator()(){++calls;uint64_t z=(state+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return float(z>>40)*0x1p-24f;}};
int main(){auto p=UnifiedEmPackage::load("data/em/ions/continuous26.bin");auto idx=build_unified_em_index(p);UnifiedEmDevice d{};
d.materials=p.materials.data();d.species=p.species.data();d.records=p.records.data();d.nodes=p.nodes.data();d.segments=p.segments.data();d.material_count=p.materials.size();d.energy_index=idx.data();d.continuous_descriptors=p.continuous_descriptors.data();d.continuous_scalars=p.continuous_scalars.data();d.continuous_ranges=p.continuous_ranges.data();d.continuous_degree=p.continuous_degree;d.continuous_scalar_count=p.continuous_scalar_count;d.continuous_lengths=p.continuous_lengths.data();unsigned n=0,invalid=0;
int ion=d.species_index(1,1);
for(int sec=-1;sec<25;++sec)for(float f:{0.f,.01f,.15f,.5f,.8f,.99f,1.f}){auto desc=p.continuous_descriptors[(sec+1)*18+ion];float rho=std::exp(desc.log_min+(desc.log_max-desc.log_min)*f);auto s=d.select(sec,rho,ion);if(!s.valid)return 2;auto point=s.lo();auto rec=point.record;
float lo=p.nodes[rec->node_offset].energy,hi=p.nodes[rec->node_offset+rec->node_count-1].energy;
for(unsigned j=0;j<100;++j){float t=lo*std::pow(hi/lo,(j+.1f)/100);auto pre=s.prepare(t);auto pp=point.prepare(t);
float restricted=0.f,delta=0.f;point.birth_stopping(t,restricted,delta);
if(restricted!=pre.lo.stopping||delta!=pre.lo.delta_stopping){std::cerr<<"birth mismatch "<<sec<<" "<<rho<<" "<<t<<"\n";return 6;}
auto fields=[](const UnifiedEmPointStep& x){return std::array{x.factor,x.range,x.stopping,x.q2,x.dispersion,x.universal_dispersion,x.delta_stopping,x.delta_variance,x.stopping_slope,x.delta_slope,x.delta_partition_rate};};if(fields(pre.lo)!=fields(pp)||pre.lo.ion_fluctuation!=pp.ion_fluctuation)return 3;
for(float h:{.001f,.1f,pre.lo.range*.9f,pre.lo.range*1.1f})for(bool fluct:{false,true}){Uniform a{},b{};auto x=unified_em_explicit_loss(s,t,h,fluct,pre,a);auto y=unified_em_explicit_loss<false,true>(s,t,h,fluct,pre,b,&point);
auto lossfields=[](const UnifiedEmLoss& x){return std::array{x.loss,x.continuous,x.delta,x.mean,x.delta_mean,x.delta_variance};};
auto xf=lossfields(x),yf=lossfields(y);for(unsigned k=0;k<xf.size();++k)if(xf[k]!=yf[k]&&!(std::isnan(xf[k])&&std::isnan(yf[k])))return 4;
if(x.valid!=y.valid||x.failure_stage!=y.failure_stage||x.proposed!=y.proposed||x.accepted!=y.accepted||a.calls!=b.calls||a.state!=b.state)return 5;++n;if(!x.valid)++invalid;
}}}
std::cout<<"{\"cases\":"<<n<<",\"exact_fields_rng_match\":true,\"reference_invalid\":"<<invalid<<"}\n";
}
