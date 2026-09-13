#include "carbon/unified_em_view.hpp"
#include <iostream>
int main(int argc,char** argv){
 if(argc!=3)throw std::runtime_error("usage: unified_em_density package sha");
 auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);unsigned count=0;float max_mean=0,max_range=0,max_rate=0;
 for(unsigned m=1;m+1<p.materials.size();++m){
  if(p.materials[m-1].section!=p.materials[m].section || p.materials[m+1].section!=p.materials[m].section)continue;
  float rho=p.materials[m].density,w=(rho-p.materials[m-1].density)/(p.materials[m+1].density-p.materials[m-1].density);
  for(int s=0;s<18;++s){auto point=[&](int j){auto* r=&p.records[j*18+s];return carbon::UnifiedEmPoint{r,p.nodes.data()+r->node_offset,p.segments.data(),rho/p.materials[j].density};};
   auto lo=point(m-1),hi=point(m+1),truth=point(m);auto mix=[&](float a,float b){return a+w*(b-a);};
   for(float e:{.1f,1.f,2.f,10.f,60.f,100.f,300.f}){float t=e*truth.record->a,h=truth.range(t)*.01f;
    auto rel=[](float value,float ref){return std::abs(value-ref)/std::max(ref,1e-6f);};
    float err=rel(mix(lo.mean(t,h),hi.mean(t,h)),truth.mean(t,h));if(err>max_mean){max_mean=err;std::cerr<<"maximum_mean section="<<p.materials[m].section<<" rho="<<rho<<" Z="<<truth.record->z<<" A="<<truth.record->a<<" E="<<e<<" relative="<<err<<"\n";}
    max_range=std::max(max_range,rel(mix(lo.range(t),hi.range(t)),truth.range(t)));
    float peak=0;for(unsigned k=0;k<truth.record->node_count;++k)peak=std::max(peak,truth.nodes[k].lambda);
    max_rate=std::max(max_rate,std::abs(mix(lo.rate(t,lo.factor(t)),hi.rate(t,hi.factor(t)))-truth.rate(t,truth.factor(t)))/std::max(truth.rate(t,truth.factor(t)),peak*.001f));++count;
   }
  }
 }
 std::cout<<"withheld_node_queries="<<count<<" max_mean_relative="<<max_mean<<" max_range_relative="<<max_range<<" max_rate_relative="<<max_rate<<'\n';
 // Diagnostic only: density production acceptance needs independent extra nodes.
}
