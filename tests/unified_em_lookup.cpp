#include "carbon/unified_em_view.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv){
    if(argc!=3)throw std::runtime_error("usage: unified_em_lookup package sha256");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);
    float max_sp=0,max_range=0,max_rate=0,max_rate_scaled=0;std::uint64_t queries=0;
    for(std::size_t i=0;i<p.records.size();++i){
        const auto& r=p.records[i];carbon::UnifiedEmPoint v{&r,p.nodes.data()+r.node_offset,p.segments.data(),1};
        float peak_rate=0;for(unsigned j=0;j<r.node_count;++j)peak_rate=std::max(peak_rate,v.nodes[j].lambda);
        for(unsigned j=0;j<r.node_count;j+=37){
            auto n=v.nodes[j];float t=n.energy*r.a;
            auto error=[](float a,float b){return b>1e-20f?std::abs(a-b)/b:std::abs(a-b);};
            float es=error(v.raw(0,t*r.ratio)*n.factor,n.stopping);
            if(es>max_sp && es>.001f)std::cerr<<"sp failure record="<<i<<" Z="<<r.z<<" A="<<r.a<<" E="<<n.energy<<" calc="<<v.raw(0,t*r.ratio)*n.factor<<" ref="<<n.stopping<<"\n";
            max_sp=std::max(max_sp,es);
            max_range=std::max(max_range,error(v.range(t),n.range));
            float er=error(v.rate(t,n.factor),n.lambda);
            if(er>max_rate && er>.001f)std::cerr<<"lambda failure record="<<i<<" E="<<n.energy<<" calc="<<v.rate(t,n.factor)<<" ref="<<n.lambda<<"\n";
            max_rate=std::max(max_rate,er);
            max_rate_scaled=std::max(max_rate_scaled,std::abs(v.rate(t,n.factor)-n.lambda)/std::max(n.lambda,peak_rate*.001f));
            ++queries;
        }
    }
    std::cerr<<"records="<<p.records.size()<<" queries="<<queries<<" stopping="<<max_sp<<" range="<<max_range<<" lambda="<<max_rate<<" mixed_relative="<<max_rate_scaled<<"\n";
    if(max_sp>0.001 || max_range>0.001 || max_rate_scaled>0.001)throw std::runtime_error("Native table reconstruction failed");
}
