#include "carbon/unified_em_view.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok) { if(!ok)throw std::runtime_error("Shared EM preparation regression"); }
bool equal(const carbon::UnifiedEmPointStep& a,const carbon::UnifiedEmPointStep& b) {
    return a.factor==b.factor && a.range==b.range && a.stopping==b.stopping &&
        a.q2==b.q2 && a.dispersion==b.dispersion && a.universal_dispersion==b.universal_dispersion &&
        a.ion_fluctuation==b.ion_fluctuation && a.delta_stopping==b.delta_stopping &&
        a.delta_variance==b.delta_variance && a.stopping_slope==b.stopping_slope &&
        a.delta_slope==b.delta_slope && a.delta_partition_rate==b.delta_partition_rate;
}
}
int main() {
    auto p=carbon::UnifiedEmPackage::load("data/em/unified_em_v1.bin");
    require(p.shared_prepare_grid && carbon::unified_em_has_shared_prepare_grid(p));
    auto index=carbon::build_unified_em_index(p);
    carbon::UnifiedEmDevice v{p.materials.data(),p.species.data(),p.records.data(),p.nodes.data(),
        p.segments.data(),unsigned(p.materials.size()),nullptr,index.data()};
    v.shared_prepare_grid=p.shared_prepare_grid;
    unsigned checked=0;
    for(unsigned m=0;m<p.materials.size();++m)for(unsigned ion=0;ion<p.species.size();++ion) {
        float density=p.materials[m].density;
        if(m+1<p.materials.size() && p.materials[m+1].section==p.materials[m].section)
            density=(density+p.materials[m+1].density)*.5f;
        auto s=v.select(p.materials[m].section,density,ion);require(s.valid);
        const auto& r=*s.lo().record;
        // Native node boundaries, including immediate FP32 neighbours.
        for(unsigned j:{0u,r.node_count/2,r.node_count-1}) {
            const float mid=p.nodes[r.node_offset+j].energy*r.a;
            for(float e:{std::nextafter(mid,0.f),mid,std::nextafter(mid,INFINITY)}) {
                const auto actual=s.prepare(e);
                require(equal(actual.lo,s.lo().prepare(e)) && equal(actual.hi,s.hi().prepare(e)));
                ++checked;
            }
        }
    }
    // A different valid package must disable reuse; the two searches remain exact.
    auto& r=p.records[2*p.species.size()];
    for(float* x:{&r.ratio,&p.nodes[r.node_offset+1].energy,
                  &p.segments[r.offsets[0]+1].lower,&p.segments[r.offsets[1]+1].lower}) {
        const float old=*x;*x=std::nextafter(old,INFINITY);
        v.shared_prepare_grid=carbon::unified_em_has_shared_prepare_grid(p);
        require(!v.shared_prepare_grid);
        auto s=v.select(p.materials[2].section,p.materials[2].density,0);
        for(float e:{.11f,1.f,100.f}) {
            const auto actual=s.prepare(e);
            require(equal(actual.lo,s.lo().prepare(e)) && equal(actual.hi,s.hi().prepare(e)));
        }
        *x=old;
    }
    std::cout<<"Shared EM preparation: "<<checked<<" boundaries and four fallback cases passed\n";
}
