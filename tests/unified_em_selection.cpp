#include "carbon/unified_em_view.hpp"
#include <bit>
#include <iostream>
#include <limits>

// Frozen pre-optimization linear search is the independent reference.
carbon::UnifiedEmState reference(const carbon::UnifiedEmDevice& d,int section,float density,int ion) {
    carbon::UnifiedEmState s;
    if(ion<0 || ion>=18 || section< -1 || section>24 || !(density>0))return s;
    int begin=-1,end=-1;
    for(unsigned i=0;i<d.material_count;++i)if(d.materials[i].section==section){if(begin<0)begin=i;end=i;}
    if(begin<0 || density<d.materials[begin].density*(1-2e-6f) || density>d.materials[end].density*(1+2e-6f))return s;
    int low=begin;while(low<end && d.materials[low+1].density<density)++low;
    int high=std::min(low+1,end);
    s.weight=high==low?0:std::clamp((density-d.materials[low].density)/(d.materials[high].density-d.materials[low].density),0.f,1.f);
    auto point=[&](int i){auto* r=d.records+i*18+ion;return carbon::UnifiedEmPoint{r,d.nodes+r->node_offset,d.segments,density/d.materials[i].density};};
    s.lo=point(low);s.hi=point(high);s.density=density;s.section=section;s.valid=true;return s;
}
int main(int argc,char** argv) {
    if(argc!=3)throw std::runtime_error("usage: unified_em_selection package sha");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);
    std::array<carbon::UnifiedEmSectionRange,26> ranges;
    for(unsigned i=0;i<p.materials.size();++i){auto& r=ranges[p.materials[i].section+1];if(r.begin<0)r.begin=i;r.end=i;}
    carbon::UnifiedEmDevice d{p.materials.data(),p.species.data(),p.records.data(),p.nodes.data(),p.segments.data(),unsigned(p.materials.size()),ranges.data()};
    unsigned checks=0;
    auto check=[&](int section,float rho,int ion) {
        auto expected=reference(d,section,rho,ion),actual=d.select(section,rho,ion);
        auto same=[](float a,float b){return std::bit_cast<unsigned>(a)==std::bit_cast<unsigned>(b);};
        if(expected.valid!=actual.valid || (actual.valid &&
            (expected.lo.record!=actual.lo.record || expected.hi.record!=actual.hi.record ||
             expected.lo.nodes!=actual.lo.nodes || expected.hi.nodes!=actual.hi.nodes ||
             !same(expected.weight,actual.weight) || !same(expected.lo.density_scale,actual.lo.density_scale) ||
             !same(expected.hi.density_scale,actual.hi.density_scale))))throw std::runtime_error("Selection changed");
        ++checks;
    };
    for(unsigned m=0;m<p.materials.size();++m)for(int ion=0;ion<18;++ion) {
        auto material=p.materials[m];float rho=material.density;
        for(float x:{rho,std::nextafter(rho,0.f),std::nextafter(rho,std::numeric_limits<float>::infinity()),rho*(1-3e-6f),rho*(1+3e-6f)})check(material.section,x,ion);
        if(m+1<p.materials.size() && p.materials[m+1].section==material.section)
            for(int k=1;k<16;++k)check(material.section,rho+(p.materials[m+1].density-rho)*k/16,ion);
    }
    for(int section=-2;section<=25;++section)for(int ion:{-1,0,17,18})
        for(float rho:{-1.f,0.f,1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})check(section,rho,ion);
    std::cout<<"selection comparisons="<<checks<<" bit-exact; invalid domains rejected\n";
}
