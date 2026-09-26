// Independent held-out real Geant4 steps, evaluated by the production FP32 view.
#include "carbon/urban_mcs_package.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
struct Probe {std::uint32_t m,s;float v[10];};
struct MfpProbe {std::uint32_t m,s;float e,value;};
static_assert(sizeof(Probe)==48 && sizeof(MfpProbe)==16);
int main(int argc,char** argv) {
 try {
    if(argc!=5)throw std::runtime_error("usage: urban_mcs_reference package sha probes mfp_probes");
    auto package=carbon::UrbanMcsPackage::load(argv[1],argv[2]);auto view=package.view();
    const char* names[]={"range","dedx","lambda","inverse","factor","query_range","query_dedx","query_energy","query_lambda","wide_mfp"};
    std::array<float,10> worst{};std::array<Probe,10> where{};unsigned long long count=0,failed=0;
    auto check=[&](int k,float got,float ref,const Probe& p) {
        float error=std::fabs(got/ref-1.f);
        if(!std::isfinite(error) || error>worst[k]){worst[k]=error;where[k]=p;}
        if(!std::isfinite(error) || error>5.e-4f)++failed;
    };
    std::ifstream in(argv[3],std::ios::binary);if(!in)throw std::runtime_error("missing probes");Probe p;
    while(in.read(reinterpret_cast<char*>(&p),sizeof(p))) {
        const float e=p.v[0];const auto c=view.at(p.m,p.s,e);
        if(!c.valid())throw std::runtime_error("missing held-out step context");
        const float range=c.range(e);
        const float got[]={range,c.dedx(e),c.lambda(e),c.energy(range),c.factor,
            c.range(.37f*e),c.dedx(.37f*e),c.energy(.37f*range),c.lambda(.37f*e)};
        for(int k=0;k<9;++k)check(k,got[k],p.v[k+1],p);
        ++count;
    }
    if(!in.eof() || count==0)throw std::runtime_error("truncated or empty probes");
    std::ifstream mi(argv[4],std::ios::binary);if(!mi)throw std::runtime_error("missing MFP probes");MfpProbe mp;
    while(mi.read(reinterpret_cast<char*>(&mp),sizeof(mp))) {
        const auto c=view.at(mp.m,mp.s,1.f);
        Probe context{mp.m,mp.s,{mp.e}};check(9,c.lambda(mp.e),mp.value,context);
    }
    if(!mi.eof())throw std::runtime_error("truncated MFP probes");
    std::cout<<std::setprecision(9);
    for(int k=0;k<10;++k) {
        const auto& ion=package.species[where[k].s];
        std::cout<<names[k]<<" max_relative_error="<<worst[k]<<" material="<<where[k].m
                 <<" Z="<<ion.z<<" A="<<ion.a<<" E="<<where[k].v[0]<<'\n';
    }
    std::cout<<"FP32_REFERENCE="<<(failed?"FAIL":"PASS")<<" steps="<<count<<" failed_queries="<<failed<<'\n';
    return failed?1:0;
 } catch(const std::exception& e) {std::cerr<<"FP32_REFERENCE=FAIL "<<e.what()<<'\n';return 2;}
}
