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
namespace {
// The runtime keeps Urban energy support and MFP only. This offline extraction
// audit must still compare the original numerical factor with held-out G4
// probes, so read that validated disk payload here rather than restoring it
// to every production GPU/host context.
struct NativeFactorRecord {
    float mass_ratio,minimum_scaled_energy;
    std::uint32_t offset,count,mfp_offset,mfp_count;
    std::uint32_t spline_offsets[3],spline_counts[3];
};
static_assert(sizeof(NativeFactorRecord)==48);
struct NativeFactorReference {
    std::vector<NativeFactorRecord> records;
    std::vector<carbon::UrbanMcsNode> nodes;
    unsigned species_count{};
    explicit NativeFactorReference(const char* path) {
        std::ifstream in(path,std::ios::binary);char magic[8];std::array<std::uint32_t,7> h{};
        in.read(magic,8);in.read(reinterpret_cast<char*>(h.data()),28);
        if(!in || std::string(magic,8)!="URBANV22" || h[0]!=2 || h[6]!=1)
            throw std::runtime_error("invalid native factor reference header");
        species_count=h[2];records.resize(std::uint64_t(h[1])*h[2]);nodes.resize(h[3]);
        in.seekg(36ULL+h[1]*20ULL+h[2]*12ULL);
        in.read(reinterpret_cast<char*>(records.data()),records.size()*sizeof(records[0]));
        in.read(reinterpret_cast<char*>(nodes.data()),nodes.size()*sizeof(nodes[0]));
        if(!in)throw std::runtime_error("truncated native factor reference");
    }
    float at(unsigned material,unsigned ion,float energy) const {
        const auto& r=records.at(std::uint64_t(material)*species_count+ion);
        return carbon::UrbanMcsContext::interpolate(nodes.data()+r.offset,r.count,energy);
    }
};
}

int main(int argc,char** argv) {
 try {
    if(argc!=4)throw std::runtime_error("usage: urban_mcs_reference package probes mfp_probes");
    auto package=carbon::UrbanMcsPackage::load(argv[1]);auto view=package.view();
    NativeFactorReference native_factor(argv[1]);
    const char* names[]={"lambda","factor","query_lambda","wide_mfp"};
    std::array<float,4> worst{};std::array<Probe,4> where{};unsigned long long count=0,failed=0;
    auto check=[&](int k,float got,float ref,const Probe& p) {
        float error=std::fabs(got/ref-1.f);
        if(!std::isfinite(error) || error>worst[k]){worst[k]=error;where[k]=p;}
        if(!std::isfinite(error) || error>5.e-4f)++failed;
    };
    std::ifstream in(argv[2],std::ios::binary);if(!in)throw std::runtime_error("missing probes");Probe p;
    while(in.read(reinterpret_cast<char*>(&p),sizeof(p))) {
        const float e=p.v[0];const auto c=view.at(p.m,p.s,e);
        if(!c.valid())throw std::runtime_error("missing held-out step context");
        check(0,c.lambda(e),p.v[3],p);
        check(1,native_factor.at(p.m,p.s,e),p.v[5],p);
        check(2,c.lambda(.37f*e),p.v[9],p);
        ++count;
    }
    if(!in.eof() || count==0)throw std::runtime_error("truncated or empty probes");
    std::ifstream mi(argv[3],std::ios::binary);if(!mi)throw std::runtime_error("missing MFP probes");MfpProbe mp;
    while(mi.read(reinterpret_cast<char*>(&mp),sizeof(mp))) {
        const auto c=view.at(mp.m,mp.s,1.f);
        Probe context{mp.m,mp.s,{mp.e}};check(3,c.lambda(mp.e),mp.value,context);
    }
    if(!mi.eof())throw std::runtime_error("truncated MFP probes");
    std::cout<<std::setprecision(9);
    for(int k=0;k<4;++k) {
        const auto& ion=package.species[where[k].s];
        std::cout<<names[k]<<" max_relative_error="<<worst[k]<<" material="<<where[k].m
                 <<" Z="<<ion.z<<" A="<<ion.a<<" E="<<where[k].v[0]<<'\n';
    }
    std::cout<<"FP32_REFERENCE="<<(failed?"FAIL":"PASS")<<" steps="<<count<<" failed_queries="<<failed<<'\n';
    return failed?1:0;
 } catch(const std::exception& e) {std::cerr<<"FP32_REFERENCE=FAIL "<<e.what()<<'\n';return 2;}
}
