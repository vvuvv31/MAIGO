#include "carbon/unified_em_package.hpp"
#include "carbon/sha256.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace carbon {
UnifiedEmPackage UnifiedEmPackage::load(const std::filesystem::path& path,const std::string& sha) {
    if(std::endian::native!=std::endian::little)throw std::runtime_error("EMJOINT1 requires little endian");
    if(sha.size()!=64 || !file_sha256_matches(path,sha))throw std::runtime_error("Unified EM SHA256 mismatch");
    std::ifstream in(path,std::ios::binary);
    char magic[8];std::uint32_t h[6];in.read(magic,8);in.read(reinterpret_cast<char*>(h),sizeof h);
    if(!in || std::memcmp(magic,"EMJOINT1",8) || h[0]!=1 || h[1]<26 || h[1]>256 || h[2]!=18 || h[3]!=h[1]*h[2])
        throw std::runtime_error("Invalid unified EM header/registry");
    const std::uint64_t expected=32ULL+8ULL*h[1]+8ULL*h[2]+104ULL*h[3]+52ULL*h[4]+24ULL*h[5];
    if(std::filesystem::file_size(path)!=expected || expected>4ULL*1024*1024*1024)
        throw std::runtime_error("Unified EM size mismatch");
    UnifiedEmPackage p;
    auto read=[&](auto& v,std::uint32_t count){v.resize(count);in.read(reinterpret_cast<char*>(v.data()),count*sizeof(v[0]));if(!in)throw std::runtime_error("Truncated unified EM payload");};
    read(p.materials,h[1]);read(p.species,h[2]);read(p.records,h[3]);read(p.nodes,h[4]);read(p.segments,h[5]);
    const std::set<std::pair<unsigned,unsigned>> required={{1,1},{1,2},{1,3},{2,3},{2,4},{2,6},{3,6},{3,7},{4,6},{4,7},{4,9},{4,10},{5,8},{5,10},{5,11},{6,10},{6,11},{6,12}};
    std::set<std::pair<unsigned,unsigned>> found;
    for(auto s:p.species)found.emplace(s.z,s.a);
    if(found!=required)throw std::runtime_error("Unified EM missing or duplicate charged species");
    std::array<int,26> coverage{};int previous=-2;float rho=0;
    for(const auto& m:p.materials){
        if(m.section< -1 || m.section>24 || !std::isfinite(m.density) || m.density<=0 || m.section<previous || (m.section==previous && m.density<=rho))
            throw std::runtime_error("Invalid unified EM material/density directory");
        ++coverage[m.section+1];previous=m.section;rho=m.density;
    }
    for(int count:coverage)if(!count)throw std::runtime_error("Unified EM missing water/Schneider section");
    if(coverage[0]!=1 || p.materials[0].density!=1)throw std::runtime_error("Invalid reference water entry");
    for(std::size_t i=0;i<p.records.size();++i){
        const auto& r=p.records[i];const auto s=p.species[i%18];
        for(float value:std::array{r.mass,r.ratio,r.cut,r.excitation,r.e0,r.spin,r.step_fraction,r.final_range,r.linear_limit,r.is_ion,r.form_factor,r.magnetic_moment2,r.lowest_kinetic,r.peak})
            if(!std::isfinite(value))throw std::runtime_error("Non-finite unified EM record");
        if(r.z!=s.z || r.a!=s.a || r.node_count<2 || std::uint64_t(r.node_offset)+r.node_count>p.nodes.size() ||
           !(r.mass>0 && r.ratio>0 && r.cut>0 && r.excitation>0 && r.e0>0 && r.step_fraction>0 && r.step_fraction<=1 && r.final_range>0 && r.linear_limit>0 && r.linear_limit<=1))
            throw std::runtime_error("Invalid unified EM record");
        float e=0;
        for(std::uint32_t j=0;j<r.node_count;++j){const auto& n=p.nodes[r.node_offset+j];
            for(float value:std::array{n.energy,n.full,n.restricted,n.stopping,n.range,n.lambda,n.factor,n.correction,n.dispersion,n.universal_dispersion,n.q2,n.model,n.fluctuation})
                if(!std::isfinite(value))throw std::runtime_error("Non-finite unified EM node");
            if(!std::isfinite(n.energy) || n.energy<=e || !std::isfinite(n.correction) || n.stopping<0 || n.range<0 || n.lambda<0 || n.factor<=0 || n.dispersion<0 || n.universal_dispersion<0 || (n.fluctuation!=0 && n.fluctuation!=1) || n.model<0 || n.model>3)
                throw std::runtime_error("Invalid unified EM node");
            e=n.energy;
        }
        for(int k=0;k<4;++k){
            if(!r.counts[k] || std::uint64_t(r.offsets[k])+r.counts[k]>p.segments.size())throw std::runtime_error("Invalid EM segment bounds");
            float end=-1;
            for(std::uint32_t j=0;j<r.counts[k];++j){const auto& v=p.segments[r.offsets[k]+j];
                if(!(v.upper>v.lower) || v.lower<end || !std::isfinite(v.y0) || !std::isfinite(v.ythird) || !std::isfinite(v.ytwothirds) || !std::isfinite(v.y1))throw std::runtime_error("Invalid EM native spline");
                end=v.upper;
            }
        }
    }
    return p;
}
} // namespace carbon
