#include "carbon/urban_mcs_package.hpp"
#include "carbon/sha256.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace carbon {
UrbanMcsPackage UrbanMcsPackage::load(const std::filesystem::path& path,const std::string& pin) {
    if(pin.size()!=64 || compute_file_sha256_hex(path)!=pin)
        throw std::runtime_error("Urban MSC package SHA256 mismatch");
    std::ifstream stream(path,std::ios::binary);
    char magic[8];std::array<std::uint32_t,7> header{};
    stream.read(magic,8);stream.read(reinterpret_cast<char*>(header.data()),28);
    const auto [version,nm,ns,nn,nf,ng,validated]=header;
    if(!stream || std::memcmp(magic,"URBANV22",8) || version!=2 || validated!=1 ||
       nm==0 || nm>2048 || ns==0 || ns>200 || nn<2 || nf<2 || ng<3)
        throw std::runtime_error("Invalid or unvalidated Urban MSC package header");
    const std::uint64_t nr=std::uint64_t(nm)*ns;
    const std::uint64_t bytes=36+std::uint64_t(nm)*20+std::uint64_t(ns)*12+nr*48+
        std::uint64_t(nn)*8+std::uint64_t(nf)*8+std::uint64_t(ng)*24;
    if(bytes!=std::filesystem::file_size(path) || bytes>4ULL*1024*1024*1024)
        throw std::runtime_error("Urban MSC package size is invalid");
    UrbanMcsPackage result;
    result.materials.resize(nm);result.species.resize(ns);result.records.resize(nr);
    result.nodes.resize(nn);result.mfps.resize(nf);result.segments.resize(ng);
    auto read=[&](auto& v){stream.read(reinterpret_cast<char*>(v.data()),
        static_cast<std::streamsize>(v.size()*sizeof(v[0])));};
    read(result.materials);read(result.species);read(result.records);
    read(result.nodes);read(result.mfps);read(result.segments);
    if(!stream)throw std::runtime_error("Truncated Urban MSC package");
    std::set<std::pair<int,float>> material_keys;
    for(const auto& m:result.materials)
        if(m.section< -2 || m.section>24 || !(m.density>0 && m.zeff>0 &&
           m.radiation_length_mm>0 && m.production_cut_mm==.05f) ||
           !std::isfinite(m.density+m.zeff+m.radiation_length_mm) ||
           !material_keys.emplace(m.section,m.density).second)
            throw std::runtime_error("Invalid or duplicate Urban material couple");
    std::set<std::pair<unsigned,unsigned>> ions;
    for(const auto& ion:result.species)
        if(ion.z==0 || ion.a<ion.z || ion.a>250 || !(ion.mass_mev>0) ||
           !std::isfinite(ion.mass_mev) || !ions.emplace(ion.z,ion.a).second)
            throw std::runtime_error("Invalid or duplicate Urban projectile");
    auto check_nodes=[](const auto& v,unsigned start,unsigned n) {
        if(n<2 || std::uint64_t(start)+n>v.size())throw std::runtime_error("Invalid Urban node bounds");
        for(unsigned j=0;j<n;++j) {
            const auto& x=v[start+j];
            if(!(x.energy>0 && x.value>0) || !std::isfinite(x.energy+x.value) ||
               (j && !(x.energy>v[start+j-1].energy)))
                throw std::runtime_error("Invalid Urban factor/MFP nodes");
        }
    };
    for(const auto& r:result.records) {
        if(!(r.mass_ratio>0 && r.minimum_scaled_energy>0) ||
           !std::isfinite(r.mass_ratio+r.minimum_scaled_energy))
            throw std::runtime_error("Invalid Urban mass scaling");
        check_nodes(result.nodes,r.node_offset,r.node_count);
        check_nodes(result.mfps,r.mfp_offset,r.mfp_count);
        for(unsigned k=0;k<3;++k) {
            if(!r.counts[k] || std::uint64_t(r.offsets[k])+r.counts[k]>ng)
                throw std::runtime_error("Invalid Urban spline bounds");
            for(unsigned j=0;j<r.counts[k];++j) {
                const auto& s=result.segments[r.offsets[k]+j];
                if(!(s.lower>0 && s.upper>s.lower && s.y0>0 && s.ythird>0 &&
                     s.ytwothirds>0 && s.y1>0) ||
                   !std::isfinite(s.lower+s.upper+s.y0+s.ythird+s.ytwothirds+s.y1) ||
                   (j && s.lower!=result.segments[r.offsets[k]+j-1].upper))
                    throw std::runtime_error("Invalid Urban native spline interval");
            }
        }
    }
    return result;
}
} // namespace carbon
