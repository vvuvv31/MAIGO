#include "unified_em_package.hpp"
#include "unrestricted_csda.hpp"
#include "detail/native_spline_wire.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace carbon {
namespace {
// Disk-only schema. Keep strict validation of the existing native package,
// then retain only fields used by unrestricted transport. None of these wire
// types, restricted splines or rate coefficients are exposed to the GPU.
struct FileRecord {
    float mass,ratio,cut,excitation,e0,spin,step_fraction,final_range,linear_limit;
    float is_ion,form_factor,magnetic_moment2,lowest_kinetic,peak;
    std::uint32_t node_offset,node_count,offsets[4],counts[4],z,a;
};
struct FileNode {
    float energy,full,restricted,stopping,range,lambda,factor,correction;
    float dispersion,universal_dispersion,q2,model,fluctuation;
};
struct FileContinuousDescriptor {
    float log_min,log_max,base_density,base_cut,base_coefficient;
    float base_c,base_m,base_a,base_x0,base_x1,base_d0,rate_max_energy;
    std::uint32_t range_offset,range_count,rate_offset,rate_count;
};
static_assert(sizeof(FileRecord)==104 && sizeof(FileNode)==52 &&
              sizeof(FileContinuousDescriptor)==64);
UnifiedEmPackage parse_unified_em(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    char magic[8]{};std::uint32_t h[6]{};in.read(magic,8);in.read(reinterpret_cast<char*>(h),sizeof h);
    if(!in || std::memcmp(magic,"EMCNT001",8)!=0 || h[0]!=1 || h[1]!=26 || h[2]!=18 || h[3]!=h[1]*h[2])
        throw std::runtime_error("Invalid unified EM header/registry");
    const std::uint64_t expected=32ULL+8ULL*h[1]+8ULL*h[2]+104ULL*h[3]+52ULL*h[4]+24ULL*h[5];
    if(expected>4ULL*1024*1024*1024 ||
       std::filesystem::file_size(path)>4ULL*1024*1024*1024)
        throw std::runtime_error("Unified EM size mismatch");
    UnifiedEmPackage p;
    std::vector<FileRecord> records;
    std::vector<FileNode> nodes;
    std::vector<FileContinuousDescriptor> descriptors;
    std::vector<detail::NativeSplineSegment> segments;
    std::vector<float> continuous_ranges;
    auto read=[&](auto& v,std::uint32_t count){v.resize(count);in.read(reinterpret_cast<char*>(v.data()),count*sizeof(v[0]));if(!in)throw std::runtime_error("Truncated unified EM payload");};
    read(p.materials,h[1]);read(p.species,h[2]);read(records,h[3]);
    read(nodes,h[4]);read(segments,h[5]);
    {
        if(h[1]!=26)throw std::runtime_error("Continuous EM must have exactly26 compositions");
        unsigned ext[4]{};in.read(reinterpret_cast<char*>(ext),sizeof ext);
        if(!in || ext[0]<2 || ext[0]>64 || ext[1]<2 || ext[1]>4096 || ext[2]!=468 || ext[3]>1000000)
            throw std::runtime_error("Invalid continuous EM extension");
        p.continuous_scalar_count=ext[1];
        const auto bytes=expected+16+64ULL*ext[2]+26ULL*ext[1]*8*4+4ULL*ext[3]*(ext[0]+1);
        if(std::filesystem::file_size(path)!=bytes)throw std::runtime_error("Continuous EM size mismatch");
        read(descriptors,ext[2]);read(p.continuous_scalars,26*ext[1]*8);read(continuous_ranges,ext[3]*(ext[0]+1));
        std::uint64_t next=0;
        for(unsigned i=0;i<468;++i){const auto& d=descriptors[i];
            for(float v:std::array{d.log_min,d.log_max,d.base_density,d.base_cut,d.base_coefficient,d.base_c,d.base_m,d.base_a,d.base_x0,d.base_x1,d.base_d0,d.rate_max_energy})
                if(!std::isfinite(v))throw std::runtime_error("Invalid continuous descriptor scalar");
            if(d.log_max<d.log_min || !(d.base_density>0 && d.base_cut>0 && d.base_coefficient>0) ||
               d.base_density!=p.materials[i/18].density || d.range_offset!=next || d.range_count<2 || next+d.range_count>ext[3])
                throw std::runtime_error("Invalid continuous density domain/range directory");
            float energy=0;
            for(unsigned j=0;j<d.range_count;++j){const float e=continuous_ranges[(next+j)*(ext[0]+1)];
                if(!(e>energy))throw std::runtime_error("Continuous range knots not ordered");energy=e;}
            if(d.rate_offset!=next+d.range_count || d.rate_count<2 || d.rate_offset+d.rate_count>ext[3])
                throw std::runtime_error("Invalid continuous rate directory");
            const unsigned stride=ext[0]+1;
            for(unsigned j=0;j<d.rate_count;++j){const float u=continuous_ranges[(d.rate_offset+j)*stride];
                if(u<0 || u>1 || (j && u<=continuous_ranges[(d.rate_offset+j-1)*stride]))throw std::runtime_error("Invalid normalized rate knot");}
            next+=d.range_count+d.rate_count;
        }
        if(next!=ext[3])throw std::runtime_error("Continuous range coverage mismatch");
        for(unsigned i=0;i<26*ext[1];++i) {
            const float* v=p.continuous_scalars.data()+8*i;
            for(unsigned k=0;k<8;++k)if(!std::isfinite(v[k]))throw std::runtime_error("Nonfinite material scalar");
            if(!(v[0]>0 && v[1]>0 && v[3]>0 && v[6]>v[5] && v[7]>=0))throw std::runtime_error("Invalid cut/density-effect material scalar");
        }
        for(float v:continuous_ranges)if(!std::isfinite(v))throw std::runtime_error("Nonfinite range coefficient");

    }
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
    for(std::size_t i=0;i<records.size();++i){
        const auto& r=records[i];const auto s=p.species[i%18];
        for(float value:std::array{r.mass,r.ratio,r.cut,r.excitation,r.e0,r.spin,r.step_fraction,r.final_range,r.linear_limit,r.is_ion,r.form_factor,r.magnetic_moment2,r.lowest_kinetic,r.peak})
            if(!std::isfinite(value))throw std::runtime_error("Non-finite unified EM record");
        if(r.z!=s.z || r.a!=s.a || r.node_count<2 || std::uint64_t(r.node_offset)+r.node_count>nodes.size() ||
           !(r.mass>0 && r.ratio>0 && r.cut>0 && r.excitation>0 && r.e0>0 && r.step_fraction>0 && r.step_fraction<=1 && r.final_range>0 && r.linear_limit>0 && r.linear_limit<=1))
            throw std::runtime_error("Invalid unified EM record");
        float e=0;
        for(std::uint32_t j=0;j<r.node_count;++j){const auto& n=nodes[r.node_offset+j];
            for(float value:std::array{n.energy,n.full,n.restricted,n.stopping,n.range,n.lambda,n.factor,n.correction,n.dispersion,n.universal_dispersion,n.q2,n.model,n.fluctuation})
                if(!std::isfinite(value))throw std::runtime_error("Non-finite unified EM node");
            if(!std::isfinite(n.energy) || n.energy<=e || !std::isfinite(n.correction) || n.stopping<0 || n.range<0 || n.lambda<0 || n.factor<=0 || n.dispersion<0 || n.universal_dispersion<0 || (n.fluctuation!=0 && n.fluctuation!=1) || n.model<0 || n.model>3)
                throw std::runtime_error("Invalid unified EM node");
            e=n.energy;
        }
        for(int k=0;k<4;++k){
            if(!r.counts[k] || std::uint64_t(r.offsets[k])+r.counts[k]>segments.size())throw std::runtime_error("Invalid EM segment bounds");
            float end=-1;
            for(std::uint32_t j=0;j<r.counts[k];++j){const auto& v=segments[r.offsets[k]+j];
                if(!(v.upper>v.lower) || v.lower<end || !std::isfinite(v.y0) || !std::isfinite(v.ythird) || !std::isfinite(v.ytwothirds) || !std::isfinite(v.y1))throw std::runtime_error("Invalid EM native spline");
                end=v.upper;
            }
        }
    }
    p.continuous_descriptors.reserve(descriptors.size());
    for (const auto& d : descriptors)
        p.continuous_descriptors.push_back({d.log_min,d.log_max,d.base_cut,
            d.base_c,d.base_m,d.base_a,d.base_x0,d.base_x1,d.base_d0});
    p.records.reserve(records.size());
    for (const auto& r : records)
        p.records.push_back({r.mass,r.ratio,r.excitation,r.e0,r.step_fraction,
            r.final_range,r.linear_limit,r.is_ion,r.lowest_kinetic,r.node_offset,r.node_count,r.z,r.a});
    p.nodes.reserve(nodes.size());
    for (const auto& n : nodes)
        p.nodes.push_back({n.energy,n.full,n.factor,n.dispersion,
            n.universal_dispersion,n.fluctuation});
    p.unrestricted_ranges.resize(p.nodes.size());
    for (const auto& r : p.records) {
        const auto* node = p.nodes.data() + r.node_offset;
        // Native low-energy continuation S(E) proportional to sqrt(E).
        // Transport coverage/cutoffs remain unchanged; no missing response is
        // replaced by a local-deposition or density-node fallback.
        if (!(node[0].full > 0))
            throw std::runtime_error("Nonpositive unrestricted electronic stopping");
        double range = 2.0 * node[0].energy * r.a / node[0].full;
        p.unrestricted_ranges[r.node_offset] = static_cast<float>(range);
        for (unsigned j = 1; j < r.node_count; ++j) {
            if (!(node[j].full > 0))
                throw std::runtime_error("Nonpositive unrestricted electronic stopping");
            range += linear_stopping_path(
                (double(node[j].energy) - node[j-1].energy) * r.a,
                double(node[j-1].full), double(node[j].full));
            const float value = static_cast<float>(range);
            if (!std::isfinite(value) || value <= p.unrestricted_ranges[r.node_offset+j-1])
                throw std::runtime_error("Nonmonotone unrestricted CSDA range");
            p.unrestricted_ranges[r.node_offset+j] = value;
        }
    }
    return p;
}
} // namespace

UnifiedEmPackage UnifiedEmPackage::load(const std::filesystem::path& path) {
    if(std::endian::native!=std::endian::little)throw std::runtime_error("EMCNT001 requires little endian");
    return parse_unified_em(path);
}
} // namespace carbon
