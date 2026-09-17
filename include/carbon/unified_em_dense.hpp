#pragma once
// Independent research attachment, never a replacement physics package.
#include "carbon/unified_em_view.hpp"
#include "carbon/delta_moments_data.hpp"
#include "carbon/sha256.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
namespace carbon {
struct UnifiedEmDenseTable {
    std::vector<UnifiedEmDenseNode> nodes;
    std::vector<unsigned char> valid;
    double max_value_error{},max_slope_error{};
    std::size_t accepted{};
};
inline UnifiedEmDenseTable build_unified_em_dense(const UnifiedEmPackage& p) {
    UnifiedEmDenseTable out;out.nodes.resize(p.nodes.size());out.valid.resize(p.nodes.size());
    for(const auto& r:p.records) {
        UnifiedEmPoint point{&r,p.nodes.data()+r.node_offset,p.segments.data(),1};
        for(unsigned i=0;i<r.node_count;++i) {
            float x=(point.nodes[i].energy*r.a)*r.ratio;
            auto& d=out.nodes[r.node_offset+i];
            d.stopping=point.raw(0,x,&d.stopping_slope);d.range=point.raw(1,x,&d.range_slope);
        }
        for(unsigned i=0;i+1<r.node_count;++i) {
            const auto& n=point.nodes[i];const auto& next=point.nodes[i+1];
            float x0=(n.energy*r.a)*r.ratio,x1=(next.energy*r.a)*r.ratio,h=x1-x0;
            unsigned ci=point.locate_cubic(0,x0);
            const auto& seg=p.segments[r.offsets[0]+ci];
            if(n.model!=next.model || n.fluctuation!=next.fluctuation || !(h>0) || x0<seg.lower || x1>seg.upper || point.locate_cubic(0,x1)!=ci || point.locate_cubic(1,x0)!=ci || point.locate_cubic(1,x1)!=ci)continue;
            // Do not interpolate across known mean-loss model switches.
            const float threshold=2.f*r.mass/938.272013f;
            if(n.energy*r.a<=threshold && next.energy*r.a>=threshold)continue;
            const auto a=out.nodes[r.node_offset+i],b=out.nodes[r.node_offset+i+1];
            bool ok=true;double ve=0,se=0;
            for(unsigned k=0;k<=8;++k) {
                float x=x0+h*(k/8.f),u=(x-x0)/h;
                for(unsigned kind=0;kind<2;++kind) {
                    float ds=0,refds=0;
                    float y=kind?unified_em_hermite(a.range,b.range,a.range_slope,b.range_slope,h,u,&ds):unified_em_hermite(a.stopping,b.stopping,a.stopping_slope,b.stopping_slope,h,u,&ds);
                    float ref=point.raw(kind,x,&refds);
                    if(!(ref>1e-20f) || !std::isfinite(y) || !std::isfinite(ds) || ds*refds<0){ok=false;break;}
                    double err=std::abs((y-ref)/ref);
                    double serr=std::abs(ds-refds)/std::max(std::abs(refds),1e-20f);
                    ve=std::max(ve,err);se=std::max(se,serr);
                    if(err>2e-4 || serr>2e-3)ok=false;
                }
            }
            if(ok){out.valid[r.node_offset+i]=1;++out.accepted;out.max_value_error=std::max(out.max_value_error,ve);out.max_slope_error=std::max(out.max_slope_error,se);}
        }
    }
    return out;
}
inline void save_unified_em_dense(const UnifiedEmDenseTable& t,const std::filesystem::path& path) {
    std::ofstream f(path,std::ios::binary);std::uint64_t n=t.nodes.size();
    f.write("EMDENSE1",8);f.write(delta_moments_source_sha256,64);f.write(reinterpret_cast<const char*>(&n),8);
    f.write(reinterpret_cast<const char*>(t.nodes.data()),n*sizeof(UnifiedEmDenseNode));f.write(reinterpret_cast<const char*>(t.valid.data()),n);
    if(!f)throw std::runtime_error("Cannot write dense EM research attachment");
}
inline UnifiedEmDenseTable load_unified_em_dense(const std::filesystem::path& path,const std::string& sha,std::size_t count) {
    if(sha.size()!=64 || compute_file_sha256_hex(path)!=sha)throw std::runtime_error("Dense EM attachment SHA mismatch");
    if(std::filesystem::file_size(path)!=80+count*17)throw std::runtime_error("Dense EM attachment size mismatch");
    std::ifstream f(path,std::ios::binary);char magic[8],source[64];std::uint64_t n=0;
    f.read(magic,8);f.read(source,64);f.read(reinterpret_cast<char*>(&n),8);
    if(std::memcmp(magic,"EMDENSE1",8) || std::memcmp(source,delta_moments_source_sha256,64) || n!=count)throw std::runtime_error("Dense EM attachment provenance mismatch");
    UnifiedEmDenseTable t;t.nodes.resize(n);t.valid.resize(n);
    f.read(reinterpret_cast<char*>(t.nodes.data()),n*sizeof(UnifiedEmDenseNode));f.read(reinterpret_cast<char*>(t.valid.data()),n);
    if(!f)throw std::runtime_error("Truncated dense EM attachment");
    for(std::size_t i=0;i<n;++i) {
        const auto d=t.nodes[i];
        if(t.valid[i]>1 || !std::isfinite(d.stopping) || !std::isfinite(d.range) || !std::isfinite(d.stopping_slope) || !std::isfinite(d.range_slope))
            throw std::runtime_error("Invalid dense EM research values");
        if(t.valid[i] && (!(d.stopping>0) || !(d.range>0) || i+1==n))
            throw std::runtime_error("Invalid dense EM research eligibility mask");
    }
    return t;
}
} // namespace carbon
