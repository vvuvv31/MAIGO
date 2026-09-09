#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace carbon {
struct ElectronStateReference {
    bool valid{};
    std::uint32_t source{};
    std::uint64_t row{},incoming_edge{std::numeric_limits<std::uint64_t>::max()};
    bool post{};
};
// Views only: the host must SHA-verify the buffer and source manifest before
// upload. Decode packed little-endian integers explicitly on host/device.
struct ElectronStateReferenceView {
    const unsigned char* data{};
    std::size_t bytes{},sample_offset{},node_offset{};
    std::uint32_t sources{};
    std::uint64_t samples{},nodes{};
    bool valid{};
    static std::uint64_t integer(const unsigned char* p,unsigned n) {
        std::uint64_t v=0;
        for(unsigned i=0;i<n;++i)v|=std::uint64_t(p[i])<<(8*i);
        return v;
    }
    static ElectronStateReferenceView parse(const unsigned char* p,std::size_t size) {
        ElectronStateReferenceView v;
        if(!p || size<32)return v;
        constexpr char magic[]="ESTREF01";
        for(unsigned i=0;i<8;++i)if(p[i]!=magic[i])return v;
        if(integer(p+8,4)!=1)return v;
        v.sources=static_cast<std::uint32_t>(integer(p+12,4));
        v.samples=integer(p+16,8);v.nodes=integer(p+24,8);
        if(!v.sources || !v.samples || !v.nodes || v.sources>(size-32)/8)return v;
        v.sample_offset=32+std::size_t(v.sources)*8;
        if(v.samples>(size-v.sample_offset)/12)return v;
        v.node_offset=v.sample_offset+std::size_t(v.samples)*12;
        if(v.nodes>(size-v.node_offset)/24 || v.nodes*24!=size-v.node_offset)return v;
        v.data=p;v.bytes=size;v.valid=true;
        return v;
    }
    ElectronStateReference at(std::uint64_t i,bool node) const {
        ElectronStateReference r;
        if(!valid || i>=(node?nodes:samples))return r;
        const auto* p=data+(node?node_offset:sample_offset)+std::size_t(i)*(node?24:12);
        r.source=static_cast<std::uint32_t>(integer(p,4));r.row=integer(p+4,8);
        if(r.source>=sources)return r;
        const auto limit=integer(data+32+std::size_t(r.source)*8,8);
        if(r.row>=limit)return r;
        if(node) {
            r.incoming_edge=integer(p+12,8);
            if(p[20]>1 || p[21] || p[22] || p[23] ||
               (r.incoming_edge!=std::numeric_limits<std::uint64_t>::max() && r.incoming_edge>=limit))return r;
            r.post=p[20]!=0;
        }
        r.valid=true;return r;
    }
};
} // namespace carbon
