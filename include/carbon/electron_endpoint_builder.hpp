#pragma once
#include "carbon/electron_compact_builder.hpp"
namespace carbon {
struct ElectronEndpointBuilder {
    std::vector<ElectronEndpointRecord> records;
    std::vector<ElectronEndpoint> endpoints;
    std::size_t shared{};
    static bool equal(const ElectronEndpoint& a,const ElectronEndpoint& b) {
        return std::bit_cast<std::array<std::uint64_t,7>>(a)==std::bit_cast<std::array<std::uint64_t,7>>(b);
    }
    static ElectronEndpointBuilder build(const std::vector<ElectronCompactRecord>& source,
                                          const std::vector<std::uint32_t>& next) {
        if(source.empty() || source.size()>=UINT32_MAX || next.size()!=source.size())
            throw std::runtime_error("Endpoint source dimensions invalid");
        ElectronEndpointBuilder out;out.records.reserve(source.size());
        for(std::size_t i=0;i<source.size();++i) {
            const auto& r=source[i];auto post=next[i];
            if(r.reserved || (post!=UINT32_MAX && post>=source.size()))throw std::runtime_error("Endpoint source/link invalid");
            const ElectronEndpoint e{r.post_position,r.post_direction,r.post_energy_MeV};
            if(post!=UINT32_MAX && equal(e,{source[post].pre_position,source[post].pre_direction,source[post].pre_energy_MeV}))++out.shared;
            else { // Nonidentical continuations remain explicit: no tolerance or inferred endpoint.
                if(source.size()+out.endpoints.size()>=UINT32_MAX)throw std::runtime_error("Endpoint dictionary exceeds uint32 domain");
                post=static_cast<std::uint32_t>(source.size()+out.endpoints.size());out.endpoints.push_back(e);
            }
            out.records.push_back({{r.pre_position,r.pre_direction,r.pre_energy_MeV},r.deposited_MeV,r.length_mm,r.track,r.medium,r.step,0,post,0});
        }
        return out;
    }
};
}
