#pragma once
#include "carbon/electron_continuation_sampling.hpp"
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
namespace carbon {
// Same ordered proposals as ElectronEnergyCrossings; adjacent energy bins with
// exactly the same row membership share storage. Boundary events certify
// equality without a probabilistic hash or material/energy approximation.
struct ElectronCompactCrossings {
    std::vector<double> edges;
    std::vector<std::uint64_t> offsets,starts;
    std::vector<std::uint32_t> rows;
    std::size_t unique_bins{};
    static ElectronCompactCrossings build(const ElectronContinuationView& source,std::size_t byte_budget) {
        ElectronCompactCrossings out;out.edges.push_back(0);
        for(unsigned i=0;i<=256;++i)out.edges.push_back(std::exp(std::log(1e-9)+i*(std::log(32.)-std::log(1e-9))/256));
        out.edges.back()=32.;const auto bins=out.edges.size()-1;
        const auto n=source.bytes/kElectronStateV4RecordBytes;
        if(n>std::uint64_t{UINT32_MAX}+1)throw std::runtime_error("Compact crossing source exceeds uint32 range");
        struct Span {std::uint32_t row;std::uint16_t first,last;};
        std::vector<Span> spans;
        std::vector<std::uint64_t> counts(bins),events(bins);
        for(std::uint64_t row=0;row<n;++row) {
            const auto step=source.read(row);
            if(!step.valid)throw std::runtime_error("Invalid compact crossing source");
            if(step.pdg!=11 || step.deposited_MeV==0)continue;
            if(prepare_electron_replay_step(source,row).status!=ElectronReplayStatus::ready)
                throw std::runtime_error("Unclosed compact crossing source");
            const auto upper=step.pre_energy_MeV,lower=upper-step.deposited_MeV;
            if(!(upper>0) || upper>out.edges.back() || lower< -1e-8)
                throw std::runtime_error("Compact crossing energy outside domain");
            const auto bucket=[&](double e){return std::min(bins-1,std::size_t(std::upper_bound(out.edges.begin(),out.edges.end(),e)-out.edges.begin()-1));};
            const auto first=bucket(std::max(0.,lower)),last=bucket(upper);
            spans.push_back({static_cast<std::uint32_t>(row),static_cast<std::uint16_t>(first),static_cast<std::uint16_t>(last)});
            ++events[first];if(last+1<bins)++events[last+1];
            for(auto b=first;b<=last;++b)++counts[b];
        }
        std::vector<std::size_t> group(bins);std::vector<std::uint64_t> group_offsets{0};
        for(std::size_t b=0;b<bins;++b) {
            if(b && events[b])group_offsets.push_back(group_offsets.back()+counts[b-1]);
            group[b]=group_offsets.size()-1;
        }
        group_offsets.push_back(group_offsets.back()+counts.back());out.unique_bins=group_offsets.size()-1;
        out.offsets.assign(bins+1,0);out.starts.resize(bins);
        for(std::size_t b=0;b<bins;++b) {out.offsets[b+1]=out.offsets[b]+counts[b];out.starts[b]=group_offsets[group[b]];}
        const auto fixed=out.edges.size()*8+out.offsets.size()*8+out.starts.size()*8;
        if(fixed>byte_budget || group_offsets.back()>(byte_budget-fixed)/4)
            throw std::runtime_error("Compact crossing rows exceed budget");
        out.rows.resize(group_offsets.back());auto cursor=group_offsets;
        for(const auto& span:spans)for(auto g=group[span.first];g<=group[span.last];++g)out.rows[cursor[g]++]=span.row;
        for(std::size_t g=0;g<out.unique_bins;++g)if(cursor[g]!=group_offsets[g+1])throw std::logic_error("Compact crossing count mismatch");
        return out;
    }
};
}
