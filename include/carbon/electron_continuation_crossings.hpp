#pragma once
#include "carbon/electron_continuation_sampling.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace carbon {
// Host index built from the same verified source/links uploaded to the device.
struct ElectronEnergyCrossings {
    std::vector<double> edges;
    std::vector<std::uint64_t> offsets,rows;
    static ElectronEnergyCrossings build(const ElectronContinuationView& source,std::size_t byte_budget) {
        ElectronEnergyCrossings out;
        out.edges.push_back(0);
        for(unsigned i=0;i<=256;++i)out.edges.push_back(std::exp(std::log(1e-9)+i*(std::log(32.)-std::log(1e-9))/256));
        out.edges.back()=32.;
        const auto bins=out.edges.size()-1;
        out.offsets.assign(bins+1,0);
        const auto fixed=out.edges.size()*sizeof(double)+out.offsets.size()*sizeof(std::uint64_t);
        if(fixed>byte_budget)throw std::runtime_error("Electron crossing index exceeds budget");
        const auto n=source.bytes/kElectronStateV4RecordBytes;
        struct Span {std::uint64_t row;std::size_t first,last;};
        std::vector<Span> spans;
        for(std::uint64_t row=0;row<n;++row) {
            const auto step=read_electron_continuation_step(source.payload,source.bytes,row);
            if(!step.valid)throw std::invalid_argument("Invalid electron crossing source");
            if(step.pdg!=11 || step.deposited_MeV==0)continue;
            const auto prepared=prepare_electron_replay_step(source,row);
            if(prepared.status!=ElectronReplayStatus::ready)throw std::invalid_argument("Unclosed electron crossing source");
            const auto upper=step.pre_energy_MeV,lower=upper-step.deposited_MeV;
            if(!(upper>0) || upper>out.edges.back() || lower< -1e-8)
                throw std::invalid_argument("Electron crossing energy outside compiled domain");
            const auto bucket=[&](double value) {
                return std::min(bins-1,std::size_t(std::upper_bound(out.edges.begin(),out.edges.end(),value)-out.edges.begin()-1));
            };
            const auto first=bucket(std::max(0.,lower)),last=bucket(upper);
            spans.push_back({row,first,last});
            for(auto i=first;i<=last;++i)++out.offsets[i+1];
        }
        for(std::size_t i=1;i<=bins;++i)out.offsets[i]+=out.offsets[i-1];
        if(out.offsets.back()>(byte_budget-fixed)/8)throw std::runtime_error("Electron crossing rows exceed device budget");
        out.rows.resize(out.offsets.back());
        auto cursor=out.offsets;
        for(const auto& span:spans)for(auto bin=span.first;bin<=span.last;++bin)out.rows[cursor[bin]++]=span.row;
        return out;
    }
};
} // namespace carbon
