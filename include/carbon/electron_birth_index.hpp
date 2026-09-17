#pragma once
#include "carbon/electron_birth_sampling.hpp"
#include "carbon/electron_continuation_replay.hpp"
#include "carbon/water_electron_response.hpp"
#include <stdexcept>

namespace carbon {
// Derived exclusively from pinned raw primary steps and exact child links.
// Each root occurs once, weighted by KE at birth, NOT by its deposited energy
// or number of subsequent steps. No package yield/energy modification.
class ElectronBirthIndex {
public:
    explicit ElectronBirthIndex(const std::vector<WaterElectronChannel>& channels):reference_(channels),
        roots_(channels.size()),loss_(channels.size()),local_(channels.size()),birth_(channels.size()) {}
    void add(const ElectronContinuationView& source,std::uint32_t source_id) {
        const auto n=source.bytes/kElectronStateV4RecordBytes;
        if(!source.payload || !source.child_offsets || (source.child_count && !source.child_rows))
            throw std::invalid_argument("Missing birth source genealogy");
        for(std::uint64_t row=0;row<n;++row) {
            const auto parent=read_electron_continuation_step(source.payload,source.bytes,row);
            if(!parent.valid)throw std::invalid_argument("Invalid primary birth source");
            if(parent.parent!=0)continue;
            if(parent.pdg!=1000060120)throw std::invalid_argument("Non-C12 electron birth source");
            const auto bin=energy_bin(parent.pre_energy_MeV/12.);
            const auto loss=parent.pre_energy_MeV-parent.post_energy_MeV;
            if(loss<0)throw std::invalid_argument("Negative primary energy loss");
            loss_[bin]+=loss;local_[bin]+=parent.deposited_MeV;
            const auto begin=source.child_offsets[row],end=source.child_offsets[row+1];
            if(begin>end || end>source.child_count)throw std::invalid_argument("Invalid primary child range");
            for(auto i=begin;i<end;++i) {
                const auto child_row=source.child_rows[i];
                const auto child=read_electron_continuation_step(source.payload,source.bytes,child_row);
                if(!generated_by_electron_step(parent,child) || child.pdg!=11 ||
                   std::abs(child.birth_energy_MeV-child.pre_energy_MeV)>1e-8)
                    throw std::invalid_argument("Unsupported/mismatched primary electron root");
                if(child.birth_energy_MeV==0)continue;
                birth_[bin]+=child.birth_energy_MeV;
                roots_[bin].push_back({child.birth_energy_MeV,child_row,source_id});
            }
        }
    }
    void finish(std::size_t byte_budget) {
        if(!channels.empty())throw std::logic_error("Birth index already finalized");
        std::uint64_t count=0;for(const auto& roots:roots_)count+=roots.size();
        const auto fixed=reference_.size()*sizeof(ElectronBirthChannel);
        if(fixed>byte_budget || count>(byte_budget-fixed)/sizeof(ElectronBirthSample))
            throw std::runtime_error("Electron birth index exceeds device budget");
        samples.reserve(count);
        for(std::size_t b=0;b<reference_.size();++b) {
            if(!(loss_[b]>0))throw std::invalid_argument("Missing birth energy exposure");
            const auto fraction=birth_[b]/loss_[b];
            const auto closure=(local_[b]+birth_[b]-loss_[b])/loss_[b];
            if(fraction<0 || fraction>1 || std::abs(closure)>(b==0?1e-3:1e-7) ||
               std::abs(fraction-reference_[b].fraction-reference_[b].unresolved)>1e-7)
                throw std::invalid_argument("Raw birth energy does not match pinned response fractions");
            channels.push_back({reference_[b].low,reference_[b].high,fraction,samples.size(),roots_[b].size()});
            double sum=0;
            for(auto root:roots_[b]) {sum+=root.cdf;root.cdf=sum/birth_[b];samples.push_back(root);}
            if(!roots_[b].empty())samples.back().cdf=1.;
        }
    }
    std::vector<ElectronBirthChannel> channels;
    std::vector<ElectronBirthSample> samples;
private:
    std::size_t energy_bin(double energy) const {
        if(reference_.empty() || !std::isfinite(energy) || energy<reference_.front().low || energy>reference_.back().high)
            throw std::invalid_argument("Primary birth energy outside response domain");
        std::size_t lo=0,hi=reference_.size();
        while(lo+1<hi) {const auto mid=lo+(hi-lo)/2;if(reference_[mid].low<=energy)lo=mid;else hi=mid;}
        if(energy>reference_[lo].high)throw std::invalid_argument("Gap in primary birth energy domain");
        return lo;
    }
    std::vector<WaterElectronChannel> reference_;
    std::vector<std::vector<ElectronBirthSample>> roots_;
    std::vector<double> loss_,local_,birth_;
};
} // namespace carbon
