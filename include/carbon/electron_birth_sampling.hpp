#pragma once
#include "carbon/electron_state_refs.hpp"
#include <cmath>

namespace carbon {
struct ElectronBirthChannel {double low{},high{},fraction{};std::uint64_t offset{},count{};};
struct ElectronBirthSample {double cdf{};std::uint64_t row{};std::uint32_t source{};};
struct ElectronBirthDraw {bool valid{};double fraction{};ElectronStateReference reference{};};
inline ElectronBirthDraw sample_electron_birth(double carbon_energy_MeVu,double u,
    const ElectronBirthChannel* channels,std::size_t channel_count,
    const ElectronBirthSample* samples,std::uint64_t sample_count) {
    ElectronBirthDraw out;
    if(!channels || !channel_count || !std::isfinite(carbon_energy_MeVu) ||
       !std::isfinite(u) || u<0 || u>=1 || carbon_energy_MeVu<channels[0].low ||
       carbon_energy_MeVu>channels[channel_count-1].high)return out;
    std::size_t lo=0,hi=channel_count;
    while(lo+1<hi) {const auto mid=lo+(hi-lo)/2;if(channels[mid].low<=carbon_energy_MeVu)lo=mid;else hi=mid;}
    const auto c=channels[lo];
    if(carbon_energy_MeVu>c.high || !std::isfinite(c.fraction) || c.fraction<0 || c.fraction>1 ||
       c.offset>sample_count || c.count>sample_count-c.offset)return out;
    out.fraction=c.fraction;
    if(!c.count) {out.valid=c.fraction==0;return out;}
    if(!samples || !(c.fraction>0))return out;
    std::uint64_t first=c.offset,last=first+c.count-1;
    while(first<last) {const auto mid=first+(last-first)/2;if(u<samples[mid].cdf)last=mid;else first=mid+1;}
    const auto selected=samples[first];
    out.reference.valid=true;out.reference.source=selected.source;out.reference.row=selected.row;
    out.valid=true;return out;
}
} // namespace carbon
