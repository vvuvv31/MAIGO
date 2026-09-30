#pragma once
#include "carbon/electron_joint_response.hpp"

namespace carbon {
// Only sampled joint endpoints enter this queue. No parent transport state,
// energy-lineage cursor or mutable RNG state crosses the kernel boundary.
struct FeJointReplayTask {
    std::array<float,3> birth{},direction{};
    float phi{},cdf_u{},packet_MeV{};
    std::uint32_t channel{},source_voxel{},source_bin{};
};
static_assert(sizeof(FeJointReplayTask)==48);

struct FeJointChannelProbe {
    ElectronJointStatus status{ElectronJointStatus::invalid_payload};
    std::uint32_t channel{};
    double fraction{};
};

// Same domain/payload decisions as sample_electron_joint_device. Preserve
// per-query domain checks even for roulette misses; defer only the CDF/path.
template<std::size_t N>
inline FeJointChannelProbe probe_fe_joint_channel(unsigned section,double energy,double u,
    const ElectronJointChannel* channels,const ElectronJointSample* samples,
    const std::array<double,N>& minimum,const std::array<double,N>& maximum,double ceiling) {
    static_assert(N==2 || N==kElectronJointSections);
    FeJointChannelProbe out;
    if((N==2 && section!=0 && section!=8) || section>=kElectronJointSections) {
        out.status=ElectronJointStatus::unsupported_section;return out;
    }
    const int material=N==2 ? section==8 : section;
    if(!std::isfinite(minimum[material])) {
        out.status=ElectronJointStatus::unsupported_section;return out;
    }
    if(!std::isfinite(energy) || energy<minimum[material] || energy>maximum[material]) {
        out.status=ElectronJointStatus::energy_domain;return out;
    }
    if(!channels || !samples || !std::isfinite(u) || u<0 || u>1)return out;
    constexpr int stride=N==2 ? 37 : kElectronJointEnergyBins;
    if(energy<0 || energy>ceiling)return out;
    const int eb=energy==ceiling ? static_cast<int>(std::ceil(ceiling/5))-1 : static_cast<int>(energy/5);
    if(eb<0 || eb>=stride)return out;
    out.channel=material*stride+eb;
    const auto ch=channels[out.channel];
    if(!ch.count)return out;
    out.status=ElectronJointStatus::hit;out.fraction=ch.fraction;
    return out;
}

inline std::uint32_t select_fe_joint_sample(const ElectronJointChannel& ch,double u,
    const ElectronJointSample* samples) {
    std::uint32_t lo=0,hi=ch.count-1;
    while(lo<hi) {
        const auto mid=lo+(hi-lo)/2;
        if(u<samples[ch.offset+mid].cdf)hi=mid;else lo=mid+1;
    }
    return ch.offset+lo;
}
} // namespace carbon
