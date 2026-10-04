#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_joint_sampling.hpp"
#include <array>
#include <cstdint>
#include <limits>
namespace carbon {
struct ElectronBirthChannel {double low{},high{},fraction{};std::uint64_t offset{},count{};};
struct ElectronBirthStateSample {double cdf{},energy{},cosine{};};
struct ElectronBirthNodeView {
    int section{-1};double density{};
    const ElectronBirthChannel* channels{};const ElectronBirthStateSample* samples{};
    std::uint64_t sample_count{};unsigned channel_count{};
};
struct ElectronBirthBankView {
    const ElectronBirthNodeView* nodes{};unsigned count{};
    const unsigned* section_offsets{};
};
struct IndependentElectronBirthDraw {
    bool valid{};double fraction{};unsigned node{};
};
inline const ElectronBirthChannel* electron_birth_channel(const ElectronBirthNodeView& n,double energy) {
    if(!n.channels || !n.channel_count || !std::isfinite(energy) || energy<n.channels[0].low || energy>n.channels[n.channel_count-1].high)return nullptr;
    unsigned lo=0,hi=n.channel_count;
    while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(n.channels[mid].low<=energy)lo=mid;else hi=mid;}
    const auto* c=n.channels+lo;return energy<=c->high?c:nullptr;
}
// Birth yield and joint (energy,angle) are separately measured at every density
// node. Only geometry in the electron propagation bank is density-scaled.
inline IndependentElectronBirthDraw prepare_independent_electron_birth(
    ElectronBirthBankView bank,int section,double density,double energy,double mix) {
    IndependentElectronBirthDraw out;
    if(!bank.nodes||!bank.section_offsets||section<0||section>=kElectronMaterialCount||!std::isfinite(density)||density<=0||!std::isfinite(mix)||mix<0||mix>=1)return out;
    const auto first=bank.section_offsets[section],end=bank.section_offsets[section+1];
    if(first>=end||end>bank.count||density<bank.nodes[first].density||density>bank.nodes[end-1].density)return out;
    unsigned hi=first;while(hi+1<end&&bank.nodes[hi].density<density)++hi;
    const auto lo=bank.nodes[hi].density==density?hi:hi-1;
    const auto* a=electron_birth_channel(bank.nodes[lo],energy);const auto* b=electron_birth_channel(bank.nodes[hi],energy);
    if(!a||!b)return out;
    const double blend=lo==hi?0:(density-bank.nodes[lo].density)/(bank.nodes[hi].density-bank.nodes[lo].density);
    const double left=(1-blend)*a->fraction,right=blend*b->fraction,total=left+right;
    if(!std::isfinite(total)||total<0||total>1)return out;
    out.node=(right>0&&(left==0||mix>=left/total))?hi:lo;out.fraction=total;out.valid=true;return out;
}
inline ElectronBirthStateSample sample_independent_electron_birth(const ElectronBirthNodeView& n,double energy,double u) {
    const auto* c=electron_birth_channel(n,energy);
    if(!c||!std::isfinite(u)||u<0||u>=1||!c->count||!n.samples||c->offset>n.sample_count||c->count>n.sample_count-c->offset)return {};
    auto lo=c->offset,hi=lo+c->count-1;
    while(lo<hi){const auto mid=lo+(hi-lo)/2;if(u<n.samples[mid].cdf)hi=mid;else lo=mid+1;}
    return n.samples[lo];
}
inline electron_joint_research::V independent_electron_direction(
    electron_joint_research::V parent,double cosine,double spin) {
    using namespace electron_joint_research;
    const auto frame=azimuth(parent,6.2831853071795864769*spin);
    return std::sqrt(std::max(0.,1-cosine*cosine))*frame.u+cosine*parent;
}
} // namespace carbon
