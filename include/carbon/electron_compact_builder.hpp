#pragma once
#include "carbon/electron_compact_state.hpp"
#include <bit>
#include <unordered_map>
#include <vector>
#include <stdexcept>
namespace carbon {
template<std::size_t N> struct ElectronWordHash {
    std::size_t operator()(const std::array<std::uint64_t,N>& a)const {
        std::uint64_t h=0xcbf29ce484222325ULL;
        for(auto x:a) { h^=x;h*=0x100000001b3ULL;h^=h>>32; }
        return h;
    }
};
struct ElectronCompactBuilder {
    std::vector<ElectronCompactRecord> records;
    std::vector<ElectronCompactTrack> tracks;
    std::vector<ElectronCompactMedium> media;
    std::unordered_map<std::array<std::uint64_t,8>,std::uint32_t,ElectronWordHash<8>> track_ids;
    std::unordered_map<std::array<std::uint64_t,4>,std::uint32_t,ElectronWordHash<4>> medium_ids;
    template<class T,class Map> static std::uint32_t intern(const T& t,Map& map,std::vector<T>& values) {
        auto key=std::bit_cast<std::array<std::uint64_t,sizeof(T)/8>>(t);
        if(auto it=map.find(key);it!=map.end())return it->second;
        if(values.size()>=UINT32_MAX)throw std::runtime_error("Compact field dictionary exceeds uint32 range");
        const auto id=static_cast<std::uint32_t>(values.size());map.emplace(key,id);values.push_back(t);return id;
    }
    void add(const ElectronContinuationStep& s) {
        if(!s.valid || (s.pdg!=11 && s.pdg!=22))throw std::runtime_error("Compact state requires validated EM row");
        ElectronCompactTrack t{s.parent_direction,s.birth_energy_MeV,s.run,s.event,s.track,s.parent,s.pdg,s.parent_valid,s.parent_step,0};
        ElectronCompactMedium m{s.pre_density_g_cm3,s.post_density_g_cm3,s.pre_material,s.post_material,s.post_step_status,s.track_status};
        records.push_back({s.pre_position,s.post_position,s.pre_direction,s.post_direction,
                          s.pre_energy_MeV,s.post_energy_MeV,s.deposited_MeV,s.length_mm,
                          intern(t,track_ids,tracks),intern(m,medium_ids,media),s.step,0});
    }
    ElectronLegacyCompactStateView view()const{return {records.data(),tracks.data(),media.data(),records.size(),tracks.size(),media.size()};}
    std::size_t bytes()const{return records.size()*sizeof(ElectronCompactRecord)+tracks.size()*sizeof(ElectronCompactTrack)+media.size()*sizeof(ElectronCompactMedium);}
    void release_indices(){decltype(track_ids){}.swap(track_ids);decltype(medium_ids){}.swap(medium_ids);}
};
inline bool electron_state_fields_equal(const ElectronContinuationStep& a,const ElectronContinuationStep& b) {
    auto real=[](double x,double y){return std::bit_cast<std::uint64_t>(x)==std::bit_cast<std::uint64_t>(y);};
    if(a.valid!=b.valid || a.run!=b.run || a.event!=b.event || a.track!=b.track || a.parent!=b.parent || a.pdg!=b.pdg || a.step!=b.step || a.parent_valid!=b.parent_valid || a.parent_step!=b.parent_step || a.pre_material!=b.pre_material || a.post_material!=b.post_material || a.post_step_status!=b.post_step_status || a.track_status!=b.track_status)return false;
    for(unsigned i=0;i<3;++i)if(!real(a.pre_position[i],b.pre_position[i]) || !real(a.post_position[i],b.post_position[i]) || !real(a.pre_direction[i],b.pre_direction[i]) || !real(a.post_direction[i],b.post_direction[i]) || !real(a.parent_direction[i],b.parent_direction[i]))return false;
    return real(a.pre_energy_MeV,b.pre_energy_MeV) && real(a.post_energy_MeV,b.post_energy_MeV) && real(a.deposited_MeV,b.deposited_MeV) && real(a.length_mm,b.length_mm) && real(a.birth_energy_MeV,b.birth_energy_MeV) && real(a.pre_density_g_cm3,b.pre_density_g_cm3) && real(a.post_density_g_cm3,b.post_density_g_cm3);
}
}
