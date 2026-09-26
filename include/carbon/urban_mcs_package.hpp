#pragma once
#include "carbon/primary_em_mean_candidate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace carbon {
struct UrbanMcsMaterial {
    std::int32_t section;
    float density, zeff, radiation_length_mm, production_cut_mm;
};
struct UrbanMcsSpecies { std::uint32_t z, a; float mass_mev; };
struct UrbanMcsRecord {
    float mass_ratio, minimum_scaled_energy;
    std::uint32_t node_offset, node_count, mfp_offset, mfp_count;
    std::uint32_t offsets[3], counts[3]; // native DEDX, range, inverse range
};
struct UrbanMcsNode { float energy, value; };
static_assert(sizeof(UrbanMcsMaterial)==20 && sizeof(UrbanMcsSpecies)==12);
static_assert(sizeof(UrbanMcsRecord)==48 && sizeof(UrbanMcsNode)==8);

// One active step, with its dynamic charge/density factor held fixed across
// all range, stopping and inverse queries, as in G4VEnergyLossProcess.
struct UrbanMcsContext {
    const UrbanMcsRecord* record{};
    const UrbanMcsNode* nodes{};
    const UrbanMcsNode* mfps{};
    const EmCubicSegmentCandidate<float>* segments{};
    float factor{};
    static float interpolate(const UrbanMcsNode* v,unsigned n,float e) {
        if(!v || n<2 || e<v[0].energy || e>v[n-1].energy)
            return std::numeric_limits<float>::quiet_NaN();
        unsigned lo=0,hi=n-1;
        while(hi-lo>1) { unsigned mid=(lo+hi)/2; if(v[mid].energy<=e)lo=mid;else hi=mid; }
        float t=(e-v[lo].energy)/(v[hi].energy-v[lo].energy);
        return v[lo].value+t*(v[hi].value-v[lo].value);
    }
    bool valid() const { return record && factor>0 && std::isfinite(factor); }
    float raw(unsigned kind,float x) const {
        const auto* v=segments+record->offsets[kind];
        unsigned lo=0,hi=record->counts[kind]-1;
        while(lo<hi) {unsigned mid=(lo+hi+1)/2;if(v[mid].lower<=x)lo=mid;else hi=mid-1;}
        return v[lo].value(x); // native vectors clamp outside their knot domain
    }
    float range(float energy) const {
        float e=energy*record->mass_ratio;
        float r=std::max(0.f,raw(1,e))/(factor*record->mass_ratio);
        if(e<record->minimum_scaled_energy)r*=std::sqrt(std::max(0.f,e)/record->minimum_scaled_energy);
        return r;
    }
    float dedx(float energy) const {
        float e=energy*record->mass_ratio,d=raw(0,e)*factor;
        if(e<record->minimum_scaled_energy)d*=std::sqrt(std::max(0.f,e)/record->minimum_scaled_energy);
        return d;
    }
    float energy(float range_mm) const {
        const float r=range_mm*factor*record->mass_ratio;
        const float rmin=segments[record->offsets[2]].lower;
        if(r<=0)return 0;
        const float x=r/rmin;
        return (r<rmin?record->minimum_scaled_energy*x*x:raw(2,r))/record->mass_ratio;
    }
    float lambda(float energy_mev) const {
        return interpolate(mfps+record->mfp_offset,record->mfp_count,energy_mev);
    }
};
struct UrbanMcsView {
    const UrbanMcsMaterial* materials{};
    const UrbanMcsSpecies* species{};
    const UrbanMcsRecord* records{};
    const UrbanMcsNode* nodes{};
    const UrbanMcsNode* mfps{};
    const EmCubicSegmentCandidate<float>* segments{};
    std::uint32_t material_count{}, species_count{};
    int projectile(int z,int a) const {
        for(std::uint32_t i=0;i<species_count;++i)
            if(species[i].z==static_cast<unsigned>(z) && species[i].a==static_cast<unsigned>(a))
                return static_cast<int>(i);
        return -1;
    }
    int material(int section,float density) const {
        for(std::uint32_t i=0;i<material_count;++i)
            if(materials[i].section==section &&
               std::fabs(materials[i].density-density)<=2.e-6f*materials[i].density)
                return static_cast<int>(i);
        return -1;
    }
    UrbanMcsContext at(int m,int ion,float energy) const {
        if(m<0 || ion<0 || unsigned(m)>=material_count || unsigned(ion)>=species_count)return {};
        auto* r=records+std::uint64_t(m)*species_count+ion;
        const float f=UrbanMcsContext::interpolate(nodes+r->node_offset,r->node_count,energy);
        return {r,nodes,mfps,segments,f};
    }
};
class UrbanMcsPackage {
 public:
    std::vector<UrbanMcsMaterial> materials;
    std::vector<UrbanMcsSpecies> species;
    std::vector<UrbanMcsRecord> records;
    std::vector<UrbanMcsNode> nodes,mfps;
    std::vector<EmCubicSegmentCandidate<float>> segments;
    static UrbanMcsPackage load(const std::filesystem::path&,const std::string& sha256);
    UrbanMcsView view() const {
        return {materials.data(),species.data(),records.data(),nodes.data(),mfps.data(),segments.data(),
            static_cast<std::uint32_t>(materials.size()),static_cast<std::uint32_t>(species.size())};
    }
};
} // namespace carbon
