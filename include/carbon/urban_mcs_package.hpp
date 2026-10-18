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
    float density_ratio{1.f}; // actual/reference, fixed for the active step
    static float interpolate(const UrbanMcsNode* v,unsigned n,float e) {
        if(!v || n<2 || e<v[0].energy || e>v[n-1].energy)
            return std::numeric_limits<float>::quiet_NaN();
        unsigned lo=0,hi=n-1;
        while(hi-lo>1) { unsigned mid=(lo+hi)/2; if(v[mid].energy<=e)lo=mid;else hi=mid; }
        float t=(e-v[lo].energy)/(v[hi].energy-v[lo].energy);
        return v[lo].value+t*(v[hi].value-v[lo].value);
    }
    bool valid() const { return record && factor>0 && std::isfinite(factor) &&
        density_ratio>0 && std::isfinite(density_ratio); }
    float lambda(float energy_mev) const {
        return interpolate(mfps+record->mfp_offset,record->mfp_count,energy_mev)/density_ratio;
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
    bool ct_density_scaling{false};
    int projectile(int z,int a) const {
        for(std::uint32_t i=0;i<species_count;++i)
            if(species[i].z==static_cast<unsigned>(z) && species[i].a==static_cast<unsigned>(a))
                return static_cast<int>(i);
        return -1;
    }
    int material(int section,float density) const {
        if(!(density>0) || !std::isfinite(density))return -1;
        for(std::uint32_t i=0;i<material_count;++i)
            if(materials[i].section==section &&
               std::fabs(materials[i].density-density)<=2.e-6f*materials[i].density)
                return static_cast<int>(i);
        // Research CT fallback stays within the same elemental composition.
        // Exact material/cut couples above always take precedence.
        if(ct_density_scaling && section>=0) {
            int best=-1;float distance=std::numeric_limits<float>::infinity();
            for(std::uint32_t i=0;i<material_count;++i)if(materials[i].section==section) {
                const float ratio=density/materials[i].density;
                const float d=std::max(ratio,1.f/ratio);
                if(d<distance){distance=d;best=static_cast<int>(i);}
            }
            return best;
        }
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
    static UrbanMcsPackage load(const std::filesystem::path&);
    UrbanMcsView view() const {
        return {materials.data(),species.data(),records.data(),nodes.data(),mfps.data(),segments.data(),
            static_cast<std::uint32_t>(materials.size()),static_cast<std::uint32_t>(species.size())};
    }
};
} // namespace carbon
