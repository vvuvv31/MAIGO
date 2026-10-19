#pragma once
#include "carbon/unrestricted_csda.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

namespace carbon {
// Full ELECTRONIC loss only. Cu nuclear recoils remain separate beamline
// sinks. Material indices are fixed: 0=G4_Cu, 1=G4_AIR.
struct BeamlineEmMaterial { float density, excitation_MeV, e0_MeV; };
struct BeamlineEmSpecies { std::uint32_t z,a; float mass_MeV; };
struct BeamlineEmRecord { std::uint32_t offset,count; };
struct BeamlineEmNode { float energy,stopping,ion_variance,universal_variance; };
static_assert(sizeof(BeamlineEmMaterial)==12 && sizeof(BeamlineEmSpecies)==12 &&
              sizeof(BeamlineEmRecord)==8 && sizeof(BeamlineEmNode)==16);
struct BeamlineEmPoint {
    const BeamlineEmNode* nodes{};
    const double* ranges{};
    std::uint32_t count{};
    BeamlineEmMaterial material{};
    float density_scale{1.f}; // recoil mass-stopping bank: actual density; beamline: 1
    bool valid() const { return nodes && ranges && count>=2 && density_scale>0 && std::isfinite(density_scale); }
    unsigned cell(float e) const {
        unsigned lo=0,hi=count-1;
        while(hi-lo>1){unsigned mid=(lo+hi)/2;if(nodes[mid].energy<=e)lo=mid;else hi=mid;}
        return lo;
    }
    BeamlineEmNode response(float e) const {
        if(!valid() || !(e>=0) || e>nodes[count-1].energy)
            return {e,NAN,NAN,NAN};
        if(e<nodes[0].energy) {
            const float scale=std::sqrt(e/nodes[0].energy)*density_scale;
            return {e,nodes[0].stopping*scale,nodes[0].ion_variance*density_scale,nodes[0].universal_variance*density_scale};
        }
        const unsigned i=cell(e);const auto l=nodes[i],r=nodes[i+1];
        const float w=(e-l.energy)/(r.energy-l.energy);
        return {e,(l.stopping+w*(r.stopping-l.stopping))*density_scale,
            (l.ion_variance+w*(r.ion_variance-l.ion_variance))*density_scale,
            (l.universal_variance+w*(r.universal_variance-l.universal_variance))*density_scale};
    }
    float dedx(float e) const { return response(e).stopping; }
    float range(float e) const {
        if(!valid() || !(e>=0) || e>nodes[count-1].energy)return NAN;
        if(e<nodes[0].energy)return ranges[0]*std::sqrt(e/nodes[0].energy)/density_scale;
        const unsigned i=cell(e);const auto l=nodes[i],r=nodes[i+1];
        const float stop=l.stopping+(e-l.energy)*(r.stopping-l.stopping)/(r.energy-l.energy);
        return (ranges[i]+linear_stopping_path(e-l.energy,l.stopping,stop))/density_scale;
    }
    float energy(float path) const {
        path*=density_scale;
        if(!valid() || !(path>=0) || path>ranges[count-1])return NAN;
        if(path<ranges[0]){const float x=path/ranges[0];return nodes[0].energy*x*x;}
        unsigned lo=0,hi=count-1;
        while(hi-lo>1){unsigned m=(lo+hi)/2;if(ranges[m]<=path)lo=m;else hi=m;}
        const auto l=nodes[lo],r=nodes[lo+1];
        return l.energy+linear_stopping_energy(float(path-ranges[lo]),l.stopping,
            (r.stopping-l.stopping)/(r.energy-l.energy));
    }
    // Only for an already validated same-energy/material Urban proposal.
    float mean_loss_with_range(float e,float path,float r) const {
        if(path>=r)return e;
        const float loss=dedx(e)*path;
        return loss>.02f*e ? e-energy(r-path) : loss;
    }
    float mean_loss(float e,float path) const {
        return mean_loss_with_range(e,path,range(e));
    }
};
struct BeamlineEmView {
    const BeamlineEmMaterial* materials{};
    const BeamlineEmSpecies* species{};
    const BeamlineEmRecord* records{};
    const BeamlineEmNode* nodes{};
    const double* ranges{};
    unsigned species_count{};
    int projectile(int z,int a) const {
        for(unsigned i=0;i<species_count;++i)if(species[i].z==unsigned(z)&&species[i].a==unsigned(a))return i;
        return -1;
    }
    BeamlineEmPoint at(unsigned material,int ion) const {
        if(material>=2 || ion<0 || unsigned(ion)>=species_count)return {};
        const auto r=records[material*species_count+ion];
        return {nodes+r.offset,ranges+r.offset,r.count,materials[material]};
    }
};
struct BeamlineEmPackage {
    std::vector<BeamlineEmMaterial> materials;
    std::vector<BeamlineEmSpecies> species;
    std::vector<BeamlineEmRecord> records;
    std::vector<BeamlineEmNode> nodes;
    std::vector<double> ranges;
    static BeamlineEmPackage load(const std::filesystem::path&);
    BeamlineEmView view() const {return {materials.data(),species.data(),records.data(),nodes.data(),ranges.data(),unsigned(species.size())};}
};
}
