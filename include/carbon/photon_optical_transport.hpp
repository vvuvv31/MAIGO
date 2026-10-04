#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_ct_boundary.hpp"
#include <algorithm>

namespace carbon {
// Straight photon flight integrates one sampled optical depth through actual
// voxel materials. It does not replay a finite source trajectory and never
// assigns energy to an interface. Interaction outcomes are handled separately.
struct PhotonAttenuation { bool valid{}; double per_mm{}; };
enum class PhotonFlightStatus { invalid, interaction, escaped, coverage_missing };
struct PhotonFlight {
    PhotonFlightStatus status{PhotonFlightStatus::invalid};
    std::array<double,3> position{};
    double distance_mm{},remaining_optical_depth{};
    std::uint8_t section{};float density{};
    std::uint64_t voxel_crossings{};
};
// rates(section, physical_density, energy) must return an independently
// validated attenuation coefficient, including all retained photon processes.
// Optical depth is -log(U), U strictly between zero and one. No new U is drawn
// at a material change: one collision hazard is integrated over the whole ray.
template<class Rates>
inline PhotonFlight transport_photon_optical_depth(const ElectronCtGeometry& g,
    const std::array<double,3>& start,const std::array<double,3>& direction,
    double energy_MeV,double optical_depth,Rates rates) {
    PhotonFlight out;out.position=start;out.remaining_optical_depth=optical_depth;
    if(!g.density || !g.section || !std::isfinite(energy_MeV) || energy_MeV<=0 ||
       !std::isfinite(optical_depth) || optical_depth<0)return out;
    std::array<std::int64_t,3> cell{};
    std::array<int,3> sign{};
    std::array<double,3> next{},delta{};
    double norm=0;bool outward=false;
    for(unsigned k=0;k<3;++k) {
        if(!g.bins[k] || !std::isfinite(g.origin[k]) || !std::isfinite(g.spacing[k]) || g.spacing[k]<=0 ||
           !std::isfinite(start[k]) || !std::isfinite(direction[k]))return out;
        norm+=direction[k]*direction[k];
        const double u=(start[k]-g.origin[k])/g.spacing[k];
        if(!std::isfinite(u) || u<0 || u>g.bins[k])return out;
        cell[k]=static_cast<std::int64_t>(std::floor(u));
        if(direction[k]<0 && u==std::floor(u))--cell[k];
        // A photon exactly on the physical outer face and directed outward
        // has escaped. Tangential rays on the upper face own the inner cell.
        if(direction[k]==0 && u==g.bins[k])--cell[k];
        if(cell[k]<0 || cell[k]>=g.bins[k])outward=true;
        sign[k]=(direction[k]>0)-(direction[k]<0);
        next[k]=delta[k]=std::numeric_limits<double>::infinity();
        if(sign[k]) {
            const double face=g.origin[k]+(cell[k]+(sign[k]>0))*g.spacing[k];
            next[k]=(face-start[k])/direction[k];delta[k]=g.spacing[k]/std::abs(direction[k]);
            if(next[k]<0 && !outward)return out;
        }
    }
    if(std::abs(norm-1)>1e-9)return out;
    if(outward){out.status=PhotonFlightStatus::escaped;return out;}
    double previous=0;
    const auto limit=std::uint64_t(g.bins[0])+g.bins[1]+g.bins[2]+3;
    for(std::uint64_t hop=0;hop<limit;++hop) {
        const auto index=(std::uint64_t(cell[2])*g.bins[1]+cell[1])*g.bins[0]+cell[0];
        out.section=g.section[index];out.density=g.density[index];
        if(out.section>=kElectronMaterialCount || !std::isfinite(out.density) || out.density<=0)return out;
        const auto coefficient=rates(out.section,out.density,energy_MeV);
        if(!coefficient.valid) {out.status=PhotonFlightStatus::coverage_missing;return out;}
        if(!std::isfinite(coefficient.per_mm) || coefficient.per_mm<0)return out;
        const double face=std::min(next[0],std::min(next[1],next[2]));
        const double segment=face-previous;
        if(!std::isfinite(face) || segment<0)return out;
        const double hazard=coefficient.per_mm*segment;
        if(coefficient.per_mm>0 && out.remaining_optical_depth<hazard) {
            const double distance=previous+out.remaining_optical_depth/coefficient.per_mm;
            for(unsigned k=0;k<3;++k)out.position[k]=start[k]+distance*direction[k];
            out.distance_mm=distance;out.remaining_optical_depth=0;
            out.status=PhotonFlightStatus::interaction;return out;
        }
        out.remaining_optical_depth-=hazard;
        if(!std::isfinite(out.remaining_optical_depth) || out.remaining_optical_depth<0)return out;
        out.distance_mm=face;
        for(unsigned k=0;k<3;++k)out.position[k]=start[k]+face*direction[k];
        for(unsigned k=0;k<3;++k)if(next[k]==face) {
            // Exact face ownership, not epsilon displacement into a neighbour.
            out.position[k]=g.origin[k]+(cell[k]+(sign[k]>0))*g.spacing[k];
            cell[k]+=sign[k];next[k]+=delta[k];
        }
        ++out.voxel_crossings;
        for(unsigned k=0;k<3;++k)if(cell[k]<0 || cell[k]>=g.bins[k]) {
            out.status=PhotonFlightStatus::escaped;return out;
        }
        previous=face;
    }
    return out;
}
} // namespace carbon
