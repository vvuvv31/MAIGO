#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace carbon {
struct ElectronCtGeometry {
    std::array<std::uint32_t,3> bins{};
    std::array<double,3> origin{},spacing{};
    const float* density{};
    const std::uint8_t* section{};
};
enum class ElectronCtBoundaryStatus { invalid, contained, material_change, escaped };
struct ElectronCtPointMaterial {bool valid{};float density{};std::uint8_t section{};};
// Resolve the actual electron birth point using the same directed, double
// coordinate ownership as electron face traversal. Never inherit a rounded
// primary voxel index, snap the point, or add an epsilon at a material face.
inline ElectronCtPointMaterial electron_ct_point_material(const ElectronCtGeometry& g,
    const std::array<double,3>& position,const std::array<double,3>& direction) {
    if(!g.density || !g.section)return {};
    std::array<std::uint32_t,3> cell{};
    for(unsigned k=0;k<3;++k) {
        if(!g.bins[k] || !std::isfinite(g.spacing[k]) || g.spacing[k]<=0 ||
           !std::isfinite(g.origin[k]) || !std::isfinite(position[k]) || !std::isfinite(direction[k]))return {};
        const double u=(position[k]-g.origin[k])/g.spacing[k];
        if(!std::isfinite(u) || u<0 || u>g.bins[k])return {};
        auto i=static_cast<std::int64_t>(std::floor(u));
        if(direction[k]<0 && u==std::floor(u))--i;
        if(i<0 || i>=g.bins[k])return {};
        cell[k]=static_cast<std::uint32_t>(i);
    }
    const auto i=(std::uint64_t(cell[2])*g.bins[1]+cell[1])*g.bins[0]+cell[0];
    const auto rho=g.density[i];const auto section=g.section[i];
    if(section>=25 || !std::isfinite(rho) || rho<=0)return {};
    return {true,rho,section};
}
struct ElectronCtBoundary {
    ElectronCtBoundaryStatus status{ElectronCtBoundaryStatus::invalid};
    double fraction{};
    std::array<double,3> position{};
    std::uint8_t from_section{},to_section{};
    float from_density{},to_density{};
};
// Exact voxel-face traversal of a geometric segment. No 2% density tolerance,
// endpoint-only shortcut or artificial minimum step. Momentum/path length are
// separate recorded quantities and are NOT reconstructed from this chord.
inline ElectronCtBoundary first_electron_ct_boundary(const ElectronCtGeometry& g,
    const std::array<double,3>& a,const std::array<double,3>& b) {
    ElectronCtBoundary out;
    if(!g.density || !g.section)return out;
    std::array<std::int64_t,3> cell{};
    std::array<int,3> sign{};
    std::array<double,3> next{},delta{};
    bool moving=false;
    for(unsigned k=0;k<3;++k) {
        if(!g.bins[k] || !(g.spacing[k]>0) || !std::isfinite(g.spacing[k]) ||
           !std::isfinite(g.origin[k]) || !std::isfinite(a[k]) || !std::isfinite(b[k]))return out;
        const double d=b[k]-a[k],u=(a[k]-g.origin[k])/g.spacing[k];
        if(!std::isfinite(d) || !std::isfinite(u) || u<0 || u>g.bins[k])return out;
        sign[k]=(d>0)-(d<0);moving=moving || sign[k]!=0;
        cell[k]=static_cast<std::int64_t>(std::floor(u));
        if(d<0 && u==std::floor(u))--cell[k];
        if(cell[k]<0 || cell[k]>=g.bins[k])return out;
        next[k]=delta[k]=std::numeric_limits<double>::infinity();
        if(sign[k]) {
            const auto face=g.origin[k]+(cell[k]+(sign[k]>0))*g.spacing[k];
            next[k]=(face-a[k])/d;delta[k]=g.spacing[k]/std::abs(d);
            if(next[k]<0)return out;
        }
    }
    const auto index=[&]() {return (std::uint64_t(cell[2])*g.bins[1]+cell[1])*g.bins[0]+cell[0];};
    out.from_section=g.section[index()];out.from_density=g.density[index()];
    if(out.from_section>=25 || !std::isfinite(out.from_density) || out.from_density<=0)return out;
    const auto finish=[&](ElectronCtBoundaryStatus status,double t) {
        out.status=status;out.fraction=t;
        for(unsigned k=0;k<3;++k)out.position[k]=a[k]+t*(b[k]-a[k]);
        return out;
    };
    if(!moving)return finish(ElectronCtBoundaryStatus::contained,1);
    const auto limit=std::uint64_t(g.bins[0])+g.bins[1]+g.bins[2]+3;
    for(std::uint64_t hop=0;hop<limit;++hop) {
        double t=next[0]<next[1]?next[0]:next[1];if(next[2]<t)t=next[2];
        if(t>1)return finish(ElectronCtBoundaryStatus::contained,1);
        std::array<bool,3> crossed{};
        std::array<double,3> face{};
        for(unsigned k=0;k<3;++k)if(next[k]==t) {
            crossed[k]=true;face[k]=g.origin[k]+(cell[k]+(sign[k]>0))*g.spacing[k];
        }
        const auto finish_crossing=[&](ElectronCtBoundaryStatus status) {
            auto result=finish(status,t);
            // Keep the exact voxel face, not a cancellation-rounded a+t*d.
            // This is ownership bookkeeping, never an epsilon displacement.
            for(unsigned k=0;k<3;++k)if(crossed[k])result.position[k]=face[k];
            return result;
        };
        for(unsigned k=0;k<3;++k)if(next[k]==t) {cell[k]+=sign[k];next[k]+=delta[k];}
        for(unsigned k=0;k<3;++k)if(cell[k]<0 || cell[k]>=g.bins[k])return finish_crossing(ElectronCtBoundaryStatus::escaped);
        out.to_section=g.section[index()];out.to_density=g.density[index()];
        if(out.to_section>=25 || !std::isfinite(out.to_density) || out.to_density<=0)return out;
        if(out.to_section!=out.from_section || out.to_density!=out.from_density)
            return finish_crossing(ElectronCtBoundaryStatus::material_change);
    }
    return out;
}
} // namespace carbon
