#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace carbon {
// Geometry-only diagnostic integrator. Each visitor receives exactly the path
// inside ONE voxel. False stops at an unsupported material, never traverses it.
struct LongitudinalMarchResult {
    double traversed_mm{0};
    bool escaped{false};
    bool blocked{false};
    bool invalid{false};
};
template<class Visitor>
inline LongitudinalMarchResult march_longitudinal_segments(
    const std::array<double,3>& source, const std::array<double,3>& direction,
    const std::array<double,3>& origin, const std::array<double,3>& spacing,
    const std::array<int,3>& dims, double range_mm, Visitor visit) {
    LongitudinalMarchResult result;
    std::array<int,3> cell{}, sign{};
    std::array<double,3> face{}, delta{};
    double norm2 = 0;
    if (!(range_mm > 0) || !std::isfinite(range_mm)) {
        result.invalid = true; return result;
    }
    for (int a=0; a<3; ++a) {
        if (!(spacing[a]>0) || !std::isfinite(spacing[a]) || dims[a]<=0 ||
            !std::isfinite(source[a]) || !std::isfinite(origin[a]) ||
            !std::isfinite(direction[a])) { result.invalid=true; return result; }
        const double u=(source[a]-origin[a])/spacing[a];
        // Reject far-outside inputs before integer conversion.
        if (u < -1 || u > static_cast<double>(dims[a])+1) {
            result.escaped=true; return result;
        }
        const double floored=std::floor(u);
        cell[a]=static_cast<int>(floored);
        sign[a]=(direction[a]>0)-(direction[a]<0);
        if (sign[a]<0 && u==floored) --cell[a];
        face[a]=delta[a]=1e300;
        if (sign[a]!=0) {
            const double edge=origin[a]+(cell[a]+(sign[a]>0))*spacing[a];
            face[a]=(edge-source[a])/direction[a];
            delta[a]=spacing[a]/std::abs(direction[a]);
        }
        norm2+=direction[a]*direction[a];
    }
    if (std::abs(norm2-1)>1e-5) { result.invalid=true; return result; }
    const auto limit=static_cast<std::size_t>(dims[0])+dims[1]+dims[2]+3U;
    for (std::size_t step=0;step<limit;++step) {
        for (int a=0;a<3;++a) if(cell[a]<0 || cell[a]>=dims[a]) {
            result.escaped=true; return result;
        }
        double end=range_mm;
        for(int a=0;a<3;++a) if(face[a]<end) end=face[a];
        if (end < result.traversed_mm) { result.invalid=true; return result; }
        if (end>result.traversed_mm &&
            !visit(cell,end-result.traversed_mm)) {
            result.blocked=true; return result;
        }
        result.traversed_mm=end;
        if(end>=range_mm) return result;
        for(int a=0;a<3;++a) if(face[a]<=end) {
            cell[a]+=sign[a]; face[a]+=delta[a];
        }
    }
    result.invalid=true; return result;
}
// Diagnostic mass-column transport. Range is in g/cm2; geometry remains mm.
// A terminal partial cell consumes only its remaining mass budget. No source-
// density extrapolation across an interface, and no interface reflection.
template<class Density, class Visitor>
inline LongitudinalMarchResult march_longitudinal_mass_segments(
    const std::array<double,3>& source, const std::array<double,3>& direction,
    const std::array<double,3>& origin, const std::array<double,3>& spacing,
    const std::array<int,3>& dims, double mass_range_g_cm2,
    Density density, Visitor visit) {
    LongitudinalMarchResult result;
    if (!(mass_range_g_cm2>0) || !std::isfinite(mass_range_g_cm2)) {
        result.invalid=true; return result;
    }
    double remaining=mass_range_g_cm2, geometric_limit=0;
    bool invalid_density=false;
    for(int a=0;a<3;++a) geometric_limit+=dims[a]*spacing[a];
    auto march=march_longitudinal_segments(source,direction,origin,spacing,dims,
        geometric_limit,[&](const std::array<int,3>& cell,double length) {
            if (remaining<=0) return false;
            const double rho=density(cell);
            if (!(rho>0) || !std::isfinite(rho)) {invalid_density=true;return false;}
            const double available=rho*length/10.0;
            const double consumed=std::min(remaining,available);
            visit(cell,consumed/mass_range_g_cm2);
            result.traversed_mm+=10.0*consumed/rho;
            remaining=consumed>=remaining ? 0.0 : remaining-consumed;
            return remaining>0;
        });
    result.invalid=march.invalid || invalid_density;
    result.escaped=march.escaped && remaining>0;
    result.blocked=remaining>0 && !result.escaped && !result.invalid;
    return result;
}
// Experimental geometry primitive only; NOT wired into production transport.
// Keep the ordered electron path instead of collapsing all turns into one ray.
// Each vector is a signed displacement in mass coordinates (g/cm2). Only the
// endpoint represents a sampled deposit: visitors must not deposit per segment.
struct MassPathResult {
    std::array<double,3> endpoint{};
    bool escaped{false},invalid{false};
    std::size_t completed_segments{};
};
template<class Density>
inline MassPathResult replay_mass_polyline(
    const std::array<double,3>& birth,const std::array<double,3>* segments,std::size_t count,
    const std::array<double,3>& origin,const std::array<double,3>& spacing,
    const std::array<int,3>& dims,Density density) {
    MassPathResult out;out.endpoint=birth;
    if(!segments || !count) {out.invalid=true;return out;}
    for(std::size_t i=0;i<count;++i) {
        const auto v=segments[i];double norm=0;
        for(double x:v)norm+=x*x;
        norm=std::sqrt(norm);
        if(!std::isfinite(norm)) {out.invalid=true;return out;}
        if(norm==0) {++out.completed_segments;continue;}
        const std::array<double,3> direction{v[0]/norm,v[1]/norm,v[2]/norm};
        const auto m=march_longitudinal_mass_segments(out.endpoint,direction,origin,spacing,dims,norm,density,
            [](const std::array<int,3>&,double){});
        for(int a=0;a<3;++a)out.endpoint[a]+=direction[a]*m.traversed_mm;
        if(m.invalid || m.blocked) {out.invalid=true;return out;}
        if(m.escaped) {out.escaped=true;return out;}
        ++out.completed_segments;
    }
    return out;
}
} // namespace carbon
