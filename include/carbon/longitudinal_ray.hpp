#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// Scalar-templated marchers. Real=double preserves the validated diagnostic
// behavior bit-for-bit; Real=float accelerates device electron replay
// (FP32 throughput on sm_75 outweighs FP64 by an order of magnitude).
// Float endpoints can differ from double within ~1e-4 mm, far below the
// 0.5 mm voxel scale; face-exact decisions may flip in rare cases, so float
// results are validated by gamma equivalence, not bitwise identity.
namespace carbon {
// Geometry-only diagnostic integrator. Each visitor receives exactly the path
// inside ONE voxel. False stops at an unsupported material, never traverses it.
struct LongitudinalMarchResult {
    double traversed_mm{0};
    bool escaped{false};
    bool blocked{false};
    bool invalid{false};
};
template<class Real = double, class Visitor>
inline LongitudinalMarchResult march_longitudinal_segments(
    const std::array<Real,3>& source, const std::array<Real,3>& direction,
    const std::array<Real,3>& origin, const std::array<Real,3>& spacing,
    const std::array<int,3>& dims, Real range_mm, Visitor visit) {
    LongitudinalMarchResult result;
    std::array<int,3> cell{}, sign{};
    std::array<Real,3> face{}, delta{};
    Real norm2 = 0;
    if (!(range_mm > 0) || !std::isfinite(range_mm)) {
        result.invalid = true; return result;
    }
    for (int a=0; a<3; ++a) {
        if (!(spacing[a]>0) || !std::isfinite(spacing[a]) || dims[a]<=0 ||
            !std::isfinite(source[a]) || !std::isfinite(origin[a]) ||
            !std::isfinite(direction[a])) { result.invalid=true; return result; }
        const Real u=(source[a]-origin[a])/spacing[a];
        // Reject far-outside inputs before integer conversion.
        if (u < -1 || u > static_cast<Real>(dims[a])+1) {
            result.escaped=true; return result;
        }
        const Real floored=std::floor(u);
        cell[a]=static_cast<int>(floored);
        sign[a]=(direction[a]>0)-(direction[a]<0);
        if (sign[a]<0 && u==floored) --cell[a];
        face[a]=delta[a]=std::numeric_limits<Real>::infinity();
        if (sign[a]!=0) {
            const Real edge=origin[a]+(cell[a]+(sign[a]>0))*spacing[a];
            face[a]=(edge-source[a])/direction[a];
            delta[a]=spacing[a]/std::abs(direction[a]);
        }
        norm2+=direction[a]*direction[a];
    }
    if (std::abs(norm2-1)>Real(1e-5)) { result.invalid=true; return result; }
    const auto limit=static_cast<std::size_t>(dims[0])+dims[1]+dims[2]+3U;
    for (std::size_t step=0;step<limit;++step) {
        for (int a=0;a<3;++a) if(cell[a]<0 || cell[a]>=dims[a]) {
            result.escaped=true; return result;
        }
        Real end=range_mm;
        for(int a=0;a<3;++a) if(face[a]<end) end=face[a];
        if (end < static_cast<Real>(result.traversed_mm)) { result.invalid=true; return result; }
        if (end>static_cast<Real>(result.traversed_mm) &&
            !visit(cell,end-static_cast<Real>(result.traversed_mm))) {
            result.blocked=true; return result;
        }
        result.traversed_mm=static_cast<double>(end);
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
template<class Real = double, class Density, class Visitor>
inline LongitudinalMarchResult march_longitudinal_mass_segments(
    const std::array<Real,3>& source, const std::array<Real,3>& direction,
    const std::array<Real,3>& origin, const std::array<Real,3>& spacing,
    const std::array<int,3>& dims, Real mass_range_g_cm2,
    Density density, Visitor visit) {
    LongitudinalMarchResult result;
    if (!(mass_range_g_cm2>0) || !std::isfinite(mass_range_g_cm2)) {
        result.invalid=true; return result;
    }
    Real remaining=mass_range_g_cm2;
    Real geometric_limit=0;
    bool invalid_density=false;
    for(int a=0;a<3;++a) geometric_limit+=static_cast<Real>(dims[a])*spacing[a];
    auto march=march_longitudinal_segments<Real>(source,direction,origin,spacing,dims,
        geometric_limit,[&](const std::array<int,3>& cell,Real length) {
            if (remaining<=0) return false;
            const Real rho=density(cell);
            if (!(rho>0) || !std::isfinite(rho)) {invalid_density=true;return false;}
            const Real available=rho*length/Real(10.0);
            const Real consumed=std::min(remaining,available);
            visit(cell,consumed/mass_range_g_cm2);
            result.traversed_mm+=static_cast<double>(Real(10.0)*consumed/rho);
            remaining=consumed>=remaining ? Real(0) : remaining-consumed;
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
// A straight segment whose endpoints are strictly inside one voxel cannot
// cross an interface. Use its constant density directly; uncertain faces,
// invalid inputs and crossings must fall through to the reference marcher.
template<class Real = double, class Density>
inline bool try_same_voxel_mass_segment(std::array<Real,3>& point,
    const std::array<Real,3>& v,const std::array<Real,3>& origin,
    const std::array<Real,3>& spacing,const std::array<int,3>& dims,Density density) {
    std::array<int,3> cell{};
    std::array<Real,3> low{},high{},guard{},next{};
    for(int a=0;a<3;++a) {
        if(!(spacing[a]>0) || !std::isfinite(spacing[a]) || dims[a]<=0 ||
            !std::isfinite(point[a]) || !std::isfinite(origin[a]) || !std::isfinite(v[a]))return false;
        const Real u=(point[a]-origin[a])/spacing[a];
        if(!(u>=0 && u<dims[a]))return false; // Before integer conversion.
        cell[a]=static_cast<int>(std::floor(u));
        low[a]=origin[a]+cell[a]*spacing[a];high[a]=low[a]+spacing[a];
        guard[a]=Real(64)*std::numeric_limits<Real>::epsilon()*
            std::max(Real(1.0),std::max(std::abs(low[a]),std::abs(high[a])));
        if(!(point[a]>low[a]+guard[a] && point[a]<high[a]-guard[a]))return false;
    }
    const Real rho=density(cell);
    if(!(rho>0) || !std::isfinite(rho))return false;
    const Real scale=Real(10.0)/rho;
    for(int a=0;a<3;++a) {
        next[a]=point[a]+scale*v[a];
        if(!(next[a]>low[a]+guard[a] && next[a]<high[a]-guard[a]))return false;
    }
    point=next;return true;
}
template<class Real = double>
struct MassPathBoundsT {
    std::array<Real,3> low{},high{},net{};
};
using MassPathBounds = MassPathBoundsT<double>;
template<class Real = double, class Segment>
inline MassPathBoundsT<Real> mass_path_bounds(std::size_t count,Segment segment) {
    MassPathBoundsT<Real> b;
    for(std::size_t i=0;i<count;++i) {
        const auto v=segment(i);
        for(int a=0;a<3;++a) {
            const Real x=static_cast<Real>(v[a]);
            b.net[a]+=x;b.low[a]=std::min(b.low[a],b.net[a]);
            b.high[a]=std::max(b.high[a],b.net[a]);
        }
    }
    return b;
}
// Conservative rotated bounds include EVERY prefix and the source, not just
// the chord endpoints. Only skip the polyline when all of it fits one voxel.
template<class Real = double, class Density>
inline bool try_same_voxel_mass_path(std::array<Real,3>& point,const MassPathBoundsT<Real>& b,
    const std::array<std::array<Real,3>,3>& basis,
    const std::array<Real,3>& origin,const std::array<Real,3>& spacing,
    const std::array<int,3>& dims,Density density) {
    std::array<int,3> cell{};std::array<Real,3> next{};
    for(int a=0;a<3;++a) {
        if(!(spacing[a]>0) || !std::isfinite(spacing[a]) || dims[a]<=0 ||
            !std::isfinite(point[a]) || !std::isfinite(origin[a]))return false;
        const Real u=(point[a]-origin[a])/spacing[a];
        if(!(u>=0 && u<dims[a]))return false;
        cell[a]=static_cast<int>(std::floor(u));
    }
    const Real rho=density(cell);
    if(!(rho>0) || !std::isfinite(rho))return false;
    const Real scale=Real(10.0)/rho;
    for(int a=0;a<3;++a) {
        Real low=0,high=0,net=0;
        for(int k=0;k<3;++k) {
            if(!std::isfinite(b.low[k]) || !std::isfinite(b.high[k]) || !std::isfinite(b.net[k]) ||
                !std::isfinite(basis[k][a]) || b.low[k]>0 || b.high[k]<0 || b.net[k]<b.low[k] || b.net[k]>b.high[k])return false;
            const Real l=b.low[k]*basis[k][a],h=b.high[k]*basis[k][a];
            low+=std::min(l,h);high+=std::max(l,h);net+=b.net[k]*basis[k][a];
        }
        const Real face=origin[a]+cell[a]*spacing[a],end=face+spacing[a];
        const Real guard=Real(64)*std::numeric_limits<Real>::epsilon()*std::max(Real(1.0),std::max(std::abs(face),std::abs(end)));
        if(!(point[a]+scale*low>face+guard && point[a]+scale*high<end-guard))return false;
        next[a]=point[a]+scale*net;
        if(!(next[a]>face+guard && next[a]<end-guard))return false;
    }
    point=next;return true;
}
template<class Real = double, bool SameVoxelFast = true, class Segment, class Density>
inline MassPathResult replay_mass_polyline_indexed(
    const std::array<Real,3>& birth,std::size_t count,Segment segment,
    const std::array<Real,3>& origin,const std::array<Real,3>& spacing,
    const std::array<int,3>& dims,Density density) {
    MassPathResult out;
    for(int a=0;a<3;++a)out.endpoint[a]=static_cast<double>(birth[a]);
    if(!count) {out.invalid=true;return out;}
    for(std::size_t i=0;i<count;++i) {
        const auto v=segment(i);Real norm=0;
        for(int a=0;a<3;++a) {const Real x=static_cast<Real>(v[a]);norm+=x*x;}
        if(!std::isfinite(norm)) {out.invalid=true;return out;}
        if(norm==0) {++out.completed_segments;continue;}
        if constexpr(SameVoxelFast) {
            std::array<Real,3> probe;
            for(int a=0;a<3;++a)probe[a]=static_cast<Real>(out.endpoint[a]);
            if(try_same_voxel_mass_segment<Real>(probe,v,origin,spacing,dims,density)) {
                for(int a=0;a<3;++a)out.endpoint[a]=static_cast<double>(probe[a]);
                ++out.completed_segments;continue;
            }
        }
        norm=std::sqrt(norm);
        const std::array<Real,3> direction{
            static_cast<Real>(v[0])/norm,static_cast<Real>(v[1])/norm,static_cast<Real>(v[2])/norm};
        std::array<Real,3> start;
        for(int a=0;a<3;++a)start[a]=static_cast<Real>(out.endpoint[a]);
        const auto m=march_longitudinal_mass_segments<Real>(start,direction,origin,spacing,dims,norm,density,
            [](const std::array<int,3>&,Real){});
        for(int a=0;a<3;++a)out.endpoint[a]+=static_cast<double>(direction[a])*m.traversed_mm;
        if(m.invalid || m.blocked) {out.invalid=true;return out;}
        if(m.escaped) {out.escaped=true;return out;}
        ++out.completed_segments;
    }
    return out;
}
template<class Real = double, class Density>
inline MassPathResult replay_mass_polyline(
    const std::array<Real,3>& birth,const std::array<Real,3>* segments,std::size_t count,
    const std::array<Real,3>& origin,const std::array<Real,3>& spacing,
    const std::array<int,3>& dims,Density density) {
    if(!segments) {MassPathResult r;r.endpoint=birth;r.invalid=true;return r;}
    return replay_mass_polyline_indexed<Real,true>(birth,count,[&](std::size_t i){return segments[i];},origin,spacing,dims,density);
}
} // namespace carbon
