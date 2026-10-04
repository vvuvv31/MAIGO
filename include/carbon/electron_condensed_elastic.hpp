#pragma once
#include "carbon/electron_fp32.hpp"

namespace carbon::electron_condensed_research {
namespace f=electron_fp32;
using V=f::joint_math::V;
// Research primitive at fixed energy in a homogeneous segment. Not connected
// to transport: material/scoring boundaries and energy loss need separate work.
struct SoftLaw {
    f::ElectronElasticView elastic{};
    float screening{},cutoff_q{},lower_row_probability{},mean_q{},second_q{};
    unsigned lower_row{};
};
struct Segment {bool valid{};V position{},direction{};unsigned events{},mode{};};

inline V rotate(V direction,float q,float azimuth) {
    const auto frame=f::joint_math::azimuth(direction,6.2831853071795864769f*azimuth);
    return f::joint_math::unit(std::sqrt(std::max(0.f,q*(2-q)))*frame.u+(1-q)*direction);
}
template<class Uniform>inline float single_soft(const SoftLaw& law,Uniform& uniform) {
    const unsigned row=law.lower_row+(float(uniform())>=law.lower_row_probability);
    const float a=law.screening,upper=law.cutoff_q*(1+a)/(2*a+law.cutoff_q);
    for(unsigned k=0;k<1000;++k){
        float u=float(uniform())*upper;
        if(u>=upper)u=std::nextafter(upper,0.f);
        const float q=2*a*u/(1-u+a);
        const float acceptance=f::elastic_mott_rejection(law.elastic,row,q);
        if(!std::isfinite(acceptance)||q<0||q>law.cutoff_q*(1+8*std::numeric_limits<float>::epsilon()))return NAN;
        if(float(uniform())<=acceptance)return q;
    }
    return NAN;
}
template<class Uniform>inline float gamma(float shape,Uniform& uniform) {
    if(!(shape>0)||!std::isfinite(shape))return NAN;
    const bool augment=shape<1;
    const float d=(augment?shape+1:shape)-1.f/3,c=1/std::sqrt(9*d);
    for(unsigned k=0;k<1000;++k){
        const float x=std::sqrt(-2*std::log(float(uniform())))*f::cos32(6.2831853071795864769f*float(uniform()));
        const float root=1+c*x;if(root<=0)continue;
        const float v=root*root*root,u=float(uniform());
        if(u<1-.0331f*x*x*x*x||std::log(u)<.5f*x*x+d*(1-v+std::log(v))){
            const float value=d*v;
            return augment?value*std::exp(std::log(float(uniform()))/shape):value;
        }
    }
    return NAN;
}
// Exact compound-Poisson single-soft reference, including collision locations.
template<class Uniform>inline Segment explicit_segment(const SoftLaw& law,float mu,float length,V direction,Uniform& uniform) {
    Segment out;out.direction=direction;out.mode=0;
    if(!(mu>0)||!(length>0)||length>.25f)return out;
    float along=0;
    for(unsigned k=0;k<10000;++k){
        const float next=-std::log(float(uniform()))/mu;
        const float end=std::min(1.f,along+next);
        out.position=out.position+(length*(end-along))*out.direction;
        if(end==1){out.valid=true;return out;}
        const float q=single_soft(law,uniform);if(!std::isfinite(q))return out;
        out.direction=rotate(out.direction,q,float(uniform()));++out.events;along=end;
    }
    return out;
}
// Preserve exact zero/one-collision probabilities. For N>=2, fit a bounded
// beta law on q in [0,2] to the first two compound-Poisson Legendre moments.
// Two fixed quadrature hinges plus an independent transverse bridge supply
// lateral displacement for multiple collisions. All subsegments have positive
// lengths and unit directions; no unconstrained Gaussian endpoint is placed
// outside the path-length sphere. This spatial law still requires screening.
template<class Uniform>inline Segment condensed_segment(const SoftLaw& law,float mu,float length,V direction,Uniform& uniform) {
    const float a=mu*law.mean_q,b=mu*law.second_q;
    if(mu<.5f||a>.02f)return explicit_segment(law,mu,length,direction,uniform);
    Segment out;out.direction=direction;
    if(!(mu>0)||!std::isfinite(mu)||!(length>0)||length>.25f)return out;
    const float p0=std::exp(-mu),p1=mu*p0,pm=-std::expm1(-mu)-p1;
    const float first=-std::expm1(-a);
    // Stable 2[(1-exp(-a))-(1-exp(-3a))/3], expanded through a^5.
    const float curvature=2*a*a*(1+a*(-4.f/3+a*(13.f/12-a*2.f/3)));
    const float second=curvature+(2.f/3)*std::exp(-3*a)*std::expm1(1.5f*b);
    const float m=(first-p1*law.mean_q)/pm,s=(second-p1*law.second_q)/pm;
    const float variance=s-m*m,concentration=m*(2-m)/variance-1;
    const float alpha=.5f*m*concentration,beta=(1-.5f*m)*concentration;
    if(!(pm>0)||!(alpha>0)||!(beta>0)||!std::isfinite(alpha)||!std::isfinite(beta))return out;
    const float choice=uniform();float q=0;
    if(choice<p0){out.position=length*direction;out.mode=1;out.valid=true;return out;}
    if(choice<p0+p1){q=single_soft(law,uniform);out.mode=2;out.events=1;}
    else {const float ga=gamma(alpha,uniform),gb=gamma(beta,uniform);q=2*ga/(ga+gb);out.mode=3;}
    if(!std::isfinite(q)||q<0||q>2)return out;
    const auto next=rotate(direction,q,float(uniform()));
    if(out.mode==2){
        const float hinge=uniform();out.position=(length*hinge)*direction+(length*(1-hinge))*next;
    }else{
        const auto frame=f::joint_math::basis(direction);
        const float radial=std::sqrt(-2*std::log(float(uniform())));
        const auto sc=f::sincos32(6.2831853071795864769f*float(uniform()));
        const float sigma=std::sqrt((m-.5f*s)*.25f);
        const auto middle=f::joint_math::unit(.5f*(direction+next)+
            (sigma*radial*sc[0])*frame.u+(sigma*radial*sc[1])*frame.v);
        constexpr float distance=.5773502691896258f,outer=(1-distance)*.5f;
        out.position=(length*outer)*direction+(length*distance)*middle+(length*outer)*next;
        out.mode=4;
    }
    out.direction=next;out.valid=true;return out;
}
// Research isolation of the accurate no-energy-loss GS spatial formula from
// Geant4 11.3.2 G4GoudsmitSaundersonMscModel::SampleMSC (Geant4 Collaboration).
// See LICENSES/Geant4.txt and the pinned source manifest in
// out/electron_condensed_20261004/native_source/. This is not the full GS model:
// both half-angle distributions below use exact explicit soft scatters, and
// G2/G1 belongs to our truncated Mott law. Never suppress lateral displacement.
template<class Uniform>inline Segment gs_spatial_segment(const SoftLaw& law,float mu,float length,V direction,Uniform& uniform,unsigned exact_limit=0) {
    Segment out;out.direction=direction;
    if(!(mu>0)||!(length>0)||length>.25f)return out;
    if(mu*law.mean_q>.02f)return explicit_segment(law,mu,length,direction,uniform);
    const auto first=explicit_segment(law,mu*.5f,length*.5f,{0,0,1},uniform);
    const auto second=explicit_segment(law,mu*.5f,length*.5f,{0,0,1},uniform);
    if(!first.valid||!second.valid)return out;
    out.events=first.events+second.events;out.mode=0;
    if(!out.events){out.position=length*direction;out.valid=true;return out;}
    // Join the independently generated exact half-trajectories when the total
    // multiplicity is small. The second half is expressed in the first half's
    // outgoing frame, preserving positions and their correlation with angle.
    // This is a spatial-isolation fixture, not a fast angular sampler yet.
    if(exact_limit>=2&&out.events<=exact_limit){
        const auto mid=f::joint_math::basis(first.direction);
        const auto initial=f::joint_math::basis(direction);
        const auto p=first.position+second.position.x*mid.u+second.position.y*mid.v+second.position.z*first.direction;
        const auto d=second.direction.x*mid.u+second.direction.y*mid.v+second.direction.z*first.direction;
        out.position=p.x*initial.u+p.y*initial.v+p.z*direction;
        out.direction=f::joint_math::unit(d.x*initial.u+d.y*initial.v+d.z*direction);
        out.valid=true;return out;
    }
    const float c1=first.direction.z,c2=second.direction.z;
    const float s1=std::sqrt(first.direction.x*first.direction.x+first.direction.y*first.direction.y);
    const float s2=std::sqrt(second.direction.x*second.direction.x+second.direction.y*second.direction.y);
    const auto phi1=f::sincos32(6.2831853071795864769f*float(uniform()));
    const auto phi2=f::sincos32(6.2831853071795864769f*float(uniform()));
    const float sp1=phi1[0],cp1=phi1[1],u2=s2*phi2[1],v2=s2*phi2[0];
    const float u2p=c1*u2+s1*c2;
    const float ux=u2p*cp1-v2*sp1,uy=u2p*sp1+v2*cp1,uz=c1*c2-s1*u2;
    const auto frame=f::joint_math::basis(direction);
    out.direction=f::joint_math::unit(ux*frame.u+uy*frame.v+uz*direction);
    if(exact_limit==1&&out.events==1){
        const float hinge=uniform();
        out.position=(length*hinge)*direction+(length*(1-hinge))*out.direction;
        out.valid=true;return out;
    }
    const float ratio=3-1.5f*law.second_q/law.mean_q;
    const float eta=std::sqrt(float(uniform())),eta1=.5f*(1-eta);
    const float delta=.9082483f-(.1020621f-.0263747f*ratio)*mu*law.mean_q;
    const float b=eta*delta,c=eta*(1-delta),w1v2=c1*v2;
    const float px=b*s1*cp1+c*(cp1*u2-sp1*w1v2)+eta1*ux;
    const float py=b*s1*sp1+c*(sp1*u2+cp1*w1v2)+eta1*uy;
    const float pz=eta1+b*c1+c*c2+eta1*uz;
    out.position=length*(px*frame.u+py*frame.v+pz*direction);
    out.mode=6;
    out.valid=std::isfinite(px)&&std::isfinite(py)&&std::isfinite(pz);
    return out;
}
} // namespace carbon::electron_condensed_research
