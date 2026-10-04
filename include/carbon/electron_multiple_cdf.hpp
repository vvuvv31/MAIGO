#pragma once
#include "carbon/electron_condensed_elastic.hpp"
#include "carbon/electron_elastic_inverse_cdf.hpp"

namespace carbon::electron_multiple_cdf_research {
namespace c=electron_condensed_research;
namespace f=electron_fp32;
using V=c::V;
struct View {const float* quantiles{};unsigned max_count{},bins{};};

template<class Uniform>inline float explicit_angle(const c::SoftLaw& law,unsigned count,Uniform& uniform){
    V direction{0,0,1};
    for(unsigned i=0;i<count;++i){
        const float q=c::single_soft(law,uniform);if(!std::isfinite(q))return NAN;
        direction=c::rotate(direction,q,float(uniform()));
    }
    // Avoid cancellation of 1-cos(theta) for near-forward FP32 directions.
    return direction.z>0?(direction.x*direction.x+direction.y*direction.y)/(1+direction.z):1-direction.z;
}
template<class Uniform>inline float angle(const c::SoftLaw& law,View table,unsigned count,Uniform& uniform){
    if(!count)return 0;
    if(count==1)return c::single_soft(law,uniform);
    if(count>table.max_count)return explicit_angle(law,count,uniform);
    if(!table.quantiles||!table.bins)return NAN;
    return electron_inverse_cdf_research::sample(table.quantiles+std::size_t(count-2)*(48*table.bins+1),table.bins,float(uniform()));
}
// Product form of the Poisson sampler avoids rounded cumulative-sum tails.
// The fixed-energy research domain has half-mu<=32; no underflow cutoff.
template<class Uniform>inline unsigned poisson(float zero_probability,Uniform& uniform){
    float product=1;
    for(unsigned n=0;n<10000;++n){product*=float(uniform());if(product<=zero_probability)return n;}
    return UINT32_MAX;
}
template<class Uniform>inline c::Segment low_count_path(const c::SoftLaw& law,unsigned count,float length,V direction,Uniform& uniform){
    c::Segment out;out.direction=direction;out.events=count;
    if(count>4||!(length>0)||length>.25f)return out;
#ifdef CARBON_MULTIPLE_SCALAR_POSITIONS
    float a=count>0?float(uniform()):1;
    float b=count>1?float(uniform()):1;
    float d=count>2?float(uniform()):1;
    float e=count>3?float(uniform()):1;
    auto order=[](float& x,float& y){const float lower=std::min(x,y);y=std::max(x,y);x=lower;};
    order(a,b);order(d,e);order(a,d);order(b,e);order(b,d);
#else
    float positions[4]{};
    for(unsigned i=0;i<count;++i){
        positions[i]=float(uniform());
        for(unsigned j=i;j&&positions[j]<positions[j-1];--j){const float a=positions[j];positions[j]=positions[j-1];positions[j-1]=a;}
    }
#endif
    float previous=0;
    for(unsigned i=0;i<count;++i){
#ifdef CARBON_MULTIPLE_SCALAR_POSITIONS
        const float position=i==0?a:i==1?b:i==2?d:e;
#else
        const float position=positions[i];
#endif
        out.position=out.position+(length*(position-previous))*out.direction;
        const float q=c::single_soft(law,uniform);if(!std::isfinite(q))return out;
        out.direction=c::rotate(out.direction,q,float(uniform()));previous=position;
    }
    out.position=out.position+(length*(1-previous))*out.direction;out.valid=true;return out;
}
// Same no-energy-loss GS spatial formula as gs_spatial_segment, now with
// independent Poisson half-counts and their full convolution angle laws.
// Geant4-derived spatial equations: see LICENSES/Geant4.txt and the original
// attribution/pinned source in electron_condensed_elastic.hpp. No suppression
// of lateral displacement based on remaining range or geometry safety.
template<class Uniform>
#ifdef CARBON_MULTIPLE_FORCE_INLINE
[[gnu::always_inline]]
#endif
inline c::Segment segment(const c::SoftLaw& law,View table,float mu,float length,V direction,Uniform& uniform){
    c::Segment out;out.direction=direction;
    if(!(mu>0)||!std::isfinite(mu)||!(length>0)||length>.25f)return out;
    if(mu*law.mean_q>.02f||mu>64)return c::explicit_segment(law,mu,length,direction,uniform);
    const float p0=std::exp(-.5f*mu);
    const unsigned n1=poisson(p0,uniform),n2=poisson(p0,uniform);
    if(n1==UINT32_MAX||n2==UINT32_MAX)return out;
    const unsigned count=n1+n2;
    if(count<=4)return low_count_path(law,count,length,direction,uniform);
    const float q1=angle(law,table,n1,uniform),q2=angle(law,table,n2,uniform);
    if(!std::isfinite(q1)||!std::isfinite(q2)||q1<0||q1>2||q2<0||q2>2)return out;
    const float c1=1-q1,c2=1-q2,s1=std::sqrt(q1*(2-q1)),s2=std::sqrt(q2*(2-q2));
    const auto phi1=f::sincos32(6.2831853071795864769f*float(uniform()));
    const auto phi2=f::sincos32(6.2831853071795864769f*float(uniform()));
    const float sp1=phi1[0],cp1=phi1[1],u2=s2*phi2[1],v2=s2*phi2[0];
    const float u2p=c1*u2+s1*c2;
    const float ux=u2p*cp1-v2*sp1,uy=u2p*sp1+v2*cp1,uz=c1*c2-s1*u2;
    const auto frame=f::joint_math::basis(direction);
    out.direction=f::joint_math::unit(ux*frame.u+uy*frame.v+uz*direction);
    const float ratio=3-1.5f*law.second_q/law.mean_q;
    const float eta=std::sqrt(float(uniform())),eta1=.5f*(1-eta);
    const float delta=.9082483f-(.1020621f-.0263747f*ratio)*mu*law.mean_q;
    const float b=eta*delta,cweight=eta*(1-delta),w1v2=c1*v2;
    const float px=b*s1*cp1+cweight*(cp1*u2-sp1*w1v2)+eta1*ux;
    const float py=b*s1*sp1+cweight*(sp1*u2+cp1*w1v2)+eta1*uy;
    const float pz=eta1+b*c1+cweight*c2+eta1*uz;
    out.position=length*(px*frame.u+py*frame.v+pz*direction);
    out.events=count;out.mode=6;out.valid=std::isfinite(px)&&std::isfinite(py)&&std::isfinite(pz);
    return out;
}
} // namespace carbon::electron_multiple_cdf_research
