#pragma once
#include "carbon/electron_elastic.hpp"
#include "carbon/electron_joint_sampling.hpp"
#include <array>
namespace carbon {
// Same GS/Mott law and source tables as electron_elastic.hpp. Keep 1-cos(theta)
// explicitly: reconstructing it from a rounded FP32 cosine loses small angles.
struct ElectronElasticFloatDraw {
    bool valid{};float one_minus_cosine{};unsigned attempts{},energy_index{};
};
struct ElasticFloatBracket {unsigned low{};float high_weight{};};
inline ElasticFloatBracket elastic_float_bracket(float energy,float mass) {
    const float p2=energy*(energy+2*mass),beta=p2/(p2+mass*mass);
    const float midp=.1f*(.1f+2*mass),midbeta=midp/(midp+mass*mass);
    if(beta>=.9999f)return {30,0};
    float x=0;
    if(beta>=midbeta)x=15+15*(beta-midbeta)/(.9999f-midbeta);
    else if(energy>.001f)x=15*(std::log(energy)-std::log(.001f))/std::log(100.f);
    const auto lo=static_cast<unsigned>(x);return {lo,x-lo};
}
inline float elastic_float_rejection(ElectronElasticView view,unsigned row_index,float delta) {
    if(!view.rows||row_index>=31||!std::isfinite(delta)||delta<0||delta>2)return -1;
    const float y=std::sqrt(.5f*delta),x=y*31;
    const unsigned i=static_cast<unsigned>(x);const auto* row=view.rows+38*row_index+3;
    if(i<30){const float a=row[i],b=row[i+1];return a+(x-i)*(b-a);}
    const float t=(y-30.f/31)*31;
    const float a=row[31],b=row[32],c=row[33],d=row[34];
    return -4.5f*(t-1.f/3)*(t-2.f/3)*(t-1)*a
        +13.5f*t*(t-2.f/3)*(t-1)*b
        -13.5f*t*(t-1.f/3)*(t-1)*c+4.5f*t*(t-1.f/3)*(t-2.f/3)*d;
}
template<class Uniform>inline ElectronElasticFloatDraw sample_electron_elastic_fp32(
    ElectronElasticView view,double energy,Uniform& uniform) {
    ElectronElasticFloatDraw out;
    if(!view.rows||!std::isfinite(energy)||energy<1e-5||energy>.65||view.mass<=0||view.bc<=0||view.xc2<=0)return out;
    const float e=energy,mass=view.mass,bc=view.bc,xc2=view.xc2;
    const auto b=elastic_float_bracket(e,mass);const float first=view.rows[38*b.low];
    const float factor=first+(b.high_weight>0?b.high_weight*(float(view.rows[38*(b.low+1)])-first):0);
    const float a=xc2/(4*e*(e+2*mass)*bc)*factor;if(!(a>0))return out;
    // One double subtraction preserves the extreme U~1 tail before narrowing;
    // the screening, divisions, interpolation and square root remain FP32.
    auto proposal=[&](double u){return 2*a*float(u)/(float(1-u)+a);};
    const double u=uniform();if(!std::isfinite(u)||u<0||u>=1)return out;
    out.one_minus_cosine=proposal(u);
    const double choose=uniform();if(!std::isfinite(choose)||choose<0||choose>=1)return out;
    out.energy_index=b.low+(choose<b.high_weight);
    for(unsigned i=0;i<1000;++i){
        ++out.attempts;const float acceptance=elastic_float_rejection(view,out.energy_index,out.one_minus_cosine);
        const double coin=uniform();if(!std::isfinite(acceptance)||!std::isfinite(coin)||coin<0||coin>=1)return out;
        if(coin<=acceptance){out.valid=true;return out;}
        if(i+1<1000){const double next=uniform();if(!std::isfinite(next)||next<0||next>=1)return out;out.one_minus_cosine=proposal(next);}
    }
    return out;
}
// FP32 frame/trigonometry, with a final FP64 normalization at the geometry
// interface. Exact-face navigation retains its existing unit-vector contract.
inline std::array<double,3> electron_rotate_fp32(std::array<double,3> incoming,
    double cosine,double delta,double azimuth) {
    struct V {float x,y,z;};
    auto cross=[](V a,V b){return V{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};};
    const V d{float(incoming[0]),float(incoming[1]),float(incoming[2])};
    const V h=std::abs(d.x)<=std::abs(d.y)&&std::abs(d.x)<=std::abs(d.z)?V{1,0,0}:
        (std::abs(d.y)<=std::abs(d.z)?V{0,1,0}:V{0,0,1});
    V u=cross(h,d);const float inv=1/std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z);u={u.x*inv,u.y*inv,u.z*inv};
    const V v=cross(d,u);const float phi=6.2831853071795864769f*float(azimuth);
    const float c=std::cos(phi),s=std::sin(phi),ct=cosine,dt=delta;
    const float st=std::sqrt(std::max(0.f,dt*(2-dt)));
    const auto n=electron_joint_research::unit({st*(c*u.x+s*v.x)+ct*d.x,
        st*(c*u.y+s*v.y)+ct*d.y,st*(c*u.z+s*v.z)+ct*d.z});
    return {n.x,n.y,n.z};
}
} // namespace carbon
