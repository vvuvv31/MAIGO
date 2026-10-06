#pragma once
#include "carbon/electron_fp32.hpp"
#include <algorithm>
#include <cmath>

namespace carbon::electron_response26 {
using Draw=electron_fp32::Draw;
// degree is the number of implicit density partitions in this prototype
// layout (0 for a singleton); it is not an energy-polynomial degree.
struct Domain {float low{},high{};unsigned degree{};};
struct JointView {
    const std::uint64_t* cdf{};const float* energy{};const float* cosine{};
    const std::int32_t* pdg{};const float* totals{};unsigned cells{};
};
struct RateView {
    const float* edges{};const float* values{};unsigned intervals{},degree{};
    float mass_majorant{};
};
struct State {bool valid{};int pdg{};float energy{},cosine{};unsigned control{};};
inline float coordinate(Domain d,float density) {
    if(!std::isfinite(density)||density<d.low||density>d.high||d.low<=0.f||d.high<d.low||d.degree>64)return NAN;
    return d.low==d.high?0.f:(density-d.low)/(d.high-d.low);
}
struct Location {bool valid{};unsigned low{},high{};float fraction{};};
// The descriptor chooses composition only. Partition coordinates are implicit
// function parameters, not material-density response records or native knots.
inline Location location(Domain d,float density) {
    const float x=coordinate(d,density);if(!std::isfinite(x))return {};
    if(!d.degree)return {true,0,0,0.f};
    if(d.degree==1)return {true,0,1,x};
    const float lower=std::log(d.low),width=(std::log(d.high)-lower)/d.degree;
    if(!(width>0.f))return {};
    const unsigned low=std::min(unsigned(std::max(0.f,(std::log(density)-lower)/width)),d.degree-1);
    const float a=low==0?d.low:std::exp(lower+low*width);
    const float b=low+1==d.degree?d.high:std::exp(lower+(low+1)*width);
    return {true,low,low+1,std::clamp((density-a)/(b-a),0.f,1.f)};
}
template<class Value> inline float mean(Domain d,float density,Value value) {
    const auto p=location(d,density);if(!p.valid)return NAN;
    return (1.f-p.fraction)*value(p.low)+p.fraction*value(p.high);
}
// Choose between neighboring controls with the smaller probability explicit.
// This avoids rounding 1-small_probability to1 and discarding rare controls.
template<class Uniform,class Value> inline int choose(Domain d,float density,Uniform& uniform,Value value) {
    const auto p=location(d,density);if(!p.valid)return -1;
    const float a=value(p.low),b=value(p.high);
    if(!std::isfinite(a)||!std::isfinite(b)||a<0.f||b<0.f)return -1;
    const float left=(1.f-p.fraction)*a,right=p.fraction*b;
    if(left<=0.f)return right>0.f?int(p.high):-1;
    if(right<=0.f)return int(p.low);
    if(left<=right)return uniform()<left/(left+right)?int(p.low):int(p.high);
    return uniform()<right/(left+right)?int(p.high):int(p.low);
}
inline State sample_control(JointView v,unsigned control,Draw draw) {
    State out;if(!v.cells||!v.cdf||!v.energy||!v.cosine||!v.pdg)return out;
    const std::uint64_t offset=std::uint64_t(control)*v.cells;
    if(v.cdf[offset+v.cells-1]!=UINT64_MAX)return out;
    unsigned low=0,high=v.cells-1;
    while(low<high){const auto middle=low+(high-low)/2;
        if(draw.bits<v.cdf[offset+middle])high=middle;else low=middle+1;}
    const auto index=offset+low;
    out={true,v.pdg[low],v.energy[index],v.cosine[index],control};
    out.valid=std::isfinite(out.energy)&&out.energy>0.f&&std::isfinite(out.cosine)&&
        std::abs(out.cosine)<=1.f&&(out.pdg==0||out.pdg==11||out.pdg==22);
    return out;
}
inline float birth_fraction(Domain d,float density,JointView channel) {
    if(!channel.totals)return NAN;
    const float value=mean(d,density,[&](unsigned i){return channel.totals[i];});
    return std::isfinite(value)&&value>=0.f&&value<=1.000002f?std::min(1.f,value):NAN;
}
// Alternative26-only yield source: expected restricted and delta means from
// the existing continuous ion EM model, at the same energy/cut/path. Never use
// an individual sampled aggregate delta draw as the numerator of this law.
// This is not interchangeable with a5MeV-binned state CDF: onset support and
// the parent energy at the hard event must match before coupling the two.
inline float birth_fraction_from_em(float restricted_mean,float delta_mean) {
    if(!std::isfinite(restricted_mean)||!std::isfinite(delta_mean)||
       restricted_mean<0.f||delta_mean<0.f)return NAN;
    const float total=restricted_mean+delta_mean;
    if(!std::isfinite(total))return NAN;
    return total>0.f?delta_mean/total:0.f;
}
struct Partition {bool valid{};float local{},electron{};};
inline Partition partition(float already_sampled_loss,float fraction) {
    if(!std::isfinite(already_sampled_loss)||already_sampled_loss<0.f||
       !std::isfinite(fraction)||fraction<0.f||fraction>1.f)return {};
    const float electron=already_sampled_loss*fraction;
    return {true,already_sampled_loss-electron,electron};
}
template<class Uniform> inline State sample_birth(Domain d,float density,JointView channel,Uniform& uniform) {
    if(!channel.totals)return {};
    const int c=choose(d,density,uniform,[&](unsigned i){return channel.totals[i];});
    return c<0?State{}:sample_control(channel,unsigned(c),uniform());
}
inline float raw_control_rate(RateView v,unsigned control,float energy) {
    if(!v.edges||!v.values||!v.intervals||control>v.degree||!std::isfinite(energy)||
       energy<v.edges[0]||energy>v.edges[v.intervals])return NAN;
    unsigned low=0,high=v.intervals;
    while(low+1<high){const auto middle=low+(high-low)/2;if(v.edges[middle]>energy)high=middle;else low=middle;}
    const float t=(energy-v.edges[low])/(v.edges[low+1]-v.edges[low]);
    const float* values=v.values+(std::uint64_t(control)*v.intervals+low)*4;
    constexpr float x[]={.125f,.375f,.625f,.875f};float result=0.f;
    for(unsigned j=0;j<4;++j){float term=values[j];for(unsigned k=0;k<4;++k)if(k!=j)term*=(t-x[k])/(x[j]-x[k]);result+=term;}
    return std::max(0.f,result);
}
inline float rate(Domain d,float density,RateView v,float energy) {
    if(v.degree!=d.degree)return NAN;
    return density*mean(d,density,[&](unsigned i){return raw_control_rate(v,i,energy);});
}
template<class Uniform> inline State sample_collision(Domain d,float density,RateView rate,
    float energy,JointView channel,Uniform& uniform) {
    if(rate.degree!=d.degree)return {};
    const int c=choose(d,density,uniform,[&](unsigned i){return raw_control_rate(rate,i,energy);});
    return c<0?State{}:sample_control(channel,unsigned(c),uniform());
}
template<class Uniform> inline State sample_photon(Domain d,float density,JointView channel,Uniform& uniform) {
    const int c=choose(d,density,uniform,[](unsigned){return 1.f;});
    return c<0?State{}:sample_control(channel,unsigned(c),uniform());
}
} // namespace carbon::electron_response26
