#pragma once
// Isolated research kernel. No production config can select this operator.
// FP64 geometric/table arithmetic is not a dose scorer: GPU dose is FP32.
#include "carbon/rng.hpp"
#include <cmath>
#include <cstdint>

namespace electron_joint_research {
struct V { double x{},y{},z{}; };
inline V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V operator*(double a,V b){return {a*b.x,a*b.y,a*b.z};}
inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline V unit(V v){return (1./std::sqrt(dot(v,v)))*v;}
struct Frame {V u,v,d;};
inline Frame basis(V d){
    V h=std::abs(d.x)<=std::abs(d.y) && std::abs(d.x)<=std::abs(d.z)?V{1,0,0}:
        (std::abs(d.y)<=std::abs(d.z)?V{0,1,0}:V{0,0,1});
    const auto u=unit(cross(h,d));return {u,cross(d,u),d};
}
inline Frame azimuth(V d,double phi){auto b=basis(d);const auto c=std::cos(phi),s=std::sin(phi);
    return {c*b.u+s*b.v,(-s)*b.u+c*b.v,d};}
inline V rotate(Frame b,V v){return v.x*b.u+v.y*b.v+v.z*b.d;}
inline V project(Frame b,V v){return {dot(v,b.u),dot(v,b.v),dot(v,b.d)};}
struct SampledStep {double cdf;V delta,post,source;double energy,length,fraction[4];std::int32_t child[2];};
struct GraphStep {V delta;double length,fraction[4];V postdir;std::int32_t next,child[2],pdg,inside,pad;};
static_assert(sizeof(SampledStep)==136 && sizeof(GraphStep)==112);
struct View {
    const SampledStep* steps{};const GraphStep* graph{};
    const double* edges{};const std::uint32_t* offsets{};const std::uint32_t* nearest{};
    std::uint32_t continuing{},entry{},graph_count{},bins{};
    double incident{};
};
struct Result {double x{},y{},z{},energy{};std::uint32_t status{5},advances{},graph_advances{},below{},reason{},iteration{},sampled_row{};std::int32_t cursor{};};
inline double boundary(V p,V d){
    double f=1.;
    for(int i=0;i<3;++i){const double x=i==0?p.x:(i==1?p.y:p.z),v=i==0?d.x:(i==1?d.y:d.z);
        if(v!=0.){const double t=((v>0.?20.:-20.)-x)/v;if(t<f)f=t;}}
    return f<0.?0.:f;
}
inline Result transport(View model,std::uint64_t seed,std::uint64_t history){
    auto uniform=[&](std::uint32_t iteration,std::uint32_t dimension){
        return (double(carbon::rng::random_u32(seed,history,iteration,dimension))+.5)*0x1p-32;};
    V p{.5*uniform(4001,0)-.25,2*uniform(4001,1)-1,.25},d{0,0,1};
    double energy=model.incident;bool photon=false;std::int32_t cursor=-1;
    Frame original{{1,0,0},{0,1,0},{0,0,1}},target=original;
    Result result;
    for(std::uint32_t iteration=0;iteration<4000;++iteration){
        result.iteration=iteration;
        const bool exact=cursor>=0;
        const GraphStep* g=nullptr;const SampledStep* s=nullptr;Frame sample_frame{};
        V delta;const double* f;int pdg=11;
        if(exact){
            if(std::uint32_t(cursor)>=model.graph_count){result.status=4;result.reason=1;break;}
            g=model.graph+cursor;
            if(!g->inside){result.status=3;break;}
            delta=rotate(target,project(original,g->delta));f=g->fraction;pdg=g->pdg;
            ++result.graph_advances;
        }else{
            std::uint32_t lo=model.continuing,hi=lo+model.entry;
            if(iteration || !model.entry){
                if(!(energy>0 && energy<=model.edges[model.bins]*(1+1e-7))){result.status=4;result.reason=2;break;}
                std::uint32_t a=0,b=model.bins;
                while(a<b){auto mid=a+(b-a)/2;if(model.edges[mid+1]<=energy)a=mid+1;else b=mid;}
                auto bin=model.nearest[a<model.bins?a:model.bins-1];
                lo=model.offsets[bin];hi=model.offsets[bin+1];
                if(energy<model.edges[0])++result.below;
            }
            const auto coin=uniform(iteration,0);
            while(lo+1<hi){auto mid=lo+(hi-lo)/2;if(coin<model.steps[mid-1].cdf)hi=mid;else lo=mid;}
            s=model.steps+lo;
            result.sampled_row=lo;
            if((iteration || !model.entry) && energy<model.edges[0] && s->fraction[0]<1-1e-7){result.status=4;result.reason=3;break;}
            sample_frame=azimuth(d,6.2831853071795864769*uniform(iteration,1));
            delta=rotate(sample_frame,s->delta);f=s->fraction;
        }
        if(pdg==11 && (exact?g->length:s->length)>.250001){result.status=4;result.reason=4;break;}
        const auto clipped=boundary(p,delta),coin=uniform(iteration,2);
        const bool leaves=clipped<1.;
        const auto deposited=f[0]*(pdg==11?clipped:(leaves?0.:1.));
        ++result.advances;
        if(coin<deposited){
            p=p+(pdg==11?uniform(iteration,3)*clipped:1.)*delta;
            result.status=photon?2:0;break;
        }
        if(leaves){p=p+clipped*delta;result.status=1;break;}
        p=p+delta;
        unsigned channel=1;double cumulative=f[0]+f[1];
        while(channel<3 && coin>=cumulative)cumulative+=f[++channel];
        if(!(f[channel]>0)){result.status=4;result.reason=5;break;}
        energy*=f[channel];
        if(channel==1){
            if(exact){cursor=g->next;
                if(cursor==-2){
                    if(pdg!=11){result.status=4;result.reason=6;break;}
                    d=rotate(target,project(original,g->postdir));
                }else if(cursor<0){result.status=3;break;}}
            else d=rotate(sample_frame,s->post);
        }else{
            cursor=exact?g->child[channel-2]:s->child[channel-2];
            if(cursor<0 || std::uint32_t(cursor)>=model.graph_count){result.status=4;result.reason=7;break;}
            photon=photon || model.graph[cursor].pdg==22;
            if(!exact){original=basis(s->source);target=sample_frame;}
        }
    }
    result.x=p.x;result.y=p.y;result.z=p.z;result.energy=energy;result.cursor=cursor;return result;
}
} // namespace electron_joint_research
