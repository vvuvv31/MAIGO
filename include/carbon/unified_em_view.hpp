#pragma once
#include "carbon/unified_em_package.hpp"
#include "carbon/discrete_delta_candidate.hpp"
#include "carbon/restricted_fluctuation_candidate.hpp"
#include <algorithm>
#include <cmath>
#include <bit>
#include <limits>
// Compile-time performance ablations; defaults preserve the validated fast path.
#ifndef CARBON_EM_MATERIAL_CACHE
#define CARBON_EM_MATERIAL_CACHE 1
#endif
#ifndef CARBON_EM_STEP_CACHE
#define CARBON_EM_STEP_CACHE 1
#endif
#ifndef CARBON_EM_LOCAL_AUDIT
#define CARBON_EM_LOCAL_AUDIT 0
#endif
#ifndef CARBON_EM_EXACT_INDEX
#define CARBON_EM_EXACT_INDEX 0
#endif
namespace carbon {
// Coarse exponent buckets only narrow the search; original knots and arithmetic stay intact.
inline constexpr unsigned unified_em_index_stride=5*257;
inline std::vector<unsigned> build_unified_em_index(const UnifiedEmPackage& package) {
    std::vector<unsigned> index(package.records.size()*unified_em_index_stride);
    for(unsigned r=0;r<package.records.size();++r) {
        const auto& record=package.records[r];
        for(unsigned kind=0;kind<5;++kind) {
            unsigned cursor=0,count=kind==0?record.node_count:record.counts[kind-1];
            auto lower=[&](unsigned i){return kind==0?package.nodes[record.node_offset+i].energy:
                package.segments[record.offsets[kind-1]+i].lower;};
            for(unsigned bucket=0;bucket<=256;++bucket) {
                const float edge=bucket==0?0.f:bucket>=255?std::numeric_limits<float>::infinity():std::ldexp(1.f,int(bucket)-127);
                while(cursor<count && lower(cursor)<edge)++cursor;
                index[r*unified_em_index_stride+kind*257+bucket]=cursor;
            }
        }
    }
    return index;
}

// Values at one fixed step-start energy. Reuse across step selection, mean
// loss, fluctuations and the post-step delta acceptance without new searches.
struct UnifiedEmPointStep {
    float factor{},range{},stopping{},q2{},dispersion{},universal_dispersion{};
    bool ion_fluctuation{};
};
struct UnifiedEmStep { UnifiedEmPointStep lo,hi; };
struct UnifiedEmPoint {
    const UnifiedEmRecord* record{};
    const UnifiedEmNode* nodes{};
    const EmCubicSegmentCandidate<float>* segments{};
    float density_scale{1};
    const unsigned* index{};
    void bounds(unsigned kind,float x,unsigned& lo,unsigned& hi)const {
        if constexpr(CARBON_EM_EXACT_INDEX) {
            if(index && x>0 && std::isfinite(x)) {
                unsigned bucket=(std::bit_cast<unsigned>(x)>>23)&255;
                const auto* v=index+kind*257;
                lo=std::min(hi,v[bucket]>0?v[bucket]-1:0);
                hi=std::min(hi,v[bucket+1]);
            }
        }
    }
    bool covers(float kinetic)const {
        float e=kinetic/record->a;
        return e>=nodes[0].energy && e<=nodes[record->node_count-1].energy;
    }
    UnifiedEmNode at(float e)const {
        unsigned lo=0,hi=record->node_count-2;bounds(0,e,lo,hi);
        while(lo<hi){auto m=(lo+hi+1)/2;if(nodes[m].energy<=e)lo=m;else hi=m-1;}
        const auto& a=nodes[lo];const auto& b=nodes[lo+1];
        float w=std::clamp((e-a.energy)/(b.energy-a.energy),0.f,1.f);
        auto mix=[&](float x,float y){return x+w*(y-x);};
        return {e,mix(a.full,b.full),mix(a.restricted,b.restricted),mix(a.stopping,b.stopping),mix(a.range,b.range),mix(a.lambda,b.lambda),mix(a.factor,b.factor),mix(a.correction,b.correction),mix(a.dispersion,b.dispersion),mix(a.universal_dispersion,b.universal_dispersion),mix(a.q2,b.q2),a.model,a.fluctuation};
    }
    float raw(int kind,float x)const {
        const auto* v=segments+record->offsets[kind];unsigned lo=0,hi=record->counts[kind]-1;
        if(x<v[0].lower){
            if(kind==3)return 0;
            float w=std::max(0.f,x/v[0].lower);
            return v[0].y0*(kind==2?w*w:std::sqrt(w));
        }
        bounds(kind+1,x,lo,hi);
        while(lo<hi){auto m=(lo+hi+1)/2;if(v[m].lower<=x)lo=m;else hi=m-1;}
        return v[lo].value(x);
    }
    float factor(float kinetic)const {return at(kinetic/record->a).factor*density_scale;}
    float range(float kinetic)const {return raw(1,kinetic*record->ratio)/(factor(kinetic)*record->ratio);}
    float rate(float kinetic,float pre_factor)const {return std::max(0.f,pre_factor*raw(3,kinetic*record->ratio));}
    UnifiedEmPointStep prepare(float kinetic)const {
        const auto& r=*record;const auto n=at(kinetic/r.a);
        const float f=n.factor*density_scale;
        return {f,raw(1,kinetic*r.ratio)/(f*r.ratio),f*raw(0,kinetic*r.ratio),
                n.q2,n.dispersion*density_scale,n.universal_dispersion*density_scale,
                n.fluctuation==1};
    }
    float mean(float kinetic,float length,const UnifiedEmPointStep& pre)const {
        const auto& r=*record;const float f=pre.factor;
        return primary_restricted_mean_candidate(kinetic,length,pre.stopping,pre.range,
            [&](float x){return raw(2,x*f*r.ratio)/r.ratio;},
            [&](float mid,float loss,float h){
                if(!r.is_ion)return loss;
                const auto m=at(mid/r.a);
                if(mid*(938.272013f/r.mass)<=2.f){
                    if(r.z>2)return (m.stopping+m.correction)*density_scale*h;
                    return loss*m.q2/std::max(pre.q2,1e-30f);
                }
                return loss+m.correction*density_scale*h;
            },r.linear_limit).energy_loss;
    }
    float mean(float kinetic,float length)const {return mean(kinetic,length,prepare(kinetic));}
};
struct UnifiedEmState {
    UnifiedEmPoint lo,hi;
    float weight{},density{};int section{-2};bool valid{false};
    float mix(float a,float b)const{return a+weight*(b-a);}
    bool covers(float t)const{return valid && lo.covers(t) && hi.covers(t);}
    float range(float t)const{return mix(lo.range(t),hi.range(t));}
    float mean(float t,float h)const{return mix(lo.mean(t,h),hi.mean(t,h));}
    UnifiedEmStep prepare(float t)const {return {lo.prepare(t),hi.prepare(t)};}
    float mean(float t,float h,const UnifiedEmStep& pre)const {
        return mix(lo.mean(t,h,pre.lo),hi.mean(t,h,pre.hi));
    }
    float cut()const{return mix(lo.record->cut,hi.record->cut);}
    float peak()const{return mix(lo.record->peak,hi.record->peak)*lo.record->a;}
    float step_from_range(float r)const {
        float f=lo.record->step_fraction,x=lo.record->final_range;
        return r>x?f*r+x*(1-f)*(2-x/r):r;
    }
    float step(float t)const {return step_from_range(range(t));}
    float step(const UnifiedEmStep& pre)const {return step_from_range(mix(pre.lo.range,pre.hi.range));}
    float research_step(float kinetic,const UnifiedEmStep& pre,float scale)const {
        const float native=step(pre);
        if(scale==1.f || kinetic/lo.record->a<20.f || density<.2f ||
           pre.lo.range<5.f || pre.hi.range<5.f)return native;
        return native*scale;
    }
    float rate(float t,float f0,float f1)const{return mix(lo.rate(t,f0),hi.rate(t,f1));}
};
struct UnifiedEmSectionRange { int begin{-1}, end{-1}; };
struct UnifiedEmDevice {
    const UnifiedEmMaterial* materials{};
    const UnifiedEmSpecies* species{};
    const UnifiedEmRecord* records{};
    const UnifiedEmNode* nodes{};
    const EmCubicSegmentCandidate<float>* segments{};
    unsigned material_count{};
    const UnifiedEmSectionRange* section_ranges{};
    const unsigned* energy_index{};
    int species_index(unsigned z,unsigned a)const {
        for(int i=0;i<18;++i)if(species[i].z==z && species[i].a==a)return i;
        return -1;
    }
    UnifiedEmState select(int section,float density,int ion)const {
        UnifiedEmState state;if(ion<0 || ion>=18 || section< -1 || section>24 || !(density>0))return state;
        int begin=-1,end=-1;
        if(CARBON_EM_MATERIAL_CACHE && section_ranges) {
            begin=section_ranges[section+1].begin;
            end=section_ranges[section+1].end;
        } else {
            // Host diagnostic views may omit the precomputed directory.
            for(unsigned i=0;i<material_count;++i)if(materials[i].section==section){if(begin<0)begin=i;end=i;}
        }
        if(begin<0 || density<materials[begin].density*(1-2e-6f) || density>materials[end].density*(1+2e-6f))return state;
        // Preserve the original strict '<' endpoint convention, including
        // densities exactly on a node. Do not change interpolation arithmetic.
        int low=begin;
        if constexpr(CARBON_EM_MATERIAL_CACHE) {
            int upper=end;
            while(low<upper) {
                int middle=low+(upper-low+1)/2;
                if(materials[middle].density<density)low=middle;
                else upper=middle-1;
            }
        } else {
            while(low<end && materials[low+1].density<density)++low;
        }
        int high=std::min(low+1,end);
        float w=high==low?0:std::clamp((density-materials[low].density)/(materials[high].density-materials[low].density),0.f,1.f);
        auto point=[&](int i){auto* r=records+i*18+ion;return UnifiedEmPoint{r,nodes+r->node_offset,segments,density/materials[i].density,energy_index?energy_index+(i*18+ion)*unified_em_index_stride:nullptr};};
        state.lo=point(low);state.hi=point(high);state.weight=w;state.density=density;state.section=section;state.valid=true;return state;
    }
};
struct UnifiedEmClock {
    DiscreteDeltaClockCandidate<float> clock;
    float threshold=std::numeric_limits<float>::max(),rate0{},rate1{},last_density{};
    int last_section=-2;
    float update(const UnifiedEmState& s,float kinetic,float f0,float f1){
        if(last_section!=s.section || last_density!=s.density){threshold=std::numeric_limits<float>::max();last_section=s.section;last_density=s.density;}
        float peak=s.peak();
        if(kinetic<=peak){if(kinetic*1.25f<threshold){rate0=s.lo.rate(kinetic,f0);rate1=s.hi.rate(kinetic,f1);threshold=(rate0+rate1)>0?kinetic:0;}}
        else if(kinetic<threshold){threshold=std::max(peak,kinetic*.8f);rate0=s.lo.rate(threshold,f0);rate1=s.hi.rate(threshold,f1);}
        return s.mix(rate0,rate1);
    }
};
struct UnifiedEmLoss {float loss{},continuous{},delta{};bool valid{false};unsigned proposed{},accepted{};};
template<class Uniform>
UnifiedEmLoss unified_em_explicit_loss(const UnifiedEmState& s,UnifiedEmClock& cache,float t,float h,float proposal_rate,float distance,bool fluctuations,const UnifiedEmStep& pre,Uniform& uniform){
    UnifiedEmLoss out;const auto& r=*s.lo.record;
    float mean=CARBON_EM_STEP_CACHE?s.mean(t,h,pre):s.mean(t,h),tau=t/r.mass,ratio=.51099891f/r.mass;
    float tmax=2*.51099891f*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);
    float cut=std::min(s.cut(),tmax);float loss=mean;
    if(fluctuations && mean<t){
        UnifiedEmPointStep fluct_lo=pre.lo,fluct_hi=pre.hi;
        if constexpr(!CARBON_EM_STEP_CACHE) {
            auto a=s.lo.at(t/r.a),b=s.hi.at(t/r.a);
            fluct_lo.dispersion=a.dispersion*s.lo.density_scale;
            fluct_hi.dispersion=b.dispersion*s.hi.density_scale;
            fluct_lo.universal_dispersion=a.universal_dispersion*s.lo.density_scale;
            fluct_hi.universal_dispersion=b.universal_dispersion*s.hi.density_scale;
            fluct_lo.ion_fluctuation=a.fluctuation==1;
        }
        RestrictedFluctuationInput<float> input{t,r.mass,mean,s.mix(fluct_lo.dispersion,fluct_hi.dispersion)*h,s.mix(fluct_lo.universal_dispersion,fluct_hi.universal_dispersion)*h,cut,tmax,s.mix(r.excitation,s.hi.record->excitation),s.mix(r.e0,s.hi.record->e0)};
        RestrictedFluctuationSampler<float,Uniform> sampler(uniform);auto draw=sampler.sample(input,fluct_lo.ion_fluctuation,float(r.z));
        if(!draw.valid)return out;
        loss=std::min(t,draw.loss);
    }
    if(t-loss<=r.lowest_kinetic)loss=t;
    out.continuous=loss;
    const bool selected=h>=distance;cache.clock.consume(h,proposal_rate,selected);out.proposed=selected;
    if(selected){
        cache.threshold=std::numeric_limits<float>::max();float post=t-loss;
        if(post>0 && uniform()*proposal_rate<s.rate(post,CARBON_EM_STEP_CACHE?pre.lo.factor:s.lo.factor(t),CARBON_EM_STEP_CACHE?pre.hi.factor:s.hi.factor(t))){
            float u=post/r.mass,beta2=u*(u+2)/((u+1)*(u+1));
            float max_transfer=2*.51099891f*u*(u+2)/(1+2*(u+1)*ratio+ratio*ratio);
            auto delta=sample_charged_delta_candidate(s.cut(),max_transfer,beta2,r.form_factor,r.mass,post,r.spin,r.magnetic_moment2,uniform);
            if(delta.status==DeltaDrawStatus::invalid || delta.status==DeltaDrawStatus::exhausted)return out;
            if(delta.status==DeltaDrawStatus::accepted){out.delta=std::min(post,delta.energy_MeV);loss+=out.delta;out.accepted=true;}
        }
    }
    out.loss=loss;out.valid=std::isfinite(loss) && loss>=0 && loss<=t;return out;
}
template<class Uniform>
UnifiedEmLoss unified_em_loss(const UnifiedEmState& s,UnifiedEmClock& cache,float t,float h,float proposal_rate,float distance,bool fluctuations,const UnifiedEmStep& pre,Uniform& uniform){
    return unified_em_explicit_loss(s,cache,t,h,proposal_rate,distance,fluctuations,pre,uniform);
}
} // namespace carbon
