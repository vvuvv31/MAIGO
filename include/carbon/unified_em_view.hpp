#pragma once
#include "carbon/delta_partition.hpp"
#include "carbon/delta_loss_sampling.hpp"
#include "carbon/unified_em_package.hpp"
#include "carbon/discrete_delta_candidate.hpp"
#include "carbon/restricted_fluctuation_candidate.hpp"
#include <algorithm>
#include <cmath>
#include <bit>
#include <limits>
#include <cstdint>
#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif
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
#ifndef CARBON_EM_EMPTY_BUCKET_FAST_PATH
#define CARBON_EM_EMPTY_BUCKET_FAST_PATH 0
#endif
#ifndef CARBON_EM_EMPTY_BUCKET_AUDIT
#define CARBON_EM_EMPTY_BUCKET_AUDIT 0
#endif
#ifndef CARBON_EM_SEARCH_KEY_AUDIT
#define CARBON_EM_SEARCH_KEY_AUDIT 0
#endif
#ifndef CARBON_EM_PAIR_PREPARE
#define CARBON_EM_PAIR_PREPARE 0
#endif
// Candidate A: split the binary-search keys out of the 52 B node records and
// 24 B segment records into contiguous float arrays so each cache line holds
// 16/32 keys instead of 2/5. Search arithmetic, interval choice and every
// downstream interpolation stay byte-for-byte identical; only the memory read
// during the search changes. 0 keeps the validated path.
#ifndef CARBON_EM_SPLIT_SEARCH_KEYS
#define CARBON_EM_SPLIT_SEARCH_KEYS 0
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
    float delta_stopping{},delta_variance{},stopping_slope{},delta_slope{},delta_partition_rate{};
};
struct UnifiedEmStep { UnifiedEmPointStep lo,hi; };
struct UnifiedEmSearchAuditRecord {
    std::uint64_t launch{};
    std::uint64_t key{};
    std::uint32_t warp{};
    std::uint32_t step{};
    std::uint16_t ordinal{};
    std::uint8_t kind{};
    std::uint8_t reserved{};
};
struct UnifiedEmSearchAuditState {
    UnifiedEmSearchAuditRecord* records{};
    std::uint32_t* count{};
    std::uint32_t capacity{};
    std::uint64_t launch{};
    std::uint32_t warp{};
    std::uint32_t step{};
    std::uint16_t ordinal{};
};
struct UnifiedEmPoint {
    const UnifiedEmRecord* record{};
    const UnifiedEmNode* nodes{};
    const EmCubicSegmentCandidate<float>* segments{};
    float density_scale{1};
    const unsigned* index{};
    const float* delta_means{};
    // Candidate A search keys (global node index / global segment index).
    const float* node_energy_keys{};
    const float* segment_lower_keys{};
#if CARBON_EM_EMPTY_BUCKET_AUDIT
    std::uint64_t* empty_bucket_audit{};
#endif
#if CARBON_EM_SEARCH_KEY_AUDIT
    UnifiedEmSearchAuditState* search_audit{};
#endif
    void bounds(unsigned kind,float x,unsigned& lo,unsigned& hi)const {
        if constexpr(CARBON_EM_EXACT_INDEX) {
            if(index && x>0 && std::isfinite(x)) {
                unsigned bucket=(std::bit_cast<unsigned>(x)>>23)&255;
                const auto* v=index+kind*257;
                const unsigned begin=v[bucket];
                const unsigned end=v[bucket+1];
                lo=std::min(hi,begin>0?begin-1:0);
                hi=std::min(hi,end);
#if CARBON_EM_EMPTY_BUCKET_AUDIT && defined(CARBON_HAS_SYCL)
                if(empty_bucket_audit) {
                    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        total(empty_bucket_audit[2*kind]);
                    total.fetch_add(1);
                    if(begin==end) {
                        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            empty(empty_bucket_audit[2*kind+1]);
                        empty.fetch_add(1);
                    }
                }
#endif
                if constexpr(CARBON_EM_EMPTY_BUCKET_FAST_PATH) {
                    if(begin==end)hi=lo;
                }
#if CARBON_EM_SEARCH_KEY_AUDIT && defined(CARBON_HAS_SYCL)
                if(search_audit && lo<hi) {
                    const unsigned m=(lo+hi+1)/2;
                    const auto* key_address=kind==0
                        ? static_cast<const void*>(nodes+m)
                        : static_cast<const void*>(segments+record->offsets[kind-1]+m);
                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        next(*search_audit->count);
                    const auto slot=next.fetch_add(1);
                    const auto ordinal=search_audit->ordinal++;
                    if(slot<search_audit->capacity)search_audit->records[slot]={
                        search_audit->launch,
                        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key_address)),
                        search_audit->warp,search_audit->step,ordinal,
                        static_cast<std::uint8_t>(kind),0};
                }
#endif
            }
        }
    }
    bool covers(float kinetic)const {
        float e=kinetic/record->a;
        return e>=nodes[0].energy && e<=nodes[record->node_count-1].energy;
    }
    UnifiedEmNode at(float e,float* delta_mean=nullptr,float* delta_variance=nullptr,float* delta_slope=nullptr)const {
        unsigned lo=0,hi=record->node_count-2;bounds(0,e,lo,hi);
        if constexpr(CARBON_EM_SPLIT_SEARCH_KEYS) {
            const float* key=node_energy_keys;
            if(key)while(lo<hi){auto m=(lo+hi+1)/2;if(key[m]<=e)lo=m;else hi=m-1;}
            else while(lo<hi){auto m=(lo+hi+1)/2;if(nodes[m].energy<=e)lo=m;else hi=m-1;}
        } else {
            while(lo<hi){auto m=(lo+hi+1)/2;if(nodes[m].energy<=e)lo=m;else hi=m-1;}
        }
        const auto& a=nodes[lo];const auto& b=nodes[lo+1];
        float w=std::clamp((e-a.energy)/(b.energy-a.energy),0.f,1.f);
        auto mix=[&](float x,float y){return x+w*(y-x);};
        if(delta_mean)*delta_mean=delta_means?mix(delta_means[2*lo],delta_means[2*lo+2]):0.f;
        if(delta_variance)*delta_variance=delta_means?mix(delta_means[2*lo+1],delta_means[2*lo+3]):0.f;
        if(delta_slope)*delta_slope=delta_means?(delta_means[2*lo+2]-delta_means[2*lo])/(b.energy-a.energy)/record->a:0.f;
        return {e,mix(a.full,b.full),mix(a.restricted,b.restricted),mix(a.stopping,b.stopping),mix(a.range,b.range),mix(a.lambda,b.lambda),mix(a.factor,b.factor),mix(a.correction,b.correction),mix(a.dispersion,b.dispersion),mix(a.universal_dispersion,b.universal_dispersion),mix(a.q2,b.q2),a.model,a.fluctuation};
    }
    float raw(int kind,float x,float* slope=nullptr)const {
        const auto* v=segments+record->offsets[kind];unsigned lo=0,hi=record->counts[kind]-1;
        const float* lk=nullptr;
        if constexpr(CARBON_EM_SPLIT_SEARCH_KEYS) lk=segment_lower_keys?segment_lower_keys+record->offsets[kind]:nullptr;
        if(x<v[0].lower){
            if(kind==3){if(slope)*slope=0;return 0;}
            float w=std::max(0.f,x/v[0].lower);
            if(slope)*slope=x>0?v[0].y0*(kind==2?2*w/v[0].lower:1/(2*std::sqrt(w)*v[0].lower)):0.f;
            return v[0].y0*(kind==2?w*w:std::sqrt(w));
        }
        bounds(kind+1,x,lo,hi);
        if(lk)while(lo<hi){auto m=(lo+hi+1)/2;if(lk[m]<=x)lo=m;else hi=m-1;}
        else while(lo<hi){auto m=(lo+hi+1)/2;if(v[m].lower<=x)lo=m;else hi=m-1;}
        if(slope)*slope=v[lo].derivative(x);
        return v[lo].value(x);
    }
    float factor(float kinetic)const {return at(kinetic/record->a).factor*density_scale;}
    float range(float kinetic)const {return raw(1,kinetic*record->ratio)/(factor(kinetic)*record->ratio);}
    float rate(float kinetic,float pre_factor)const {return std::max(0.f,pre_factor*raw(3,kinetic*record->ratio));}
    UnifiedEmPointStep prepare(float kinetic)const {
        const auto& r=*record;float delta=0,variance=0,delta_slope=0,raw_slope=0;const auto n=at(kinetic/r.a,&delta,&variance,&delta_slope);
        const float f=n.factor*density_scale;
        const float stop=f*raw(0,kinetic*r.ratio,&raw_slope);
        return {f,raw(1,kinetic*r.ratio)/(f*r.ratio),stop,
                n.q2,n.dispersion*density_scale,n.universal_dispersion*density_scale,
                n.fluctuation==1,delta*density_scale,variance*density_scale,f*r.ratio*raw_slope,delta_slope*density_scale,n.lambda*density_scale};
    }
    float mean(float kinetic,float length,const UnifiedEmPointStep& pre)const {
        const auto& r=*record;const float f=pre.factor;
        // Correct only the native linear restricted branch. Range inversion
        // already integrates energy dependence and must not be corrected twice.
        float stopping=pre.stopping;
        if(length<pre.range && stopping*length<=r.linear_limit*kinetic) {
            const float fraction=delta_partition_fraction(pre.delta_partition_rate*length);
            const float corrected=stopping-.5f*length*(stopping*fraction+pre.delta_stopping)*pre.stopping_slope;
            if(corrected>0 && corrected*length<=r.linear_limit*kinetic)stopping=corrected;
        }
        return primary_restricted_mean_candidate(kinetic,length,stopping,pre.range,
            [&](float x){return raw(2,x*f*r.ratio)/r.ratio;},
            [&](float mid,float loss,float h){
                if(!r.is_ion)return loss;
                mid=std::max(.5f*kinetic,mid-.5f*pre.delta_stopping*h);
                const auto m=at(mid/r.a);
                if(mid*(938.272013f/r.mass)<=2.f){
                    if(r.z>2)return (m.stopping+m.correction)*density_scale*h;
                    return loss*m.q2/std::max(pre.q2,1e-30f);
                }
                return loss+m.correction*density_scale*h;
            },r.linear_limit).energy_loss;
    }
    float mean(float kinetic,float length)const {return mean(kinetic,length,prepare(kinetic));}
    // Native restricted-only table probe, before condensed delta integration.
    float native_mean(float kinetic,float length)const {
        auto pre=prepare(kinetic);pre.stopping_slope=0;pre.delta_stopping=0;
        return mean(kinetic,length,pre);
    }
};
struct UnifiedEmDevice;
struct alignas(16) UnifiedEmState {
    const UnifiedEmDevice* tables{};
    const UnifiedEmRecord* host_record{};
    unsigned lo_mat{}, hi_mat{};
    int ion{-1};
    float weight{}, density{}, lo_scale{1}, hi_scale{1};
    int section{-2};
    bool valid{false};
#if CARBON_EM_SEARCH_KEY_AUDIT
    UnifiedEmSearchAuditState* search_audit{};
#endif
    UnifiedEmPoint lo() const;
    UnifiedEmPoint hi() const;
    const UnifiedEmRecord* record() const { return tables ? lo().record : host_record; }
    float mix(float a,float b)const{return a+weight*(b-a);}
    bool covers(float t)const{return valid && lo().covers(t) && hi().covers(t);}
    float range(float t)const{return mix(lo().range(t),hi().range(t));}
    float mean(float t,float h)const{return mix(lo().mean(t,h),hi().mean(t,h));}
    UnifiedEmStep prepare(float t)const {
        if constexpr(CARBON_EM_PAIR_PREPARE) {
            const auto low=lo();
            const auto high=hi();
            return {low.prepare(t),high.prepare(t)};
        }
        return {lo().prepare(t),hi().prepare(t)};
    }
    float mean(float t,float h,const UnifiedEmStep& pre)const {
        return mix(lo().mean(t,h,pre.lo),hi().mean(t,h,pre.hi));
    }
    float cut()const{const auto* a=record();const auto b=hi();return mix(a->cut,b.record->cut);}
    float peak()const{const auto* a=record();const auto b=hi();return mix(a->peak,b.record->peak)*a->a;}
    float step_from_range(float r)const {
        const auto* rec=record();
        float f=rec->step_fraction,x=rec->final_range;
        return r>x?f*r+x*(1-f)*(2-x/r):r;
    }
    float step(float t)const {return step_from_range(range(t));}
    float step(const UnifiedEmStep& pre)const {return step_from_range(mix(pre.lo.range,pre.hi.range));}
    float research_step(float kinetic,const UnifiedEmStep& pre,float scale)const {
        const float native=step(pre);
        const auto* rec=record();
        if(scale==1.f || kinetic/rec->a<20.f || density<.2f ||
           pre.lo.range<5.f || pre.hi.range<5.f)return native;
        return native*scale;
    }
    float rate(float t,float f0,float f1)const{return mix(lo().rate(t,f0),hi().rate(t,f1));}
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
    const float* delta_means{};
    // Candidate A search keys (global arrays; null disables the split path).
    const float* node_energy_keys{};
    const float* segment_lower_keys{};
#if CARBON_EM_EMPTY_BUCKET_AUDIT
    std::uint64_t* empty_bucket_audit{};
#endif
    int species_index(unsigned z,unsigned a)const {
        for(int i=0;i<18;++i)if(species[i].z==z && species[i].a==a)return i;
        return -1;
    }
    UnifiedEmPoint point(unsigned material,int ion,float density_scale
#if CARBON_EM_SEARCH_KEY_AUDIT
                         ,UnifiedEmSearchAuditState* search_audit=nullptr
#endif
                         )const {
        auto* r=records+material*18+ion;
        return {r,nodes+r->node_offset,segments,density_scale,
            energy_index?energy_index+(material*18+ion)*unified_em_index_stride:nullptr,
            delta_means?delta_means+2*r->node_offset:nullptr,
            node_energy_keys?node_energy_keys+r->node_offset:nullptr,
            segment_lower_keys
#if CARBON_EM_EMPTY_BUCKET_AUDIT
            ,empty_bucket_audit
#endif
#if CARBON_EM_SEARCH_KEY_AUDIT
            ,search_audit
#endif
            };
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
        state.tables=this;state.lo_mat=static_cast<unsigned>(low);state.hi_mat=static_cast<unsigned>(high);
        state.ion=ion;state.lo_scale=density/materials[low].density;state.hi_scale=density/materials[high].density;
        state.weight=w;state.density=density;state.section=section;state.valid=true;return state;
    }
};
inline UnifiedEmPoint UnifiedEmState::lo() const {
    return tables?tables->point(lo_mat,ion,lo_scale
#if CARBON_EM_SEARCH_KEY_AUDIT
        ,search_audit
#endif
        ):UnifiedEmPoint{};
}
inline UnifiedEmPoint UnifiedEmState::hi() const {
    return tables?tables->point(hi_mat,ion,hi_scale
#if CARBON_EM_SEARCH_KEY_AUDIT
        ,search_audit
#endif
        ):UnifiedEmPoint{};
}
struct UnifiedEmClock {
    DiscreteDeltaClockCandidate<float> clock;
    float threshold=std::numeric_limits<float>::max(),rate0{},rate1{},last_density{};
    int last_section=-2;
    float update(const UnifiedEmState& s,float kinetic,float f0,float f1){
        if(last_section!=s.section || last_density!=s.density){threshold=std::numeric_limits<float>::max();last_section=s.section;last_density=s.density;}
        float peak=s.peak();
        if(kinetic<=peak){if(kinetic*1.25f<threshold){rate0=s.lo().rate(kinetic,f0);rate1=s.hi().rate(kinetic,f1);threshold=(rate0+rate1)>0?kinetic:0;}}
        else if(kinetic<threshold){threshold=std::max(peak,kinetic*.8f);rate0=s.lo().rate(threshold,f0);rate1=s.hi().rate(threshold,f1);}
        return s.mix(rate0,rate1);
    }
};
struct UnifiedEmLoss {float loss{},continuous{},delta{};bool valid{false};unsigned proposed{},accepted{};};
template<class Uniform>
UnifiedEmLoss unified_em_explicit_loss(const UnifiedEmState& s,UnifiedEmClock& cache,float t,float h,float proposal_rate,float distance,bool fluctuations,const UnifiedEmStep& pre,Uniform& uniform){
    UnifiedEmLoss out;const auto lo=s.lo(),hi=s.hi();const auto& r=*lo.record;
    float mean=CARBON_EM_STEP_CACHE?s.mean(t,h,pre):s.mean(t,h),tau=t/r.mass,ratio=.51099891f/r.mass;
    float tmax=2*.51099891f*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);
    float cut=std::min(s.cut(),tmax);float loss=mean;
    if(fluctuations && mean<t){
        UnifiedEmPointStep fluct_lo=pre.lo,fluct_hi=pre.hi;
        if constexpr(!CARBON_EM_STEP_CACHE) {
            auto a=lo.at(t/r.a),b=hi.at(t/r.a);
            fluct_lo.dispersion=a.dispersion*lo.density_scale;
            fluct_hi.dispersion=b.dispersion*hi.density_scale;
            fluct_lo.universal_dispersion=a.universal_dispersion*lo.density_scale;
            fluct_hi.universal_dispersion=b.universal_dispersion*hi.density_scale;
            fluct_lo.ion_fluctuation=a.fluctuation==1;
        }
        RestrictedFluctuationInput<float> input{t,r.mass,mean,s.mix(fluct_lo.dispersion,fluct_hi.dispersion)*h,s.mix(fluct_lo.universal_dispersion,fluct_hi.universal_dispersion)*h,cut,tmax,s.mix(r.excitation,hi.record->excitation),s.mix(r.e0,hi.record->e0)};
        RestrictedFluctuationSampler<float,Uniform> sampler(uniform);auto draw=sampler.sample(input,fluct_lo.ion_fluctuation,float(r.z));
        if(!draw.valid)return out;
        loss=std::min(t,draw.loss);
    }
    if(t-loss<=r.lowest_kinetic)loss=t;
    out.continuous=loss;
    // Aggregate delta draw; native restricted fluctuation remains independent.
    const float delta0=s.mix(pre.lo.delta_stopping,pre.hi.delta_stopping);
    const float delta_slope=s.mix(pre.lo.delta_slope,pre.hi.delta_slope);
    const float delta_mean=std::max(0.f,delta0-.5f*(mean+delta0*h)*delta_slope)*h;
    if(!(delta_mean>=0 && std::isfinite(delta_mean)))return out;
    const float delta_variance=s.mix(pre.lo.delta_variance,pre.hi.delta_variance)*h;
    const float delta=fluctuations?delta_moment_draw(delta_mean,delta_variance,uniform):delta_mean;
    if(!(delta>=0 && std::isfinite(delta)))return out;
    out.delta=std::min(t-loss,delta);loss+=out.delta;
    out.loss=loss;out.valid=std::isfinite(loss) && loss>=0 && loss<=t;return out;
}
template<class Uniform>
UnifiedEmLoss unified_em_loss(const UnifiedEmState& s,UnifiedEmClock& cache,float t,float h,float proposal_rate,float distance,bool fluctuations,const UnifiedEmStep& pre,Uniform& uniform){
    return unified_em_explicit_loss(s,cache,t,h,proposal_rate,distance,fluctuations,pre,uniform);
}
} // namespace carbon
