#pragma once
// Research energy-loss aggregation with every elastic collision explicit.
// Not connected to the production worker. New approximations: one Urban loss
// over a protected segment, pre-segment elastic rate, midpoint-loss angular
// energy. Photons/hard null tests retain their original endpoint treatment.
#include "carbon/electron_fp32.hpp"
#include "carbon/electron_range_guard.hpp"
#include "carbon/electron_batch_original.hpp"
namespace carbon::electron_elastic_batch_research {
namespace f=electron_fp32;
struct Result {f::ChargedStep step{};unsigned elastic_events{};bool batched{},guard_violation{};};
struct Bounds {bool valid{};float safety{};f::Position low{},high{};};
inline Bounds bounds(const f::ElectronCtGeometry& material,const f::ElectronCtGeometry& scoring,
    f::Position start,std::uint64_t material_cell){
    Bounds out;out.safety=INFINITY;
    const auto score=electron_range_guard_research::scoring_index(scoring,start);
    if(score==UINT64_MAX)return out;
    float coordinate_scale=1;
    for(unsigned k=0;k<3;++k){out.low[k]=-INFINITY;out.high[k]=INFINITY;coordinate_scale=std::max(coordinate_scale,std::abs(start[k]));}
    for(unsigned which=0;which<2;++which){
        const auto& g=which?scoring:material;auto cell=which?score:material_cell;
        for(unsigned k=0;k<3;++k){
            if(!g.bins[k]||g.bins[k]>16777215u||!(g.spacing[k]>0))return out;
            const auto coordinate=cell%g.bins[k];cell/=g.bins[k];
            coordinate_scale=std::max(coordinate_scale,std::max(std::abs(g.origin[k]),(coordinate+1)*g.spacing[k]));
            out.low[k]=std::max(out.low[k],g.origin[k]+coordinate*g.spacing[k]);
            out.high[k]=std::min(out.high[k],g.origin[k]+(coordinate+1)*g.spacing[k]);
        }
    }
    for(unsigned k=0;k<3;++k)out.safety=std::min(out.safety,std::min(start[k]-out.low[k],out.high[k]-start[k]));
    // Include both face reconstruction and (position-origin)/spacing in the
    // FP32 error budget. Near the origin of patient coordinates, abs(position)
    // alone misses cancellation against a distant grid origin. The supported
    // integer indices are exactly representable; retain independent runtime
    // index audits rather than claiming validation from this margin alone.
    const float rounding=8*std::numeric_limits<float>::epsilon()*coordinate_scale;
    out.safety=std::max(0.f,out.safety-rounding);out.valid=true;return out;
}
inline bool contains(const Bounds& b,f::Position p){
    for(unsigned k=0;k<3;++k)if(!std::isfinite(p[k])||p[k]<=b.low[k]||p[k]>=b.high[k])return false;
    return true;
}
inline std::array<float,3> rotate(std::array<float,3> direction,float q,float azimuth){
    const f::joint_math::V d{direction[0],direction[1],direction[2]};
    const auto frame=f::joint_math::azimuth(d,6.2831853071795864769f*azimuth);
    const auto next=f::joint_math::unit(std::sqrt(std::max(0.f,q*(2-q)))*frame.u+(1-q)*d);
    return {next.x,next.y,next.z};
}
template<class Model,class Uniform>[[gnu::always_inline]] inline Result transport_face(const Model& model,
    const f::ElectronCtGeometry& material,const f::ElectronCtGeometry& scoring,
    f::Position start,std::array<float,3> direction,float energy,
    const std::array<float,2>& majorants,const f::ChargedFace& face,Uniform& uniform,float loss_fraction=.005f){
    auto original=[&](const f::ChargedParameters* prepared=nullptr){Result r;
        r.step=prepared?f::elastic_batch_reference::original_prepared(model,face,start,direction,energy,majorants,*prepared,uniform,.05f,loss_fraction):
            f::voxel_cache_research::transport_step_face(model,face,start,direction,energy,majorants,uniform,.05f,loss_fraction);
        r.elastic_events=r.step.elastic;return r;};
    // Keep the original terminal/coverage logic and explicit face navigation.
    if(!std::isfinite(energy)||energy<=4*model.lowest_energy)return original();
    if(!face.valid||face.escaped||face.section!=model.section||face.density!=model.density)return original();
    const auto p=f::charged_parameters_certified(model,energy);
    if(!p.valid)return original(&p);
    const auto box=bounds(material,scoring,start,face.voxel);
    const float nominal=std::min(.05f,std::min(loss_fraction*energy/p.dedx,p.range));
    const float allowed=std::min(nominal,box.safety);
    if(!box.valid||!(allowed>0)||allowed<.25f*nominal||allowed*p.elastic_rate<1)return original(&p);
    Result result;result.batched=true;auto& out=result.step;out.position=start;out.direction=direction;out.energy=energy;
    const float hard=majorants[0]+majorants[1];
    if(!std::isfinite(hard)||hard<0)return result;
    const float u=uniform();if(!(u>0&&u<1))return result;
    const float flight=hard>0?-std::log(u)/hard:INFINITY;
    out.length=std::min(allowed,flight);
    if(!(out.length>0)||out.length>.25f)return result;
    const auto loss=f::sample_charged_loss_prepared(model,energy,out.length,uniform,p);
    if(!loss.valid){out.status=f::ChargedStepStatus::coverage_missing;return result;}
    out.raw_loss=loss.raw;out.physical_deposit=loss.deposit;out.stopped=loss.stopped;
    const float branch=uniform();if(!(branch>=0&&branch<1))return result;
    const bool deposited=branch<loss.deposit/energy||loss.stopped;
    float limit=out.length;
    if(deposited){const float along=uniform();if(!(along>=0&&along<1))return result;limit*=along;}
    // Uniform loss location follows the explicit CURVED trajectory; it is
    // never replaced by the straight chord or by initial-position scoring.
    // Midpoint energy is an approximation to be tested, not a cutoff/clamp.
    const float angular_energy=energy-.5f*loss.deposit;
    if(!(angular_energy>=model.lowest_energy))return result;
    f::Position offset{};float along=0;bool finished=false;
    for(unsigned n=0;n<10000;++n){
        const float elastic_u=uniform();if(!(elastic_u>0&&elastic_u<1))return result;
        const float next=along-std::log(elastic_u)/p.elastic_rate;
        const float end=std::min(limit,next),length=end-along;
        for(unsigned k=0;k<3;++k){offset[k]+=length*out.direction[k];out.position[k]=start[k]+offset[k];}
        // All vertices lie inside the same two convex cells; consequently
        // every represented straight subleg stays inside as well. Any
        // violation rejects the experiment, never drops or snaps the packet.
        if(!contains(box,out.position)){result.guard_violation=true;return result;}
#ifdef CARBON_ELASTIC_BATCH_GEOMETRY_AUDIT
        // Independently check actual FP32 scorer/material indexing, including
        // cancellation between grid origin and coordinate. Bounds alone are
        // not evidence that the runtime floor convention kept the same cell.
        if(electron_range_guard_research::scoring_index(scoring,out.position)!=
               electron_range_guard_research::scoring_index(scoring,start)||
           electron_range_guard_research::scoring_index(material,out.position)!=face.voxel){
            result.guard_violation=true;return result;
        }
#endif
        if(end==limit){finished=true;break;}
        const auto scatter=f::sample_electron_elastic(model.elastic,angular_energy,uniform);
        if(!scatter.valid)return result;
        out.direction=rotate(out.direction,scatter.one_minus_cosine,float(uniform()));
        ++result.elastic_events;out.elastic=true;along=end;
    }
    if(!finished)return result;
    if(deposited){out.status=f::ChargedStepStatus::deposited;return result;}
    out.energy=loss.remaining;
    if(out.length!=flight){out.status=f::ChargedStepStatus::continued;return result;}
    out.hard_proposal=true;out.process=float(uniform())*hard<majorants[0]?0:1;
    const float raw=f::charged_raw_rate(model,out.process,out.energy);
    if(!std::isfinite(raw))return result;
    const float rate=std::max(0.f,raw);if(rate>majorants[out.process])return result;
    if(float(uniform())*majorants[out.process]>=rate){out.status=f::ChargedStepStatus::continued;return result;}
    const auto collision=f::sample_charged_collision(model,out.energy,out.process,uniform());
    if(!collision.valid){out.status=f::ChargedStepStatus::coverage_missing;return result;}
    out.hard_collision=true;out.energy=collision.energy;
    if(collision.pdg==0){out.status=f::ChargedStepStatus::deposited;return result;}
    out.direction=rotate(out.direction,1-collision.cosine,float(uniform()));
    out.status=collision.pdg==22?f::ChargedStepStatus::photon:f::ChargedStepStatus::continued;
    return result;
}
template<class Model,class Uniform>inline Result transport(const Model& model,
    const f::ElectronCtGeometry& material,const f::ElectronCtGeometry& scoring,
    f::Position start,std::array<float,3> direction,float energy,
    const std::array<float,2>& majorants,Uniform& uniform,float loss_fraction=.005f){
    const auto face=f::next_charged_face(material,start,direction);
    return transport_face(model,material,scoring,start,direction,energy,majorants,face,uniform,loss_fraction);
}
} // namespace carbon::electron_elastic_batch_research
