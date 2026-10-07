#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_charged_density.hpp"
#include "carbon/electron_ct_boundary.hpp"
namespace carbon {
struct ChargedFace {
    bool valid{},escaped{};double distance{};std::array<double,3> position{};
    std::uint64_t voxel{};std::uint8_t section{};float density{};
};
// Actual ray distance to the next voxel face, before sampling any energy loss.
// Exact face ownership; simultaneous corner crossings use all tied axes.
inline ChargedFace next_charged_face(const ElectronCtGeometry& g,
    std::array<double,3> position,std::array<double,3> direction) {
    ChargedFace out;out.position=position;if(!g.density||!g.section)return out;
    std::int64_t cell[3];double next[3],face[3],norm=0;bool outward=false;
    for(unsigned k=0;k<3;++k){
        if(!g.bins[k]||!std::isfinite(g.origin[k])||!std::isfinite(g.spacing[k])||g.spacing[k]<=0||!std::isfinite(position[k])||!std::isfinite(direction[k]))return out;
        norm+=direction[k]*direction[k];double x=(position[k]-g.origin[k])/g.spacing[k];
        const double nearest=std::round(x);
        if(std::isfinite(nearest)&&nearest>=0&&nearest<=g.bins[k]&&position[k]==g.origin[k]+nearest*g.spacing[k])x=nearest;
        if(!std::isfinite(x)||x<0||x>g.bins[k])return out;
        cell[k]=static_cast<std::int64_t>(std::floor(x));
        if(direction[k]<0&&x==std::floor(x))--cell[k];if(direction[k]==0&&x==g.bins[k])--cell[k];
        if(cell[k]<0||cell[k]>=g.bins[k])outward=true;
        face[k]=g.origin[k]+(cell[k]+(direction[k]>0))*g.spacing[k];
        next[k]=direction[k]?(face[k]-position[k])/direction[k]:std::numeric_limits<double>::infinity();
    }
    if(std::abs(norm-1)>1e-9)return out;
    if(outward){out.valid=out.escaped=true;return out;}
    out.voxel=(std::uint64_t(cell[2])*g.bins[1]+cell[1])*g.bins[0]+cell[0];out.section=g.section[out.voxel];out.density=g.density[out.voxel];
    if(out.section>=kElectronMaterialCount||!std::isfinite(out.density)||out.density<=0)return out;
    out.distance=std::min(next[0],std::min(next[1],next[2]));if(!std::isfinite(out.distance)||out.distance<=0)return out;
    for(unsigned k=0;k<3;++k)out.position[k]=next[k]==out.distance?face[k]:position[k]+out.distance*direction[k];
    out.valid=true;return out;
}
enum class ChargedStepStatus {invalid,continued,deposited,photon,escaped,coverage_missing};
struct ChargedStep {
    ChargedStepStatus status{ChargedStepStatus::invalid};std::array<double,3> position{},direction{};
    double energy{},length{},raw_loss{},physical_deposit{};
    bool face{},elastic{},hard_proposal{},hard_collision{},stopped{};unsigned process{};
};
// Single-energy-weighted-packet step: W remains unchanged. Sample the branch
// using soft-loss/E, then all positive hard parent/child/local outcomes.
// No queue is needed for this unbiased single-branch energy estimator.
// New approximation: elastic rate frozen at pre-step E, with a mean-loss
// limiter (default 0.5%) and explicit single scattering. Convergence and
// native interface-dose validation are required before CT promotion.
// Hard events use a global energy-domain majorant and exact endpoint-rate
// null test, so losing energy cannot request an unsupported hard channel.
template<class Uniform,class Model>inline ChargedStep transport_charged_step(const Model& model,
    const ElectronCtGeometry& geometry,std::array<double,3> start,std::array<double,3> direction,
    double energy,const std::array<double,2>& hard_majorant,Uniform& uniform,
    double maximum_step=.05,double mean_loss_fraction=.005) {
    ChargedStep out;out.position=start;out.direction=direction;out.energy=energy;
    if(!std::isfinite(energy)||energy<=0||!std::isfinite(maximum_step)||maximum_step<=0||maximum_step>.25||!std::isfinite(mean_loss_fraction)||mean_loss_fraction<=0||mean_loss_fraction>.01)return out;
    const auto face=next_charged_face(geometry,start,direction);if(!face.valid)return out;
    if(face.escaped){out.status=ChargedStepStatus::escaped;return out;}
    // This isolated primitive uses exact native density nodes. Main-runtime
    // density interpolation must be supplied and independently validated.
    if(face.section!=model.section||double(face.density)!=model.density){out.status=ChargedStepStatus::coverage_missing;return out;}
    if(energy<=model.lowest_energy){out.physical_deposit=energy;out.stopped=true;out.status=ChargedStepStatus::deposited;return out;}
    const auto p=charged_parameters(model,energy);if(!p.valid){out.status=ChargedStepStatus::coverage_missing;return out;}
    double hard=0;for(unsigned k=0;k<2;++k){if(!std::isfinite(hard_majorant[k])||hard_majorant[k]<p.hard_rate[k]||hard_majorant[k]<0)return out;hard+=hard_majorant[k];}
    const double rate=p.elastic_rate+hard;
    const double u=uniform();if(!std::isfinite(u)||u<=0||u>=1)return out;
    const double flight=-std::log(u)/rate;
    out.length=std::min(std::min(maximum_step,mean_loss_fraction*energy/p.dedx),std::min(p.range,std::min(face.distance,flight)));
    if(!std::isfinite(out.length)||out.length<=0)return out;
    out.face=out.length==face.distance;
    if(out.face)out.position=face.position;else for(unsigned k=0;k<3;++k)out.position[k]=start[k]+out.length*direction[k];
    const auto loss=sample_charged_loss(model,energy,out.length,uniform);
    if(!loss.valid){out.status=ChargedStepStatus::coverage_missing;return out;}
    out.raw_loss=loss.raw;out.physical_deposit=loss.deposit;out.stopped=loss.stopped;
    const double branch=uniform();if(!std::isfinite(branch)||branch<0||branch>=1)return out;
    if(branch<loss.deposit/energy||loss.stopped){
        const double along=uniform();if(!std::isfinite(along)||along<0||along>=1)return out;
        for(unsigned k=0;k<3;++k)out.position[k]=start[k]+along*out.length*direction[k];
        out.status=ChargedStepStatus::deposited;return out;
    }
    out.energy=loss.remaining;
    if(out.length!=flight||out.face){out.status=ChargedStepStatus::continued;return out;}
    const double event=uniform();if(!std::isfinite(event)||event<0||event>=1)return out;
    const double choice=event*rate;double cosine;
    if(choice<p.elastic_rate){
        const auto sampled=sample_electron_elastic(model.elastic,out.energy,uniform);if(!sampled.valid)return out;
        cosine=sampled.cosine;out.elastic=true;
    }else{
        out.hard_proposal=true;out.process=choice<p.elastic_rate+hard_majorant[0]?0:1;
        const double raw_rate=charged_raw_rate(model,out.process,out.energy);if(!std::isfinite(raw_rate))return out;
        const double effective=std::max(0.,raw_rate);
        if(!std::isfinite(effective)||effective>hard_majorant[out.process])return out;
        const double accept=uniform();if(!std::isfinite(accept)||accept<0||accept>=1)return out;
        if(accept*hard_majorant[out.process]>=effective){out.status=ChargedStepStatus::continued;return out;}
        const auto sampled=sample_charged_collision(model,out.energy,out.process,uniform());
        if(!sampled.valid){out.status=ChargedStepStatus::coverage_missing;return out;}
        out.hard_collision=true;out.energy=sampled.energy;cosine=sampled.cosine;
        if(sampled.pdg==0){out.status=ChargedStepStatus::deposited;return out;}
        if(sampled.pdg==22)out.status=ChargedStepStatus::photon;
    }
    const double azimuth=uniform();if(!std::isfinite(azimuth)||azimuth<0||azimuth>=1)return out;
    const electron_joint_research::V dir{direction[0],direction[1],direction[2]};
    const auto frame=electron_joint_research::azimuth(dir,6.2831853071795864769*azimuth);
    const auto next=std::sqrt(std::max(0.,1-cosine*cosine))*frame.u+cosine*dir;out.direction={next.x,next.y,next.z};
    if(out.status!=ChargedStepStatus::photon)out.status=ChargedStepStatus::continued;return out;
}
} // namespace carbon
