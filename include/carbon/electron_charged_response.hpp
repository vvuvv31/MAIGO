#pragma once
#include "carbon/electron_elastic.hpp"
#include "carbon/electron_urban_fluctuation.hpp"
#include "carbon/electron_joint_sampling.hpp"
namespace carbon {
struct ChargedParameter {double energy{},elastic_mfp{},dedx{},range{},diagnostic_raw_rate[2]{};};
struct ChargedRatePiece {double low{},high{},value[4]{};};
struct ChargedOutcome {double cdf{},fraction{},cosine{};std::int32_t pdg{},pad{};};
struct ChargedChannel {std::uint64_t offset{},count{};};
struct ChargedResponseView {
    const ChargedParameter* parameters{};const ChargedRatePiece* rate[2]{};
    const double* edges{};const ChargedChannel* channels{};const ChargedOutcome* outcomes{};
    ElectronElasticView elastic{};ElectronUrbanMaterial urban{};
    unsigned parameter_count{},rate_count[2]{},bins{};std::uint64_t outcome_count{};
    int section{-1};double density{},electron_cut{},photon_cut{},lowest_energy{},linear_loss_limit{};
};
inline double charged_raw_rate(const ChargedResponseView& v,unsigned process,double energy) {
    if(process>=2||!v.rate[process]||!v.rate_count[process]||!std::isfinite(energy))return NAN;
    const auto* rows=v.rate[process];const auto count=v.rate_count[process];
    if(energy<rows[0].low||energy>rows[count-1].high)return NAN;
    unsigned lo=0,hi=count;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(rows[m].low>energy)hi=m;else lo=m;}
    const auto& r=rows[lo];const double t=(energy-r.low)/(r.high-r.low);
    // Native cubic represented by its values at 1/8,3/8,5/8,7/8.
    constexpr double x[4]={.125,.375,.625,.875};double result=0;
    for(unsigned j=0;j<4;++j){double term=r.value[j];for(unsigned k=0;k<4;++k)if(k!=j)term*=(t-x[k])/(x[j]-x[k]);result+=term;}
    return result;
}
struct ChargedParameters {bool valid{};double elastic_rate{},dedx{},range{},hard_rate[2]{};};
inline ChargedParameters charged_parameters(const ChargedResponseView& v,double energy) {
    ChargedParameters out;
    if(!v.parameters||v.parameter_count<2||!std::isfinite(energy)||energy<v.parameters[0].energy||energy>v.parameters[v.parameter_count-1].energy)return out;
    unsigned lo=0,hi=v.parameter_count-1;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(v.parameters[m].energy>energy)hi=m;else lo=m;}
    const auto a=v.parameters[lo],b=v.parameters[hi];const double f=(energy-a.energy)/(b.energy-a.energy);
    out.elastic_rate=1/(a.elastic_mfp+f*(b.elastic_mfp-a.elastic_mfp));
    out.dedx=a.dedx+f*(b.dedx-a.dedx);out.range=a.range+f*(b.range-a.range);
    for(unsigned p=0;p<2;++p){const double r=charged_raw_rate(v,p,energy);if(!std::isfinite(r))return {};out.hard_rate[p]=std::max(0.,r);}
    out.valid=std::isfinite(out.elastic_rate)&&out.elastic_rate>0&&std::isfinite(out.dedx)&&out.dedx>0&&std::isfinite(out.range)&&out.range>0;return out;
}
// Approximate inverse of the monotonic native restricted-range samples.
// Return unsupported for sub-table ranges; never clamp energy to coverage.
inline double charged_inverse_range(const ChargedResponseView& v,double range) {
    if(!v.parameters||v.parameter_count<2||!std::isfinite(range)||range<v.parameters[0].range||range>v.parameters[v.parameter_count-1].range)return NAN;
    unsigned lo=0,hi=v.parameter_count-1;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(v.parameters[m].range>range)hi=m;else lo=m;}
    const auto a=v.parameters[lo],b=v.parameters[hi];return a.energy+(range-a.range)/(b.range-a.range)*(b.energy-a.energy);
}
struct ChargedLoss {bool valid{},stopped{};double mean{},raw{},deposit{},remaining{};};
// G4VEnergyLossProcess AlongStep accounting with PIXE/subcut disabled as in
// the extraction. Cut and MaxSecondaryEnergy are model-specific: PenIoni
// inherits E below/equal 100keV; MollerBhabha uses E/2 above 100keV.
// Only stopping/range interpolation is approximated here. An over-energy
// fluctuation is handled by the native finalT<=lowestEnergy rule, not redrawn.
template<class Uniform,class Model>inline ChargedLoss sample_charged_loss(const Model& v,double energy,double length,Uniform& uniform) {
    ChargedLoss out;
    if(!std::isfinite(energy)||energy<=0||!std::isfinite(length)||length<0||length>.25||!(v.lowest_energy>0))return out;
    // Physical termination applies even below the model's 10eV table minimum.
    if(energy<=v.lowest_energy){out.valid=out.stopped=true;out.mean=out.raw=out.deposit=energy;return out;}
    const auto p=charged_parameters(v,energy);if(!p.valid)return out;
    if(length>=p.range){out.valid=out.stopped=true;out.mean=out.raw=out.deposit=energy;return out;}
    if(length==0){out.valid=true;out.remaining=energy;return out;}
    out.mean=length*p.dedx;
    if(out.mean>v.linear_loss_limit*energy){
        const double remaining=charged_inverse_range(v,p.range-length);
        if(!std::isfinite(remaining))return out;
        const double loss=energy-remaining;if(loss>0)out.mean=loss;
    }
    out.raw=out.mean;
    if(out.mean>=energy)out.raw=energy;
    else {
        const double tmax=energy<=.1?energy:.5*energy;
        const auto sampled=sample_electron_urban_loss(v.urban,energy,out.mean,std::min(v.electron_cut,tmax),uniform);
        if(!sampled.valid)return out;out.raw=sampled.loss;
    }
    const double final_energy=energy-out.raw;
    if(final_energy<=v.lowest_energy){out.deposit=energy;out.remaining=0;out.stopped=true;}
    else {out.deposit=out.raw;out.remaining=final_energy;}
    out.valid=true;return out;
}
struct ChargedCollision {bool valid{};std::int32_t pdg{};double energy{},cosine{};unsigned process{};};
inline ChargedCollision sample_charged_collision(const ChargedResponseView& v,double energy,unsigned process,double u) {
    ChargedCollision out;out.process=process;
    if(process>=2||!std::isfinite(u)||u<0||u>=1||!std::isfinite(energy)||!v.edges||!v.bins||!v.channels||!v.outcomes||energy<v.edges[0]||energy>v.edges[v.bins])return out;
    const double rate=charged_raw_rate(v,process,energy);if(!std::isfinite(rate)||rate<=0)return out;
    unsigned lo=0,hi=v.bins;while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(v.edges[m]>energy)hi=m;else lo=m;}
    const auto ch=v.channels[2*lo+process];if(!ch.count||ch.offset>v.outcome_count||ch.count>v.outcome_count-ch.offset)return out;
    auto a=ch.offset,b=a+ch.count-1;while(a<b){const auto m=a+(b-a)/2;if(u<v.outcomes[m].cdf)b=m;else a=m+1;}
    const auto s=v.outcomes[a];out.energy=energy*s.fraction;out.cosine=s.cosine;out.pdg=s.pdg;out.valid=out.energy>0;return out;
}
} // namespace carbon
