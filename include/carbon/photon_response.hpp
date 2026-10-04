#pragma once
#include "carbon/photon_optical_transport.hpp"
#include "carbon/electron_joint_sampling.hpp"

namespace carbon {
struct PhotonRateRow { double energy{},rate[3]{}; };
struct PhotonOutcome { double cdf{},fraction{},cosine{};std::int32_t pdg{},pad{}; };
struct PhotonChannel { std::uint64_t offset{},count{}; };
struct PhotonResponseView {
    const PhotonRateRow* rates{};const double* edges{};
    const PhotonChannel* channels{};const PhotonOutcome* outcomes{};
    std::uint32_t rate_count{},bins{};std::uint64_t outcome_count{};
    int section{-1};double density{};
};
struct PhotonRates { bool valid{};double rate[3]{},total{}; };
inline PhotonRates photon_rates(PhotonResponseView model,int section,double density,double energy) {
    PhotonRates out;
    if(!model.rates || model.rate_count<2 || section!=model.section || !std::isfinite(density) || density<=0 ||
       !std::isfinite(model.density) || model.density<=0 || !std::isfinite(energy) ||
       energy<model.rates[0].energy || energy>model.rates[model.rate_count-1].energy)return out;
    unsigned lo=0,hi=model.rate_count-1;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(model.rates[m].energy>energy)hi=m;else lo=m;}
    const auto& a=model.rates[lo];const auto& b=model.rates[hi];
    const double f=(energy-a.energy)/(b.energy-a.energy),scale=density/model.density;
    for(unsigned i=0;i<3;++i) {
        out.rate[i]=((1-f)*a.rate[i]+f*b.rate[i])*scale;
        if(!std::isfinite(out.rate[i]) || out.rate[i]<0)return {};
        out.total+=out.rate[i];
    }
    out.valid=std::isfinite(out.total);return out;
}
struct PhotonCollisionDraw {bool valid{};std::int32_t pdg{};double energy{};electron_joint_research::V direction{};unsigned process{};};
// Energy-weighted single-branch estimator. Statistical W is unchanged by this
// function; physical continuation energy is E*f. A pdg=0 outcome deposits W
// exactly at the sampled interaction, never smeared along the photon flight.
// Every positive native branch is retained in the conditional CDF.
inline PhotonCollisionDraw sample_photon_collision(PhotonResponseView model,
    int section,double physical_density,double energy,electron_joint_research::V direction,
    double process_u,double outcome_u,double azimuth_u) {
    PhotonCollisionDraw out;
    if(!std::isfinite(process_u)||process_u<0||process_u>=1||!std::isfinite(outcome_u)||outcome_u<0||outcome_u>=1||
       !std::isfinite(azimuth_u)||azimuth_u<0||azimuth_u>=1||
       !std::isfinite(electron_joint_research::dot(direction,direction))||std::abs(electron_joint_research::dot(direction,direction)-1)>1e-9)return out;
    // The conditional collision outcomes use exact density nodes for now.
    // Density interpolation of outcomes is a separate, unvalidated extension.
    if(!model.channels||!model.outcomes||!model.edges||!model.bins||
       physical_density!=model.density||energy<model.edges[0]||energy>model.edges[model.bins])return out;
    const auto rates=photon_rates(model,section,physical_density,energy);if(!rates.valid||rates.total<=0)return out;
    const double coin=process_u*rates.total;
    unsigned process=coin<rates.rate[0]?0:(coin<rates.rate[0]+rates.rate[1]?1:2);
    if(rates.rate[process]<=0)return out;
    unsigned lo=0,hi=model.bins;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(model.edges[m]>energy)hi=m;else lo=m;}
    const auto channel=model.channels[3*lo+process];out.process=process;
    if(!channel.count || channel.offset>model.outcome_count || channel.count>model.outcome_count-channel.offset)return out;
    auto a=channel.offset,b=a+channel.count-1;
    while(a<b){const auto m=a+(b-a)/2;if(outcome_u<model.outcomes[m].cdf)b=m;else a=m+1;}
    const auto& selected=model.outcomes[a];
    out.pdg=selected.pdg;out.energy=selected.fraction*energy;
    if(out.pdg!=0) {
        const auto frame=electron_joint_research::azimuth(direction,6.2831853071795864769*azimuth_u);
        out.direction=std::sqrt(std::max(0.,1-selected.cosine*selected.cosine))*frame.u+selected.cosine*direction;
    }
    out.valid=out.energy>0;return out;
}
} // namespace carbon
