#pragma once
#include "carbon/photon_response.hpp"
namespace carbon {
struct PhotonResponseBankView {
    const PhotonResponseView* nodes{};unsigned count{};
    const unsigned* section_offsets{};
};
struct PhotonDensityBracket {bool valid{};unsigned low{},high{};double upper_weight{};};
inline PhotonDensityBracket photon_density_bracket(PhotonResponseBankView bank,int section,double density) {
    PhotonDensityBracket out;
    if(!bank.nodes||!bank.section_offsets||section<0||section>=25||!std::isfinite(density)||density<=0)return out;
    const auto first=bank.section_offsets[section],end=bank.section_offsets[section+1];
    if(first>=end||end>bank.count||density<bank.nodes[first].density||density>bank.nodes[end-1].density)return out;
    unsigned hi=first;while(hi+1<end&&bank.nodes[hi].density<density)++hi;
    const unsigned lo=bank.nodes[hi].density==density?hi:hi-1;
    out={true,lo,hi,lo==hi?0:(density-bank.nodes[lo].density)/(bank.nodes[hi].density-bank.nodes[lo].density)};return out;
}
inline PhotonAttenuation photon_bank_attenuation(PhotonResponseBankView bank,int section,double density,double energy) {
    const auto bracket=photon_density_bracket(bank,section,density);if(!bracket.valid)return {};
    const auto rates=photon_rates(bank.nodes[bracket.low],section,density,energy);
    return {rates.valid,rates.total};
}
enum class IndependentPhotonStatus { invalid, deposited, escaped, electron, coverage_missing };
struct IndependentPhotonResult {
    IndependentPhotonStatus status{IndependentPhotonStatus::invalid};
    std::array<double,3> position{},direction{};
    double energy{},physical_density{};
    unsigned node{};std::uint64_t interactions{},voxel_crossings{};
};
// Photon-only chain. A selected electron returns its complete physical state
// for charged transport. A selected photon child starts a new independent
// flight. No branch is locally deposited merely because it lacks a record.
// The caller retains the original energy weight W through every transition.
template<class Uniform>
inline IndependentPhotonResult transport_independent_photon(PhotonResponseBankView bank,
    const ElectronCtGeometry& geometry,std::array<double,3> position,
    std::array<double,3> direction,double energy,Uniform& uniform) {
    IndependentPhotonResult out;out.position=position;out.direction=direction;out.energy=energy;
    for(unsigned i=0;i<65536;++i) {
        const double u=uniform();if(!std::isfinite(u)||u<=0||u>=1)return out;
        auto rates=[&](int section,double rho,double e){return photon_bank_attenuation(bank,section,rho,e);};
        const auto flight=transport_photon_optical_depth(geometry,out.position,out.direction,out.energy,-std::log(u),rates);
        out.position=flight.position;out.voxel_crossings+=flight.voxel_crossings;out.physical_density=flight.density;
        if(flight.status==PhotonFlightStatus::invalid)return out;
        if(flight.status==PhotonFlightStatus::escaped){out.status=IndependentPhotonStatus::escaped;return out;}
        if(flight.status==PhotonFlightStatus::coverage_missing){out.status=IndependentPhotonStatus::coverage_missing;return out;}
        const auto bracket=photon_density_bracket(bank,flight.section,flight.density);
        if(!bracket.valid){out.status=IndependentPhotonStatus::coverage_missing;return out;}
        const double density_u=uniform();if(!std::isfinite(density_u)||density_u<0||density_u>=1)return out;
        out.node=density_u<bracket.upper_weight?bracket.high:bracket.low;
        const auto model=bank.nodes[out.node];
        // Optical depth used actual density. Collision final-state tables at
        // bracketing cut/density nodes are mixed with their geometric weights.
        // This finite-density interpolation remains an explicit approximation.
        const double process_u=uniform(),outcome_u=uniform(),azimuth_u=uniform();
        const auto draw=sample_photon_collision(model,model.section,model.density,out.energy,
            {out.direction[0],out.direction[1],out.direction[2]},process_u,outcome_u,azimuth_u);
        if(!draw.valid){out.status=IndependentPhotonStatus::coverage_missing;return out;}
        ++out.interactions;
        if(draw.pdg==0){out.status=IndependentPhotonStatus::deposited;return out;}
        out.energy=draw.energy;out.direction={draw.direction.x,draw.direction.y,draw.direction.z};
        if(draw.pdg==11){out.status=IndependentPhotonStatus::electron;return out;}
        if(draw.pdg!=22)return out;
    }
    return out; // iteration cap is a failure, never dose/escape
}
} // namespace carbon
