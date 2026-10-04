#pragma once
#include "carbon/electron_charged_bank.hpp"
#include "carbon/electron_charged_step.hpp"
#include "carbon/electron_packet_state.hpp"
namespace carbon {
template<class Uniform,class Tally,class Score=NoExpectedScore>
inline ElectronEnergyPacket transport_independent_charged_packet(ElectronEnergyPacket packet,
    ChargedBankView bank,const ElectronCtGeometry& geometry,Uniform& uniform,Tally& tally,
    Score&& score=Score{},double expected_density=-1.,double expected_min_energy=0.){
    tally(0,1);
    const auto face=next_charged_face(geometry,packet.cursor.position,packet.cursor.direction);
    if(!face.valid){packet.status=ElectronPacketStatus::invalid;tally(7,1);return packet;}
    if(face.escaped){packet.status=ElectronPacketStatus::escaped;return packet;}
    const auto bracket=charged_density_bracket(bank,face.section,face.density);
    if(!bracket.valid){packet.status=ElectronPacketStatus::invalid;tally(6,1);return packet;}
    const auto model=charged_density_model(bank.nodes+bracket.low,bank.nodes+bracket.high,face.density);
    // Expected scoring is selected by the current voxel; it never alters steps.
    const bool expected=face.density<=expected_density&&packet.cursor.energy_MeV>=expected_min_energy;
    const auto result=transport_charged_step(model,geometry,packet.cursor.position,packet.cursor.direction,
        packet.cursor.energy_MeV,charged_density_majorant(bank,bracket,face.density),uniform,.05,.005,expected);
    if(result.ev_fraction>0){
        // Exact telescoping split: scored + remaining == previous weight.
        const double w=packet.weight_MeV,rest=w-w*result.ev_fraction;
        score(result.ev_position,w-rest);packet.weight_MeV=rest;
    }
    packet.cursor.section=face.section;packet.cursor.density_g_cm3=face.density;packet.cursor.physical_density_g_cm3=face.density;
    packet.cursor.position=result.position;packet.cursor.direction=result.direction;packet.cursor.energy_MeV=result.energy;
    packet.cursor.row=kElectronContinuationEnd;++packet.advances;
    tally(1,result.elastic);tally(2,result.hard_proposal);tally(3,result.hard_collision);tally(4,result.face);tally(5,result.stopped);
    if(result.status==ChargedStepStatus::continued)packet.cursor.pdg=11;
    else if(result.status==ChargedStepStatus::photon){packet.cursor.pdg=22;++packet.photon_children;}
    else if(result.status==ChargedStepStatus::deposited){packet.deposit_position=result.position;packet.status=ElectronPacketStatus::deposited;}
    else if(result.status==ChargedStepStatus::escaped)packet.status=ElectronPacketStatus::escaped;
    else {packet.status=ElectronPacketStatus::invalid;tally(result.status==ChargedStepStatus::coverage_missing?6:7,1);}
    return packet;
}
} // namespace carbon
