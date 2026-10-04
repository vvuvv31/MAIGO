#pragma once
#include "carbon/electron_elastic_batch.hpp"
namespace carbon::electron_elastic_batch_research {
struct Work {unsigned clock_ticks{1},batched{},elastic{},guard_violation{};};
// Research worker adapter. Same packet energy estimator and hard/photon
// statuses as the current FP32 worker; no new child queue or local cutoff.
template<class Uniform,class Tally>inline f::ElectronEnergyPacket transport_packet(
    f::ElectronEnergyPacket packet,f::ChargedBankView bank,
    const f::ElectronCtGeometry& geometry,const f::ElectronCtGeometry& scoring,
    f::voxel_cache_research::Cache& cache,Uniform& uniform,Tally& tally,Work& work){
    work={};tally(0,1);
    const auto face=f::voxel_cache_research::cached_face(geometry,packet.cursor.position,packet.cursor.direction,cache);
    if(!face.valid){packet.status=f::ElectronPacketStatus::invalid;tally(7,1);return packet;}
    if(face.escaped){packet.status=f::ElectronPacketStatus::escaped;return packet;}
    if(!f::voxel_cache_research::prepare_material(bank,geometry,face,cache)){
        packet.status=f::ElectronPacketStatus::invalid;tally(6,1);return packet;
    }
    const auto proposed=transport_face(cache.model,geometry,scoring,packet.cursor.position,
        packet.cursor.direction,packet.cursor.energy_MeV,cache.majorant,face,uniform);
    const auto& r=proposed.step;
    work.batched=proposed.batched;work.elastic=proposed.elastic_events;work.guard_violation=proposed.guard_violation;
    // One original call for the segment endpoint, plus its interior elastic
    // events. This is a defined retry clock, not a claim of identical steps.
    work.clock_ticks=proposed.batched?1+proposed.elastic_events:1;
    packet.cursor.section=face.section;packet.cursor.density_g_cm3=face.density;
    packet.cursor.physical_density_g_cm3=face.density;packet.cursor.position=r.position;
    packet.cursor.direction=r.direction;packet.cursor.energy_MeV=r.energy;
    packet.cursor.row=carbon::kElectronContinuationEnd;++packet.advances;
    tally(1,proposed.elastic_events);tally(2,r.hard_proposal);tally(3,r.hard_collision);tally(4,r.face);tally(5,r.stopped);
    if(r.status==f::ChargedStepStatus::continued)packet.cursor.pdg=11;
    else if(r.status==f::ChargedStepStatus::photon){packet.cursor.pdg=22;++packet.photon_children;}
    else if(r.status==f::ChargedStepStatus::deposited){packet.deposit_position=r.position;packet.status=f::ElectronPacketStatus::deposited;}
    else if(r.status==f::ChargedStepStatus::escaped)packet.status=f::ElectronPacketStatus::escaped;
    else{packet.status=f::ElectronPacketStatus::invalid;tally(r.status==f::ChargedStepStatus::coverage_missing?6:7,1);}
    return packet;
}
} // namespace carbon::electron_elastic_batch_research
