#pragma once
#include "carbon/electron_energy_lineage.hpp"
#include "carbon/material_electron_density.hpp"

namespace carbon {
// W is an energy-valued statistical packet, distinct from the sampled
// particle's kinetic energy. Exactly one terminal bucket owns W. Pending
// states retain the cursor and W; they are NOT dose or patient escape.
enum class ElectronPacketStatus { active, deposited, escaped, coverage_missing, invalid };
enum class ElectronPacketGap { none, material, electron_energy, photon_continuation };
struct ElectronEnergyPacket {
    ElectronPacketStatus status{ElectronPacketStatus::invalid};
    ElectronPacketGap gap{ElectronPacketGap::none};
    ElectronContinuationCursor cursor{};
    double weight_MeV{};
    std::array<double,3> deposit_position{};
    std::uint64_t advances{},electron_children{},photon_children{},source_restarts{},boundary_restarts{};
    std::size_t material_table_index{std::numeric_limits<std::size_t>::max()};
};

// Same-material finite electron sources are continued by the existing exact
// continuous-energy crossing law. No KE clamp/scale, material alias, local
// dump, or finite-source -> patient-escape conversion. Photon energy is
// discrete: the electron crossing index MUST NOT be used for photon flights.
template<class Uniform>
inline ElectronEnergyPacket resume_electron_packet(
    ElectronEnergyPacket packet,const MaterialElectronResponseView& target,Uniform& uniform,
    double physical_density=0) {
    if(packet.cursor.pdg==22) {
        packet.status=ElectronPacketStatus::coverage_missing;
        packet.gap=ElectronPacketGap::photon_continuation;return packet;
    }
    if(packet.cursor.pdg!=11) {packet.status=ElectronPacketStatus::invalid;return packet;}
    const auto sampled=sample_material_electron_continuation(target,11,packet.cursor.energy_MeV,uniform);
    if(sampled.status!=ElectronCrossingStatus::ready) {
        if(sampled.status==ElectronCrossingStatus::invalid || sampled.status==ElectronCrossingStatus::unsupported_particle) {
            packet.status=ElectronPacketStatus::invalid;return packet;
        }
        packet.status=ElectronPacketStatus::coverage_missing;
        packet.gap=ElectronPacketGap::electron_energy;return packet;
    }
    const auto rebound=bind_electron_continuation(sampled,target,packet.cursor.position,
                                                 packet.cursor.direction,uniform());
    if(!rebound.valid) {packet.status=ElectronPacketStatus::invalid;return packet;}
    packet.cursor=rebound;packet.cursor.physical_density_g_cm3=physical_density;
    packet.status=ElectronPacketStatus::active;
    packet.gap=ElectronPacketGap::none;return packet;
}

// One bounded transaction, callable from a GPU packet queue. Active output
// is requeued with unchanged W, never counted as a deposit. A caller-imposed
// observation/queue limit retains active packets, not an artificial terminal.
template<class Uniform>
inline ElectronEnergyPacket transport_electron_packet_step(
    ElectronEnergyPacket packet,const MaterialElectronResponseView* tables,std::size_t count,
    const ElectronCtGeometry& geometry,Uniform& uniform) {
    if(packet.status!=ElectronPacketStatus::active)return packet;
    if(!packet.cursor.valid || !std::isfinite(packet.weight_MeV) || packet.weight_MeV<=0 ||
       !std::isfinite(packet.cursor.energy_MeV) || packet.cursor.energy_MeV<=0 || !tables) {
        packet.status=ElectronPacketStatus::invalid;return packet;
    }
    std::size_t current=packet.material_table_index;
    if(current==std::numeric_limits<std::size_t>::max()) {
        current=count;
        for(std::size_t t=0;t<count;++t)
            if(tables[t].section==packet.cursor.section && tables[t].density_g_cm3==packet.cursor.density_g_cm3)current=t;
    } else if(current>=count || tables[current].section!=packet.cursor.section ||
              tables[current].density_g_cm3!=packet.cursor.density_g_cm3) {
        packet.status=ElectronPacketStatus::invalid;return packet;
    }
    if(current==count) {
        packet.status=ElectronPacketStatus::coverage_missing;packet.gap=ElectronPacketGap::material;return packet;
    }
    packet.material_table_index=current;
    if(packet.cursor.row==kElectronContinuationEnd) {
        packet=resume_electron_packet(packet,tables[current],uniform,packet.cursor.physical_density_g_cm3);
        if(packet.status==ElectronPacketStatus::active)++packet.source_restarts;
        return packet;
    }
    const auto advanced=advance_electron_continuation(packet.cursor,tables[current],geometry);
    // Sequence RNG explicitly; function argument evaluation order is not fixed.
    const double choice=uniform(),along=uniform();
    const auto selected=choose_electron_energy_lineage(packet.cursor,advanced,tables[current],choice,along);
    ++packet.advances;
    switch(selected.status) {
    case ElectronLineageStatus::deposited:
        packet.deposit_position=selected.deposit_position;
        packet.status=ElectronPacketStatus::deposited;return packet;
    case ElectronLineageStatus::escaped:
        packet.cursor=selected.cursor;packet.status=ElectronPacketStatus::escaped;return packet;
    case ElectronLineageStatus::electron_child:
        ++packet.electron_children;packet.cursor=selected.cursor;return packet;
    case ElectronLineageStatus::photon_pending:
        // The historical status name identifies PDG22, not a transport stop.
        ++packet.photon_children;packet.cursor=selected.cursor;return packet;
    case ElectronLineageStatus::parent_continuation:
    case ElectronLineageStatus::source_exhausted:
        packet.cursor=selected.cursor;return packet;
    case ElectronLineageStatus::material_boundary: {
        packet.cursor=selected.cursor;
        const auto bracket=material_electron_density_bracket(advanced.boundary.to_section,
            advanced.boundary.to_density,tables,count);
        if(!bracket.valid) {
            packet.status=ElectronPacketStatus::coverage_missing;packet.gap=ElectronPacketGap::material;return packet;
        }
        const double mix=uniform();
        if(!std::isfinite(mix) || mix<0 || mix>=1) {packet.status=ElectronPacketStatus::invalid;return packet;}
        const auto destination=mix<bracket.upper_weight?bracket.high:bracket.low;
        packet=resume_electron_packet(packet,tables[destination],uniform,advanced.boundary.to_density);
        packet.material_table_index=destination;
        if(packet.status==ElectronPacketStatus::active)++packet.boundary_restarts;
        return packet;
    }
    default:packet.status=ElectronPacketStatus::invalid;return packet;
    }
}
} // namespace carbon
