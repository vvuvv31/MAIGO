#pragma once
#include "carbon/electron_packet_transport.hpp"

namespace carbon {
inline double electron_short_range_length(const ElectronContinuationView& view,std::uint64_t row) {
    double length=0;
    for(unsigned i=0;i<64;++i) {
        const auto step=prepare_electron_replay_step(view,row);
        if(step.status!=ElectronReplayStatus::ready || step.state.pdg!=11 ||
           step.children_begin!=step.children_end || step.state.post_material!=step.state.pre_material ||
           step.state.post_density_g_cm3!=step.state.pre_density_g_cm3)return -1;
        double chord2=0;
        for(unsigned k=0;k<3;++k) {
            const double d=step.state.post_position[k]-step.state.pre_position[k];chord2+=d*d;
        }
        length+=std::max(step.state.length_mm,std::sqrt(chord2));
        if(!std::isfinite(length) || length>=0.25)return -1;
        if(step.next==kElectronContinuationEnd)
            return step.state.post_energy_MeV==0 && step.state.track_status==2?length:-1;
        row=step.next;
    }
    return -1;
}
// Conservative research shortcut at a packet birth. Never infer range from W.
// Bounded lookahead falls back to the original transport on any uncertainty.
inline bool electron_short_range_contained(const ElectronEnergyPacket& packet,
    const MaterialElectronResponseView& table,const ElectronCtGeometry& geometry,double threshold) {
    const auto& c=packet.cursor;
    if(!(threshold>0) || !std::isfinite(threshold) || !c.valid || c.pdg!=11 || c.fraction!=0 ||
       c.source>=table.state_source_count || !table.state_sources)return false;
    const auto material=electron_ct_point_material(geometry,c.position,c.direction);
    if(!material.valid || material.section!=c.section ||
       material.density!=static_cast<float>(c.physical_density_g_cm3>0?c.physical_density_g_cm3:c.density_g_cm3))return false;
    double clearance=threshold;
    for(unsigned k=0;k<3;++k) {
        const double u=(c.position[k]-geometry.origin[k])/geometry.spacing[k];
        const double f=u-std::floor(u);
        clearance=std::min(clearance,std::min(f,1-f)*geometry.spacing[k]);
    }
    if(!(clearance>0))return false;
    const auto& source=table.state_sources[c.source];
    if(source.short_range_lengths) {
        if(c.row>=source.bytes/kElectronStateV4RecordBytes)return false;
        const double length=source.short_range_lengths[c.row];
        return length>=0 && length<clearance;
    }
    const auto view=table.state_sources[c.source].continuation();
    auto row=c.row;double length=0;
    for(unsigned i=0;i<64;++i) {
        const auto step=prepare_electron_replay_step(view,row);
        if(step.status!=ElectronReplayStatus::ready || step.state.pdg!=11 ||
           step.children_begin!=step.children_end || step.state.post_material!=step.state.pre_material ||
           step.state.post_density_g_cm3!=step.state.pre_density_g_cm3)return false;
        double chord2=0;
        for(unsigned k=0;k<3;++k) {
            const double d=step.state.post_position[k]-step.state.pre_position[k];chord2+=d*d;
        }
        length+=std::max(step.state.length_mm,std::sqrt(chord2));
        if(!std::isfinite(length) || length>=clearance)return false;
        if(step.next==kElectronContinuationEnd)
            return step.state.post_energy_MeV==0 && step.state.track_status==2;
        row=step.next;
    }
    return false;
}
} // namespace carbon
