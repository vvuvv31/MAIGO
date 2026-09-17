#pragma once
#include "carbon/material_electron_replay.hpp"

namespace carbon {
// Candidate direction interpolation, not a Brownian-bridge reconstruction of
// unrecorded substep scattering. Chords and momentum remain separate.
inline bool electron_step_direction(const ElectronContinuationStep& step,double fraction,
                                    std::array<double,3>& direction) {
    if(!std::isfinite(fraction) || fraction<0 || fraction>1)return false;
    for(unsigned k=0;k<3;++k)
        direction[k]=(1-fraction)*step.pre_direction[k]+fraction*step.post_direction[k];
    return electron_unit_vector(direction);
}
struct ElectronContinuationCursor {
    bool valid{};
    std::size_t source{};
    std::uint64_t row{kElectronContinuationEnd};
    double fraction{},energy_MeV{};
    MaterialElectronFrame raw_to_world{};
    std::array<double,3> position{},direction{};
    int section{-1};double density_g_cm3{};
    // Physical voxel density may lie between sampled response-table nodes.
    // Zero preserves the exact-node API for existing callers/tests.
    double physical_density_g_cm3{};
    int pdg{11};
};
inline ElectronContinuationCursor bind_electron_continuation(
    const ElectronEnergyCrossingDraw& sampled,const MaterialElectronResponseView& target,
    const std::array<double,3>& position,std::array<double,3> incoming,double azimuth_uniform) {
    ElectronContinuationCursor cursor;
    if(sampled.status!=ElectronCrossingStatus::ready || sampled.source>=target.state_source_count ||
       !std::isfinite(azimuth_uniform) || azimuth_uniform<0 || azimuth_uniform>=1)return cursor;
    std::array<double,3> target_raw{};
    if(!electron_step_direction(sampled.replay.state,sampled.continuous_loss_fraction,target_raw))return cursor;
    if(!electron_unit_vector(incoming))return cursor;
    MaterialElectronFrame from,to;
    if(!electron_direction_basis(target_raw,from) || !electron_direction_basis(incoming,to))return cursor;
    const auto x=to.axes[0],y=to.axes[1];
    const double angle=6.2831853071795864769*azimuth_uniform,c=std::cos(angle),s=std::sin(angle);
    for(unsigned k=0;k<3;++k) {to.axes[0][k]=c*x[k]+s*y[k];to.axes[1][k]=-s*x[k]+c*y[k];}
    cursor.raw_to_world=electron_frame_rotation(from,to);
    cursor.fraction=sampled.continuous_loss_fraction;
    const auto& step=sampled.replay.state;
    std::array<double,3> entry{};
    for(unsigned k=0;k<3;++k)entry[k]=step.pre_position[k]+cursor.fraction*(step.post_position[k]-step.pre_position[k]);
    const auto rotated=cursor.raw_to_world.vector(entry);
    for(unsigned k=0;k<3;++k)cursor.raw_to_world.origin[k]=position[k]-rotated[k];
    cursor.source=sampled.source;cursor.row=sampled.row;
    cursor.energy_MeV=sampled.energy_MeV;
    cursor.position=position;cursor.direction=incoming;
    cursor.section=target.section;cursor.density_g_cm3=target.density_g_cm3;
    cursor.valid=cursor.raw_to_world.valid();
    return cursor;
}
inline ElectronContinuationCursor bind_electron_boundary_continuation(
    const MaterialElectronBoundaryRequest& request,const MaterialElectronBoundaryContinuation& continued,
    const MaterialElectronResponseView& target,double azimuth_uniform) {
    if(!request.raw_frame_valid || target.section!=request.boundary.to_section ||
       static_cast<float>(target.density_g_cm3)!=request.boundary.to_density)return {};
    std::array<double,3> incoming{};
    if(!electron_step_direction(request.source_step,continued.source_step_fraction,incoming))return {};
    return bind_electron_continuation(continued.sampled,target,request.boundary.position,
                                     request.raw_to_world.vector(incoming),azimuth_uniform);
}
// Birth binding from a carbon-energy birth draw to a physical electron cursor.
// The recorded root parent direction is aligned to the incident carbon world
// direction (with a random azimuth spin); the recorded electron pre direction
// is carried by the SAME rigid rotation, preserving the birth cone. The raw
// root pre position is anchored to the given world position. Cursor energy is
// the physical root kinetic energy; the dose packet statistical weight stays
// with the caller and is never set here. Zero-transfer births never launch.
inline ElectronContinuationCursor bind_electron_birth(
    const ElectronBirthDraw& draw,const MaterialElectronResponseView& table,
    const std::array<double,3>& world_position,std::array<double,3> carbon_direction,
    double azimuth_uniform) {
    ElectronContinuationCursor cursor;
    if(!draw.valid || !std::isfinite(draw.fraction) || !(draw.fraction>0) || draw.fraction>1)return cursor;
    if(!draw.reference.valid || draw.reference.post)return cursor;
    if(!table.state_sources || draw.reference.source>=table.state_source_count)return cursor;
    for(double v:world_position)if(!std::isfinite(v))return cursor;
    if(!std::isfinite(azimuth_uniform) || azimuth_uniform<0 || azimuth_uniform>=1)return cursor;
    if(!electron_unit_vector(carbon_direction))return cursor;
    const auto root=table.state_at(draw.reference);
    if(!root.valid || root.pdg!=11 || root.step!=1 || root.parent_valid!=1)return cursor;
    if(!std::isfinite(root.birth_energy_MeV) || !(root.birth_energy_MeV>0))return cursor;
    if(!std::isfinite(root.pre_energy_MeV) || !(root.pre_energy_MeV>0))return cursor;
    if(std::abs(root.birth_energy_MeV-root.pre_energy_MeV)>1e-8)return cursor;
    MaterialElectronFrame from,to;
    if(!electron_direction_basis(root.parent_direction,from))return cursor;
    if(!electron_direction_basis(carbon_direction,to))return cursor;
    const auto x=to.axes[0],y=to.axes[1];
    const double angle=6.2831853071795864769*azimuth_uniform,c=std::cos(angle),s=std::sin(angle);
    for(unsigned k=0;k<3;++k) {to.axes[0][k]=c*x[k]+s*y[k];to.axes[1][k]=-s*x[k]+c*y[k];}
    cursor.raw_to_world=electron_frame_rotation(from,to);
    const auto rotated=cursor.raw_to_world.vector(root.pre_position);
    for(unsigned k=0;k<3;++k)cursor.raw_to_world.origin[k]=world_position[k]-rotated[k];
    if(!cursor.raw_to_world.valid())return {};
    cursor.source=draw.reference.source;cursor.row=draw.reference.row;
    cursor.fraction=0;cursor.energy_MeV=root.pre_energy_MeV;
    cursor.position=world_position;
    cursor.direction=cursor.raw_to_world.vector(root.pre_direction);
    if(!electron_unit_vector(cursor.direction))return {};
    cursor.section=table.section;cursor.density_g_cm3=table.density_g_cm3;
    cursor.pdg=11;cursor.valid=true;
    return cursor;
}
enum class ElectronAdvanceStatus { invalid, step_complete, material_boundary, escaped,
                                   stopped, source_exhausted };
struct ElectronContinuationAdvance {
    ElectronAdvanceStatus status{ElectronAdvanceStatus::invalid};
    ElectronContinuationCursor next{};
    std::array<double,3> start{},end{};
    double deposited_MeV{},child_energy_MeV{},energy_residual_MeV{};
    std::uint64_t children_begin{},children_end{};
    ElectronCtBoundary boundary{};
};
// Prepare an immutable transaction: caller scores/enqueues ONLY after status is
// valid. Children are exposed only at a completed source step, never before an
// interface. An exhausted finite source retains KE and requires explicit action.
inline ElectronContinuationAdvance advance_electron_continuation(
    const ElectronContinuationCursor& cursor,const MaterialElectronResponseView& table,
    const ElectronCtGeometry& geometry) {
    ElectronContinuationAdvance out;
    if(!cursor.valid || (cursor.pdg!=11 && cursor.pdg!=22) || !table.state_sources || cursor.source>=table.state_source_count ||
       table.section!=cursor.section || table.density_g_cm3!=cursor.density_g_cm3 ||
       !std::isfinite(cursor.energy_MeV) || cursor.energy_MeV<=0 ||
       !std::isfinite(cursor.fraction) || cursor.fraction<0 || cursor.fraction>=1 ||
       !cursor.raw_to_world.valid())return out;
    const auto replay=prepare_electron_replay_step(table.state_sources[cursor.source].continuation(),cursor.row);
    if(replay.status!=ElectronReplayStatus::ready || replay.state.pdg!=cursor.pdg)return out;
    const auto& step=replay.state;
    const bool photon=cursor.pdg==22;
    const double expected=step.pre_energy_MeV-(photon?0:cursor.fraction*step.deposited_MeV);
    if(std::abs(expected-cursor.energy_MeV)>1e-8+1e-10*expected)return out;
    std::array<double,3> raw_start{};
    for(unsigned k=0;k<3;++k)raw_start[k]=step.pre_position[k]+cursor.fraction*(step.post_position[k]-step.pre_position[k]);
    const auto start=cursor.raw_to_world.world(raw_start),end=cursor.raw_to_world.world(step.post_position);
    for(unsigned k=0;k<3;++k)if(std::abs(start[k]-cursor.position[k])>1e-6)return out;
    out.boundary=first_electron_ct_boundary(geometry,cursor.position,end);
    if(out.boundary.status==ElectronCtBoundaryStatus::invalid)return out;
    out.next=cursor;out.start=cursor.position;
    // At an exact face a newly sampled chord may point back into the previous
    // material. Report a zero-length recrossing rather than scoring in the
    // wrong material or inventing an epsilon displacement.
    const double physical_density=cursor.physical_density_g_cm3>0?cursor.physical_density_g_cm3:cursor.density_g_cm3;
    if(!std::isfinite(physical_density) || physical_density<=0)return {};
    if(out.boundary.from_section!=cursor.section ||
       out.boundary.from_density!=static_cast<float>(physical_density)) {
        bool on_face=false;
        for(unsigned k=0;k<3;++k) {
            const auto u=(cursor.position[k]-geometry.origin[k])/geometry.spacing[k];
            on_face=on_face || u==std::floor(u);
        }
        if(!on_face)return {};
        out.boundary.to_section=out.boundary.from_section;out.boundary.to_density=out.boundary.from_density;
        out.boundary.from_section=static_cast<std::uint8_t>(cursor.section);
        out.boundary.from_density=static_cast<float>(physical_density);
        out.boundary.status=ElectronCtBoundaryStatus::material_change;
        out.boundary.fraction=0;out.boundary.position=cursor.position;
        out.end=cursor.position;out.status=ElectronAdvanceStatus::material_boundary;
        return out;
    }
    const bool clipped=out.boundary.status!=ElectronCtBoundaryStatus::contained;
    const double relative=clipped?out.boundary.fraction:1;
    if(!std::isfinite(relative) || relative<0 || relative>1)return {};
    const double consumed=(1-cursor.fraction)*relative;
    // Photon transfer occurs only at the recorded interaction endpoint.
    // A clipped flight has neither continuous deposition nor child creation.
    out.deposited_MeV=photon?(clipped?0:step.deposited_MeV):consumed*step.deposited_MeV;
    out.end=clipped?out.boundary.position:end;
    out.next.position=out.end;out.next.fraction=cursor.fraction+consumed;
    if(clipped) {
        out.next.energy_MeV=cursor.energy_MeV-out.deposited_MeV;
        if(!std::isfinite(out.next.energy_MeV) || out.next.energy_MeV<0)return {};
        out.status=out.boundary.status==ElectronCtBoundaryStatus::escaped ?
            ElectronAdvanceStatus::escaped:ElectronAdvanceStatus::material_boundary;
        if(out.next.energy_MeV==0) {
            out.status=ElectronAdvanceStatus::stopped;
            out.next.row=kElectronContinuationEnd;out.next.fraction=0;
        }
    } else {
        out.child_energy_MeV=replay.child_energy_MeV;
        out.children_begin=replay.children_begin;out.children_end=replay.children_end;
        out.next.energy_MeV=step.post_energy_MeV;
        out.next.row=replay.next;out.next.fraction=0;
        out.status=step.post_energy_MeV==0?ElectronAdvanceStatus::stopped:
            (replay.next==kElectronContinuationEnd?ElectronAdvanceStatus::source_exhausted:ElectronAdvanceStatus::step_complete);
    }
    std::array<double,3> direction{};
    if(out.next.energy_MeV>0) {
        if(photon) {
            direction=clipped?step.pre_direction:step.post_direction;
            if(!electron_unit_vector(direction))return {};
        } else if(!electron_step_direction(step,cursor.fraction+consumed,direction))return {};
        out.next.direction=cursor.raw_to_world.vector(direction);
        if(!electron_unit_vector(out.next.direction))return {};
    }
    out.energy_residual_MeV=cursor.energy_MeV-out.deposited_MeV-out.child_energy_MeV-out.next.energy_MeV;
    if(!std::isfinite(out.energy_residual_MeV) ||
       std::abs(out.energy_residual_MeV)>1e-8+1e-10*cursor.energy_MeV)return {};
    return out;
}
// Preserve sibling correlations: children inherit the SAME raw-to-world frame,
// with their recorded birth position/energy/direction. Photons are explicitly
// identified for a separate transport route, never relabelled as electrons.
inline ElectronContinuationCursor electron_continuation_child(
    const ElectronContinuationCursor& parent,const ElectronContinuationAdvance& advance,
    const MaterialElectronResponseView& table,std::uint64_t child_index) {
    ElectronContinuationCursor child;
    if(advance.status!=ElectronAdvanceStatus::step_complete && advance.status!=ElectronAdvanceStatus::stopped &&
       advance.status!=ElectronAdvanceStatus::source_exhausted)return child;
    if(!parent.valid || table.section!=parent.section || table.density_g_cm3!=parent.density_g_cm3 ||
       !table.state_sources || parent.source>=table.state_source_count ||
       child_index<advance.children_begin || child_index>=advance.children_end)return child;
    const auto source=table.state_sources[parent.source].continuation();
    if(!source.child_rows || child_index>=source.child_count)return child;
    const auto row=source.child_rows[child_index];
    const auto state=read_electron_continuation_step(source.payload,source.bytes,row);
    const auto parent_state=read_electron_continuation_step(source.payload,source.bytes,parent.row);
    if(!generated_by_electron_step(parent_state,state) || (state.pdg!=11 && state.pdg!=22) ||
       std::abs(state.pre_energy_MeV-state.birth_energy_MeV)>1e-8)return child;
    child=parent;child.row=row;child.fraction=0;child.energy_MeV=state.birth_energy_MeV;child.pdg=state.pdg;
    child.position=parent.raw_to_world.world(state.pre_position);
    child.direction=parent.raw_to_world.vector(state.pre_direction);
    for(unsigned k=0;k<3;++k)if(std::abs(child.position[k]-advance.end[k])>1e-6)return {};
    if(child.energy_MeV>0 && !electron_unit_vector(child.direction))return {};
    return child;
}
} // namespace carbon
