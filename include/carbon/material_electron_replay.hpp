#pragma once
#include "carbon/material_electron_response.hpp"
#include "carbon/electron_ct_boundary.hpp"

namespace carbon {
enum class MaterialElectronPathVisit { completed, interrupted, invalid };
// Columns map response-local coordinates to patient coordinates. This is NOT
// the transform from the original TOPAS frame of the recorded momentum.
struct MaterialElectronFrame {
    std::array<double,3> origin{};
    std::array<std::array<double,3>,3> axes{{{1,0,0},{0,1,0},{0,0,1}}};
    bool valid() const {
        for(double x:origin)if(!std::isfinite(x))return false;
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
            double dot=0;
            for(unsigned k=0;k<3;++k)dot+=axes[i][k]*axes[j][k];
            if(!std::isfinite(dot) || std::abs(dot-(i==j?1.0:0.0))>1e-5)return false;
        }
        const double det=axes[0][0]*(axes[1][1]*axes[2][2]-axes[1][2]*axes[2][1])-
            axes[1][0]*(axes[0][1]*axes[2][2]-axes[0][2]*axes[2][1])+
            axes[2][0]*(axes[0][1]*axes[1][2]-axes[0][2]*axes[1][1]);
        return det>0;
    }
    std::array<double,3> world(const std::array<double,3>& p) const {
        auto result=origin;
        for(unsigned k=0;k<3;++k)for(unsigned i=0;i<3;++i)result[k]+=axes[i][k]*p[i];
        return result;
    }
    std::array<double,3> vector(const std::array<double,3>& p) const {
        std::array<double,3> result{};
        for(unsigned k=0;k<3;++k)for(unsigned i=0;i<3;++i)result[k]+=axes[i][k]*p[i];
        return result;
    }
};
inline bool electron_unit_vector(std::array<double,3>& v) {
    double n=0;for(double x:v)n+=x*x;
    if(!(n>1e-24) || !std::isfinite(n))return false;
    n=std::sqrt(n);for(auto& x:v)x/=n;
    return true;
}
// Identical basis convention to water_electron_path_graph.parent_basis.
inline bool electron_direction_basis(std::array<double,3> z,MaterialElectronFrame& frame) {
    if(!electron_unit_vector(z))return false;
    unsigned k=0;for(unsigned i=1;i<3;++i)if(std::abs(z[i])<std::abs(z[k]))k=i;
    std::array<double,3> helper{};helper[k]=1;
    std::array<double,3> x{helper[1]*z[2]-helper[2]*z[1],helper[2]*z[0]-helper[0]*z[2],helper[0]*z[1]-helper[1]*z[0]};
    if(!electron_unit_vector(x))return false;
    const std::array<double,3> y{z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]};
    frame.origin={};frame.axes={x,y,z};return frame.valid();
}
inline MaterialElectronFrame electron_frame_rotation(const MaterialElectronFrame& from,
                                                     const MaterialElectronFrame& to) {
    MaterialElectronFrame result;
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
        result.axes[i][j]=0;
        for(unsigned k=0;k<3;++k)result.axes[i][j]+=to.axes[k][j]*from.axes[k][i];
    }
    return result;
}
struct MaterialElectronBoundaryRequest {
    MaterialElectronPathVisit path_status{MaterialElectronPathVisit::invalid};
    ElectronCtBoundary boundary{};
    ElectronContinuationStep source_step{};
    ElectronReplayStep replay_step{};
    ElectronStateReference source_reference{};
    std::array<double,3> segment_start{},segment_end{};
    MaterialElectronFrame raw_to_world{};
    bool raw_frame_valid{};
    // No inferred boundary KE: a geometric fraction is not an energy fraction.
};
// Feed the complete ancestry in chronological order to the transport adapter.
// The visitor receives (before, after, raw_step, source_reference). Coordinates
// are millimetres in the generating primary's local frame; raw_step positions
// and momenta remain in the original TOPAS frame, NOT the patient frame.
// it owns world rotation and boundary continuation. This layer does NOT
// pretend that density scaling supplies a new material's scattering kernel.
template<class Visitor>
inline MaterialElectronPathVisit visit_material_electron_path(
    const MaterialElectronDraw& draw,const MaterialElectronResponseView& table,Visitor visit) {
    if(draw.status!=MaterialElectronStatus::hit || !draw.response.valid ||
       !table.nodes || !table.node_count || draw.response.head==kWaterPathNone ||
       !table.state_references.valid || !table.state_sources ||
       table.state_references.nodes!=table.node_count ||
       draw.response.sample_index>=table.state_references.samples)
        return MaterialElectronPathVisit::invalid;
    std::size_t depth=0;
    auto node=draw.response.head;
    while(node!=kWaterPathNone) {
        if(node>=table.node_count)return MaterialElectronPathVisit::invalid;
        const auto previous=table.nodes[node].previous;
        if(previous!=kWaterPathNone && previous>=node)return MaterialElectronPathVisit::invalid;
        ++depth;node=previous;
    }
    std::array<double,3> before{};
    // Exact reference traversal; no fixed-size per-track stack or truncation.
    // A future skip-index can accelerate this without changing the callback.
    for(std::size_t remaining=depth;remaining>0;--remaining) {
        node=draw.response.head;
        for(std::size_t back=1;back<remaining;++back)node=table.nodes[node].previous;
        const auto after=table.nodes[node].point;
        for(double v:after)if(!std::isfinite(v))return MaterialElectronPathVisit::invalid;
        if(remaining==depth && after!=std::array<double,3>{})return MaterialElectronPathVisit::invalid;
        if(after!=before) {
            auto reference=table.state_references.at(node,true);
            if(!reference.valid || reference.incoming_edge==std::numeric_limits<std::uint64_t>::max())
                return MaterialElectronPathVisit::invalid;
            // A birth node carries child PRE state, but its incoming segment is
            // the generating PARENT step. Use the edge row, not the node row.
            reference.row=reference.incoming_edge;reference.post=false;
            const auto state=table.state_at(reference);
            if(!state.valid || (state.pdg!=11 && state.pdg!=22))return MaterialElectronPathVisit::invalid;
            if(!visit(before,after,state,reference))return MaterialElectronPathVisit::interrupted;
        }
        before=after;
    }
    for(double v:draw.response.point)if(!std::isfinite(v))return MaterialElectronPathVisit::invalid;
    if(draw.response.point!=before) {
        const auto reference=table.state_references.at(draw.response.sample_index,false);
        const auto state=table.state_at(reference);
        if(!state.valid || (state.pdg!=11 && state.pdg!=22))return MaterialElectronPathVisit::invalid;
        if(!visit(before,draw.response.point,state,reference))return MaterialElectronPathVisit::interrupted;
    }
    return MaterialElectronPathVisit::completed;
}
// Find the FIRST interface on the actual chronological ancestry, not the chord
// from birth to deposition. No scoring occurs here: callers must resolve the
// continuation request before committing a deposit. An interrupted path never
// means that the original homogeneous endpoint is safe to score.
inline MaterialElectronBoundaryRequest locate_material_electron_boundary(
    const MaterialElectronDraw& draw,const MaterialElectronResponseView& table,
    const ElectronCtGeometry& geometry,const MaterialElectronFrame& frame) {
    MaterialElectronBoundaryRequest result;
    if(!frame.valid())return result;
    if(!table.nodes || !table.state_references.valid || draw.response.head==kWaterPathNone)return result;
    auto root=draw.response.head;
    for(;;) {
        if(root>=table.node_count)return result;
        const auto previous=table.nodes[root].previous;
        if(previous==kWaterPathNone)break;
        if(previous>=root)return result;
        root=previous;
    }
    const auto root_state=table.state_at(table.state_references.at(root,true));
    MaterialElectronFrame raw_basis;
    if(!root_state.valid || root_state.pdg!=11 || root_state.step!=1 ||
       !electron_direction_basis(root_state.parent_direction,raw_basis))return result;
    result.raw_to_world=electron_frame_rotation(raw_basis,frame);
    const auto root_rotated=result.raw_to_world.vector(root_state.pre_position);
    for(unsigned k=0;k<3;++k)result.raw_to_world.origin[k]=frame.origin[k]-root_rotated[k];
    result.raw_frame_valid=result.raw_to_world.valid();
    if(!result.raw_frame_valid)return result;
    const auto initial=first_electron_ct_boundary(geometry,frame.origin,frame.origin);
    if(initial.status!=ElectronCtBoundaryStatus::contained)return result;
    result.boundary=initial;
    bool invalid=false;
    result.path_status=visit_material_electron_path(draw,table,
        [&](const auto& before,const auto& after,const auto& state,const auto& reference) {
            const auto a=frame.world(before),b=frame.world(after);
            const auto expected=result.raw_to_world.world(state.pre_position);
            for(unsigned k=0;k<3;++k)if(std::abs(expected[k]-a[k])>1e-6) {invalid=true;return false;}
            const auto boundary=first_electron_ct_boundary(geometry,a,b);
            if(boundary.status==ElectronCtBoundaryStatus::invalid) {invalid=true;return false;}
            result.boundary=boundary;
            if(boundary.status==ElectronCtBoundaryStatus::contained)return true;
            const auto replay=table.replay_at(reference);
            if(replay.status!=ElectronReplayStatus::ready) {invalid=true;return false;}
            result.replay_step=replay;
            result.source_step=state;result.source_reference=reference;
            result.segment_start=a;result.segment_end=b;
            return false;
        });
    if(invalid)result.path_status=MaterialElectronPathVisit::invalid;
    return result;
}
struct MaterialElectronBoundaryContinuation {
    ElectronEnergyCrossingDraw sampled{};
    double source_step_fraction{},source_partial_deposit_MeV{},boundary_energy_MeV{};
};
// Experimental boundary sampling only, not a dose-packet weight law. Uniform
// continuous loss along a recorded chord is an explicit approximation. The
// discrete child energy is retained in the boundary incident KE, NOT deposited
// or spawned early in the old material. Momentum rotation and downstream family
// scoring remain the transport adapter's responsibility.
template<class Uniform>
inline MaterialElectronBoundaryContinuation sample_candidate_material_boundary_continuation(
    const MaterialElectronBoundaryRequest& request,const MaterialElectronResponseView& target,
    Uniform uniform) {
    MaterialElectronBoundaryContinuation result;
    if(request.path_status!=MaterialElectronPathVisit::interrupted ||
       request.boundary.status!=ElectronCtBoundaryStatus::material_change ||
       request.replay_step.status!=ElectronReplayStatus::ready ||
       target.section!=request.boundary.to_section ||
       static_cast<float>(target.density_g_cm3)!=request.boundary.to_density)return result;
    const auto& step=request.replay_step.state;
    if(step.pdg!=11) {result.sampled.status=ElectronCrossingStatus::unsupported_particle;return result;}
    double full2=0,partial2=0;
    for(unsigned k=0;k<3;++k) {
        const double full=step.post_position[k]-step.pre_position[k];
        const double partial=request.boundary.position[k]-request.segment_start[k];
        full2+=full*full;partial2+=partial*partial;
    }
    if(!(full2>0) || !std::isfinite(full2) || !std::isfinite(partial2))return result;
    const auto fraction=std::sqrt(partial2/full2);
    if(fraction>1+1e-8)return result;
    result.source_step_fraction=fraction>1?1:fraction;
    result.source_partial_deposit_MeV=result.source_step_fraction*step.deposited_MeV;
    result.boundary_energy_MeV=step.pre_energy_MeV-result.source_partial_deposit_MeV;
    result.sampled=sample_material_electron_continuation(target,11,result.boundary_energy_MeV,uniform);
    return result;
}
} // namespace carbon
