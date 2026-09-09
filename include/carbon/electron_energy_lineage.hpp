#pragma once
#include "carbon/electron_continuation_transport.hpp"

namespace carbon {
// Energy-weighted lineage choice over one immutable electron advance.
//
// Partition law (tight tolerance 1e-8 + 1e-10 * E_in, same as replay/advance):
//   E_in = deposited_MeV + next_parent_KE + sum(exact child birth KE).
// For clipped advances (material_boundary / escaped) the child sum must be
// exactly zero and the transaction must expose no children. The exact child
// sum is re-derived from correlated child cursors and checked against the
// reported child_energy_MeV; any mismatch, malformed CSR range, non-finite/
// nonnegative energy, or broken closure is a hard invalid -- never
// renormalized, never dumped as residual deposit.
//
// Source-row genealogy is re-validated BEFORE selection, never trusted from
// the supplied CSR window alone: for completed steps (step_complete /
// source_exhausted / stopped) the supplied [children_begin, children_end) must
// exactly equal the raw `prepare_electron_replay_step(incoming.row)` range.
// An omitted child family with forged matching energies is therefore rejected.
// A malformed empty window (out-of-bounds begin==end, or empty window where
// the raw row owns children) is rejected even when the random draw falls in
// the deposit band. Clipped transactions must expose an empty in-bounds
// window. Geometry is NOT re-run here; only the immutable advance plus the raw
// genealogy range are checked.
//
// Exactly ONE outcome is selected with weight proportional to its energy
// (threshold = u_choice * E_in). The dose packet statistical weight W is
// unchanged: it is NOT multiplied by the selection probability, and the full
// packet is NOT split to all children. Deposit exposes only the physical
// energy on the clipped electron segment for diagnostic placement; it is never
// the dose packet score. Photon deposit is at the interaction endpoint only.
// Children inherit the exact correlated frame. The historical photon_pending
// status identifies a selected PDG22 child: the packet driver advances it as
// a photon, never relabels it or dumps its energy locally.
//
// A stopped advance carries a zero-energy parent cursor with no selection
// probability: only deposit / electron_child / photon_pending bands are
// selectable. `ElectronLineageStatus::stopped` is therefore never returned by
// `choose_electron_energy_lineage()`; it is retained only so callers can name
// the advance-level termination distinctly from a selectable lineage outcome.
enum class ElectronLineageStatus {
    invalid,
    deposited,
    parent_continuation,
    material_boundary,
    escaped,
    source_exhausted,
    stopped, // never returned by choose (zero-energy parent has no weight)
    electron_child,
    photon_pending
};
struct ElectronEnergyLineage {
    ElectronLineageStatus status{ElectronLineageStatus::invalid};
    std::array<double,3> deposit_position{};
    // Physical energy on the selected deposit segment for placement only.
    // NEVER score this as the dose packet weight; the packet weight W stays
    // with the caller and is unchanged by selection.
    double physical_selected_deposit_MeV{};
    ElectronContinuationCursor cursor{};
    std::uint64_t child_index{kElectronContinuationEnd};
};
inline ElectronEnergyLineage choose_electron_energy_lineage(
    const ElectronContinuationCursor& incoming,const ElectronContinuationAdvance& advance,
    const MaterialElectronResponseView& table,double u_choice,double u_along) {
    ElectronEnergyLineage out;
    if(!incoming.valid || (incoming.pdg!=11 && incoming.pdg!=22))return out;
    if(!table.state_sources || incoming.source>=table.state_source_count)return out;
    if(table.section!=incoming.section || table.density_g_cm3!=incoming.density_g_cm3)return out;
    if(!incoming.raw_to_world.valid())return out;
    if(!std::isfinite(u_choice) || u_choice<0 || u_choice>=1)return out;
    if(!std::isfinite(u_along) || u_along<0 || u_along>=1)return out;
    if(advance.status==ElectronAdvanceStatus::invalid)return out;
    if(!std::isfinite(incoming.energy_MeV) || incoming.energy_MeV<=0)return out;
    if(!std::isfinite(advance.deposited_MeV) || advance.deposited_MeV<0)return out;
    if(!std::isfinite(advance.child_energy_MeV) || advance.child_energy_MeV<0)return out;
    if(!std::isfinite(advance.next.energy_MeV) || advance.next.energy_MeV<0)return out;
    for(unsigned k=0;k<3;++k)
        if(!std::isfinite(advance.start[k]) || !std::isfinite(advance.end[k]))return out;
    for(unsigned k=0;k<3;++k)if(std::abs(advance.start[k]-incoming.position[k])>1e-6)return out;
    const double total=incoming.energy_MeV;
    const double tolerance=1e-8+1e-10*total;
    const bool is_clipped=advance.status==ElectronAdvanceStatus::material_boundary ||
        advance.status==ElectronAdvanceStatus::escaped;
    const bool is_terminal=advance.status==ElectronAdvanceStatus::step_complete ||
        advance.status==ElectronAdvanceStatus::source_exhausted ||
        advance.status==ElectronAdvanceStatus::stopped;
    if(!is_clipped && !is_terminal)return out;
    // Validate the source parent row itself before any branch selection, so an
    // invalid parent is rejected even when the draw falls in the deposit band.
    // Do not re-run geometry; only check row bounds plus raw genealogy range.
    const auto& source_state=table.state_sources[incoming.source];
    const std::size_t row_count=source_state.bytes/kElectronStateV4RecordBytes;
    if(source_state.bytes%kElectronStateV4RecordBytes || incoming.row>=row_count ||
       incoming.row==kElectronContinuationEnd)return out;
    const auto raw_view=source_state.continuation();
    const std::uint64_t raw_child_count=raw_view.child_count;
    // Supplied CSR window must always be in-bounds, even when empty.
    if(advance.children_begin>advance.children_end)return out;
    if(advance.children_end>raw_child_count)return out;
    if(advance.children_begin>raw_child_count)return out;
    // Positive-energy next-cursor invariant holds regardless of which branch
    // the random draw later picks (not only when the parent is selected).
    if(advance.next.energy_MeV>0) {
        if(!advance.next.valid)return out;
        if(advance.next.section!=incoming.section ||
           advance.next.density_g_cm3!=incoming.density_g_cm3)return out;
    }
    double exact_child_sum=0;
    if(is_clipped) {
        // Clipped transactions expose no children, period.
        if(advance.children_begin!=advance.children_end || advance.child_energy_MeV!=0)return out;
        if(!(advance.next.energy_MeV>0))return out; // zero KE at a face is stopped, never escape/boundary
        if(std::abs(total-advance.deposited_MeV-advance.next.energy_MeV)>tolerance)return out;
    } else {
        // Completed steps must reproduce the exact raw genealogy range.
        const auto replay=prepare_electron_replay_step(raw_view,incoming.row);
        if(replay.status!=ElectronReplayStatus::ready)return out;
        if(advance.children_begin!=replay.children_begin ||
           advance.children_end!=replay.children_end)return out;
        if(std::abs(advance.child_energy_MeV-replay.child_energy_MeV)>tolerance)return out;
        for(auto c=advance.children_begin;c<advance.children_end;++c) {
            const auto child=electron_continuation_child(incoming,advance,table,c);
            if(!child.valid || (child.pdg!=11 && child.pdg!=22))return out;
            if(!std::isfinite(child.energy_MeV) || child.energy_MeV<0)return out;
            exact_child_sum+=child.energy_MeV;
            if(!std::isfinite(exact_child_sum))return out;
        }
        if(std::abs(exact_child_sum-advance.child_energy_MeV)>tolerance)return out;
        if(std::abs(total-advance.deposited_MeV-advance.next.energy_MeV-exact_child_sum)>tolerance)return out;
        if(advance.status==ElectronAdvanceStatus::stopped) {
            if(advance.next.energy_MeV!=0)return out;
        } else if(!(advance.next.energy_MeV>0)) {
            return out;
        }
    }
    const double threshold=u_choice*total;
    if(!std::isfinite(threshold) || threshold<0 || threshold>total)return out;
    double edge=advance.deposited_MeV;
    if(threshold<edge) {
        if(!(advance.deposited_MeV>0))return out; // zero deposit carries no weight
        out.status=ElectronLineageStatus::deposited;
        // Physical placement energy only; never the dose packet score W.
        out.physical_selected_deposit_MeV=advance.deposited_MeV;
        for(unsigned k=0;k<3;++k)out.deposit_position[k]=incoming.pdg==22?advance.end[k]:
            advance.start[k]+u_along*(advance.end[k]-advance.start[k]);
        for(double v:out.deposit_position)if(!std::isfinite(v))return {};
        return out;
    }
    edge+=advance.next.energy_MeV;
    // A stopped advance has next.energy==0, so this band has zero width: the
    // zero-energy parent is never selectable; only deposit/children compete.
    if(advance.next.energy_MeV>0 && threshold<edge) {
        if(!advance.next.valid)return out;
        switch(advance.status) {
            case ElectronAdvanceStatus::step_complete:
                out.status=ElectronLineageStatus::parent_continuation;break;
            case ElectronAdvanceStatus::material_boundary:
                out.status=ElectronLineageStatus::material_boundary;break;
            case ElectronAdvanceStatus::escaped:
                out.status=ElectronLineageStatus::escaped;break;
            case ElectronAdvanceStatus::source_exhausted:
                out.status=ElectronLineageStatus::source_exhausted;break;
            case ElectronAdvanceStatus::stopped:
            default:return out;
        }
        out.cursor=advance.next;
        return out;
    }
    if(advance.children_begin==advance.children_end)return out;
    double base=advance.deposited_MeV+advance.next.energy_MeV;
    for(auto c=advance.children_begin;c<advance.children_end;++c) {
        const auto child=electron_continuation_child(incoming,advance,table,c);
        if(!child.valid)return out;
        base+=child.energy_MeV;
        if(threshold<base) {
            if(!(child.energy_MeV>0))return out; // zero-KE children carry no weight
            out.child_index=c;
            out.cursor=child;
            if(child.pdg==22)out.status=ElectronLineageStatus::photon_pending;
            else if(child.pdg==11)out.status=ElectronLineageStatus::electron_child;
            else return out;
            return out;
        }
    }
    return out;
}
} // namespace carbon
