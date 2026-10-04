#pragma once
#include "carbon/electron_continuation_state.hpp"
#include "carbon/electron_compact_state.hpp"
#include <limits>

namespace carbon {
inline constexpr auto kElectronContinuationEnd=std::numeric_limits<std::uint64_t>::max();
struct ElectronContinuationView {
    const unsigned char* payload{};
    std::size_t bytes{};
    const std::uint64_t* next_rows{};
    const std::uint64_t* child_offsets{}; // rows+1; immutable compiler-verified CSR
    const std::uint64_t* child_rows{};
    std::uint64_t child_count{};
    // Immutable host-decoded records replace the wire payload, never both.
    const ElectronContinuationStep* decoded{};
    // Built by full host replay validation on this immutable source. Negative means use original checks.
    const double* certified_child_energy{};
    const ElectronCompactStateView* compact{};
    bool graph_valid() const {
        return compact?(compact->next32 && compact->child_offsets32 && (!child_count || compact->children32)):
                       (next_rows && child_offsets && (!child_count || child_rows));
    }
    std::uint64_t next_at(std::uint64_t row) const {
        if(!compact)return next_rows[row];
        auto v=compact->next32[row];return v==UINT32_MAX?kElectronContinuationEnd:v;
    }
    std::uint64_t child_offset(std::uint64_t row) const {return compact?compact->child_offsets32[row]:child_offsets[row];}
    std::uint64_t child_at(std::uint64_t i) const {return compact?compact->children32[i]:child_rows[i];}
    ElectronContinuationStep read(std::uint64_t row) const {
        if(bytes%kElectronStateV4RecordBytes || row>=bytes/kElectronStateV4RecordBytes)return {};
        if(compact)return compact->read(row);
        return decoded?decoded[row]:read_electron_continuation_step(payload,bytes,row);
    }
};
enum class ElectronReplayStatus { ready, invalid_view, invalid_state, invalid_genealogy,
                                  invalid_continuation, energy_mismatch };
struct ElectronReplayStep {
    ElectronReplayStatus status{ElectronReplayStatus::invalid_view};
    ElectronContinuationStep state{};
    std::uint64_t next{kElectronContinuationEnd},children_begin{},children_end{};
    double child_energy_MeV{},energy_residual_MeV{};
};
// Prepare before any scoring/enqueue operation. A rejected source step must not
// leave a partial deposit or a partially spawned family. No energy renormalizing,
// residual deposition, material alias, or choice of continuation sampling law.
inline ElectronReplayStep prepare_electron_replay_step(
    const ElectronContinuationView& view,std::uint64_t row) {
    ElectronReplayStep result;
    if((!view.payload && !view.decoded && !view.compact) || !view.graph_valid() || view.bytes%kElectronStateV4RecordBytes ||
       row>=view.bytes/kElectronStateV4RecordBytes)return result;
    result.state=view.read(row);
    const auto& state=result.state;
    result.status=ElectronReplayStatus::invalid_state;
    if(!state.valid || (state.pdg!=11 && state.pdg!=22))return result;
    if(view.certified_child_energy) {
        const double energy=view.certified_child_energy[row];
        if(energy>=0 && std::isfinite(energy)) {
            result.next=view.next_at(row);
            result.children_begin=view.child_offset(row);result.children_end=view.child_offset(row+1);
            result.child_energy_MeV=energy;
            result.energy_residual_MeV=state.pre_energy_MeV-state.post_energy_MeV-state.deposited_MeV-energy;
            result.status=ElectronReplayStatus::energy_mismatch;
            if(std::abs(result.energy_residual_MeV)>1e-8+1e-10*state.pre_energy_MeV)return result;
            result.status=ElectronReplayStatus::ready;return result;
        }
    }
    result.status=ElectronReplayStatus::invalid_genealogy;
    result.children_begin=view.child_offset(row);result.children_end=view.child_offset(row+1);
    if(result.children_begin>result.children_end || result.children_end>view.child_count)return result;
    for(auto i=result.children_begin;i<result.children_end;++i) {
        const auto child=view.read(view.child_at(i));
        if(!generated_by_electron_step(state,child) || (child.pdg!=11 && child.pdg!=22))return result;
        if(std::abs(child.pre_energy_MeV-child.birth_energy_MeV)>1e-8)return result;
        // The continuation model applies the discrete branch at the endpoint.
        // Validate this against raw data instead of silently moving a birth.
        for(unsigned a=0;a<3;++a)
            if(std::abs(child.pre_position[a]-state.post_position[a])>1e-6)return result;
        result.child_energy_MeV+=child.birth_energy_MeV;
    }
    result.next=view.next_at(row);
    result.status=ElectronReplayStatus::invalid_continuation;
    if(result.next!=kElectronContinuationEnd) {
        const auto next=view.read(result.next);
        if(!same_electron_continuation_track(state,next) ||
           std::abs(next.pre_energy_MeV-state.post_energy_MeV)>1e-8)return result;
        for(unsigned a=0;a<3;++a)
            if(std::abs(next.pre_position[a]-state.post_position[a])>1e-6 ||
               (next.pre_energy_MeV>0 && std::abs(next.pre_direction[a]-state.post_direction[a])>1e-6))return result;
    }
    result.energy_residual_MeV=state.pre_energy_MeV-state.post_energy_MeV-
        state.deposited_MeV-result.child_energy_MeV;
    result.status=ElectronReplayStatus::energy_mismatch;
    if(!std::isfinite(result.child_energy_MeV) ||
       std::abs(result.energy_residual_MeV)>1e-8+1e-10*state.pre_energy_MeV)return result;
    result.status=ElectronReplayStatus::ready;
    return result;
}
} // namespace carbon
