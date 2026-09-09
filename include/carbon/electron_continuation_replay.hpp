#pragma once
#include "carbon/electron_continuation_state.hpp"
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
    if(!view.payload || !view.next_rows || !view.child_offsets ||
       (view.child_count && !view.child_rows) || view.bytes%kElectronStateV4RecordBytes ||
       row>=view.bytes/kElectronStateV4RecordBytes)return result;
    result.state=read_electron_continuation_step(view.payload,view.bytes,row);
    const auto& state=result.state;
    result.status=ElectronReplayStatus::invalid_state;
    if(!state.valid || (state.pdg!=11 && state.pdg!=22))return result;
    result.status=ElectronReplayStatus::invalid_genealogy;
    result.children_begin=view.child_offsets[row];result.children_end=view.child_offsets[row+1];
    if(result.children_begin>result.children_end || result.children_end>view.child_count)return result;
    for(auto i=result.children_begin;i<result.children_end;++i) {
        const auto child=read_electron_continuation_step(view.payload,view.bytes,view.child_rows[i]);
        if(!generated_by_electron_step(state,child) || (child.pdg!=11 && child.pdg!=22))return result;
        if(std::abs(child.pre_energy_MeV-child.birth_energy_MeV)>1e-8)return result;
        // The continuation model applies the discrete branch at the endpoint.
        // Validate this against raw data instead of silently moving a birth.
        for(unsigned a=0;a<3;++a)
            if(std::abs(child.pre_position[a]-state.post_position[a])>1e-6)return result;
        result.child_energy_MeV+=child.birth_energy_MeV;
    }
    result.next=view.next_rows[row];
    result.status=ElectronReplayStatus::invalid_continuation;
    if(result.next!=kElectronContinuationEnd) {
        const auto next=read_electron_continuation_step(view.payload,view.bytes,result.next);
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
