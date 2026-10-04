#pragma once
#include "carbon/electron_continuation_state.hpp"
#include <limits>

namespace carbon {
// Exact field factoring. All physical doubles retain their original bits.
// Track and medium records contain explicit padding, initialized to zero by
// the compiler. No fields are inferred from a neighbouring transport step.
struct ElectronCompactTrack {
    std::array<double,3> parent_direction{};
    double birth_energy_MeV{};
    std::int32_t run{},event{},track{},parent{},pdg{},parent_valid{},parent_step{},reserved{};
};
struct ElectronCompactMedium {
    double pre_density_g_cm3{},post_density_g_cm3{};
    std::int32_t pre_material{},post_material{},post_step_status{},track_status{};
};
struct ElectronCompactRecord {
    std::array<double,3> pre_position{},post_position{},pre_direction{},post_direction{};
    double pre_energy_MeV{},post_energy_MeV{},deposited_MeV{},length_mm{};
    std::uint32_t track{},medium{};
    std::int32_t step{};
    std::uint32_t reserved{};
};
static_assert(sizeof(ElectronCompactTrack)==64 && sizeof(ElectronCompactMedium)==32 &&
              sizeof(ElectronCompactRecord)==144);
// A shared endpoint preserves all seven original double bit patterns.
struct ElectronEndpoint {
    std::array<double,3> position{},direction{};
    double energy_MeV{};
};
struct ElectronEndpointRecord {
    ElectronEndpoint pre{};
    double deposited_MeV{},length_mm{};
    std::uint32_t track{},medium{};
    std::int32_t step{};
    std::uint32_t reserved{},post{},padding{};
};
static_assert(sizeof(ElectronEndpoint)==56 && sizeof(ElectronEndpointRecord)==96);
struct ElectronLegacyCompactStateView {
    const ElectronCompactRecord* records{};
    const ElectronCompactTrack* tracks{};
    const ElectronCompactMedium* media{};
    std::size_t rows{},track_count{},medium_count{};
    const std::uint32_t* next32{};
    const std::uint32_t* child_offsets32{};
    const std::uint32_t* children32{};
    ElectronContinuationStep read(std::uint64_t row) const {
        ElectronContinuationStep s;
        if(!records || !tracks || !media || row>=rows)return s;
        const auto& r=records[row];
        if(r.track>=track_count || r.medium>=medium_count || r.reserved)return s;
        const auto& t=tracks[r.track];const auto& m=media[r.medium];
        if(t.reserved)return s;
        s.valid=true;s.run=t.run;s.event=t.event;s.track=t.track;s.parent=t.parent;s.pdg=t.pdg;s.step=r.step;
        s.parent_valid=t.parent_valid;s.parent_step=t.parent_step;s.parent_direction=t.parent_direction;
        s.birth_energy_MeV=t.birth_energy_MeV;s.pre_position=r.pre_position;s.post_position=r.post_position;
        s.pre_direction=r.pre_direction;s.post_direction=r.post_direction;
        s.pre_energy_MeV=r.pre_energy_MeV;s.post_energy_MeV=r.post_energy_MeV;
        s.deposited_MeV=r.deposited_MeV;s.length_mm=r.length_mm;
        s.pre_density_g_cm3=m.pre_density_g_cm3;s.post_density_g_cm3=m.post_density_g_cm3;
        s.pre_material=m.pre_material;s.post_material=m.post_material;
        s.post_step_status=m.post_step_status;s.track_status=m.track_status;
        return s;
    }
};
struct ElectronCompactStateView {
    const ElectronEndpointRecord* records{};
    const ElectronCompactTrack* tracks{};
    const ElectronCompactMedium* media{};
    std::size_t rows{},track_count{},medium_count{};
    const ElectronEndpoint* endpoints{};
    std::size_t endpoint_count{};
    const std::uint32_t* next32{};
    const std::uint32_t* child_offsets32{};
    const std::uint32_t* children32{};
    ElectronContinuationStep read(std::uint64_t row) const {
        ElectronContinuationStep s;
        if(!records || (endpoint_count && !endpoints) || !tracks || !media || row>=rows)return s;
        const auto& r=records[row];
        if(r.track>=track_count || r.medium>=medium_count || r.reserved || r.padding || r.post>=rows+endpoint_count)return s;
        const auto& t=tracks[r.track];const auto& m=media[r.medium];
        const auto& pre=r.pre;
        const auto& post=r.post<rows?records[r.post].pre:endpoints[r.post-rows];
        if(t.reserved)return s;
        s.valid=true;s.run=t.run;s.event=t.event;s.track=t.track;s.parent=t.parent;s.pdg=t.pdg;s.step=r.step;
        s.parent_valid=t.parent_valid;s.parent_step=t.parent_step;s.parent_direction=t.parent_direction;
        s.birth_energy_MeV=t.birth_energy_MeV;s.pre_position=pre.position;s.post_position=post.position;
        s.pre_direction=pre.direction;s.post_direction=post.direction;
        s.pre_energy_MeV=pre.energy_MeV;s.post_energy_MeV=post.energy_MeV;
        s.deposited_MeV=r.deposited_MeV;s.length_mm=r.length_mm;
        s.pre_density_g_cm3=m.pre_density_g_cm3;s.post_density_g_cm3=m.post_density_g_cm3;
        s.pre_material=m.pre_material;s.post_material=m.post_material;
        s.post_step_status=m.post_step_status;s.track_status=m.track_status;
        return s;
    }
};
}
