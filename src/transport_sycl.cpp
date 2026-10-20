#include "immutable_input_cache.hpp"
#include "input_dependencies.hpp"
#include "charged_species.hpp"
#include "secondary_schedule.hpp"
#include "scattering_policy.hpp"
#include "host_prefetch.hpp"
#include "runtime_timing.hpp"
#include "unified_em_view.hpp"
#include "urban_ct_loss.hpp"
#include <fstream>
#include <charconv>
#include <sstream>
#include "cross_section.hpp"
#include "hadronic_rate_cache.hpp"
#include "ct_grid.hpp"
#include "device.hpp"
#include "beamline_fluctuation.hpp"
#include "inelastic.hpp"
#include "multiple_scattering.hpp"
#include "particle.hpp"
#include "rng.hpp"
#include "stopping_power.hpp"
#include "tps_source.hpp"
#include "nuclear_elastic_scattering.hpp"
#include "all_ion_elastic.hpp"
#include "detail/device_memory_tracker.hpp"
#include "transport.hpp"
#include "schneider_rate_table.hpp"
#include "schneider_target_sampler.hpp"
#include "secondary_rate_table.hpp"
#include "inelastic_package_v3.hpp"
#include "cinel03_event_rotation.hpp"
#include "schneider_ct_device_context.hpp"
#include "minibeam_collimator.hpp"

#include <sycl/sycl.hpp>
#include "pipelines/proton_broadbeam.hpp"
#include "pipelines/proton_minibeam.hpp"
#include "pipelines/carbon_broadbeam.hpp"
#include "pipelines/carbon_minibeam.hpp"
#include "pipelines/generic.hpp"
#include "pipelines/bucket_order.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <map>
#include <typeindex>
#include <type_traits>
#include <tuple>
#include <vector>

namespace carbon {
namespace primary_stage = pipelines::primary_stage;

template<class Pipeline, bool UniformWaterPath, int Stage, bool BroadBeam>
class PrimaryTransportKernel;
// A single transfer at the beamline/medium boundary. All physics-specific
// step states remain private to their own stage; RNG identity is preserved.
struct PrimaryMediumEntryState {
    std::uint64_t rng_history, spot_seed, global_history;
    float spot_initial_energy_MeV, energy_MeV, sampled_initial_energy_MeVu;
    float position_x_mm, position_y_mm, position_z_mm;
    float direction_x, direction_y, direction_z;
};
static_assert(sizeof(PrimaryMediumEntryState) == 64);

template<class Pipeline, bool ProductionPath = false, bool BroadBeam = false>
class SecondaryTransportKernel;

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"
#include "detail/sycl_ion_ledger_device.inc"

namespace {

#include "detail/sycl_copper_primary_task.inc"

// Slot values are a shared host/device wire layout, not physical process IDs.
namespace primary_terminal {
enum : std::size_t { inelastic = 0, escaped_ct = 1, stopped = 2, other = 3, count = 4 };
}
namespace copper_failure_field {
enum : std::size_t {
    reason = 1, atomic_number = 2, mass_number = 3, energy = 4,
    step = 5, path = 6, optical_depth = 7
};
}

constexpr std::size_t minibeam_event_counter_count = 170;
constexpr std::size_t copper_fragment_post_loss_null_slot = 169;
constexpr std::size_t copper_ion_first_failure_slot = 161; // claim + reason/ZA/energy/step/path/tau
constexpr std::size_t copper_fragment_urban_slot = 160;
constexpr std::size_t copper_fragment_elastic_slot = 158;
constexpr std::size_t copper_ion_transport_failure_slot = 159;
constexpr std::size_t copper_proton_loss_failure_slot = 153;
constexpr std::size_t copper_proton_primary_loss_slot = 154;
constexpr std::size_t copper_proton_fragment_loss_slot = 155;
constexpr std::size_t minibeam_fragment_cascade_interactions_slot = 48;
constexpr std::size_t minibeam_fragment_cascade_hits_slot = 49;
constexpr std::size_t minibeam_fragment_cascade_miss_slot = 50;
constexpr std::size_t minibeam_fragment_cascade_charged_slot = 56;
constexpr std::size_t minibeam_fragment_cascade_neutral_slot = 57;
constexpr std::size_t minibeam_fragment_cascade_unsupported_slot = 58;
constexpr std::size_t minibeam_fragment_cascade_overflow_slot = 59;
constexpr std::size_t minibeam_fragment_cascade_local_keV_slot = 60;
constexpr std::size_t minibeam_fragment_cascade_untracked_keV_slot = 61;
constexpr std::size_t minibeam_fragment_ignored_tau_micro_slot = 62;
constexpr std::size_t minibeam_fragment_miss_species_slot = 71;
constexpr std::size_t minibeam_fragment_miss_energy_slot = 80;
constexpr std::size_t minibeam_fragment_actual_input_keV_slot = 96;
constexpr std::size_t minibeam_fragment_selected_input_keV_slot = 97;
constexpr std::size_t minibeam_fragment_replay_output_keV_slot = 98;
constexpr std::size_t minibeam_fragment_selection_mismatch_keV_slot = 99;
constexpr std::size_t minibeam_fragment_closure_mismatch_keV_slot = 100;
constexpr std::size_t minibeam_fragment_mass_energy_mismatch_keV_slot = 101;
constexpr std::size_t minibeam_fragment_baryon_mismatch_slot = 102;
constexpr std::size_t minibeam_fragment_terminal_track_slot = 103;
constexpr std::size_t minibeam_fragment_ignored_probability_micro_slot = 112;
constexpr std::size_t minibeam_fragment_generation_interaction_slot = 121;
constexpr std::size_t minibeam_fragment_generation_hit_slot = 124;
constexpr std::size_t minibeam_fragment_generation_miss_slot = 127;
// Hard failure: invalid Urban proposal, zero-progress segment, or
// subdivision-cap trip. Any nonzero count fails the run (throw on
// host): a partial space path with full-macro-step energy loss must never
// pass as a successful dose. Structured record via the slot + throw message.
constexpr std::size_t minibeam_water_urban_fatal_slot = 137;
// Diagnostic: finalize-class numeric-guard trips (t<g / t>t_msc clamp
// repairs in finalize_true). These indicate real conversion inconsistency
// and must be zero in smoke; the trips are counted but non-fatal (the
// repair itself is rounding-only and safe). The limiter's sub-ulp chord
// repair (~7% of production segments by FP32 granularity) is normal
// reference-class protection with a proven <= 1 ulp bound and is
// intentionally not counted here.
constexpr std::size_t minibeam_urban_numeric_guard_slot = 138;
// First-failure record: 139 = claim flag (0/1), 140 = history id,
// 141 = (outer_step<<32)|segment_index, 142 = reason code. Written once via
// CAS by the first failing work-item; the host reports it and fails the run.
constexpr std::size_t minibeam_urban_firstfail_claim_slot = 139;
constexpr std::size_t minibeam_urban_firstfail_history_slot = 140;
constexpr std::size_t minibeam_urban_firstfail_stepseg_slot = 141;
constexpr std::size_t minibeam_urban_firstfail_reason_slot = 142;
// First numeric-guard evidence: 144 = claim flag,
// 145 = site tag (1 Cu, 2 water primary, 3 secondary), 146..151 = float
// bits of (g_final, g_prop, t_msc, lambda0, range, par1), 152 = float bits
// of par3. First trip only; lets the host reproduce the exact inputs.
constexpr std::size_t minibeam_urban_guardev_claim_slot = 144;
constexpr std::size_t minibeam_urban_guardev_base_slot = 145;

struct CopperElasticRecord {
    float energy_MeV_per_u{};
    float macroscopic_rate_per_mm{};
    float transfer_fraction{};
    float target_mass_MeV{};
    std::uint32_t target_a{};
};

struct CopperFragmentElasticBank {
    std::vector<CopperElasticRecord> records;
    std::vector<std::uint32_t> offsets;
};

CopperFragmentElasticBank load_copper_fragment_elastic(
    const std::filesystem::path& path,const std::vector<UrbanMcsSpecies>& species) {
    std::ifstream input(path);
    if(!input)throw std::runtime_error("Cannot open Cu fragment elastic bank: "+path.string());
    std::vector<std::vector<CopperElasticRecord>> groups(species.size());
    std::string line;
    while(std::getline(input,line)) {
        if(line.empty()||line[0]=='#'||std::isalpha(static_cast<unsigned char>(line[0])))continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        int z,a;CopperElasticRecord r{};
        if(!(row>>z>>a>>r.energy_MeV_per_u>>r.macroscopic_rate_per_mm>>r.transfer_fraction>>r.target_a>>r.target_mass_MeV))
            throw std::runtime_error("Malformed Cu fragment elastic row");
        if(!(r.energy_MeV_per_u>0&&r.macroscopic_rate_per_mm>=0&&r.transfer_fraction>=0&&r.transfer_fraction<=1&&r.target_mass_MeV>0) ||
           !std::isfinite(r.energy_MeV_per_u+r.macroscopic_rate_per_mm+r.transfer_fraction+r.target_mass_MeV) ||
           (r.target_a!=63&&r.target_a!=65))throw std::runtime_error("Invalid Cu fragment elastic row");
        auto it=std::find_if(species.begin(),species.end(),[&](const auto& s){return int(s.z)==z&&int(s.a)==a;});
        if(it==species.end())throw std::runtime_error("Cu fragment elastic species not in Urban bank");
        auto& group=groups[it-species.begin()];
        if(!group.empty()&&(r.energy_MeV_per_u<group.back().energy_MeV_per_u ||
           (r.energy_MeV_per_u==group.back().energy_MeV_per_u && r.macroscopic_rate_per_mm!=group.back().macroscopic_rate_per_mm)))
            throw std::runtime_error("Cu fragment elastic nodes unordered or inconsistent");
        group.push_back(r);
    }
    CopperFragmentElasticBank bank;
    for(const auto& group:groups) {
        if(group.size()<2 || group.front().energy_MeV_per_u>=group.back().energy_MeV_per_u)
            throw std::runtime_error("Missing Cu fragment elastic projectile data");
        bank.offsets.push_back(static_cast<std::uint32_t>(bank.records.size()));
        bank.records.insert(bank.records.end(),group.begin(),group.end());
    }
    bank.offsets.push_back(static_cast<std::uint32_t>(bank.records.size()));
    return bank;
}

std::vector<CopperElasticRecord> load_copper_elastic_events(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open Copper elastic bank: " + path.string());
    std::vector<CopperElasticRecord> records;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#' ||
            std::isalpha(static_cast<unsigned char>(line[first]))) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        CopperElasticRecord record{};
        if (!(row >> record.energy_MeV_per_u >> record.macroscopic_rate_per_mm >>
              record.transfer_fraction >> record.target_a >> record.target_mass_MeV)) {
            throw std::runtime_error("Malformed Copper elastic row: " + path.string());
        }
        if (!(record.energy_MeV_per_u > 0.0F) ||
            !(record.macroscopic_rate_per_mm >= 0.0F) ||
            !(record.transfer_fraction >= 0.0F && record.transfer_fraction <= 1.0F) ||
            record.target_a == 0 || !(record.target_mass_MeV > 0.0F)) {
            throw std::runtime_error("Invalid Copper elastic row: " + path.string());
        }
        records.push_back(record);
    }
    if (records.size() < 2) throw std::runtime_error("Copper elastic bank is empty");
    std::stable_sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return left.energy_MeV_per_u < right.energy_MeV_per_u;
    });
    return records;
}

inline float minibeam_linear_table(const float* energies, const float* values,
                                   std::uint32_t count, float energy) noexcept {
    if (!energies || !values || count == 0) return 0.0F;
    if (energy <= energies[0]) return values[0];
    if (energy >= energies[count - 1]) return values[count - 1];
    std::uint32_t low = 0, high = count - 1;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (energies[middle] <= energy) low = middle;
        else high = middle;
    }
    const auto fraction = (energy - energies[low]) / (energies[high] - energies[low]);
    return values[low] + fraction * (values[high] - values[low]);
}

inline float minibeam_copper_ion_inelastic_rate(
    const float* values, const std::uint8_t* species_present,
    std::uint32_t grid_size, float minimum_energy, float inverse_step,
    float kinetic_energy_MeV, int atomic_number, int mass_number) noexcept {
    if (!values || !species_present || grid_size < 2 || atomic_number <= 0 ||
        mass_number <= 0 ||
        atomic_number >= static_cast<int>(IonCrossSectionTables::atomic_number_slots) ||
        mass_number >= static_cast<int>(IonCrossSectionTables::mass_stride)) return 0.0F;
    const auto species = static_cast<std::size_t>(atomic_number) *
        IonCrossSectionTables::mass_stride + static_cast<std::size_t>(mass_number);
    if (species_present[species] == 0) return 0.0F;
    const auto floating_index =
        (kinetic_energy_MeV / static_cast<float>(mass_number) - minimum_energy) *
        inverse_step;
    auto low = static_cast<int>(sycl::floor(floating_index));
    low = sycl::max(0, sycl::min(low, static_cast<int>(grid_size) - 2));
    const auto fraction = sycl::clamp(
        floating_index - static_cast<float>(low), 0.0F, 1.0F);
    const auto base = species * grid_size + static_cast<std::size_t>(low);
    return values[base] + fraction * (values[base + 1] - values[base]);
}

inline std::uint32_t minibeam_elastic_nearest(
    const CopperElasticRecord* records, std::uint32_t count, float energy) noexcept {
    if (!records || count == 0) return 0;
    if (energy <= records[0].energy_MeV_per_u) return 0;
    if (energy >= records[count - 1].energy_MeV_per_u) return count - 1;
    std::uint32_t low = 0, high = count - 1;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (records[middle].energy_MeV_per_u <= energy) low = middle;
        else high = middle;
    }
    return energy - records[low].energy_MeV_per_u <=
                   records[high].energy_MeV_per_u - energy ? low : high;
}

// Same result as minibeam_elastic_nearest(). A host-built bracket per
// energy bin narrows the search only when the bracket is verified to hold the
// unique index with records[low] <= energy < records[low + 1]; the per-step
// rate lookup otherwise walks ~18 dependent levels of a 254k-record bank.
inline std::uint32_t minibeam_elastic_nearest_bracketed(
    const CopperElasticRecord* records, std::uint32_t count, float energy,
    const std::uint32_t* brackets, std::uint32_t bins, float first_energy,
    float inverse_bin_width) noexcept {
    if (!records || count == 0) return 0;
    if (energy <= records[0].energy_MeV_per_u) return 0;
    if (energy >= records[count - 1].energy_MeV_per_u) return count - 1;
    std::uint32_t low = 0, high = count - 1;
    if (brackets && bins > 0) {
        const auto bin = static_cast<std::uint32_t>(sycl::clamp(
            (energy - first_energy) * inverse_bin_width, 0.0F, static_cast<float>(bins - 1)));
        const auto lower = brackets[2 * bin], upper = brackets[2 * bin + 1];
        if (lower < upper && upper < count &&
            records[lower].energy_MeV_per_u <= energy &&
            records[upper].energy_MeV_per_u > energy) {
            low = lower;
            high = upper;
        }
    }
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (records[middle].energy_MeV_per_u <= energy) low = middle;
        else high = middle;
    }
    return energy - records[low].energy_MeV_per_u <=
                   records[high].energy_MeV_per_u - energy ? low : high;
}

inline std::uint32_t minibeam_elastic_local_sample(
    const CopperElasticRecord* records, std::uint32_t count, float energy,
    float uniform) noexcept {
    if (!records || count == 0) return 0;
    // The extraction records contain individual collisions along slowing-down
    // tracks, not one angular quantile table per exact incident energy.  Exact
    // nearest-neighbour lookup consequently turns the sampled transfer into a
    // deterministic function of energy.  Draw from a narrow local energy
    // window instead; over this interval the G4 elastic response varies slowly,
    // while the empirical transfer distribution is retained.
    constexpr float half_window_MeV_per_u = 0.5F;
    const auto lower_energy = energy - half_window_MeV_per_u;
    const auto upper_energy = energy + half_window_MeV_per_u;
    std::uint32_t first = 0, last = count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        if (records[middle].energy_MeV_per_u < lower_energy) first = middle + 1;
        else last = middle;
    }
    const auto begin = first;
    last = count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        if (records[middle].energy_MeV_per_u <= upper_energy) first = middle + 1;
        else last = middle;
    }
    const auto end = first;
    if (begin >= end) return minibeam_elastic_nearest(records, count, energy);
    auto offset = static_cast<std::uint32_t>(uniform * static_cast<float>(end - begin));
    if (offset >= end - begin) offset = end - begin - 1;
    return begin + offset;
}

#ifndef CARBON_EM_AUDIT_SHARDS
#define CARBON_EM_AUDIT_SHARDS 64
#endif
// State persists across launch boundaries; a pause must not finalize dose or
// diagnostics. Only survivor indices move during stable compaction.
struct EmptySecondaryUrbanState {};
struct SecondaryUrbanState {
    float sec_urban_tlimit_mm{};
    bool sec_urban_born{};
    bool sec_urban_first_step{true};
    bool sec_global_urban_boundary{true};
};
template<bool Compact>
struct alignas(16) SecondaryResumeStateT {
    bool sec_terminal_recorded{};
    bool unified_secondary_escaped_ct{};
    ContinuousSpeciesTrackTally continuous_species_tally{};
    float sec_e{};
    float sec_x{};
    float sec_y{};
    float sec_z{};
    float sec_dx{};
    float sec_dy{};
    float sec_dz{};
    float pending_sec_depth_MeV{};
    float pending_sec_voxel_MeV{};
    int pending_sec_bin{};
    int pending_sec_voxel{};
    std::uint64_t unified_secondary_counter{};
    // Preserve optical depth and the rate cache across scheduler slices.
    UnifiedEmState unified_secondary_state{};
    float nuclear_tau_remaining{};
    bool nuclear_tau_active{};
    std::uint32_t nuclear_tau_rng_step{};
    uint32_t sec_steps{};
    std::uint64_t local_sec_rate_queries{};
    std::uint64_t local_sec_steps{};
    [[no_unique_address]] std::conditional_t<Compact, EmptySecondaryUrbanState,
                                            SecondaryUrbanState> urban;
    // Sparse inelastic tallies ([3],[4] of the device audit) are written
    // directly to the global device audit at event time and are not carried
    // across continuation launches.
    std::array<double,4> he4_audit{};

};
static_assert(sizeof(SecondaryResumeStateT<false>) % 16 == 0);
static_assert(sizeof(SecondaryResumeStateT<true>) % 16 == 0);
// Secondary index reordering only: particles are already independent and keep
// their RNG identity. Within one species, sorting by an energy bucket keeps a
// warp reading the same EM node/segment region, which improves L1 locality of
// the exact-index table probes without changing transport arithmetic.
inline constexpr unsigned kSecondaryEnergyBucketCount = 16;
inline constexpr unsigned kSecondaryGroupBucketCount =
    19 * kSecondaryEnergyBucketCount;
inline unsigned secondary_group_bucket(int z, int a, float energy_MeV) {
    const int species = carbon::get_charged_species_idx(z, a);
    const unsigned sp = species >= 0 ? static_cast<unsigned>(species) : 18u;
    unsigned eb = 0u;
    if (energy_MeV > 0.0F) {
        const float level = sycl::log2(energy_MeV * 8.0F);
        eb = static_cast<unsigned>(sycl::clamp(
            level, 0.0F, static_cast<float>(kSecondaryEnergyBucketCount - 1)));
    }
    return sp * kSecondaryEnergyBucketCount + eb;
}
struct UnifiedEmFailureRecord {int reason,section,z,a;float energy,density,step;float d0,d1,d2;};
inline void record_unified_em_failure(unsigned* count,UnifiedEmFailureRecord* records,
    int reason,int section,int z,int a,float energy,float density,float step,
    float d0=0.0F,float d1=0.0F,float d2=0.0F) {
    if(!count)return;
    sycl::atomic_ref<unsigned,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> counter(*count);
    const auto slot=counter.fetch_add(1);if(slot<16)records[slot]={reason,section,z,a,energy,density,step,d0,d1,d2};
}
// Four EM counters are spread across logical threads so a warp's
// same-index increments do not serialize on one global address. Counter i lane
// s lives at global[i*kShards+s]: failures, primary/secondary draws, loss microMeV.
inline constexpr unsigned kUnifiedEmAuditShards = CARBON_EM_AUDIT_SHARDS;
inline constexpr unsigned kUnifiedEmAuditCounters = 4;
#include "detail/sycl_device_math.inc"
#include "detail/sycl_urban_global.inc"

// Urban accepted-step contract checks (device-side).
// Reason codes (0 = accept): 1 invalid proposal (includes the RNG
// outer/segment guard reject), 2 non-positive g, 3 non-finite t/g/delta,
// 4 g > t, 5 delta outside [0, t], 6 FP32 stall (traversed+g == traversed)
// with remainder above granularity dust, 7 geometric overshoot
// (g beyond the segment allowance), 8 bad direction/displacement,
// 9 invalid range-context ordering or internal-energy prediction,
// 10 accepted-path mismatch, 11 invalid physical-voxel navigation.
// A stall with remainder <= 2 ulp of the macro step is NOT a fault: the
// remainder is FP32 granularity dust (position error < 2 ulp); the caller
// completes the macro step exactly and counts a sub-ulp drop instead.
inline int urban_step_field_reason(const CorrelatedScatteringStep& s) noexcept {
    if (!s.proposal_valid) return 1;
    const float t = s.final_true_path_mm;
    const float g = s.final_geom_path_mm;
    const float d = s.stable_delta_mm;
    if (!sycl::isfinite(t) || !sycl::isfinite(g) || !sycl::isfinite(d))
        return 3;
    if (!(g > 0.0F)) return 2;
    if (g > t) return 4;
    if (!(d >= 0.0F) || d > t) return 5;
    if (!s.energy_prediction_valid ||
        !(s.current_range_mm > 0.0F) ||
        !(s.proposed_true_path_mm > 0.0F) ||
        t > s.proposed_true_path_mm +
                4.0F * urban_ulp_above(s.proposed_true_path_mm) ||
        s.proposed_true_path_mm > s.current_range_mm +
                4.0F * urban_ulp_above(s.current_range_mm) ||
        !sycl::isfinite(s.pre_step_energy_mev) ||
        !sycl::isfinite(s.predicted_internal_energy_mev) ||
        s.predicted_internal_energy_mev < 0.0F ||
        s.predicted_internal_energy_mev > s.pre_step_energy_mev +
                4.0F * urban_ulp_above(s.pre_step_energy_mev)) {
        return 9;
    }
    const float dx = s.direction.x, dy = s.direction.y, dz = s.direction.z;
    if (!sycl::isfinite(dx) || !sycl::isfinite(dy) || !sycl::isfinite(dz))
        return 8;
    const float n2 = dx * dx + dy * dy + dz * dz;
    if (!sycl::isfinite(n2) || !(n2 > 0.25F) || !(n2 < 4.0F)) return 8;
    const float ex = s.displacement_mm.x, ey = s.displacement_mm.y,
                ez = s.displacement_mm.z;
    if (!sycl::isfinite(ex) || !sycl::isfinite(ey) || !sycl::isfinite(ez))
        return 8;
    return 0;
}

// Returns 0 (accept: caller advances traversed by g), 6 (FP32 stall with
// significant remainder: fault), 7 (overshoot: fault), or sets
// subulp_complete (caller completes the macro step exactly + counts).

namespace {
inline void urban_record_first_failure(std::uint64_t* counts,
                                       const std::uint64_t history_id,
                                       const std::uint64_t outer_step,
                                       const std::uint32_t segment_index,
                                       const int reason) noexcept {
    if (counts == nullptr) return;
    // Ticket dispenser (fetch_add is supported on the PTX backend;
    // 64-bit compare_exchange is not). Ticket 0 wins and writes the record;
    // the kernel completes before the host reads, so the plain writes are
    // visible. Counts array is zero-filled on the host before launch.
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        claim(counts[minibeam_urban_firstfail_claim_slot]);
    if (claim.fetch_add(1ULL) == 0ULL) {
        counts[minibeam_urban_firstfail_history_slot] = history_id;
        counts[minibeam_urban_firstfail_stepseg_slot] =
            (outer_step << 32U) | segment_index;
        counts[minibeam_urban_firstfail_reason_slot] =
            static_cast<std::uint64_t>(reason);
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        fatal(counts[minibeam_water_urban_fatal_slot]);
    fatal.fetch_add(1U);
}
}  // namespace

namespace {
inline void urban_record_guard_evidence(std::uint64_t* counts,
                                        const std::uint64_t site_tag,
                                        const float epre_mev,
                                        const float t_msc_mm,
                                        const float g_final_mm) noexcept {
    if (counts == nullptr) return;
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        claim(counts[minibeam_urban_guardev_claim_slot]);
    if (claim.fetch_add(1ULL) == 0ULL) {
        counts[minibeam_urban_guardev_base_slot] = site_tag;
        counts[minibeam_urban_guardev_base_slot + 1] =
            sycl::bit_cast<std::uint32_t>(epre_mev);
        counts[minibeam_urban_guardev_base_slot + 2] =
            sycl::bit_cast<std::uint32_t>(t_msc_mm);
        counts[minibeam_urban_guardev_base_slot + 3] =
            sycl::bit_cast<std::uint32_t>(g_final_mm);
    }
}
}  // namespace

using carbon::detail::DeviceMemoryTracker;

// Independent Schneider-CT nuclear diagnostics (named schema, NOT the
// CINEL02 water array). All increments are relaxed device atomics.
inline void schneider_diag_add_device(std::uint64_t* diag, SchneiderDiagSlot slot,
                                      std::uint64_t value) {
    if (diag == nullptr || value == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(diag[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(value);
}

inline void schneider_diag_increment_device(std::uint64_t* diag, SchneiderDiagSlot slot) {
    schneider_diag_add_device(diag, slot, 1U);
}

inline void schneider_float_add_device(float* floats, SchneiderFloatSlot slot, float value) {
    if (floats == nullptr || value == 0.0F) {
        return;
    }
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(floats[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(value);
}

inline void schneider_energy_add_device(std::uint64_t* diag, SchneiderDiagSlot slot,
                                          float value_MeV) {
    if (diag == nullptr || !(value_MeV > 0.0F)) {
        return;
    }
    const auto fixed =
        static_cast<std::uint64_t>(static_cast<double>(value_MeV) * 1.0e6);
    if (fixed == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(diag[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(fixed);
}

inline void schneider_float_max_device(float* floats, SchneiderFloatSlot slot, float value) {
    if (floats == nullptr || !(value > 0.0F)) {
        return;
    }
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(floats[static_cast<std::uint32_t>(slot)]);
    float current = ref.load();
    while (value > current) {
        if (ref.compare_exchange_strong(current, value)) {
            break;
        }
    }
}

inline std::uint8_t schneider_lookup_status_code(Cinel03LookupStatus status) noexcept {
    switch (status) {
    case Cinel03LookupStatus::Hit: return 255;
    case Cinel03LookupStatus::MissingProjectile: return 2;
    case Cinel03LookupStatus::MissingTarget: return 3;
    case Cinel03LookupStatus::BelowEnergyDomain: return 0;
    case Cinel03LookupStatus::AboveEnergyDomain: return 1;
    case Cinel03LookupStatus::EnergyGapTooLarge: return 4;
    case Cinel03LookupStatus::EmptyNode: return 5;
    }
    return 5;
}

// Bounded per-miss record log. Misses are rare (10^2 in 50k runs); the
// buffer holds 2^19 entries and counts drops instead of wrapping.
inline void schneider_log_miss_device(SchneiderMissRecord* log, std::uint32_t* counts,
                                      std::uint32_t cap, bool is_primary,
                                      int pz, int pa, int tz, std::uint8_t section,
                                      std::uint8_t generation,
                                      const Cinel03LookupResult& result,
                                      float query_e_u, float step_dE_MeV,
                                      float total_rate_per_mm, float hazard_step_mm,
                                      float incident_e_MeV, float birth_e_MeV) {
    if (log == nullptr || counts == nullptr) {
        return;
    }
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        count_ref(counts[0]);
    const std::uint32_t slot = count_ref.fetch_add(1U);
    if (slot >= cap) {
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            drop_ref(counts[1]);
        drop_ref.fetch_add(1U);
        return;
    }
    SchneiderMissRecord rec{};
    rec.projectile_z = static_cast<std::int16_t>(pz);
    rec.projectile_a = static_cast<std::int16_t>(pa);
    rec.target_z = static_cast<std::int16_t>(tz);
    rec.status = schneider_lookup_status_code(result.status);
    rec.section_id = section;
    rec.is_primary = is_primary ? 1 : 0;
    rec.generation = generation;
    rec.query_energy_MeV_per_u = query_e_u;
    rec.step_dE_MeV = step_dE_MeV;
    rec.total_rate_per_mm = total_rate_per_mm;
    rec.hazard_step_mm = hazard_step_mm;
    rec.incident_energy_MeV = incident_e_MeV;
    rec.birth_energy_MeV = birth_e_MeV;
    log[slot] = rec;
}

// Bounded per-track log for registry-unknown projectiles (isotope census).
inline void schneider_log_unsupported_track_device(
    SchneiderUnsupportedTrack* log, std::uint32_t* counts, std::uint32_t cap,
    int pz, int pa, std::uint16_t generation, float birth_e_MeV,
    float x_mm, float y_mm, float z_mm) {
    if (log == nullptr || counts == nullptr) {
        return;
    }
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        count_ref(counts[0]);
    const std::uint32_t slot = count_ref.fetch_add(1U);
    if (slot >= cap) {
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            drop_ref(counts[1]);
        drop_ref.fetch_add(1U);
        return;
    }
    SchneiderUnsupportedTrack rec{};
    rec.projectile_z = static_cast<std::int16_t>(pz);
    rec.projectile_a = static_cast<std::int16_t>(pa);
    rec.generation = generation;
    rec.birth_energy_MeV = birth_e_MeV;
    rec.birth_x_mm = x_mm;
    rec.birth_y_mm = y_mm;
    rec.birth_z_mm = z_mm;
    log[slot] = rec;
}

// Record one CINEL03 lookup outcome into the named Schneider counters.
// incident_energy_MeV is the parent kinetic energy at the vertex; domain
// misses itemize it under OutOfDomainEnergy, all other lookup failures
// under LookupFailureEnergy (never silently inside untracked).
inline void schneider_record_lookup_device(std::uint64_t* diag, float* floats,
                                           bool is_primary,
                                           const Cinel03LookupResult& result,
                                           float incident_energy_MeV) {
    if (diag == nullptr) {
        return;
    }
    const float incident = incident_energy_MeV > 0.0F ? incident_energy_MeV : 0.0F;
    switch (result.status) {
    case Cinel03LookupStatus::Hit:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryExactTargetHits
                             : SchneiderDiagSlot::SecondaryExactTargetHits);
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEventsReplayed
                             : SchneiderDiagSlot::SecondaryEventsReplayed);
        schneider_float_add_device(
            floats, is_primary ? SchneiderFloatSlot::PrimaryMismatchSum
                               : SchneiderFloatSlot::SecondaryMismatchSum,
            result.absolute_energy_mismatch_MeV_per_u);
        schneider_float_max_device(
            floats, is_primary ? SchneiderFloatSlot::PrimaryMismatchMax
                               : SchneiderFloatSlot::SecondaryMismatchMax,
            result.absolute_energy_mismatch_MeV_per_u);
        break;
    case Cinel03LookupStatus::MissingProjectile:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryMissingProjectile
                             : SchneiderDiagSlot::SecondaryMissingProjectile);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::MissingTarget:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryMissingTarget
                             : SchneiderDiagSlot::SecondaryMissingTarget);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::BelowEnergyDomain:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryBelowDomain
                             : SchneiderDiagSlot::SecondaryBelowDomain);
        schneider_float_add_device(floats, SchneiderFloatSlot::OutOfDomainEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::AboveEnergyDomain:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryAboveDomain
                             : SchneiderDiagSlot::SecondaryAboveDomain);
        schneider_float_add_device(floats, SchneiderFloatSlot::OutOfDomainEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::EnergyGapTooLarge:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEnergyGapMisses
                             : SchneiderDiagSlot::SecondaryEnergyGapMisses);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::EmptyNode:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEmptyNodes
                             : SchneiderDiagSlot::SecondaryEmptyNodes);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    }
}

float cuda_clock_warmup(sycl::queue& queue, DeviceMemoryTracker& tracker) {
    auto* dummy = tracker.allocate<float>(1024);
    if (dummy == nullptr) {
        return 0.0F;
    }
    queue.memset(dummy, 0, 1024 * sizeof(float)).wait_and_throw();
    const auto start = std::chrono::steady_clock::now();
    auto event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{1024}, sycl::range<1>{128}},
        [=](sycl::nd_item<1> item) {
            const auto lane = item.get_global_linear_id();
            if (lane < 1024) {
                dummy[lane] = static_cast<float>(lane) * 1.001F;
            }
        });
    event.wait_and_throw();
    tracker.free(dummy);
    return static_cast<float>(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace

[[gnu::noinline]] SchneiderCtDeviceContext upload_schneider_ct_device_context(
    sycl::queue& queue,
    carbon::detail::DeviceMemoryTracker& mem_tracker,
    const TransportConfig& config,
    SchneiderHostCache* host_cache = nullptr,
    ResidentPhysicsTables* resident = nullptr) {

    SchneiderCtDeviceContext ctx{};
    ctx.mode = config.material_physics_mode;
    ctx.unified_water = config.unified_water_nuclear_transport;

    if (!ctx.uses_cinel03()) {
        return ctx;
    }
    const auto upload = [&](const char* name, const auto& values) {
        using T = typename std::remove_cvref_t<decltype(values)>::value_type;
        if (resident) return resident->upload(name, values, mem_tracker);
        auto* pointer = mem_tracker.allocate<T>(values.size());
        if (pointer) queue.copy(values.data(), pointer, values.size()).wait_and_throw();
        return pointer;
    };
    PureWaterMaterial water_material{};
    SchneiderMaterialTable water_source_materials{};
    if (ctx.unified_water) {
        water_material = PureWaterMaterial::from_probe(config.unified_water_material_file);
        water_source_materials = SchneiderMaterialTable::from_topas_file(
            config.ct_schneider_file.empty() ? std::filesystem::path("data/materials/schneider/hu_to_material.txt") : config.ct_schneider_file);
        ctx.water_radiation_length_g_cm2 = water_material.radiation_length_g_cm2;
    }
    const auto upload_water_rates = [&](const MaterialNuclearRates& rates) {
        auto view = rates.view();
        auto* partials = mem_tracker.allocate<double>(rates.partials().size());
        auto* domains = mem_tracker.allocate<MaterialRateDomain>(rates.domains().size());
        if (!partials || !domains) throw std::bad_alloc();
        queue.copy(rates.partials().data(),partials,rates.partials().size()).wait_and_throw();
        queue.copy(rates.domains().data(),domains,rates.domains().size()).wait_and_throw();
        view.partials=partials;view.domains=domains;return view;
    };

    std::filesystem::path primary_rate_file = config.ct_schneider_primary_rate_file;
    std::filesystem::path c12_cinel_file = config.ct_schneider_primary_cinel03_file;
    std::filesystem::path sec_rate_file = config.ct_schneider_secondary_rate_file;
    std::filesystem::path sec_cinel_file = config.ct_schneider_secondary_cinel03_file;

    // Channel support and bounds live in the rate binary and CINPKG04 nodes.
    // Compare their FP32 endpoints, which are also used by device lookup.
    const auto check_domain = [](const auto& rate, const auto& channel) {
        if (static_cast<bool>(rate.has_support) != channel.found_target ||
            (rate.has_support &&
             (static_cast<float>(rate.energy_min_mevu) != channel.energy_min_MeV_per_u ||
              static_cast<float>(rate.energy_max_mevu) != channel.energy_max_MeV_per_u)))
            throw std::runtime_error("Nuclear rate/package channel domain mismatch");
    };
    const SchneiderRateTable* rate_table = nullptr;
    std::shared_ptr<SchneiderRateTable> local_rate_table;
    const SecondaryRateTable* sec_rate_table = nullptr;
    std::shared_ptr<SecondaryRateTable> local_sec_rate_table;

    // 1. Primary Target Sampler
    if (std::filesystem::exists(primary_rate_file)) {
        const SchneiderTargetSampler* target_sampler = nullptr;
        std::shared_ptr<SchneiderTargetSampler> local_target_sampler;
        if (host_cache && host_cache->primary_rate && host_cache->primary_sampler) {
            rate_table = host_cache->primary_rate.get();
            target_sampler = host_cache->primary_sampler.get();
            ++host_cache->hits;
        } else {
            local_rate_table = std::make_shared<SchneiderRateTable>(
                SchneiderRateTable::from_binary(primary_rate_file));
            local_target_sampler = std::make_shared<SchneiderTargetSampler>(*local_rate_table);
            rate_table = local_rate_table.get();
            target_sampler = local_target_sampler.get();
            if (host_cache) {
                host_cache->primary_rate = local_rate_table;
                host_cache->primary_sampler = local_target_sampler;
                ++host_cache->misses;
            }
        }
        if (ctx.unified_water) ctx.water_primary_rates = upload_water_rates(
            MaterialNuclearRates::water_primary(*rate_table,water_source_materials,water_material.hydrogen_mass_fraction,
                config.primary_atomic_number,config.primary_mass_number));
        ctx.primary_sampler = target_sampler->device_table();
        // Current masked partials and per-target domains only.
        {
            const auto& partials = target_sampler->partial_rates();
            auto* dev_partial = upload("dev_partial", partials);
            auto* dev_emin = upload("dev_emin", target_sampler->domain_emin());
            auto* dev_emax = upload("dev_emax", target_sampler->domain_emax());
            auto* dev_has = upload("dev_has", target_sampler->domain_has());
            if (dev_partial == nullptr || dev_emin == nullptr || dev_emax == nullptr ||
                dev_has == nullptr) {
                throw std::bad_alloc();
            }
            ctx.primary_sampler.partial_rates = dev_partial;
            ctx.primary_sampler.domain_emin = dev_emin;
            ctx.primary_sampler.domain_emax = dev_emax;
            ctx.primary_sampler.domain_has = dev_has;
        }
    }

    // 2. Primary C12 CINEL03 Package
    if (std::filesystem::exists(c12_cinel_file)) {
        const Cinel03DeviceTables* c12_dev = nullptr;
        std::shared_ptr<InelasticPackageV3Table> local_c12_pkg;
        std::shared_ptr<Cinel03DeviceTables> local_c12_dev;
        if (host_cache && host_cache->primary_package && host_cache->primary_device_tables) {
            c12_dev = host_cache->primary_device_tables.get();
            ++host_cache->hits;
        } else {
            auto loaded = take_cinel03(c12_cinel_file);
            local_c12_pkg = std::move(loaded.package);
            local_c12_dev = std::move(loaded.tables);
            c12_dev = local_c12_dev.get();
            if (host_cache) {
                host_cache->primary_package = local_c12_pkg;
                host_cache->primary_device_tables = local_c12_dev;
                ++host_cache->misses;
            }
        }
        const auto& primary_package = host_cache && host_cache->primary_package
            ? *host_cache->primary_package : *local_c12_pkg;
        for (const auto& event : primary_package.interactions())
            config.require_primary_reference_mass(event.projectile_rest_mass, "Primary CINEL03 package", PrimaryMassConvention::Kinematic);
        for (const auto& node : c12_dev->energy_nodes) {
            if (node.projectile_z != config.primary_atomic_number ||
                node.projectile_a != config.primary_mass_number)
                throw std::runtime_error("Primary CINEL03 binary projectile does not match YAML source Z/A");
        }
        if (!rate_table) throw std::runtime_error("Missing primary nuclear rate table");
        for (std::size_t target = 0; target < kSchneiderNumTargets; ++target)
            check_domain(rate_table->channel_domain(target), primary_package.channel_domain(
                config.primary_atomic_number, config.primary_mass_number, kSchneiderCanonicalZ[target]));
        auto* dev_c12_nodes = upload("dev_c12_nodes", c12_dev->energy_nodes);
        auto* dev_c12_offsets = upload("dev_c12_offsets", c12_dev->event_offsets);
        auto* dev_c12_indices = upload("dev_c12_indices", c12_dev->event_indices);
        auto* dev_c12_ints = upload("dev_c12_ints", c12_dev->interactions);
        auto* dev_c12_prods = upload("dev_c12_prods", c12_dev->products);
        if (dev_c12_nodes == nullptr || dev_c12_offsets == nullptr || dev_c12_indices == nullptr ||
            dev_c12_ints == nullptr || dev_c12_prods == nullptr) {
            throw std::bad_alloc();
        }

        ctx.primary_energy_nodes = dev_c12_nodes;
        ctx.primary_event_offsets = dev_c12_offsets;
        ctx.primary_event_indices = dev_c12_indices;
        ctx.primary_interactions = dev_c12_ints;
        ctx.primary_products = dev_c12_prods;
        ctx.primary_node_count = static_cast<std::uint32_t>(c12_dev->energy_nodes.size());
        ctx.primary_total_events = static_cast<std::uint32_t>(c12_dev->interactions.size());
        ctx.primary_total_products = static_cast<std::uint32_t>(c12_dev->products.size());
    }

    // 3. Secondary Rates (when secondary transport or production mode is active)
    if (std::filesystem::exists(sec_rate_file) &&
        (config.enable_secondary_transport || config.run_mode == RunMode::production || !config.ct_schneider_secondary_rate_file.empty())) {
        if (host_cache && host_cache->secondary_rate) {
            sec_rate_table = host_cache->secondary_rate.get();
            ++host_cache->hits;
        } else {
            local_sec_rate_table = std::make_shared<SecondaryRateTable>(
                SecondaryRateTable::from_binary(sec_rate_file));
            sec_rate_table = local_sec_rate_table.get();
            if (host_cache) {
                host_cache->secondary_rate = local_sec_rate_table;
                ++host_cache->misses;
            }
        }
        if (ctx.unified_water) ctx.water_secondary_rates = upload_water_rates(
            MaterialNuclearRates::water_secondary(*sec_rate_table,water_source_materials,water_material.hydrogen_mass_fraction));
        std::vector<float> sec_partial_rates_float(sec_rate_table->mass_partial_rates().size());
        for (std::size_t i = 0; i < sec_rate_table->mass_partial_rates().size(); ++i) {
            sec_partial_rates_float[i] = static_cast<float>(sec_rate_table->mass_partial_rates()[i]);
        }
        auto* dev_sec_partial = upload("dev_sec_partial", sec_partial_rates_float);
        if (dev_sec_partial == nullptr) {
            throw std::bad_alloc();
        }

        ctx.sec_partial_rates = dev_sec_partial;
        // v3: binary projectile registry + per-(projectile,target)
        // domain for the only supported, masked device path.
        {
            const auto& projs = sec_rate_table->projectiles();
            std::vector<std::int32_t> keys;
            keys.reserve(projs.size() * 2);
            for (const auto& p : projs) {
                keys.push_back(p.z);
                keys.push_back(p.a);
            }
            auto* dev_keys = upload("dev_keys", keys);
            if (dev_keys == nullptr) {
                throw std::bad_alloc();
            }
            ctx.sec_proj_keys = dev_keys;
            const auto& doms = sec_rate_table->channel_domains();
            std::vector<float> demin;
            std::vector<float> demax;
            std::vector<unsigned char> dhas;
            demin.reserve(doms.size());
            demax.reserve(doms.size());
            dhas.reserve(doms.size());
            for (const auto& d : doms) {
                demin.push_back(static_cast<float>(d.energy_min_mevu));
                demax.push_back(static_cast<float>(d.energy_max_mevu));
                dhas.push_back(d.has_support);
            }
            auto* dev_demin = upload("dev_demin", demin);
            auto* dev_demax = upload("dev_demax", demax);
            auto* dev_dhas = upload("dev_dhas", dhas);
            if (dev_demin == nullptr || dev_demax == nullptr || dev_dhas == nullptr) {
                throw std::bad_alloc();
            }
            ctx.sec_domain_emin = dev_demin;
            ctx.sec_domain_emax = dev_demax;
            ctx.sec_domain_has = dev_dhas;
        }
        ctx.sec_num_projectiles = static_cast<std::uint32_t>(sec_rate_table->num_projectiles());
        ctx.sec_num_energies = static_cast<std::uint32_t>(sec_rate_table->num_energies());
        ctx.sec_energy_min_MeV_per_u = static_cast<float>(sec_rate_table->energy_min_mevu());
        ctx.sec_inv_energy_step = static_cast<float>(1.0 / sec_rate_table->energy_step_mevu());
    }

    // 4. Secondary CINEL03 Package
    if (std::filesystem::exists(sec_cinel_file) &&
        (config.enable_secondary_transport || config.run_mode == RunMode::production || !config.ct_schneider_secondary_cinel03_file.empty())) {
        const Cinel03DeviceTables* sec_dev = nullptr;
        std::shared_ptr<InelasticPackageV3Table> local_sec_pkg;
        std::shared_ptr<Cinel03DeviceTables> local_sec_dev;
        if (host_cache && host_cache->secondary_cinel03 &&
            host_cache->secondary_cinel03_device_tables) {
            sec_dev = host_cache->secondary_cinel03_device_tables.get();
            ++host_cache->hits;
        } else {
            auto loaded = take_cinel03(sec_cinel_file);
            local_sec_pkg = std::move(loaded.package);
            local_sec_dev = std::move(loaded.tables);
            sec_dev = local_sec_dev.get();
            if (host_cache) {
                host_cache->secondary_cinel03 = local_sec_pkg;
                host_cache->secondary_cinel03_device_tables = local_sec_dev;
                ++host_cache->misses;
            }
        }
        if (!sec_rate_table) throw std::runtime_error("Missing secondary nuclear rate table");
        const auto& secondary_package = host_cache && host_cache->secondary_cinel03
            ? *host_cache->secondary_cinel03 : *local_sec_pkg;
        for (const auto& node : sec_dev->energy_nodes)
            if (sec_rate_table->projectile_index(node.projectile_z, node.projectile_a) < 0)
                throw std::runtime_error("Secondary package projectile absent from rate binary");
        for (std::size_t projectile = 0; projectile < sec_rate_table->num_projectiles(); ++projectile) {
            const auto& key = sec_rate_table->projectiles()[projectile];
            for (std::size_t target = 0; target < kSecondaryNumTargets; ++target)
                check_domain(sec_rate_table->channel_domain(projectile, target),
                    secondary_package.channel_domain(key.z, key.a, kSchneiderCanonicalZ[target]));
        }
        auto* dev_sec_nodes = upload("dev_sec_nodes", sec_dev->energy_nodes);
        auto* dev_sec_offsets = upload("dev_sec_offsets", sec_dev->event_offsets);
        auto* dev_sec_indices = upload("dev_sec_indices", sec_dev->event_indices);
        auto* dev_sec_ints = upload("dev_sec_ints", sec_dev->interactions);
        auto* dev_sec_prods = upload("dev_sec_prods", sec_dev->products);
        if (dev_sec_nodes == nullptr || dev_sec_offsets == nullptr || dev_sec_indices == nullptr ||
            dev_sec_ints == nullptr || dev_sec_prods == nullptr) {
            throw std::bad_alloc();
        }

        ctx.sec_energy_nodes = dev_sec_nodes;
        ctx.sec_event_offsets = dev_sec_offsets;
        ctx.sec_event_indices = dev_sec_indices;
        ctx.sec_interactions = dev_sec_ints;
        ctx.sec_products = dev_sec_prods;
        ctx.sec_node_count = static_cast<std::uint32_t>(sec_dev->energy_nodes.size());
        ctx.sec_total_events = static_cast<std::uint32_t>(sec_dev->interactions.size());
        ctx.sec_total_products = static_cast<std::uint32_t>(sec_dev->products.size());
    }

    queue.wait_and_throw();

    return ctx;
}

UnifiedEmRequest unified_em_request(const TransportConfig& config) {
    return {config.em_package_file};
}

template<class Pipeline>
[[gnu::noinline]] TransportResult transport_pipeline(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    constexpr bool BroadBeam = Pipeline::broadbeam;
    // These fragments form one function body and share its local scope.
    // setup owns uploads/allocations; primary appends charged children;
    // secondary drains their generations; finalize reads scores and audits.
#include "detail/sycl_pipeline_setup.inc"
#include "detail/sycl_pipeline_primary.inc"
#include "detail/sycl_pipeline_secondary.inc"
#include "detail/sycl_pipeline_finalize.inc"

}

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
    const StoppingPowerTable& stopping_power,const std::string& device_name,SyclTransportContext* context) {
    config.validate();

    const auto run = [&](auto pipeline) {
        using Pipeline = decltype(pipeline);
        std::cout << "[transport-pipeline] name=" << Pipeline::name
                  << " medium=" << (config.enable_ct_grid ? "ct" : "water") << '\n';
        return transport_pipeline<Pipeline>(config, stopping_power, device_name, context);
    };
#if CARBON_ENABLE_PROTON_BROADBEAM
    if (config.transport_pipeline == "proton_broadbeam") return run(pipelines::ProtonBroadbeam{});
#endif
#if CARBON_ENABLE_PROTON_MINIBEAM
    if (config.transport_pipeline == "proton_minibeam") return run(pipelines::ProtonMinibeam{});
#endif
#if CARBON_ENABLE_CARBON_BROADBEAM
    if (config.transport_pipeline == "carbon_broadbeam") return run(pipelines::CarbonBroadbeam{});
#endif
#if CARBON_ENABLE_CARBON_MINIBEAM
    if (config.transport_pipeline == "carbon_minibeam") return run(pipelines::CarbonMinibeam{});
#endif
#if CARBON_ENABLE_GENERIC
    if (config.transport_pipeline == "generic") {
        if (config.transport_profile == "broadbeam_fast") return run(pipelines::Generic<true>{});
        return run(pipelines::Generic<false>{});
    }
#endif
    throw std::invalid_argument("Transport pipeline is not compiled: " + config.transport_pipeline);
}

}  // namespace carbon
