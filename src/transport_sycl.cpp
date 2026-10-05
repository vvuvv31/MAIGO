#include "carbon/immutable_input_cache.hpp"
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
#include "carbon/electron_segment_diagnostics.hpp"
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
#include "carbon/electron_range_guard.hpp"
#endif
#include "carbon/input_dependencies.hpp"
#include "carbon/nuclear_collision.hpp"
#include "carbon/fe_moments.hpp"
#include "carbon/charged_species.hpp"
#include "carbon/secondary_schedule.hpp"
#include "carbon/delta_moments_data.hpp"
#include "carbon/host_prefetch.hpp"
#include "carbon/runtime_timing.hpp"
#include "carbon/unified_em_view.hpp"
#ifndef CARBON_SECONDARY_STEP_PROFILE
#define CARBON_SECONDARY_STEP_PROFILE 0
#endif
#ifndef CARBON_SECONDARY_CONTEXT_POINTER
#define CARBON_SECONDARY_CONTEXT_POINTER 0
#endif
#ifndef CARBON_SECONDARY_EXPLICIT_ND_RANGE
#define CARBON_SECONDARY_EXPLICIT_ND_RANGE 0
#endif
#ifndef CARBON_SECONDARY_ND_RANGE_SIZE
#define CARBON_SECONDARY_ND_RANGE_SIZE 64
#endif
#ifndef CARBON_SECONDARY_REUSE_PROJECTILE_INDEX
#define CARBON_SECONDARY_REUSE_PROJECTILE_INDEX 0
#endif
#ifndef CARBON_SECONDARY_PRODUCTION_SPECIALIZE
#define CARBON_SECONDARY_PRODUCTION_SPECIALIZE 0
#endif
#ifndef CARBON_PRIMARY_PRODUCTION_SPECIALIZE
#define CARBON_PRIMARY_PRODUCTION_SPECIALIZE 0
#endif
// Pass the immutable primary-kernel closures (Schneider CT context and unified
// EM device view) through a device-resident context pointer instead of capturing
// them by value. Required on GPU backends whose kernel-argument limit is 2048 B
// (Intel Arc OpenCL/Level Zero); harmless and ABI-compatible on CUDA.
#ifndef CARBON_PRIMARY_CONTEXT_POINTER
#define CARBON_PRIMARY_CONTEXT_POINTER 0
#endif
// When the primary context pointer is enabled, also move the unified EM device
// view out of the argument list. Set to 0 to keep unified_device by value (only
// SchneiderCtDeviceContext moved) for isolation/debugging.
#ifndef CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
#define CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED 1
#endif
#ifndef CARBON_SECONDARY_KNOWN_SPECIES_PROBE
#define CARBON_SECONDARY_KNOWN_SPECIES_PROBE 0
#endif
#ifndef CARBON_SECONDARY_NON_HE4_PROBE
#define CARBON_SECONDARY_NON_HE4_PROBE 0
#endif
#ifndef CARBON_SECONDARY_EXACT_SPECIES_PROBE
#define CARBON_SECONDARY_EXACT_SPECIES_PROBE -1
#endif
#ifndef CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE
#define CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE 0
#endif
#if CARBON_SECONDARY_CONTEXT_POINTER
#define CARBON_SECONDARY_CONTEXT_FIELD(name) secondary_transport_ctx->name
#else
#define CARBON_SECONDARY_CONTEXT_FIELD(name) name
#endif
#include <fstream>
#include <charconv>
#include <sstream>
#include "carbon/cross_section.hpp"
#include "carbon/hadronic_cache_candidate.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/electron_ct_birth_position.hpp"
#include "carbon/device.hpp"
#include "carbon/broadbeam_electron_region.hpp"
#include "carbon/broadbeam_electron_distance.hpp"
#include "carbon/broadbeam_body_mask.hpp"
#include "carbon/electron_charged_packet.hpp"
#include "carbon/electron_birth_task.hpp"
#include "carbon/electron_fp32_device.hpp"
#ifdef CARBON_ELECTRON_VOXEL_CACHE
#include "carbon/electron_voxel_cache.hpp"
#endif
#include "carbon/copper_proton_fluctuation.hpp"
#include "carbon/inelastic.hpp"
#include "carbon/inelastic_package_v2.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/nuclear_elastic_scattering.hpp"
#include "carbon/all_ion_elastic.hpp"
#include "carbon/detail/device_memory_tracker.hpp"
#include "carbon/transport.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/longitudinal_ray.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/secondary_rate_table.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/cinel03_event_rotation.hpp"
#include "carbon/schneider_ct_device_context.hpp"
#if defined(CARBON_ENABLE_MINIBEAM)
#include "carbon/minibeam_collimator.hpp"
#endif

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include <tuple>
#include <type_traits>
#include <vector>

namespace carbon {
// JointResponse: 0 disabled, 1 inline, 2 separate selected-response replay.
template<bool ProductionPath = false, bool UniformWaterPath = false, int Stage = 0, bool BroadBeam = false, bool Gaussian = false, int JointResponse = 0>
class CarbonPrimaryTransportKernel;
template<bool Gaussian> class FeJointReplayKernel;
// A single transfer at the beamline/medium boundary. All physics-specific
// step states remain private to their own stage; RNG identity is preserved.
struct PrimaryMediumEntryState {
    std::uint64_t rng_history, spot_seed, global_history;
    float spot_initial_energy_MeV, energy_MeV, sampled_initial_energy_MeVu;
    float position_x_mm, position_y_mm, position_z_mm;
    float direction_x, direction_y, direction_z;
};
static_assert(sizeof(PrimaryMediumEntryState) == 64);

template<bool ProductionPath = false, int ExactSpecies = -1, bool BroadBeam = false, bool Gaussian = false, int JointResponse = 0>
class CarbonSecondaryTransportKernel;

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_profile.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"
#include "detail/sycl_cinel02_device.inc"

namespace {

#include "detail/sycl_minibeam_copper_diagnostic.inc"

#if defined(CARBON_ENABLE_MINIBEAM)
constexpr std::size_t minibeam_event_counter_count = 158;
constexpr std::size_t air_ion_proposal_slot = 156;
constexpr std::size_t air_ion_step_slot = 157;
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
constexpr std::size_t minibeam_secondary_c12_fe_step_slot = 130;
constexpr std::size_t minibeam_secondary_c12_fe_segment_slot = 131;
constexpr std::size_t minibeam_secondary_c12_raw_loss_micro_slot = 132;
constexpr std::size_t minibeam_secondary_c12_scaled_loss_micro_slot = 133;
constexpr std::size_t minibeam_secondary_c12_scaled_loss_step_slot = 134;
// Diagnostic: water-Urban subdivision iterations beyond any physical need
// (legit worst case ~200 micro-segments for sub-table-floor energies; normal
// use needs <= 4). Trips indicate a non-progress pathology, never physics.
constexpr std::size_t minibeam_water_urban_subdiv_cap_slot = 135;
// Diagnostic: secondary-C12 Urban step entries (proves the research branch
// executes; the cap slot above only fires on pathology).
constexpr std::size_t minibeam_water_secondary_c12_urban_step_slot = 136;
// Hard failure: invalid Urban proposal, zero-progress segment, or
// subdivision-cap trip (fix B5). Any nonzero count fails the run (throw on
// host): a partial space path with full-macro-step energy loss must never
// pass as a successful dose. Structured record via the slot + throw message.
constexpr std::size_t minibeam_water_urban_fatal_slot = 137;
// Diagnostic: C1 finalize-class numeric-guard trips (t<g / t>t_msc clamp
// repairs in finalize_true). These indicate real conversion inconsistency
// and must be zero in smoke; the trips are counted but non-fatal (the
// repair itself is rounding-only and safe). The limiter's sub-ulp chord
// repair (~7% of production segments by FP32 granularity) is normal
// reference-class protection with a proven <= 1 ulp bound and is
// intentionally not counted here.
constexpr std::size_t minibeam_urban_numeric_guard_slot = 138;
// First-failure record (fix C2.3): 139 = claim flag (0/1), 140 = history id,
// 141 = (outer_step<<32)|segment_index, 142 = reason code. Written once via
// CAS by the first failing work-item; the host reports it and fails the run.
constexpr std::size_t minibeam_urban_firstfail_claim_slot = 139;
constexpr std::size_t minibeam_urban_firstfail_history_slot = 140;
constexpr std::size_t minibeam_urban_firstfail_stepseg_slot = 141;
constexpr std::size_t minibeam_urban_firstfail_reason_slot = 142;
// Diagnostic: sub-ulp remainder completions (FP32 granularity, position
// error < 2 ulp of the macro step; counted, non-fatal).
constexpr std::size_t minibeam_urban_subulp_drop_slot = 143;
// First numeric-guard evidence (fix C2.3 investigation): 144 = claim flag,
// 145 = site tag (1 Cu, 2 water primary, 3 secondary), 146..151 = float
// bits of (g_final, g_prop, t_msc, lambda0, range, par1), 152 = float bits
// of par3. First trip only; lets the host reproduce the exact inputs.
constexpr std::size_t minibeam_urban_guardev_claim_slot = 144;
constexpr std::size_t minibeam_urban_guardev_base_slot = 145;
constexpr std::size_t minibeam_urban_guardev_count = 8;
#endif

#if defined(CARBON_ENABLE_MINIBEAM)
struct CopperElasticRecord {
    float energy_MeV_per_u{};
    float macroscopic_rate_per_mm{};
    float transfer_fraction{};
    float target_mass_MeV{};
    std::uint32_t target_a{};
};

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

inline float minibeam_survivor_stopping_scale(
    float energy_MeV_per_u, float fallback, const float* energies,
    const float* scales, std::size_t count) noexcept {
    if (!energies || !scales || count == 0) return fallback;
    if (energy_MeV_per_u <= energies[0]) return scales[0];
    if (energy_MeV_per_u >= energies[count - 1]) return scales[count - 1];
    std::size_t low = 0;
    std::size_t high = count - 1;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (energies[middle] <= energy_MeV_per_u) low = middle;
        else high = middle;
    }
    const auto fraction = sycl::clamp(
        (energy_MeV_per_u - energies[low]) /
            (energies[high] - energies[low]),
        0.0F, 1.0F);
    return scales[low] + fraction * (scales[high] - scales[low]);
}

inline float minibeam_copper_ion_stopping(
    const float* energies, const float* carbon_values, std::uint32_t count,
    const float* species_ratios, const std::uint8_t* species_present,
    float kinetic_energy_MeV, int atomic_number, int mass_number,
    int reference_atomic_number) noexcept {
    if (!energies || !carbon_values || count < 2 || atomic_number <= 0 ||
        mass_number <= 0) return 0.0F;
    const auto energy_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    std::uint32_t low = 0;
    if (energy_u >= energies[count - 1]) {
        low = count - 2;
    } else if (energy_u > energies[0]) {
        std::uint32_t high = count - 1;
        while (high - low > 1) {
            const auto middle = low + (high - low) / 2;
            if (energies[middle] <= energy_u) low = middle;
            else high = middle;
        }
    }
    const auto fraction = sycl::clamp(
        (energy_u - energies[low]) / (energies[low + 1] - energies[low]),
        0.0F, 1.0F);
    const auto carbon_stopping = carbon_values[low] + fraction *
        (carbon_values[low + 1] - carbon_values[low]);
    const auto species = static_cast<std::size_t>(atomic_number) *
        IonStoppingPowerTables::mass_stride + static_cast<std::size_t>(mass_number);
    if (species_ratios && species_present &&
        atomic_number < static_cast<int>(IonStoppingPowerTables::atomic_number_slots) &&
        mass_number < static_cast<int>(IonStoppingPowerTables::mass_stride) &&
        species_present[species] != 0) {
        const auto base = species * count;
        const auto ratio = species_ratios[base + low] + fraction *
            (species_ratios[base + low + 1] - species_ratios[base + low]);
        return carbon_stopping * ratio;
    }
    const auto charge = static_cast<float>(atomic_number);
    return carbon_stopping * charge * charge /
        static_cast<float>(reference_atomic_number * reference_atomic_number);
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

#endif

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
    UnifiedEmClock urban_ionization_clock{};
    UnifiedEmState unified_secondary_state{};
#if CARBON_EM_LOCAL_AUDIT
    std::array<std::uint64_t,8> unified_secondary_audit{};
#endif
    uint32_t sec_steps{};
    std::uint64_t local_sec_rate_queries{};
    std::uint64_t local_sec_steps{};
    [[no_unique_address]] std::conditional_t<Compact, EmptySecondaryUrbanState,
                                            SecondaryUrbanState> urban;
    // Sparse inelastic tallies ([3],[4] of the device audit) are written
    // directly to the global device audit at event time and are not carried
    // across continuation launches.
    std::array<double,4> he4_audit{};
    std::array<std::uint64_t,CARBON_SECONDARY_STEP_PROFILE?60:0> sec_prof{};
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
// The eight diagnostic counters are spread across logical threads so a warp's
// same-index increments do not serialize on one global address. Counter i lane
// s lives at global[i*kShards+s]; flush writes shard 0, and shipment merges all
// shards back into the eight original counters without changing their values.
inline constexpr unsigned kUnifiedEmAuditShards = CARBON_EM_AUDIT_SHARDS;
inline constexpr unsigned kUnifiedEmAuditCounters = 8;
inline void flush_unified_em_audit(std::uint64_t* global,
                                  const std::array<std::uint64_t,8>& local) {
#pragma unroll
    for(int i=0;i<8;++i) {
        if(local[i]) {
            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                sycl::memory_scope::device,sycl::access::address_space::global_space>
                count(global[static_cast<std::size_t>(i)*kUnifiedEmAuditShards]);
            count.fetch_add(local[i]);
        }
    }
}

#include "detail/sycl_device_math.inc"
#include "detail/sycl_urban_global.inc"

// Read-only, opt-in capture. It neither samples RNG nor modifies dose/ledgers.
struct NeutronBirthBuffer {
    NeutronBirthRecord* records{};
    std::uint64_t* count{};
    std::uint64_t capacity{};
};
inline void record_neutron_birth(
    NeutronBirthBuffer buffer, const Cinel03DeviceProduct& product,
    float x, float y, float z, Direction3F parent_direction,
    float event_cos, float event_sin, std::uint64_t history,
    std::uint64_t parent_stream, std::uint32_t collision,
    std::uint32_t product_index, std::uint32_t generation,
    int material, int parent_z, int parent_a, float parent_weight = 1.f) {
    if (!buffer.records || product.pdg != 2112) return;
    const auto local = rotate_cinel03_event_azimuth(
        product.local_direction_x, product.local_direction_y,
        product.local_direction_z, event_cos, event_sin);
    const auto direction = rotate_local_direction(local.x, local.y, local.z, parent_direction);
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space> count(*buffer.count);
    const auto index = count.fetch_add(1);
    if (index >= buffer.capacity) return; // Host rejects truncation unconditionally.
    NeutronBirthRecord r{};
    r.history=history; r.collision=collision; r.product_index=product_index;
    r.rng_stream=rng::event_product_stream(parent_stream, collision,
        generation == 0 ? rng::branch_role_primary_neutral : rng::branch_role_cascade_neutral,
        product_index);
    r.generation=generation; r.material_section=material;
    r.parent_z=parent_z; r.parent_a=parent_a; r.role=product.role;
    r.kinetic_energy_MeV=product.kinetic_energy_MeV;
    r.x_mm=x; r.y_mm=y; r.z_mm=z;
    r.direction_x=direction.x; r.direction_y=direction.y; r.direction_z=direction.z;
    r.weight=parent_weight*product.weight;
    buffer.records[index]=r;
}


// Fix C2.3: Urban accepted-step contract checks (device-side).
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
inline int urban_step_progress_reason(const CorrelatedScatteringStep& s,
                                      const float traversed_mm,
                                      const float step_mm,
                                      const float allowance_mm,
                                      bool& subulp_complete) noexcept {
    subulp_complete = false;
    const float g = s.final_geom_path_mm;
    const float allow_ulp = urban_ulp_above(allowance_mm);
    if (g > allowance_mm + allow_ulp) return 7;
    const float next = traversed_mm + g;
    if (next > traversed_mm) {
        const float step_ulp = urban_ulp_above(step_mm);
        if (next > step_mm + step_ulp) return 7;
        return 0;
    }
    const float rem = step_mm - traversed_mm;
    const float step_ulp = urban_ulp_above(step_mm);
    if (rem <= 2.0F * step_ulp) {
        subulp_complete = true;
        return 0;
    }
    return 6;
}

#if defined(CARBON_ENABLE_MINIBEAM)
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
#endif

#if defined(CARBON_ENABLE_MINIBEAM)
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
#endif
#include "detail/sycl_score_device.inc"

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

inline void score_minibeam_energy_band_roi_device(
    DoseAtomicT* scores, const int z, const int a,
    const float kinetic_energy_MeV, const int voxel,
    const std::size_t number_of_depth_bins, const std::size_t voxel_bins_x,
    const std::size_t voxel_bins_y, const std::uint8_t* region_by_x_bin,
    const float deposited_energy_MeV) {
    if (scores == nullptr || voxel < 0 || a <= 0 ||
        !(deposited_energy_MeV > 0.0F)) return;
    const auto species = minibeam_energy_band_species_category(z, a);
    if (species >= minibeam_energy_band_species_count) return;
    const auto plane = voxel_bins_x * voxel_bins_y;
    const auto depth = static_cast<std::size_t>(voxel) / plane;
    if (depth >= number_of_depth_bins) return;
    const auto x_bin = static_cast<std::size_t>(voxel) % voxel_bins_x;
    if (region_by_x_bin == nullptr) return;
    const auto region = static_cast<std::size_t>(region_by_x_bin[x_bin]);
    if (region >= minibeam_fixed_region_count) return;
    const auto energy_per_u = kinetic_energy_MeV / static_cast<float>(a);
    const std::size_t energy_band = energy_per_u < 50.0F ? 0U :
        (energy_per_u <= 300.0F ? 1U : 2U);
    const auto index =
        (((species * minibeam_energy_band_count + energy_band) *
           minibeam_fixed_region_count + region) * number_of_depth_bins) + depth;
    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic(scores[index]);
    atomic.fetch_add(static_cast<DoseAtomicT>(deposited_energy_MeV));
}

inline void score_minibeam_primary_roi_device(
    DoseAtomicT* scores, const std::size_t kind,
    const float kinetic_energy_MeV, const int mass_number, const int voxel,
    const std::size_t number_of_depth_bins, const std::size_t voxel_bins_x,
    const std::size_t voxel_bins_y, const std::uint8_t* region_by_x_bin,
    const float value) {
    if (scores == nullptr || voxel < 0 || kind >= minibeam_primary_roi_kind_count ||
        !(value > 0.0F)) return;
    const auto plane = voxel_bins_x * voxel_bins_y;
    const auto depth = static_cast<std::size_t>(voxel) / plane;
    if (depth >= number_of_depth_bins) return;
    const auto x_bin = static_cast<std::size_t>(voxel) % voxel_bins_x;
    if (region_by_x_bin == nullptr) return;
    const auto region = static_cast<std::size_t>(region_by_x_bin[x_bin]);
    if (region >= minibeam_fixed_region_count) return;
    const std::size_t energy_band = primary_energy_band(kinetic_energy_MeV, mass_number);
    if (energy_band >= minibeam_energy_band_count) return;
    const auto index =
        (((kind * minibeam_energy_band_count + energy_band) *
           minibeam_fixed_region_count + region) * number_of_depth_bins) + depth;
    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic(scores[index]);
    atomic.fetch_add(static_cast<DoseAtomicT>(value));
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

// Content identity for cached read-only tables: FNV-1a over the bytes that are
// actually uploaded. Paths are never used as equivalence proof.
inline std::uint64_t fnv1a64_bytes(const void* data, std::size_t bytes,
                                   std::uint64_t h = 1469598103934665603ULL) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}
template <class T>
inline std::uint64_t fnv1a64_vec(const std::vector<T>& values, std::uint64_t h) {
    return fnv1a64_bytes(values.data(), values.size() * sizeof(T), h);
}

}  // namespace

[[gnu::noinline]] SchneiderCtDeviceContext upload_schneider_ct_device_context(
    sycl::queue& queue,
    carbon::detail::DeviceMemoryTracker& mem_tracker,
    const TransportConfig& config,
    SchneiderHostCache* host_cache = nullptr) {

    SchneiderCtDeviceContext ctx{};
    ctx.mode = config.material_physics_mode;
    ctx.unified_water = config.unified_water_nuclear_transport;

    if (!ctx.uses_cinel03() || config.is_primary_attenuation_only_mode()) {
        return ctx;
    }
    PureWaterMaterial water_material{};
    SchneiderMaterialTable water_source_materials{};
    if (ctx.unified_water) {
        water_material = PureWaterMaterial::from_probe(config.unified_water_material_file);
        water_source_materials = SchneiderMaterialTable::from_topas_file(
            config.ct_schneider_file.empty() ? std::filesystem::path("data/HUtoMaterialSchneider.txt") : config.ct_schneider_file);
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

    // 1. Primary Target Sampler
    if (std::filesystem::exists(primary_rate_file)) {
        const SchneiderRateTable* rate_table = nullptr;
        const SchneiderTargetSampler* target_sampler = nullptr;
        std::shared_ptr<SchneiderRateTable> local_rate_table;
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
        const auto cdf_size = target_sampler->cdf_table().size();
        const auto tot_size = target_sampler->total_mass_rates().size();
        float* dev_cdf = mem_tracker.allocate<float>(cdf_size);
        float* dev_total_rates = mem_tracker.allocate<float>(tot_size);
        if (dev_cdf == nullptr || dev_total_rates == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(target_sampler->cdf_table().data(), dev_cdf, cdf_size);
        queue.copy(target_sampler->total_mass_rates().data(), dev_total_rates, tot_size);
        ctx.primary_sampler = target_sampler->device_table();
        ctx.primary_sampler.cdf_table = dev_cdf;
        ctx.primary_sampler.total_mass_rates = dev_total_rates;
        // v3: raw partials + per-target domain for the masked device path.
        if (target_sampler->rate_version() == 3) {
            const auto& partials = target_sampler->partial_rates();
            float* dev_partial = mem_tracker.allocate<float>(partials.size());
            float* dev_emin = mem_tracker.allocate<float>(target_sampler->domain_emin().size());
            float* dev_emax = mem_tracker.allocate<float>(target_sampler->domain_emax().size());
            unsigned char* dev_has =
                mem_tracker.allocate<unsigned char>(target_sampler->domain_has().size());
            if (dev_partial == nullptr || dev_emin == nullptr || dev_emax == nullptr ||
                dev_has == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(partials.data(), dev_partial, partials.size());
            queue.copy(target_sampler->domain_emin().data(), dev_emin,
                       target_sampler->domain_emin().size());
            queue.copy(target_sampler->domain_emax().data(), dev_emax,
                       target_sampler->domain_emax().size());
            queue.copy(target_sampler->domain_has().data(), dev_has,
                       target_sampler->domain_has().size());
            ctx.primary_sampler.partial_rates = dev_partial;
            ctx.primary_sampler.domain_emin = dev_emin;
            ctx.primary_sampler.domain_emax = dev_emax;
            ctx.primary_sampler.domain_has = dev_has;
        }
    }

    // 1b. Elastic C12 SCHNELXS sampler (diagnostic only; absent file
    // keeps elastic disabled with zero behavior change).
    if (!config.ct_elastic_section_rate_file.empty()) {
        const auto elastic_table = SchneiderRateTable::from_binary(
            config.ct_elastic_section_rate_file, {}, "SCHNELXS", 3);
        const SchneiderTargetSampler elastic_sampler(elastic_table);
        const auto cdf_size = elastic_sampler.cdf_table().size();
        const auto tot_size = elastic_sampler.total_mass_rates().size();
        float* dev_cdf = mem_tracker.allocate<float>(cdf_size);
        float* dev_total_rates = mem_tracker.allocate<float>(tot_size);
        if (dev_cdf == nullptr || dev_total_rates == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(elastic_sampler.cdf_table().data(), dev_cdf, cdf_size);
        queue.copy(elastic_sampler.total_mass_rates().data(), dev_total_rates, tot_size);
        ctx.elastic_sampler = elastic_sampler.device_table();
        ctx.elastic_sampler.cdf_table = dev_cdf;
        ctx.elastic_sampler.total_mass_rates = dev_total_rates;
        const auto& partials = elastic_sampler.partial_rates();
        float* dev_partial = mem_tracker.allocate<float>(partials.size());
        float* dev_emin = mem_tracker.allocate<float>(elastic_sampler.domain_emin().size());
        float* dev_emax = mem_tracker.allocate<float>(elastic_sampler.domain_emax().size());
        unsigned char* dev_has =
            mem_tracker.allocate<unsigned char>(elastic_sampler.domain_has().size());
        if (dev_partial == nullptr || dev_emin == nullptr || dev_emax == nullptr ||
            dev_has == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(partials.data(), dev_partial, partials.size());
        queue.copy(elastic_sampler.domain_emin().data(), dev_emin,
                   elastic_sampler.domain_emin().size());
        queue.copy(elastic_sampler.domain_emax().data(), dev_emax,
                   elastic_sampler.domain_emax().size());
        queue.copy(elastic_sampler.domain_has().data(), dev_has,
                   elastic_sampler.domain_has().size());
        ctx.elastic_sampler.partial_rates = dev_partial;
        ctx.elastic_sampler.domain_emin = dev_emin;
        ctx.elastic_sampler.domain_emax = dev_emax;
        ctx.elastic_sampler.domain_has = dev_has;
        std::cout << "[schneider-elastic-rates] mode=elastic-diagnostic sections=25 "
                  << "targets=13 energies=921 source="
                  << config.ct_elastic_section_rate_file << '\n';
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
        if (config.primary_rest_mass_MeV > 0.0 ||
            config.primary_atomic_number != 6 || config.primary_mass_number != 12) {
            for (const auto& event : primary_package.interactions())
                config.require_primary_reference_mass(event.projectile_rest_mass, "Primary CINEL03 package");
        }
        for (const auto& node : c12_dev->energy_nodes) {
            if (node.projectile_z != config.primary_atomic_number ||
                node.projectile_a != config.primary_mass_number)
                throw std::runtime_error("Primary CINEL03 binary projectile does not match YAML source Z/A");
        }
        auto* dev_c12_nodes = mem_tracker.allocate<Cinel03EnergyNode>(c12_dev->energy_nodes.size());
        auto* dev_c12_offsets = mem_tracker.allocate<std::uint32_t>(c12_dev->event_offsets.size());
        auto* dev_c12_indices = mem_tracker.allocate<std::uint32_t>(c12_dev->event_indices.size());
        auto* dev_c12_ints = mem_tracker.allocate<Cinel03DeviceInteraction>(c12_dev->interactions.size());
        auto* dev_c12_prods = mem_tracker.allocate<Cinel03DeviceProduct>(c12_dev->products.size());
        if (dev_c12_nodes == nullptr || dev_c12_offsets == nullptr || dev_c12_indices == nullptr ||
            dev_c12_ints == nullptr || dev_c12_prods == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(c12_dev->energy_nodes.data(), dev_c12_nodes, c12_dev->energy_nodes.size());
        queue.copy(c12_dev->event_offsets.data(), dev_c12_offsets, c12_dev->event_offsets.size());
        queue.copy(c12_dev->event_indices.data(), dev_c12_indices, c12_dev->event_indices.size());
        queue.copy(c12_dev->interactions.data(), dev_c12_ints, c12_dev->interactions.size());
        queue.copy(c12_dev->products.data(), dev_c12_prods, c12_dev->products.size());

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
        const SecondaryRateTable* sec_rate_table = nullptr;
        std::shared_ptr<SecondaryRateTable> local_sec_rate_table;
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
        std::vector<float> sec_total_rates_float(sec_rate_table->mass_total_rates().size());
        for (std::size_t i = 0; i < sec_rate_table->mass_total_rates().size(); ++i) {
            sec_total_rates_float[i] = static_cast<float>(sec_rate_table->mass_total_rates()[i]);
        }
        std::vector<float> sec_partial_rates_float(sec_rate_table->mass_partial_rates().size());
        for (std::size_t i = 0; i < sec_rate_table->mass_partial_rates().size(); ++i) {
            sec_partial_rates_float[i] = static_cast<float>(sec_rate_table->mass_partial_rates()[i]);
        }
        float* dev_sec_total = mem_tracker.allocate<float>(sec_total_rates_float.size());
        float* dev_sec_partial = mem_tracker.allocate<float>(sec_partial_rates_float.size());
        if (dev_sec_total == nullptr || dev_sec_partial == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(sec_total_rates_float.data(), dev_sec_total, sec_total_rates_float.size());
        queue.copy(sec_partial_rates_float.data(), dev_sec_partial, sec_partial_rates_float.size());

        ctx.sec_total_rates = dev_sec_total;
        ctx.sec_partial_rates = dev_sec_partial;
        // v3: bundle-ordered projectile registry + per-(projectile,target)
        // domain for the only supported, masked device path.
        {
            const auto& projs = sec_rate_table->projectiles();
            std::vector<std::int32_t> keys;
            keys.reserve(projs.size() * 2);
            for (const auto& p : projs) {
                keys.push_back(p.z);
                keys.push_back(p.a);
            }
            std::int32_t* dev_keys = mem_tracker.allocate<std::int32_t>(keys.size());
            if (dev_keys == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(keys.data(), dev_keys, keys.size());
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
            float* dev_demin = mem_tracker.allocate<float>(demin.size());
            float* dev_demax = mem_tracker.allocate<float>(demax.size());
            unsigned char* dev_dhas = mem_tracker.allocate<unsigned char>(dhas.size());
            if (dev_demin == nullptr || dev_demax == nullptr || dev_dhas == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(demin.data(), dev_demin, demin.size());
            queue.copy(demax.data(), dev_demax, demax.size());
            queue.copy(dhas.data(), dev_dhas, dhas.size());
            ctx.sec_domain_emin = dev_demin;
            ctx.sec_domain_emax = dev_demax;
            ctx.sec_domain_has = dev_dhas;
        }
        ctx.sec_num_projectiles = static_cast<std::uint32_t>(sec_rate_table->num_projectiles());
        ctx.sec_num_sections = static_cast<std::uint32_t>(kSecondaryNumSections);
        ctx.sec_num_targets = static_cast<std::uint32_t>(kSecondaryNumTargets);
        ctx.sec_num_energies = static_cast<std::uint32_t>(sec_rate_table->num_energies());
        ctx.sec_energy_min_MeV_per_u = static_cast<float>(sec_rate_table->energy_min_mevu());
        ctx.sec_energy_step_MeV_per_u = static_cast<float>(sec_rate_table->energy_step_mevu());
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
        auto* dev_sec_nodes = mem_tracker.allocate<Cinel03EnergyNode>(sec_dev->energy_nodes.size());
        auto* dev_sec_offsets = mem_tracker.allocate<std::uint32_t>(sec_dev->event_offsets.size());
        auto* dev_sec_indices = mem_tracker.allocate<std::uint32_t>(sec_dev->event_indices.size());
        auto* dev_sec_ints = mem_tracker.allocate<Cinel03DeviceInteraction>(sec_dev->interactions.size());
        auto* dev_sec_prods = mem_tracker.allocate<Cinel03DeviceProduct>(sec_dev->products.size());
        if (dev_sec_nodes == nullptr || dev_sec_offsets == nullptr || dev_sec_indices == nullptr ||
            dev_sec_ints == nullptr || dev_sec_prods == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(sec_dev->energy_nodes.data(), dev_sec_nodes, sec_dev->energy_nodes.size());
        queue.copy(sec_dev->event_offsets.data(), dev_sec_offsets, sec_dev->event_offsets.size());
        queue.copy(sec_dev->event_indices.data(), dev_sec_indices, sec_dev->event_indices.size());
        queue.copy(sec_dev->interactions.data(), dev_sec_ints, sec_dev->interactions.size());
        queue.copy(sec_dev->products.data(), dev_sec_prods, sec_dev->products.size());

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
    return {config.em_package_file,
            config.em_delta_moments_file.empty()
                ? config.em_package_file.parent_path() / delta_moments_filename
                : config.em_delta_moments_file};
}

template<bool BroadBeam, bool Gaussian, int JointResponse = 0>
[[gnu::noinline]] TransportResult transport_sycl_impl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    using SecondaryResumeState = SecondaryResumeStateT<BroadBeam>;
    ImmutableInputCache local_inputs;
    auto& inputs = context ? context->impl_->immutable_inputs : local_inputs;
    const auto input_hits_before = inputs.hits;
    const auto input_misses_before = inputs.misses;
    RuntimeScope runtime_transport("transport_total");
    RuntimeScope runtime_setup("transport_setup_including_upload");
    config.validate();
    std::cout<<"[transport-profile] requested="<<config.transport_profile
             <<" compiled_kernel="<<(BroadBeam?(JointResponse==3?"broadbeam_fe_independent":"broadbeam_fe"):"reference_detailed")<<"\n";
    validate_schneider_ct_startup(config);
    // Parse the large read-only packages on background threads while the
    // setup below proceeds; each consumer takes the same loader's result.
    // Nothing is prefetched when a process-level cache will serve it.
    {
        SchneiderCtDeviceContext uses{};
        uses.mode = config.material_physics_mode;
        uses.unified_water = config.unified_water_nuclear_transport;
        if (uses.uses_cinel03() && !config.is_primary_attenuation_only_mode() &&
            !(context != nullptr && context->impl_->schneider_host_cache.valid)) {
            prefetch_cinel03(config.ct_schneider_primary_cinel03_file);
            prefetch_cinel03(config.ct_schneider_secondary_cinel03_file);
        }
#if defined(CARBON_ENABLE_MINIBEAM)
        if ((!BroadBeam && config.enable_minibeam) && config.minibeam_transport_mode == "copper_em" &&
            !inputs.contains<LoadedCinel03>(input_cache_key(config.minibeam_copper_inclxx_file)))
            prefetch_cinel03(config.minibeam_copper_inclxx_file);
#endif
        if ( !(context != nullptr && context->impl_->unified_em_cache.valid))
            prefetch_unified_em(unified_em_request(config));
        if ((!BroadBeam && config.enable_neutron_transport)) prefetch_neutron_inputs(config,&inputs);
    }

    const auto start = std::chrono::steady_clock::now();
    const auto& table_energies = stopping_power.energies();
    if (table_energies.size() < 2 || !is_uniform_grid(table_energies)) {
        throw std::invalid_argument("Stopping power table must have a uniform energy grid");
    }

    const auto resolved_device_name =
        device_name.empty() ? std::string("gpu") : device_name;

    sycl::queue local_queue = (context != nullptr)
                                  ? context->impl_->queue
                                  : make_sycl_queue(resolved_device_name);
    auto& queue = local_queue;
    const auto device = queue.get_device();
    const auto backend = queue.get_backend();
    const bool is_cuda_backend = backend == sycl::backend::ext_oneapi_cuda;

    const auto device_memory_budget_bytes =
        config.device_memory_budget_gib > 0.0
            ? static_cast<std::size_t>(config.device_memory_budget_gib *
                                       1024.0 * 1024.0 * 1024.0)
            : std::size_t{0};
    DeviceMemoryTracker mem_tracker{queue, device_memory_budget_bytes};

    const auto free_device = [&](auto* pointer) {
        mem_tracker.free(pointer);
    };

    const auto number_of_histories = config.number_of_histories;
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_voxels = config.number_of_voxels();
    const auto table_size = stopping_power.values().size();
    // Independent nuclear/recoil lookup tables retain their reference grid.
    const auto cross_section_table_size = table_size;

    const bool reuse_immutable_buffers = context != nullptr;
    if (context != nullptr) {
        // Identity covers the reference energy grid and nuclear cross sections,
        // not the file path. A zero cross-section hashes differently
        // from a real one.
        std::uint64_t physics_hash = 1469598103934665603ULL;
        physics_hash = fnv1a64_vec(stopping_power.energies(), physics_hash);
        physics_hash = fnv1a64_vec(cross_section.energies(), physics_hash);
        physics_hash = fnv1a64_vec(cross_section.values(), physics_hash);
        const std::string physics_signature = std::to_string(physics_hash);
        context->impl_->ensure_initialized(stopping_power, cross_section,
                                           physics_signature);
    }

    float* cross_section_device =
        context != nullptr ? context->impl_->cross_section_device
                           : mem_tracker.allocate<float>(cross_section_table_size);
    float* target_h_fraction_device =
        mem_tracker.allocate<float>(cross_section_table_size);

    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers && pointer != nullptr) {
            mem_tracker.free(pointer);
        }
    };

    float* schneider_primary_xs_device = nullptr;
    std::uint32_t schneider_xs_sections = 0;
    std::uint32_t schneider_xs_energies = 0;
    float schneider_xs_e_min = 0.0F;
    float schneider_xs_inv_dE = 0.0F;
    const bool use_unified_water = config.unified_water_nuclear_transport;

    bool use_schneider_primary_xs = use_unified_water;
    Cinel02DeviceInteraction* cinel02_interactions_device = nullptr;
    Cinel02DeviceProduct* cinel02_products_device = nullptr;
    Cinel02EnergyNode* cinel02_energy_nodes_device = nullptr;
    std::uint32_t* cinel02_event_offsets_device = nullptr;
    std::uint32_t* cinel02_event_indices_device = nullptr;
    Cinel02RateGroup* cinel02_rate_groups_device = nullptr;
    Cinel02RateSample* cinel02_rate_samples_device = nullptr;
    Cinel02MaterialRateGroup* cinel02_ct_rate_groups_device = nullptr;
    Cinel02RateSample* cinel02_ct_rate_samples_device = nullptr;
    std::uint32_t cinel02_interaction_count = 0U;
    std::uint32_t cinel02_product_count = 0U;
    std::uint32_t cinel02_energy_node_count = 0U;
    std::uint32_t cinel02_rate_group_count = 0U;
    std::uint32_t cinel02_rate_sample_count = 0U;
    std::uint32_t cinel02_ct_rate_group_count = 0U;
    std::uint32_t cinel02_ct_rate_sample_count = 0U;
    float cinel02_ct_rate_reference_density_g_per_cm3 = 1.0F;
    constexpr std::size_t kSchneiderDiagSlots =
        static_cast<std::size_t>(SchneiderDiagSlot::Count);
    constexpr std::size_t kSchneiderFloatSlots =
        static_cast<std::size_t>(SchneiderFloatSlot::Count);
    std::uint64_t* schneider_diag_device = nullptr;
    float* schneider_float_device = nullptr;
    // Bounded per-record logs (Schneider CT only). Capacities are generous
    // for full-shard runs (50k needs ~10^2 entries); overflow is counted,
    // never silently wrapped.
    constexpr std::uint32_t kSchneiderMissLogCap = 1U << 19;   // 524288
    constexpr std::uint32_t kSchneiderTrackLogCap = 1U << 19;  // 524288
    SchneiderMissRecord* schneider_miss_device = nullptr;
    SchneiderUnsupportedTrack* schneider_track_log_device = nullptr;
    std::uint32_t* schneider_miss_count_device = nullptr;
    std::uint32_t* schneider_track_count_device = nullptr;
    constexpr std::size_t kCinel02SpeciesEnergySlots =
        TransportResult::species_ledger_species_count *
        TransportResult::species_ledger_metric_count;
    float* cinel02_species_energy_device = nullptr;
    double* grid_deposited_in_device = nullptr;
    double* grid_deposited_out_device = nullptr;
    constexpr std::size_t kCinel02SpeciesTerminalSlots =
        Cinel02SpeciesLedgerSchema::species_count *
        Cinel02SpeciesLedgerSchema::terminal_reason_count;
    std::uint64_t* cinel02_species_terminal_device = nullptr;
    // CINEL02-only package/rate uploads and diagnostic allocations retired.

    // Read-only species ledger is mode-independent: secondary EM transport
    // records per-species deposited/escaped energy unconditionally on device
    // (null-checked increments), so these two buffers remain in the shared
    // CINEL03 framework. No physics effect:
    // ledger writes are isolated atomics into dedicated buffers.
    if (cinel02_species_energy_device == nullptr) {
        cinel02_species_energy_device =
            mem_tracker.allocate<float>(kCinel02SpeciesEnergySlots);
        if (cinel02_species_energy_device == nullptr) throw std::bad_alloc();
    }
    if (cinel02_species_terminal_device == nullptr) {
        cinel02_species_terminal_device =
            mem_tracker.allocate<std::uint64_t>(kCinel02SpeciesTerminalSlots);
        if (cinel02_species_terminal_device == nullptr) throw std::bad_alloc();
    }
    queue.fill(cinel02_species_energy_device, 0.0F,
               kCinel02SpeciesEnergySlots).wait_and_throw();
    queue.fill(cinel02_species_terminal_device, std::uint64_t{0},
               kCinel02SpeciesTerminalSlots).wait_and_throw();

    // Explicit in-grid/outside-grid deposited-energy sinks (global MeV).
    // Every site crediting deposited ledgers splits the same amount here
    // using that site's paired voxel-scorer guard, so total ≈ in + outside
    // and voxel_sum ≈ in_grid hold by construction. Double precision:
    // global accumulators reach 1e10 MeV where float32 atomics would absorb
    // MeV-scale adds.
    grid_deposited_in_device = mem_tracker.allocate<double>(1);
    grid_deposited_out_device = mem_tracker.allocate<double>(1);
    if (grid_deposited_in_device == nullptr ||
        grid_deposited_out_device == nullptr) {
        throw std::bad_alloc();
    }
    queue.fill(grid_deposited_in_device, 0.0, 1).wait_and_throw();
    queue.fill(grid_deposited_out_device, 0.0, 1).wait_and_throw();

    // Slab layers
    const auto enable_layered_phantom = config.enable_layered_phantom;
    const auto slab_layer_count =
        enable_layered_phantom ? config.slab_layers.size() : std::size_t{0};
    std::vector<float> slab_z_ends_host(slab_layer_count);
    std::vector<float> slab_densities_host(slab_layer_count);
    std::vector<float> slab_radiation_lengths_host(slab_layer_count);
    for (std::size_t i = 0; i < slab_layer_count; ++i) {
        slab_z_ends_host[i] = static_cast<float>(config.slab_layers[i].z_end_mm);
        slab_densities_host[i] = static_cast<float>(config.slab_layers[i].density_g_per_cm3);
        slab_radiation_lengths_host[i] =
            i < config.slab_radiation_lengths_g_per_cm2.size()
                ? static_cast<float>(config.slab_radiation_lengths_g_per_cm2[i])
                : static_cast<float>(water_radiation_length_g_per_cm2);
    }
    float* slab_z_ends_device = slab_layer_count > 0
                                   ? mem_tracker.allocate<float>(slab_layer_count)
                                   : nullptr;
    float* slab_densities_device = slab_layer_count > 0
                                       ? mem_tracker.allocate<float>(slab_layer_count)
                                       : nullptr;
    float* slab_radiation_lengths_device =
        slab_layer_count > 0
            ? mem_tracker.allocate<float>(slab_layer_count)
            : nullptr;
    if (slab_layer_count > 0) {
        queue.copy(slab_z_ends_host.data(), slab_z_ends_device, slab_layer_count);
        queue.copy(slab_densities_host.data(), slab_densities_device, slab_layer_count);
        queue.copy(slab_radiation_lengths_host.data(), slab_radiation_lengths_device,
                   slab_layer_count).wait_and_throw();
    }

    const auto enable_hetero_insert = config.enable_hetero_insert;
    const auto insert_x_min = static_cast<float>(config.hetero_insert.x_min_mm);
    const auto insert_x_max = static_cast<float>(config.hetero_insert.x_max_mm);
    const auto insert_y_min = static_cast<float>(config.hetero_insert.y_min_mm);
    const auto insert_y_max = static_cast<float>(config.hetero_insert.y_max_mm);
    const auto insert_z_min = static_cast<float>(config.hetero_insert.z_min_mm);
    const auto insert_z_max = static_cast<float>(config.hetero_insert.z_max_mm);
    const auto insert_density_g_per_cm3 =
        static_cast<float>(config.hetero_insert.density_g_per_cm3);
    const auto insert_radiation_length_g_per_cm2 =
        static_cast<float>(config.insert_radiation_length_g_per_cm2);

    // CT grid
    const auto enable_ct_grid = config.enable_ct_grid;
    float ct_origin_x = 0.0F, ct_origin_y = 0.0F, ct_origin_z = 0.0F;
    float ct_spacing_x = 0.0F, ct_spacing_y = 0.0F, ct_spacing_z = 0.0F;
    std::uint32_t ct_nx = 0, ct_ny = 0, ct_nz = 0;
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    float* ct_xs_device = nullptr;
    bool use_ct_material_xs = false;
    bool ct_material_ids_are_schneider_sections = false;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;

    if (std::getenv("CARBON_JOINT_EM_DATA") || std::getenv("CARBON_DIAGNOSTIC_PRIMARY_START_DEDX"))
        throw std::invalid_argument("Obsolete isolated EM environment switch; use em_model: g4_material_joint_v1");
    // Water/Schneider charged-particle energy loss has one implementation.
    const float em_primary_step_scale=static_cast<float>(config.em_primary_step_scale);
    const float maximum_mean_loss_fraction = BroadBeam ? static_cast<float>(config.broadbeam_maximum_mean_loss_fraction) : .01f;
    const float em_secondary_step_scale=static_cast<float>(config.em_secondary_step_scale);
    std::cout<<"[em-performance] material_cache="<<CARBON_EM_MATERIAL_CACHE
        <<" step_cache="<<CARBON_EM_STEP_CACHE<<" local_audit="<<CARBON_EM_LOCAL_AUDIT
        <<" exact_index="<<CARBON_EM_EXACT_INDEX<<"\n";
    UnifiedEmDevice unified_device;
    UnifiedEmMaterial* unified_materials=nullptr;
    UnifiedEmSpecies* unified_species=nullptr;
    UnifiedEmSectionRange* unified_sections=nullptr;
    unsigned* unified_energy_index=nullptr;
    unsigned* unified_failure_count=nullptr;
    UnifiedEmFailureRecord* unified_failure_records=nullptr;
    UnifiedEmRecord* unified_records=nullptr;
    UnifiedEmNode* unified_nodes=nullptr;
    float* delta_mean_device=nullptr;
    EmCubicSegmentCandidate<float>* unified_segments=nullptr;
    const float* unified_node_energy_keys=nullptr;
    const float* unified_segment_lower_keys=nullptr;
    std::uint64_t* unified_audit=nullptr;
    std::uint64_t* unified_empty_bucket_audit=nullptr;
    std::uint64_t* unified_interval_width_audit=nullptr;
    UnifiedEmSearchAuditRecord* unified_search_audit_records=nullptr;
    std::uint32_t* unified_search_audit_count=nullptr;
    constexpr std::uint32_t kUnifiedSearchAuditCapacity=4000000;
    std::uint64_t* sec_step_profile_device=nullptr;
    if(CARBON_SECONDARY_STEP_PROFILE) {
        sec_step_profile_device=mem_tracker.allocate<std::uint64_t>(60);
        if(!sec_step_profile_device)throw std::bad_alloc();
        queue.fill(sec_step_profile_device,std::uint64_t{0},60).wait_and_throw();
    }
    const bool global_urban=!BroadBeam && config.multiple_scattering_model=="urban_v2";
    const bool uses_urban_package=global_urban;
    const bool retain_physics_device = context != nullptr && [] {
        const char* value=std::getenv("CARBON_RETAIN_PHYSICS_DEVICE");
        if (!value || std::string_view(value)=="1") return true;
        if (std::string_view(value)=="0") return false;
        throw std::invalid_argument("CARBON_RETAIN_PHYSICS_DEVICE must be 0 or 1");
    }();
    if (context && !retain_physics_device) {
        context->impl_->resident_em.reset(); context->impl_->resident_urban.reset();
        context->impl_->resident_air.reset();
    }
    if (context && retain_physics_device) {
        if (!uses_urban_package) context->impl_->resident_urban.reset();
        if (config.minibeam_air_mcs_model != "urban_v2") context->impl_->resident_air.reset();
    }
    const auto resident_tables = [&](auto& slot,const std::string& key,const char* name) {
        const bool hit=slot && slot->signature==key;
        if (!hit) slot=std::make_shared<ResidentPhysicsTables>(queue,key);
        mem_tracker.account_retained(slot->bytes);
        std::cout<<"[device-physics-cache] "<<name<<" "<<(hit?"hit":"miss")
                 <<" resident_bytes="<<slot->bytes<<"\n";
        return slot;
    };

    const float urban_water_half_width_mm=static_cast<float>(config.urban_water_half_width_mm);
    UrbanMcsView urban_device{};
    std::vector<UrbanMcsSpecies> urban_species_host;
    std::uint64_t* urban_step_counts=nullptr;
    if(uses_urban_package) {
        const auto& package = inputs.get<UrbanMcsPackage>(input_cache_key(config.urban_mcs_package_file),
            [&] { return UrbanMcsPackage::load(config.urban_mcs_package_file); });
        if (global_urban) {
            const auto index = package.view().projectile(config.primary_atomic_number, config.primary_mass_number);
            if (index < 0) throw std::invalid_argument("Urban package lacks the YAML primary projectile");
            config.require_primary_reference_mass(package.species[index].mass_mev, "Urban package");
            if ((!BroadBeam && config.enable_minibeam) && config.minibeam_copper_mcs_model == "urban_v2" &&
                package.view().material(-2, static_cast<float>(config.minibeam_copper_density_g_per_cm3)) < 0)
                throw std::invalid_argument("Global minibeam Urban requires a G4_Cu (-2) material record");
        }

        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_urban,
            input_cache_key(config.urban_mcs_package_file),"urban") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        urban_device={upload("materials",package.materials),upload("species",package.species),upload("records",package.records),
            upload("nodes",package.nodes),upload("mfps",package.mfps),upload("segments",package.segments),
            static_cast<unsigned>(package.materials.size()),static_cast<unsigned>(package.species.size())};
        urban_species_host=package.species;
        urban_device.ct_density_scaling=config.urban_ct_density_scaling;
        if(config.urban_ct_density_scaling)
            std::cout<<"[urban-v2] research CT reference-density scaling enabled; native-density dose validation pending\n";
        urban_step_counts=mem_tracker.allocate<std::uint64_t>(2*package.species.size());
        if(!urban_step_counts)throw std::bad_alloc();
        queue.fill(urban_step_counts,std::uint64_t{0},2*package.species.size()).wait_and_throw();
        std::cout<<"[urban-v2] "
                 <<"global all transported charged ions"
                 <<"; materials="<<package.materials.size()
                 <<" species="<<package.species.size()<<"\n";
    }
    const bool minibeam_air_urban = config.minibeam_air_mcs_model == "urban_v2";
    const float air_slit_max_step = static_cast<float>(config.minibeam_air_slit_max_step_mm);
    UrbanMcsView air_urban_device{};
    int air_urban_projectile = -1;
    if (minibeam_air_urban) {
        if (!(!BroadBeam && config.enable_minibeam) || config.minibeam_transport_mode != "copper_em" ||
            !config.minibeam_copper_exact_material_boundaries)
            throw std::invalid_argument("Air Urban requires copper_em with exact material boundaries");
        const auto& package = inputs.get<UrbanMcsPackage>(input_cache_key(config.minibeam_air_urban_package_file),
            [&] { return UrbanMcsPackage::load(config.minibeam_air_urban_package_file); });
        air_urban_projectile = package.view().projectile(config.primary_atomic_number, config.primary_mass_number);
        if (package.materials.size() != 1 || package.materials[0].section != -3 || air_urban_projectile < 0)
            throw std::invalid_argument("Air Urban requires a G4_AIR (-3) bank containing source ZA");
        if (!config.minibeam_air_ionization_rate_file.empty() &&
            (std::abs(package.materials[0].density-0.00120479F)>1e-9F ||
             std::abs(package.materials[0].production_cut_mm-0.05F)>1e-7F))
            throw std::invalid_argument("Air ionization clock and Urban material/cut must match");
        config.require_primary_reference_mass(package.species[air_urban_projectile].mass_mev, "Air Urban package");
        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_air,
            input_cache_key(config.minibeam_air_urban_package_file),"air") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        air_urban_device={upload("materials",package.materials),upload("species",package.species),upload("records",package.records),
            upload("nodes",package.nodes),upload("mfps",package.mfps),upload("segments",package.segments),
            static_cast<unsigned>(package.materials.size()),static_cast<unsigned>(package.species.size())};
        std::cout << "[air-urban] primary beamline G4_AIR\n";
    }
    bool nonion_primary_package_safe=true;
    {
        const std::string unified_em_signature =
            input_cache_key(config.em_package_file)+"|"+
            input_cache_key(unified_em_request(config).delta_file);
        std::shared_ptr<const UnifiedEmPackage> unified_package;
        std::shared_ptr<const std::vector<float>> unified_delta_means;
        bool unified_em_cache_hit=false;
        if(context!=nullptr){
            auto& cached=context->impl_->unified_em_cache;
            if(cached.valid && cached.signature==unified_em_signature){
                unified_package=cached.package;unified_delta_means=cached.delta_means;
                unified_em_cache_hit=true;++context->impl_->physics_cache_hits;
            }
        }
        if(!unified_package){
            // Package and delta moments are parsed together (possibly already
            // on a prefetch thread); the second stage reports zero here.
            auto loaded=runtime_call("unified_em_package_load_verify",[&]{return take_unified_em(unified_em_request(config));});
            unified_package=std::move(loaded.package);
            unified_delta_means=runtime_call("delta_moments_load_verify",[&]{return std::move(loaded.delta_means);});
            if(context!=nullptr){
                auto& cached=context->impl_->unified_em_cache;
                cached.valid=true;cached.signature=unified_em_signature;
                cached.package=unified_package;cached.delta_means=unified_delta_means;
                cached.package_node_count=unified_package->nodes.size();
                ++context->impl_->physics_cache_misses;
            }
        }
        const auto& package=*unified_package;
        const auto& delta_means=*unified_delta_means;
        bool primary_record_found = false;
        for (const auto& record : package.records) {
            if (record.z == static_cast<unsigned>(config.primary_atomic_number) &&
                record.a == static_cast<unsigned>(config.primary_mass_number)) {
                primary_record_found = true;
                if(record.is_ion!=0)nonion_primary_package_safe=false;
                for(unsigned j=0;j<record.node_count;++j)
                    if(package.nodes[record.node_offset+j].fluctuation==1)nonion_primary_package_safe=false;
                config.require_primary_reference_mass(record.mass, "Unified EM package");
            }
        }
        if (!primary_record_found)
            throw std::invalid_argument("Unified EM package lacks the YAML primary projectile");
        auto resident=retain_physics_device ? resident_tables(context->impl_->resident_em,unified_em_signature,"em") : nullptr;
        auto upload=[&]<class T>(const char* name,const std::vector<T>& values){
            if (resident) return resident->upload(name,values,mem_tracker);
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        std::cout<<"[physics-cache] unified_em="<<(unified_em_cache_hit?"hit":"miss")
                 <<" nodes="<<package.nodes.size()<<"\n";
        RuntimeScope t_em_device("setup_em_device_upload");
        delta_mean_device=upload("delta_means",delta_means);
        if(uses_urban_package)std::cout<<"[urban-ionization-clock] native proposal clock limits true transport paths; aggregate energy law unchanged\n";
        std::cout<<"[delta-moments] condensed_partition_v1; aggregate Gamma; nodes="
                 <<package.nodes.size()<<"\n";
        std::vector<UnifiedEmSectionRange> sections(26);
        for(unsigned i=0;i<package.materials.size();++i) {
            auto& range=sections[package.materials[i].section+1];
            if(range.begin<0)range.begin=i;
            range.end=i;
        }
        unified_failure_count=mem_tracker.allocate<unsigned>(1);
        unified_failure_records=mem_tracker.allocate<UnifiedEmFailureRecord>(16);
        if(!unified_failure_count || !unified_failure_records)throw std::bad_alloc();
        queue.fill(unified_failure_count,0u,1).wait_and_throw();
        unified_sections=upload("sections",sections);
        if constexpr(CARBON_EM_EXACT_INDEX) {
            if (resident && resident->contains("energy_index"))
                unified_energy_index=resident->template get<std::remove_pointer_t<decltype(unified_energy_index)>>("energy_index");
            else {
                const auto index=build_unified_em_index(package);
                unified_energy_index=upload("energy_index",index);
            }
        }
        unified_materials=upload("materials",package.materials);unified_species=upload("species",package.species);
        unified_records=upload("records",package.records);unified_nodes=upload("nodes",package.nodes);unified_segments=upload("segments",package.segments);
        if constexpr(CARBON_EM_SPLIT_SEARCH_KEYS) {
            std::vector<float> node_keys(package.nodes.size());
            for(std::size_t i=0;i<package.nodes.size();++i)node_keys[i]=package.nodes[i].energy;
            std::vector<float> segment_keys(package.segments.size());
            for(std::size_t i=0;i<package.segments.size();++i)segment_keys[i]=package.segments[i].lower;
            unified_node_energy_keys=upload("node_keys",node_keys);
            unified_segment_lower_keys=upload("segment_keys",segment_keys);
            std::cout<<"[em-search-keys] node_keys="<<node_keys.size()
                     <<" segment_keys="<<segment_keys.size()<<"\n";
        }
        unified_audit=mem_tracker.allocate<std::uint64_t>(kUnifiedEmAuditCounters*kUnifiedEmAuditShards);if(!unified_audit)throw std::bad_alloc();
        queue.fill(unified_audit,std::uint64_t{0},kUnifiedEmAuditCounters*kUnifiedEmAuditShards).wait_and_throw();
        if constexpr(CARBON_EM_EMPTY_BUCKET_AUDIT) {
            unified_empty_bucket_audit=mem_tracker.allocate<std::uint64_t>(10);
            if(!unified_empty_bucket_audit)throw std::bad_alloc();
            queue.fill(unified_empty_bucket_audit,std::uint64_t{0},10).wait_and_throw();
        }
        if constexpr(CARBON_EM_INTERVAL_WIDTH_AUDIT) {
            unified_interval_width_audit=mem_tracker.allocate<std::uint64_t>(15);
            if(!unified_interval_width_audit)throw std::bad_alloc();
            queue.fill(unified_interval_width_audit,std::uint64_t{0},15).wait_and_throw();
        }
        if constexpr(CARBON_EM_SEARCH_KEY_AUDIT) {
            unified_search_audit_records=
                mem_tracker.allocate<UnifiedEmSearchAuditRecord>(kUnifiedSearchAuditCapacity);
            unified_search_audit_count=mem_tracker.allocate<std::uint32_t>(1);
            if(!unified_search_audit_records || !unified_search_audit_count)
                throw std::bad_alloc();
            queue.fill(unified_search_audit_count,std::uint32_t{0},1).wait_and_throw();
        }
        unified_device={unified_materials,unified_species,unified_records,unified_nodes,unified_segments,static_cast<unsigned>(package.materials.size()),unified_sections,unified_energy_index,delta_mean_device,unified_node_energy_keys,unified_segment_lower_keys
#if CARBON_EM_EMPTY_BUCKET_AUDIT
            ,unified_empty_bucket_audit
#endif
#if CARBON_EM_INTERVAL_WIDTH_AUDIT
            ,unified_interval_width_audit
#endif
        };
        unified_device.shared_prepare_grid=package.shared_prepare_grid;
        std::cout<<"[em-shared-prepare-grid] certified="<<unified_device.shared_prepare_grid<<"\n";

        t_em_device.finish();
        std::cout<<"[unified-em] all 18 charged ions; water + Schneider density nodes; native particle step parameters plus 1% combined mean-loss guard; local aggregate delta deposition; density cut-onset/patient accuracy validation pending\n";
    }
    // One host-table cache per process for the Schneider nuclear packages.
    // Reset only when
    // the validated inputs or device change.
    SchneiderHostCache* schneider_host_cache = nullptr;
    if (context != nullptr) {
        auto& cache = context->impl_->schneider_host_cache;
        const std::string schneider_signature =
            std::to_string(config.primary_atomic_number) + ":" + std::to_string(config.primary_mass_number) + "|" +
            config.ct_schneider_primary_rate_file.string() + "|" +
            config.ct_schneider_primary_cinel03_file.string() + "|" +
            config.ct_schneider_secondary_rate_file.string() + "|" +
            config.ct_schneider_secondary_cinel03_file.string() + "|" +
            config.ct_elastic_section_rate_file.string() + "|" +
            (config.unified_water_nuclear_transport ? "water" : "ct") + "|" +
            device_name;
        if (!cache.valid || cache.signature != schneider_signature) {
            cache = SchneiderHostCache{};
            cache.signature = schneider_signature;
            cache.valid = true;
        }
        schneider_host_cache = &cache;
    }
    const auto ct_secondary_exact_faces = config.ct_secondary_exact_faces;
    const auto ct_secondary_mcs_off = config.ct_secondary_mcs_off_diagnostic;

    // Legacy C12-only diagnostic is separate from the all-projectile bank.
    const bool use_ct_elastic = config.ct_elastic_diagnostic;
    const bool ct_elastic_all_targets = config.ct_elastic_all_targets;
    // Production speed switch: skip reporting-only joint diagnostic atomics.
    // Slots 1 (energy domain) and 2 (miss/blocked) stay always-on: they feed
    // fail-closed quality gates. Skipped counters never feed transport.
    const bool enable_electron_joint_diagnostics = config.electron_joint_diagnostics;
    const bool use_independent_charged=(!BroadBeam || JointResponse==3) && !config.material_electron_charged_index_file.empty();
    const bool use_electron_fp32=use_independent_charged;
    const bool use_material_electron=use_independent_charged;
    const bool use_material_ct=use_material_electron && config.enable_ct_grid;
    std::uint8_t* broadbeam_electron_mask=nullptr;
    std::uint8_t* broadbeam_birth_regions=nullptr;
    std::uint64_t* broadbeam_region_work=nullptr;
    const bool retain_electron_ct=config.material_electron_retain_bank && context;
    bool retained_ct_arrays=false;

    std::uint64_t* material_untracked_device=nullptr; // photon / other, micro-MeV
    struct MaterialPacketFailure {
        std::uint64_t history{},step{},rng{};
        electron_fp32::ElectronEnergyPacket before{},after{};
    };
    MaterialPacketFailure* material_failure_device=nullptr;
    const bool defer_material_packets=use_material_electron;
    std::shared_ptr<IndependentElectronFloatDevice> independent_float_bank;
    electron_fp32::ElectronBirthBankView float_birth{};
    electron_fp32::ElectronJointPilotView float_joint{};
    electron_fp32::ChargedBankView float_charged{};
    electron_fp32::PhotonResponseBankView float_photon{};
    float* float_electron_grid_ledger=nullptr;
    float* float_electron_vr_ledger=nullptr; // per-lane signed roulette weight gain
    std::uint64_t* electron_vr_work=nullptr; // killed / survived roulette tasks
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
    electron_segment_diagnostics::Sample* float_segment_samples=nullptr;
    std::uint64_t* float_segment_counts=nullptr;
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
    float* float_range_delta=nullptr;
    std::uint64_t* float_range_counts=nullptr;
#endif
#ifdef CARBON_ELECTRON_STEP_DIAGNOSTICS
    std::uint64_t* float_step_diagnostics=nullptr;
#endif
    std::uint64_t* independent_charged_work=nullptr;
    std::uint64_t* independent_photon_work=nullptr;
    std::uint64_t* joint_pilot_work=nullptr;

    struct MaterialBirthTask {
        QueuedElectronBirth birth;
        std::array<double,3> position;
        std::array<float,3> direction;
        double density;
        std::uint64_t rng,history;
        std::uint32_t step,table;
        double weight;
    };
    static_assert(sizeof(MaterialBirthTask)==104);
    MaterialBirthTask* material_tasks=nullptr;
    std::uint64_t* material_task_counts=nullptr; // produced / consumed
    std::uint64_t* material_queue_work=nullptr; // advances / source restarts / boundary restarts
    const auto material_task_capacity=static_cast<std::uint32_t>(
        std::uint64_t(config.material_electron_queue_budget_MiB)*1024*1024/sizeof(MaterialBirthTask));
    const auto material_queue_chunk=config.material_electron_queue_chunk;
    const auto material_queue_lanes=config.material_electron_queue_lanes;
    const auto material_group_size=config.material_electron_group_size;
    // Order indices, retaining each birth's draws and transport RNG stream.
    const bool electron_order_enabled=BroadBeam && use_electron_fp32 &&
        config.broadbeam_independent_electrons && config.broadbeam_electron_birth_order;
    constexpr unsigned electron_order_bins=33;
    std::uint32_t* electron_order=nullptr;
    unsigned char* electron_order_keys=nullptr;
    std::uint32_t* electron_order_counts=nullptr;
    std::uint32_t* electron_order_offsets=nullptr;
    if(electron_order_enabled) {
        electron_order=mem_tracker.allocate<std::uint32_t>(material_task_capacity);
        electron_order_keys=mem_tracker.allocate<unsigned char>(material_task_capacity);
        electron_order_counts=mem_tracker.allocate<std::uint32_t>(electron_order_bins);
        electron_order_offsets=mem_tracker.allocate<std::uint32_t>(electron_order_bins);
        if(!electron_order || !electron_order_keys || !electron_order_counts || !electron_order_offsets)throw std::bad_alloc();
        std::cout<<"[electron-birth-order] mode=1 bins="<<electron_order_bins<<"\n";
    }

    if(use_electron_fp32) {
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
        if(!enable_ct_grid||!config.enable_voxel_scoring||config.voxel_dose_mhd_output_file.empty())
            throw std::invalid_argument("Segment diagnostic requires FP32 CT with MHD dose output");
        float_segment_samples=mem_tracker.allocate<electron_segment_diagnostics::Sample>(electron_segment_diagnostics::capacity);
        float_segment_counts=mem_tracker.allocate<std::uint64_t>(2);
        if(!float_segment_samples||!float_segment_counts)throw std::bad_alloc();
        queue.fill(float_segment_counts,std::uint64_t{0},2).wait_and_throw();
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
        if(!enable_ct_grid||!config.enable_voxel_scoring||config.voxel_dose_mhd_output_file.empty())
            throw std::invalid_argument("Range diagnostic requires FP32 CT with MHD dose output");
        float_range_delta=mem_tracker.allocate<float>(number_of_voxels);
        float_range_counts=mem_tracker.allocate<std::uint64_t>(10);
        if(!float_range_delta||!float_range_counts)throw std::bad_alloc();
        queue.fill(float_range_delta,0.f,number_of_voxels);
        queue.fill(float_range_counts,std::uint64_t{0},10).wait_and_throw();
#endif
#ifdef CARBON_ELECTRON_STEP_DIAGNOSTICS
        float_step_diagnostics=mem_tracker.allocate<std::uint64_t>(64*1536);
        if(!float_step_diagnostics)throw std::bad_alloc();
        queue.fill(float_step_diagnostics,std::uint64_t{0},64*1536).wait_and_throw();
#endif
        float_electron_grid_ledger=mem_tracker.allocate<float>(2*material_queue_lanes);
        if(!float_electron_grid_ledger)throw std::bad_alloc();
        queue.fill(float_electron_grid_ledger,0.f,2*material_queue_lanes).wait_and_throw();
        float_electron_vr_ledger=mem_tracker.allocate<float>(material_queue_lanes);
        electron_vr_work=mem_tracker.allocate<std::uint64_t>(2);
        if(!float_electron_vr_ledger||!electron_vr_work)throw std::bad_alloc();
        queue.fill(float_electron_vr_ledger,0.f,material_queue_lanes);
        queue.fill(electron_vr_work,std::uint64_t{0},2).wait_and_throw();
    }
    // Low-density electron variance reduction (unbiased; see transport_config).
    // A negative density disables a feature: physical densities are positive.
    const bool electron_vr_scope=use_electron_fp32 && config.broadbeam_electron_tracking &&
        config.broadbeam_independent_electrons && config.enable_ct_grid;
    if((config.broadbeam_electron_low_density_survival!=1 || config.broadbeam_electron_tissue_survival!=1 || config.broadbeam_electron_expected_scoring) && !electron_vr_scope)
        throw std::invalid_argument("Low-density electron variance reduction requires the FP32 CT broadbeam shared-electron worker");
#ifndef CARBON_ELECTRON_VOXEL_CACHE
    if(config.broadbeam_electron_expected_scoring)
        throw std::invalid_argument("Expected-value electron scoring requires CARBON_ELECTRON_VOXEL_CACHE");
#endif
    const float electron_roulette_survival=static_cast<float>(config.broadbeam_electron_low_density_survival);
    const float electron_tissue_survival=static_cast<float>(config.broadbeam_electron_tissue_survival);
    const float electron_tissue_density=electron_vr_scope?static_cast<float>(config.broadbeam_electron_density_threshold):INFINITY;
    const auto electron_tissue_bits=electron_fp32::probability_threshold(electron_tissue_survival);
    if(electron_vr_scope)std::cout<<"[electron-tissue-roulette] survival="<<electron_tissue_survival<<" density_gt="<<electron_tissue_density<<"\n";

    const float electron_roulette_density=electron_vr_scope && electron_roulette_survival<1.f?
        static_cast<float>(config.broadbeam_electron_density_threshold):-1.f;
    const std::uint64_t electron_roulette_bits=electron_fp32::probability_threshold(electron_roulette_survival);
    const float electron_expected_density=electron_vr_scope && config.broadbeam_electron_expected_scoring?
        static_cast<float>(config.broadbeam_electron_density_threshold):-1.f;
    const float electron_expected_min_energy=static_cast<float>(config.broadbeam_electron_expected_min_energy_MeV);
    if(electron_vr_scope)
        std::cout<<"[electron-variance-reduction] roulette_survival="<<electron_roulette_survival
                 <<" roulette_density_le="<<electron_roulette_density<<" expected_scoring="<<config.broadbeam_electron_expected_scoring
                 <<" expected_density_le="<<electron_expected_density<<" expected_min_energy_MeV="<<electron_expected_min_energy<<"\n";
    if (defer_material_packets) {
        if(!use_material_electron || config.run_mode!=RunMode::research || config.enable_let_scoring ||
           config.enable_charged_origin_voxel_scoring)
            throw std::invalid_argument("Queued electrons require research transport without LET/component scoring");
        material_tasks=mem_tracker.allocate<MaterialBirthTask>(material_task_capacity);
        material_task_counts=mem_tracker.allocate<std::uint64_t>(2);
        material_queue_work=mem_tracker.allocate<std::uint64_t>(3);
        if(!material_queue_work)throw std::bad_alloc();
        queue.fill(material_queue_work,std::uint64_t{0},3);
        if(!material_tasks || !material_task_counts)throw std::bad_alloc();
        std::cout<<"[material-response-queue] capacity="<<material_task_capacity
                 <<" task_bytes="<<sizeof(MaterialBirthTask)<<" max_chunk="<<material_queue_chunk<<" lanes="<<material_queue_lanes
                 <<" group_size="<<material_group_size<<"\n";
        std::cout<<"[electron-step-refill] enabled=1 transactions_per_iteration=1\n";
    }

    std::uint64_t* electron_joint_diag_device=nullptr;

    // Electrons share one geometric transport interface. A water voxel spans
    // the physical phantom, independently of the scoring ROI; ion transport
    // still uses its native water EM/nuclear route (enable_ct_grid stays false).
    if (enable_ct_grid || use_material_electron) {
        std::shared_ptr<const CtGrid> grid_storage;
        const auto make_geometry = [&]() {
            if(enable_ct_grid) return std::make_shared<CtGrid>(CtGrid::load(
                config.ct_grid_file,config.ct_schneider_file,config.ct_dicom_origin_mode));
            auto water=std::make_shared<CtGrid>();
            water->file_version=CtGrid::version_value;
            water->nx=water->ny=water->nz=1;
            water->origin_x_mm=water->origin_y_mm=-urban_water_half_width_mm;
            water->spacing_x_mm=water->spacing_y_mm=2*urban_water_half_width_mm;
            water->spacing_z_mm=static_cast<float>(config.phantom_length_mm);
            water->density_g_per_cm3={static_cast<float>(config.water_density_g_per_cm3)};
            water->material_id={kElectronWaterSection};
            return water;
        };
        std::string geometry_key;
        if(enable_ct_grid) geometry_key=ct_input_key(config.ct_grid_file,config.ct_schneider_file,config.ct_dicom_origin_mode);
        else {
            char key[160];std::snprintf(key,sizeof key,"Water_75eV:%a:%a:%a",
                double(urban_water_half_width_mm),config.phantom_length_mm,config.water_density_g_per_cm3);
            geometry_key=key;
        }
        if(context) {
            auto& cached=context->impl_->electron_ct;
            const bool hit=cached.grid && cached.geometry_key==geometry_key;
            if(!hit) {
                queue.wait_and_throw();
                if(cached.density)sycl::free(cached.density,queue);
                if(cached.material)sycl::free(cached.material,queue);
                cached={};cached.grid=make_geometry();cached.geometry_key=geometry_key;
            }
            grid_storage=cached.grid;
            std::cout<<"[electron-geometry-cache] "<<(hit?"hit":"miss")
                     <<" medium="<<(enable_ct_grid?"Schneider_CT":"Water_75eV")<<'\n';
        } else grid_storage=make_geometry();
        const auto& grid=*grid_storage;
        if(enable_ct_grid) {
            if (grid.file_version < CtGrid::version_v2)
                throw std::invalid_argument("Unified EM CT requires a Schneider grid with section IDs");
            for (const auto section : grid.material_id)
                if (section >= 25)
                    throw std::invalid_argument("Unified EM CT material section must be in [0,24]");
        }
        std::vector<std::uint8_t> electron_region;
        if constexpr(BroadBeam && JointResponse==3) {
            std::vector<float> uncached_distance;
            const std::vector<float>* distance=nullptr;
            if(config.broadbeam_electron_region_diagnostics || config.broadbeam_electron_region_metric=="euclidean") {
                if(context) {
                    auto& cached=context->impl_->electron_ct;
                    if(cached.electron_distance_mm.empty() || cached.electron_distance_threshold!=config.broadbeam_electron_density_threshold) {
                        cached.electron_distance_mm=broadbeam_electron_box_distance(grid.density_g_per_cm3,{grid.nx,grid.ny,grid.nz},
                            {grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm},config.broadbeam_electron_density_threshold);
                        cached.electron_distance_threshold=config.broadbeam_electron_density_threshold;
                    }
                    distance=&cached.electron_distance_mm;
                } else {
                    uncached_distance=broadbeam_electron_box_distance(grid.density_g_per_cm3,{grid.nx,grid.ny,grid.nz},
                        {grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm},config.broadbeam_electron_density_threshold);
                    distance=&uncached_distance;
                }
            }
            if(config.broadbeam_electron_region_metric=="euclidean") {
                electron_region.resize(distance->size());
                for(std::size_t i=0;i<distance->size();++i)electron_region[i]=(*distance)[i]<=config.broadbeam_electron_halo_mm+1e-6;
            } else if(context) {
                // The cached grid is reset with its geometry key; reuse the
                // identical box dilation across batches.
                auto& cached=context->impl_->electron_ct;
                char key[96];std::snprintf(key,sizeof key,"%a:%a",config.broadbeam_electron_density_threshold,config.broadbeam_electron_halo_mm);
                if(cached.electron_region_key!=key){
                    cached.electron_region_box=broadbeam_electron_region(grid.density_g_per_cm3,{grid.nx,grid.ny,grid.nz},
                        {grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm},
                        config.broadbeam_electron_density_threshold,config.broadbeam_electron_halo_mm);
                    cached.electron_region_key=key;
                }
                electron_region=cached.electron_region_box;
            } else
            electron_region=broadbeam_electron_region(grid.density_g_per_cm3,{grid.nx,grid.ny,grid.nz},
                {grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm},
                config.broadbeam_electron_density_threshold,config.broadbeam_electron_halo_mm);
            std::vector<std::uint8_t> uncached_body;
            std::vector<float> uncached_body_distance;
            const std::vector<std::uint8_t>* body=nullptr;
            if(config.broadbeam_electron_external_air_local ||
               (config.broadbeam_electron_region_diagnostics && !config.broadbeam_electron_body_mask_file.empty())) {
                const auto key=input_cache_key(config.broadbeam_electron_body_mask_file)+geometry_key;
                auto& mask=context?context->impl_->electron_ct.electron_body_mask:uncached_body;
                auto& dist=context?context->impl_->electron_ct.electron_body_distance_mm:uncached_body_distance;
                if(!context || context->impl_->electron_ct.electron_body_key!=key) {
                    mask=load_broadbeam_body_mask(config.broadbeam_electron_body_mask_file,
                        {grid.nx,grid.ny,grid.nz},{grid.origin_x_mm,grid.origin_y_mm,grid.origin_z_mm},{grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm});
                    dist.clear();if(context)context->impl_->electron_ct.electron_body_key=key;
                }
                body=&mask;
                if(config.broadbeam_electron_external_air_local && config.broadbeam_electron_external_air_margin_mm>0 && dist.empty()) {
                    std::vector<float> seed(mask.size());
                    for(std::size_t i=0;i<mask.size();++i)seed[i]=mask[i]?.01f:1.f;
                    dist=broadbeam_electron_box_distance(seed,{grid.nx,grid.ny,grid.nz},{grid.spacing_x_mm,grid.spacing_y_mm,grid.spacing_z_mm},.1);
                }
                if(config.broadbeam_electron_external_air_local) {
                std::size_t localized=0;
                for(std::size_t i=0;i<mask.size();++i)if(!mask[i] && grid.density_g_per_cm3[i]<=config.broadbeam_electron_density_threshold &&
                   (config.broadbeam_electron_external_air_margin_mm==0 || dist[i]>config.broadbeam_electron_external_air_margin_mm+1e-6)) {
                    localized+=electron_region[i];electron_region[i]=0;
                }
                std::cout<<"[broadbeam-external-air] body_mask=pinned margin_mm="<<config.broadbeam_electron_external_air_margin_mm<<" localized_voxels="<<localized<<'\n';
                }
            }
            broadbeam_electron_mask=mem_tracker.allocate<std::uint8_t>(electron_region.size());
            if(!broadbeam_electron_mask)throw std::bad_alloc();
            queue.copy(electron_region.data(),broadbeam_electron_mask,electron_region.size()).wait_and_throw();
            std::cout<<"[broadbeam-electron-region] density_threshold="<<config.broadbeam_electron_density_threshold
                     <<" metric="<<config.broadbeam_electron_region_metric
                     <<" halo_mm="<<config.broadbeam_electron_halo_mm<<" selected_voxels="
                     <<std::count(electron_region.begin(),electron_region.end(),std::uint8_t{1})
                     <<" total_voxels="<<electron_region.size()<<" continuation=unrestricted\n";
            if(config.broadbeam_electron_region_diagnostics) {
                auto regions=broadbeam_electron_birth_regions(grid.density_g_per_cm3,{grid.nx,grid.ny,grid.nz},
                    config.broadbeam_electron_density_threshold,*distance);
                if(body)for(std::size_t i=0;i<regions.size();++i)if(grid.density_g_per_cm3[i]<=config.broadbeam_electron_density_threshold)regions[i]=(*body)[i]?1:0;
                std::cout<<"[electron-birth-region-definition] low_density="<<(body?"anatomical_BODY":"six_connected_CT_boundary_proxy")<<'\n';
                broadbeam_birth_regions=mem_tracker.allocate<std::uint8_t>(regions.size());
                broadbeam_region_work=mem_tracker.allocate<std::uint64_t>(7*12);
                if(!broadbeam_birth_regions||!broadbeam_region_work)throw std::bad_alloc();
                queue.copy(regions.data(),broadbeam_birth_regions,regions.size());queue.fill(broadbeam_region_work,std::uint64_t{0},7*12).wait_and_throw();
                for(unsigned r=0;r<6;++r) {
                    std::size_t selected=0,total=0;double mass=0;
                    for(std::size_t i=0;i<regions.size();++i)if(regions[i]==r){++total;if(electron_region[i]){++selected;mass+=grid.density_g_per_cm3[i]*grid.spacing_x_mm*grid.spacing_y_mm*grid.spacing_z_mm/1000.;}}
                    std::cout<<"[electron-birth-region-geometry] region="<<r<<" voxels="<<total<<" selected="<<selected<<" selected_mass_g="<<mass<<'\n';
                }
            }
        }
        if(use_material_electron) {
            if(grid.file_version<CtGrid::version_v2)
                throw std::invalid_argument("Material electron transport requires Schneider section IDs");
            if(enable_ct_grid && (grid.nx!=config.voxel_bins_x || grid.ny!=config.voxel_bins_y || grid.nz!=config.voxel_bins_z ||
               std::abs(grid.spacing_x_mm-config.voxel_size_x_mm)>1e-6 ||
               std::abs(grid.spacing_y_mm-config.voxel_size_y_mm)>1e-6 ||
               std::abs(grid.spacing_z_mm-config.voxel_size_z_mm)>1e-6 || std::abs(grid.origin_z_mm)>1e-6))
                throw std::invalid_argument("Material CT packet scorer must align with CT grid, z origin zero");


                    const auto key=input_cache_key(config.material_electron_birth_file)+
                        input_cache_key(config.material_electron_joint_pilot_file)+
                        electron_bank_input_key(config.material_electron_charged_index_file,"MAIECB02")+
                        electron_bank_input_key(config.material_photon_response_index_file,"MAIPHB02")+
                        ":Z"+std::to_string(config.primary_atomic_number)+":A"+std::to_string(config.primary_mass_number);
                    const bool hit=retain_electron_ct&&context->impl_->electron_ct.fp32&&context->impl_->electron_ct.fp32_key==key;
                    if(hit)independent_float_bank=context->impl_->electron_ct.fp32;
                    else independent_float_bank=std::make_shared<IndependentElectronFloatDevice>(queue,
                        config.material_electron_birth_file,
                        config.material_electron_joint_pilot_file,
                        config.material_electron_charged_index_file,
                        config.material_photon_response_index_file,
                        config.primary_atomic_number,config.primary_mass_number);
                    if(retain_electron_ct){context->impl_->electron_ct.fp32=independent_float_bank;context->impl_->electron_ct.fp32_key=key;}
                    float_birth=independent_float_bank->birth;float_joint=independent_float_bank->joint;
                    float_charged=independent_float_bank->charged;float_photon=independent_float_bank->photon;
                    {std::array<float,256> lows,highs;lows.fill(INFINITY);highs.fill(0);
                    for(const auto& node:independent_float_bank->charged_nodes)if(node.section>=0&&node.section<256){
                        lows[node.section]=std::min(lows[node.section],node.density);highs[node.section]=std::max(highs[node.section],node.density);}
                    for(std::size_t i=0;i<grid.material_id.size();++i){const int section=grid.material_id[i];const float rho=grid.density_g_per_cm3[i];
                        if(!std::isfinite(rho)||rho<lows[section]||rho>highs[section])throw std::runtime_error("FP32 charged bank misses CT material/density");}}
                    if constexpr(BroadBeam && JointResponse==3) {
                        std::size_t misses=0;int first_section=-1;float first_density=0;
                        std::array<float,256> lows,highs;lows.fill(INFINITY);highs.fill(0);
                        for(const auto& node:independent_float_bank->birth_materials)if(node.first>=0&&node.first<256){
                            lows[node.first]=std::min(lows[node.first],node.second);highs[node.first]=std::max(highs[node.first],node.second);}
                        for(std::size_t i=0;i<electron_region.size();++i)if(electron_region[i]) {
                            const int section=grid.material_id[i];const float rho=grid.density_g_per_cm3[i];
                            if(rho<lows[section]||rho>highs[section]){if(!misses){first_section=section;first_density=rho;}++misses;}
                        }
                        if(misses)throw std::runtime_error("Broadbeam selected region misses projectile birth coverage: voxels="+
                            std::to_string(misses)+" first_section="+std::to_string(first_section)+" density="+std::to_string(first_density));
                    }
                    independent_charged_work=mem_tracker.allocate<std::uint64_t>(8);
                    independent_photon_work=mem_tracker.allocate<std::uint64_t>(8);
                    joint_pilot_work=mem_tracker.allocate<std::uint64_t>(9);
                    if(!independent_charged_work||!independent_photon_work||!joint_pilot_work)throw std::bad_alloc();
                    queue.fill(independent_charged_work,std::uint64_t{0},8);queue.fill(independent_photon_work,std::uint64_t{0},8);queue.fill(joint_pilot_work,std::uint64_t{0},9).wait_and_throw();
                    std::cout<<"[electron-arithmetic] precision=4 transport=FP32 geometry=FP32 CDF=uint64 dose=FP32\n";
                    std::cout<<"[independent-fp32-bank] cache="<<(hit?"hit":"miss")<<" bytes="<<independent_float_bank->bytes
                        <<" mapped_bytes="<<independent_float_bank->mapped_bytes<<" source_bytes="<<independent_float_bank->source_bytes<<" legacy_loaded=0 legacy_fallback=0\n";


            electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
            material_untracked_device=mem_tracker.allocate<std::uint64_t>(3);
            material_failure_device=mem_tracker.allocate<MaterialPacketFailure>(1);
            if(!material_failure_device)throw std::bad_alloc();

            if(!electron_joint_diag_device || !material_untracked_device)throw std::bad_alloc();
            queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
            queue.fill(material_untracked_device,std::uint64_t{0},3).wait_and_throw();
        }
        ct_origin_x = grid.origin_x_mm;
        ct_origin_y = grid.origin_y_mm;
        ct_origin_z = grid.origin_z_mm;
        ct_spacing_x = grid.spacing_x_mm;
        ct_spacing_y = grid.spacing_y_mm;
        ct_spacing_z = grid.spacing_z_mm;
        ct_nx = grid.nx;
        ct_ny = grid.ny;
        ct_nz = grid.nz;
        ct_material_ids_are_schneider_sections =
            enable_ct_grid && grid.file_version >= CtGrid::version_v2;

        const auto voxel_count = static_cast<std::size_t>(ct_nx) * ct_ny * ct_nz;
        if(context) {
            auto& cached=context->impl_->electron_ct;
            if(!cached.density || !cached.material) {
                if(cached.density) {sycl::free(cached.density,queue);cached.density=nullptr;}
                if(cached.material) {sycl::free(cached.material,queue);cached.material=nullptr;}
                cached.density=sycl::malloc_device<float>(voxel_count,queue);
                cached.material=sycl::malloc_device<std::uint8_t>(voxel_count,queue);
                if(!cached.density || !cached.material)throw std::bad_alloc();
                queue.copy(grid.density_g_per_cm3.data(),cached.density,voxel_count);
                queue.copy(grid.material_id.data(),cached.material,voxel_count).wait_and_throw();
            }
            ct_density_device=cached.density;ct_material_device=cached.material;
            retained_ct_arrays=true;
        } else {
        ct_density_device = mem_tracker.allocate<float>(voxel_count);
        ct_material_device = mem_tracker.allocate<std::uint8_t>(voxel_count);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();
        }



        use_ct_material_xs = enable_ct_grid && (!config.ct_bone_cross_section_file.empty() ||
                             !config.ct_schneider_cross_section_file.empty());

        if(enable_ct_grid) use_schneider_primary_xs =
            ((ct_material_ids_are_schneider_sections && grid.mass_sp_za_rel.size() == 25) ||
             !config.ct_schneider_file.empty() ||
             !config.ct_schneider_cross_section_file.empty() ||
             config.is_schneider_ct_mode()) &&
            config.nuclear_model != "none";

        if (enable_ct_grid && use_schneider_primary_xs) {
            if (config.ct_schneider_physics_bundle_file.empty() &&
                (config.primary_atomic_number != 6 || config.primary_mass_number != 12)) {
                throw std::runtime_error(
                    "Schneider primary cross section is validated for C12 (Z=6, A=12) primaries only, got Z=" +
                    std::to_string(config.primary_atomic_number) + ", A=" + std::to_string(config.primary_mass_number));
            }

            // Validate all voxel material_id < 25 before kernel launch
            for (std::size_t i = 0; i < grid.material_id.size(); ++i) {
                if (grid.material_id[i] >= SchneiderResampledCrossSectionGrid::kExpectedSections) {
                    throw std::runtime_error(
                        "Invalid Schneider material_id " +
                        std::to_string(static_cast<unsigned>(grid.material_id[i])) +
                        " at voxel " + std::to_string(i) + " (must be < 25)");
                }
            }

            // v3 primary rate: the hazard comes from the masked rate-binary
            // partials (single source with the target sampler), so no CSV XS
            // table is loaded. The CSV key MUST be empty for v3 (fail-fast on
            // v1/v2.1 mixing is enforced at startup); v1 keeps this path.
            const std::filesystem::path v3_primary_rate_probe =
                config.ct_schneider_primary_rate_file;
            const bool primary_rate_is_v3 =
                std::filesystem::exists(v3_primary_rate_probe) &&
                schneider_rate_binary_version(v3_primary_rate_probe) == 3;
            if (primary_rate_is_v3) {
                if (!config.ct_schneider_cross_section_file.empty()) {
                    throw std::runtime_error(
                        "v3 primary rate requires empty ct_schneider_cross_section_file "
                        "(masked-binary hazard; refusing CSV/v3 mixing)");
                }
                if (config.is_primary_attenuation_only_mode()) {
                    throw std::runtime_error(
                        "primary-attenuation-only mode requires the CSV XS table; v3 has none");
                }
                std::cout << "[schneider-primary-xs] mode=primary-c12-masked-binary-v3 (no CSV)\n";
            } else {
                const auto schneider_host_grid = prepare_schneider_primary_xs(config);
                schneider_xs_sections =
                    static_cast<std::uint32_t>(SchneiderResampledCrossSectionGrid::kExpectedSections);
                schneider_xs_energies =
                    static_cast<std::uint32_t>(schneider_host_grid.energy_nodes());
                schneider_xs_e_min = static_cast<float>(schneider_host_grid.transport_energies_MeVu.front());
                const float dE = static_cast<float>(
                    schneider_host_grid.transport_energies_MeVu[1] - schneider_host_grid.transport_energies_MeVu[0]);
                schneider_xs_inv_dE = 1.0F / dE;

                const std::size_t total_elements =
                    static_cast<std::size_t>(schneider_xs_sections) * schneider_xs_energies;
                if (schneider_host_grid.mass_xs_per_mm_at_1g_cm3.size() != total_elements) {
                    throw std::runtime_error("Schneider cross section host payload size mismatch");
                }
                if (schneider_xs_sections != 25) {
                    throw std::runtime_error("schneider_xs_sections must be exactly 25");
                }
                if (schneider_xs_energies != schneider_host_grid.energy_nodes()) {
                    throw std::runtime_error("schneider_xs_energies must match grid size");
                }

                schneider_primary_xs_device = mem_tracker.allocate<float>(total_elements);
                if (schneider_primary_xs_device == nullptr) {
                    throw std::bad_alloc();
                }
                queue.copy(schneider_host_grid.mass_xs_per_mm_at_1g_cm3.data(),
                           schneider_primary_xs_device, total_elements).wait_and_throw();

                // Log table dimensions, byte count, source file and mode once
                std::cout << "[schneider-primary-xs] mode=primary-c12-section-resolved\n"
                          << "  sections=" << schneider_xs_sections << "\n"
                          << "  energies=" << schneider_xs_energies << "\n"
                          << "  bytes=" << total_elements * sizeof(float) << "\n"
                          << "  E_min=" << schneider_xs_e_min << " MeV/u\n"
                          << "  inv_dE=" << schneider_xs_inv_dE << "\n"
                          << "  source=" << config.ct_schneider_cross_section_file << "\n"
                          << "\n";
            }
            if (config.is_primary_attenuation_only_mode()) {
                std::cout << "[ct-validation-mode] primary-attenuation-only\n"
                          << "\n";
            }
        }

    }

    // Spot batching / TPS source
    const auto primary_spot_count = config.primary_spot_batch.size();
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            mem_tracker.allocate<PrimarySpotBatchEntry>(primary_spot_count);
        queue.copy(config.primary_spot_batch.data(), primary_spots_device, primary_spot_count)
            .wait_and_throw();
    }

    const bool use_all_elastic = !config.all_ion_elastic_file.empty();
    AllIonElasticView all_elastic{};
    const int primary_elastic_projectile = all_elastic.projectile(
        config.primary_atomic_number, config.primary_mass_number);
    if (use_all_elastic && primary_elastic_projectile < 0)
        throw std::invalid_argument("All-ion elastic has no record for the YAML primary projectile");
    ElasticRecoilStoppingView recoil_stopping{};
    float* elastic_energy = nullptr; // local, queued charged, overflow
    std::uint32_t* elastic_audit = nullptr; // primary, secondary, bad query, unsupported recoil
    if(use_all_elastic) {
        const auto& stop = inputs.get<ElasticRecoilStoppingTable>(input_cache_key(config.elastic_recoil_stopping_file),
            [&] { return ElasticRecoilStoppingTable::load(config.elastic_recoil_stopping_file); });
        auto* keys=mem_tracker.allocate<std::int32_t>(stop.keys.size());
        auto* se=mem_tracker.allocate<float>(stop.energies.size());auto* sv=mem_tracker.allocate<float>(stop.values.size());
        if(!keys||!se||!sv)throw std::bad_alloc();
        queue.copy(stop.keys.data(),keys,stop.keys.size());queue.copy(stop.energies.data(),se,stop.energies.size());queue.copy(stop.values.data(),sv,stop.values.size()).wait_and_throw();
        recoil_stopping={keys,se,sv,static_cast<std::uint32_t>(stop.keys.size()/2),static_cast<std::uint32_t>(stop.energies.size())};
        const auto& bank = inputs.get<AllIonElasticTable>(input_cache_key(config.all_ion_elastic_file),
            [&] { return AllIonElasticTable::load(config.all_ion_elastic_file); });
        config.require_primary_reference_mass(bank.masses[primary_elastic_projectile], "Elastic package");
        auto* e=mem_tracker.allocate<float>(bank.energies.size());
        auto* r=mem_tracker.allocate<float>(bank.rates.size());
        auto* a=mem_tracker.allocate<ElasticSample>(bank.samples.size());
        auto* m=mem_tracker.allocate<double>(18);
        elastic_audit=mem_tracker.allocate<std::uint32_t>(12);
        elastic_energy=mem_tracker.allocate<float>(3);
        if(!elastic_energy)throw std::bad_alloc();
        queue.fill(elastic_energy,0.0F,3);
        if(!e||!r||!a||!m||!elastic_audit)throw std::bad_alloc();
        queue.copy(bank.energies.data(),e,bank.energies.size());queue.copy(bank.rates.data(),r,bank.rates.size());
        queue.copy(bank.samples.data(),a,bank.samples.size());queue.copy(bank.masses.data(),m,18);
        queue.fill(elastic_audit,std::uint32_t{0},12).wait_and_throw();
        all_elastic={e,r,a,m,static_cast<std::uint32_t>(bank.energies.size()),bank.nq};
    }
#if defined(CARBON_ENABLE_MINIBEAM)
    const bool minibeam_copper_transport = (!BroadBeam && config.enable_minibeam) &&
        config.minibeam_transport_mode == "copper_em";
    float* minibeam_copper_sp_energies_device = nullptr;
    float* minibeam_copper_sp_values_device = nullptr;
    std::uint32_t minibeam_copper_sp_count = 0;
    float* minibeam_copper_loss_e_device = nullptr;
    float* minibeam_copper_loss_r_device = nullptr;
    float* minibeam_copper_loss_d_device = nullptr;
    std::uint32_t minibeam_copper_loss_count = 0;

    float* air_ion_e_device=nullptr;
    float* air_ion_rate_device=nullptr;
    std::uint32_t air_ion_count=0;
    float air_ion_peak=0;
    float* minibeam_air_sp_energies_device = nullptr;
    float* minibeam_air_sp_values_device = nullptr;
    std::uint32_t minibeam_air_sp_count = 0;
    float* minibeam_copper_xs_energies_device = nullptr;
    float* minibeam_copper_xs_values_device = nullptr;
    std::uint32_t minibeam_copper_xs_count = 0;
    CopperElasticRecord* minibeam_copper_elastic_device = nullptr;
    std::uint32_t minibeam_copper_elastic_count = 0;
    std::uint32_t* minibeam_copper_elastic_brackets_device = nullptr;
    std::uint32_t minibeam_copper_elastic_bracket_bins = 0;
    float minibeam_copper_elastic_bracket_first_energy = 0.0F;
    float minibeam_copper_elastic_bracket_inverse_width = 0.0F;
    float* minibeam_copper_ion_sp_ratios_device = nullptr;
    std::uint8_t* minibeam_copper_ion_sp_present_device = nullptr;
    float* minibeam_copper_ion_xs_device = nullptr;
    std::uint8_t* minibeam_copper_ion_xs_present_device = nullptr;
    std::uint32_t minibeam_copper_ion_xs_grid_size = 0;
    float minibeam_copper_ion_xs_minimum_energy = 0.0F;
    float minibeam_copper_ion_xs_inverse_step = 0.0F;
    Cinel03EnergyNode* minibeam_copper_nodes_device = nullptr;
    std::uint32_t* minibeam_copper_offsets_device = nullptr;
    std::uint32_t* minibeam_copper_indices_device = nullptr;
    Cinel03DeviceInteraction* minibeam_copper_interactions_device = nullptr;
    Cinel03DeviceProduct* minibeam_copper_products_device = nullptr;
    std::uint32_t minibeam_copper_node_count = 0;
    std::uint32_t minibeam_copper_event_count = 0;
    std::uint32_t minibeam_copper_product_count = 0;
    // Host-side float-pair uploader shared by the Copper tables and the
    // mode-independent water-Urban loss-range table below.
    const auto upload_float_pair = [&](const std::vector<double>& energies,
                                       const std::vector<double>& values,
                                       float*& out_energies,
                                       float*& out_values) {
        std::vector<float> e(energies.begin(), energies.end());
        std::vector<float> v(values.begin(), values.end());
        out_energies = mem_tracker.allocate<float>(e.size());
        out_values = mem_tracker.allocate<float>(v.size());
        if (!out_energies || !out_values) throw std::bad_alloc();
        queue.copy(e.data(), out_energies, e.size());
        queue.copy(v.data(), out_values, v.size());
    };
    if (!config.minibeam_air_ionization_rate_file.empty()) {
        int z=0,a=0;double density=0,cut=0,peak=0;std::string material;
        std::ifstream meta(config.minibeam_air_ionization_rate_file);
        std::string line;
        while(std::getline(meta,line)) {
            std::istringstream in(line);std::string marker,key;in>>marker>>key;
            if(marker!="#")continue;
            if(key=="projectile_z")in>>z;else if(key=="projectile_a")in>>a;
            else if(key=="material")in>>material;else if(key=="density_g_cm3")in>>density;
            else if(key=="production_cut_mm")in>>cut;else if(key=="peak_energy_MeV")in>>peak;
        }
        if(z!=config.primary_atomic_number || a!=config.primary_mass_number ||
           material!="G4_AIR" || std::abs(density-0.00120479)>1e-10 ||
           std::abs(cut-0.05)>1e-12 || !(peak>0) || !std::isfinite(peak))
            throw std::runtime_error("Air ionization rate metadata mismatch");
        const auto& rates=inputs.get<CrossSectionTable>(input_cache_key(config.minibeam_air_ionization_rate_file),
            [&]{return CrossSectionTable::from_csv(config.minibeam_air_ionization_rate_file);});
        if(rates.energies().front()*a>config.energy_cutoff_MeV ||
           rates.energies().back()*a<config.initial_energy_MeVu*a)
            throw std::runtime_error("Air ionization rate energy domain does not cover source");
        upload_float_pair(rates.energies(),rates.values(),air_ion_e_device,air_ion_rate_device);
        air_ion_count=static_cast<std::uint32_t>(rates.energies().size());air_ion_peak=static_cast<float>(peak);
        std::cout<<"[air-ionization-clock] native proposal rate; aggregate air energy law unchanged\n";
    }
    if (minibeam_copper_transport) {
        const auto& copper_sp = inputs.get<StoppingPowerTable>(input_cache_key(config.minibeam_copper_stopping_power_file),
            [&] { return StoppingPowerTable::from_csv(config.minibeam_copper_stopping_power_file); });
        const auto& air_sp = inputs.get<StoppingPowerTable>(input_cache_key(config.minibeam_air_stopping_power_file),
            [&] { return StoppingPowerTable::from_csv(config.minibeam_air_stopping_power_file); });
        const auto& copper_xs = inputs.get<CrossSectionTable>(input_cache_key(config.minibeam_copper_cross_section_file),
            [&] { return CrossSectionTable::from_csv(config.minibeam_copper_cross_section_file); });
        const auto& elastic = inputs.get<std::vector<CopperElasticRecord>>(input_cache_key(config.minibeam_copper_elastic_file),
            [&] { return load_copper_elastic_events(config.minibeam_copper_elastic_file); });
        std::vector<float> copper_ion_sp_ratios;
        std::vector<std::uint8_t> copper_ion_sp_present;
        std::vector<float> copper_ion_xs;
        std::vector<std::uint8_t> copper_ion_xs_present;
        if (!config.minibeam_copper_ion_stopping_power_file.empty()) {
            const auto& ion_stopping = inputs.get<IonStoppingPowerTables>(
                input_cache_key(config.minibeam_copper_ion_stopping_power_file) + "|base=" +
                input_cache_key(config.minibeam_copper_stopping_power_file), [&] {
                    return IonStoppingPowerTables::from_csv(config.minibeam_copper_ion_stopping_power_file, copper_sp); });
            copper_ion_sp_ratios = ion_stopping.ratios_to_carbon();
            copper_ion_sp_present = ion_stopping.species_present();
        }
        if (!config.minibeam_copper_ion_cross_section_file.empty()) {
            const auto& ion_xs = inputs.get<IonCrossSectionTables>(input_cache_key(config.minibeam_copper_ion_cross_section_file),
            [&] { return IonCrossSectionTables::from_csv(config.minibeam_copper_ion_cross_section_file); });
            copper_ion_xs = ion_xs.values();
            copper_ion_xs_present = ion_xs.species_present();
            minibeam_copper_ion_xs_grid_size =
                static_cast<std::uint32_t>(ion_xs.energy_grid_size());
            minibeam_copper_ion_xs_minimum_energy = ion_xs.minimum_energy_MeVu();
            minibeam_copper_ion_xs_inverse_step = 1.0F / ion_xs.energy_step_MeVu();
        }
        const auto& loaded_cascade = inputs.get<LoadedCinel03>(input_cache_key(config.minibeam_copper_inclxx_file),
            [&] { return take_cinel03(config.minibeam_copper_inclxx_file); });
        const auto& package = *loaded_cascade.package;
        const auto& tables = *loaded_cascade.tables;
        upload_float_pair(copper_sp.energies(), copper_sp.values(),
                          minibeam_copper_sp_energies_device,
                          minibeam_copper_sp_values_device);
        minibeam_copper_sp_count = static_cast<std::uint32_t>(copper_sp.energies().size());
        if (!config.minibeam_copper_loss_range_file.empty()) {
            const auto& loss_table = inputs.get<UrbanLossRangeTable>(input_cache_key(config.minibeam_copper_loss_range_file),
            [&] { return UrbanLossRangeTable::from_csv(config.minibeam_copper_loss_range_file); });
            if (config.minibeam_copper_mcs_model == "urban_v2" &&
                !loss_table.is_active_reference(config.resolved_primary_urban_reference_particle(),
                     config.resolved_primary_urban_ionisation_process(), "G4_Cu",
                     config.minibeam_copper_production_cut_mm)) {
                throw std::runtime_error(
                    "copper urban_v2 requires a lifecycle-validated active "
                    "Urban/loss-process table matching the YAML primary, G4_Cu and production cuts "
                    "couple; the configured table is not a production oracle: " +
                    config.minibeam_copper_loss_range_file.string());
            }
            upload_float_pair(loss_table.energies_total_mev(),
                              loss_table.ranges_mm(),
                              minibeam_copper_loss_e_device,
                              minibeam_copper_loss_r_device);
            std::vector<float> d(loss_table.dedx_values().begin(),
                                 loss_table.dedx_values().end());
            minibeam_copper_loss_d_device = mem_tracker.allocate<float>(d.size());
            if (!minibeam_copper_loss_d_device) throw std::bad_alloc();
            queue.copy(d.data(), minibeam_copper_loss_d_device, d.size());
            minibeam_copper_loss_count =
                static_cast<std::uint32_t>(loss_table.energies_total_mev().size());
            std::cout << "[minibeam-copper] loss-range nodes="
                      << minibeam_copper_loss_count << " max-inverse-residual-MeV="
                      << loss_table.max_inverse_residual_mev() << '\n';
        }
        upload_float_pair(air_sp.energies(), air_sp.values(),
                          minibeam_air_sp_energies_device,
                          minibeam_air_sp_values_device);
        minibeam_air_sp_count = static_cast<std::uint32_t>(air_sp.energies().size());
        upload_float_pair(copper_xs.energies(), copper_xs.values(),
                          minibeam_copper_xs_energies_device,
                          minibeam_copper_xs_values_device);
        minibeam_copper_xs_count = static_cast<std::uint32_t>(copper_xs.energies().size());
        minibeam_copper_elastic_device =
            mem_tracker.allocate<CopperElasticRecord>(elastic.size());
        if (!copper_ion_sp_ratios.empty()) {
            minibeam_copper_ion_sp_ratios_device =
                mem_tracker.allocate<float>(copper_ion_sp_ratios.size());
            minibeam_copper_ion_sp_present_device =
                mem_tracker.allocate<std::uint8_t>(copper_ion_sp_present.size());
        }
        if (!copper_ion_xs.empty()) {
            minibeam_copper_ion_xs_device =
                mem_tracker.allocate<float>(copper_ion_xs.size());
            minibeam_copper_ion_xs_present_device =
                mem_tracker.allocate<std::uint8_t>(copper_ion_xs_present.size());
        }
        minibeam_copper_nodes_device =
            mem_tracker.allocate<Cinel03EnergyNode>(tables.energy_nodes.size());
        minibeam_copper_offsets_device =
            mem_tracker.allocate<std::uint32_t>(tables.event_offsets.size());
        minibeam_copper_indices_device =
            mem_tracker.allocate<std::uint32_t>(tables.event_indices.size());
        minibeam_copper_interactions_device =
            mem_tracker.allocate<Cinel03DeviceInteraction>(tables.interactions.size());
        minibeam_copper_products_device =
            mem_tracker.allocate<Cinel03DeviceProduct>(tables.products.size());
        if (!minibeam_copper_elastic_device ||
            (!copper_ion_sp_ratios.empty() &&
             (!minibeam_copper_ion_sp_ratios_device ||
              !minibeam_copper_ion_sp_present_device)) ||
            (!copper_ion_xs.empty() &&
             (!minibeam_copper_ion_xs_device ||
              !minibeam_copper_ion_xs_present_device)) ||
            !minibeam_copper_nodes_device ||
            !minibeam_copper_offsets_device || !minibeam_copper_indices_device ||
            !minibeam_copper_interactions_device || !minibeam_copper_products_device) {
            throw std::bad_alloc();
        }
        queue.copy(elastic.data(), minibeam_copper_elastic_device, elastic.size());
        if (!copper_ion_sp_ratios.empty()) {
            queue.copy(copper_ion_sp_ratios.data(),
                       minibeam_copper_ion_sp_ratios_device,
                       copper_ion_sp_ratios.size());
            queue.copy(copper_ion_sp_present.data(),
                       minibeam_copper_ion_sp_present_device,
                       copper_ion_sp_present.size());
        }
        if (!copper_ion_xs.empty()) {
            queue.copy(copper_ion_xs.data(), minibeam_copper_ion_xs_device,
                       copper_ion_xs.size());
            queue.copy(copper_ion_xs_present.data(),
                       minibeam_copper_ion_xs_present_device,
                       copper_ion_xs_present.size());
        }
        queue.copy(tables.energy_nodes.data(), minibeam_copper_nodes_device,
                   tables.energy_nodes.size());
        queue.copy(tables.event_offsets.data(), minibeam_copper_offsets_device,
                   tables.event_offsets.size());
        queue.copy(tables.event_indices.data(), minibeam_copper_indices_device,
                   tables.event_indices.size());
        queue.copy(tables.interactions.data(), minibeam_copper_interactions_device,
                   tables.interactions.size());
        queue.copy(tables.products.data(), minibeam_copper_products_device,
                   tables.products.size()).wait_and_throw();
        minibeam_copper_elastic_count = static_cast<std::uint32_t>(elastic.size());
        {
            // Search brackets for the per-step nearest-record rate lookup;
            // each is widened by one record and verified again on device.
            constexpr std::uint32_t bins = 8192;
            const double first = elastic.front().energy_MeV_per_u;
            const double width = (static_cast<double>(elastic.back().energy_MeV_per_u) - first) / bins;
            if (elastic.size() > 2 && width > 0.0) {
                std::vector<float> energies(elastic.size());
                for (std::size_t i = 0; i < elastic.size(); ++i) energies[i] = elastic[i].energy_MeV_per_u;
                std::vector<std::uint32_t> brackets(2 * bins);
                for (std::uint32_t bin = 0; bin < bins; ++bin) {
                    const auto low_energy = static_cast<float>(first + width * bin);
                    const auto high_energy = static_cast<float>(first + width * (bin + 1));
                    auto lower = static_cast<std::size_t>(
                        std::upper_bound(energies.begin(), energies.end(), low_energy) - energies.begin());
                    lower = lower >= 2 ? lower - 2 : 0;
                    auto upper = static_cast<std::size_t>(
                        std::upper_bound(energies.begin(), energies.end(), high_energy) - energies.begin());
                    upper = std::min(upper + 1, energies.size() - 1);
                    brackets[2 * bin] = static_cast<std::uint32_t>(lower);
                    brackets[2 * bin + 1] = static_cast<std::uint32_t>(upper);
                }
                minibeam_copper_elastic_brackets_device = mem_tracker.allocate<std::uint32_t>(brackets.size());
                if (!minibeam_copper_elastic_brackets_device) throw std::bad_alloc();
                queue.copy(brackets.data(), minibeam_copper_elastic_brackets_device, brackets.size()).wait_and_throw();
                minibeam_copper_elastic_bracket_bins = bins;
                minibeam_copper_elastic_bracket_first_energy = static_cast<float>(first);
                minibeam_copper_elastic_bracket_inverse_width = static_cast<float>(1.0 / width);
            }
        }
        minibeam_copper_node_count = static_cast<std::uint32_t>(tables.energy_nodes.size());
        minibeam_copper_event_count = static_cast<std::uint32_t>(tables.interactions.size());
        minibeam_copper_product_count = static_cast<std::uint32_t>(tables.products.size());
        std::cout << "[minibeam-copper] stopping=" << minibeam_copper_sp_count
                  << " inelastic-rate=" << minibeam_copper_xs_count
                  << " elastic-events=" << minibeam_copper_elastic_count
                  << " INCLXX-events/products=" << minibeam_copper_event_count
                  << '/' << minibeam_copper_product_count << '\n';
    }
#endif
#if defined(CARBON_ENABLE_MINIBEAM)
    // Water-Urban loss-range table: independent of the Copper transport mode.
#endif
    RuntimeScope t_sch_ctx("setup_schneider_ctx_host_and_device");
    const SchneiderCtDeviceContext schneider_ct_device_ctx = upload_schneider_ct_device_context(
        queue, mem_tracker, config, schneider_host_cache);
    t_sch_ctx.finish();
    if (context != nullptr && schneider_host_cache != nullptr) {
        std::cout << "[physics-cache] schneider hits=" << schneider_host_cache->hits
                  << " misses=" << schneider_host_cache->misses << "\n";
    }
    if (schneider_ct_device_ctx.uses_cinel03() &&
        !config.is_primary_attenuation_only_mode()) {
        schneider_diag_device = mem_tracker.allocate<std::uint64_t>(kSchneiderDiagSlots);
        schneider_float_device = mem_tracker.allocate<float>(kSchneiderFloatSlots);
        schneider_miss_device =
            mem_tracker.allocate<SchneiderMissRecord>(kSchneiderMissLogCap);
        schneider_track_log_device =
            mem_tracker.allocate<SchneiderUnsupportedTrack>(kSchneiderTrackLogCap);
        schneider_miss_count_device = mem_tracker.allocate<std::uint32_t>(2);
        schneider_track_count_device = mem_tracker.allocate<std::uint32_t>(2);
        if (schneider_diag_device == nullptr || schneider_float_device == nullptr ||
            schneider_miss_device == nullptr || schneider_track_log_device == nullptr ||
            schneider_miss_count_device == nullptr || schneider_track_count_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(schneider_diag_device, std::uint64_t{0}, kSchneiderDiagSlots)
            .wait_and_throw();
        queue.fill(schneider_float_device, 0.0F, kSchneiderFloatSlots).wait_and_throw();
        // [0] = written count, [1] = dropped (over-capacity) count.
        queue.fill(schneider_miss_count_device, std::uint32_t{0}, 2).wait_and_throw();
        queue.fill(schneider_track_count_device, std::uint32_t{0}, 2).wait_and_throw();
    }

    // Scorers & Result buffers
    const auto cinel02_max_secondary_inelastic_generations =
        config.cinel02_max_secondary_inelastic_generations;
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_let_scoring = false;
    const auto voxel_scorer_clamps_transport = config.voxel_scorer_clamps_transport;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    // Resolve the legacy water grid exactly as its allocation/header do.
    // Raw voxel_bins_z/voxel_size_z_mm can both be zero when the grid is
    // defined by phantom_length_mm and depth_bin_width_mm.
    const auto voxel_bins_z = number_of_bins;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.scorer_spacing_z_mm());
    const auto voxel_plane_size = voxel_bins_x * voxel_bins_y;
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    if (enable_ct_grid &&
        std::abs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
        std::abs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
        voxel_min_x_mm = ct_origin_x;
        voxel_min_y_mm = ct_origin_y;
        voxel_max_x_mm =
            ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
        voxel_max_y_mm =
            ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    }

    auto* dose_device = mem_tracker.allocate<DepthAtomicT>(number_of_bins);
    std::uint64_t* primary_survival_device = nullptr;
    std::uint64_t* inelastic_reaction_device = nullptr;
    // Diagnostic-only: fragment-species scoring also needs per-depth
    // survival/reaction counts. No transport effect.
    if (config.needs_primary_survival_buffers()) {
        primary_survival_device =
            mem_tracker.allocate<std::uint64_t>(number_of_bins);
        inelastic_reaction_device =
            mem_tracker.allocate<std::uint64_t>(number_of_bins);
        if (primary_survival_device == nullptr || inelastic_reaction_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(primary_survival_device, std::uint64_t{0}, number_of_bins);
        queue.fill(inelastic_reaction_device, std::uint64_t{0}, number_of_bins);
    }
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? mem_tracker.allocate<DoseAtomicT>(number_of_voxels)
                                  : nullptr;
    const auto enable_primary_voxel_fluence =
        !config.primary_voxel_fluence_mhd_output_file.empty();
    auto* primary_voxel_track_length_device =
        enable_primary_voxel_fluence
            ? mem_tracker.allocate<DoseAtomicT>(number_of_voxels)
            : nullptr;
    const auto enable_charged_origin_voxel_scoring =
        config.enable_charged_origin_voxel_scoring;
    const auto enable_minibeam_component_voxel_scoring =
        (!BroadBeam && config.enable_minibeam) && enable_charged_origin_voxel_scoring;
    const auto enable_minibeam_energy_band_roi_scoring =
        config.enable_minibeam_energy_band_roi_scoring;
    const auto enable_minibeam_primary_roi_scoring =
        config.enable_minibeam_primary_roi_scoring;
    auto* he4_hazard_audit_device = config.fragment_birth_spectrum_output_file.empty()
        ? nullptr : mem_tracker.allocate<double>(6);
    if (!config.fragment_birth_spectrum_output_file.empty()) {
        if (!he4_hazard_audit_device) throw std::bad_alloc();
        queue.memset(he4_hazard_audit_device, 0, 6 * sizeof(double)).wait_and_throw();
    }
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(charged_origin_category_count * number_of_voxels)
            : nullptr;
    auto* minibeam_component_voxel_dose_device =
        enable_minibeam_component_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(
                  minibeam_component_category_count * number_of_voxels)
            : nullptr;
    const auto minibeam_energy_band_roi_value_count =
        minibeam_energy_band_species_count * minibeam_energy_band_count *
        minibeam_fixed_region_count * number_of_bins;
    auto* minibeam_energy_band_roi_dose_device =
        enable_minibeam_energy_band_roi_scoring
            ? mem_tracker.allocate<DoseAtomicT>(
                  minibeam_energy_band_roi_value_count)
            : nullptr;
    const auto minibeam_primary_roi_value_count =
        minibeam_primary_roi_kind_count * minibeam_energy_band_count *
        minibeam_fixed_region_count * number_of_bins;
    auto* minibeam_primary_roi_dose_device =
        enable_minibeam_primary_roi_scoring
            ? mem_tracker.allocate<DoseAtomicT>(minibeam_primary_roi_value_count)
            : nullptr;
    auto* minibeam_spatial_audit_device =
        enable_minibeam_primary_roi_scoring
            ? mem_tracker.allocate<std::uint64_t>(minibeam_spatial_audit_slot_count)
            : nullptr;
    auto* minibeam_fixed_region_by_x_bin_device =
        (enable_minibeam_energy_band_roi_scoring ||
         enable_minibeam_primary_roi_scoring)
            ? mem_tracker.allocate<std::uint8_t>(voxel_bins_x)
            : nullptr;
    if ((enable_minibeam_energy_band_roi_scoring ||
         enable_minibeam_primary_roi_scoring) &&
        minibeam_fixed_region_by_x_bin_device != nullptr) {
        std::vector<std::uint8_t> regions(voxel_bins_x, 255U);
        for (std::size_t ix = 0; ix < voxel_bins_x; ++ix) {
            const auto x =
                -0.5 * static_cast<double>(voxel_bins_x - 1U) *
                    config.voxel_size_x_mm +
                static_cast<double>(ix) * config.voxel_size_x_mm;
            if (std::abs(x) > 18.0) continue;
            auto folded = std::fmod(x + 1.8, 3.6);
            if (folded < 0.0) folded += 3.6;
            const auto absolute = std::abs(folded - 1.8);
            regions[ix] = absolute < 0.25 ? 0U :
                (absolute < 0.9 ? 1U : 2U);
        }
        queue.copy(regions.data(), minibeam_fixed_region_by_x_bin_device,
                   voxel_bins_x).wait_and_throw();
    }
    auto* be_isotope_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(be_isotope_origin_category_count * number_of_voxels)
            : nullptr;
    auto* he_isotope_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(he_isotope_origin_category_count * number_of_voxels)
            : nullptr;
    auto* let_moments_device = enable_let_scoring
                                   ? mem_tracker.allocate<LetAtomicT>(4 * number_of_bins)
                                   : nullptr;
    auto* voxel_let_moments_device =
        enable_let_scoring && enable_voxel_scoring
            ? mem_tracker.allocate<LetAtomicT>(4 * number_of_voxels)
            : nullptr;

    const bool is_primary_attenuation_only = config.is_primary_attenuation_only_mode();
    const auto enable_inelastic = config.enable_inelastic;
    if (enable_inelastic && !use_schneider_primary_xs) {
        throw std::runtime_error("Nuclear transport requires the validated Schneider/unified-water CINEL03 path");
    }
    const auto enable_secondary_transport = config.enable_secondary_transport;
    auto* deposited_device = mem_tracker.allocate<float>(number_of_histories);
    auto* sampled_incident_device = mem_tracker.allocate<float>(number_of_histories);
    auto* escaped_device = mem_tracker.allocate<float>(number_of_histories);
    auto* steps_device = mem_tracker.allocate<std::uint32_t>(number_of_histories);
    auto* untracked_nuclear_device = mem_tracker.allocate<float>(number_of_histories);
    auto* other_terminal_energy_device = mem_tracker.allocate<float>(number_of_histories);
    auto* cutoff_stopped_energy_device = mem_tracker.allocate<float>(number_of_histories);
#if defined(CARBON_ENABLE_MINIBEAM)
    const bool phase_at_copper_exit = config.minibeam_phase_space_at_copper_exit;
    const std::string copper_capture_path = std::getenv("CARBON_MINIBEAM_COPPER_CAPTURE")
        ? std::getenv("CARBON_MINIBEAM_COPPER_CAPTURE") : "";
    const std::string copper_diagnostic_path = !copper_capture_path.empty() ? copper_capture_path+"/input.csv" : std::getenv("CARBON_MINIBEAM_COPPER_DIAGNOSTIC")
        ? std::getenv("CARBON_MINIBEAM_COPPER_DIAGNOSTIC") : "";
    const std::string copper_history_mask_path = std::getenv("CARBON_MINIBEAM_DIAGNOSTIC_HISTORY_MASK")
        ? std::getenv("CARBON_MINIBEAM_DIAGNOSTIC_HISTORY_MASK") : "";
    if ((!copper_diagnostic_path.empty() || !copper_history_mask_path.empty()) &&
        (BroadBeam || config.run_mode != RunMode::research || !config.enable_minibeam ||
         config.enable_ct_grid || config.enable_tps_source || config.enable_tps_coordinate_system ||
         config.primary_atomic_number != 1 || config.primary_mass_number != 1 ||
         config.multiple_scattering_model != "urban_v2" ||
         config.minibeam_transport_mode != "copper_em"))
        throw std::invalid_argument("Copper history diagnostics require research proton homogeneous-water Urban minibeam without TPS/replay");
    const auto copper_history_mask_host = read_copper_history_mask(
        copper_history_mask_path, number_of_histories, config.random_seed);
    auto* copper_history_mask_device = copper_history_mask_host.empty() ? nullptr
        : mem_tracker.allocate<std::uint8_t>(number_of_histories);
    auto* copper_diagnostic_device = copper_diagnostic_path.empty() ? nullptr
        : mem_tracker.allocate<CopperHistoryDiagnostic>(number_of_histories);
    if ((!copper_history_mask_host.empty() && !copper_history_mask_device) ||
        (!copper_diagnostic_path.empty() && !copper_diagnostic_device)) throw std::bad_alloc();
    if (copper_history_mask_device) {
        queue.copy(copper_history_mask_host.data(),copper_history_mask_device,number_of_histories).wait_and_throw();
        std::cout << "[copper-diagnostic-mask] selected="
                  << std::count(copper_history_mask_host.begin(),copper_history_mask_host.end(),std::uint8_t{1})
                  << " of=" << number_of_histories << " source_histories; subset dose is normalized to the full source count; path="
                  << copper_history_mask_path << '\n';
    }
    if (copper_diagnostic_device)
        queue.memset(copper_diagnostic_device,0,number_of_histories*sizeof(CopperHistoryDiagnostic));
    auto* copper_capture_entry_device=copper_capture_path.empty() ? nullptr
        : mem_tracker.allocate<PrimaryMediumEntryState>(number_of_histories);
    if(!copper_capture_path.empty() && !copper_capture_entry_device)throw std::bad_alloc();
    const auto enable_minibeam_phase_space =
        (!BroadBeam && config.enable_minibeam) &&
        !config.minibeam_phase_space_output_file.empty();
    auto* minibeam_phase_space_device = enable_minibeam_phase_space
        ? mem_tracker.allocate<MinibeamPhaseSpaceRecord>(number_of_histories)
        : nullptr;
    const auto minibeam_water_primary_plane_count =
        config.minibeam_water_primary_plane_depths_mm.size();
    const auto enable_minibeam_water_primary_planes =
        (!BroadBeam && config.enable_minibeam) &&
        !config.minibeam_water_primary_plane_output_file.empty() &&
        minibeam_water_primary_plane_count != 0;
    auto* minibeam_water_primary_plane_depths_device =
        enable_minibeam_water_primary_planes
            ? mem_tracker.allocate<float>(minibeam_water_primary_plane_count)
            : nullptr;
    auto* minibeam_water_primary_plane_records_device =
        enable_minibeam_water_primary_planes
            ? mem_tracker.allocate<MinibeamWaterPrimaryPlaneRecord>(
                  number_of_histories * minibeam_water_primary_plane_count)
            : nullptr;
    auto* beamline_removed_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_primary_survivor_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_air_loss_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* minibeam_event_counts_device = (!BroadBeam && config.enable_minibeam)
        ? mem_tracker.allocate<std::uint64_t>(minibeam_event_counter_count)
        : nullptr;
    const bool enable_minibeam_fragment_miss_joint =
        (!BroadBeam && config.enable_minibeam) &&
        config.minibeam_copper_fragment_cascade_generations > 0;
    constexpr auto minibeam_fragment_miss_joint_cell_count =
        MinibeamDiagnostics::fragment_miss_joint_cell_count;
    auto* minibeam_fragment_miss_joint_counts_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_energy_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_depth_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
    auto* minibeam_fragment_miss_joint_remaining_device =
        enable_minibeam_fragment_miss_joint
            ? mem_tracker.allocate<std::uint64_t>(
                  minibeam_fragment_miss_joint_cell_count)
            : nullptr;
#endif
    auto* schneider_inelastic_device =
        use_schneider_primary_xs ? mem_tracker.allocate<std::uint64_t>(1) : nullptr;
    auto* primary_terminal_counts_device = mem_tracker.allocate<std::uint64_t>(4);

    if (deposited_device == nullptr || sampled_incident_device == nullptr || escaped_device == nullptr || steps_device == nullptr ||
        untracked_nuclear_device == nullptr || other_terminal_energy_device == nullptr ||
        cutoff_stopped_energy_device == nullptr || primary_terminal_counts_device == nullptr ||
#if defined(CARBON_ENABLE_MINIBEAM)
        ((!BroadBeam && config.enable_minibeam) && (beamline_removed_device == nullptr ||
         beamline_primary_survivor_device == nullptr ||
         beamline_air_loss_device == nullptr ||
         minibeam_event_counts_device == nullptr ||
         (enable_minibeam_fragment_miss_joint &&
          (minibeam_fragment_miss_joint_counts_device == nullptr ||
           minibeam_fragment_miss_joint_energy_device == nullptr ||
           minibeam_fragment_miss_joint_depth_device == nullptr ||
           minibeam_fragment_miss_joint_remaining_device == nullptr)) ||
         (enable_minibeam_phase_space &&
          minibeam_phase_space_device == nullptr) ||
         (enable_minibeam_water_primary_planes &&
          (minibeam_water_primary_plane_depths_device == nullptr ||
           minibeam_water_primary_plane_records_device == nullptr)))) ||
#endif
        (use_schneider_primary_xs && schneider_inelastic_device == nullptr)) {
        free_device(deposited_device);
        free_device(sampled_incident_device);
        free_device(escaped_device);
        free_device(steps_device);
        free_device(untracked_nuclear_device);
        free_device(other_terminal_energy_device);
        free_device(cutoff_stopped_energy_device);
        free_device(schneider_inelastic_device);
        free_device(primary_terminal_counts_device);
#if defined(CARBON_ENABLE_MINIBEAM)
        free_device(beamline_removed_device);
        free_device(beamline_primary_survivor_device);
        free_device(beamline_air_loss_device);
    free_device(minibeam_event_counts_device);

        free_device(minibeam_fragment_miss_joint_counts_device);
        free_device(minibeam_fragment_miss_joint_energy_device);
        free_device(minibeam_fragment_miss_joint_depth_device);
        free_device(minibeam_fragment_miss_joint_remaining_device);
        free_device(minibeam_phase_space_device);
        free_device(minibeam_water_primary_plane_depths_device);
        free_device(minibeam_water_primary_plane_records_device);
#endif
        throw std::bad_alloc();
    }

    queue.fill(untracked_nuclear_device, 0.0F, number_of_histories);
    queue.fill(other_terminal_energy_device, 0.0F, number_of_histories);
    queue.fill(cutoff_stopped_energy_device, 0.0F, number_of_histories);
#if defined(CARBON_ENABLE_MINIBEAM)
    if (beamline_removed_device != nullptr) {
        queue.fill(beamline_removed_device, 0.0F, number_of_histories);
        queue.fill(beamline_primary_survivor_device, 0.0F, number_of_histories);
        queue.fill(beamline_air_loss_device, 0.0F, number_of_histories);
        queue.fill(minibeam_event_counts_device, std::uint64_t{0},
                   minibeam_event_counter_count);


        if (minibeam_fragment_miss_joint_counts_device != nullptr) {
            queue.fill(minibeam_fragment_miss_joint_counts_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_energy_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_depth_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
            queue.fill(minibeam_fragment_miss_joint_remaining_device,
                       std::uint64_t{0},
                       minibeam_fragment_miss_joint_cell_count);
        }
        if (minibeam_phase_space_device != nullptr) {
            queue.memset(minibeam_phase_space_device, 0,
                         number_of_histories *
                             sizeof(MinibeamPhaseSpaceRecord));
        }
        if (minibeam_water_primary_plane_records_device != nullptr) {
            std::vector<float> plane_depths(
                minibeam_water_primary_plane_count);
            for (std::size_t plane = 0;
                 plane < minibeam_water_primary_plane_count; ++plane) {
                plane_depths[plane] = static_cast<float>(
                    config.minibeam_water_primary_plane_depths_mm[plane]);
            }
            queue.copy(
                plane_depths.data(),
                minibeam_water_primary_plane_depths_device,
                minibeam_water_primary_plane_count).wait();
            queue.memset(
                minibeam_water_primary_plane_records_device, 0,
                number_of_histories * minibeam_water_primary_plane_count *
                    sizeof(MinibeamWaterPrimaryPlaneRecord));
        }
    }
#endif
    if (schneider_inelastic_device != nullptr) {
        queue.fill(schneider_inelastic_device, std::uint64_t{0}, 1);
    }
    queue.fill(primary_terminal_counts_device, std::uint64_t{0}, 4).wait_and_throw();

    const bool record_first_interactions =
        is_primary_attenuation_only || config.validation_scorers();
    PrimaryFirstInteractionRecord* first_interactions_device = nullptr;
    std::uint32_t* first_interactions_count_device = nullptr;
    if (record_first_interactions) {
        first_interactions_device = mem_tracker.allocate<PrimaryFirstInteractionRecord>(number_of_histories);
        first_interactions_count_device = mem_tracker.allocate<std::uint32_t>(1);
        if (first_interactions_device == nullptr || first_interactions_count_device == nullptr) {
            free_device(primary_terminal_counts_device);
            free_device(deposited_device);
            free_device(sampled_incident_device);
            free_device(escaped_device);
            free_device(steps_device);
            free_device(untracked_nuclear_device);
            free_device(other_terminal_energy_device);
            free_device(cutoff_stopped_energy_device);
            free_device(schneider_inelastic_device);
            free_device(first_interactions_device);
            free_device(first_interactions_count_device);
            throw std::bad_alloc();
        }
        queue.fill(first_interactions_count_device, 0U, 1).wait_and_throw();
    }

    const std::size_t max_secondaries = config.secondary_queue_capacity;
    const bool need_secondary_buffers =
        !is_primary_attenuation_only &&
        enable_inelastic;
    auto* secondary_queue_device =
        need_secondary_buffers
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* secondary_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    uint32_t* secondary_overflow_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    float* secondary_overflow_energy_device =
        need_secondary_buffers
            ? mem_tracker.allocate<float>(1)
            : nullptr;
    NeutronBirthBuffer neutron_birth_buffer{};
    if ((!BroadBeam && config.enable_neutron_transport) || !config.neutron_birth_output_file.empty()) {
        neutron_birth_buffer.records = mem_tracker.allocate<NeutronBirthRecord>(config.neutron_birth_capacity);
        neutron_birth_buffer.count = mem_tracker.allocate<std::uint64_t>(1);
        neutron_birth_buffer.capacity = config.neutron_birth_capacity;
        if (!neutron_birth_buffer.records || !neutron_birth_buffer.count)
            throw std::runtime_error("Neutron birth buffer allocation failed");
        queue.fill(neutron_birth_buffer.count, std::uint64_t{0}, 1).wait_and_throw();
    }
    // CT uses the identical upstream Cu/air transport and entry record, but
    // must retain the generic material-aware downstream kernel. Never apply
    // the uniform-water substitutions to patient transport.
    const bool ct_minibeam_split = !BroadBeam && config.enable_minibeam &&
        config.enable_ct_grid && config.minibeam_primary_pipeline != "combined";
    // The shared beamline and generic material-aware stage 4 retain the
    // projectile ZA and total kinetic energy in their runtime state. C12 may
    // use this handoff with its own pinned packages. The proton-only stage 11
    // and Cu-child queue eligibility checks below remain unchanged.
    const bool ct_minibeam_projectile_supported =
        (config.primary_atomic_number == 1 && config.primary_mass_number == 1) ||
        (config.primary_atomic_number == 6 && config.primary_mass_number == 12);
    if(ct_minibeam_split && (config.run_mode!=RunMode::research ||
        config.enable_tps_source || config.enable_tps_coordinate_system ||
        !ct_minibeam_projectile_supported ||
        config.multiple_scattering_model!="urban_v2" || config.minibeam_primary_pipeline!="compact" ||
        config.minibeam_transport_mode!="copper_em" || !config.enable_secondary_transport))
        throw std::invalid_argument("CT split minibeam requires research proton/C12 Urban copper transport without TPS/replay");
#if defined(CARBON_ENABLE_MINIBEAM)
    const bool queue_copper_primary_children = [&] {
        const char* value=std::getenv("CARBON_MINIBEAM_CU_CHARGED_QUEUE");
        const char* specialization_override=std::getenv("CARBON_MINIBEAM_PRIMARY_SPECIALIZE");
        const char* pipeline_override=std::getenv("CARBON_MINIBEAM_PRIMARY_SPLIT");
        if(!value)return !(specialization_override && std::string_view(specialization_override)=="0") &&
            !(pipeline_override && std::string_view(pipeline_override)=="0") &&
            !BroadBeam && config.run_mode==RunMode::research &&
            config.enable_minibeam && (!config.enable_ct_grid || ct_minibeam_split) && !config.enable_tps_source &&
            !config.enable_tps_coordinate_system && config.primary_atomic_number==1 &&
            config.primary_mass_number==1 && config.multiple_scattering_model=="urban_v2" &&
            config.minibeam_transport_mode=="copper_em" && config.enable_secondary_transport &&
            (config.minibeam_primary_specialization || ct_minibeam_split) && config.minibeam_primary_pipeline=="compact";
        if(std::string_view(value)=="0")return false;
        if(std::string_view(value)!="1")throw std::invalid_argument("CARBON_MINIBEAM_CU_CHARGED_QUEUE must be0 or1");
        if(BroadBeam || config.run_mode!=RunMode::research || !config.enable_minibeam || (config.enable_ct_grid && !ct_minibeam_split) ||
            config.enable_tps_source || config.enable_tps_coordinate_system ||
            config.primary_atomic_number!=1 || config.primary_mass_number!=1 ||
            config.multiple_scattering_model!="urban_v2" || !config.enable_secondary_transport ||
            config.minibeam_primary_pipeline=="legacy")
            throw std::invalid_argument("Cu charged queue requires research proton water/CT split/compact Urban minibeam");
        return true;
    }();
    const auto copper_primary_child_capacity=std::min<std::size_t>(max_secondaries,
        std::max<std::size_t>(4096,2*number_of_histories));
    auto* copper_primary_child_tasks=queue_copper_primary_children
        ? mem_tracker.allocate<CopperPrimaryChildTask>(copper_primary_child_capacity) : nullptr;
    auto* copper_primary_child_counts=queue_copper_primary_children
        ? mem_tracker.allocate<std::uint32_t>(3) : nullptr;
    if(queue_copper_primary_children && (!copper_primary_child_tasks || !copper_primary_child_counts))throw std::bad_alloc();
    if(copper_primary_child_counts)queue.fill(copper_primary_child_counts,0U,3).wait_and_throw();
    auto* minibeam_copper_cascade_queue_device =
        need_secondary_buffers && (!BroadBeam && config.enable_minibeam) &&
                config.minibeam_copper_fragment_cascade_generations > 0
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* minibeam_copper_cascade_count_device =
        minibeam_copper_cascade_queue_device
            ? mem_tracker.allocate<std::uint32_t>(1)
            : nullptr;
    const auto enable_minibeam_fragment_phase_space =
        (!BroadBeam && config.enable_minibeam) && need_secondary_buffers &&
        !config.minibeam_fragment_phase_space_output_file.empty();
    auto* minibeam_fragment_phase_space_device =
        enable_minibeam_fragment_phase_space
            ? mem_tracker.allocate<MinibeamFragmentPhaseSpaceRecord>(
                  max_secondaries)
            : nullptr;
    auto* minibeam_fragment_phase_space_count_device =
        enable_minibeam_fragment_phase_space
            ? mem_tracker.allocate<std::uint32_t>(1)
            : nullptr;
#else
    constexpr bool queue_copper_primary_children=false;
#endif
    if (need_secondary_buffers) {
        if (!secondary_queue_device || !secondary_count_device ||
            !secondary_overflow_count_device || !secondary_overflow_energy_device
#if defined(CARBON_ENABLE_MINIBEAM)
            || (enable_minibeam_fragment_phase_space &&
                (!minibeam_fragment_phase_space_device ||
                 !minibeam_fragment_phase_space_count_device))
            || ((!BroadBeam && config.enable_minibeam) &&
                config.minibeam_copper_fragment_cascade_generations > 0 &&
                !minibeam_copper_cascade_count_device)
#endif
            )
            throw std::bad_alloc();
        queue.fill(secondary_count_device, 0U, 1);
        queue.fill(secondary_overflow_count_device, 0U, 1);
        queue.fill(secondary_overflow_energy_device, 0.0F, 1);
#if defined(CARBON_ENABLE_MINIBEAM)
        if (minibeam_copper_cascade_count_device) {
            queue.fill(minibeam_copper_cascade_count_device, 0U, 1);
        }
        if (minibeam_fragment_phase_space_count_device) {
            queue.fill(minibeam_fragment_phase_space_count_device, 0U, 1);
        }
#endif
        queue.wait_and_throw();
    }

    std::uint64_t* primary_loss_query_audit = nullptr;
    if (config.enable_primary_loss_query_audit) {
        primary_loss_query_audit = mem_tracker.allocate<std::uint64_t>(28);
        if (!primary_loss_query_audit) throw std::bad_alloc();
        queue.fill(primary_loss_query_audit, std::uint64_t{0}, 28).wait_and_throw();
    }

    float* generic_recoil_reference_sp_device = nullptr;
    if (use_all_elastic) {
        const auto path = use_unified_water ? config.particle_stopping_power_file
            : std::filesystem::path("data/ion_stopping_power_water_geant4_11_3_2.csv");
        if (use_unified_water)
            (void)IonStoppingPowerTables::from_csv(path, stopping_power);
        const auto reference = load_ion_species_stopping_power_lut(path, table_size, 1.0F);
        // Preserve the existing independent generic-recoil interpolation.
        // Supported ions never query this reference table.
        generic_recoil_reference_sp_device = mem_tracker.allocate<float>(table_size);
        if (!generic_recoil_reference_sp_device) throw std::bad_alloc();
        queue.copy(reference.data(), generic_recoil_reference_sp_device, table_size).wait_and_throw();
    }

    auto* in_fov_dose_device =
        enable_voxel_scoring ? mem_tracker.allocate<DepthAtomicT>(number_of_bins) : nullptr;
    if (in_fov_dose_device != nullptr) {
        queue.fill(in_fov_dose_device, DepthAtomicT{0}, number_of_bins).wait_and_throw();
    }

    if (dose_device == nullptr || deposited_device == nullptr ||
        escaped_device == nullptr || steps_device == nullptr ||
        (enable_voxel_scoring && (voxel_dose_device == nullptr || in_fov_dose_device == nullptr)) ||
        (enable_primary_voxel_fluence &&
         primary_voxel_track_length_device == nullptr) ||
        (enable_charged_origin_voxel_scoring &&
         (charged_origin_voxel_dose_device == nullptr ||
          be_isotope_origin_voxel_dose_device == nullptr ||
          he_isotope_origin_voxel_dose_device == nullptr)) ||
        (enable_minibeam_component_voxel_scoring &&
         minibeam_component_voxel_dose_device == nullptr) ||
        (enable_minibeam_energy_band_roi_scoring &&
         (minibeam_energy_band_roi_dose_device == nullptr ||
          minibeam_fixed_region_by_x_bin_device == nullptr)) ||
        (enable_minibeam_primary_roi_scoring &&
         (minibeam_primary_roi_dose_device == nullptr ||
          minibeam_fixed_region_by_x_bin_device == nullptr ||
          minibeam_spatial_audit_device == nullptr)) ||
        (enable_let_scoring && let_moments_device == nullptr)) {
        throw std::bad_alloc();
    }

    if (!reuse_immutable_buffers) {
        const auto resampled_xs =
            resample_cross_section_grid(cross_section, table_energies);
        queue.copy(resampled_xs.macroscopic_per_mm.data(), cross_section_device,
                   cross_section_table_size);
        if (target_h_fraction_device != nullptr) {
            queue.copy(resampled_xs.target_h_fraction.data(), target_h_fraction_device,
                       cross_section_table_size);
        }
        queue.wait_and_throw();
    } else if (target_h_fraction_device != nullptr) {
        const auto resampled_xs =
            resample_cross_section_grid(cross_section, table_energies);
        queue.copy(resampled_xs.target_h_fraction.data(), target_h_fraction_device,
                   cross_section_table_size).wait_and_throw();
    }

    queue.memset(dose_device, 0, number_of_bins * sizeof(DepthAtomicT));
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_primary_voxel_fluence) {
        queue.memset(primary_voxel_track_length_device, 0,
                     number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
        queue.memset(be_isotope_origin_voxel_dose_device, 0,
                     be_isotope_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
        queue.memset(he_isotope_origin_voxel_dose_device, 0,
                     he_isotope_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
    }
    if (enable_minibeam_component_voxel_scoring) {
        queue.memset(minibeam_component_voxel_dose_device, 0,
                     minibeam_component_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
    }
    if (enable_minibeam_energy_band_roi_scoring) {
        queue.memset(minibeam_energy_band_roi_dose_device, 0,
                     minibeam_energy_band_roi_value_count *
                         sizeof(DoseAtomicT));
    }
    if (enable_minibeam_primary_roi_scoring) {
        queue.memset(minibeam_primary_roi_dose_device, 0,
                     minibeam_primary_roi_value_count *
                         sizeof(DoseAtomicT));
        queue.memset(minibeam_spatial_audit_device, 0,
                     minibeam_spatial_audit_slot_count * sizeof(std::uint64_t));
    }
    if (enable_let_scoring) {
        queue.memset(let_moments_device, 0, 4 * number_of_bins * sizeof(LetAtomicT));
        if (voxel_let_moments_device != nullptr) {
            queue.memset(voxel_let_moments_device, 0, 4 * number_of_voxels * sizeof(LetAtomicT));
        }
    }

    std::size_t local_size = is_cuda_backend ? 128U : (device.is_gpu() ? 256U : 128U);
    // Diagnostic/tuning override for the primary work-group size (non-CUDA
    // backends). Must stay a positive multiple of the subgroup size.
    if (!is_cuda_backend) {
        if (const char* requested = std::getenv("CARBON_PRIMARY_LOCAL_SIZE")) {
            const auto parsed = std::strtoul(requested, nullptr, 10);
            if (parsed > 0) {
                local_size = static_cast<std::size_t>(parsed);
                std::cout << "[primary-local-size] override=" << local_size << "\n";
            }
        }
    }
    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        if (is_cuda_backend) {
            // The dedicated FE kernel needs more simultaneous primary tracks
            // than the detailed reference path for <=1M-history launches.
            // Explicit YAML/environment overrides remain authoritative.
            history_chunk = number_of_histories > 1000000 ? 34816
                : (BroadBeam ? 17408 : 4096);
        } else if (backend == sycl::backend::ext_oneapi_level_zero &&
                   device.is_gpu()) {
            // Arc B580 needs substantially more than one device-wide wave of
            // primary work-groups per launch.  The former 34,816-history CT
            // chunk produced 187 short kernels for RT07575; 1,114,112 reduces
            // this to six launches without the slowdown seen for a single
            // whole-shard kernel.
            history_chunk = 1114112;
        } else {
            history_chunk = device.is_gpu() ? 8192 : number_of_histories;
        }
        std::cout << "[history-chunk-size] backend default=" << history_chunk
                  << "\n";
    }
    if (const char* requested = std::getenv("CARBON_HISTORY_CHUNK_SIZE")) {
        const auto parsed = std::strtoull(requested, nullptr, 10);
        if (parsed == 0U)
            throw std::invalid_argument(
                "CARBON_HISTORY_CHUNK_SIZE must be a positive integer");
        history_chunk = static_cast<std::size_t>(parsed);
        std::cout << "[history-chunk-size] override=" << history_chunk << "\n";
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);
    if (defer_material_packets)history_chunk=std::min<std::size_t>(history_chunk,material_queue_chunk);

    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto inverse_depth_bin_width_mm = 1.0F / depth_bin_width_mm;
    const auto inverse_voxel_size_x_mm = 1.0F / voxel_size_x_mm;
    const auto inverse_voxel_size_y_mm = 1.0F / voxel_size_y_mm;
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_primary_steps = config.maximum_primary_steps;
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);

    if (is_cuda_backend && !(!BroadBeam && config.enable_minibeam)) {
        cuda_clock_warmup(queue, mem_tracker);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_secondary_energy_straggling = config.enable_secondary_energy_straggling;

    const auto multiple_scattering_scale =
        static_cast<float>(config.multiple_scattering_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto active_water_radiation_length = use_unified_water
        ? schneider_ct_device_ctx.water_radiation_length_g_cm2 : water_radiation_length_g_per_cm2;
    const auto enable_multiple_scattering = BroadBeam || config.enable_multiple_scattering;
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    const auto c12_fermi_eyges =
        (BroadBeam || config.multiple_scattering_model == "fermi_eyges");
    const auto fermi_eyges_species_scope = BroadBeam ? 3 :
        config.fermi_eyges_species == "all_charged" ? 3 :
        (config.fermi_eyges_species == "c12_he4_pdt" ? 2 :
         (config.fermi_eyges_species == "c12_he4" ? 1 : 0));
    const auto fermi_eyges_use_species_water_parameters =
        BroadBeam || config.fermi_eyges_parameter_set == "species_water";
    const bool primary_fermi_eyges = c12_fermi_eyges &&
        ion_uses_fermi_eyges(fermi_eyges_species_scope,
                           config.primary_atomic_number, config.primary_mass_number);
    const bool fe_gaussian_only=BroadBeam ? Gaussian : config.fermi_eyges_gaussian_only;
    const bool fe_moments=config.fermi_eyges_sampling=="moments";
    const bool fe_diagnostics=std::getenv("CARBON_FE_DIAGNOSTICS") &&
        std::string_view(std::getenv("CARBON_FE_DIAGNOSTICS"))=="1";
    const bool fe_simple_geometry=!config.enable_hetero_insert && config.slab_layers.empty();
    auto* fe_counts=mem_tracker.allocate<std::uint64_t>(8);
    if (!fe_counts) throw std::bad_alloc();
    if (fe_counts) queue.fill(fe_counts,std::uint64_t{0},8).wait_and_throw();
    const auto c12_fermi_eyges_max_segment_mm =
        static_cast<float>(config.fermi_eyges_max_segment_mm);
    if (c12_fermi_eyges) {
        std::cout << "[ion-mcs] model=fermi_eyges gaussian_only=" << fe_gaussian_only << " species="
                  << config.fermi_eyges_species
                  << " parameters=" << config.fermi_eyges_parameter_set
                  << " unselected_species=highland max_segment_mm="
                  << c12_fermi_eyges_max_segment_mm << '\n';
    }
    const auto enable_tps_source = config.uses_fixed_patient_coordinates();
    const auto random_seed = config.random_seed;
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const bool minibeam_pbs_source = config.spots_geometry_mode == "minibeam_pbs_y";
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime = static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime = static_cast<float>(config.emittance_sigma_y_prime);
    const auto source_origin_x_mm = static_cast<float>(config.source_origin_x_mm);
    const auto source_origin_y_mm = static_cast<float>(config.source_origin_y_mm);
    const auto source_origin_z_mm = static_cast<float>(config.source_origin_z_mm);
    const auto beam_ux_x = static_cast<float>(config.beam_ux_x);
    const auto beam_ux_y = static_cast<float>(config.beam_ux_y);
    const auto beam_ux_z = static_cast<float>(config.beam_ux_z);
    const auto beam_uy_x = static_cast<float>(config.beam_uy_x);
    const auto beam_uy_y = static_cast<float>(config.beam_uy_y);
    const auto beam_uy_z = static_cast<float>(config.beam_uy_z);
    const auto beam_uz_x = static_cast<float>(config.beam_uz_x);
    const auto beam_uz_y = static_cast<float>(config.beam_uz_y);
    const auto beam_uz_z = static_cast<float>(config.beam_uz_z);
#if defined(CARBON_ENABLE_MINIBEAM)
    const auto enable_minibeam = (!BroadBeam && config.enable_minibeam);
    const auto minibeam_angle_rad = static_cast<float>(
        config.minibeam_collimator_angle_deg * 0.01745329251994329577);
    const auto minibeam_cos = std::cos(minibeam_angle_rad);
    const auto minibeam_sin = std::sin(minibeam_angle_rad);
    const auto minibeam_block_radius =
        static_cast<float>(config.minibeam_radius_mm);
    const auto minibeam_block_thickness =
        static_cast<float>(config.minibeam_collimator_thickness_mm);
    const auto minibeam_slit_length =
        static_cast<float>(config.minibeam_slit_length_mm);
    const auto minibeam_slit_thickness =
        static_cast<float>(config.minibeam_slit_thickness_mm);
    const auto minibeam_slit_width = static_cast<float>(config.minibeam_slit_width_mm);
    const auto minibeam_slit_pitch = static_cast<float>(config.minibeam_slit_pitch_mm);
    const auto minibeam_slit_offset = static_cast<float>(config.minibeam_slit_offset_mm);
    const auto minibeam_slit_count = config.minibeam_slit_count;
    const auto minibeam_block_center_z = -static_cast<float>(
        config.minibeam_water_entrance_world_y_mm +
        config.minibeam_collimator_center_to_isocenter_mm);
    const auto minibeam_copper_max_step =
        static_cast<float>(config.minibeam_copper_max_step_mm);
    const auto minibeam_copper_exact_material_boundaries =
        config.minibeam_copper_exact_material_boundaries;
    const auto minibeam_copper_density =
        static_cast<float>(config.minibeam_copper_density_g_per_cm3);
    const auto minibeam_copper_radiation_length =
        static_cast<float>(config.minibeam_copper_radiation_length_g_per_cm2);
    const auto minibeam_copper_mcs_scale =
        static_cast<float>(config.minibeam_copper_mcs_scale);
    const auto minibeam_copper_fragment_mcs_scale =
        static_cast<float>(config.minibeam_copper_fragment_mcs_scale);
    const auto minibeam_copper_fragment_cascade_generations =
        config.minibeam_copper_fragment_cascade_generations;
    const auto minibeam_copper_urban_v2_msc =
        config.minibeam_copper_mcs_model == "urban_v2";
    const auto minibeam_copper_correlated_scattering =
        minibeam_copper_urban_v2_msc;
    const auto minibeam_copper_enable_mcs = config.minibeam_copper_enable_mcs;
    const bool copper_proton_native_fluctuation =
        config.minibeam_copper_proton_straggling_model != "legacy_clipped_gaussian";
    const bool copper_primary_native_mean = config.minibeam_copper_primary_native_mean_loss;
    const bool copper_proton_finite_cut =
        config.minibeam_copper_proton_straggling_model == "g4_universal_delta";
    const auto minibeam_copper_enable_energy_straggling =
        config.minibeam_copper_enable_energy_straggling;
    const auto minibeam_copper_straggling_scale =
        static_cast<float>(config.minibeam_copper_straggling_scale);
    const auto minibeam_copper_fragment_enable_energy_straggling =
        config.minibeam_copper_fragment_enable_energy_straggling;
    const auto minibeam_copper_fragment_straggling_scale =
        static_cast<float>(config.minibeam_copper_fragment_straggling_scale);
    const auto minibeam_copper_enable_elastic =
        config.minibeam_copper_enable_elastic;
    const auto minibeam_copper_enable_inelastic =
        config.minibeam_copper_enable_nuclear_attenuation;
    const auto minibeam_water_primary_stopping_power_scale = static_cast<float>(
        config.minibeam_water_primary_stopping_power_scale);
    const auto minibeam_copper_survivor_energy_loss_scale = static_cast<float>(
        config.minibeam_copper_survivor_energy_loss_scale);
    constexpr std::size_t minibeam_survivor_calibration_capacity = 16;
    std::array<float, minibeam_survivor_calibration_capacity>
        minibeam_survivor_calibration_energies{};
    std::array<float, minibeam_survivor_calibration_capacity>
        minibeam_survivor_calibration_scales{};
    const auto minibeam_survivor_calibration_count =
        config.minibeam_copper_survivor_energy_loss_energies_MeVu.size();
    for (std::size_t index = 0;
         index < minibeam_survivor_calibration_count; ++index) {
        minibeam_survivor_calibration_energies[index] = static_cast<float>(
            config.minibeam_copper_survivor_energy_loss_energies_MeVu[index]);
        minibeam_survivor_calibration_scales[index] = static_cast<float>(
            config.minibeam_copper_survivor_energy_loss_scales[index]);
    }
#endif
    const auto emittance_correlation_x = static_cast<float>(config.emittance_correlation_x);

    const auto emittance_correlation_y = static_cast<float>(config.emittance_correlation_y);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
    const auto primary_mcs_mass_override_MeV =
        static_cast<float>(config.primary_mcs_mass_override_MeV());
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);

#if CARBON_PRIMARY_CONTEXT_POINTER
    // Move the two large immutable closures out of the primary kernel's argument
    // list. Intel GPU backends cap kernel arguments at 2048 B; the by-value
    // captures alone exceed that. The kernel body aliases these pointers with the
    // original names, so the physics code is unchanged.
    static_assert(std::is_trivially_copyable_v<UnifiedEmDevice>);
    static_assert(std::is_trivially_copyable_v<SchneiderCtDeviceContext>);
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
    auto* primary_unified_device_ptr = mem_tracker.allocate<UnifiedEmDevice>(1);
#endif
    auto* primary_schneider_ct_ptr = mem_tracker.allocate<SchneiderCtDeviceContext>(1);
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
    if (primary_unified_device_ptr == nullptr || primary_schneider_ct_ptr == nullptr) {
#else
    if (primary_schneider_ct_ptr == nullptr) {
#endif
        throw std::bad_alloc();
    }
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
    queue.memcpy(primary_unified_device_ptr, &unified_device, sizeof(UnifiedEmDevice));
#endif
    queue.memcpy(primary_schneider_ct_ptr, &schneider_ct_device_ctx,
                 sizeof(SchneiderCtDeviceContext));
    queue.wait_and_throw();
    std::cout << "[primary-context] device context pointer enabled; moved "
              << (sizeof(SchneiderCtDeviceContext)
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
                  + sizeof(UnifiedEmDevice)
#endif
                  )
              << " B of closures out of the kernel argument list\n";
#endif

    std::cout << "[input-cache] hits=" << inputs.hits-input_hits_before
              << " misses=" << inputs.misses-input_misses_before << "\n";
    runtime_setup.finish();
    // Broadbeam CT (shared FP32 electrons): fold the fixed configuration into
    // dedicated primary/secondary kernels. Ineligible runs keep generic stage 6.
    const auto broadbeam_ct_switch=[](const char* name) {
        const char* v=std::getenv(name);
        if(!v)return true;
        const std::string_view s(v);
        if(s!="0"&&s!="1")throw std::invalid_argument(std::string(name)+" must be 0 or 1");
        return s=="1";
    };
    bool broadbeam_ct_primary_specialize=false,broadbeam_ct_secondary_specialize=false;
    if constexpr(BroadBeam && JointResponse==3) {
        const bool primary_eligible = enable_ct_grid && !enable_hetero_insert && !use_unified_water &&
            enable_voxel_scoring && enable_multiple_scattering && enable_inelastic && use_schneider_primary_xs &&
            use_material_electron && use_material_ct && defer_material_packets &&
            !use_ct_elastic &&
             slab_layer_count==0 && !enable_charged_origin_voxel_scoring &&
            !enable_minibeam_component_voxel_scoring && !enable_minibeam_energy_band_roi_scoring &&
            use_electron_fp32 && use_independent_charged && !enable_primary_voxel_fluence && !voxel_scorer_clamps_transport;
        const bool secondary_eligible = enable_ct_grid && enable_voxel_scoring && ct_secondary_exact_faces &&
            ct_skip_homogeneous_face_clamp && !use_unified_water && !enable_charged_origin_voxel_scoring &&
            !enable_minibeam_component_voxel_scoring && !enable_minibeam_energy_band_roi_scoring;
        broadbeam_ct_primary_specialize=primary_eligible && broadbeam_ct_switch("CARBON_BROADBEAM_CT_PRIMARY_SPECIALIZE");
        broadbeam_ct_secondary_specialize=secondary_eligible && broadbeam_ct_switch("CARBON_BROADBEAM_CT_SECONDARY_SPECIALIZE");
        std::cout<<"[broadbeam-ct-specialize] primary_eligible="<<primary_eligible<<" primary="<<broadbeam_ct_primary_specialize
                 <<" secondary_eligible="<<secondary_eligible<<" secondary="<<broadbeam_ct_secondary_specialize<<"\n";
    }
    RuntimeScope runtime_steps("transport_loop_including_scoring_and_queue_transfers");
    double primary_kernel_seconds = 0.0;
#if CARBON_PRIMARY_PRODUCTION_SPECIALIZE
    const bool primary_production_path_eligible =
        enable_ct_grid && enable_voxel_scoring && enable_inelastic &&
        enable_multiple_scattering && enable_ct_material_mcs &&
        use_schneider_primary_xs &&
        !use_unified_water && !voxel_scorer_clamps_transport &&
        primary_loss_query_audit == nullptr &&
        !enable_primary_voxel_fluence && !enable_charged_origin_voxel_scoring &&
        !use_all_elastic;
    bool use_production_primary_path = primary_production_path_eligible;
    if(const char* requested=std::getenv("CARBON_PRIMARY_PATH_DIAGNOSTIC")) {
        const std::string_view mode(requested);
        if(mode=="generic")use_production_primary_path=false;
        else if(mode=="specialized") {
            if(!primary_production_path_eligible)
                throw std::invalid_argument(
                    "CARBON_PRIMARY_PATH_DIAGNOSTIC=specialized requested for an ineligible configuration");
            use_production_primary_path=true;
        } else throw std::invalid_argument(
            "CARBON_PRIMARY_PATH_DIAGNOSTIC must be generic or specialized");
        std::cout<<"[primary-path-diagnostic] same_binary="<<mode<<"\n";
    }
#endif

    // Opt-in homogeneous-water Urban minibeam specialization. Eligibility
    // proves every compile-time substitution below against runtime settings.
    bool use_minibeam_primary_path = config.minibeam_primary_specialization;
    if (const char* requested = std::getenv("CARBON_MINIBEAM_PRIMARY_SPECIALIZE")) {
        const std::string_view value(requested);
        if (value != "0" && value != "1")
            throw std::invalid_argument("CARBON_MINIBEAM_PRIMARY_SPECIALIZE must be 0 or 1");
        use_minibeam_primary_path = value == "1";
    }
    if (use_minibeam_primary_path && !(
        (!BroadBeam && config.enable_minibeam) && global_urban && use_unified_water &&
        !enable_ct_grid && !enable_hetero_insert && slab_layer_count == 0))
        throw std::invalid_argument("Minibeam primary specialization requires uniform-water Urban minibeam");
    std::cout << "[minibeam-primary-specialization] " << use_minibeam_primary_path << "\n";
    std::cout << "[ct-minibeam-split] " << ct_minibeam_split << "\n";

    // Both entry routes share the same compile-time uniform-water physics body.
    // Stage 5 samples the source and transports water, with copper disabled at
    // compile time. No state handoff or active-track partition is needed.
    const bool use_water_primary_path = config.water_primary_specialization;
    if (use_water_primary_path && !(
        !(!BroadBeam && config.enable_minibeam) && global_urban && use_unified_water &&
        !enable_ct_grid && !enable_hetero_insert && slab_layer_count == 0))
        throw std::invalid_argument("Water primary specialization requires non-minibeam uniform-water Urban");
    std::cout << "[water-primary-specialization] " << use_water_primary_path << "\n";
    const bool primary_minibeam_enabled = (!BroadBeam && config.enable_minibeam);
    const bool nonion_primary = [&] {
        if constexpr(!BroadBeam || JointResponse!=0)return false;
        const char* env=std::getenv("CARBON_PROTON_NONION_PRIMARY");
        if(env && std::string_view(env)!="0" && std::string_view(env)!="1")
            throw std::invalid_argument("CARBON_PROTON_NONION_PRIMARY must be 0 or 1");
        if(env && std::string_view(env)=="0")return false;
        // Package models are the authority. Do not infer their behavior from Z/A alone.
        if(config.primary_atomic_number!=1 || config.primary_mass_number!=1)return false;
        return nonion_primary_package_safe;
    }();
    std::cout<<"[nonion-primary] enabled="<<nonion_primary<<"\n";

    const int split_minibeam_primary = [&] {
        const int configured = config.minibeam_primary_pipeline == "compact" ? 2
            : config.minibeam_primary_pipeline == "split" ? 1 : 0;
        const char* requested = std::getenv("CARBON_MINIBEAM_PRIMARY_SPLIT");
        if (!requested) return configured;
        const std::string_view value(requested);
        if (value != "0" && value != "1" && value != "2")
            throw std::invalid_argument("CARBON_MINIBEAM_PRIMARY_SPLIT must be 0, 1 or 2");
        return value == "2" ? 2 : value == "1" ? 1 : 0;
    }();
    if (split_minibeam_primary && !use_minibeam_primary_path && !ct_minibeam_split)
        throw std::invalid_argument("Split primary requires water specialization or CT compact Urban minibeam");
    if(ct_minibeam_split && split_minibeam_primary!=2)
        throw std::invalid_argument("CT minibeam requires compact pipeline (no split environment override)");
#if defined(CARBON_ENABLE_MINIBEAM)
    if(queue_copper_primary_children && !split_minibeam_primary)
        throw std::invalid_argument("Cu charged queue requires a split primary launch");
#endif
    // Exact CT profile: every constant substituted in stage 11 is checked
    // here. The original material-aware stage 4 remains the control/fallback.
    const bool ct_queued_primary_eligible = ct_minibeam_split &&
        split_minibeam_primary == 2 &&
        global_urban == true &&
        enable_ct_grid == true &&
        enable_hetero_insert == false &&
        use_unified_water == false &&
        enable_voxel_scoring == true &&
        enable_multiple_scattering == true &&
        enable_inelastic == true &&
        use_schneider_primary_xs == true &&
        use_material_electron == true &&
        use_material_ct == true &&
        defer_material_packets == true &&

        use_ct_elastic == false &&



        slab_layer_count == 0 &&
        primary_atomic_number == 1 &&
        primary_mass_number == 1 &&
        inverse_mass_number == 1.0F;
    const bool ct_queued_primary_specialize = [&] {
        const char* value = std::getenv("CARBON_CT_QUEUED_PRIMARY_SPECIALIZE");
        if (!value) return ct_queued_primary_eligible;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) != "1")
            throw std::invalid_argument("CARBON_CT_QUEUED_PRIMARY_SPECIALIZE must be 0 or 1");
        if (!ct_queued_primary_eligible)
            throw std::invalid_argument("CT queued primary specialization requires the exact proton Urban CT queue profile");
        return true;
    }();
    std::cout << "[ct-queued-primary-specialize] enabled=" << ct_queued_primary_specialize << "\n";
    auto* primary_medium_entry = split_minibeam_primary
        ? mem_tracker.allocate<PrimaryMediumEntryState>(history_chunk) : nullptr;
    if (split_minibeam_primary && !primary_medium_entry) throw std::bad_alloc();
    auto* primary_entry_counts = split_minibeam_primary == 2
        ? mem_tracker.allocate<std::uint64_t>(2) : nullptr;
    if (split_minibeam_primary == 2 && !primary_entry_counts) throw std::bad_alloc();
#if defined(CARBON_ENABLE_MINIBEAM)
    const unsigned water_energy_buckets = [&] {
        const char* value=std::getenv("CARBON_MINIBEAM_WATER_ENERGY_BUCKETS");
        if(!value)return split_minibeam_primary==2 && (use_minibeam_primary_path || ct_minibeam_split) &&
            config.primary_atomic_number==1 && config.primary_mass_number==1 ? 16U : 0U;
        if(std::string_view(value)=="0")return 0U;
        unsigned buckets=0;
        const std::string_view view(value);
        auto parsed=std::from_chars(view.data(),view.data()+view.size(),buckets);
        if (parsed.ec!=std::errc{} || parsed.ptr!=view.data()+view.size() ||
            (buckets!=4 && buckets!=8 && buckets!=16 && buckets!=32))
            throw std::invalid_argument("CARBON_MINIBEAM_WATER_ENERGY_BUCKETS must be 0,4,8,16,32");
        if (split_minibeam_primary!=2 || (!use_minibeam_primary_path && !ct_minibeam_split))
            throw std::invalid_argument("Entry energy ordering requires compact water/CT minibeam");
        return buckets;
    }();
    auto* water_energy_order = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(history_chunk) : nullptr;
    auto* water_energy_counts = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(water_energy_buckets+1) : nullptr;
    auto* water_energy_offsets = water_energy_buckets ? mem_tracker.allocate<std::uint32_t>(water_energy_buckets+1) : nullptr;
    if(water_energy_buckets && (!water_energy_order || !water_energy_counts || !water_energy_offsets))throw std::bad_alloc();
    std::vector<sycl::event> water_order_events;
    std::cout << "[minibeam-candidates] water_energy_buckets=" << water_energy_buckets << '\n';
    // Beamline lane order: a straight-ray prediction from the sampled flat
    // source groups copper-bound histories (~600 Cu steps) apart from open-slit
    // ones (a few air steps), so fewer warps idle on a single long lane. Only
    // the lane->history map changes; every history keeps its index, RNG
    // stream and output slots. CARBON_MINIBEAM_BEAMLINE_ORDER=0 disables it.
    const bool order_beamline_lanes = [&] {
        const char* value = std::getenv("CARBON_MINIBEAM_BEAMLINE_ORDER");
        if (value && std::string_view(value) != "0" && std::string_view(value) != "1")
            throw std::invalid_argument("CARBON_MINIBEAM_BEAMLINE_ORDER must be 0 or 1");
        return split_minibeam_primary != 0 && enable_minibeam && enable_flat_source &&
            !enable_emittance_source && primary_spot_count == 0 &&
            !(value && std::string_view(value) == "0");
    }();
    auto* beamline_order = order_beamline_lanes
        ? mem_tracker.allocate<std::uint32_t>(history_chunk) : nullptr;
    auto* beamline_order_counts = order_beamline_lanes
        ? mem_tracker.allocate<std::uint32_t>(2) : nullptr;
    if (order_beamline_lanes && (!beamline_order || !beamline_order_counts)) throw std::bad_alloc();
    std::cout << "[beamline-order] enabled=" << order_beamline_lanes << "\n";
#else
    constexpr std::uint32_t* beamline_order = nullptr;
#endif
    std::cout << "[primary-split] enabled=" << split_minibeam_primary
              << " entry_bytes=" << sizeof(PrimaryMediumEntryState) << "\n";
    std::vector<sycl::event> electron_events;
    std::uint64_t electron_tasks_total=0, electron_queue_peak=0;
    std::vector<sycl::event> primary_events;
    primary_events.reserve((split_minibeam_primary ? 2 : 1) *
        ((number_of_histories + history_chunk - 1) / history_chunk));
    for (std::size_t hist_offset = 0;
         hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        if (defer_material_packets)queue.fill(material_task_counts,std::uint64_t{0},2);
        const auto launch_primary_chunk = [&](auto production_path_tag, auto uniform_water_path_tag, auto stage_tag) {
            constexpr int kKernelStage = decltype(stage_tag)::value;
            constexpr bool kQueueCopperChildren = kKernelStage == 8 || kKernelStage == 9 || kKernelStage == 10;
            constexpr bool kCtQueuedPrimaryPath = kKernelStage == 11;
            // Stage 12 is broadbeam stage 6 with host-proven CT-shared-electron
            // configuration folded at compile time (no physics difference).
            constexpr bool kBroadBeamCtPath = kKernelStage == 12 || kKernelStage == 13 || kKernelStage == 14;
            // 13/14 additionally fold the host-checked projectile (proton / C12).
            constexpr int kFoldZ = kKernelStage == 13 ? 1 : kKernelStage == 14 ? 6 : 0;
            constexpr int kFoldA = kKernelStage == 13 ? 1 : kKernelStage == 14 ? 12 : 0;
            static_assert(!kBroadBeamCtPath || (BroadBeam && JointResponse==3));
            constexpr int kPrimaryStage = kCtQueuedPrimaryPath ? 4 : kKernelStage == 8 ? 1 : (kKernelStage == 9 || kKernelStage == 10) ? 3 : kBroadBeamCtPath ? 6 : kKernelStage;
            constexpr bool kUniformWaterPrimaryPath = decltype(uniform_water_path_tag)::value;
            // Host eligibility requires global Urban for every split beamline.
            // Downstream CT material handling must not select upstream physics.
            constexpr bool kUrbanBeamlinePath = kUniformWaterPrimaryPath ||
                kPrimaryStage == 1 || kPrimaryStage == 3;
            constexpr bool kProductionPrimaryPath =
                decltype(production_path_tag)::value;
            const auto ct_runtime_global_urban = global_urban;
            const auto ct_runtime_enable_ct_grid = enable_ct_grid;
            const auto ct_runtime_enable_hetero_insert = enable_hetero_insert;
            const auto ct_runtime_use_unified_water = use_unified_water;
            const auto ct_runtime_enable_voxel_scoring = enable_voxel_scoring;
            const auto ct_runtime_enable_multiple_scattering = enable_multiple_scattering;
            const auto ct_runtime_enable_inelastic = enable_inelastic;
            const auto ct_runtime_use_schneider_primary_xs = use_schneider_primary_xs;
            const auto ct_runtime_use_material_electron = use_material_electron;
            const auto ct_runtime_use_material_ct = use_material_ct;
            const auto ct_runtime_defer_material_packets = defer_material_packets;
            const auto ct_runtime_use_ct_elastic = use_ct_elastic;
            const auto ct_runtime_slab_layer_count = slab_layer_count;
            const auto ct_runtime_primary_atomic_number = primary_atomic_number;
            const auto ct_runtime_primary_mass_number = primary_mass_number;
            const auto ct_runtime_inverse_mass_number = inverse_mass_number;
            const auto ct_runtime_enable_charged_origin_voxel_scoring = enable_charged_origin_voxel_scoring;
            const auto ct_runtime_enable_minibeam_component_voxel_scoring = enable_minibeam_component_voxel_scoring;
            const auto ct_runtime_enable_minibeam_energy_band_roi_scoring = enable_minibeam_energy_band_roi_scoring;
            const auto ct_runtime_use_electron_fp32 = use_electron_fp32;
            const auto ct_runtime_use_independent_charged = use_independent_charged;
            const auto ct_runtime_enable_primary_voxel_fluence = enable_primary_voxel_fluence;
            const auto ct_runtime_voxel_scorer_clamps_transport = voxel_scorer_clamps_transport;
            return queue.parallel_for<CarbonPrimaryTransportKernel<
                kProductionPrimaryPath, kUniformWaterPrimaryPath, kKernelStage, BroadBeam, Gaussian, JointResponse>>(
            sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                // Constexpr-selected values remove unused physics branches;
                // the host eligibility test proves their runtime equivalence.
                const auto global_urban = kCtQueuedPrimaryPath ? true : (BroadBeam ? false : ct_runtime_global_urban);
                const auto enable_ct_grid = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_ct_grid;
                const auto enable_hetero_insert = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? false : ct_runtime_enable_hetero_insert;
                const auto use_unified_water = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? false : ct_runtime_use_unified_water;
                const auto enable_voxel_scoring = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_voxel_scoring;
                const auto enable_multiple_scattering = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_multiple_scattering;
                const auto enable_inelastic = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_enable_inelastic;
                const auto use_schneider_primary_xs = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_use_schneider_primary_xs;
                const auto use_material_electron = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_use_material_electron;
                const auto use_material_ct = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_use_material_ct;
                const auto defer_material_packets = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? true : ct_runtime_defer_material_packets;
                const auto use_ct_elastic = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? false : ct_runtime_use_ct_elastic;
                const auto slab_layer_count = (kCtQueuedPrimaryPath || kBroadBeamCtPath) ? 0 : ct_runtime_slab_layer_count;
                const auto primary_atomic_number = kCtQueuedPrimaryPath ? 1 : (kFoldZ ? kFoldZ : ct_runtime_primary_atomic_number);
                const auto primary_mass_number = kCtQueuedPrimaryPath ? 1 : (kFoldA ? kFoldA : ct_runtime_primary_mass_number);
                const auto inverse_mass_number = kCtQueuedPrimaryPath ? 1.0F : (kFoldA ? 1.0F / static_cast<float>(kFoldA) : ct_runtime_inverse_mass_number);
                const bool enable_minibeam = (kPrimaryStage == 5 || BroadBeam) ? false : primary_minibeam_enabled;
                const auto enable_charged_origin_voxel_scoring = kBroadBeamCtPath ? false : ct_runtime_enable_charged_origin_voxel_scoring;
                const auto enable_minibeam_component_voxel_scoring = kBroadBeamCtPath ? false : ct_runtime_enable_minibeam_component_voxel_scoring;
                const auto enable_minibeam_energy_band_roi_scoring = kBroadBeamCtPath ? false : ct_runtime_enable_minibeam_energy_band_roi_scoring;
                const auto use_electron_fp32 = kBroadBeamCtPath ? true : ct_runtime_use_electron_fp32;
                const auto use_independent_charged = kBroadBeamCtPath ? true : ct_runtime_use_independent_charged;
                const auto enable_primary_voxel_fluence = kBroadBeamCtPath ? false : ct_runtime_enable_primary_voxel_fluence;
                const auto voxel_scorer_clamps_transport = kBroadBeamCtPath ? false : ct_runtime_voxel_scorer_clamps_transport;
                const auto lane = item.get_global_linear_id();
                if (lane >= chunk_count) {
                    return;
                }
                auto global_history = hist_offset + lane;
                if constexpr (kPrimaryStage == 1 || kPrimaryStage == 3) {
                    if (beamline_order != nullptr) global_history = hist_offset + beamline_order[lane];
                }
#if CARBON_PRIMARY_CONTEXT_POINTER
                // Shadow the host-side closures with device-resident views so the
                // rest of the kernel body is byte-for-byte identical.
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
                const auto& unified_device = *primary_unified_device_ptr;
#endif
                const auto& schneider_ct_device_ctx = *primary_schneider_ct_ptr;
#endif

                PrimaryMediumEntryState entry{};
#include "detail/sycl_primary_beamline.inc"
                if constexpr (kPrimaryStage == 1 || kPrimaryStage == 3) {
                    std::size_t destination = lane;
                    if constexpr (kPrimaryStage == 3) {
                        // Partition all histories, never discard a terminated track.
                        // The two ends cannot overlap: active + inactive <= chunk_count.
                        const bool active = entry.energy_MeV > energy_cutoff_MeV;
                        const auto rank = sycl::atomic_ref<std::uint64_t,
                            sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                primary_entry_counts[active ? 0 : 1]).fetch_add(1U);
                        destination = active ? rank : chunk_count - 1 - rank;
                    }
                    primary_medium_entry[destination] = entry;
                } else {
                    if constexpr (kPrimaryStage == 2 || kPrimaryStage == 4) {
#if defined(CARBON_ENABLE_MINIBEAM)
                        entry = primary_medium_entry[water_energy_order ? water_energy_order[lane] : lane];
#else
                        entry = primary_medium_entry[lane];
#endif
                        global_history = entry.global_history;
                    }
                const auto rng_history = entry.rng_history;
                const auto spot_seed = entry.spot_seed;
                const auto spot_initial_energy_MeV = entry.spot_initial_energy_MeV;
                auto energy_MeV = entry.energy_MeV;
                const auto sampled_initial_energy_MeVu = entry.sampled_initial_energy_MeVu;
                auto position_x_mm = entry.position_x_mm;
                auto position_y_mm = entry.position_y_mm;
                auto position_z_mm = entry.position_z_mm;
                auto direction_x = entry.direction_x;
                auto direction_y = entry.direction_y;
                auto direction_z = entry.direction_z;
                auto history_deposited_MeV = 0.0F;
                // Broadbeam: the grid in/out energy ledgers are reduced per
                // history instead of one single-address FP64 atomic per step.
                double grid_in_pending_MeV = 0.0, grid_out_pending_MeV = 0.0;
                auto grid_split = [&](bool in_grid, float value_MeV) {
                    if constexpr (BroadBeam) {
                        if (sycl::isfinite(value_MeV) && value_MeV > 0.0F) {
                            if (in_grid) grid_in_pending_MeV += value_MeV;
                            else grid_out_pending_MeV += value_MeV;
                        }
                    } else grid_deposit_split_device(grid_deposited_in_device, grid_deposited_out_device, in_grid, value_MeV);
                };
                MaterialBirthTask grouped_task{};
                std::uint32_t grouped_count=0;
                double grouped_weight=0;
                // Separate stream: never consume transport or selected replay RNG.
                std::uint64_t group_draw=0;
                auto flush_electron_group=[&]() {
                    if(!grouped_count)return;
                    grouped_task.weight=grouped_weight;
                    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                        sycl::access::address_space::global_space> produced(material_task_counts[0]);
                    const auto slot=produced.fetch_add(1);
                    if(slot<material_task_capacity)material_tasks[slot]=grouped_task;
                    grouped_count=0;grouped_weight=0;
                };

                std::uint64_t local_schneider_rate_queries = 0;
                std::uint32_t steps = 0;
                // Research-only water Urban state (one fMinimal track per
                // primary history; first water segment starts at a boundary).
                UrbanV2TrackState global_urban_state{};
                bool global_urban_boundary=true;
                const int global_urban_ion=(kUniformWaterPrimaryPath || global_urban)?urban_device.projectile(primary_atomic_number,primary_mass_number):-1;
                bool water_urban_escaped_geometry = false;

                double pending_primary_depth_MeV = 0.0;
                double pending_let_numerator = 0.0;
                double pending_let_denominator = 0.0;
                double pending_voxel_let_numerator = 0.0;
                double pending_voxel_let_denominator = 0.0;
                int pending_primary_bin = 0;
                double pending_primary_voxel_MeV = 0.0;
                std::size_t pending_primary_voxel = 0;
                auto last_primary_stopping_power_MeV_per_mm = 0.0F;
                auto last_primary_density_g_per_cm3 = 0.0F;

                float nuclear_tau_remaining = 0.0F;
                HadronicIncreasingCacheCandidate primary_hadronic_cache;
                bool nuclear_tau_active = false;
                std::uint32_t nuclear_tau_rng_step = 0;
                bool primary_inelastic_occurred = false;
                bool unified_primary_escaped_ct = false;
                UnifiedEmClock unified_primary_clock;
                UnifiedEmState unified_primary_state;
                std::array<std::uint64_t,8> unified_primary_audit{};
                std::uint64_t unified_primary_counter=0;
                const int unified_primary_species=unified_device.species_index(primary_atomic_number,primary_mass_number);

                std::uint32_t interaction_section = 0;
                float interaction_density = 0.0F;

                const std::uint32_t max_primary_steps = maximum_primary_steps;
                int last_survival_bin = -1;
                while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    // The rate/target hazard is evaluated from the energy at
                    // the beginning of this step.  The replay event is
                    // queried after the continuous EM loss below, so retain
                    // both values for the compact replay ledger.
                    const auto primary_rate_query_energy_MeV =
                        sycl::fmax(0.0F, energy_MeV);
                    const auto escaped_z =
                        position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                    if (escaped_z) break;
                    if((kUniformWaterPrimaryPath || global_urban) && !(kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) &&
                       urban_global_outside_water({position_x_mm,position_y_mm,position_z_mm},
                           {direction_x,direction_y,direction_z},urban_water_half_width_mm,phantom_length_mm)) {
                        water_urban_escaped_geometry=true;break;
                    }

                    const auto absolute_direction_x = sycl::fabs(direction_x);
                    const auto absolute_direction_y = sycl::fabs(direction_y);
                    const auto absolute_direction_z = sycl::fabs(direction_z);

                    auto bin = direction_z < 0.0F
                                   ? static_cast<int>(
                                         sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                   : static_cast<int>(
                                         sycl::floor(position_z_mm / depth_bin_width_mm));
                    bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                    if (primary_survival_device != nullptr && bin != last_survival_bin) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            survival(primary_survival_device[bin]);
                        survival.fetch_add(1U);
                        last_survival_bin = bin;
                    }

                    auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                    auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                    if ((kProductionPrimaryPath || enable_voxel_scoring)) {
                        const auto x_coordinate =
                            (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                        const auto y_coordinate =
                            (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                        voxel_x = direction_x < 0.0F
                                      ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(x_coordinate));
                        voxel_y = direction_y < 0.0F
                                      ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(y_coordinate));
                        voxel_x = sycl::max(
                            0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                        voxel_y = sycl::max(
                            0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                    }

                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const auto scored=urban_voxel_cell({position_x_mm,position_y_mm,position_z_mm},
                            {direction_x,direction_y,direction_z},{voxel_min_x_mm,voxel_min_y_mm,0},
                            {voxel_max_x_mm,voxel_max_y_mm,phantom_length_mm},
                            {voxel_size_x_mm,voxel_size_y_mm,depth_bin_width_mm},voxel_bins_x,voxel_bins_y,number_of_bins);
                        if(scored.valid){voxel_x=scored.ix;voxel_y=scored.iy;bin=scored.iz;}
                    }
                    const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                             static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                             static_cast<std::size_t>(voxel_x);

                    const auto energy_MeVu = energy_MeV * inverse_mass_number;
                    auto floating_index = (energy_MeVu - minimum_table_energy) * inverse_table_step;
                    auto index = static_cast<int>(sycl::floor(floating_index));
                    index = sycl::max(0, sycl::min(index, static_cast<int>(table_size) - 2));
                    const auto fraction =
                        sycl::clamp(floating_index - static_cast<float>(index), 0.0F, 1.0F);

                    const auto in_insert =
                        (!kUniformWaterPrimaryPath && enable_hetero_insert) &&
                        inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                             insert_x_min, insert_x_max, insert_y_min,
                                             insert_y_max, insert_z_min, insert_z_max);
                    auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                        position_z_mm, slab_z_ends_device, slab_densities_device,
                        (kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count), water_density_g_per_cm3);
                    if (in_insert) {
                        local_density_g_per_cm3 = insert_density_g_per_cm3;
                    }
                    std::uint8_t ct_material = 2;
                    auto in_ct = false;
                    if ((kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid))) {
                        float ct_rho = water_density_g_per_cm3;
                        in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                          ct_origin_x, ct_origin_y, ct_origin_z, ct_spacing_x,
                                          ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz,
                                          ct_density_device, ct_material_device, ct_rho,
                                          ct_material, direction_x, direction_y, direction_z);
                        if (in_ct) {
                            local_density_g_per_cm3 = ct_rho;
                        }
                    }

                    // Mirror secondary CT escape: the EM package has no exterior material.
                    if((kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) && !in_ct) {
                        unified_primary_escaped_ct=true;
                        break;
                    }

                    const auto layer_for_material =
                        (kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count) > 0
                            ? slab_layer_index(position_z_mm, slab_z_ends_device, (kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count))
                            : 0U;
                    float stopping_power_MeV_per_mm = 0.0F;
                    float step_mm = 0.0F;
                    UnifiedEmStep unified_primary_pre;
                    float unified_primary_rate=0,unified_primary_distance=std::numeric_limits<float>::infinity();
                    auto unified_uniform=[&](){return (rng::random_u32(spot_seed,rng_history,unified_primary_counter++,120)>>8)*0x1p-24f;};
                    auto unified_count=[&](int index,std::uint64_t count=1){
                        if(count==0)return; // zero-increment diagnostics must not touch globals
                        if constexpr(CARBON_EM_LOCAL_AUDIT) unified_primary_audit[index]+=count;
                            else {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(unified_audit[static_cast<std::size_t>(index)*kUnifiedEmAuditShards+lane%kUnifiedEmAuditShards]);a.fetch_add(count);
                            }
                    };
                    {
                        const int section=(kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid))?(in_ct?int(ct_material):-2):-1;
                        if(!CARBON_EM_MATERIAL_CACHE || !unified_primary_state.valid || unified_primary_state.section!=section ||
                           unified_primary_state.density!=local_density_g_per_cm3)
                            unified_primary_state=unified_device.select(section,local_density_g_per_cm3,unified_primary_species);
                        if(!unified_primary_state.covers(energy_MeV)){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,1,section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,0);
                            unified_count(0);break;}
                        if(unified_primary_clock.last_section!=unified_primary_state.section || unified_primary_clock.last_density!=unified_primary_state.density)primary_hadronic_cache.valid=false;
                        unified_primary_pre=unified_primary_state.prepare(energy_MeV);
                        stopping_power_MeV_per_mm = unified_primary_state.mix(
                            unified_primary_pre.lo.stopping + unified_primary_pre.lo.delta_stopping,
                            unified_primary_pre.hi.stopping + unified_primary_pre.hi.delta_stopping);
                        // Match ionIoni proposal segmentation while retaining aggregate loss.
                        // Debit only the true path accepted after geometry/nuclear competition.
                        if((kUniformWaterPrimaryPath || global_urban)) {
                            unified_primary_rate=unified_primary_clock.update(unified_primary_state,energy_MeV,
                                unified_primary_pre.lo.factor,unified_primary_pre.hi.factor);
                            if(!unified_primary_clock.clock.active)
                                unified_primary_clock.clock.arm(-sycl::log(sycl::fmax(unified_uniform(),0x1p-24f)));
                            unified_primary_distance=unified_primary_clock.clock.distance(unified_primary_rate);
                        } else {
                            unified_primary_clock.last_section=unified_primary_state.section;
                            unified_primary_clock.last_density=unified_primary_state.density;
                        }
                        step_mm=sycl::fmin((em_primary_step_scale!=1.f?unified_primary_state.research_step(energy_MeV,CARBON_EM_STEP_CACHE?unified_primary_pre:unified_primary_state.prepare(energy_MeV),em_primary_step_scale):(CARBON_EM_STEP_CACHE?unified_primary_state.step(unified_primary_pre):unified_primary_state.step(energy_MeV))),unified_primary_distance);
                        // Bound combined mean loss, not the old restricted range alone.
                        const float delta_sp=unified_primary_state.mix(unified_primary_pre.lo.delta_stopping,unified_primary_pre.hi.delta_stopping);
                        if(delta_sp>0) step_mm=sycl::fmin(step_mm,maximum_mean_loss_fraction*energy_MeV/sycl::fmax(1e-12f,delta_sp+unified_primary_state.mix(unified_primary_pre.lo.stopping,unified_primary_pre.hi.stopping)));
                    }

                    if (!(kUniformWaterPrimaryPath || global_urban) && absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                        if (dz_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dz_step);
                        }
                    }
                    if ((kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count) > 0) {
                        const auto slab_step = distance_to_slab_interface_mm(
                            position_z_mm, direction_z, slab_z_ends_device, (kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count),
                            phantom_length_mm);
                        if (slab_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, slab_step);
                        }
                    }
                    if ((!kUniformWaterPrimaryPath && enable_hetero_insert)) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                            insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                        if (insert_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, insert_step);
                        }
                    }
                    CtFaceClampResult ct_clamp_res{step_mm, false, 0};
                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct) {
                        ct_clamp_res = clamp_step_to_ct_faces_exact(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz);
                        step_mm = ct_clamp_res.step_mm;
                    }
                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        const auto dx_step = (boundary_x_mm - position_x_mm) / direction_x;
                        if (dx_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dx_step);
                        }
                    }
                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        const auto dy_step = (boundary_y_mm - position_y_mm) / direction_y;
                        if (dy_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dy_step);
                        }
                    }


                    UrbanGlobalProposal global_urban_proposal{};
                    CorrelatedScatteringStep global_urban_scattering{};
                    int global_urban_material=-1;
                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const auto cell=in_ct?urban_voxel_cell(
                            {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},
                            {ct_origin_x,ct_origin_y,ct_origin_z},
                            {ct_origin_x+ct_nx*ct_spacing_x,ct_origin_y+ct_ny*ct_spacing_y,ct_origin_z+ct_nz*ct_spacing_z},
                            {ct_spacing_x,ct_spacing_y,ct_spacing_z},ct_nx,ct_ny,ct_nz):
                            urban_global_water_cell({position_x_mm,position_y_mm,position_z_mm},
                                {direction_x,direction_y,direction_z},{voxel_min_x_mm,voxel_min_y_mm,0},
                                {voxel_max_x_mm,voxel_max_y_mm,phantom_length_mm},
                                {voxel_size_x_mm,voxel_size_y_mm,depth_bin_width_mm},voxel_bins_x,voxel_bins_y,number_of_bins,
                                urban_water_half_width_mm);
                        global_urban_material=urban_device.material(in_ct?int(ct_material):-1,local_density_g_per_cm3);
                        global_urban_proposal=urban_global_prepare(urban_device,global_urban_material,global_urban_ion,
                            energy_MeV,sycl::fmin(step_mm,maximum_step_mm),
                            {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},
                            cell,global_urban_boundary,global_urban_state,spot_seed,rng_history,steps,
                            local_density_g_per_cm3);
                        if(!global_urban_proposal.valid) {
                            record_unified_em_failure(unified_failure_count,unified_failure_records,70,
                                in_ct?int(ct_material):-1,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,step_mm);
                            unified_count(0);break;
                        }
                        step_mm=global_urban_proposal.true_path;
                    }
                    float accepted_loss_path_mm = step_mm;
                    bool urban_range_limited = false;
                    bool inelastic_this_step = false;
                    // Elastic candidates are resolved only by the configured CT/water bank.
                    bool ct_elastic_this_step = false;
                    std::int16_t cinel02_target_z_step = 0;
                    std::int16_t cinel02_target_a_step = 0;
                    if ((kProductionPrimaryPath || enable_inelastic) &&
                        energy_MeV > energy_cutoff_MeV) {
                        const auto cur_e_u = energy_MeV * inverse_mass_number;
                        if ((kProductionPrimaryPath || use_schneider_primary_xs)) {
                            if (in_ct || (!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water))) {
                                const std::uint32_t section_id = (!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water)) ? 255U : static_cast<std::uint32_t>(
                                    sycl::min(static_cast<std::uint32_t>(ct_material), 24U));
                                // v3: hazard from the masked rate-binary partials
                                // (single source with the target sampler); v1
                                // keeps the CSV XS table EXACTLY.
                                float mass_rate = 0.0F;
                                if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                                    mass_rate = schneider_ct_device_ctx.primary_rates(
                                        section_id, cur_e_u).total;
                                } else {
                                    mass_rate = schneider_primary_mass_xs(
                                        schneider_primary_xs_device, schneider_xs_sections,
                                        schneider_xs_energies, schneider_xs_e_min, schneider_xs_inv_dE,
                                        section_id, cur_e_u);
                                }
                                // Density enters exactly once, in the total
                                // hazard; target fractions from the sampler CDF
                                // are density-independent.
                                // Full-section elastic hazard (diagnostic): same
                                // optical-depth competition as inelastic; the
                                // branch below splits them exclusively, so no
                                // double counting of the total rate.
                                float macro_el_ct = 0.0F;
                                const bool elastic_armed =
                                    ((!kProductionPrimaryPath && use_all_elastic) && (in_ct || (!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water)))) ||
                                    (use_ct_elastic && in_ct &&
                                    schneider_ct_device_ctx.elastic_sampler.rate_version == 3);
                                if ((!kProductionPrimaryPath && use_all_elastic) && elastic_armed) {
                                    const float rate=all_elastic.rate(primary_elastic_projectile,(!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water))?25:section_id,cur_e_u);
                                    if(rate<0) {
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[4]).fetch_add(1);
                                        break;
                                    }
                                    macro_el_ct=local_density_g_per_cm3*rate;
                                } else if (elastic_armed) {
                                    const auto el_rates =
                                        schneider_ct_device_ctx.elastic_rates(
                                            section_id, cur_e_u);
                                    // Legacy diagnostic H-only target selection;
                                    // all-target isotropic needs explicit
                                    // opt-in (unphysical for heavy nuclei).
                                    macro_el_ct = local_density_g_per_cm3 *
                                        (ct_elastic_all_targets
                                             ? el_rates.total
                                             : el_rates.partials[0]);
                                }
                                float macro_inelastic = local_density_g_per_cm3 * mass_rate;
                                {
                                // Cache only the process whose post-step acceptance uses
                                // this rate. Elastic retains its own local hazard and must
                                // never enter the inelastic acceptance denominator.
                                macro_inelastic = (kProductionPrimaryPath || enable_inelastic)
                                    ? primary_hadronic_cache.update(cur_e_u, macro_inelastic)
                                    : 0.0F;
                                }
                                const float macro_tot = macro_inelastic + macro_el_ct;
                                ++local_schneider_rate_queries;

                                float u_nuc = 1.0F;
                                if (!nuclear_tau_active) {
                                    u_nuc = rng::uniform01(
                                        spot_seed, rng_history, nuclear_tau_rng_step++, 8);
                                }
                                const bool collision = consume_schneider_optical_depth_segment(
                                    nuclear_tau_remaining, nuclear_tau_active,
                                    accepted_loss_path_mm, macro_tot, u_nuc);
                                bool elastic_branch = false;
                                if (collision && elastic_armed && macro_tot > 0.0F) {
                                    const float u_br_el = rng::uniform01(
                                        spot_seed, rng_history, steps, 9);
                                    elastic_branch = (u_br_el * macro_tot < macro_el_ct);
                                }
                                if (elastic_branch) {
                                    ct_elastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                } else if (collision && (kProductionPrimaryPath || enable_inelastic)) {
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryHazards);
                                    if (schneider_inelastic_device != nullptr) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            nuc_ref(*schneider_inelastic_device);
                                        nuc_ref.fetch_add(1U);
                                    }
                                    if (inelastic_reaction_device != nullptr) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            reaction(inelastic_reaction_device[bin]);
                                        reaction.fetch_add(1U);
                                    }
                                    inelastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                }
                            }
                        }
                    }

                    if((kUniformWaterPrimaryPath || global_urban)) {
                        const bool ionization_proposal=accepted_loss_path_mm>=unified_primary_distance;
                        unified_primary_clock.clock.consume(accepted_loss_path_mm,unified_primary_rate,ionization_proposal);
                        global_urban_scattering=urban_global_finalize(global_urban_proposal,accepted_loss_path_mm,
                            spot_seed,rng_history,steps);
                        const int reason=urban_step_field_reason(global_urban_scattering);
                        if(reason) {
                            record_unified_em_failure(unified_failure_count,unified_failure_records,80+reason,
                                in_ct?int(ct_material):-1,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,accepted_loss_path_mm);
                            unified_count(0);break;
                        }
                        step_mm=global_urban_scattering.final_geom_path_mm;
                        urban_range_limited=global_urban_scattering.range_terminal;
                    }
                    // Diagnostic only: CT face clamping above guarantees the
                    // segment stays in its starting CT voxel. Do not score a
                    // laterally escaped segment into a clamped edge voxel.
                    if ((!kProductionPrimaryPath && enable_primary_voxel_fluence) &&
                        (!(kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) || in_ct)) {
                        sycl::atomic_ref<
                            DoseAtomicT, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_track_length(
                                primary_voxel_track_length_device[voxel_index]);
                        atomic_track_length.fetch_add(
                            static_cast<DoseAtomicT>(accepted_loss_path_mm));
                    }

                    float mean_loss_MeV = 0.0F;
                    {
                        // The unified loss sampler computes the physical mean below.
                        // This separate query is needed only by the optional audit.
                        if ((kProductionPrimaryPath ? nullptr : primary_loss_query_audit))
                            mean_loss_MeV=unified_primary_state.mean(
                                energy_MeV, accepted_loss_path_mm);
                    }
                    float deposited_MeV = 0.0F;

                    if ((kProductionPrimaryPath ? nullptr : primary_loss_query_audit) && mean_loss_MeV > 0 && energy_MeV > 0) {
                        // Bin 0: f<1e-12; bins 1..12: decades; bin 13: f>=1.
                        const float f = mean_loss_MeV / energy_MeV;
                        const int b = sycl::clamp(static_cast<int>(sycl::floor(sycl::log10(f)))+13, 0, 13);
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>
                            count((kProductionPrimaryPath ? nullptr : primary_loss_query_audit)[b]);
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>
                            energy_sum((kProductionPrimaryPath ? nullptr : primary_loss_query_audit)[b+14]);
                        count.fetch_add(1);
                        energy_sum.fetch_add(static_cast<std::uint64_t>(mean_loss_MeV*1.0e6F+0.5F));
                    }

                    if (urban_range_limited) deposited_MeV = energy_MeV;
                    float em_continuous_sampled_MeV =
                        urban_range_limited ? energy_MeV : 0.0F;
                    float em_delta_sampled_MeV = 0.0F;
                    if(!urban_range_limited){
                        auto draw=unified_em_loss<kPrimaryStage==7>(
                            unified_primary_state, unified_primary_clock,
                            energy_MeV, accepted_loss_path_mm,
                            unified_primary_rate, unified_primary_distance,
                            enable_energy_straggling, unified_primary_pre,
                            unified_uniform);
                        if(!draw.valid){
                            record_unified_em_failure(
                                unified_failure_count, unified_failure_records,
                                20 + static_cast<int>(draw.failure_stage),
                                unified_primary_state.section,
                                primary_atomic_number, primary_mass_number,
                                energy_MeV, local_density_g_per_cm3,
                                accepted_loss_path_mm, draw.mean,
                                draw.delta_mean, draw.delta_variance);
                            unified_count(0);break;}
                        deposited_MeV=draw.loss;unified_count(1);unified_count(3,draw.proposed);unified_count(4,draw.accepted);
                        unified_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                        em_continuous_sampled_MeV = draw.continuous;
                        em_delta_sampled_MeV = draw.delta;
                    }
                    const float em_loss_before_scale_MeV = deposited_MeV;
#if defined(CARBON_ENABLE_MINIBEAM)
                    if (enable_minibeam) {
                        deposited_MeV = sycl::fmin(
                            energy_MeV,
                            deposited_MeV * minibeam_water_primary_stopping_power_scale);
                    }
#endif
                    const float em_scale_factor =
                        em_loss_before_scale_MeV > 0.0F
                            ? deposited_MeV / em_loss_before_scale_MeV : 0.0F;
                    const float em_continuous_after_scale_MeV =
                        em_continuous_sampled_MeV * em_scale_factor;
                    const float em_delta_after_scale_MeV =
                        em_delta_sampled_MeV * em_scale_factor;
                    auto local_voxel_deposit_MeV = deposited_MeV;
                    auto same_voxel_electron_packet_MeV = 0.0F;
                    auto delta_tail_escaped_scorer_MeV = 0.0F;
                    // Forward-redistributed energy leaves the source depth bin, so the
                    // 1-D depth scorers must not credit it at the source bin (the 3-D
                    // march deposits below credit the destination bins instead).
                    auto forward_shifted_MeV = 0.0F;
                    auto transverse_relocated_MeV = 0.0F;
                    auto transverse_escaped_MeV = 0.0F;
                    auto water_physical_escape_MeV=0.0F;
                    float response_scored_separately_MeV=0.0F;
                    float material_untracked_MeV=0;
                    // Only the primary-loop EM loss exists here. Nuclear recoil
                    // and interaction deposits are scored later and must never
                    // be folded into this ionization response packet weight.
                    if(use_material_electron && (!enable_ct_grid || in_ct) && deposited_MeV>0) {
                        auto count=[&](int slot,std::uint64_t value) {
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);a.fetch_add(value);
                        };
                        auto untracked=[&](double weight,bool photon) {
                            material_untracked_MeV+=static_cast<float>(weight);
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(material_untracked_device[photon?0:1]);
                            a.fetch_add(static_cast<std::uint64_t>(weight*1e6+0.5));
                        };
                        // A separate deterministic packet stream does not advance the
                        // carbon transport RNG. Rebinding consumes this stream only.
                        std::uint64_t packet_rng=spot_seed ^ ((rng_history+1)*0x9e3779b97f4a7c15ULL) ^
                            (static_cast<std::uint64_t>(steps)<<32) ^ 0x6a09e667f3bcc909ULL;
                        auto uniform=[&]() -> double {return float(electron_fp32::random_draw(packet_rng));};
                        count(0,1);
                        const double density_u=uniform();
                        (void)uniform(); // Preserve the independent birth CDF draw and RNG order.
                        const auto birth_cdf_bits=packet_rng;
                        const double along=uniform();
                        const auto float_geometry=electron_fp32::ElectronCtGeometry{{ct_nx,ct_ny,ct_nz},
                            {float(ct_origin_x),float(ct_origin_y),float(ct_origin_z)},
                            {float(ct_spacing_x),float(ct_spacing_y),float(ct_spacing_z)},ct_density_device,ct_material_device};
                        const auto birth_position=[&]() -> std::array<double,3> {
                            const auto p=electron_fp32::electron_ct_birth_position(float_geometry,
                                {position_x_mm,position_y_mm,position_z_mm},{direction_x,direction_y,direction_z},step_mm,float(along));
                            return {p[0],p[1],p[2]};
                        }();
                        const auto birth_material=[&]() -> ElectronCtPointMaterial {
                            const auto p=electron_fp32::electron_ct_point_material(float_geometry,
                                {electron_fp32::Coordinate(birth_position[0]),electron_fp32::Coordinate(birth_position[1]),electron_fp32::Coordinate(birth_position[2])},
                                {direction_x,direction_y,direction_z});return {p.valid,p.density,p.section};
                        }();
                        electron_fp32::IndependentElectronBirthDraw birth;
                        bool selected_birth=true;
                        if constexpr(BroadBeam && JointResponse==3) {
                            if(birth_material.valid) {
                                const int bx=static_cast<int>(sycl::floor((birth_position[0]-ct_origin_x)/ct_spacing_x));
                                const int by=static_cast<int>(sycl::floor((birth_position[1]-ct_origin_y)/ct_spacing_y));
                                const int bz=static_cast<int>(sycl::floor((birth_position[2]-ct_origin_z)/ct_spacing_z));
                                if(bx>=0&&by>=0&&bz>=0&&bx<int(ct_nx)&&by<int(ct_ny)&&bz<int(ct_nz)) {
                                    selected_birth=broadbeam_electron_mask[(std::size_t(bz)*ct_ny+by)*ct_nx+bx];
                                    // Unselected ionization remains local by explicit research policy.
                                    if(!selected_birth)birth.valid=true;
                                }
                            }
                        }
                        if(birth_material.valid && selected_birth) {

                                const auto fresh=electron_fp32::prepare_independent_electron_birth(float_birth,birth_material.section,
                                    birth_material.density,energy_MeVu,float(density_u));
                                birth=fresh;
                                                    }
                        const auto ti=birth.node;
                        if(!birth.valid) {
                            count(1,1);untracked(deposited_MeV,false);
                            local_voxel_deposit_MeV=0;forward_shifted_MeV+=deposited_MeV;
                        } else if(birth.fraction>0 &&
                                  deposited_MeV*static_cast<float>(birth.fraction)>0.0F) {
                            // FP32 loss*yield can underflow to zero. A zero-weight
                            // packet carries no energy and must not enter transport.
                            // Response packets partition the already sampled total loss
                            // by the pinned energy-weighted birth fraction, as in water.
                            // Do not combine a binned birth law with individual aggregate
                            // delta draws: their cut-onset support need not coincide.
                            // No energy is added and the transport loss/RNG is unchanged.
                            const float weight=deposited_MeV*static_cast<float>(birth.fraction);
                            local_voxel_deposit_MeV-=weight;forward_shifted_MeV+=weight;

                                MaterialBirthTask task{QueuedElectronBirth{energy_MeVu,birth_cdf_bits,birth.node},birth_position,{direction_x,direction_y,direction_z},
                                    birth_material.density,packet_rng,global_history,static_cast<std::uint32_t>(steps),
                                    static_cast<std::uint32_t>(ti),weight};
                                grouped_weight=float(grouped_weight)+weight;
                                bool select=grouped_count==0;
                                if(!select) {
                                    const auto block=carbon::rng::philox4x32_10(
                                        {static_cast<std::uint32_t>(rng_history),static_cast<std::uint32_t>(rng_history>>32),
                                         static_cast<std::uint32_t>(group_draw),static_cast<std::uint32_t>(group_draw>>32)},
                                        static_cast<std::uint32_t>(spot_seed)^0xe1ec7a91U,
                                        static_cast<std::uint32_t>(spot_seed>>32)^0x47525031U);
                                    ++group_draw;
                                    const std::uint64_t bits=(std::uint64_t(block[0])<<32)|block[1];
                                    select=electron_fp32::Draw{bits}<weight/float(grouped_weight);
                                }
                                if(select)grouped_task=task;
                                if(++grouped_count==material_group_size)flush_electron_group();
                                // Pending W is excluded from deposited/grid energy until
                                // the task terminates. No untracked sink is recorded here.
                                material_untracked_MeV+=weight;
                        }
                    }


                    if (bin != pending_primary_bin) {
                        if (pending_primary_depth_MeV > 0.0) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_primary_bin]);
                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
                            pending_primary_depth_MeV = 0.0;
                        }
                        if (enable_let_scoring) {
                            flush_letd_moments_device(
                                let_moments_device, number_of_bins,
                                static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                                pending_let_numerator, pending_let_denominator, true);
                            pending_let_numerator = 0.0;
                            pending_let_denominator = 0.0;
                        }
                        pending_primary_bin = bin;
                    }
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && voxel_index != pending_primary_voxel) {
                        if (pending_primary_voxel_MeV > 0.0 && pending_primary_voxel >= 0 &&
                            pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                            atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            if ((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring)) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_primary_origin(
                                        charged_origin_voxel_dose_device[pending_primary_voxel]);
                                atomic_primary_origin.fetch_add(
                                    static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            }
                            if ((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring)) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_primary_component(
                                        minibeam_component_voxel_dose_device[pending_primary_voxel]);
                                atomic_primary_component.fetch_add(
                                    static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            }
                        }
                        pending_primary_voxel_MeV = 0.0;
                        if (enable_let_scoring && voxel_let_moments_device != nullptr && pending_primary_voxel >= 0) {
                            flush_letd_moments_device(
                                voxel_let_moments_device, number_of_voxels,
                                pending_primary_voxel, nullptr, 0, 0,
                                pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                            pending_voxel_let_numerator = 0.0;
                            pending_voxel_let_denominator = 0.0;
                        }
                        pending_primary_voxel = voxel_index;
                    }

                    // Keep the legacy unrestricted-depth escape convention, but
                    // remove relocated energy now tallied at its destination.
                    pending_primary_depth_MeV += deposited_MeV - forward_shifted_MeV -
                        transverse_relocated_MeV;
                    pending_primary_depth_MeV += same_voxel_electron_packet_MeV;
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0) {
                        pending_primary_voxel_MeV += local_voxel_deposit_MeV;
                        pending_primary_voxel_MeV += same_voxel_electron_packet_MeV;
                        if (enable_minibeam_primary_roi_scoring) {
                            const auto score_c12 = [&](std::size_t kind, float value) {
                                score_minibeam_primary_roi_device(
                                    minibeam_primary_roi_dose_device, kind, energy_MeV,
                                    primary_mass_number, voxel_index, number_of_bins, voxel_bins_x,
                                    voxel_bins_y, minibeam_fixed_region_by_x_bin_device,
                                    value);
                            };
                            score_c12(minibeam_primary_roi_local_total_deposit,
                                      local_voxel_deposit_MeV +
                                          same_voxel_electron_packet_MeV);
                            score_c12(minibeam_primary_roi_continuous_sampled,
                                      em_continuous_sampled_MeV);
                            score_c12(minibeam_primary_roi_delta_sampled,
                                      em_delta_sampled_MeV);
                            score_c12(minibeam_primary_roi_continuous_after_scale,
                                      em_continuous_after_scale_MeV);
                            score_c12(minibeam_primary_roi_delta_after_scale,
                                      em_delta_after_scale_MeV);
                            score_c12(minibeam_primary_roi_fluence, step_mm);
                        }
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(
                                deposited_MeV - forward_shifted_MeV -
                                transverse_relocated_MeV - transverse_escaped_MeV)+
                                static_cast<DepthAtomicT>(same_voxel_electron_packet_MeV));
                        }
                    }
                    history_deposited_MeV += deposited_MeV-water_physical_escape_MeV-material_untracked_MeV;
                    grid_split(
                        (kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0,
                        deposited_MeV - delta_tail_escaped_scorer_MeV-water_physical_escape_MeV-
                            material_untracked_MeV-response_scored_separately_MeV);
                    grid_split(false, delta_tail_escaped_scorer_MeV);
                    last_primary_stopping_power_MeV_per_mm = stopping_power_MeV_per_mm;
                    last_primary_density_g_per_cm3 = local_density_g_per_cm3;

                    if (enable_let_scoring) {
                        const auto step_numerator =
                            static_cast<double>(deposited_MeV) *
                            static_cast<double>(stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(local_density_g_per_cm3, 1.0e-6F));
                        const auto step_denominator = static_cast<double>(deposited_MeV);
                        pending_let_numerator += step_numerator;
                        pending_let_denominator += step_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += step_numerator;
                            pending_voxel_let_denominator += step_denominator;
                        }
                    }

                    const float seg_dir_x = direction_x;
                    const float seg_dir_y = direction_y;
                    const float seg_dir_z = direction_z;
                    Direction3F water_scattering_displacement{
                        0.0F, 0.0F, 0.0F};

                    if ((kProductionPrimaryPath || enable_multiple_scattering)) {
                        auto radiation_length_g_per_cm2 =
                            static_cast<float>(active_water_radiation_length);
                        if ((kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct && (kProductionPrimaryPath || enable_ct_material_mcs)) {
                            if (ct_material_ids_are_schneider_sections) {
                                radiation_length_g_per_cm2 = static_cast<float>(
                                    select_transport_radiation_length_g_per_cm2(
                                        ct_material_ids_are_schneider_sections, in_ct,
                                        ct_material, (kProductionPrimaryPath || enable_ct_material_mcs),
                                        water_radiation_length_g_per_cm2));
                            } else {
                                radiation_length_g_per_cm2 = static_cast<float>(
                                    ct_material_radiation_length_g_per_cm2(
                                        ct_material_class(ct_material, false)));
                            }
                        } else if (in_insert) {
                            radiation_length_g_per_cm2 = insert_radiation_length_g_per_cm2;
                        } else if ((kUniformWaterPrimaryPath ? std::size_t{0} : slab_layer_count) > 0) {
                            radiation_length_g_per_cm2 =
                                slab_radiation_lengths_device[layer_for_material];
                        }
                        // Generic primary FE path. Yields to the minibeam water
                        // branch below when a non-legacy minibeam primary
                        // model is selected (FE-tail or Urban v2); otherwise
                        // the minibeam branch would be dead code whenever the
                        // generic fermi_eyges model is on.
                        if((kUniformWaterPrimaryPath || global_urban)) {
                            direction_x=global_urban_scattering.direction.x;
                            direction_y=global_urban_scattering.direction.y;
                            direction_z=global_urban_scattering.direction.z;
                            water_scattering_displacement=global_urban_scattering.displacement_mm;
                            global_urban_state=global_urban_proposal.state;
                            urban_global_count(urban_step_counts,global_urban_ion,false);
                        } else if (primary_fermi_eyges
                        ) {
                            auto observation_path_mm = -1.0F;
#if defined(CARBON_ENABLE_MINIBEAM)
                            auto observation_plane =
                                minibeam_water_primary_plane_count;
                            if (minibeam_water_primary_plane_records_device !=
                                    nullptr &&
                                seg_dir_z > 0.0F) {
                                const auto straight_end_z =
                                    position_z_mm + seg_dir_z * step_mm;
                                for (std::size_t plane = 0;
                                     plane < minibeam_water_primary_plane_count;
                                     ++plane) {
                                    auto& candidate =
                                        minibeam_water_primary_plane_records_device[
                                            global_history *
                                                minibeam_water_primary_plane_count +
                                            plane];
                                    const auto plane_depth_mm =
                                        minibeam_water_primary_plane_depths_device[
                                            plane];
                                    if (!candidate.valid &&
                                        position_z_mm < plane_depth_mm &&
                                        straight_end_z >= plane_depth_mm) {
                                        observation_plane = plane;
                                        observation_path_mm = sycl::clamp(
                                            (plane_depth_mm - position_z_mm) /
                                                seg_dir_z,
                                            1.0e-6F,
                                            step_mm - 1.0e-6F);
                                        break;
                                    }
                                }
                            }
#endif
                            const auto correlated =
                                ion_fermi_eyges_transport_step(
                                    Direction3F{direction_x, direction_y,
                                                direction_z},
                                    energy_MeV, primary_atomic_number,
                                    primary_mass_number, deposited_MeV, step_mm,
                                    c12_fermi_eyges_max_segment_mm,
                                    local_density_g_per_cm3,
                                    radiation_length_g_per_cm2, spot_seed,
                                    rng_history,
                                    static_cast<std::uint64_t>(steps), 40U,
                                    observation_path_mm, true, primary_mcs_mass_override_MeV, fe_gaussian_only,
                                    fe_moments, fe_moments ? fe_geometry_safety(
                                        {position_x_mm,position_y_mm,position_z_mm},enable_ct_grid,in_ct,
                                        {ct_origin_x,ct_origin_y,ct_origin_z},{ct_spacing_x,ct_spacing_y,ct_spacing_z},
                                        ct_nx,ct_ny,ct_nz,phantom_length_mm,fe_simple_geometry) : 0.0F,
                                    fe_counts,fe_diagnostics);
#if defined(CARBON_ENABLE_MINIBEAM)
                            if (correlated.observation_valid &&
                                observation_plane <
                                    minibeam_water_primary_plane_count) {
                                const auto plane_depth_mm =
                                    minibeam_water_primary_plane_depths_device[
                                        observation_plane];
                                auto crossing_x = position_x_mm +
                                    seg_dir_x * observation_path_mm +
                                    correlated.observation_displacement_mm.x;
                                auto crossing_y = position_y_mm +
                                    seg_dir_y * observation_path_mm +
                                    correlated.observation_displacement_mm.y;
                                const auto crossing_z = position_z_mm +
                                    seg_dir_z * observation_path_mm +
                                    correlated.observation_displacement_mm.z;
                                if (correlated.observation_direction.z >
                                    1.0e-6F) {
                                    const auto residual_path =
                                        (plane_depth_mm - crossing_z) /
                                        correlated.observation_direction.z;
                                    crossing_x += residual_path *
                                        correlated.observation_direction.x;
                                    crossing_y += residual_path *
                                        correlated.observation_direction.y;
                                }
                                auto& record =
                                    minibeam_water_primary_plane_records_device[
                                        global_history *
                                            minibeam_water_primary_plane_count +
                                        observation_plane];
                                const auto step_fraction = sycl::clamp(
                                    observation_path_mm / step_mm, 0.0F, 1.0F);
                                record.history = global_history;
                                record.particle_id = global_history;
                                record.rng_stream = rng_history;
                                record.plane_index =
                                    static_cast<std::uint32_t>(observation_plane);
                                record.atomic_number = static_cast<std::int16_t>(
                                    primary_atomic_number);
                                record.mass_number = static_cast<std::int16_t>(
                                    primary_mass_number);
                                record.depth_mm = plane_depth_mm;
                                record.kinetic_energy_MeV = sycl::fmax(
                                    0.0F, energy_MeV -
                                        step_fraction * deposited_MeV);
                                record.weight = 1.0F;
                                record.x_mm = crossing_x;
                                record.y_mm = crossing_y;
                                record.direction_x =
                                    correlated.observation_direction.x;
                                record.direction_y =
                                    correlated.observation_direction.y;
                                record.direction_z =
                                    correlated.observation_direction.z;
                                record.valid = 1U;
                            }
#endif
                            direction_x = correlated.direction.x;
                            direction_y = correlated.direction.y;
                            direction_z = correlated.direction.z;
                            water_scattering_displacement =
                                correlated.displacement_mm;
                        }
                        else {
                            auto theta0 = highland_projected_rms_angle_device(
                                energy_MeV, primary_atomic_number, primary_mass_number,
                                step_mm, local_density_g_per_cm3,
                                radiation_length_g_per_cm2, primary_mcs_mass_override_MeV) * multiple_scattering_scale;
                            float angular_scale = 1.0F;
                            const auto u0 = rng::uniform01(
                                spot_seed, rng_history, steps, 3);
                            const auto u1 = rng::uniform01(
                                spot_seed, rng_history, steps, 4);
                            const auto u2 = rng::uniform01(
                                spot_seed, rng_history, steps, 5);
                            const auto u3 = rng::uniform01(
                                spot_seed, rng_history, steps, 6);
                            // Unified MSC transport: Highland mode returns the
                            // scattered direction with zero intra-step
                            // displacement (endpoint scattering). Correlated
                            // displacement lives in the FE-tail / Urban-v2
                            // branches above, selected before this fallback.
                            const auto msc = msc_highland_primary_step(
                                Direction3F{direction_x, direction_y,
                                            direction_z},
                                angular_scale * theta0, u0, u1, u2, u3);
                            direction_x = msc.direction.x;
                            direction_y = msc.direction.y;
                            direction_z = msc.direction.z;
                        }
                    }

                    position_x_mm += seg_dir_x * step_mm +
                        water_scattering_displacement.x;
                    position_y_mm += seg_dir_y * step_mm +
                        water_scattering_displacement.y;
                    position_z_mm += seg_dir_z * step_mm +
                        water_scattering_displacement.z;
                    if((kUniformWaterPrimaryPath || global_urban)) {
                        global_urban_boundary=global_urban_scattering.boundary_crossed;
                        const auto p=urban_voxel_snap_endpoint(global_urban_proposal.cell,step_mm,
                            {position_x_mm,position_y_mm,position_z_mm});
                        position_x_mm=p.x;position_y_mm=p.y;position_z_mm=p.z;
                    }

                    if (enable_minibeam_primary_roi_scoring &&
                        minibeam_spatial_audit_device != nullptr &&
                        voxel_index >= 0 && step_mm > 0.0F) {
                        const float x0 = position_x_mm - seg_dir_x * step_mm -
                            water_scattering_displacement.x;
                        const float y0 = position_y_mm - seg_dir_y * step_mm -
                            water_scattering_displacement.y;
                        const float z0 = position_z_mm - seg_dir_z * step_mm -
                            water_scattering_displacement.z;
                        auto voxel_at = [&](float x, float y, float z) {
                            const int ix = static_cast<int>(sycl::floor(
                                (x - voxel_min_x_mm) * inverse_voxel_size_x_mm));
                            const int iy = static_cast<int>(sycl::floor(
                                (y - voxel_min_y_mm) * inverse_voxel_size_y_mm));
                            const int iz = static_cast<int>(
                                sycl::floor(z / voxel_size_z_mm));
                            if (ix < 0 || iy < 0 || iz < 0 ||
                                ix >= static_cast<int>(voxel_bins_x) ||
                                iy >= static_cast<int>(voxel_bins_y) ||
                                iz >= static_cast<int>(number_of_bins)) {
                                return -1;
                            }
                            return (iz * static_cast<int>(voxel_bins_y) + iy) *
                                static_cast<int>(voxel_bins_x) + ix;
                        };
                        const int v_straight = voxel_at(
                            x0 + seg_dir_x * step_mm,
                            y0 + seg_dir_y * step_mm,
                            z0 + seg_dir_z * step_mm);
                        const int v_end = voxel_at(
                            position_x_mm, position_y_mm, position_z_mm);
                        auto roi_of = [&](int voxel) -> int {
                            if (voxel < 0 ||
                                minibeam_fixed_region_by_x_bin_device == nullptr) {
                                return -1;
                            }
                            return static_cast<int>(
                                minibeam_fixed_region_by_x_bin_device[
                                    static_cast<std::size_t>(voxel) %
                                    voxel_bins_x]);
                        };
                        const int roi0 = roi_of(voxel_index);
                        const int roi_straight = roi_of(v_straight);
                        const int roi_end = roi_of(v_end);
                        auto add_slot = [&](std::size_t slot, std::uint64_t value) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                a(minibeam_spatial_audit_device[slot]);
                            a.fetch_add(value);
                        };
                        add_slot(0, 1);
                        if (v_straight != voxel_index) add_slot(1, 1);
                        if (roi_straight != roi0 && roi_straight >= 0 && roi0 >= 0)
                            add_slot(2, 1);
                        if (v_end != voxel_index) add_slot(3, 1);
                        if (roi_end != roi0 && roi_end >= 0 && roi0 >= 0)
                            add_slot(4, 1);
                        const auto mev = static_cast<std::uint64_t>(
                            deposited_MeV * 1.0e6f + 0.5f);
                        add_slot(5, mev);
                        if (roi_straight != roi0 && roi_straight >= 0 && roi0 >= 0)
                            add_slot(6, mev);
                        if (roi_end != roi0 && roi_end >= 0 && roi0 >= 0)
                            add_slot(7, mev);
                    }

#if defined(CARBON_ENABLE_MINIBEAM)
                    if ((!(kProductionPrimaryPath || enable_multiple_scattering) ||
                         !primary_fermi_eyges) &&
                        minibeam_water_primary_plane_records_device != nullptr &&
                        seg_dir_z > 0.0F) {
                        const auto previous_z_mm = position_z_mm -
                            seg_dir_z * step_mm -
                            water_scattering_displacement.z;
                        const auto previous_x_mm = position_x_mm -
                            seg_dir_x * step_mm -
                            water_scattering_displacement.x;
                        const auto previous_y_mm = position_y_mm -
                            seg_dir_y * step_mm -
                            water_scattering_displacement.y;
                        for (std::size_t plane = 0;
                             plane < minibeam_water_primary_plane_count;
                             ++plane) {
                            const auto plane_depth_mm =
                                minibeam_water_primary_plane_depths_device[plane];
                            if (!(previous_z_mm < plane_depth_mm &&
                                  position_z_mm >= plane_depth_mm)) {
                                continue;
                            }
                            auto& record =
                                minibeam_water_primary_plane_records_device[
                                    global_history *
                                        minibeam_water_primary_plane_count +
                                    plane];
                            if (record.valid) continue;
                            const auto fraction = sycl::clamp(
                                (plane_depth_mm - previous_z_mm) /
                                    (position_z_mm - previous_z_mm),
                                0.0F, 1.0F);
                            record.history = global_history;
                            record.particle_id = global_history;
                            record.rng_stream = rng_history;
                            record.plane_index =
                                static_cast<std::uint32_t>(plane);
                            record.atomic_number = static_cast<std::int16_t>(
                                primary_atomic_number);
                            record.mass_number = static_cast<std::int16_t>(
                                primary_mass_number);
                            record.depth_mm = plane_depth_mm;
                            // The transport step moves along seg_dir and applies
                            // its angular kick at the end.  Keep every plane
                            // field at the same interpolated crossing rather
                            // than mixing the crossing position with the
                            // post-kick direction and step-start energy.
                            record.kinetic_energy_MeV = sycl::fmax(
                                0.0F, energy_MeV - fraction * deposited_MeV);
                            record.weight = 1.0F;
                            record.x_mm = previous_x_mm +
                                fraction * (seg_dir_x * step_mm +
                                            water_scattering_displacement.x);
                            record.y_mm = previous_y_mm +
                                fraction * (seg_dir_y * step_mm +
                                            water_scattering_displacement.y);
                            const auto record_direction = Direction3F{
                                seg_dir_x, seg_dir_y, seg_dir_z};
                            record.direction_x = record_direction.x;
                            record.direction_y = record_direction.y;
                            record.direction_z = record_direction.z;
                            record.valid = 1U;
                        }
                    }
#endif

                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || (!kUniformWaterPrimaryPath && enable_ct_grid)) && in_ct && ct_clamp_res.hit_face && !inelastic_this_step && !ct_elastic_this_step) {
                        if ((ct_clamp_res.axis_mask & 1) != 0 && sycl::fabs(seg_dir_x) > 1.0e-6F) {
                            const float fx = (position_x_mm - ct_origin_x) / ct_spacing_x;
                            const int face_x = static_cast<int>(sycl::round(fx));
                            const float b_x = ct_origin_x + static_cast<float>(face_x) * ct_spacing_x;
                            position_x_mm = sycl::nextafter(b_x, seg_dir_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 2) != 0 && sycl::fabs(seg_dir_y) > 1.0e-6F) {
                            const float fy = (position_y_mm - ct_origin_y) / ct_spacing_y;
                            const int face_y = static_cast<int>(sycl::round(fy));
                            const float b_y = ct_origin_y + static_cast<float>(face_y) * ct_spacing_y;
                            position_y_mm = sycl::nextafter(b_y, seg_dir_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 4) != 0 && sycl::fabs(seg_dir_z) > 1.0e-6F) {
                            const float fz = (position_z_mm - ct_origin_z) / ct_spacing_z;
                            const int face_z = static_cast<int>(sycl::round(fz));
                            const float b_z = ct_origin_z + static_cast<float>(face_z) * ct_spacing_z;
                            position_z_mm = sycl::nextafter(b_z, seg_dir_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    if (!(kUniformWaterPrimaryPath || global_urban) && absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if (sycl::fabs(boundary_z_mm - position_z_mm) <= 1.0e-5F) {
                            position_z_mm = sycl::nextafter(
                                boundary_z_mm,
                                direction_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        if (sycl::fabs(boundary_x_mm - position_x_mm) <= 1.0e-5F) {
                            position_x_mm = sycl::nextafter(
                                boundary_x_mm,
                                direction_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (!(kUniformWaterPrimaryPath || global_urban) && (kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        if (sycl::fabs(boundary_y_mm - position_y_mm) <= 1.0e-5F) {
                            position_y_mm = sycl::nextafter(
                                boundary_y_mm,
                                direction_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    energy_MeV -= deposited_MeV;

                    if(defer_material_packets && (ct_elastic_this_step || inelastic_this_step))flush_electron_group();

                    // Elastic endpoint: full bank uses target-specific TOPAS samples;
                    // the separate legacy diagnostic retains its explicit isotropic law.
                    // Supported recoils are queued; below-cutoff energy is scored locally.
                    if((!kProductionPrimaryPath && use_all_elastic) && ct_elastic_this_step && energy_MeV>energy_cutoff_MeV &&
                       all_elastic.rate(primary_elastic_projectile,(!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water))?25:interaction_section,energy_MeV*inverse_mass_number)==0) {
                        ct_elastic_this_step=false;
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[10]).fetch_add(1);
                    }
                    if ((use_ct_elastic || (!kProductionPrimaryPath && use_all_elastic)) && ct_elastic_this_step &&
                        energy_MeV > energy_cutoff_MeV) {
                        const float u_tgt = rng::uniform01(
                            spot_seed, rng_history, steps, 12);
                        const auto draw = (!kProductionPrimaryPath && use_all_elastic) ? all_elastic.draw(primary_elastic_projectile,
                            (!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water))?25:interaction_section,energy_MeV,direction_x,direction_y,direction_z,
                            u_tgt,rng::uniform01(spot_seed,rng_history,steps,70),
                            rng::uniform01(spot_seed,rng_history,steps,71),rng::uniform01(spot_seed,rng_history,steps,11)) : ElasticDraw{};
                        if((!kProductionPrimaryPath && use_all_elastic) && !draw.valid) {
                            sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[5]).fetch_add(1);
                            break;
                        }
                        const int target_z=(!kProductionPrimaryPath && use_all_elastic)?draw.target_z:(ct_elastic_all_targets?
                            sample_schneider_target_device(schneider_ct_device_ctx.elastic_sampler,interaction_section,
                                energy_MeV*inverse_mass_number,u_tgt):1);
                        const int target_idx=carbon::elastic_target_index_from_z(target_z);
                        const int target_a=(!kProductionPrimaryPath && use_all_elastic)?draw.target_a:static_cast<int>(carbon::kElasticTargetMassU[target_idx<0?0:target_idx]);
                        if (target_idx >= 0) {
                            const auto scat=(!kProductionPrimaryPath && use_all_elastic)?draw.outcome:carbon::sample_c12_target_elastic(
                                energy_MeV,direction_x,direction_y,direction_z,
                                rng::uniform01(spot_seed,rng_history,steps,10),rng::uniform01(spot_seed,rng_history,steps,11),
                                carbon::kElasticTargetMassU[target_idx]);
                            if((!kProductionPrimaryPath && use_all_elastic))sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[0]).fetch_add(1);
                            energy_MeV = scat.projectile_ke_MeV;
                            direction_x = scat.proj_dir_x;
                            direction_y = scat.proj_dir_y;
                            direction_z = scat.proj_dir_z;
                            if ((carbon::get_charged_species_idx(target_z,target_a)>=0 ||
                                 ((!kProductionPrimaryPath && use_all_elastic)&&recoil_stopping.projectile(target_z,target_a)>=0)) && enable_secondary_transport &&
                                scat.recoil_ke_MeV > energy_cutoff_MeV &&
                                secondary_queue_device != nullptr) {
                                // Elastic recoil born accounting: mirrors the
                                // inelastic born/queued pair so the born
                                // conservation gate stays exact.
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedBorn);
                                auto count_ref = sycl::atomic_ref<
                                    uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *secondary_count_device);
                                const auto base_idx = count_ref.fetch_add(1U);
                                if (base_idx < max_secondaries) {
                                    SecondaryParticle proton{};
                                    proton.z = target_z;
                                    proton.a = target_a;
                                    if(carbon::get_charged_species_idx(target_z,target_a)<0)proton.generation=cinel02_max_secondary_inelastic_generations;
                                    proton.energy_MeV = scat.recoil_ke_MeV;
                                    proton.pos_x_mm = position_x_mm;
                                    proton.pos_y_mm = position_y_mm;
                                    proton.pos_z_mm = position_z_mm;
                                    proton.dir_x = scat.recoil_dir_x;
                                    proton.dir_y = scat.recoil_dir_y;
                                    proton.dir_z = scat.recoil_dir_z;
                                    proton.weight = 1.0F;
                                    proton.parent_history = global_history;
                                    proton.rng_stream = rng::child_stream(
                                        rng_history, rng::branch_tag(
                                            rng::branch_role_primary_charged, steps));
                                    secondary_queue_device[base_idx] = proton;
                                    if((!kProductionPrimaryPath && use_all_elastic)){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[1]).fetch_add(scat.recoil_ke_MeV);}
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryChargedQueued);
                                    if (cinel02_species_energy_device != nullptr) {
                                        cinel02_record_queued_secondary_birth_device(
                                            cinel02_species_energy_device, proton.z, proton.a,
                                            proton.energy_MeV);
                                    }
                                } else if (secondary_overflow_count_device != nullptr) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_ov(*secondary_overflow_count_device);
                                    atomic_ov.fetch_add(1U);
                                    if((!kProductionPrimaryPath && use_all_elastic)){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[2]).fetch_add(scat.recoil_ke_MeV);sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[11]).fetch_add(1);}
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_energy_device).fetch_add(scat.recoil_ke_MeV);
                                }
                            } else if((!kProductionPrimaryPath && use_all_elastic) && scat.recoil_ke_MeV>energy_cutoff_MeV) {
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[3]).fetch_add(1);
                                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(untracked_nuclear_device[global_history]).fetch_add(scat.recoil_ke_MeV);
                            } else if (scat.recoil_ke_MeV > 0.0F) {
                                if ((kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0 &&
                                    voxel_index < number_of_voxels) {
                                    pending_primary_voxel_MeV += scat.recoil_ke_MeV;
                                }
                                history_deposited_MeV += scat.recoil_ke_MeV;
                                if((!kProductionPrimaryPath && use_all_elastic)){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[0]).fetch_add(scat.recoil_ke_MeV);}
                                pending_primary_depth_MeV += scat.recoil_ke_MeV;
                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                    (kProductionPrimaryPath || enable_voxel_scoring)&&voxel_index>=0,scat.recoil_ke_MeV);
                            }
                        }
                    }


                    if ((kProductionPrimaryPath || use_schneider_primary_xs) && (in_ct || (!kProductionPrimaryPath && (kUniformWaterPrimaryPath || use_unified_water))) && inelastic_this_step) {
                        if (is_primary_attenuation_only) {
                            primary_inelastic_occurred = true;
                            if (first_interactions_device != nullptr && first_interactions_count_device != nullptr) {
                                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    count_ref(*first_interactions_count_device);
                                const auto slot = count_ref.fetch_add(1U);
                                if (slot < number_of_histories) {
                                    first_interactions_device[slot] = PrimaryFirstInteractionRecord{
                                        position_x_mm,
                                        position_y_mm,
                                        position_z_mm,
                                        energy_MeV * inverse_mass_number,
                                        interaction_section,
                                        interaction_density
                                    };
                                }
                            }
                            if (primary_terminal_counts_device != nullptr) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    term_ref(primary_terminal_counts_device[0]);
                                term_ref.fetch_add(1U);
                            }
                            if (untracked_nuclear_device != nullptr) {
                                untracked_nuclear_device[global_history] = energy_MeV;
                            }
                            energy_MeV = 0.0F;
                            break;
                        }

                        // Production Schneider CT primary inelastic collision.
                        // Exact (C12, target_Z) channel, bounded energy domain,
                        // stochastic bracketing. Tag 15 drives the bracket
                        // choice, tag 16 the intra-node event pick.
                        const float cur_primary_e_u = energy_MeV * inverse_mass_number;
                        const float u_target = rng::uniform01(spot_seed, rng_history, steps, 14);
                        // v3: masked draw from the same partials as the hazard.
                        // A non-positive result (masked hazard fired but
                        // slowing emptied every channel before the collision
                        // point) is a primary post-EM null collision: no
                        // package query, track continues, never
                        // UnsupportedTargets. v1 keeps the legacy call and
                        // flows to lookup EXACTLY as before.
                        int target_z = 0;
                        if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                            const auto masked = schneider_ct_device_ctx.primary_rates(
                                interaction_section, cur_primary_e_u);
                            target_z = sample_masked_schneider_target_device(
                                masked.partials, u_target);
                            {
                            // Dedicated dimension 44; do not reuse source, loss,
                            // target, event or electron-response variates.
                            const bool accepted = primary_hadronic_cache.accept(
                                interaction_density * masked.total,
                                rng::uniform01(spot_seed, rng_history, steps, 44));
                            // Candidate-only: rejected proposal reuses the
                            // existing energy-preserving post-EM null route.
                            if (!accepted) target_z = 0;
                            }
                        } else {
                            target_z = sample_schneider_target_device(
                                schneider_ct_device_ctx.primary_sampler,
                                interaction_section,
                                cur_primary_e_u,
                                u_target);
                        }
                        if (schneider_ct_device_ctx.primary_sampler.rate_version == 3 &&
                            target_z <= 0) {
                            // Null collision: retain the surviving track and complete this EM step.
                            inelastic_this_step = false;
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::PrimaryPostEmNullCollisions);
                            schneider_float_add_device(
                                schneider_float_device,
                                SchneiderFloatSlot::PostEmNullEnergy,
                                sycl::fmax(0.0F, energy_MeV));
                        } else {

                        const float u_bracket = rng::uniform01(spot_seed, rng_history, steps, 15);
                        const float u_event = rng::uniform01(spot_seed, rng_history, steps, 16);
                        const auto primary_lookup = cinel03_lookup_event_device(
                            schneider_ct_device_ctx.primary_energy_nodes,
                            schneider_ct_device_ctx.primary_node_count,
                            schneider_ct_device_ctx.primary_event_offsets,
                            schneider_ct_device_ctx.primary_event_indices,
                            schneider_ct_device_ctx.primary_total_events,
                            primary_atomic_number, primary_mass_number, target_z,
                            cur_primary_e_u,
                            u_bracket, u_event);
                        schneider_record_lookup_device(schneider_diag_device,
                                                       schneider_float_device,
                                                       true, primary_lookup,
                                                       energy_MeV);

                        if (primary_lookup.status != Cinel03LookupStatus::Hit) {
                            // Per-miss log: recompute the macro total exactly
                            // as at hazard time (density x mass, once).
                            // Per-miss macro total recomputed exactly as at
                            // hazard time: v3 uses the masked binary total
                            // (no CSV exists for v3), v1 the CSV XS table.
                            float primary_miss_macro = 0.0F;
                            if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                                primary_miss_macro = interaction_density *
                                    schneider_ct_device_ctx.primary_rates(
                                        interaction_section, cur_primary_e_u).total;
                            } else {
                                primary_miss_macro = interaction_density *
                                    schneider_primary_mass_xs(
                                        schneider_primary_xs_device, schneider_xs_sections,
                                        schneider_xs_energies, schneider_xs_e_min, schneider_xs_inv_dE,
                                        interaction_section, cur_primary_e_u);
                            }
                            schneider_log_miss_device(
                                schneider_miss_device, schneider_miss_count_device,
                                kSchneiderMissLogCap, true,
                                primary_atomic_number, primary_mass_number, target_z,
                                interaction_section > 255 ? 255
                                                          : static_cast<std::uint8_t>(interaction_section),
                                0, primary_lookup, cur_primary_e_u, deposited_MeV,
                                primary_miss_macro, 0.0F, energy_MeV,
                                spot_initial_energy_MeV);
                            if (untracked_nuclear_device != nullptr) {
                                untracked_nuclear_device[global_history] = energy_MeV;
                            }
                            energy_MeV = 0.0F;
                            break;
                        }


                        // A hazard proposal (including a post-EM null collision)
                        // is not an inelastic reaction. Mark only an event hit.
                        primary_inelastic_occurred = true;
                        const auto& event = schneider_ct_device_ctx.primary_interactions[primary_lookup.event_index];
                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);

                        pending_primary_depth_MeV += local_deposit;
                        if ((kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0) {
                            pending_primary_voxel_MeV += local_deposit;
                        }
                        history_deposited_MeV += local_deposit;
                        grid_deposit_split_device(
                            grid_deposited_in_device, grid_deposited_out_device,
                            (kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0, local_deposit);

                        float charged_accounted_MeV = 0.0F;
                        float neutral_accounted_MeV = 0.0F;
                        float unsupported_accounted_MeV = 0.0F;

                        const std::uint32_t prod_offset = event.product_offset;
                        const std::uint32_t prod_count = event.direct_product_count;

                        // Finite event-library azimuths are not a preferred
                        // laboratory direction. Rotate the whole event once.
                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                            spot_seed, rng_history, steps, cinel03_event_azimuth_dimension);
                        const float event_cos = sycl::cos(event_phi);
                        const float event_sin = sycl::sin(event_phi);

                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                            if (prod_offset + ip >= schneider_ct_device_ctx.primary_total_products) break;
                            const auto& product = schneider_ct_device_ctx.primary_products[prod_offset + ip];
                            record_neutron_birth(neutron_birth_buffer,product,
                                position_x_mm,position_y_mm,position_z_mm,
                                {direction_x,direction_y,direction_z},event_cos,event_sin,
                                global_history,rng_history,steps,ip,0,-1,
                                primary_atomic_number,primary_mass_number);

                            if (product.role == 2) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                unsupported_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                continue;
                            }
                            if (product.z <= 0 || product.a <= 0) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                neutral_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                continue;
                            }
                            if (product.role != 0) {
                                continue;
                            }
                            // Be6 keeps the frozen TopasCompatKill policy:
                            // explicit counter + energy, never queued.
                            if (product.z == 4 && product.a == 6) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryBe6Kills);
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::Be6TopasCompatKills);
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::Be6KillEnergy, ke);
                                continue;
                            }
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::PrimaryChargedBorn);

                            if (enable_secondary_transport && product.kinetic_energy_MeV > energy_cutoff_MeV) {
                                const auto local_direction = rotate_cinel03_event_azimuth(
                                    product.local_direction_x, product.local_direction_y,
                                    product.local_direction_z, event_cos, event_sin);
                                const auto child_direction = rotate_local_direction(
                                    local_direction.x,
                                    local_direction.y,
                                    local_direction.z,
                                    Direction3F{direction_x, direction_y, direction_z});

                                if (secondary_queue_device != nullptr) {
                                    auto count_ref = sycl::atomic_ref<
                                        uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(
                                        *secondary_count_device);
                                    const auto output = count_ref.fetch_add(1U);
                                    if (output < max_secondaries) {
                                        SecondaryParticle child{};
                                        child.z = product.z;
                                        child.a = product.a;
                                        child.energy_MeV = product.kinetic_energy_MeV;
                                        child.pos_x_mm = position_x_mm;
                                        child.pos_y_mm = position_y_mm;
                                        child.pos_z_mm = position_z_mm;
                                        child.dir_x = child_direction.x;
                                        child.dir_y = child_direction.y;
                                        child.dir_z = child_direction.z;
                                        child.weight = 1.0F;
                                        child.parent_history = global_history;
                                        child.rng_stream = rng::event_product_stream(
                                            rng_history, steps, rng::branch_role_primary_charged, ip);
                                        secondary_queue_device[output] = child;
                                        charged_accounted_MeV += product.kinetic_energy_MeV;
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryChargedQueued);
                                        if (cinel02_species_energy_device != nullptr) {
                                            cinel02_record_queued_secondary_birth_device(
                                                cinel02_species_energy_device, child.z, child.a,
                                                child.energy_MeV);
                                        }
                                    } else {
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryQueueOverflows);
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::QueueOverflows);
                                        if (secondary_overflow_count_device != nullptr) {
                                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_ov(*secondary_overflow_count_device);
                                            atomic_ov.fetch_add(1U);
                                        }
                                        if (secondary_overflow_energy_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_ove(*secondary_overflow_energy_device);
                                            atomic_ove.fetch_add(product.kinetic_energy_MeV);
                                        }
                                    }
                                }
                            } else {
                                pending_primary_depth_MeV += product.kinetic_energy_MeV;
                                if ((kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0) {
                                    pending_primary_voxel_MeV += product.kinetic_energy_MeV;
                                }
                                if ((!kProductionPrimaryPath &&
                                     enable_minibeam_component_voxel_scoring) &&
                                    voxel_index >= 0) {
                                    const auto child_category =
                                        minibeam_component_category(
                                            product.z, product.a,
                                            minibeam_birth_region_water);
                                    sycl::atomic_ref<DoseAtomicT,
                                                     sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>(
                                        minibeam_component_voxel_dose_device[
                                            child_category * number_of_voxels +
                                            voxel_index])
                                        .fetch_add(static_cast<DoseAtomicT>(
                                            product.kinetic_energy_MeV));
                                    sycl::atomic_ref<DoseAtomicT,
                                                     sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>(
                                        minibeam_component_voxel_dose_device[
                                            minibeam_primary_component_category *
                                                number_of_voxels + voxel_index])
                                        .fetch_add(static_cast<DoseAtomicT>(
                                            -product.kinetic_energy_MeV));
                                }
                                history_deposited_MeV += product.kinetic_energy_MeV;
                                grid_deposit_split_device(
                                    grid_deposited_in_device,
                                    grid_deposited_out_device,
                                    (kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0,
                                    product.kinetic_energy_MeV);
                                charged_accounted_MeV += product.kinetic_energy_MeV;
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedCutoffKills);
                            }
                        }

                        schneider_float_add_device(
                            schneider_float_device, SchneiderFloatSlot::ReactionQResidual,
                            sycl::fmax(0.0F, energy_MeV - local_deposit -
                                                  charged_accounted_MeV -
                                                  neutral_accounted_MeV -
                                                  unsupported_accounted_MeV));
                        // NO-DOUBLE-COUNT RULE: the legacy untracked sink below
                        // already contains neutral + unsupported + Q-residual
                        // energy (E - local - charged). The split float slots
                        // above are INFORMATIONAL ONLY and must never be added
                        // into the global closure alongside untracked; the
                        // closure in run_quality uses the legacy sink alone.
                        // A unit test pins this (split fields leave the
                        // residual bitwise unchanged).
                        const float untracked_MeV = sycl::fmax(0.0F, energy_MeV - local_deposit - charged_accounted_MeV);
                        if (untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_untracked(untracked_nuclear_device[global_history]);
                            atomic_untracked.fetch_add(untracked_MeV);
                        }

                        if (primary_terminal_counts_device != nullptr) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                term_ref(primary_terminal_counts_device[0]);
                            term_ref.fetch_add(1U);
                        }
                        energy_MeV = 0.0F;
                        break;
                        }  // else of the v3 masked-sampler empty-draw guard
                    }



                    ++steps;
                }


                if(CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit(unified_audit,unified_primary_audit);

                const bool inside_phantom = !unified_primary_escaped_ct &&
                    !water_urban_escaped_geometry &&
                    (position_z_mm >= 0.0F && position_z_mm < phantom_length_mm);

                if (energy_MeV > 0.0F && energy_MeV <= energy_cutoff_MeV && inside_phantom) {
                    const auto cutoff_energy_MeV = energy_MeV;
                    pending_primary_depth_MeV += cutoff_energy_MeV;
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                        pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                        pending_primary_voxel_MeV += cutoff_energy_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(cutoff_energy_MeV));
                        }
                    }
                    history_deposited_MeV += cutoff_energy_MeV;
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device,
                        (kProductionPrimaryPath || enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                            pending_primary_voxel <
                                static_cast<std::size_t>(number_of_voxels),
                        cutoff_energy_MeV);
                    if (cutoff_stopped_energy_device != nullptr) {
                        cutoff_stopped_energy_device[global_history] = cutoff_energy_MeV;
                    }
                    if (enable_let_scoring) {
                        const auto cutoff_numerator =
                            static_cast<double>(cutoff_energy_MeV) *
                            static_cast<double>(last_primary_stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(last_primary_density_g_per_cm3, 1.0e-6F));
                        const auto cutoff_denominator = static_cast<double>(cutoff_energy_MeV);
                        pending_let_numerator += cutoff_numerator;
                        pending_let_denominator += cutoff_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += cutoff_numerator;
                            pending_voxel_let_denominator += cutoff_denominator;
                        }
                    }
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[2]); // [2] = stopped_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                    energy_MeV = 0.0F;
                } else if (energy_MeV > 0.0F && !inside_phantom) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[1]); // [1] = escaped_ct_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                } else if (!primary_inelastic_occurred) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[3]); // [3] = other_terminal
                        term_ref.fetch_add(1U);
                    }
                    if (energy_MeV > 0.0F) {
                        if (other_terminal_energy_device != nullptr) {
                            other_terminal_energy_device[global_history] = energy_MeV;
                        }
                        energy_MeV = 0.0F;
                    }
                }

                if (pending_primary_depth_MeV > 0.0) {
                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
                }
                if (enable_let_scoring) {
                    flush_letd_moments_device(
                        let_moments_device, number_of_bins,
                        static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                        pending_let_numerator, pending_let_denominator, true);
                }
                if ((kProductionPrimaryPath || enable_voxel_scoring) && pending_primary_voxel >= 0 &&
                    pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                    if (pending_primary_voxel_MeV > 0.0) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                        atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        if ((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_primary_origin(
                                    charged_origin_voxel_dose_device[pending_primary_voxel]);
                            atomic_primary_origin.fetch_add(
                                static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        }
                        if ((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_primary_component(
                                    minibeam_component_voxel_dose_device[pending_primary_voxel]);
                            atomic_primary_component.fetch_add(
                                static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        }
                    }
                    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
                        flush_letd_moments_device(
                            voxel_let_moments_device, number_of_voxels,
                            pending_primary_voxel, nullptr, 0, 0,
                            pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                    }
                }

                schneider_diag_add_device(schneider_diag_device,
                                              SchneiderDiagSlot::PrimaryRateQueries,
                                              local_schneider_rate_queries);
                if(defer_material_packets)flush_electron_group();
                if constexpr (BroadBeam) {
                    if (grid_in_pending_MeV != 0.0 && grid_deposited_in_device != nullptr)
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*grid_deposited_in_device).fetch_add(grid_in_pending_MeV);
                    if (grid_out_pending_MeV != 0.0 && grid_deposited_out_device != nullptr)
                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*grid_deposited_out_device).fetch_add(grid_out_pending_MeV);
                }
                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV;
                steps_device[global_history] = steps;
                } // water stage (or original combined path)
            });
        };
        const auto dispatch_primary_chunk = [&]() {
            if constexpr(BroadBeam) {
                if constexpr(JointResponse==0)if(nonion_primary)
                    return launch_primary_chunk(std::false_type{},std::false_type{},std::integral_constant<int,7>{});
                if constexpr(JointResponse==3)if(broadbeam_ct_primary_specialize) {
                    if(primary_atomic_number==1 && primary_mass_number==1)
                        return launch_primary_chunk(std::false_type{},std::false_type{},std::integral_constant<int,13>{});
                    if(primary_atomic_number==6 && primary_mass_number==12)
                        return launch_primary_chunk(std::false_type{},std::false_type{},std::integral_constant<int,14>{});
                    return launch_primary_chunk(std::false_type{},std::false_type{},std::integral_constant<int,12>{});
                }
                return launch_primary_chunk(std::false_type{},std::false_type{},std::integral_constant<int,6>{});
            } else {
            {
                if (use_water_primary_path)
                    return launch_primary_chunk(std::false_type{}, std::true_type{}, std::integral_constant<int, 5>{});
                if (use_minibeam_primary_path || ct_minibeam_split) {
                    const auto launch_minibeam_stage = [&](auto stage) {
                        // The Cu/air stages share one compiled kernel for water
                        // and CT. Only downstream transport selects a material
                        // policy; never apply uniform-water substitutions to CT.
                        if constexpr (decltype(stage)::value == 4) {
                            if (ct_minibeam_split) {
                                if (ct_queued_primary_specialize)
                                    return launch_primary_chunk(std::false_type{}, std::false_type{}, std::integral_constant<int,11>{});
                                return launch_primary_chunk(std::false_type{}, std::false_type{}, stage);
                            }
                        }
                        return launch_primary_chunk(std::false_type{}, std::true_type{}, stage);
                    };
#if defined(CARBON_ENABLE_MINIBEAM)
                    if (beamline_order != nullptr) {
                        queue.fill(beamline_order_counts, 0U, 2);
                        queue.parallel_for(sycl::range<1>(chunk_count), [=](sycl::id<1> id) {
                            const auto lane = static_cast<std::uint32_t>(id[0]);
                            const std::uint64_t history = hist_offset + lane;
                            const auto local_x = flat_source_half_width_x_mm *
                                (2.0F * rng::uniform01(random_seed, history, 0, 34) - 1.0F);
                            const auto local_y = flat_source_half_width_y_mm *
                                (2.0F * rng::uniform01(random_seed, history, 0, 35) - 1.0F);
                            const auto direction = compose_tps_direction(
                                0.0F, 0.0F, 1.0F, beam_ux_x, beam_ux_y, beam_ux_z,
                                beam_uy_x, beam_uy_y, beam_uy_z, beam_uz_x, beam_uz_y, beam_uz_z);
                            const bool copper = minibeam_ray_hits_cylindrical_copper(
                                source_origin_x_mm + beam_ux_x * local_x + beam_uy_x * local_y,
                                source_origin_y_mm + beam_ux_y * local_x + beam_uy_y * local_y,
                                source_origin_z_mm + beam_ux_z * local_x + beam_uy_z * local_y,
                                direction.x, direction.y, direction.z,
                                minibeam_cos, minibeam_sin, minibeam_block_radius,
                                minibeam_block_center_z, minibeam_block_thickness,
                                minibeam_slit_count, minibeam_slit_width, minibeam_slit_pitch,
                                0.5F * minibeam_slit_length, minibeam_slit_thickness,
                                minibeam_slit_offset);
                            bool expensive=copper;

                            const auto rank = sycl::atomic_ref<std::uint32_t,
                                sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                    beamline_order_counts[expensive ? 0 : 1]).fetch_add(1U);
                            beamline_order[expensive ? rank : chunk_count - 1 - rank] = lane;
                        });
                    }
#endif
                    if (split_minibeam_primary == 2) {
                        queue.fill(primary_entry_counts, std::uint64_t{0}, 2);
                        primary_events.push_back(queue_copper_primary_children
                            ? launch_minibeam_stage(std::integral_constant<int,9>{})
                            : launch_minibeam_stage(std::integral_constant<int,3>{}));
#if defined(CARBON_ENABLE_MINIBEAM)
                        if (water_energy_order) {
                            queue.fill(water_energy_counts,0U,water_energy_buckets+1);
                            auto bucket=[=](float energy) {
                                if(energy<=energy_cutoff_MeV)return water_energy_buckets;
                                return sycl::min(water_energy_buckets-1,
                                    static_cast<unsigned>(sycl::fmax(0.F,energy)/initial_energy_MeV*water_energy_buckets));
                            };
                            water_order_events.push_back(queue.parallel_for(sycl::range<1>(chunk_count),[=](sycl::id<1> index){
                                const auto b=bucket(primary_medium_entry[index[0]].energy_MeV);
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(water_energy_counts[b]).fetch_add(1U);
                            }));
                            water_order_events.push_back(queue.single_task([=]{
                                unsigned offset=0;
                                for(unsigned b=0;b<=water_energy_buckets;++b){
                                    water_energy_offsets[b]=offset;offset+=water_energy_counts[b];water_energy_counts[b]=0;
                                }
                            }));
                            water_order_events.push_back(queue.parallel_for(sycl::range<1>(chunk_count),[=](sycl::id<1> index){
                                const auto b=bucket(primary_medium_entry[index[0]].energy_MeV);
                                const auto rank=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(water_energy_counts[b]).fetch_add(1U);
                                water_energy_order[water_energy_offsets[b]+rank]=index[0];
                            }));
                        }
#endif
                        return launch_minibeam_stage(std::integral_constant<int, 4>{});
                    }
                    if (split_minibeam_primary) {
                        primary_events.push_back(queue_copper_primary_children
                            ? launch_primary_chunk(std::false_type{},std::true_type{},std::integral_constant<int,8>{})
                            : launch_primary_chunk(std::false_type{},std::true_type{},std::integral_constant<int,1>{}));
                        return launch_primary_chunk(
                            std::false_type{}, std::true_type{}, std::integral_constant<int, 2>{});
                    }
                    return launch_primary_chunk(std::false_type{}, std::true_type{}, std::integral_constant<int, 0>{});
                }
            }
#if CARBON_PRIMARY_PRODUCTION_SPECIALIZE
            if (use_production_primary_path)
                return launch_primary_chunk(std::true_type{}, std::false_type{}, std::integral_constant<int, 0>{});
#endif
            return launch_primary_chunk(std::false_type{}, std::false_type{}, std::integral_constant<int, 0>{});
            }
        };
        auto kernel_event = dispatch_primary_chunk();
        primary_events.push_back(kernel_event);

        if (defer_material_packets) {
            std::uint64_t count_host=0;
            queue.copy(material_task_counts,&count_host,1).wait_and_throw();
            if(count_host>material_task_capacity)
                throw std::runtime_error("Electron response task queue overflow; batch rejected, no tasks discarded as accepted dose");
            electron_tasks_total+=count_host;
            electron_queue_peak=std::max<std::uint64_t>(electron_queue_peak,count_host);
            if(count_host) {
                const std::uint32_t lanes=static_cast<std::uint32_t>(std::min<std::uint64_t>(count_host,material_queue_lanes));
                if(electron_order) {
                    queue.fill(electron_order_counts,0U,electron_order_bins);
                    electron_events.push_back(queue.parallel_for(sycl::range<1>(count_host),[=](sycl::id<1> index){
                        namespace f=electron_fp32;
                        const auto t=material_tasks[index[0]];
                        unsigned bucket=32;
                        if(t.birth.source<float_birth.count) {
                            const auto node=float_birth.nodes[t.birth.source];
                            const auto e=f::sample_independent_electron_birth(node,float(t.birth.energy),f::Draw{t.birth.cdf_bits}).energy;
                            const unsigned eb=(e>.001f)+(e>.003f)+(e>.01f)+(e>.03f)+(e>.1f)+(e>.3f)+(e>.65f);
                            const unsigned db=(t.density>.01)+(t.density>.1)+(t.density>.5);
                            bucket=db*8+eb;
                        }
                        electron_order_keys[index[0]]=static_cast<unsigned char>(bucket);
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(electron_order_counts[bucket]).fetch_add(1U);
                    }));
                    electron_events.push_back(queue.single_task([=]{
                        std::uint32_t offset=0;
                        for(unsigned b=0;b<electron_order_bins;++b){
                            electron_order_offsets[b]=offset;offset+=electron_order_counts[b];electron_order_counts[b]=0;
                        }
                    }));
                    electron_events.push_back(queue.parallel_for(sycl::range<1>(count_host),[=](sycl::id<1> index){
                        const auto b=electron_order_keys[index[0]];
                        const auto rank=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(electron_order_counts[b]).fetch_add(1U);
                        electron_order[electron_order_offsets[b]+rank]=static_cast<std::uint32_t>(index[0]);
                    }));
                }


                    auto launch_float_response=[&](auto region_tag) {
                        constexpr bool region_diagnostics=decltype(region_tag)::value;
#include "detail/sycl_independent_electron_fp32.inc"
                    };
                    if constexpr(BroadBeam && JointResponse==3) {
                        if(broadbeam_region_work)launch_float_response(std::true_type{});
                        else launch_float_response(std::false_type{});
                    } else launch_float_response(std::false_type{});

            }
        }
    }
    if(broadbeam_region_work) {
        std::array<std::uint64_t,7*12> rows{};
        queue.copy(broadbeam_region_work,rows.data(),rows.size()).wait_and_throw();
        std::array<std::uint64_t,12> total{};
        const char* names[]={"tasks","charged_calls","elastic","hard_proposals","hard_collisions","faces","joint_calls","joint_advances","photon_calls","photon_interactions","weight_microMeV","completed"};
        for(unsigned r=0;r<7;++r) {
            std::cout<<"[electron-birth-region-work] region="<<r;
            for(unsigned k=0;k<12;++k){std::cout<<' '<<names[k]<<'='<<rows[12*r+k];total[k]+=rows[12*r+k];}
            std::cout<<'\n';
        }
        std::array<std::uint64_t,9> joint{};std::array<std::uint64_t,8> charged{},photon{};
        queue.copy(joint_pilot_work,joint.data(),9);queue.copy(independent_charged_work,charged.data(),8);queue.copy(independent_photon_work,photon.data(),8).wait_and_throw();
        if(total[0]!=electron_tasks_total || total[11]!=electron_tasks_total || total[1]!=charged[0] || total[2]!=charged[1] ||
           total[3]!=charged[2] || total[4]!=charged[3] || total[5]!=charged[4] || total[6]!=joint[0] || total[7]!=joint[2] ||
           total[8]!=photon[0] || total[9]!=photon[1] || rows[6*12])
            throw std::runtime_error("Electron birth-region diagnostic closure failed");
        std::cout<<"[electron-birth-region-audit] closure=pass attribution=queued_birth includes_descendants=1\n";
    }
    if(joint_pilot_work) {
        std::array<std::uint64_t,9> work{};queue.copy(joint_pilot_work,work.data(),9).wait_and_throw();
        sycl::free(joint_pilot_work,queue);
        const char* names[]={"eligible","started","advances","fallback","below","deposited","escaped","photon_gap","invalid"};
        std::cout<<"[electron-joint-pilot-work]";
        for(unsigned k=0;k<9;++k)std::cout<<" "<<names[k]<<"="<<work[k];
        std::cout<<"\n";
    }
    if(independent_charged_work) {
        std::array<std::uint64_t,8> work{};queue.copy(independent_charged_work,work.data(),8).wait_and_throw();
#ifdef CARBON_ELECTRON_STEP_DIAGNOSTICS
        if(float_step_diagnostics){
            std::vector<std::uint64_t> shards(64*1536);
            queue.copy(float_step_diagnostics,shards.data(),shards.size()).wait_and_throw();
            std::array<std::uint64_t,1536> bins{};std::uint64_t total=0;
            for(unsigned shard=0;shard<64;++shard)for(unsigned k=0;k<1536;++k)bins[k]+=shards[shard*1536+k];
            for(unsigned k=0;k<1536;++k)if(bins[k]){
                total+=bins[k];
                std::cout<<"[electron-step-diagnostic] energy_bin="<<k/256
                         <<" limiter_mask="<<(k/8)%32<<" event="<<k%8<<" count="<<bins[k]<<"\n";
            }
            std::cout<<"[electron-step-diagnostic-total] calls="<<work[0]<<" counted="<<total<<"\n";
            if(total!=work[0])throw std::runtime_error("Electron diagnostic histogram does not close");
            free_device(float_step_diagnostics);
        }
#endif
        const char* names[]={"calls","elastic","hard_proposals","hard_collisions","faces","stopped","coverage_missing","invalid"};
        std::cout<<"[independent-charged-work]";for(unsigned k=0;k<8;++k)std::cout<<" "<<names[k]<<"="<<work[k];std::cout<<"\n";
    }
    if(independent_photon_work) {
        std::array<std::uint64_t,8> work{};queue.copy(independent_photon_work,work.data(),8).wait_and_throw();
        const char* names[]={"calls","interactions","voxel_crossings","electron","deposited","escaped","coverage_missing","invalid"};
        std::cout<<"[independent-photon-work]";
        for(unsigned k=0;k<8;++k)std::cout<<" "<<names[k]<<"="<<work[k];
        std::cout<<"\n";
    }
    std::array<std::uint64_t,3> electron_work{};
#ifdef CARBON_ELECTRON_SEGMENT_DIAGNOSTICS
    if(float_segment_samples){
        namespace diagnostic=electron_segment_diagnostics;
        std::array<std::uint64_t,2> counts{};queue.copy(float_segment_counts,counts.data(),2).wait_and_throw();
        std::array<std::uint64_t,8> work{};queue.copy(independent_charged_work,work.data(),8).wait_and_throw();
        if(counts[0]!=work[0]||counts[1]>diagnostic::capacity)
            throw std::runtime_error("Segment diagnostic count closure or sample capacity failed");
        std::cout<<"[electron-segment-diagnostic] charged_total="<<counts[0]<<" selected="<<counts[1]
                 <<" denominator="<<diagnostic::selection_mask+1<<" capacity="<<diagnostic::capacity<<"\n";
        const auto path=config.voxel_dose_mhd_output_file.parent_path()/"electron_segment_samples.bin";
        if(std::filesystem::exists(path))throw std::runtime_error("Segment diagnostic refuses to overwrite: "+path.string());
        std::filesystem::create_directories(path.parent_path());
        std::vector<diagnostic::Sample> samples(counts[1]);queue.copy(float_segment_samples,samples.data(),samples.size()).wait_and_throw();
        std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(samples.data()),samples.size()*sizeof(diagnostic::Sample));
        if(!file)throw std::runtime_error("Segment diagnostic write failed");
        free_device(float_segment_samples);free_device(float_segment_counts);
    }
#endif
#ifdef CARBON_ELECTRON_RANGE_DIAGNOSTICS
    if(float_range_delta){
        std::array<std::uint64_t,10> counts{};queue.copy(float_range_counts,counts.data(),counts.size()).wait_and_throw();
        const char* names[]={"eligible","charged_after_guard","photon_calls_after_guard","joint_calls_after_guard",
            "different_deposit","escaped_after_guard","same_deposit","invalid_after_guard","charged_total","tasks_total"};
        std::cout<<"[electron-range-diagnostic] factor=1.5";
        for(unsigned k=0;k<10;++k)std::cout<<" "<<names[k]<<"="<<counts[k];std::cout<<"\n";
        const auto directory=config.voxel_dose_mhd_output_file.parent_path();std::filesystem::create_directories(directory);
        auto write=[&](const char* name,const float* device){
            const auto path=directory/name;
            if(std::filesystem::exists(path))throw std::runtime_error("Range diagnostic refuses to overwrite: "+path.string());
            std::vector<float> values(number_of_voxels);queue.copy(device,values.data(),values.size()).wait_and_throw();
            std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
            if(!file)throw std::runtime_error("Range diagnostic write failed");
        };
        write("range_guard_delta_MeV.f32",float_range_delta);write("range_guard_density.f32",ct_density_device);
        free_device(float_range_delta);free_device(float_range_counts);
    }
#endif
    if (defer_material_packets) {
        queue.copy(material_queue_work,electron_work.data(),3).wait_and_throw();
        free_device(material_queue_work);
        queue.wait_and_throw();free_device(material_tasks);free_device(material_task_counts);
        free_device(electron_order);free_device(electron_order_keys);free_device(electron_order_counts);free_device(electron_order_offsets);
    }

    double electron_kernel_seconds=0;
    for(const auto& event:electron_events)electron_kernel_seconds+=event_duration_seconds(event);
    if(defer_material_packets)
        std::cout<<"[electron-queue-time] seconds="<<electron_kernel_seconds
                 <<" tasks="<<electron_tasks_total<<" peak_tasks="<<electron_queue_peak
                 <<" advances="<<electron_work[0]<<" source_restarts="<<electron_work[1]
                 <<" boundary_restarts="<<electron_work[2]<<"\n";
    // The queue is in order, so all primary chunks retain exactly the previous
    // execution order.  Waiting once on the final event avoids a host/device
    // round trip and a flushed progress line between every chunk.
    if (!primary_events.empty()) {
        primary_events.back().wait_and_throw();
        double split_beamline_seconds = 0.0, split_water_seconds = 0.0;
        for (std::size_t i=0; i<primary_events.size(); ++i) {
            const auto seconds = event_duration_seconds(primary_events[i]);
            primary_kernel_seconds += seconds;
            if (split_minibeam_primary) {
                if (i%2 == 0) split_beamline_seconds += seconds;
                else split_water_seconds += seconds;
            }
        }
        if (split_minibeam_primary)
            std::cout << "[primary-split-time] beamline_s=" << split_beamline_seconds
                      << " water_s=" << split_water_seconds << "\n";
        std::cout << "[progress] primary batches completed: "
                  << number_of_histories << "/" << number_of_histories
                  << " histories (" << primary_events.size() << " launches)\n";
    }

    if (primary_medium_entry) mem_tracker.free(primary_medium_entry);
    if (primary_entry_counts) mem_tracker.free(primary_entry_counts);
#if defined(CARBON_ENABLE_MINIBEAM)
    double water_order_seconds=0;
    for(const auto& event:water_order_events)water_order_seconds+=event_duration_seconds(event);
    if(water_energy_order)std::cout << "[water-energy-order] seconds=" << water_order_seconds << '\n';
    if(water_energy_order)mem_tracker.free(water_energy_order);
    if(water_energy_counts)mem_tracker.free(water_energy_counts);
    if(water_energy_offsets)mem_tracker.free(water_energy_offsets);
    if (beamline_order) mem_tracker.free(beamline_order);
    if (beamline_order_counts) mem_tracker.free(beamline_order_counts);

#endif

#if defined(CARBON_ENABLE_MINIBEAM)
    double copper_primary_child_seconds=0;
    if(copper_primary_child_tasks) {
        std::uint32_t count=0;
        queue.copy(copper_primary_child_counts,&count,1).wait_and_throw();
        if(count>copper_primary_child_capacity)
            throw std::runtime_error("Cu primary-child queue overflow; batch rejected without truncation");
        if(count) {
            const auto lanes=std::min<std::size_t>(count,17408);
            auto event=queue.parallel_for(sycl::nd_range<1>{sycl::range<1>{((lanes+127)/128)*128},sycl::range<1>{128}},
                [=](sycl::nd_item<1>) {
                    for(;;) {
                        const auto slot=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                            (copper_primary_child_counts[1]).fetch_add(1U);
                        if(slot>=count)break;
                        const auto task=copper_primary_child_tasks[slot];
                        const auto& product=minibeam_copper_products_device[task.flat_product_index];
                        const auto product_index=task.product_index;
                        const auto global_history=task.global_history;
                        const auto rng_history=task.rng_history;
                        const auto spot_seed=task.spot_seed;
                        const auto beamline_step=task.beamline_step;
                        const auto position_x_mm=task.x,position_y_mm=task.y,position_z_mm=task.z;
                        const auto direction_x=task.dx,direction_y=task.dy,direction_z=task.dz;
                        const auto event_cos=task.event_cos,event_sin=task.event_sin;
                        const auto block_exit_z=minibeam_block_center_z+.5F*minibeam_block_thickness;
                        // Started tasks are all finished when this event completes,
                        // including the original stop/backward early-continue paths.
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                            (copper_primary_child_counts[2]).fetch_add(1U);
                        float queued_energy=0;
#include "detail/sycl_copper_primary_child.inc"
                        if(queued_energy>0)
                            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(beamline_removed_device[global_history]).fetch_sub(queued_energy);
                    }
                });
            event.wait_and_throw();copper_primary_child_seconds=event_duration_seconds(event);
        }
        std::uint32_t completed=0;
        queue.copy(copper_primary_child_counts+2,&completed,1).wait_and_throw();
        if(completed!=count)throw std::runtime_error("Cu primary-child queue drain mismatch");
        std::cout << "[copper-primary-child-queue] seconds=" << copper_primary_child_seconds
                  << " tasks=" << count << " completed=" << completed << " overflow=0\n";
        free_device(copper_primary_child_tasks);free_device(copper_primary_child_counts);
    }
    // Process-faithful fragment+Cu products are transported through the
    // remaining collimator in generation batches. Copper generation lives in
    // this dedicated queue and never consumes the downstream water budget.
    if (minibeam_copper_cascade_count_device != nullptr) {
        std::uint32_t copper_cascade_count = 0;
        queue.copy(minibeam_copper_cascade_count_device,
                   &copper_cascade_count, 1).wait_and_throw();
        copper_cascade_count = std::min<std::uint32_t>(
            copper_cascade_count, static_cast<std::uint32_t>(max_secondaries));
        std::uint32_t copper_generation_begin = 0;
        std::uint32_t copper_generation_end = copper_cascade_count;
        while (copper_generation_begin < copper_generation_end) {
            const auto generation_begin = copper_generation_begin;
            const auto generation_size = copper_generation_end - generation_begin;
            queue.parallel_for(sycl::range<1>(generation_size),
                [=](sycl::id<1> item) {
                    const auto cascade =
                        minibeam_copper_cascade_queue_device[
                            generation_begin + item[0]];
                    const auto species_category =
                        cascade.z == 6 ? 0U : cascade.z == 5 ? 1U :
                        cascade.z == 4 ? 2U : cascade.z == 3 ? 3U :
                        cascade.z == 2 ? 4U :
                        (cascade.z == 1 && cascade.a == 1) ? 5U :
                        (cascade.z == 1 && cascade.a == 2) ? 6U :
                        (cascade.z == 1 && cascade.a == 3) ? 7U : 8U;
                    auto energy = cascade.energy_MeV;
                    auto x = cascade.pos_x_mm;
                    auto y = cascade.pos_y_mm;
                    auto z = cascade.pos_z_mm;
                    Direction3F direction{
                        cascade.dir_x, cascade.dir_y, cascade.dir_z};
                    auto copper_segment_path = 0.0F;
                    auto ignored_nuclear_tau = 0.0F;
                    auto nuclear_tau_remaining = 0.0F;
                    auto nuclear_tau_active = false;
                    auto collided = false;
                    std::uint32_t step = 0;
                    const auto block_exit_z = minibeam_block_center_z +
                        0.5F * minibeam_block_thickness;
                    const auto terminal_copper_generation =
                        cascade.generation >=
                            minibeam_copper_fragment_cascade_generations;
                    const auto reaction_generation_index = sycl::min(
                        2U, static_cast<std::uint32_t>(cascade.generation));
                    while (energy > energy_cutoff_MeV &&
                           direction.z > 1.0e-8F &&
                           z < block_exit_z - 1.0e-6F &&
                           step < 100000U) {
                        const auto nominal_path = sycl::fmin(
                            minibeam_copper_max_step,
                            (block_exit_z - z) / direction.z);
                        const auto boundary_path =
                            minibeam_copper_exact_material_boundaries
                                ? minibeam_path_to_material_boundary(
                                      x, y, direction.x, direction.y,
                                      minibeam_cos, minibeam_sin,
                                      minibeam_block_radius,
                                      minibeam_slit_count,
                                      minibeam_slit_width,
                                      minibeam_slit_pitch,
                                      0.5F * minibeam_slit_length,
                                      minibeam_slit_offset,
                                      nominal_path)
                                : nominal_path;
                        const auto mid_x = x + 0.5F * boundary_path * direction.x;
                        const auto mid_y = y + 0.5F * boundary_path * direction.y;
                        const auto in_copper = minibeam_point_in_copper(
                            mid_x, mid_y, minibeam_cos, minibeam_sin,
                            minibeam_block_radius, minibeam_slit_count,
                            minibeam_slit_width, minibeam_slit_pitch,
                            0.5F * minibeam_slit_length,
                            minibeam_slit_offset);
                        auto inelastic_rate = 0.0F;
                        auto collision_in_step = false;
                        auto path = boundary_path;
                        if (in_copper) {
                            inelastic_rate = minibeam_copper_ion_inelastic_rate(
                                minibeam_copper_ion_xs_device,
                                minibeam_copper_ion_xs_present_device,
                                minibeam_copper_ion_xs_grid_size,
                                minibeam_copper_ion_xs_minimum_energy,
                                minibeam_copper_ion_xs_inverse_step,
                                energy, cascade.z, cascade.a);
                            if (terminal_copper_generation) {
                                ignored_nuclear_tau +=
                                    inelastic_rate * boundary_path;
                            } else if (inelastic_rate > 0.0F) {
                                if (!nuclear_tau_active) {
                                    const auto optical_uniform = sycl::fmax(
                                        1.0e-7F,
                                        rng::uniform01(random_seed,
                                                       cascade.rng_stream,
                                                       step, 2));
                                    nuclear_tau_remaining =
                                        -sycl::log(optical_uniform);
                                    nuclear_tau_active = true;
                                }
                                collision_in_step =
                                    nuclear_tau_remaining <=
                                        inelastic_rate * boundary_path;
                                if (collision_in_step) {
                                    path = nuclear_tau_remaining / inelastic_rate;
                                }
                            }
                        }
                        x += path * direction.x;
                        y += path * direction.y;
                        z += path * direction.z;
                        if (in_copper) {
                            const auto stopping = minibeam_copper_ion_stopping(
                                minibeam_copper_sp_energies_device,
                                minibeam_copper_sp_values_device,
                                minibeam_copper_sp_count,
                                minibeam_copper_ion_sp_ratios_device,
                                minibeam_copper_ion_sp_present_device,
                                energy, cascade.z, cascade.a, primary_atomic_number);
                            const auto predicted_loss = stopping * path;
                            const auto midpoint_energy = sycl::fmax(
                                0.0F, energy - 0.5F * predicted_loss);
                            const auto midpoint_stopping =
                                minibeam_copper_ion_stopping(
                                    minibeam_copper_sp_energies_device,
                                    minibeam_copper_sp_values_device,
                                    minibeam_copper_sp_count,
                                    minibeam_copper_ion_sp_ratios_device,
                                    minibeam_copper_ion_sp_present_device,
                                    midpoint_energy, cascade.z, cascade.a, primary_atomic_number);
                            const auto mean_loss = midpoint_stopping * path;
                            auto loss = mean_loss;
                            if (minibeam_copper_fragment_enable_energy_straggling) {
                                constexpr float copper_z_over_a_rel_water =
                                    (29.0F / 63.546F) / 0.55509F;
                                const auto energy_u = midpoint_energy /
                                    static_cast<float>(cascade.a);
                                const auto effective_charge =
                                    ion_effective_charge_device(cascade.z, energy_u);
                                const auto variance =
                                    condensed_total_loss_variance_MeV2_device(
                                        energy_u, cascade.a, effective_charge,
                                        path, minibeam_copper_density,
                                        copper_z_over_a_rel_water);
                                const auto gaussian_u0 = sycl::fmax(
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 10),
                                    1.0e-12F);
                                const auto gaussian_u1 = rng::uniform01(
                                    random_seed, cascade.rng_stream, step, 11);
                                const auto gaussian = sycl::sqrt(
                                    -2.0F * sycl::log(gaussian_u0)) *
                                    sycl::cos(6.2831853071795864769F * gaussian_u1);
                                loss = sycl::fmax(
                                    0.0F,
                                    mean_loss +
                                        minibeam_copper_fragment_straggling_scale *
                                            sycl::sqrt(sycl::fmax(0.0F, variance)) *
                                            gaussian);
                            }
                            if (minibeam_copper_fragment_enable_energy_straggling) {
                                if (copper_proton_native_fluctuation && cascade.z == 1 && cascade.a == 1) {
                                    std::uint64_t draw_index = 0;
                                    auto uniform = [&]() { return (rng::random_u32(
                                        random_seed, cascade.rng_stream, (std::uint64_t(step) << 32) | draw_index++, 124) >> 8) * 0x1p-24f; };
                                    const auto draw = copper_proton_total_fluctuation(
                                        energy, 938.272013F, mean_loss, path, minibeam_copper_density, uniform, copper_proton_finite_cut);
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                        minibeam_event_counts_device[draw.valid ? copper_proton_fragment_loss_slot : copper_proton_loss_failure_slot]).fetch_add(1U);
                                    if (!draw.valid) break; // host rejects the entire run
                                    loss = draw.loss;
                                }
                            }
                            if (loss >= energy - energy_cutoff_MeV) {
                                energy = 0.0F;
                                break;
                            }
                            const auto scatter_energy = energy - 0.5F * loss;
                            energy -= loss;
                            if (minibeam_copper_enable_mcs) {
                                const auto previous_path = copper_segment_path;
                                copper_segment_path += path;
                                const auto total_rms =
                                    highland_projected_rms_angle_device(
                                        scatter_energy, cascade.z, cascade.a,
                                        copper_segment_path,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length);
                                const auto previous_rms =
                                    highland_projected_rms_angle_device(
                                        scatter_energy, cascade.z, cascade.a,
                                        previous_path,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length);
                                direction = scatter_direction(
                                    direction,
                                    minibeam_copper_fragment_mcs_scale *
                                        sycl::sqrt(sycl::fmax(
                                            0.0F, total_rms * total_rms -
                                                      previous_rms * previous_rms)),
                                    random_seed, cascade.rng_stream, step, 0);
                            }
                            if (!terminal_copper_generation &&
                                !collision_in_step && inelastic_rate > 0.0F &&
                                nuclear_tau_active) {
                                nuclear_tau_remaining -= inelastic_rate * path;
                            }
                            if (collision_in_step) {
                                collided = true;
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    reaction_count(minibeam_event_counts_device[
                                        minibeam_fragment_cascade_interactions_slot]);
                                reaction_count.fetch_add(1U);
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    generation_reaction_count(
                                        minibeam_event_counts_device[
                                            minibeam_fragment_generation_interaction_slot +
                                            reaction_generation_index]);
                                generation_reaction_count.fetch_add(1U);
                                const auto lookup = cinel03_lookup_event_device(
                                    minibeam_copper_nodes_device,
                                    minibeam_copper_node_count,
                                    minibeam_copper_offsets_device,
                                    minibeam_copper_indices_device,
                                    minibeam_copper_event_count,
                                    cascade.z, cascade.a, 29,
                                    energy / static_cast<float>(cascade.a),
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 20),
                                    rng::uniform01(random_seed,
                                                   cascade.rng_stream,
                                                   step, 21));
                                if (lookup.status == Cinel03LookupStatus::Hit) {
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        hit_count(minibeam_event_counts_device[
                                            minibeam_fragment_cascade_hits_slot]);
                                    hit_count.fetch_add(1U);
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generation_hit_count(
                                            minibeam_event_counts_device[
                                                minibeam_fragment_generation_hit_slot +
                                                reaction_generation_index]);
                                    generation_hit_count.fetch_add(1U);
                                    const auto& event =
                                        minibeam_copper_interactions_device[
                                            lookup.event_index];
                                    const auto phi = 6.2831853071795864769F *
                                        rng::uniform01(random_seed,
                                                       cascade.rng_stream,
                                                       step, 22);
                                    const auto event_cos = sycl::cos(phi);
                                    const auto event_sin = sycl::sin(phi);
                                    auto accounted_charged = 0.0F;
                                    auto replay_product_energy = 0.0F;
                                    auto replay_product_rest_mass = 0.0F;
                                    std::int32_t replay_product_a = 0;
                                    for (std::uint32_t product_index = 0;
                                         product_index < event.direct_product_count;
                                         ++product_index) {
                                        const auto flat_index =
                                            event.product_offset + product_index;
                                        if (flat_index >=
                                            minibeam_copper_product_count) break;
                                        const auto& product =
                                            minibeam_copper_products_device[flat_index];
                                        record_neutron_birth(neutron_birth_buffer, product,
                                            x,y,z,direction,event_cos,event_sin,cascade.parent_history,
                                            cascade.rng_stream,step,product_index,cascade.generation+1U,
                                            -2,cascade.z,cascade.a,cascade.weight);
                                        replay_product_energy += sycl::fmax(
                                            0.0F, product.kinetic_energy_MeV);
                                        replay_product_rest_mass += sycl::fmax(
                                            0.0F, product.rest_mass);
                                        replay_product_a += sycl::max(
                                            0, static_cast<int>(product.a));
                                        const bool supported_charged =
                                            (product.role == 0 || product.role == 1) &&
                                            product.z > 0 && product.a > 0;
                                        const auto count_slot = supported_charged
                                            ? minibeam_fragment_cascade_charged_slot
                                            : (product.z <= 0
                                                   ? minibeam_fragment_cascade_neutral_slot
                                                   : minibeam_fragment_cascade_unsupported_slot);
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            product_count(minibeam_event_counts_device[
                                                count_slot]);
                                        product_count.fetch_add(1U);
                                        if (!supported_charged ||
                                            product.kinetic_energy_MeV <=
                                                energy_cutoff_MeV) continue;
                                        accounted_charged +=
                                            product.kinetic_energy_MeV;
                                        const auto local =
                                            rotate_cinel03_event_azimuth(
                                                product.local_direction_x,
                                                product.local_direction_y,
                                                product.local_direction_z,
                                                event_cos, event_sin);
                                        const auto child_direction =
                                            rotate_local_direction(
                                                local.x, local.y, local.z,
                                                direction);
                                        sycl::atomic_ref<
                                            std::uint32_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            queue_count(
                                                *minibeam_copper_cascade_count_device);
                                        const auto output = queue_count.fetch_add(1U);
                                        if (output < max_secondaries) {
                                            SecondaryParticle child{};
                                            child.z = product.z;
                                            child.a = product.a;
                                            child.energy_MeV =
                                                product.kinetic_energy_MeV;
                                            child.pos_x_mm = x;
                                            child.pos_y_mm = y;
                                            child.pos_z_mm = z;
                                            child.dir_x = child_direction.x;
                                            child.dir_y = child_direction.y;
                                            child.dir_z = child_direction.z;
                                            child.weight = 1.0F;
                                            child.parent_history =
                                                cascade.parent_history;
                                            child.rng_stream =
                                                rng::event_product_stream(
                                                    cascade.rng_stream, step,
                                                    rng::branch_role_cascade_charged,
                                                    product_index);
                                            child.generation = static_cast<std::uint16_t>(
                                                cascade.generation + 1U);
                                            child.birth_region =
                                                minibeam_birth_region_copper;
                                            minibeam_copper_cascade_queue_device[
                                                output] = child;
                                        } else {
                                            sycl::atomic_ref<
                                                std::uint64_t,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                overflow_count(
                                                    minibeam_event_counts_device[
                                                        minibeam_fragment_cascade_overflow_slot]);
                                            overflow_count.fetch_add(1U);
                                        }
                                    }
                                    const auto add_fixed =
                                        [&](const std::size_t slot,
                                            const float value) {
                                            sycl::atomic_ref<
                                                std::uint64_t,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                counter(minibeam_event_counts_device[slot]);
                                            counter.fetch_add(
                                                static_cast<std::uint64_t>(
                                                    sycl::fmax(0.0F, value) *
                                                        1000.0F + 0.5F));
                                        };
                                    add_fixed(
                                        minibeam_fragment_cascade_local_keV_slot,
                                        event.process_local_deposit_MeV);
                                    add_fixed(
                                        minibeam_fragment_cascade_untracked_keV_slot,
                                        sycl::fmax(
                                            0.0F,
                                            energy - event.process_local_deposit_MeV -
                                                accounted_charged));
                                    const auto selected_input =
                                        lookup.selected_energy_MeV_per_u *
                                        static_cast<float>(cascade.a);
                                    const auto replay_output = sycl::fmax(
                                        0.0F, event.parent_energy_MeV) +
                                        sycl::fmax(
                                            0.0F,
                                            event.process_local_deposit_MeV) +
                                        replay_product_energy;
                                    add_fixed(
                                        minibeam_fragment_actual_input_keV_slot,
                                        energy);
                                    add_fixed(
                                        minibeam_fragment_selected_input_keV_slot,
                                        selected_input);
                                    add_fixed(
                                        minibeam_fragment_replay_output_keV_slot,
                                        replay_output);
                                    add_fixed(
                                        minibeam_fragment_selection_mismatch_keV_slot,
                                        sycl::fabs(energy - selected_input));
                                    add_fixed(
                                        minibeam_fragment_closure_mismatch_keV_slot,
                                        sycl::fabs(selected_input - replay_output));
                                    const bool parent_survives =
                                        event.parent_energy_MeV > 0.0F;
                                    const auto copper_target_mass =
                                        event.target_a == 63
                                            ? 58603.7301743F
                                            : (event.target_a == 65
                                                   ? 60465.0342192F
                                                   : sycl::fmax(
                                                         0.0F,
                                                         static_cast<float>(
                                                             event.target_a) *
                                                                 931.49410242F -
                                                             29.0F * 0.51099895F));
                                    add_fixed(
                                        minibeam_fragment_mass_energy_mismatch_keV_slot,
                                        sycl::fabs(
                                            selected_input + event.parent_rest_mass +
                                                copper_target_mass -
                                            (replay_output +
                                             replay_product_rest_mass +
                                             (parent_survives
                                                  ? event.parent_rest_mass
                                                  : 0.0F))));
                                    const auto input_a =
                                        static_cast<int>(cascade.a) +
                                        static_cast<int>(event.target_a);
                                    const auto output_a = replay_product_a +
                                        (parent_survives
                                             ? static_cast<int>(event.parent_a)
                                             : 0);
                                    if (input_a != output_a) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            baryon_miss(minibeam_event_counts_device[
                                                minibeam_fragment_baryon_mismatch_slot]);
                                        baryon_miss.fetch_add(1U);
                                    }
                                } else {
                                    const auto miss_index =
                                        static_cast<std::size_t>(lookup.status) - 1U;
                                    if (miss_index < 6U) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                        miss_count(minibeam_event_counts_device[
                                            minibeam_fragment_cascade_miss_slot +
                                            miss_index]);
                                        miss_count.fetch_add(1U);
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generation_miss_count(
                                            minibeam_event_counts_device[
                                                minibeam_fragment_generation_miss_slot +
                                                reaction_generation_index]);
                                    generation_miss_count.fetch_add(1U);
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        species_miss(minibeam_event_counts_device[
                                            minibeam_fragment_miss_species_slot +
                                            species_category]);
                                    species_miss.fetch_add(1U);
                                    const auto miss_energy_bin = sycl::min(
                                        15U,
                                        static_cast<std::uint32_t>(
                                            sycl::fmax(
                                                0.0F,
                                                energy /
                                                    static_cast<float>(cascade.a)) /
                                            25.0F));
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        energy_miss(minibeam_event_counts_device[
                                            minibeam_fragment_miss_energy_slot +
                                            miss_energy_bin]);
                                    energy_miss.fetch_add(1U);
                                    if (minibeam_fragment_miss_joint_counts_device != nullptr &&
                                        miss_index < MinibeamDiagnostics::fragment_miss_reason_count) {
                                        const auto za_index =
                                            cascade.z >= 0 && cascade.z <= 6 &&
                                                    cascade.a >= 0 && cascade.a <= 12
                                                ? static_cast<std::size_t>(cascade.z) * 13U +
                                                      static_cast<std::size_t>(cascade.a)
                                                : MinibeamDiagnostics::fragment_miss_za_count - 1U;
                                        const auto energy_u = energy /
                                            static_cast<float>(cascade.a);
                                        const auto joint_energy_bin = sycl::min(
                                            127U,
                                            static_cast<std::uint32_t>(
                                                sycl::fmax(0.0F, energy_u) / 5.0F));
                                        const auto joint_index =
                                            (((static_cast<std::size_t>(
                                                   reaction_generation_index) *
                                                   MinibeamDiagnostics::fragment_miss_za_count +
                                               za_index) *
                                                  MinibeamDiagnostics::fragment_miss_reason_count +
                                              miss_index) *
                                                 MinibeamDiagnostics::fragment_miss_joint_energy_bin_count) +
                                            joint_energy_bin;
                                        const auto block_entry_z =
                                            minibeam_block_center_z -
                                            0.5F * minibeam_block_thickness;
                                        const auto collision_depth_um =
                                            static_cast<std::uint64_t>(sycl::fmax(
                                                0.0F, z - block_entry_z) * 1000.0F + 0.5F);
                                        const auto path_to_plane = direction.z > 1.0e-8F
                                            ? sycl::fmax(0.0F,
                                                  (block_exit_z - z) / direction.z)
                                            : 0.0F;
                                        const auto remaining_um =
                                            static_cast<std::uint64_t>(
                                                minibeam_straight_copper_path_to_plane(
                                                    x, y, direction.x, direction.y,
                                                    minibeam_cos, minibeam_sin,
                                                    minibeam_block_radius,
                                                    minibeam_slit_count,
                                                    minibeam_slit_width,
                                                    minibeam_slit_pitch,
                                                    0.5F * minibeam_slit_length,
                                                    minibeam_slit_offset,
                                                    path_to_plane) *
                                                    1000.0F +
                                                0.5F);
                                        const auto energy_keV =
                                            static_cast<std::uint64_t>(
                                                sycl::fmax(0.0F, energy) * 1000.0F +
                                                0.5F);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_counts_device[joint_index])
                                            .fetch_add(1U);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_energy_device[joint_index])
                                            .fetch_add(energy_keV);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_depth_device[joint_index])
                                            .fetch_add(collision_depth_um);
                                        sycl::atomic_ref<std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(
                                                minibeam_fragment_miss_joint_remaining_device[joint_index])
                                            .fetch_add(remaining_um);
                                    }
                                }
                                energy = 0.0F;
                                break;
                            }
                        } else {
                            copper_segment_path = 0.0F;
                        }
                        ++step;
                    }
                    if (terminal_copper_generation) {
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            terminal_tracks(minibeam_event_counts_device[
                                minibeam_fragment_terminal_track_slot +
                                species_category]);
                        terminal_tracks.fetch_add(1U);
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ignored_tau(minibeam_event_counts_device[
                                minibeam_fragment_ignored_tau_micro_slot +
                                species_category]);
                        ignored_tau.fetch_add(static_cast<std::uint64_t>(
                            sycl::fmax(0.0F, ignored_nuclear_tau) * 1.0e6F +
                            0.5F));
                        sycl::atomic_ref<
                            std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ignored_probability(minibeam_event_counts_device[
                                minibeam_fragment_ignored_probability_micro_slot +
                                species_category]);
                        ignored_probability.fetch_add(
                            static_cast<std::uint64_t>(
                                (1.0F - sycl::exp(-sycl::fmax(
                                            0.0F, ignored_nuclear_tau))) *
                                    1.0e6F +
                                0.5F));
                    }
                    if (collided) return;
                    if (energy <= energy_cutoff_MeV ||
                        direction.z <= 1.0e-8F || step == 100000U) return;
                    const auto to_water = -z / direction.z;
                    if (to_water < 0.0F) return;
                    x += to_water * direction.x;
                    y += to_water * direction.y;

                    sycl::atomic_ref<
                        std::uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        secondary_count(*secondary_count_device);
                    const auto output = secondary_count.fetch_add(1U);
                    if (output >= max_secondaries) {
                        sycl::atomic_ref<
                            std::uint32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            overflow(*secondary_overflow_count_device);
                        overflow.fetch_add(1U);
                        sycl::atomic_ref<
                            float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            overflow_energy(*secondary_overflow_energy_device);
                        overflow_energy.fetch_add(energy);
                        return;
                    }
                    SecondaryParticle water_child = cascade;
                    water_child.energy_MeV = energy;
                    water_child.pos_x_mm = x;
                    water_child.pos_y_mm = y;
                    water_child.pos_z_mm = 0.0F;
                    water_child.dir_x = direction.x;
                    water_child.dir_y = direction.y;
                    water_child.dir_z = direction.z;
                    // Copper and water use independent generation budgets.
                    // The dedicated Copper queue already enforced its cap;
                    // every survivor starts water at generation zero.
                    water_child.generation = 0U;
                    secondary_queue_device[output] = water_child;

                    // The primary kernel initially booked the complete
                    // terminal fragment energy as beamline-removed.  Restore
                    // exactly the kinetic energy that now reaches water.
                    sycl::atomic_ref<
                        float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        removed_energy(
                            beamline_removed_device[cascade.parent_history]);
                    removed_energy.fetch_sub(energy);

                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        survivor_count(minibeam_event_counts_device[2]);
                    survivor_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        species_count(
                            minibeam_event_counts_device[3U + species_category]);
                    species_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        species_energy(minibeam_event_counts_device[
                            12U + species_category]);
                    species_energy.fetch_add(static_cast<std::uint64_t>(
                        sycl::fmax(0.0F, energy) * 1000.0F + 0.5F));

                    if (minibeam_fragment_phase_space_device != nullptr) {
                        sycl::atomic_ref<
                            std::uint32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            phase_count(
                                *minibeam_fragment_phase_space_count_device);
                        const auto phase_index = phase_count.fetch_add(1U);
                        if (phase_index < max_secondaries) {
                            MinibeamFragmentPhaseSpaceRecord record{};
                            record.history = cascade.parent_history;
                            record.atomic_number = cascade.z;
                            record.mass_number = cascade.a;
                            record.kinetic_energy_MeV = energy;
                            record.x_mm = x;
                            record.y_mm = y;
                            record.direction_x = direction.x;
                            record.direction_y = direction.y;
                            record.direction_z = direction.z;
                            minibeam_fragment_phase_space_device[phase_index] =
                                record;
                        }
                    }
                }).wait_and_throw();
            copper_generation_begin = copper_generation_end;
            std::uint32_t next_end = 0;
            queue.copy(minibeam_copper_cascade_count_device, &next_end, 1)
                .wait_and_throw();
            copper_generation_end = std::min<std::uint32_t>(
                next_end, static_cast<std::uint32_t>(max_secondaries));
        }
        std::cout << "[minibeam-copper-cascade] generation_limit="
                  << minibeam_copper_fragment_cascade_generations
                  << " initial/final-queued=" << copper_cascade_count << '/'
                  << copper_generation_end << '\n';
    }
#endif

    double secondary_kernel_seconds = 0.0;
    const bool segment_secondaries = config.secondary_step_chunking;
    constexpr unsigned secondary_tail_threshold = 8192U;
    const bool secondary_tail_diag =
        std::getenv("CARBON_SECONDARY_TAIL_DIAG") != nullptr;
    // Reorder indices only: RNG streams and parent histories belong to particles.
    const bool group_secondaries = config.secondary_species_grouping &&
        enable_inelastic && enable_secondary_transport;
    std::uint32_t* secondary_order = nullptr;
    std::uint32_t* secondary_group_counts = nullptr;
    std::uint32_t* secondary_group_cursors = nullptr;
    double secondary_group_seconds = 0.0;
    if (group_secondaries && secondary_queue_device) {
        secondary_order = mem_tracker.allocate<std::uint32_t>(max_secondaries);
        secondary_group_counts =
            mem_tracker.allocate<std::uint32_t>(kSecondaryGroupBucketCount);
        secondary_group_cursors =
            mem_tracker.allocate<std::uint32_t>(kSecondaryGroupBucketCount);
        if (!secondary_order || !secondary_group_counts || !secondary_group_cursors)
            throw std::bad_alloc();
    }
    std::cout << "[secondary-schedule] group=" << group_secondaries
              << "; particle state and RNG identities preserved\n";
    std::vector<SecondaryParticle> birth_secondaries_host;
#if defined(CARBON_ENABLE_MINIBEAM)
    std::vector<SecondaryParticle> copper_capture_charged_host;
    if(!copper_capture_path.empty() && secondary_count_device && secondary_queue_device) {
        std::uint32_t count=0;
        queue.copy(secondary_count_device,&count,1).wait_and_throw();
        if(count>max_secondaries)throw std::runtime_error("Cu capture rejects secondary queue overflow");
        std::vector<SecondaryParticle> particles(count);
        if(count)queue.copy(secondary_queue_device,particles.data(),count).wait_and_throw();
        for(const auto& particle:particles)
            if(particle.birth_region==minibeam_birth_region_copper)copper_capture_charged_host.push_back(particle);
    }
#endif
    if (enable_secondary_transport &&
        secondary_count_device != nullptr && secondary_queue_device != nullptr) {
        uint32_t secondary_count_host = 0;
        queue.copy(secondary_count_device, &secondary_count_host, 1).wait_and_throw();
        if (secondary_count_host > max_secondaries) {
            secondary_count_host = static_cast<uint32_t>(max_secondaries);
        }
        if (secondary_count_host > 0) {
            std::uint32_t generation_begin = 0U;
            std::uint32_t generation_end = secondary_count_host;
            while (generation_begin < generation_end) {
                const auto batch_begin = generation_begin;
                unsigned species0_initial_end = 0;
                unsigned species1_initial_end = 0;
                if (group_secondaries) {
                    const auto grouping_start = std::chrono::steady_clock::now();
                    queue.fill(secondary_group_counts, 0u,
                               kSecondaryGroupBucketCount).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const auto frag = secondary_queue_device[generation_begin + id[0]];
                            const unsigned bucket =
                                secondary_group_bucket(frag.z, frag.a, frag.energy_MeV);
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                count(secondary_group_counts[bucket]);
                            count.fetch_add(1);
                        }).wait_and_throw();
                    queue.single_task([=]() {
                        unsigned total = 0;
                        for (unsigned k = 0; k < kSecondaryGroupBucketCount; ++k) {
                            secondary_group_cursors[k] = total;
                            total += secondary_group_counts[k];
                        }
                    }).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const unsigned index = generation_begin + id[0];
                            const auto frag = secondary_queue_device[index];
                            const unsigned bucket =
                                secondary_group_bucket(frag.z, frag.a, frag.energy_MeV);
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                cursor(secondary_group_cursors[bucket]);
                            secondary_order[cursor.fetch_add(1)] = index;
                        }).wait_and_throw();
                    if constexpr(CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE) {
                        queue.copy(secondary_group_cursors + 15,
                                   &species0_initial_end, 1).wait_and_throw();
                        queue.copy(secondary_group_cursors + 31,
                                   &species1_initial_end, 1).wait_and_throw();
                    }
                    secondary_group_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - grouping_start).count();
                }
                const unsigned resume_capacity = generation_end - generation_begin;
                const unsigned state_capacity = segment_secondaries ? resume_capacity : 0;
                auto* resume_states = mem_tracker.allocate<SecondaryResumeState>(state_capacity);
                auto* resume_ready = mem_tracker.allocate<unsigned>(state_capacity);
                auto* active_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* next_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* keep = mem_tracker.allocate<unsigned>(state_capacity);
                auto* ranks = mem_tracker.allocate<unsigned>(state_capacity);
                auto* block_counts = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                auto* block_offsets = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                constexpr unsigned kSecondaryActiveCountSlots =
                    CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE ? 3u : 1u;
                auto* active_count = mem_tracker.allocate<unsigned>(
                    segment_secondaries ? kSecondaryActiveCountSlots : 0);
                if (segment_secondaries) {
                    if (!resume_states || !resume_ready || !active_order || !next_order ||
                        !keep || !ranks || !block_counts || !block_offsets || !active_count)
                        throw std::runtime_error("Secondary continuation allocation failed; reduce histories per shard");
                    queue.fill(resume_ready, 0u, resume_capacity).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(resume_capacity), [=](sycl::id<1> i) {
                        active_order[i[0]] = i[0];
                    }).wait_and_throw();
                    std::cout << "[segmented-secondary] step_limit=" << kSecondarySegmentSteps << " tail_threshold="
                              << secondary_tail_threshold << " state_bytes="
                              << sizeof(SecondaryResumeState) << " capacity=" << resume_capacity << "\n";
                }
                unsigned resume_active = resume_capacity, segment_rounds = 0;
                unsigned species0_active = species0_initial_end;
                unsigned species1_active = species1_initial_end - species0_initial_end;
                unsigned tail_rounds = 0;
                double compact_seconds = 0, generation_kernel_seconds = 0, generation_tail_seconds = 0;
#if CARBON_SECONDARY_CONTEXT_POINTER
                struct SecondaryTransportContext {
                    NeutronBirthBuffer neutron_birth_buffer;
                    bool group_secondaries;
                    bool enable_inelastic;
                    std::uint32_t * secondary_order;
                    std::uint32_t generation_begin;
                    carbon::SecondaryParticle * secondary_queue_device;
                    std::size_t number_of_voxels;
                    float energy_cutoff_MeV;
                    float * cinel02_species_energy_device;
                    std::uint64_t * cinel02_species_terminal_device;
                    float inverse_depth_bin_width_mm;
                    std::size_t number_of_bins;
                    DepthAtomicT * dose_device;
                    bool enable_voxel_scoring;
                    float voxel_min_x_mm;
                    float inverse_voxel_size_x_mm;
                    float voxel_min_y_mm;
                    float inverse_voxel_size_y_mm;
                    std::size_t voxel_bins_x;
                    std::size_t voxel_bins_y;
                    DoseAtomicT * voxel_dose_device;
                    bool enable_charged_origin_voxel_scoring;
                    DoseAtomicT * charged_origin_voxel_dose_device;
                    bool enable_minibeam_component_voxel_scoring;
                    DoseAtomicT * minibeam_component_voxel_dose_device;
                    bool enable_minibeam_energy_band_roi_scoring;
                    DoseAtomicT * minibeam_energy_band_roi_dose_device;
                    std::uint8_t * minibeam_fixed_region_by_x_bin_device;
                    DoseAtomicT * be_isotope_origin_voxel_dose_device;
                    DoseAtomicT * he_isotope_origin_voxel_dose_device;
                    DepthAtomicT * in_fov_dose_device;
                    float * deposited_device;
                    std::uint64_t * schneider_diag_device;
                    double * grid_deposited_in_device;
                    double * grid_deposited_out_device;
                    bool use_all_elastic;
                    ElasticRecoilStoppingView recoil_stopping;
                    float* generic_recoil_reference_sp_device;
                    bool global_urban;
                    float urban_water_half_width_mm;
                    UrbanMcsView urban_device;
                    std::uint64_t* urban_step_counts;
                    float * untracked_nuclear_device;
                    std::size_t table_size;
                    UnifiedEmDevice unified_device;
                    std::uint64_t * unified_audit;
                    SchneiderCtDeviceContext schneider_ct_device_ctx;
                    float * schneider_float_device;
                    std::uint32_t cinel02_max_secondary_inelastic_generations;
                    SchneiderUnsupportedTrack * schneider_track_log_device;
                    std::uint32_t * schneider_track_count_device;
                    float phantom_length_mm;
                    float water_density_g_per_cm3;
                    bool enable_ct_grid;
                    float ct_origin_x;
                    float ct_origin_y;
                    float ct_origin_z;
                    float ct_spacing_x;
                    float ct_spacing_y;
                    float ct_spacing_z;
                    std::uint32_t ct_nx;
                    std::uint32_t ct_ny;
                    std::uint32_t ct_nz;
                    float * ct_density_device;
                    std::uint8_t * ct_material_device;
                    float minimum_table_energy;
                    float inverse_table_step;
                    std::uint32_t * elastic_audit;
                    float maximum_step_mm;
                    unsigned int * unified_failure_count;
                    UnifiedEmFailureRecord * unified_failure_records;
                    float em_secondary_step_scale;
                    float maximum_mean_loss_fraction;
                    float depth_bin_width_mm;
                    bool ct_secondary_exact_faces;
                    bool ct_skip_homogeneous_face_clamp;
                    bool use_unified_water;
                    AllIonElasticView all_elastic;
                    bool enable_secondary_energy_straggling;
                    double * he4_hazard_audit_device;
                    float voxel_max_x_mm;
                    float voxel_max_y_mm;
                    unsigned int * secondary_count_device;
                    std::size_t max_secondaries;
                    std::uint32_t * secondary_overflow_count_device;
                    float * secondary_overflow_energy_device;
                    SchneiderMissRecord * schneider_miss_device;
                    std::uint32_t * schneider_miss_count_device;
                    Cinel02EnergyNode * cinel02_energy_nodes_device;
                    std::uint32_t cinel02_energy_node_count;
                    std::uint32_t * cinel02_event_offsets_device;
                    std::uint32_t * cinel02_event_indices_device;
                    Cinel02DeviceInteraction * cinel02_interactions_device;
                    std::uint32_t cinel02_interaction_count;
                    std::uint32_t cinel02_product_count;
                    Cinel02DeviceProduct * cinel02_products_device;
                    bool enable_multiple_scattering;
                    bool ct_secondary_mcs_off;
                    bool c12_fermi_eyges;
                    int fermi_eyges_species_scope;
                    bool fermi_eyges_use_species_water_parameters;
                    float c12_fermi_eyges_max_segment_mm;
#if defined(CARBON_ENABLE_MINIBEAM)


                    std::uint64_t * minibeam_event_counts_device;
                    float * minibeam_water_primary_plane_depths_device;
                    MinibeamWaterPrimaryPlaneRecord *
                        minibeam_water_primary_plane_records_device;
                    std::size_t minibeam_water_primary_plane_count;
#endif
                    bool ct_material_ids_are_schneider_sections;
                    bool enable_ct_material_mcs;
                    double active_water_radiation_length;
                    float multiple_scattering_scale;
                    float * elastic_energy;
                    float voxel_size_x_mm;
                    float voxel_size_y_mm;
                    float * escaped_device;
                    std::uint64_t * sec_step_profile_device;
#if CARBON_EM_SEARCH_KEY_AUDIT
                    UnifiedEmSearchAuditRecord * unified_search_audit_records;
                    std::uint32_t * unified_search_audit_count;
                    std::uint32_t unified_search_audit_capacity;
#endif
                };
                static_assert(std::is_trivially_copyable_v<SecondaryTransportContext>);
                const SecondaryTransportContext secondary_transport{
                    neutron_birth_buffer,
                    group_secondaries,
                    enable_inelastic,
                    secondary_order,
                    generation_begin,
                    secondary_queue_device,
                    number_of_voxels,
                    energy_cutoff_MeV,
                    cinel02_species_energy_device,
                    cinel02_species_terminal_device,
                    inverse_depth_bin_width_mm,
                    number_of_bins,
                    dose_device,
                    enable_voxel_scoring,
                    voxel_min_x_mm,
                    inverse_voxel_size_x_mm,
                    voxel_min_y_mm,
                    inverse_voxel_size_y_mm,
                    voxel_bins_x,
                    voxel_bins_y,
                    voxel_dose_device,
                    enable_charged_origin_voxel_scoring,
                    charged_origin_voxel_dose_device,
                    enable_minibeam_component_voxel_scoring,
                    minibeam_component_voxel_dose_device,
                    enable_minibeam_energy_band_roi_scoring,
                    minibeam_energy_band_roi_dose_device,
                    minibeam_fixed_region_by_x_bin_device,
                    be_isotope_origin_voxel_dose_device,
                    he_isotope_origin_voxel_dose_device,
                    in_fov_dose_device,
                    deposited_device,
                    schneider_diag_device,
                    grid_deposited_in_device,
                    grid_deposited_out_device,
                    use_all_elastic,
                    recoil_stopping,
                    generic_recoil_reference_sp_device,
                    global_urban,
                    urban_water_half_width_mm,urban_device,urban_step_counts,
                    untracked_nuclear_device,
                    table_size,
                    unified_device,
                    unified_audit,
                    schneider_ct_device_ctx,
                    schneider_float_device,
                    cinel02_max_secondary_inelastic_generations,
                    schneider_track_log_device,
                    schneider_track_count_device,
                    phantom_length_mm,
                    water_density_g_per_cm3,
                    enable_ct_grid,
                    ct_origin_x,
                    ct_origin_y,
                    ct_origin_z,
                    ct_spacing_x,
                    ct_spacing_y,
                    ct_spacing_z,
                    ct_nx,
                    ct_ny,
                    ct_nz,
                    ct_density_device,
                    ct_material_device,
                    minimum_table_energy,
                    inverse_table_step,
                    elastic_audit,
                    maximum_step_mm,
                    unified_failure_count,
                    unified_failure_records,
                    em_secondary_step_scale,
                    maximum_mean_loss_fraction,
                    depth_bin_width_mm,
                    ct_secondary_exact_faces,
                    ct_skip_homogeneous_face_clamp,
                    use_unified_water,
                    all_elastic,
                    enable_secondary_energy_straggling,
                    he4_hazard_audit_device,
                    voxel_max_x_mm,
                    voxel_max_y_mm,
                    secondary_count_device,
                    max_secondaries,
                    secondary_overflow_count_device,
                    secondary_overflow_energy_device,
                    schneider_miss_device,
                    schneider_miss_count_device,
                    cinel02_energy_nodes_device,
                    cinel02_energy_node_count,
                    cinel02_event_offsets_device,
                    cinel02_event_indices_device,
                    cinel02_interactions_device,
                    cinel02_interaction_count,
                    cinel02_product_count,
                    cinel02_products_device,
                    enable_multiple_scattering,
                    ct_secondary_mcs_off,
                    c12_fermi_eyges,
                    fermi_eyges_species_scope,
                    fermi_eyges_use_species_water_parameters,
                    c12_fermi_eyges_max_segment_mm,
#if defined(CARBON_ENABLE_MINIBEAM)


                    minibeam_event_counts_device,
                    minibeam_water_primary_plane_depths_device,
                    minibeam_water_primary_plane_records_device,
                    minibeam_water_primary_plane_count,
#endif
                    ct_material_ids_are_schneider_sections,
                    enable_ct_material_mcs,
                    active_water_radiation_length,
                    multiple_scattering_scale,
                    elastic_energy,
                    voxel_size_x_mm,
                    voxel_size_y_mm,
                    escaped_device,
                    sec_step_profile_device
#if CARBON_EM_SEARCH_KEY_AUDIT
                    ,
                    unified_search_audit_records,
                    unified_search_audit_count,
                    kUnifiedSearchAuditCapacity
#endif
                };
                auto* secondary_transport_ctx =
                    mem_tracker.allocate<SecondaryTransportContext>(1);
                if (secondary_transport_ctx == nullptr) throw std::bad_alloc();
                queue.copy(&secondary_transport, secondary_transport_ctx, 1)
                    .wait_and_throw();
#endif
#if CARBON_SECONDARY_PRODUCTION_SPECIALIZE
                const bool use_production_secondary_path =
                    enable_ct_grid &&
                    enable_voxel_scoring && ct_secondary_exact_faces &&
                    ct_skip_homogeneous_face_clamp && !use_all_elastic &&
                    !use_unified_water && !enable_charged_origin_voxel_scoring;
#endif
                while (resume_active) {
                const bool finish_tail = !segment_secondaries || resume_active < secondary_tail_threshold;
#if CARBON_EM_SEARCH_KEY_AUDIT
                const std::uint64_t search_audit_launch_id =
                    (static_cast<std::uint64_t>(batch_begin)<<32) | segment_rounds;
#endif
                const auto launch_secondary_round = [&](auto production_path_tag,
                                                        auto exact_species_tag,
                                                        unsigned active_begin,
                                                        unsigned active_size) {
                constexpr bool kProductionSecondaryPath =
                    decltype(production_path_tag)::value;
                constexpr int kExactSpecies = decltype(exact_species_tag)::value;
                // -2 is the compile-only Hydrogen-class probe: one kernel serves
                // proton + deuteron (species index 0 or 1) with runtime z/a.
                constexpr bool kHydrogenClass =
                    kProductionSecondaryPath && kExactSpecies == -2;
                constexpr bool kKnownSpeciesOnly =
                    kProductionSecondaryPath &&
                    (CARBON_SECONDARY_KNOWN_SPECIES_PROBE || kExactSpecies >= 0 ||
                     kHydrogenClass);
                constexpr bool kExactSpeciesPath = kExactSpecies >= 0;
                constexpr bool kNonHe4Only =
                    (CARBON_SECONDARY_NON_HE4_PROBE && kKnownSpeciesOnly) ||
                    (kExactSpeciesPath && kExactSpecies != 4) || kHydrogenClass;
                // Proton/deuteron exact slices may run a longer continuation
                // budget; the fallback slice keeps the production default.
                constexpr unsigned kSliceSegmentSteps =
                    (kExactSpecies == 0 || kExactSpecies == 1)
                        ? carbon::kSecondaryHotSegmentSteps
                        : kSecondarySegmentSteps;
                return queue.submit([&](sycl::handler& cgh) {
                const auto secondary_kernel =
#if CARBON_SECONDARY_CONTEXT_POINTER
                    [secondary_transport_ctx, resume_states, resume_ready, active_order,
                     keep, segment_secondaries, finish_tail
#if CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE
                     , active_begin
#endif
#if CARBON_EM_SEARCH_KEY_AUDIT
                     , search_audit_launch_id
#endif
#if CARBON_SECONDARY_EXPLICIT_ND_RANGE
                     , active_size
#endif
                    ]
#else
                    [=]
#endif
#if CARBON_SECONDARY_EXPLICIT_ND_RANGE
                    (sycl::nd_item<1> item) {
                        const auto item_id = item.get_global_id();
                        if (item_id[0] >= active_size) return;
#else
                    (sycl::id<1> item_id) {
#endif
#if CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE
                        const unsigned active_pos = active_begin + item_id[0];
#else
                        const unsigned active_pos = item_id[0];
#endif
                        const unsigned state_idx = segment_secondaries ? active_order[active_pos] : active_pos;
                        const bool resumed = segment_secondaries && resume_ready[state_idx] != 0;
                        if (segment_secondaries) keep[active_pos] = 0;
                        const auto sec_idx = CARBON_SECONDARY_CONTEXT_FIELD(group_secondaries) ? CARBON_SECONDARY_CONTEXT_FIELD(secondary_order)[state_idx]
                            : CARBON_SECONDARY_CONTEXT_FIELD(generation_begin) + state_idx;
                        const auto frag = CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device)[sec_idx];
                        const int transport_z = [&] {
                            if constexpr(kExactSpeciesPath)
                                return carbon::kChargedIons[kExactSpecies].z;
                            return static_cast<int>(frag.z);
                        }();
                        const int transport_a = [&] {
                            if constexpr(kExactSpeciesPath)
                                return carbon::kChargedIons[kExactSpecies].a;
                            return static_cast<int>(frag.a);
                        }();
                        if (transport_z <= 0 || transport_a <= 0) return;
                        const auto charged_origin_category =
                            charged_origin_category_from_fragment(
                                charged_dose_category(transport_z, transport_a));
                        const auto charged_origin_voxel_offset =
                            charged_origin_category * CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels);
                        const auto minibeam_component_voxel_offset =
                            minibeam_component_category(
                                transport_z, transport_a, frag.birth_region) *
                            CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels);
                        const auto be_isotope_category =
                            be_isotope_origin_category(transport_z, transport_a);
                        const auto he_isotope_category =
                            he_isotope_origin_category(transport_z, transport_a);
                        if (frag.energy_MeV <= CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)) {
                            const auto ledger_species_idx = carbon::get_charged_species_idx(
                                static_cast<int>(transport_z), static_cast<int>(transport_a));
                            cinel02_species_energy_add_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 5U,
                                frag.energy_MeV);
                            cinel02_species_terminal_increment_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    Cinel02SpeciesLedgerSchema::initial_below_cutoff));
                            const auto bin_z = static_cast<int>(frag.pos_z_mm * CARBON_SECONDARY_CONTEXT_FIELD(inverse_depth_bin_width_mm));
                            if (bin_z >= 0 && bin_z < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dose(CARBON_SECONDARY_CONTEXT_FIELD(dose_device)[bin_z]);
                                atomic_dose.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
                            }
                            if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring))) {
                                const auto bin_x = static_cast<int>((frag.pos_x_mm - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                const auto bin_y = static_cast<int>((frag.pos_y_mm - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                if (bin_x >= 0 && bin_x < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                    bin_y >= 0 && bin_y < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) &&
                                    bin_z >= 0 && bin_z < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                    const auto cur_voxel = (bin_z * static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) + bin_y) * static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) + bin_x;
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_vox(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[cur_voxel]);
                                    atomic_vox.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                    if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[
                                                charged_origin_voxel_offset + cur_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                            be_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels), cur_voxel,
                                            static_cast<DoseAtomicT>(frag.energy_MeV));
                                        score_he_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                            he_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels), cur_voxel,
                                            static_cast<DoseAtomicT>(frag.energy_MeV));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            component(CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                minibeam_component_voxel_offset + cur_voxel]);
                                        component.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring))) {
                                        score_minibeam_energy_band_roi_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                            transport_z, transport_a, frag.energy_MeV,
                                            cur_voxel, CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                            CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                            CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                            CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device),
                                            frag.energy_MeV);
                                    }
                                    if (CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device) != nullptr) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device)[bin_z]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
                                        cinel02_species_energy_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                            6U, frag.energy_MeV);
                                    }
                                }
                            }
                            if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                atomic_dep.fetch_add(frag.energy_MeV);
                                schneider_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                    frag.energy_MeV);
                                {
                                    const auto sqx = static_cast<int>(
                                        (frag.pos_x_mm - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) *
                                        CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                    const auto sqy = static_cast<int>(
                                        (frag.pos_y_mm - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) *
                                        CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                    const auto sqz = static_cast<int>(
                                        frag.pos_z_mm * CARBON_SECONDARY_CONTEXT_FIELD(inverse_depth_bin_width_mm));
                                    const bool sq_in_grid =
                                        (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && sqx >= 0 &&
                                        sqx < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                        sqy >= 0 &&
                                        sqy < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) &&
                                        sqz >= 0 &&
                                        sqz < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins));
                                    grid_deposit_split_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device), sq_in_grid,
                                        frag.energy_MeV);
                                }
                            }
                            return;
                        }

                        const auto frag_a = static_cast<float>(transport_a);
                        const auto frag_inv_a = 1.0F / frag_a;
                        const auto charged_sp_idx = [&] {
                            if constexpr(kExactSpeciesPath) return kExactSpecies;
                            if constexpr(kHydrogenClass) return (transport_a==2)?1:0;
                            return carbon::get_charged_species_idx(transport_z, transport_a);
                        }();
                        const auto ledger_species_idx = charged_sp_idx;
                        const int recoil_sp_idx=((kProductionSecondaryPath && !BroadBeam) ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_all_elastic))?CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).projectile(transport_z,transport_a):-1;
                        const bool generic_recoil=
                            !kKnownSpeciesOnly && charged_sp_idx<0 && recoil_sp_idx>=0;
                        if (charged_sp_idx < 0 && !generic_recoil) {
                            if (CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device) != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_untracked(
                                        CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device)[frag.parent_history]);
                                atomic_untracked.fetch_add(sycl::fmax(0.0F, frag.energy_MeV));
                            }
                            return;
                        }
                        bool sec_terminal_recorded = false;
                        bool unified_secondary_escaped_ct = false;
                        ContinuousSpeciesTrackTally continuous_species_tally;
                        // Broadbeam common-step ledgers reduced per segment; flushed
                        // before suspend/terminal handling (resume state unchanged).
                        float sec_dep_pending_MeV = 0.0F;
                        std::uint64_t sec_diag_dep_pending_uMeV = 0;
                        double sec_grid_in_pending_MeV = 0.0, sec_grid_out_pending_MeV = 0.0;
                        double he4_audit[4]{};
                        float sec_e = frag.energy_MeV;
                        float sec_x = frag.pos_x_mm;
                        float sec_y = frag.pos_y_mm;
                        float sec_z = frag.pos_z_mm;
                        float sec_dx = frag.dir_x;
                        float sec_dy = frag.dir_y;
                        float sec_dz = frag.dir_z;

                        int pending_sec_bin = -1;
                        int pending_sec_voxel = -1;
                        float pending_sec_depth_MeV = 0.0F;
                        float pending_sec_voxel_MeV = 0.0F;

                        const bool unified_secondary = !generic_recoil;
                        const int unified_secondary_species=unified_secondary
                            ? ([&] {
                                if constexpr(kExactSpeciesPath) return kExactSpecies;
                                if constexpr(kHydrogenClass) return charged_sp_idx;
                                return CARBON_SECONDARY_CONTEXT_FIELD(unified_device).species_index(
                                    transport_z,transport_a);
                              }())
                            : -1;
                        UnifiedEmClock unified_secondary_clock;
                        std::uint64_t unified_secondary_counter=0;
#if CARBON_EM_SEARCH_KEY_AUDIT
                        UnifiedEmSearchAuditState unified_search_audit{
                            CARBON_SECONDARY_CONTEXT_FIELD(unified_search_audit_records),
                            CARBON_SECONDARY_CONTEXT_FIELD(unified_search_audit_count),
                            CARBON_SECONDARY_CONTEXT_FIELD(unified_search_audit_capacity),
                            search_audit_launch_id,active_pos/32,0,0};
#endif
                        // Diagnostic species profile: 20 slots x {steps,
                        // short-range steps, deposited uMeV}. Zero production
                        // impact (compiled out when profiling is off).
                        std::uint64_t sec_prof[60];
                        if constexpr(CARBON_SECONDARY_STEP_PROFILE)
                            for(int sec_pi=0;sec_pi<60;++sec_pi)sec_prof[sec_pi]=0;
                        UnifiedEmState unified_secondary_state;
#if CARBON_EM_SEARCH_KEY_AUDIT
                        unified_secondary_state.search_audit=&unified_search_audit;
#endif
#if CARBON_EM_LOCAL_AUDIT
                        std::array<std::uint64_t,8> unified_secondary_audit{};
#endif
                        auto unified_secondary_uniform=[&](){return (rng::random_u32(2026,frag.rng_stream,unified_secondary_counter++,121)>>8)*0x1p-24f;};
                        auto unified_secondary_count=[&](int index,std::uint64_t count=1){
                            if(count==0)return; // zero-increment diagnostics must not touch globals
#if CARBON_EM_LOCAL_AUDIT
                            unified_secondary_audit[index]+=count;
#else
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(CARBON_SECONDARY_CONTEXT_FIELD(unified_audit)[static_cast<std::size_t>(index)*kUnifiedEmAuditShards+active_pos%kUnifiedEmAuditShards]);a.fetch_add(count);
#endif
                        };
                        uint32_t sec_steps = 0;
                        const uint32_t kSecondaryMaxSteps = 30000U;
                        // Fix C4: one fMinimal track per secondary (persists
                        // across macro steps like the primary state; the old
                        // per-macro-step rebuild pinned tlimit at 1e10, so
                        // secondaries were never MSC-limited at low E).
                        UrbanV2TrackState sec_urban_persist{};
                        bool sec_urban_born = true;
                        bool sec_global_urban_boundary=true;
                        const bool sec_global_urban=!BroadBeam && CARBON_SECONDARY_CONTEXT_FIELD(global_urban);
                        const int sec_global_urban_ion=sec_global_urban?
                            CARBON_SECONDARY_CONTEXT_FIELD(urban_device).projectile(transport_z,transport_a):-1;
                        std::uint64_t local_sec_rate_queries = 0;
                        std::uint64_t local_sec_steps = 0;
                        const int schneider_reg_idx =
                            CARBON_SECONDARY_CONTEXT_FIELD(enable_inelastic)
                                ? secondary_projectile_lut_index_device(
                                      CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_proj_keys,
                                      CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_num_projectiles,
                                      transport_z, transport_a)
                                : -1;
                        if(!resumed){
                        // A track that reaches the stepping loop has actually
                        // started charged transport: count it and itemize its
                        // birth kinetic energy (not a config-switch inference).
                        schneider_diag_increment_device(
                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                            SchneiderDiagSlot::SecondaryTracksStarted);
                        schneider_float_add_device(
                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                            SchneiderFloatSlot::SecondaryTransportBirthEnergy,
                            sycl::fmax(0.0F, frag.energy_MeV));
                        // Per-track unsupported-projectile accounting (once per
                        // track, not per step): generation-eligible tracks whose
                        // (Z/A) has no secondary rate-table registry entry.
                        // The per-step evaluation counter below
                        // (UnsupportedProjectileSteps) is diagnostic only and
                        // must NOT drive coverage gates.
                        // v3: registry lookup over the uploaded bundle-ordered
                        // keys (any (Z,A) in the bundle is supported).

                        if (CARBON_SECONDARY_CONTEXT_FIELD(enable_inelastic) &&
                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() &&
                            frag.generation < CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations) &&
                            schneider_reg_idx < 0) {
                            schneider_diag_increment_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                SchneiderDiagSlot::UnsupportedProjectileTracks);
                            if (transport_z == 4 && transport_a == 6) {
                                schneider_diag_increment_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                    SchneiderDiagSlot::UnsupportedBe6Tracks);
                            }
                            schneider_energy_add_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV,
                                sycl::fmax(0.0F, frag.energy_MeV));
                            // Per-(Z/A) census record for the coverage audit.
                            schneider_log_unsupported_track_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_track_log_device), CARBON_SECONDARY_CONTEXT_FIELD(schneider_track_count_device),
                                kSchneiderTrackLogCap,
                                transport_z, transport_a, frag.generation,
                                sycl::fmax(0.0F, frag.energy_MeV),
                                frag.pos_x_mm, frag.pos_y_mm, frag.pos_z_mm);
                        }
                        }
                        if(resumed){
                            const auto saved=resume_states[state_idx];
                            sec_terminal_recorded=saved.sec_terminal_recorded;
                            unified_secondary_escaped_ct=saved.unified_secondary_escaped_ct;
                            continuous_species_tally=saved.continuous_species_tally;
                            sec_e=saved.sec_e;
                            sec_x=saved.sec_x;
                            sec_y=saved.sec_y;
                            sec_z=saved.sec_z;
                            sec_dx=saved.sec_dx;
                            sec_dy=saved.sec_dy;
                            sec_dz=saved.sec_dz;
                            pending_sec_depth_MeV=saved.pending_sec_depth_MeV;
                            pending_sec_voxel_MeV=saved.pending_sec_voxel_MeV;
                            pending_sec_bin=saved.pending_sec_bin;
                            pending_sec_voxel=saved.pending_sec_voxel;
                            unified_secondary_counter=saved.unified_secondary_counter;
                            unified_secondary_clock=saved.urban_ionization_clock;
                            unified_secondary_state=saved.unified_secondary_state;
                            unified_secondary_state.tables=&CARBON_SECONDARY_CONTEXT_FIELD(unified_device);
#if CARBON_EM_SEARCH_KEY_AUDIT
                            unified_secondary_state.search_audit=&unified_search_audit;
#endif
#if CARBON_EM_LOCAL_AUDIT
                            unified_secondary_audit=saved.unified_secondary_audit;
#endif
                            sec_steps=saved.sec_steps;
                            if constexpr (!BroadBeam) {
                            sec_urban_persist.tlimit_mm =
                                saved.urban.sec_urban_tlimit_mm;
                            sec_urban_born = saved.urban.sec_urban_born;
                            if(sec_global_urban)sec_urban_persist.first_step=saved.urban.sec_urban_first_step;
                            sec_global_urban_boundary=saved.urban.sec_global_urban_boundary;
                            }
                            local_sec_rate_queries=saved.local_sec_rate_queries;
                            local_sec_steps=saved.local_sec_steps;
                            if constexpr (!kNonHe4Only)
                                for(int j=0;j<4;++j)he4_audit[j]=saved.he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)sec_prof[j]=saved.sec_prof[j];
                        }
                        unsigned segment_steps=0;bool segment_paused=false;
                        while (sec_e > CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV) && sec_z >= 0.0F && sec_z < CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm) &&
                               sec_steps < kSecondaryMaxSteps) {
                            if(!finish_tail && segment_steps>=kSliceSegmentSteps){segment_paused=true;break;}
                            ++segment_steps;
#if CARBON_EM_SEARCH_KEY_AUDIT
                            unified_search_audit.step=sec_steps;
                            unified_search_audit.ordinal=0;
#endif
                            if(sec_global_urban && !(kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) &&
                               urban_global_outside_water({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},
                                   CARBON_SECONDARY_CONTEXT_FIELD(urban_water_half_width_mm),CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm))) {
                                // This escape flag also suppresses terminal deposition at an exact water face.
                                unified_secondary_escaped_ct=true;break;
                            }
                            // A surface belongs to the cell entered by the
                            // track. floor(z/dz) already has the right forward
                            // convention; one representable step upstream gives
                            // the corresponding convention for backward tracks.
                            const auto directed_sec_z = sec_dz < -1.0e-6F
                                ? sycl::nextafter(
                                      sec_z,
                                      -std::numeric_limits<float>::infinity())
                                : sec_z;
                            const auto bin_z = static_cast<int>(sycl::floor(
                                directed_sec_z * CARBON_SECONDARY_CONTEXT_FIELD(
                                                     inverse_depth_bin_width_mm)));

                            if (bin_z < 0 || bin_z >= static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                if(sec_global_urban)unified_secondary_escaped_ct=true;
                                break;
                            }

#if defined(CARBON_ENABLE_MINIBEAM)
                            // A requested plane may coincide with the current
                            // step start (notably the water-exit endpoint after
                            // an exactly truncated preceding step). Record that
                            // state before the next EM/MCS draw instead of
                            // requiring a strict interior crossing or moving
                            // the diagnostic plane upstream by a few microns.

#endif

                            const auto secondary_rate_query_energy_MeV =
                                sycl::fmax(0.0F, sec_e);
                            const auto sec_e_u = sec_e * frag_inv_a;
                            float sec_local_density_g_per_cm3 = CARBON_SECONDARY_CONTEXT_FIELD(water_density_g_per_cm3);
                            std::uint8_t sec_ct_material = 2U;
                            bool sec_in_ct = false;
                            if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid))) {
                                sec_in_ct = ct_sample(
                                    sec_x, sec_y, sec_z, CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_nx), CARBON_SECONDARY_CONTEXT_FIELD(ct_ny), CARBON_SECONDARY_CONTEXT_FIELD(ct_nz), CARBON_SECONDARY_CONTEXT_FIELD(ct_density_device),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_material_device), sec_local_density_g_per_cm3,
                                    sec_ct_material, sec_dx, sec_dy, sec_dz);
                            }
                            // The unified package has no exterior material. Leaving the
                            // CT volume is a charged-particle escape, not a lookup error.
                            if ((unified_secondary || sec_global_urban) && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && !sec_in_ct) {
                                unified_secondary_escaped_ct = true;
                                break;
                            }
                            auto recoil_linear_sp = [&](float energy_mevu) {
                                const float value = CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).stopping(
                                    recoil_sp_idx, sec_in_ct ? sec_ct_material : 25, energy_mevu);
                                if (!(value > 0.0F)) {
                                    for (const int slot : {2, 6})
                                        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                            CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[slot]).fetch_add(1);
                                    return -1.0F;
                                }
                                return value * sec_local_density_g_per_cm3;
                            };
                            float recoil_sp_at_start = 0.0F;
                            if (generic_recoil) {
                                recoil_sp_at_start = recoil_linear_sp(sec_e_u);
                                if (!(recoil_sp_at_start > 1.0e-6F)) break;
                            }

                            float sec_step_mm = CARBON_SECONDARY_CONTEXT_FIELD(maximum_step_mm);
                            UnifiedEmStep unified_secondary_pre;
                            float unified_secondary_rate=0,unified_secondary_distance=std::numeric_limits<float>::infinity();
                            if(unified_secondary){
                                const int section=(kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid))?(sec_in_ct?int(sec_ct_material):-2):-1;
                                if(!CARBON_EM_MATERIAL_CACHE || !unified_secondary_state.valid || unified_secondary_state.section!=section ||
                                   unified_secondary_state.density!=sec_local_density_g_per_cm3)
                                    unified_secondary_state=CARBON_SECONDARY_CONTEXT_FIELD(unified_device).select(section,sec_local_density_g_per_cm3,unified_secondary_species);
#if CARBON_EM_SEARCH_KEY_AUDIT
                                unified_secondary_state.search_audit=&unified_search_audit;
#endif
                                if(!unified_secondary_state.covers(sec_e)){
                                    record_unified_em_failure(CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_count),CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_records),3,section,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,0);
                                    unified_secondary_count(0);break;}
                                unified_secondary_pre=unified_secondary_state.prepare(sec_e);
                                if(sec_global_urban) {
                                    unified_secondary_rate=unified_secondary_clock.update(unified_secondary_state,sec_e,
                                        unified_secondary_pre.lo.factor,unified_secondary_pre.hi.factor);
                                    if(!unified_secondary_clock.clock.active)
                                        unified_secondary_clock.clock.arm(-sycl::log(sycl::fmax(unified_secondary_uniform(),0x1p-24f)));
                                    unified_secondary_distance=unified_secondary_clock.clock.distance(unified_secondary_rate);
                                }
                                sec_step_mm=sycl::fmin((CARBON_SECONDARY_CONTEXT_FIELD(em_secondary_step_scale)!=1.f?unified_secondary_state.research_step(sec_e,CARBON_EM_STEP_CACHE?unified_secondary_pre:unified_secondary_state.prepare(sec_e),CARBON_SECONDARY_CONTEXT_FIELD(em_secondary_step_scale)):(CARBON_EM_STEP_CACHE?unified_secondary_state.step(unified_secondary_pre):unified_secondary_state.step(sec_e))),unified_secondary_distance);
                                // Bound combined mean loss, not the old restricted range alone.
                                const float delta_sp=unified_secondary_state.mix(unified_secondary_pre.lo.delta_stopping,unified_secondary_pre.hi.delta_stopping);
                                if(delta_sp>0) sec_step_mm=sycl::fmin(sec_step_mm,CARBON_SECONDARY_CONTEXT_FIELD(maximum_mean_loss_fraction)*sec_e/sycl::fmax(1e-12f,delta_sp+unified_secondary_state.mix(unified_secondary_pre.lo.stopping,unified_secondary_pre.hi.stopping)));
                            }

                            bool depth_boundary_limited = false;
                            if (!sec_global_urban && sec_dz > 1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z + 1) * CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm);
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            } else if (!sec_global_urban && sec_dz < -1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z) * CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm);
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            }
                            CtFaceClampResult sec_face_clamp{sec_step_mm,false,0};
                            if (!sec_global_urban && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces))) {
                                sec_face_clamp = clamp_step_to_ct_faces_exact(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z), CARBON_SECONDARY_CONTEXT_FIELD(ct_nx), CARBON_SECONDARY_CONTEXT_FIELD(ct_ny), CARBON_SECONDARY_CONTEXT_FIELD(ct_nz));
                                sec_step_mm = sec_face_clamp.step_mm;
                            } else if (!sec_global_urban && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct) {
                                sec_step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z), CARBON_SECONDARY_CONTEXT_FIELD(ct_nx), CARBON_SECONDARY_CONTEXT_FIELD(ct_ny),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_nz), CARBON_SECONDARY_CONTEXT_FIELD(ct_density_device), CARBON_SECONDARY_CONTEXT_FIELD(ct_material_device),
                                    sec_local_density_g_per_cm3, sec_ct_material,
                                    (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_skip_homogeneous_face_clamp)), nullptr);
                            }
                            // Do not enlarge a real face distance to the legacy
                            // minimum step: that would cross the material again.
                            if (!unified_secondary && !depth_boundary_limited &&
                                (!(kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces)) || !sec_face_clamp.hit_face))
                                sec_step_mm = sycl::fmax(sec_step_mm, 1.0e-5F);
                            // Generic EM-only recoils take CSDA steps without
                            // fluctuations: a 5% relative-loss cap keeps
                            // stopping variation second-order while cutting
                            // the 0.5%-capped micro-step count ~10x.
                            if(generic_recoil)sec_step_mm=sycl::fmin(sec_step_mm,0.05F*sec_e/recoil_sp_at_start);
                            UrbanGlobalProposal sec_urban_proposal{};
                            CorrelatedScatteringStep sec_urban_scattering{};
                            if(sec_global_urban) {
                                const auto origin=Direction3F{CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z)};
                                const auto spacing=Direction3F{CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z)};
                                const auto cell=sec_in_ct?urban_voxel_cell({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},origin,
                                    {origin.x+CARBON_SECONDARY_CONTEXT_FIELD(ct_nx)*spacing.x,origin.y+CARBON_SECONDARY_CONTEXT_FIELD(ct_ny)*spacing.y,origin.z+CARBON_SECONDARY_CONTEXT_FIELD(ct_nz)*spacing.z},spacing,
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_nx),CARBON_SECONDARY_CONTEXT_FIELD(ct_ny),CARBON_SECONDARY_CONTEXT_FIELD(ct_nz)):
                                    urban_global_water_cell({sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},
                                        {CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm),CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm),0},
                                        {CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)+CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm),
                                         CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)+CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm),CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm)},
                                        {1/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm),1/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm),CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm)},
                                        CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),CARBON_SECONDARY_CONTEXT_FIELD(urban_water_half_width_mm));
                                const auto view=CARBON_SECONDARY_CONTEXT_FIELD(urban_device);
                                const int material=view.material(sec_in_ct?int(sec_ct_material):-1,sec_local_density_g_per_cm3);
                                sec_urban_proposal=urban_global_prepare(view,material,sec_global_urban_ion,sec_e,
                                    sycl::fmin(sec_step_mm,CARBON_SECONDARY_CONTEXT_FIELD(maximum_step_mm)),
                                    {sec_x,sec_y,sec_z},{sec_dx,sec_dy,sec_dz},cell,
                                    sec_global_urban_boundary,sec_urban_persist,2026,frag.rng_stream,sec_steps,
                                    sec_local_density_g_per_cm3);
                                if(!sec_urban_proposal.valid) {
                                    record_unified_em_failure(CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_count),CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_records),100,
                                        sec_in_ct?int(sec_ct_material):-1,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);break;
                                }
                                sec_step_mm=sec_urban_proposal.true_path;
                            }
                            bool secondary_inelastic = false;
                            bool secondary_elastic = false;
                            // A nuclear hazard is not necessarily a replayed
                            // event. Keep this separate so lookup misses and
                            // invalid records follow null-collision transport
                            // semantics and still receive normal MCS.
                            bool secondary_replay_succeeded = false;
                            std::int16_t secondary_target_z = 0;
                            std::int16_t secondary_target_a = 0;
                            // v3: set when the post-EM sampler finds no
                            // in-domain channel (lookup skipped, never queried).
                            bool skip_schneider_lookup = false;
                            // Stash for the miss log: macro total rate at
                            // sampling (density x mass, density exactly once),
                            // sampled collision distance, and section.
                            float schneider_hazard_total_rate = 0.0F;
                            float schneider_hazard_step_mm = 0.0F;
                            std::uint8_t schneider_hazard_section = 255;
                            if (CARBON_SECONDARY_CONTEXT_FIELD(enable_inelastic) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))) &&
                                frag.generation < CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations)) {
                                // Bundle-ordered registry LUT; no hardcoded isotope fallback.
                                const int proj_idx =
#if CARBON_SECONDARY_REUSE_PROJECTILE_INDEX
                                    schneider_reg_idx;
#else
                                    secondary_projectile_lut_index_device(
                                              CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_proj_keys,
                                              CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_num_projectiles,
                                              transport_z, transport_a);
#endif
                                if (proj_idx < 0) {
                                    // Unsupported secondary projectile: explicit
                                    // per-STEP evaluation counter, never a silent
                                    // zero-rate step. (Generation-ineligible tracks
                                    // skip nuclear evaluation entirely: stage-C
                                    // transport without secondary reactions.)
                                    // Coverage gates use the per-TRACK counter
                                    // recorded at track start, not this value.
                                    schneider_diag_increment_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                        SchneiderDiagSlot::UnsupportedProjectileSteps);
                                } else if (CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_total_rates != nullptr) {
                                    const std::size_t section_id = (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    ++local_sec_rate_queries;
                                    // v3: single masked computation drives hazard
                                    // AND target sampling (same partials, same
                                    // mask), without an older total/sampler alternative.
                                    {
                                        // v3: hazard from the masked total at the
                                        // step-start energy E_h. Target sampling
                                        // is deferred to the lookup site (post-EM
                                        // collision energy E_c) so the mask and
                                        // the package query share one energy.
                                        // A post-EM empty draw remains a classified null collision.
                                        const auto sec_masked = CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).secondary_rates(
                                            proj_idx, section_id, sec_e_u);
                                        const float sec_macro_xs =
                                            sec_local_density_g_per_cm3 * sec_masked.total;
                                        schneider_hazard_total_rate = sec_macro_xs;
                                        if (sec_macro_xs > 0.0F) {
                                            float collision_distance = sec_step_mm;
                                            const auto collision = sample_exponential_collision(
                                                sec_macro_xs, sec_step_mm,
                                                rng::uniform01(2026, frag.rng_stream, sec_steps, 13));
                                            secondary_inelastic = collision.occurred;
                                            collision_distance = collision.distance_mm;
                                            if (secondary_inelastic) {
                                                sec_step_mm = collision_distance;
                                                schneider_hazard_total_rate = sec_macro_xs;
                                                schneider_hazard_step_mm = sec_step_mm;
                                                schneider_hazard_section = (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) ? 255U : sec_ct_material;
                                            }
                                        }
                                    }
                                }
                            }

                            // Independent exponential elastic clock competes with the already
                            // sampled inelastic distance. Elastic is NOT generation-limited.
                            if(CARBON_SECONDARY_CONTEXT_FIELD(enable_inelastic) &&
                               ((kProductionSecondaryPath && !BroadBeam) ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_all_elastic)) && !generic_recoil && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)))) {
                                const int p=CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).projectile(transport_z,transport_a);
                                const float rate=CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).rate(p,(kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))?25:sec_ct_material,sec_e_u);
                                if(rate<0) {
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[7]).fetch_add(1);
                                    break;
                                }
                                const auto collision = sample_exponential_collision(
                                    rate * sec_local_density_g_per_cm3, sec_step_mm,
                                    rng::uniform01(2026, frag.rng_stream, sec_steps, 70));
                                secondary_elastic = collision.occurred;
                                if (secondary_elastic) {
                                    sec_step_mm = collision.distance_mm;
                                    secondary_inelastic = false;
                                }
                            }
                            if(sec_global_urban) {
                                if(unified_secondary) {
                                    const bool ionization_proposal=sec_step_mm>=unified_secondary_distance;
                                    unified_secondary_clock.clock.consume(sec_step_mm,unified_secondary_rate,ionization_proposal);
                                }
                                sec_urban_scattering=urban_global_finalize(sec_urban_proposal,sec_step_mm,2026,frag.rng_stream,sec_steps);
                                const int reason=urban_step_field_reason(sec_urban_scattering);
                                if(reason) {
                                    record_unified_em_failure(CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_count),CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_records),110+reason,
                                        sec_in_ct?int(sec_ct_material):-1,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);break;
                                }
                                // MSC AlongStep direction precedes hadronic PostStep replay.
                                sec_dx=sec_urban_scattering.direction.x;sec_dy=sec_urban_scattering.direction.y;sec_dz=sec_urban_scattering.direction.z;
                                sec_urban_persist=sec_urban_proposal.state;
                                sec_global_urban_boundary=sec_urban_scattering.boundary_crossed;
                                urban_global_count(CARBON_SECONDARY_CONTEXT_FIELD(urban_step_counts),sec_global_urban_ion,true);
                            }
                            if(secondary_inelastic)schneider_diag_increment_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),SchneiderDiagSlot::SecondaryHazards);

                            float dE = 0.0F;
                            if (generic_recoil) {
                                const auto mid_e_u = sycl::fmax(
                                    CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).energies[0],
                                    (sec_e - 0.5F * recoil_sp_at_start * sec_step_mm) * frag_inv_a);
                                if (!(recoil_linear_sp(mid_e_u) > 0.0F)) break;
                                const auto f = (mid_e_u - CARBON_SECONDARY_CONTEXT_FIELD(minimum_table_energy)) *
                                    CARBON_SECONDARY_CONTEXT_FIELD(inverse_table_step);
                                const auto i = sycl::clamp(static_cast<int>(sycl::floor(f)), 0,
                                    static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(table_size)) - 2);
                                const auto w = sycl::clamp(f - static_cast<float>(i), 0.0F, 1.0F);
                                const auto* reference = CARBON_SECONDARY_CONTEXT_FIELD(generic_recoil_reference_sp_device);
                                const auto stopping = reference[i] + w * (reference[i + 1] - reference[i]);
                                dE = sycl::fmin(stopping * sec_step_mm, sec_e);
                            }

                            // CINEL02 packages are captured at hadronic PostStepDoIt:
                            // advance the continuous EM state to the collision point
                            // before selecting and replaying the final state.
                            const auto collision_input_dx = sec_dx;
                            const auto collision_input_dy = sec_dy;
                            const auto collision_input_dz = sec_dz;
                            if(sec_global_urban && sec_urban_scattering.range_terminal)dE=sec_e;
                            if(unified_secondary && !(sec_global_urban && sec_urban_scattering.range_terminal)){
                                auto draw=unified_em_loss(unified_secondary_state,unified_secondary_clock,sec_e,sec_step_mm,unified_secondary_rate,unified_secondary_distance,CARBON_SECONDARY_CONTEXT_FIELD(enable_secondary_energy_straggling),unified_secondary_pre,unified_secondary_uniform);
                                if(!draw.valid){
                                    record_unified_em_failure(CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_count),CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_records),40+static_cast<int>(draw.failure_stage),unified_secondary_state.section,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);unified_secondary_count(7);break;}
                                dE=draw.loss;unified_secondary_count(2);unified_secondary_count(3,draw.proposed);unified_secondary_count(4,draw.accepted);
                                unified_secondary_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_secondary_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                            }
                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);
                            if constexpr (!kNonHe4Only) {
                            if (CARBON_SECONDARY_CONTEXT_FIELD(enable_inelastic) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(he4_hazard_audit_device) && transport_z == 2 && transport_a == 4 &&
                                frag.generation < CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)))) {
                                const auto p =
#if CARBON_SECONDARY_REUSE_PROJECTILE_INDEX
                                    schneider_reg_idx;
#else
                                    secondary_projectile_lut_index_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_proj_keys,
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_num_projectiles, 2, 4);
#endif
                                const auto section = (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) ? 255U : sec_ct_material;
                                const double start = schneider_hazard_total_rate;
                                const double middle = sec_local_density_g_per_cm3 *
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).secondary_rates(p, section,
                                        (sec_e - 0.5F*dE)*frag_inv_a).total;
                                const double end = sec_local_density_g_per_cm3 *
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).secondary_rates(p, section,
                                        post_em_e*frag_inv_a).total;
                                he4_audit[0] += start*sec_step_mm;
                                he4_audit[1] += (start+4*middle+end)*sec_step_mm/6;
                                he4_audit[2] += (start*sec_e+4*middle*(sec_e-0.5F*dE)+end*post_em_e)*sec_step_mm/6;
                                he4_audit[3] += sec_step_mm;
                                // Sparse inelastic tallies: rare events update the
                                // global device audit directly instead of living in
                                // the per-track continuation state.
                                if (secondary_inelastic) {
                                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>
                                        tally_count(CARBON_SECONDARY_CONTEXT_FIELD(he4_hazard_audit_device)[3]);
                                    tally_count.fetch_add(1.0);
                                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>
                                        tally_energy(CARBON_SECONDARY_CONTEXT_FIELD(he4_hazard_audit_device)[4]);
                                    tally_energy.fetch_add(static_cast<double>(post_em_e));
                                }
                            }
                            }
                            auto post_em_x = sec_x + collision_input_dx * sec_step_mm;
                            auto post_em_y = sec_y + collision_input_dy * sec_step_mm;
                            auto post_em_z = sec_z + collision_input_dz * sec_step_mm;
                            if(sec_global_urban) {
                                const float g=sec_urban_scattering.final_geom_path_mm;
                                auto end=urban_voxel_snap_endpoint(sec_urban_proposal.cell,g,
                                    {sec_x+sec_urban_proposal.direction.x*g+sec_urban_scattering.displacement_mm.x,
                                     sec_y+sec_urban_proposal.direction.y*g+sec_urban_scattering.displacement_mm.y,
                                     sec_z+sec_urban_proposal.direction.z*g+sec_urban_scattering.displacement_mm.z});
                                post_em_x=end.x;post_em_y=end.y;post_em_z=end.z;
                            }
                            if (!sec_global_urban && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces)) && sec_face_clamp.hit_face && !secondary_inelastic && !secondary_elastic) {
                                const auto endpoint=ct_finish_exact_face_step(sec_face_clamp,sec_step_mm,
                                    {sec_x,sec_y,sec_z},{collision_input_dx,collision_input_dy,collision_input_dz},
                                    {CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z)},{CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z)});
                                post_em_x=endpoint[0];post_em_y=endpoint[1];post_em_z=endpoint[2];
                            }

                            // Independent continuous optical-depth audit.  The
                            // runtime sampler uses the step-start rates above;
                            // this diagnostic evaluates the same H/O rates at
                            // the start, midpoint and post-EM endpoint and
                            // integrates them with Simpson's rule over the
                            // actual (possibly collision-truncated) segment.
                            // It never feeds back into target selection or
                            // collision sampling.
                            const auto collision_bin = sycl::max(
                                0, sycl::min(
                                       static_cast<int>(post_em_z *
                                           CARBON_SECONDARY_CONTEXT_FIELD(inverse_depth_bin_width_mm)),
                                       static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins)) - 1));
                            // Exact-face transport keeps continuous loss inside
                            // the SOURCE voxel. Do not mix source x/y with the
                            // endpoint z, or credit this loss at a nuclear vertex.
                            // Commit once here before a replay can update sec_x/y/z
                            // or break; the legacy dE commits below are suppressed.
                            const bool source_voxel_continuous_loss =
                                (sec_global_urban || ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces)) && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct)) && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring));
                            int continuous_step_voxel = -1;
                            if (source_voxel_continuous_loss) {
                                const auto source_dir=sec_urban_proposal.direction;
                                const int nx=CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x);
                                const int ny=CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y);
                                const int nz=CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins);
                                const float xmin=CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm);
                                const float ymin=CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm);
                                const float vx=1/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm);
                                const float vy=1/CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm);
                                const int sx=sec_global_urban ? urban_voxel_axis(sec_x,source_dir.x,xmin,xmin+nx*vx,vx,nx).index :
                                    static_cast<int>(sycl::floor((sec_x-xmin)/vx));
                                const int sy=sec_global_urban ? urban_voxel_axis(sec_y,source_dir.y,ymin,ymin+ny*vy,vy,ny).index :
                                    static_cast<int>(sycl::floor((sec_y-ymin)/vy));
                                const int sz=sec_global_urban ? urban_voxel_axis(sec_z,source_dir.z,0,
                                    CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm),CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm),nz).index : bin_z;
                                if (sx>=0 && sy>=0 && sz>=0 && sx<nx && sy<ny && sz<nz) {
                                    continuous_step_voxel=(sz*ny+sy)*nx+sx;
                                    sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,sycl::access::address_space::global_space>
                                        a(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[continuous_step_voxel]);
                                    a.fetch_add(static_cast<DoseAtomicT>(dE));
                                    if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                                            origin(CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[charged_origin_voxel_offset+continuous_step_voxel]);
                                        origin.fetch_add(static_cast<DoseAtomicT>(dE));
                                        score_be_isotope_origin_voxel_device(CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                            be_isotope_category,CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),continuous_step_voxel,static_cast<DoseAtomicT>(dE));
                                        score_he_isotope_origin_voxel_device(CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                            he_isotope_category,CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),continuous_step_voxel,static_cast<DoseAtomicT>(dE));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                                            component(CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                minibeam_component_voxel_offset+continuous_step_voxel]);
                                        component.fetch_add(static_cast<DoseAtomicT>(dE));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring))) {
                                        score_minibeam_energy_band_roi_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                            transport_z, transport_a, sec_e,
                                            continuous_step_voxel,
                                            CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                            CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                            CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                            CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device), dE);
                                    }
                                }
                            }
                            if (secondary_inelastic) {
                                sec_e = post_em_e;
                                // v3: sample the target HERE at the post-EM
                                // collision energy E_c (sec_e just updated),
                                // so the replay-status record below and the
                                // package query share one energy with the mask.
                                if (CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))) &&
                                    sec_e > CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV) &&
                                    frag.generation <
                                        CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations)) {
                                    const float sec_e_c = sec_e * frag_inv_a;
                                    const int proj_idx_c =
#if CARBON_SECONDARY_REUSE_PROJECTILE_INDEX
                                        schneider_reg_idx;
#else
                                        secondary_projectile_lut_index_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_proj_keys,
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_num_projectiles,
                                            transport_z, transport_a);
#endif
                                    const std::size_t section_c = (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    const auto sec_masked_c = CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).secondary_rates(
                                        proj_idx_c, section_c, sec_e_c);
                                    // Same tag-14 draw as the legacy site (same
                                    // step counter: sec_steps increments after
                                    // the lookup); only the energy moves E_h ->
                                    // E_c. Empty draw (slowing left every
                                    // channel domain): counted, never queried.
                                    const float u_target_c = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 14);
                                    const int sampled_z_c =
                                        sample_masked_secondary_target_device(
                                            sec_masked_c.partials, u_target_c);
                                    if (sampled_z_c <= 0) {
                                        // Post-EM null collision (declared research
                                        // approximation): no UnsupportedTargets,
                                        // no StoppedBeforeReplay. The track keeps
                                        // its post-EM energy/position, continues
                                        // transport, loses no energy, deposits
                                        // nothing locally, issues no lookup.
                                        schneider_diag_increment_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                            SchneiderDiagSlot::SecondaryPostEmNullCollisions);
                                        schneider_float_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                                            SchneiderFloatSlot::PostEmNullEnergy,
                                            sycl::fmax(0.0F, sec_e));
                                        secondary_inelastic = false;
                                        skip_schneider_lookup = true;
                                    } else {
                                        secondary_target_z =
                                            static_cast<std::int16_t>(sampled_z_c);
                                    }
                                }
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                                // Record the hazard before attempting package
                                // replay so isotope/target/generation misses
                                // remain distinguishable from valid events.
                                // v3 sampler-empty skips the record exactly
                                // like the v1 hazard-site empty draw (which
                                // never reaches this block).
                                if (!skip_schneider_lookup) {
                                }  // !skip_schneider_lookup (v3 sampler-empty records nothing)
                            }

                            if (bin_z != pending_sec_bin) {
                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dose(CARBON_SECONDARY_CONTEXT_FIELD(dose_device)[pending_sec_bin]);
                                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
                                    pending_sec_depth_MeV = 0.0F;
                                }
                                pending_sec_bin = bin_z;
                            }
                            pending_sec_depth_MeV += dE;
                            continuous_species_tally.add_all(dE);

                            if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) || secondary_inelastic) {
                                int cur_voxel = -1;
                                const auto score_bin_x = static_cast<int>(
                                    (sec_x - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                const auto score_bin_y = static_cast<int>(
                                    (sec_y - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                if (score_bin_x >= 0 && score_bin_x < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) && score_bin_y >= 0 && score_bin_y < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))) {
                                    // Continuous EM loss belongs to the source
                                    // step voxel.  `collision_bin` is the
                                    // post-EM endpoint/nuclear-vertex bin and
                                    // mixing it with source x/y creates depth
                                    // spikes when a step ends on a z boundary.
                                    cur_voxel = (bin_z * static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) + score_bin_y) * static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) + score_bin_x;
                                }
                                if (cur_voxel != pending_sec_voxel || secondary_inelastic) {
                                    if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_vox(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[pending_sec_voxel]);
                                        atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[
                                                charged_origin_voxel_offset + pending_sec_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                            be_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_he_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                            he_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            component(CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                minibeam_component_voxel_offset + pending_sec_voxel]);
                                        component.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                    }
                                    pending_sec_voxel_MeV = 0.0F;
                                    pending_sec_voxel = cur_voxel;
                            if (secondary_inelastic && !(sec_e > CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))) &&
                                frag.generation < CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations)) {
                                // Sampled Schneider collision whose post-EM
                                // energy is already below cutoff: continuous
                                // stopping owns the energy, no replay attempted.
                                schneider_diag_increment_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                    SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
                            }
                            if (secondary_inelastic && sec_e > CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)) {
                                // v3 sampling already happened at the post-EM
                                // update above; skip the query only when the
                                // sampler found no in-domain channel.
                                if (CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).uses_cinel03() && !skip_schneider_lookup) {
                                    // Exact (Z/A, target_Z) channel with the
                                    // secondary's own sampled target: never the
                                    // primary target, never water/O fallback.
                                    // Tag 15 drives the bracket choice, tag 16
                                    // the intra-node event pick.
                                    const float sec_u_bracket = rng::uniform01(2026, frag.rng_stream, sec_steps, 15);
                                    const float sec_u_event = rng::uniform01(2026, frag.rng_stream, sec_steps, 16);
                                    const auto sec_lookup = cinel03_lookup_event_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_energy_nodes,
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_node_count,
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_event_offsets,
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_event_indices,
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_total_events,
                                        transport_z, transport_a, secondary_target_z,
                                        sec_e * frag_inv_a,
                                        sec_u_bracket, sec_u_event);
                                    schneider_record_lookup_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device), CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                                        false, sec_lookup, sec_e);

                                    if (sec_lookup.status == Cinel03LookupStatus::Hit) {
                                        const std::uint32_t event_idx = sec_lookup.event_index;
                                        secondary_replay_succeeded = true;
                                        cinel02_species_energy_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 8U, sec_e);
                                        // Use the collision vertex, not the source
                                        // voxel or legacy pre-x/post-z mixed index.
                                        // These are subsets, never extra energy sinks.
                                        if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && replay_vertex_in_scoring_box(
                                                {post_em_x,post_em_y,post_em_z},
                                                {CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm),CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm),0.0F},
                                                {CARBON_SECONDARY_CONTEXT_FIELD(voxel_max_x_mm),CARBON_SECONDARY_CONTEXT_FIELD(voxel_max_y_mm),
                                                 static_cast<float>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))*CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm)})) {
                                            cinel02_species_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                                Cinel02SpeciesLedgerSchema::cinel03_replay_input_fov, sec_e);
                                            cinel02_species_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                                Cinel02SpeciesLedgerSchema::cinel03_replay_step_dE_fov, dE);
                                        }
                                        const auto& event = CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_interactions[event_idx];
                                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                                            2026, frag.rng_stream, sec_steps, cinel03_event_azimuth_dimension);
                                        const float event_cos = sycl::cos(event_phi);
                                        const float event_sin = sycl::sin(event_phi);
                                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);
                                        pending_sec_depth_MeV += local_deposit;
                                        if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += local_deposit;
                                        }
                                        // The break below skips the common-path
                                        // energy-band scorer; local nuclear
                                        // deposit must be recorded here or the
                                        // ROI diagnostic will not close.
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring)) &&
                                            cur_voxel >= 0) {
                                            score_minibeam_energy_band_roi_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                                transport_z, transport_a, sec_e,
                                                cur_voxel,
                                                CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device),
                                                local_deposit);
                                        }
                                        if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                            // Ledger keeps local + step dE here; the
                                            // common path is skipped by the break
                                            // below, so this is the single dE credit.
                                            atomic_dep.fetch_add(local_deposit + dE);
                                            schneider_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                local_deposit + dE);
                                            if (source_voxel_continuous_loss) {
                                                grid_deposit_split_device(CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                    cur_voxel>=0,local_deposit);
                                                grid_deposit_split_device(CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                    continuous_step_voxel>=0,dE);
                                            } else grid_deposit_split_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && cur_voxel>=0,local_deposit+dE);
                                        }
                                        // The break below skips the common-path
                                        // voxel dE commit: commit it here so the
                                        // collision step reaches history ledger,
                                        // depth, voxel and species scorers.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && !source_voxel_continuous_loss, cur_voxel, dE);

                                        float sec_charged_accounted_MeV = 0.0F;
                                        float sec_neutral_accounted_MeV = 0.0F;
                                        float sec_unsupported_accounted_MeV = 0.0F;

                                        const std::uint32_t prod_offset = event.product_offset;
                                        const std::uint32_t prod_count = event.direct_product_count;
                                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                                            if (prod_offset + ip >= CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_total_products) break;
                                            const auto& product = CARBON_SECONDARY_CONTEXT_FIELD(schneider_ct_device_ctx).sec_products[prod_offset + ip];
                                            record_neutron_birth(CARBON_SECONDARY_CONTEXT_FIELD(neutron_birth_buffer),product,
                                                post_em_x,post_em_y,post_em_z,
                                                {collision_input_dx,collision_input_dy,collision_input_dz},event_cos,event_sin,
                                                frag.parent_history,frag.rng_stream,sec_steps,ip,frag.generation+1U,
                                                -1,frag.z,frag.a,frag.weight);

                                            if (product.role == 2) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_unsupported_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                                continue;
                                            }
                                            if (product.z <= 0 || product.a <= 0) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_neutral_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                                continue;
                                            }
                                            if (product.role != 0) {
                                                continue;
                                            }

                                            // TopasCompatKill for Be6 (Z=4, A=6):
                                            // independent counter + energy, never queued.
                                            if (product.z == 4 && product.a == 6) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                schneider_diag_increment_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                    SchneiderDiagSlot::SecondaryBe6Kills);
                                                schneider_diag_increment_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                    SchneiderDiagSlot::Be6TopasCompatKills);
                                                schneider_float_add_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device),
                                                    SchneiderFloatSlot::Be6KillEnergy, ke);
                                                continue;
                                            }
                                            schneider_diag_increment_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                SchneiderDiagSlot::SecondaryChargedBorn);

                                            if (product.kinetic_energy_MeV >
                                                CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)) {
                                                // Nuclear-generation limits never turn a
                                                // charged product into local dose.  The child
                                                // is queued for EM transport; its generation
                                                // guard suppresses any further nuclear hazard.
                                                const auto local_direction = rotate_cinel03_event_azimuth(
                                                    product.local_direction_x, product.local_direction_y,
                                                    product.local_direction_z, event_cos, event_sin);
                                                const auto child_direction = rotate_local_direction(
                                                    local_direction.x,
                                                    local_direction.y,
                                                    local_direction.z,
                                                    Direction3F{collision_input_dx, collision_input_dy, collision_input_dz});

                                                if (CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device) != nullptr) {
                                                    auto count_ref = sycl::atomic_ref<
                                                        uint32_t, sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>(
                                                        *CARBON_SECONDARY_CONTEXT_FIELD(secondary_count_device));
                                                    const auto output = count_ref.fetch_add(1U);
                                                    if (output < CARBON_SECONDARY_CONTEXT_FIELD(max_secondaries)) {
                                                        SecondaryParticle child{};
                                                        child.z = product.z;
                                                        child.a = product.a;
                                                        child.energy_MeV = product.kinetic_energy_MeV;
                                                        child.pos_x_mm = post_em_x;
                                                        child.pos_y_mm = post_em_y;
                                                        child.pos_z_mm = post_em_z;
                                                        child.dir_x = child_direction.x;
                                                        child.dir_y = child_direction.y;
                                                        child.dir_z = child_direction.z;
                                                        child.weight = 1.0F;
                                                        child.generation = static_cast<std::uint16_t>(frag.generation + 1U);
                                                        child.parent_history = frag.parent_history;
                                                        child.rng_stream = rng::event_product_stream(
                                                            frag.rng_stream, sec_steps,
                                                            rng::branch_role_cascade_charged, ip);
                                                        CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device)[output] = child;
                                                        sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                        schneider_diag_increment_device(
                                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                            SchneiderDiagSlot::SecondaryChargedQueued);
                                                        if (CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device) != nullptr) {
                                                            cinel02_record_queued_secondary_birth_device(
                                                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), child.z, child.a,
                                                                child.energy_MeV);
                                                            const auto child_species_idx = carbon::get_charged_species_idx(child.z, child.a);
                                                            if (child_species_idx < 18) {
                                                                cinel02_species_energy_add_device(
                                                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), child_species_idx, 10U, child.energy_MeV);
                                                            }
                                                        }
                                                    } else {
                                                        schneider_diag_increment_device(
                                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                            SchneiderDiagSlot::SecondaryQueueOverflows);
                                                        schneider_diag_increment_device(
                                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                            SchneiderDiagSlot::QueueOverflows);
                                                        if (CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_count_device) != nullptr) {
                                                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ov(*CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_count_device));
                                                            atomic_ov.fetch_add(1U);
                                                        }
                                                        if (CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_energy_device) != nullptr) {
                                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ove(*CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_energy_device));
                                                            atomic_ove.fetch_add(product.kinetic_energy_MeV);
                                                        }
                                                    }
                                                }
                                            } else {
                                                pending_sec_depth_MeV += product.kinetic_energy_MeV;
                                                if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && cur_voxel >= 0) {
                                                    pending_sec_voxel_MeV += product.kinetic_energy_MeV;
                                                }
                                                if ((kProductionSecondaryPath ? false :
                                                     CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring)) &&
                                                    cur_voxel >= 0) {
                                                    const auto child_category =
                                                        minibeam_component_category(
                                                            product.z, product.a,
                                                            minibeam_birth_region_water);
                                                    const auto child_offset =
                                                        child_category *
                                                        CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels);
                                                    sycl::atomic_ref<DoseAtomicT,
                                                                     sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>(
                                                        CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                            child_offset + cur_voxel])
                                                        .fetch_add(static_cast<DoseAtomicT>(
                                                            product.kinetic_energy_MeV));
                                                    sycl::atomic_ref<DoseAtomicT,
                                                                     sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>(
                                                        CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                            minibeam_component_voxel_offset + cur_voxel])
                                                        .fetch_add(static_cast<DoseAtomicT>(
                                                            -product.kinetic_energy_MeV));
                                                }
                                                if ((kProductionSecondaryPath ? false :
                                                     CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring)) &&
                                                    cur_voxel >= 0) {
                                                    score_minibeam_energy_band_roi_device(
                                                        CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                                        product.z, product.a,
                                                        product.kinetic_energy_MeV, cur_voxel,
                                                        CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                                        CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                                        CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                                        CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device),
                                                        product.kinetic_energy_MeV);
                                                }
                                                if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                                    atomic_dep.fetch_add(product.kinetic_energy_MeV);
                                                    schneider_energy_add_device(
                                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                        product.kinetic_energy_MeV);
                                                    grid_deposit_split_device(
                                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                        (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && cur_voxel >= 0,
                                                        product.kinetic_energy_MeV);
                                                }
                                                sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                schneider_diag_increment_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                    SchneiderDiagSlot::SecondaryChargedCutoffKills);
                                            }
                                        }

                                        schneider_float_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_float_device), SchneiderFloatSlot::ReactionQResidual,
                                            sycl::fmax(0.0F, sec_e - local_deposit -
                                                                  sec_charged_accounted_MeV -
                                                                  sec_neutral_accounted_MeV -
                                                                  sec_unsupported_accounted_MeV));
                                        // NO-DOUBLE-COUNT RULE (same as primary
                                        // vertex): legacy untracked sink already
                                        // contains neutral + unsupported + Q;
                                        // split slots are informational only.
                                        const float sec_untracked_MeV = sycl::fmax(0.0F, sec_e - local_deposit - sec_charged_accounted_MeV);
                                        if (sec_untracked_MeV > 0.0F && CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device) != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device)[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_untracked_MeV);
                                        }
                                        sec_e = 0.0F;
                                        break;
                                    } else {
                                        // CINEL03 miss: fail closed + per-miss log.
                                        schneider_log_miss_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(schneider_miss_device), CARBON_SECONDARY_CONTEXT_FIELD(schneider_miss_count_device),
                                            kSchneiderMissLogCap, false,
                                            transport_z, transport_a, secondary_target_z,
                                            schneider_hazard_section,
                                            frag.generation > 255 ? 255
                                                                  : static_cast<std::uint8_t>(frag.generation),
                                            sec_lookup, sec_e * frag_inv_a, dE,
                                            schneider_hazard_total_rate,
                                            schneider_hazard_step_mm, sec_e,
                                            frag.energy_MeV);
                                        if (CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device) != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device)[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_e);
                                        }
                                        // cur_voxel is out of scope on the miss
                                        // path: recompute the collision voxel
                                        // with the same formula for both the
                                        // split and the commit below.
                                        int miss_voxel = -1;
                                        {
                                            const auto miss_sbx = static_cast<int>(
                                                (sec_x - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) *
                                                CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                            const auto miss_sby = static_cast<int>(
                                                (sec_y - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) *
                                                CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                            if (miss_sbx >= 0 &&
                                                miss_sbx < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                                miss_sby >= 0 &&
                                                miss_sby < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))) {
                                                miss_voxel =
                                                    (collision_bin *
                                                     static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) +
                                                     miss_sby) *
                                                        static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) +
                                                    miss_sbx;
                                            }
                                        }
                                        if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                            atomic_dep.fetch_add(dE);
                                            schneider_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                            grid_deposit_split_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                                CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && (source_voxel_continuous_loss ? continuous_step_voxel>=0 : miss_voxel>=0),
                                                dE);
                                        }
                                        // Same bypass as the replay-hit path:
                                        // the break below skips the common-path
                                        // voxel dE commit.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && !source_voxel_continuous_loss, miss_voxel, dE);
                                        sec_e = 0.0F;
                                        break;
                                    }
                                } else {
                                const auto event_index = cinel02_find_event_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_energy_nodes_device), CARBON_SECONDARY_CONTEXT_FIELD(cinel02_energy_node_count),
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_event_offsets_device), CARBON_SECONDARY_CONTEXT_FIELD(cinel02_event_indices_device),
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_interactions_device), CARBON_SECONDARY_CONTEXT_FIELD(cinel02_interaction_count),
                                    transport_z, transport_a, secondary_target_z, secondary_target_a,
                                    sec_e * frag_inv_a, 0.51F,
                                    rng::uniform01(2026, frag.rng_stream,
                                                   sec_steps, 14));
                                if (event_index !=
                                    std::numeric_limits<std::uint32_t>::max()) {
                                    const auto event =
                                        CARBON_SECONDARY_CONTEXT_FIELD(cinel02_interactions_device)[event_index];
                                    const auto product_end =
                                        static_cast<std::uint64_t>(event.product_offset) +
                                        event.product_count;
                                    const auto replay_status =
                                        product_end <= CARBON_SECONDARY_CONTEXT_FIELD(cinel02_product_count) &&
                                        (event.parent_status == 0 ||
                                         event.parent_status == 2)
                                            ? Cinel02ReplayLedgerSchema::replay_valid
                                            : Cinel02ReplayLedgerSchema::replay_invalid_event;
                                    if (product_end <= CARBON_SECONDARY_CONTEXT_FIELD(cinel02_product_count) &&
                                        (event.parent_status == 0 ||
                                         event.parent_status == 2)) {
                                        secondary_replay_succeeded = true;
                                        if (transport_z >= 1 && transport_z <= 6) {
                                            const auto incident_keV =
                                                static_cast<std::uint64_t>(sycl::fmax(
                                                    0.0F, sec_e) * 1000.0F + 0.5F);
                                        }
                                        if (transport_z == 4) {
                                            const auto be_incident_slot =
                                                cinel02_be_incident_diag_slot_device(
                                                    secondary_target_z, frag.generation,
                                                    sec_e * frag_inv_a);
                                        }
                                        const auto parent_outcome_slot =
                                            cinel02_parent_outcome_diag_slot_device(
                                                transport_z, event.parent_status,
                                                static_cast<std::uint32_t>(frag.generation));
                                        if (parent_outcome_slot !=
                                            std::numeric_limits<std::uint32_t>::max()) {
                                        }
                                        const auto local_deposit = sycl::fmax(
                                            0.0F, event.process_local_deposit_MeV);
                                        const auto parent_after = event.parent_status == 0
                                            ? sycl::fmax(0.0F, event.parent_energy_MeV)
                                            : 0.0F;
                                        const auto reaction_handoff_delta = sec_e -
                                            parent_after - local_deposit;
                                        cinel02_species_energy_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::nuclear_local_deposit_all),
                                            local_deposit);
                                        cinel02_species_energy_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::reaction_export_kinetic),
                                            sycl::fmax(0.0F, reaction_handoff_delta));
                                        cinel02_species_energy_add_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::reaction_import_kinetic),
                                            sycl::fmax(0.0F, -reaction_handoff_delta));
                                        if (local_deposit > 0.0F) {
                                            sycl::atomic_ref<
                                                DepthAtomicT, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                atomic_local_depth(CARBON_SECONDARY_CONTEXT_FIELD(dose_device)[collision_bin]);
                                            atomic_local_depth.fetch_add(
                                                static_cast<DepthAtomicT>(local_deposit));
                                        }
                                        if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && pending_sec_voxel >= 0) {
                                            pending_sec_voxel_MeV += local_deposit;
                                            cinel02_species_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                                4U, local_deposit);
                                        }
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring)) &&
                                            pending_sec_voxel >= 0) {
                                            score_minibeam_energy_band_roi_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                                transport_z, transport_a, sec_e,
                                                pending_sec_voxel,
                                                CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device),
                                                local_deposit);
                                        }
                                        if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                            sycl::atomic_ref<
                                                float, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                atomic_dep(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                            atomic_dep.fetch_add(local_deposit);
                                            schneider_energy_add_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                local_deposit);
                                            grid_deposit_split_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                                CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                                (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) &&
                                                    pending_sec_voxel >= 0,
                                                local_deposit);
                                        }
                                        float untracked_MeV = 0.0F;
                                        for (std::uint32_t ip = 0;
                                             ip < event.product_count; ++ip) {
                                            const auto product = CARBON_SECONDARY_CONTEXT_FIELD(cinel02_products_device)[
                                                event.product_offset + ip];
                                            if (product.role == 0 && product.z >= 1 && product.z <= 6) {
                                            }
                                            if (product.role == 0 && product.z == 4) {
                                                const auto birth_slot =
                                                    cinel02_be_isotope_birth_diag_slot_device(
                                                        transport_z, secondary_target_z,
                                                        frag.generation + 1U, product.a);
                                                if (birth_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                }
                                                const auto be_channel_slot =
                                                    cinel02_be_channel_diag_slot_device(
                                                        transport_z, secondary_target_z,
                                                        frag.generation, sec_e * frag_inv_a);
                                                if (be_channel_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                }
                                            }
                                            if (product.role == 0) {
                                                const auto transition_slot =
                                                    cinel02_transition_diag_slot_device(
                                                        transport_z, product.z,
                                                        static_cast<std::uint32_t>(frag.generation));
                                                if (transition_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                    const auto kinetic_keV =
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, product.kinetic_energy_MeV) *
                                                            1000.0F + 0.5F);
                                                }
                                            }

                                            if (product.role != 0 || product.z <= 0 ||
                                                product.a <= 0 ||
                                                CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device) == nullptr) {
                                                untracked_MeV += sycl::fmax(
                                                    0.0F, product.kinetic_energy_MeV);
                                                continue;
                                            }
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    product.local_direction_x,
                                                    product.local_direction_y,
                                                    product.local_direction_z,
                                                    Direction3F{collision_input_dx, collision_input_dy,
                                                                collision_input_dz});
                                            auto count_ref = sycl::atomic_ref<
                                                uint32_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>(
                                                *CARBON_SECONDARY_CONTEXT_FIELD(secondary_count_device));
                                            const auto output = count_ref.fetch_add(1U);
                                            if (output < CARBON_SECONDARY_CONTEXT_FIELD(max_secondaries)) {
                                                SecondaryParticle child{};
                                                child.z = product.z;
                                                child.a = product.a;
                                                child.energy_MeV =
                                                    product.kinetic_energy_MeV;
                                                child.pos_x_mm = sec_x;
                                                child.pos_y_mm = sec_y;
                                                child.pos_z_mm = sec_z;
                                                child.dir_x = child_direction.x;
                                                child.dir_y = child_direction.y;
                                                child.dir_z = child_direction.z;
                                                child.weight = product.weight;
                                                child.parent_history = frag.parent_history;
                                                const auto event_stream = rng::child_stream(
                                                    frag.rng_stream,
                                                    event_index ^
                                                        (sec_steps * 0x9E3779B9U));
                                                child.rng_stream = rng::child_stream(
                                                    event_stream, rng::branch_tag(
                                                        rng::branch_role_cascade_charged, ip));
                                                child.generation =
                                                    static_cast<std::uint16_t>(
                                                        frag.generation + 1U);
                                                if (product.z >= 1 && product.z <= 6) {
                                                }
                                                CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device)[output] = child;
                                                cinel02_record_queued_secondary_birth_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), product.z, product.a,
                                                    sycl::fmax(0.0F, product.kinetic_energy_MeV));
                                            } else if (CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_count_device) !=
                                                       nullptr) {
                                                sycl::atomic_ref<
                                                    uint32_t, sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    atomic_overflow(
                                                        *CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_count_device));
                                                atomic_overflow.fetch_add(1U);
                                                const auto overflow_energy = sycl::fmax(
                                                    0.0F, product.kinetic_energy_MeV);
                                                untracked_MeV += overflow_energy;
                                                if (CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_energy_device) != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_overflow_energy(
                                                            *CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_energy_device));
                                                    atomic_overflow_energy.fetch_add(overflow_energy);
                                                }
                                            }
                                        }
                                        if (untracked_MeV > 0.0F &&
                                            CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device) != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device)[frag.parent_history]);
                                            atomic_untracked.fetch_add(untracked_MeV);
                                        }
                                        if (event.parent_status == 0) {
                                            sec_e = sycl::fmax(
                                                0.0F, event.parent_energy_MeV);
                                            const auto parent_direction =
                                                rotate_local_direction(
                                                    event.parent_local_direction_x,
                                                    event.parent_local_direction_y,
                                                    event.parent_local_direction_z,
                                                    Direction3F{collision_input_dx, collision_input_dy,
                                                                collision_input_dz});
                                            sec_dx = parent_direction.x;
                                            sec_dy = parent_direction.y;
                                            sec_dz = parent_direction.z;
                                        } else {
                                            cinel02_species_terminal_increment_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                                static_cast<std::uint32_t>(
                                                    Cinel02SpeciesLedgerSchema::reaction_killed));
                                            sec_terminal_recorded = true;
                                            sec_e = 0.0F;
                                        }
                                    }
                                    else {
                                    }
                                }
                                else {
                                }
                                }
                            } else if (secondary_inelastic) {
                                // The post-EM collision energy is already at
                                // or below the transport cutoff, so no package
                                // event is eligible.  Keep the candidate in
                                // the status partition as post_em_below_cutoff.
                            }
                                    pending_sec_voxel = cur_voxel;
                                }
                                if (cur_voxel >= 0) {
                                    if (!source_voxel_continuous_loss) {
                                        pending_sec_voxel_MeV += dE;
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring))) {
                                            score_minibeam_energy_band_roi_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                                transport_z, transport_a, sec_e, cur_voxel,
                                                CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                                CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device), dE);
                                        }
                                    }
                                    if (CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device) != nullptr && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device)[pending_sec_bin]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(dE));
                                        continuous_species_tally.add_fov(dE);
                                    }
                                }
                            }

                            if (!secondary_inelastic) sec_e = post_em_e;
                            // cur_voxel is out of scope on the common path:
                            // recompute the step voxel with the same formula.
                            int com_voxel = -1;
                            {
                                const auto com_sbx = static_cast<int>(
                                    (sec_x - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) *
                                    CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                const auto com_sby = static_cast<int>(
                                    (sec_y - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) *
                                    CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                if (com_sbx >= 0 &&
                                    com_sbx < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                    com_sby >= 0 &&
                                    com_sby < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))) {
                                    com_voxel =
                                        (bin_z *
                                         static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) +
                                         com_sby) *
                                            static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) +
                                        com_sbx;
                                }
                            }
                            if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                if constexpr (BroadBeam) {
                                    // Same per-step values; integer microMeV truncation is
                                    // applied per step so the diagnostic total is unchanged.
                                    sec_dep_pending_MeV += dE;
                                    if (dE > 0.0F)
                                        sec_diag_dep_pending_uMeV += static_cast<std::uint64_t>(static_cast<double>(dE) * 1.0e6);
                                    if (sycl::isfinite(dE) && dE > 0.0F) {
                                        if (source_voxel_continuous_loss ? continuous_step_voxel>=0 : com_voxel>=0) sec_grid_in_pending_MeV += dE;
                                        else sec_grid_out_pending_MeV += dE;
                                    }
                                } else {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                atomic_dep.fetch_add(dE);
                                schneider_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                grid_deposit_split_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                    CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device), source_voxel_continuous_loss ? continuous_step_voxel>=0 : com_voxel>=0,
                                    dE);
                                }
                            }
                            if (!secondary_inelastic) {
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                            }

                            if (!sec_global_urban && cinel02_should_apply_secondary_mcs(
                                    secondary_inelastic, secondary_replay_succeeded,
                                    CARBON_SECONDARY_CONTEXT_FIELD(enable_multiple_scattering) && !CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_mcs_off), sec_e,
                                    CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV))) {
                                [[maybe_unused]] constexpr float two_pi = 6.2831853071795864769F;
                                [[maybe_unused]] float theta_scat = 0.0F;
                                [[maybe_unused]] float phi_scat = 0.0F;
                                // Secondary MCS material selection shares the
                                // primary helper: Schneider CT voxels use the
                                // section X0, everywhere else falls back to
                                // water. Density is the local CT density
                                // (== water outside CT), never double-counted.
                                const auto sec_radiation_length_g_per_cm2 =
                                    static_cast<float>(
                                        select_transport_radiation_length_g_per_cm2(
                                            CARBON_SECONDARY_CONTEXT_FIELD(ct_material_ids_are_schneider_sections),
                                            sec_in_ct,
                                            static_cast<unsigned>(sec_ct_material),
                                            CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_material_mcs),
                                            CARBON_SECONDARY_CONTEXT_FIELD(active_water_radiation_length)));
                                bool secondary_uses_fe = BroadBeam || (
                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                        c12_fermi_eyges) &&
                                    ion_uses_fermi_eyges(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            fermi_eyges_species_scope),
                                        transport_z, transport_a));
#if defined(CARBON_ENABLE_MINIBEAM)
                                // Honor the explicit secondary-C12 Urban selector,
                                // as the primary path does. Generic all-ion FE must
                                // remain available for the other fragment species.

#endif
                                if (secondary_uses_fe) {
                                    const auto route_slot =
                                        transport_z == 6 && transport_a == 12
                                            ? SchneiderDiagSlot::SecondaryFeC12Steps
                                        : transport_z == 2 && transport_a == 4
                                            ? SchneiderDiagSlot::SecondaryFeHe4Steps
                                        : transport_z == 1 && transport_a >= 1 &&
                                                  transport_a <= 3
                                            ? SchneiderDiagSlot::SecondaryFePdtSteps
                                            : SchneiderDiagSlot::SecondaryFeOtherChargedSteps;
                                    schneider_diag_increment_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            schneider_diag_device),
                                        route_slot);
                                    auto observation_path_mm = -1.0F;
#if defined(CARBON_ENABLE_MINIBEAM)
                                    const auto fe_start_x = post_em_x -
                                        collision_input_dx * sec_step_mm;
                                    const auto fe_start_y = post_em_y -
                                        collision_input_dy * sec_step_mm;
                                    const auto fe_start_z = post_em_z -
                                        collision_input_dz * sec_step_mm;
                                    auto observation_plane =
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_primary_plane_count);

#endif
                                    const auto correlated =
                                        ion_fermi_eyges_transport_step(
                                            Direction3F{collision_input_dx,
                                                        collision_input_dy,
                                                        collision_input_dz},
                                            sec_e + dE,
                                            static_cast<int>(transport_z),
                                            static_cast<int>(transport_a),
                                            dE, sec_step_mm,
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                c12_fermi_eyges_max_segment_mm),
                                            sec_local_density_g_per_cm3,
                                            sec_radiation_length_g_per_cm2,
                                            2026, frag.rng_stream,
                                            static_cast<std::uint64_t>(sec_steps),
                                            100U, observation_path_mm,
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                fermi_eyges_use_species_water_parameters), 0.0F, fe_gaussian_only,
                                            fe_moments, fe_moments ? fe_geometry_safety(
                                                {post_em_x-collision_input_dx*sec_step_mm,
                                                 post_em_y-collision_input_dy*sec_step_mm,
                                                 post_em_z-collision_input_dz*sec_step_mm},
                                                CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid),sec_in_ct,
                                                {CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z)},
                                                {CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y),CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z)},
                                                CARBON_SECONDARY_CONTEXT_FIELD(ct_nx),CARBON_SECONDARY_CONTEXT_FIELD(ct_ny),CARBON_SECONDARY_CONTEXT_FIELD(ct_nz),
                                                CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm),fe_simple_geometry) : 0.0F,
                                            fe_counts,fe_diagnostics);
#if defined(CARBON_ENABLE_MINIBEAM)
                                    if (correlated.observation_valid &&
                                        observation_plane <
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_count)) {
                                        const auto plane_depth =
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_depths_device)[
                                                observation_plane];
                                        auto crossing_x = fe_start_x +
                                            collision_input_dx * observation_path_mm +
                                            correlated.observation_displacement_mm.x;
                                        auto crossing_y = fe_start_y +
                                            collision_input_dy * observation_path_mm +
                                            correlated.observation_displacement_mm.y;
                                        const auto crossing_z = fe_start_z +
                                            collision_input_dz * observation_path_mm +
                                            correlated.observation_displacement_mm.z;
                                        if (correlated.observation_direction.z >
                                            1.0e-6F) {
                                            const auto residual_path =
                                                (plane_depth - crossing_z) /
                                                correlated.observation_direction.z;
                                            crossing_x += residual_path *
                                                correlated.observation_direction.x;
                                            crossing_y += residual_path *
                                                correlated.observation_direction.y;
                                        }
                                        auto& record =
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_records_device)[
                                                frag.parent_history *
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_water_primary_plane_count) +
                                                observation_plane];
                                        const auto step_fraction = sycl::clamp(
                                            observation_path_mm / sec_step_mm,
                                            0.0F, 1.0F);
                                        record.history = frag.parent_history;
                                        record.transport_path = 1U;
                                        record.particle_id = frag.rng_stream;
                                        record.rng_stream = frag.rng_stream;
                                        record.plane_index =
                                            static_cast<std::uint32_t>(observation_plane);
                                        record.atomic_number = static_cast<std::int16_t>(transport_z);
                                        record.mass_number = static_cast<std::int16_t>(transport_a);
                                        record.depth_mm = plane_depth;
                                        record.kinetic_energy_MeV = sycl::fmax(
                                            0.0F,
                                            sec_e + (1.0F - step_fraction) * dE);
                                        record.weight = frag.weight;
                                        record.x_mm = crossing_x;
                                        record.y_mm = crossing_y;
                                        record.direction_x =
                                            correlated.observation_direction.x;
                                        record.direction_y =
                                            correlated.observation_direction.y;
                                        record.direction_z =
                                            correlated.observation_direction.z;
                                        record.valid = 1U;
                                    }
#endif
                                    sec_x += correlated.displacement_mm.x;
                                    sec_y += correlated.displacement_mm.y;
                                    sec_z += correlated.displacement_mm.z;
                                    sec_dx = correlated.direction.x;
                                    sec_dy = correlated.direction.y;
                                    sec_dz = correlated.direction.z;
                                }
                                else {
                                    schneider_diag_increment_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            schneider_diag_device),
                                        SchneiderDiagSlot::SecondaryHighlandSteps);
                                    const auto theta_rms = highland_projected_rms_angle_device(
                                        sec_e, static_cast<int>(transport_z),
                                        static_cast<int>(transport_a), sec_step_mm,
                                        sec_local_density_g_per_cm3,
                                        sec_radiation_length_g_per_cm2) * CARBON_SECONDARY_CONTEXT_FIELD(multiple_scattering_scale);
                                    const auto u_msc0 = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 0);
                                    const auto u_msc1 = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 1);
                                    // Unified MSC transport (secondary Highland
                                    // mode): Rayleigh-theta/uniform-phi form,
                                    // same 2-D Gaussian PDF as the primary
                                    // Box-Muller form; zero displacement
                                    // (endpoint scattering).
                                    const auto msc = msc_highland_secondary_step(
                                        Direction3F{collision_input_dx, collision_input_dy,
                                                                    collision_input_dz},
                                        theta_rms, u_msc0, u_msc1);
                                    sec_dx = msc.direction.x;
                                    sec_dy = msc.direction.y;
                                    sec_dz = msc.direction.z;
                                }
                            }
                            // A hazard sampled at step start can leave a supported rate
                            // interval after EM loss. Treat the zero-rate endpoint as a
                            // null collision; retain post-EM energy, position and MCS.
                            if(secondary_elastic && sec_e>CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV) &&
                               CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).rate(CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).projectile(transport_z,transport_a),(kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))?25:sec_ct_material,sec_e*frag_inv_a)==0) {
                                secondary_elastic=false;
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[9]).fetch_add(1);
                            }
                            if(secondary_elastic && sec_e>CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)) {
                                const int p=CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).projectile(transport_z,transport_a);
                                const auto draw=CARBON_SECONDARY_CONTEXT_FIELD(all_elastic).draw(p,(kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water))?25:sec_ct_material,
                                    sec_e,sec_dx,sec_dy,sec_dz,
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,71),rng::uniform01(2026,frag.rng_stream,sec_steps,72),
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,73),rng::uniform01(2026,frag.rng_stream,sec_steps,74));
                                if(!draw.valid){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[8]).fetch_add(1);break;
                                }
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[1]).fetch_add(1);
                                const auto scat=draw.outcome;sec_e=scat.projectile_ke_MeV;
                                sec_dx=scat.proj_dir_x;sec_dy=scat.proj_dir_y;sec_dz=scat.proj_dir_z;
                                const float recoil=scat.recoil_ke_MeV;
                                const int child_species=carbon::get_charged_species_idx(draw.target_z,draw.target_a);
                                if(recoil>CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV) && (child_species>=0||CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).projectile(draw.target_z,draw.target_a)>=0)) {
                                    schneider_diag_increment_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),SchneiderDiagSlot::SecondaryChargedBorn);
                                    const auto output=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*CARBON_SECONDARY_CONTEXT_FIELD(secondary_count_device)).fetch_add(1);
                                    if(output<CARBON_SECONDARY_CONTEXT_FIELD(max_secondaries)){
                                        SecondaryParticle child{};child.z=draw.target_z;child.a=draw.target_a;child.energy_MeV=recoil;
                                        child.pos_x_mm=sec_x;child.pos_y_mm=sec_y;child.pos_z_mm=sec_z;
                                        child.dir_x=scat.recoil_dir_x;child.dir_y=scat.recoil_dir_y;child.dir_z=scat.recoil_dir_z;
                                        child.weight=1;child.generation=child_species>=0?frag.generation:CARBON_SECONDARY_CONTEXT_FIELD(cinel02_max_secondary_inelastic_generations);child.parent_history=frag.parent_history;
                                        child.rng_stream=rng::event_product_stream(frag.rng_stream,sec_steps,rng::branch_role_cascade_charged,0xE1U);
                                        CARBON_SECONDARY_CONTEXT_FIELD(secondary_queue_device)[output]=child;
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_energy)[1]).fetch_add(recoil);
                                        schneider_diag_increment_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),SchneiderDiagSlot::SecondaryChargedQueued);
                                        cinel02_record_queued_secondary_birth_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),child.z,child.a,recoil);
                                        cinel02_species_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),child_species,10U,recoil);
                                    }else{
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_count_device)).fetch_add(1);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*CARBON_SECONDARY_CONTEXT_FIELD(secondary_overflow_energy_device)).fetch_add(recoil);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_energy)[2]).fetch_add(recoil);
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[11]).fetch_add(1);
                                    }
                                    cinel02_species_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),ledger_species_idx,8U,recoil);
                                }else if(recoil>CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV)){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[3]).fetch_add(1);
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(untracked_nuclear_device)[frag.parent_history]).fetch_add(recoil);
                                    cinel02_species_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),ledger_species_idx,8U,recoil);
                                }else if(recoil>0){
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_energy)[0]).fetch_add(recoil);
                                    const int vx=static_cast<int>(sycl::floor((sec_x-CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm))/CARBON_SECONDARY_CONTEXT_FIELD(voxel_size_x_mm)));
                                    const int vy=static_cast<int>(sycl::floor((sec_y-CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm))/CARBON_SECONDARY_CONTEXT_FIELD(voxel_size_y_mm)));
                                    const int vz=static_cast<int>(sycl::floor(sec_z*CARBON_SECONDARY_CONTEXT_FIELD(inverse_depth_bin_width_mm)));
                                    const bool inside=(kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring))&&vx>=0&&vy>=0&&vz>=0&&vx<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x))&&vy<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))&&vz<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins));
                                    if(inside){const int v=(vz*CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)+vy)*CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)+vx;
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[v]).fetch_add(static_cast<DoseAtomicT>(recoil));
                                        if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                            const auto recoil_origin_category =
                                                charged_origin_category_from_fragment(
                                                    charged_dose_category(draw.target_z, draw.target_a));
                                            sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>(
                                                CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[
                                                    recoil_origin_category * CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels) + v])
                                                .fetch_add(static_cast<DoseAtomicT>(recoil));
                                            score_be_isotope_origin_voxel_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                                be_isotope_origin_category(draw.target_z, draw.target_a),
                                                CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels), v,
                                                static_cast<DoseAtomicT>(recoil));
                                            score_he_isotope_origin_voxel_device(
                                                CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                                he_isotope_origin_category(draw.target_z, draw.target_a),
                                                CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels), v,
                                                static_cast<DoseAtomicT>(recoil));
                                        }
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                            const auto recoil_component_category =
                                                minibeam_component_category(
                                                    draw.target_z, draw.target_a,
                                                    minibeam_birth_region_water);
                                            sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>(
                                                CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                    recoil_component_category * CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels) + v])
                                                .fetch_add(static_cast<DoseAtomicT>(recoil));
                                        }
                                    }
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]).fetch_add(recoil);
                                    grid_deposit_split_device(CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),inside,recoil);
                                    schneider_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),SchneiderDiagSlot::SecondaryDepositedMicroMeV,recoil);
                                    cinel02_species_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),ledger_species_idx,3U,recoil);
                                    if(inside)cinel02_species_energy_add_device(CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device),ledger_species_idx,4U,recoil);
                                }
                            }

                            ++sec_steps;
                            ++local_sec_steps;
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
                                const int sec_slot = unified_secondary_species>=0 ? unified_secondary_species : (generic_recoil ? 18 : 19);
                                sec_prof[sec_slot]+=1;
                                const float sec_range = unified_secondary ? unified_secondary_state.range(sec_e) : 1.0e30f;
                                if(sec_range<0.5f)sec_prof[20+sec_slot]+=1;
                                sec_prof[40+sec_slot]+=static_cast<std::uint64_t>(dE*1e6f);
                            }
                        }

                        if constexpr (BroadBeam) {
                            if (sec_dep_pending_MeV != 0.0F && CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr)
                                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]).fetch_add(sec_dep_pending_MeV);
                            schneider_diag_add_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                SchneiderDiagSlot::SecondaryDepositedMicroMeV, sec_diag_dep_pending_uMeV);
                            if (sec_grid_in_pending_MeV != 0.0 && CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device) != nullptr)
                                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device)).fetch_add(sec_grid_in_pending_MeV);
                            if (sec_grid_out_pending_MeV != 0.0 && CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device) != nullptr)
                                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device)).fetch_add(sec_grid_out_pending_MeV);
                            sec_dep_pending_MeV = 0.0F;sec_diag_dep_pending_uMeV = 0;
                            sec_grid_in_pending_MeV = sec_grid_out_pending_MeV = 0.0;
                        }
                        if(segment_paused){
                            SecondaryResumeState saved;
                            saved.sec_terminal_recorded=sec_terminal_recorded;
                            saved.unified_secondary_escaped_ct=unified_secondary_escaped_ct;
                            saved.continuous_species_tally=continuous_species_tally;
                            saved.sec_e=sec_e;
                            saved.sec_x=sec_x;
                            saved.sec_y=sec_y;
                            saved.sec_z=sec_z;
                            saved.sec_dx=sec_dx;
                            saved.sec_dy=sec_dy;
                            saved.sec_dz=sec_dz;
                            saved.pending_sec_depth_MeV=pending_sec_depth_MeV;
                            saved.pending_sec_voxel_MeV=pending_sec_voxel_MeV;
                            saved.pending_sec_bin=pending_sec_bin;
                            saved.pending_sec_voxel=pending_sec_voxel;
                            saved.unified_secondary_counter=unified_secondary_counter;
                            saved.urban_ionization_clock=unified_secondary_clock;
                            saved.unified_secondary_state=unified_secondary_state;
#if CARBON_EM_LOCAL_AUDIT
                            saved.unified_secondary_audit=unified_secondary_audit;
#endif
                            saved.sec_steps=sec_steps;
                            if constexpr (!BroadBeam) {
                            saved.urban.sec_urban_tlimit_mm =
                                sec_urban_persist.tlimit_mm;
                            saved.urban.sec_urban_born = sec_urban_born;
                            saved.urban.sec_urban_first_step=sec_urban_persist.first_step;
                            saved.urban.sec_global_urban_boundary=sec_global_urban_boundary;
                            }
                            saved.local_sec_rate_queries=local_sec_rate_queries;
                            saved.local_sec_steps=local_sec_steps;
                            if constexpr (!kNonHe4Only)
                                for(int j=0;j<4;++j)saved.he4_audit[j]=he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)saved.sec_prof[j]=sec_prof[j];
                            resume_states[state_idx]=saved;resume_ready[state_idx]=1;keep[active_pos]=1;
                            return; // Suspend: no terminal scoring or audit flush.
                        }
#if CARBON_EM_LOCAL_AUDIT
                        if(unified_secondary)flush_unified_em_audit(CARBON_SECONDARY_CONTEXT_FIELD(unified_audit),unified_secondary_audit);
#endif
                        if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
                            for(int sec_pi=0;sec_pi<60;++sec_pi) if(sec_prof[sec_pi]) {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> pa(CARBON_SECONDARY_CONTEXT_FIELD(sec_step_profile_device)[sec_pi]);pa.fetch_add(sec_prof[sec_pi]);
                            }
                        }
                        // Small per-step deposits must not contend directly on
                        if constexpr (!kNonHe4Only) {
                        if (CARBON_SECONDARY_CONTEXT_FIELD(he4_hazard_audit_device) && transport_z == 2 && transport_a == 4) {
                            // Carried slots 0/1/2 keep their device meaning; the
                            // fourth carried value is the total step length
                            // (original device slot 5). Device slots 3/4 are the
                            // sparse inelastic tallies already accumulated at the
                            // event and must not be flushed again.
                            constexpr int carried_slots[4] = {0, 1, 2, 5};
                            for (int i=0; i<4; ++i) {
                                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, sycl::access::address_space::global_space>
                                    tally(CARBON_SECONDARY_CONTEXT_FIELD(he4_hazard_audit_device)[carried_slots[i]]);
                                tally.fetch_add(he4_audit[i]);
                            }
                        }
                        }
                        // Small per-step deposits must not contend directly on
                        // one global species scalar. Preserve original scoring
                        // guards and reduce per track; all replay breaks arrive
                        // here too. This changes diagnostics only, not dose.
                        cinel02_species_energy_add_device(
                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 1U,
                            continuous_species_tally.all_MeV);
                        cinel02_species_energy_add_device(
                            CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 2U,
                            continuous_species_tally.fov_MeV);
                        schneider_diag_add_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                  SchneiderDiagSlot::SecondaryRateQueries,
                                                  local_sec_rate_queries);
                        schneider_diag_add_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                                  SchneiderDiagSlot::SecondarySteps,
                                                  local_sec_steps);

                        const bool step_limited =
                            sec_e > CARBON_SECONDARY_CONTEXT_FIELD(energy_cutoff_MeV) && sec_steps >= kSecondaryMaxSteps;
                        if (sec_e > 0.0F) {
                            bool sec_terminal_scored = false;
                            if (step_limited) {
                                cinel02_species_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,  9U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::step_limit));
                                sec_terminal_recorded = true;
                                if (CARBON_SECONDARY_CONTEXT_FIELD(escaped_device) != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_esc(CARBON_SECONDARY_CONTEXT_FIELD(escaped_device)[frag.parent_history]);
                                    atomic_esc.fetch_add(sec_e);
                                    schneider_energy_add_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                        SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
                                }
                            } else if (!unified_secondary_escaped_ct && sec_z >= 0.0F && sec_z < CARBON_SECONDARY_CONTEXT_FIELD(phantom_length_mm)) {
                                cinel02_species_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 5U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::terminal_deposit));
                                sec_terminal_recorded = true;
                                const auto bin_x = static_cast<int>((sec_x - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm));
                                const auto bin_y = static_cast<int>((sec_y - CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm));
                                const auto bin_z = static_cast<int>(sec_z * CARBON_SECONDARY_CONTEXT_FIELD(inverse_depth_bin_width_mm));
                                if (bin_z >= 0 && bin_z < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                    if (bin_z != pending_sec_bin) {
                                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 &&
                                            pending_sec_bin < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dose(CARBON_SECONDARY_CONTEXT_FIELD(dose_device)[pending_sec_bin]);
                                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
                                            pending_sec_depth_MeV = 0.0F;
                                        }
                                        pending_sec_bin = bin_z;
                                    }
                                    pending_sec_depth_MeV += sec_e;
                                    if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring))) {
                                        int cur_voxel = -1;
                                        if (bin_x >= 0 && bin_x < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                            bin_y >= 0 && bin_y < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))) {
                                            cur_voxel = (bin_z * static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) + bin_y) *
                                                            static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) +
                                                        bin_x;
                                        }
                                        if (cur_voxel != pending_sec_voxel) {
                                            if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 &&
                                                pending_sec_voxel < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels))) {
                                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_vox(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[pending_sec_voxel]);
                                                atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[
                                                charged_origin_voxel_offset + pending_sec_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                            be_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_he_isotope_origin_voxel_device(
                                            CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                            he_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            component(CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                                minibeam_component_voxel_offset + pending_sec_voxel]);
                                        component.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                            }
                                            pending_sec_voxel_MeV = 0.0F;
                                            pending_sec_voxel = cur_voxel;
                                        }
                                        if (cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += sec_e;
                                            if ((kProductionSecondaryPath ? false :
                                                 CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_energy_band_roi_scoring))) {
                                                score_minibeam_energy_band_roi_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(minibeam_energy_band_roi_dose_device),
                                                    transport_z, transport_a, sec_e, cur_voxel,
                                                    CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins),
                                                    CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x),
                                                    CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y),
                                                    CARBON_SECONDARY_CONTEXT_FIELD(minibeam_fixed_region_by_x_bin_device), sec_e);
                                            }
                                            sec_terminal_scored = true;
                                            if (CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device) != nullptr) {
                                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_in_fov(CARBON_SECONDARY_CONTEXT_FIELD(in_fov_dose_device)[bin_z]);
                                                atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(sec_e));
                                                cinel02_species_energy_add_device(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx,
                                                    6U, sec_e);
                                            }
                                        }
                                    }
                                }
                                if (CARBON_SECONDARY_CONTEXT_FIELD(deposited_device) != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dep(CARBON_SECONDARY_CONTEXT_FIELD(deposited_device)[frag.parent_history]);
                                    atomic_dep.fetch_add(sec_e);
                                    schneider_energy_add_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV, sec_e);
                                    grid_deposit_split_device(
                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_in_device),
                                        CARBON_SECONDARY_CONTEXT_FIELD(grid_deposited_out_device),
                                        sec_terminal_scored, sec_e);
                                }
                            } else {
                                cinel02_species_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_energy_device), ledger_species_idx, 7U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::boundary_escape));
                                sec_terminal_recorded = true;
                                if (CARBON_SECONDARY_CONTEXT_FIELD(escaped_device) != nullptr) {
                                // Escaped phantom boundaries
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_esc(CARBON_SECONDARY_CONTEXT_FIELD(escaped_device)[frag.parent_history]);
                                atomic_esc.fetch_add(sec_e);
                                schneider_energy_add_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),
                                    SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
                                }
                            }
                        }

                        if (!sec_terminal_recorded) {
                            cinel02_species_terminal_increment_device(
                                CARBON_SECONDARY_CONTEXT_FIELD(cinel02_species_terminal_device), ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    Cinel02SpeciesLedgerSchema::continuous_stop));
                        }

                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(CARBON_SECONDARY_CONTEXT_FIELD(dose_device)[pending_sec_bin]);
                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_sec_depth_MeV));
                        }
                        if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring)) && pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels))) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_vox(CARBON_SECONDARY_CONTEXT_FIELD(voxel_dose_device)[pending_sec_voxel]);
                            atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            if ((kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(enable_charged_origin_voxel_scoring))) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_origin(CARBON_SECONDARY_CONTEXT_FIELD(charged_origin_voxel_dose_device)[
                                        charged_origin_voxel_offset + pending_sec_voxel]);
                                atomic_origin.fetch_add(
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                score_be_isotope_origin_voxel_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(be_isotope_origin_voxel_dose_device),
                                    be_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                    pending_sec_voxel,
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                score_he_isotope_origin_voxel_device(
                                    CARBON_SECONDARY_CONTEXT_FIELD(he_isotope_origin_voxel_dose_device),
                                    he_isotope_category, CARBON_SECONDARY_CONTEXT_FIELD(number_of_voxels),
                                    pending_sec_voxel,
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            }
                            if ((kProductionSecondaryPath ? false :
                                 CARBON_SECONDARY_CONTEXT_FIELD(enable_minibeam_component_voxel_scoring))) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    component(CARBON_SECONDARY_CONTEXT_FIELD(minibeam_component_voxel_dose_device)[
                                        minibeam_component_voxel_offset + pending_sec_voxel]);
                                component.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            }
                        }
                                        };
#if CARBON_SECONDARY_CONTEXT_POINTER
                static_assert(std::is_trivially_copyable_v<decltype(secondary_kernel)>);
                static_assert(sizeof(secondary_kernel) <= 64,
                              "secondary kernel closure exceeded its ABI budget");
#endif
#if CARBON_SECONDARY_EXPLICIT_ND_RANGE
                constexpr std::size_t secondary_local_size =
                    CARBON_SECONDARY_ND_RANGE_SIZE;
                const auto secondary_global_size =
                    ((static_cast<std::size_t>(active_size) + secondary_local_size - 1) /
                     secondary_local_size) * secondary_local_size;
                cgh.parallel_for<CarbonSecondaryTransportKernel<
                                                                 kProductionSecondaryPath,
                                                                 kExactSpecies, BroadBeam, Gaussian, JointResponse>>(
                    sycl::nd_range<1>{secondary_global_size, secondary_local_size},
                    secondary_kernel);
#else
                cgh.parallel_for<CarbonSecondaryTransportKernel<
                                                                 kProductionSecondaryPath,
                                                                 kExactSpecies, BroadBeam, Gaussian, JointResponse>>(
                    sycl::range<1>(active_size), secondary_kernel);
#endif
            });
                };
                double round_seconds = 0.0;
                sycl::event slice_events[3];
                unsigned slice_event_count = 0;
                const auto submit_secondary_slice = [&](auto production_path_tag,
                                                        auto exact_species_tag,
                                                        unsigned active_begin,
                                                        unsigned active_size) {
                    if(active_size==0)return;
                    slice_events[slice_event_count++] = launch_secondary_round(
                        production_path_tag,exact_species_tag,active_begin,active_size);
                };
                if constexpr(BroadBeam) {
                    if constexpr(JointResponse==3) {
                        if(broadbeam_ct_secondary_specialize)
                            submit_secondary_slice(std::true_type{},std::integral_constant<int,-1>{},0,resume_active);
                        else submit_secondary_slice(std::false_type{},std::integral_constant<int,-1>{},0,resume_active);
                    } else
                    submit_secondary_slice(std::false_type{},std::integral_constant<int,-1>{},0,resume_active);
                } else {
#if CARBON_SECONDARY_PRODUCTION_SPECIALIZE
                if(use_production_secondary_path) {
#if CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE
                    if(group_secondaries) {
                        submit_secondary_slice(std::true_type{},
                            std::integral_constant<int,0>{},0,species0_active);
                        submit_secondary_slice(std::true_type{},
                            std::integral_constant<int,1>{},species0_active,
                            species1_active);
                        const unsigned fallback_begin=species0_active+species1_active;
                        submit_secondary_slice(std::true_type{},
                            std::integral_constant<int,-1>{},fallback_begin,
                            resume_active-fallback_begin);
                    } else
#endif
                    submit_secondary_slice(std::true_type{},
                        std::integral_constant<int,CARBON_SECONDARY_EXACT_SPECIES_PROBE>{},
                        0,resume_active);
                } else {
                    submit_secondary_slice(std::false_type{},
                        std::integral_constant<int,-1>{},0,resume_active);
                }
#else
                submit_secondary_slice(std::false_type{},
                    std::integral_constant<int,-1>{},0,resume_active);
#endif
                }
                // The queue is in order, so waiting on the final slice also
                // completes every earlier slice in this round.
                if (slice_event_count != 0)
                    slice_events[slice_event_count - 1].wait_and_throw();
                for (unsigned s=0; s<slice_event_count; ++s)
                    round_seconds += event_duration_seconds(slice_events[s]);
                secondary_kernel_seconds += round_seconds;
                generation_kernel_seconds += round_seconds;
                if (finish_tail) {
                    generation_tail_seconds += round_seconds;
                    ++tail_rounds;
                }
                if (secondary_tail_diag) {
                    std::cout << "[secondary-round] begin=" << batch_begin
                              << " round=" << segment_rounds << " active=" << resume_active
                              << " blocks256=" << ((resume_active + 255) / 256)
                              << (finish_tail ? " finish_tail" : " segment")
                              << " kernel_s=" << round_seconds << "\n";
                }
                ++segment_rounds;
                if (finish_tail) break; // No suspended tracks; avoid an empty compaction.
                const auto compact_start=std::chrono::steady_clock::now();
                const unsigned resume_groups=(resume_active+255)/256;
                queue.parallel_for(sycl::nd_range<1>(resume_groups*256,256),[=](sycl::nd_item<1> it){
                    const auto i=it.get_global_linear_id();const unsigned flag=i<resume_active?keep[i]:0;
                    const auto rank=sycl::exclusive_scan_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    const auto count=sycl::reduce_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    if(i<resume_active)ranks[i]=rank;
                    if(it.get_local_linear_id()==0)block_counts[it.get_group_linear_id()]=count;
                });
                if constexpr(CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE) {
                    // The compaction below is order-preserving, so the survivor
                    // counts before the old slice boundaries are exactly
                    // block_offsets[i/256]+ranks[i]. Fold them into the prefix
                    // single_task instead of a second boundary kernel.
                    queue.single_task([=](){
                        unsigned sum=0;
                        for(unsigned i=0;i<resume_groups;++i){block_offsets[i]=sum;sum+=block_counts[i];}
                        active_count[0]=sum;
                        const unsigned p_end=species0_active;
                        const unsigned d_end=species0_active+species1_active;
                        active_count[1]=(p_end==0u)?0u:((p_end>=resume_active)?sum:(block_offsets[p_end/256]+ranks[p_end]));
                        active_count[2]=(d_end==0u)?0u:((d_end>=resume_active)?sum:(block_offsets[d_end/256]+ranks[d_end]));
                    });
                    queue.parallel_for(sycl::range<1>(resume_active),[=](sycl::id<1> id){const auto i=id[0];if(keep[i])next_order[block_offsets[i/256]+ranks[i]]=active_order[i];});
                    unsigned counts[3]{};
                    queue.copy(active_count,counts,3).wait_and_throw();
                    resume_active=counts[0];
                    species0_active=counts[1];
                    species1_active=counts[2]-counts[1];
                } else {
                    queue.single_task([=](){unsigned sum=0;for(unsigned i=0;i<resume_groups;++i){block_offsets[i]=sum;sum+=block_counts[i];}*active_count=sum;});
                    queue.parallel_for(sycl::range<1>(resume_active),[=](sycl::id<1> id){const auto i=id[0];if(keep[i])next_order[block_offsets[i/256]+ranks[i]]=active_order[i];});
                    queue.copy(active_count,&resume_active,1).wait_and_throw();
                }
                std::swap(active_order,next_order);
                compact_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-compact_start).count();
                }
                if (segment_secondaries) std::cout<<"[segmented-secondary] rounds="<<segment_rounds
                    <<" tail_rounds="<<tail_rounds<<" tail_s="<<generation_tail_seconds
                    <<" kernel_s="<<generation_kernel_seconds
                    <<" compaction_s="<<compact_seconds
                    <<" begin="<<batch_begin<<" end="<<generation_end<<"\n";
#if CARBON_SECONDARY_CONTEXT_POINTER
                mem_tracker.free(secondary_transport_ctx);
#endif
                mem_tracker.free(resume_states);
                for(auto* ptr:{resume_ready,active_order,next_order,keep,ranks,block_counts,block_offsets,active_count})mem_tracker.free(ptr);
                const auto batch_end = generation_end;
                generation_begin = batch_end;
                queue.copy(secondary_count_device, &secondary_count_host, 1)
                    .wait_and_throw();
                generation_end = sycl::min(
                    secondary_count_host, static_cast<std::uint32_t>(max_secondaries));
                std::cout << "[progress] secondary batch completed: ["
                          << batch_begin << ", " << batch_end << ") particles; queued="
                          << generation_end << std::endl;
            }
            std::cout << "[secondary-schedule] grouping_seconds="
                      << secondary_group_seconds << "\n";
            if (!config.fragment_birth_spectrum_output_file.empty()) {
                birth_secondaries_host.resize(generation_end);
                queue.copy(secondary_queue_device, birth_secondaries_host.data(),
                           generation_end).wait_and_throw();
            }
        }
    }
    if (birth_secondaries_host.empty() &&
        !config.fragment_birth_spectrum_output_file.empty() &&
        secondary_count_device != nullptr && secondary_queue_device != nullptr) {
        std::uint32_t birth_count = 0U;
        queue.copy(secondary_count_device, &birth_count, 1).wait_and_throw();
        birth_count = sycl::min(
            birth_count, static_cast<std::uint32_t>(max_secondaries));
        if (birth_count > 0U) {
            birth_secondaries_host.resize(birth_count);
            queue.copy(secondary_queue_device, birth_secondaries_host.data(),
                       birth_count).wait_and_throw();
        }
    }

    runtime_steps.finish();
    RuntimeScope runtime_post("transport_readback_and_host_finalize");
    std::vector<DepthAtomicT> dose_device_host(number_of_bins);
    queue.copy(dose_device, dose_device_host.data(), number_of_bins).wait_and_throw();
    std::vector<double> dose_host(number_of_bins);
    std::transform(dose_device_host.begin(), dose_device_host.end(), dose_host.begin(),
                   [](DoseAtomicT val) { return static_cast<double>(val); });

    std::vector<double> voxel_dose_host;
    if (enable_voxel_scoring) {
        std::vector<DoseAtomicT> voxel_dose_device_host(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_device_host.data(), number_of_voxels)
            .wait_and_throw();
        voxel_dose_host.resize(number_of_voxels);
        std::transform(voxel_dose_device_host.begin(), voxel_dose_device_host.end(),
                       voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> primary_voxel_track_length_host;
    if (enable_primary_voxel_fluence) {
        std::vector<DoseAtomicT> device_host(number_of_voxels);
        queue.copy(primary_voxel_track_length_device, device_host.data(),
                   number_of_voxels).wait_and_throw();
        primary_voxel_track_length_host.resize(number_of_voxels);
        std::transform(device_host.begin(), device_host.end(),
                       primary_voxel_track_length_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> charged_origin_voxel_dose_host;
    std::vector<double> minibeam_component_voxel_dose_host;
    std::vector<double> minibeam_energy_band_roi_dose_host;
    std::array<double, 6> he4_hazard_audit_host{};
    if (he4_hazard_audit_device)
        queue.copy(he4_hazard_audit_device, he4_hazard_audit_host.data(), 6).wait_and_throw();
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            charged_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(charged_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        charged_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       charged_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    if (enable_minibeam_component_voxel_scoring) {
        const auto value_count =
            minibeam_component_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(minibeam_component_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        minibeam_component_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_component_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    if (enable_minibeam_energy_band_roi_scoring) {
        std::vector<DoseAtomicT> device_host(
            minibeam_energy_band_roi_value_count);
        queue.copy(minibeam_energy_band_roi_dose_device, device_host.data(),
                   minibeam_energy_band_roi_value_count).wait_and_throw();
        minibeam_energy_band_roi_dose_host.resize(
            minibeam_energy_band_roi_value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_energy_band_roi_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::vector<double> minibeam_primary_roi_dose_host;
    if (enable_minibeam_primary_roi_scoring) {
        std::vector<DoseAtomicT> device_host(minibeam_primary_roi_value_count);
        queue.copy(minibeam_primary_roi_dose_device, device_host.data(),
                   minibeam_primary_roi_value_count).wait_and_throw();
        minibeam_primary_roi_dose_host.resize(minibeam_primary_roi_value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_primary_roi_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::array<std::uint64_t, minibeam_spatial_audit_slot_count>
        minibeam_spatial_audit_host{};
    if (enable_minibeam_primary_roi_scoring) {
        queue.copy(minibeam_spatial_audit_device, minibeam_spatial_audit_host.data(),
                   minibeam_spatial_audit_slot_count).wait_and_throw();
    }

    std::vector<double> be_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            be_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(be_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        be_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       be_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::vector<double> he_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            he_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(he_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        he_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       he_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> in_fov_dose_host;
    if (enable_voxel_scoring && in_fov_dose_device != nullptr) {
        std::vector<DepthAtomicT> in_fov_device_host(number_of_bins);
        queue.copy(in_fov_dose_device, in_fov_device_host.data(), number_of_bins).wait_and_throw();
        in_fov_dose_host.resize(number_of_bins);
        std::transform(in_fov_device_host.begin(), in_fov_device_host.end(),
                       in_fov_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<LetAtomicT> let_moments_host;
    if (enable_let_scoring) {
        let_moments_host.resize(4 * number_of_bins);
        queue.copy(let_moments_device, let_moments_host.data(), 4 * number_of_bins)
            .wait_and_throw();
    }

    std::vector<LetAtomicT> voxel_let_moments_host;
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        voxel_let_moments_host.resize(4 * number_of_voxels);
        queue.copy(voxel_let_moments_device, voxel_let_moments_host.data(), 4 * number_of_voxels)
            .wait_and_throw();
    }

    std::vector<std::uint64_t> primary_survival_host;
    std::array<std::uint32_t,12> elastic_counts_host{};
    std::array<float,3> elastic_energy_host{};
    if(use_all_elastic) {
        auto& counts=elastic_counts_host;queue.copy(elastic_audit,counts.data(),12);
        queue.copy(elastic_energy,elastic_energy_host.data(),3).wait_and_throw();
        std::cout<<"[all-ion-elastic] primary="<<counts[0]<<" secondary="<<counts[1]
                 <<" bad_queries="<<counts[2]<<" unsupported_recoils="<<counts[3]<<'\n';
        std::cout<<"[elastic-failure-sites] primary_rate="<<counts[4]<<" primary_draw="<<counts[5]<<" recoil_stopping="<<counts[6]<<" secondary_rate="<<counts[7]<<" secondary_draw="<<counts[8]<<" null_post_em="<<counts[9]<<'\n';
        if(counts[2]||counts[3])throw std::runtime_error("All-ion elastic coverage failure: missing bank query or recoil transport; dose rejected");
    }
    std::vector<std::uint64_t> inelastic_reaction_host;
    if (primary_survival_device != nullptr) {
        primary_survival_host.resize(number_of_bins);
        inelastic_reaction_host.resize(number_of_bins);
        queue.copy(primary_survival_device, primary_survival_host.data(), number_of_bins);
        queue.copy(inelastic_reaction_device, inelastic_reaction_host.data(), number_of_bins);
    }
    std::uint64_t neutron_birth_count_host = 0;
    std::vector<NeutronBirthRecord> neutron_birth_host;
    if (neutron_birth_buffer.records) {
        queue.copy(neutron_birth_buffer.count, &neutron_birth_count_host, 1).wait_and_throw();
        const auto kept = std::min(neutron_birth_count_host, neutron_birth_buffer.capacity);
        neutron_birth_host.resize(kept);
        if (kept) queue.copy(neutron_birth_buffer.records, neutron_birth_host.data(), kept).wait_and_throw();
        free_device(neutron_birth_buffer.records);
        free_device(neutron_birth_buffer.count);
    }
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> sampled_incident_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<float> untracked_host(number_of_histories, 0.0F);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(sampled_incident_device, sampled_incident_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories);
    if (untracked_nuclear_device != nullptr) {
        queue.copy(untracked_nuclear_device, untracked_host.data(), number_of_histories);
    }
    std::vector<float> other_terminal_host(number_of_histories, 0.0F);
    if (other_terminal_energy_device != nullptr) {
        queue.copy(other_terminal_energy_device, other_terminal_host.data(), number_of_histories);
    }
    std::vector<float> cutoff_stopped_host(number_of_histories, 0.0F);
    if (cutoff_stopped_energy_device != nullptr) {
        queue.copy(cutoff_stopped_energy_device, cutoff_stopped_host.data(), number_of_histories);
    }
#if defined(CARBON_ENABLE_MINIBEAM)
    std::vector<float> beamline_removed_host;
    std::vector<float> beamline_primary_survivor_host;
    std::vector<float> beamline_air_loss_host;
    std::array<std::uint64_t, minibeam_event_counter_count>
        minibeam_event_counts_host{};
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_counts_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_energy_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_depth_host;
    std::vector<std::uint64_t> minibeam_fragment_miss_joint_remaining_host;
    std::vector<MinibeamPhaseSpaceRecord> minibeam_phase_space_host;
    std::vector<CopperHistoryDiagnostic> copper_diagnostic_host;
    std::vector<PrimaryMediumEntryState> copper_capture_entry_host;
    if(copper_capture_entry_device) {
        copper_capture_entry_host.resize(number_of_histories);
        queue.copy(copper_capture_entry_device,copper_capture_entry_host.data(),number_of_histories).wait_and_throw();
    }
    if (copper_diagnostic_device) {
        copper_diagnostic_host.resize(number_of_histories);
        queue.copy(copper_diagnostic_device,copper_diagnostic_host.data(),number_of_histories).wait_and_throw();
    }
    std::vector<MinibeamWaterPrimaryPlaneRecord>
        minibeam_water_primary_plane_records_host;
    std::vector<MinibeamFragmentPhaseSpaceRecord>
        minibeam_fragment_phase_space_host;
    if (beamline_removed_device != nullptr) {
        beamline_removed_host.resize(number_of_histories);
        beamline_primary_survivor_host.resize(number_of_histories);
        beamline_air_loss_host.resize(number_of_histories);
        queue.copy(beamline_removed_device, beamline_removed_host.data(),
                   number_of_histories);
        queue.copy(beamline_primary_survivor_device,
                   beamline_primary_survivor_host.data(), number_of_histories);
        queue.copy(beamline_air_loss_device,
                   beamline_air_loss_host.data(), number_of_histories);
        queue.copy(minibeam_event_counts_device,
                   minibeam_event_counts_host.data(),
                   minibeam_event_counts_host.size());
        queue.wait_and_throw();
        if (copper_proton_native_fluctuation) {
            if(air_ion_count)std::cout<<"AIR_IONIZATION_CLOCK steps="
                <<minibeam_event_counts_host[air_ion_step_slot]<<" proposals="
                <<minibeam_event_counts_host[air_ion_proposal_slot]<<'\n';
            std::cout << "COPPER_PROTON_LOSS_QUALITY failures="
                      << minibeam_event_counts_host[copper_proton_loss_failure_slot]
                      << " primary=" << minibeam_event_counts_host[copper_proton_primary_loss_slot]
                      << " fragment=" << minibeam_event_counts_host[copper_proton_fragment_loss_slot] << '\n';
            if (minibeam_event_counts_host[copper_proton_loss_failure_slot])
                throw std::runtime_error("Invalid copper proton fluctuation draw: reject partial dose");
        }


        if (minibeam_fragment_miss_joint_counts_device != nullptr) {
            minibeam_fragment_miss_joint_counts_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_energy_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_depth_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            minibeam_fragment_miss_joint_remaining_host.resize(
                minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_counts_device,
                       minibeam_fragment_miss_joint_counts_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_energy_device,
                       minibeam_fragment_miss_joint_energy_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_depth_device,
                       minibeam_fragment_miss_joint_depth_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
            queue.copy(minibeam_fragment_miss_joint_remaining_device,
                       minibeam_fragment_miss_joint_remaining_host.data(),
                       minibeam_fragment_miss_joint_cell_count);
        }
        if (minibeam_phase_space_device != nullptr) {
            minibeam_phase_space_host.resize(number_of_histories);
            queue.copy(minibeam_phase_space_device,
                       minibeam_phase_space_host.data(), number_of_histories);
        }
        if (minibeam_water_primary_plane_records_device != nullptr) {
            minibeam_water_primary_plane_records_host.resize(
                number_of_histories * minibeam_water_primary_plane_count);
            queue.copy(
                minibeam_water_primary_plane_records_device,
                minibeam_water_primary_plane_records_host.data(),
                minibeam_water_primary_plane_records_host.size());
        }
        if (minibeam_fragment_phase_space_device != nullptr) {
            std::uint32_t count = 0;
            queue.copy(minibeam_fragment_phase_space_count_device, &count, 1)
                .wait_and_throw();
            count = std::min<std::uint32_t>(
                count, static_cast<std::uint32_t>(max_secondaries));
            minibeam_fragment_phase_space_host.resize(count);
            if (count != 0) {
                queue.copy(minibeam_fragment_phase_space_device,
                           minibeam_fragment_phase_space_host.data(), count);
            }
        }
    }
#endif
        if(uses_urban_package) {
            std::vector<std::uint64_t> counts(2*urban_species_host.size());
            queue.copy(urban_step_counts,counts.data(),counts.size()).wait_and_throw();
            for(unsigned i=0;i<urban_species_host.size();++i)
                std::cout<<"[urban-v2-steps] Z="<<urban_species_host[i].z<<" A="<<urban_species_host[i].a
                         <<" primary="<<counts[2*i]<<" secondary="<<counts[2*i+1]<<"\n";
        }
        {
        std::array<std::uint64_t,kUnifiedEmAuditCounters*kUnifiedEmAuditShards> shard_audit{};
        queue.copy(unified_audit,shard_audit.data(),shard_audit.size()).wait_and_throw();
        std::uint64_t audit[8]{};
        for(unsigned shard=0;shard<kUnifiedEmAuditShards;++shard)
            for(unsigned counter=0;counter<kUnifiedEmAuditCounters;++counter)
                audit[counter]+=shard_audit[static_cast<std::size_t>(counter)*kUnifiedEmAuditShards+shard];
        std::cout<<"[unified-em-audit]";for(auto count:audit)std::cout<<" "<<count;std::cout<<"\n";
        if(audit[0]) {
            unsigned count=0;queue.copy(unified_failure_count,&count,1).wait_and_throw();
            std::vector<UnifiedEmFailureRecord> records(std::min(count,16u));
            queue.copy(unified_failure_records,records.data(),records.size()).wait_and_throw();
            for(const auto& f:records)std::cerr<<"[unified-em-failure] reason="<<f.reason<<" section="<<f.section<<" Z="<<f.z<<" A="<<f.a<<" kinetic_MeV="<<f.energy<<" density="<<f.density<<" step_mm="<<f.step<<" mean="<<f.d0<<" delta_mean="<<f.d1<<" delta_variance="<<f.d2<<"\n";
            std::cerr<<"[runtime-rejected-kernel] primary_s="<<primary_kernel_seconds<<" secondary_s="<<secondary_kernel_seconds<<" histories="<<number_of_histories<<"\n";
            throw std::runtime_error("Unified EM missing domain or sampling failure; dose rejected");
        }
        mem_tracker.free(unified_failure_count);mem_tracker.free(unified_failure_records);
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
            std::uint64_t secprof[60]{};queue.copy(sec_step_profile_device,secprof,60).wait_and_throw();
            std::cout<<"[secondary-step-profile] slot steps short depMeV\n";
            for(int sec_si=0;sec_si<20;++sec_si)
                std::cout<<"[secondary-step-profile] "<<sec_si<<" "<<secprof[sec_si]<<" "<<secprof[20+sec_si]<<" "<<(static_cast<double>(secprof[40+sec_si])*1e-6)<<"\n";
        }
        if constexpr(CARBON_EM_EMPTY_BUCKET_AUDIT) {
            std::uint64_t counts[10]{};
            queue.copy(unified_empty_bucket_audit,counts,10).wait_and_throw();
            constexpr const char* names[5]={"node","raw0","raw1","raw2","raw3"};
            for(unsigned kind=0;kind<5;++kind) {
                const auto total=counts[2*kind];
                const auto empty=counts[2*kind+1];
                const double fraction=total?static_cast<double>(empty)/static_cast<double>(total):0.0;
                std::cout<<"[em-empty-bucket-audit] kind="<<names[kind]
                         <<" total="<<total<<" empty="<<empty
                         <<" fraction="<<fraction<<"\n";
            }
        }
        if constexpr(CARBON_EM_INTERVAL_WIDTH_AUDIT) {
            std::uint64_t widths[15]{};
            queue.copy(unified_interval_width_audit,widths,15).wait_and_throw();
            constexpr const char* names[3]={"node","raw0","raw1"};
            for(unsigned kind=0;kind<3;++kind) {
                std::uint64_t total=0;
                for(unsigned w=0;w<5;++w)total+=widths[kind*5+w];
                std::cout<<"[em-interval-width-audit] kind="<<names[kind]<<" total="<<total;
                for(unsigned w=0;w<4;++w)
                    std::cout<<" w"<<w<<"="<<widths[kind*5+w];
                std::cout<<" w>=4="<<widths[kind*5+4]<<"\n";
            }
        }
        if constexpr(CARBON_EM_SEARCH_KEY_AUDIT) {
            std::uint32_t observed=0;
            queue.copy(unified_search_audit_count,&observed,1).wait_and_throw();
            const auto sampled=std::min(observed,kUnifiedSearchAuditCapacity);
            std::vector<UnifiedEmSearchAuditRecord> records(sampled);
            queue.copy(unified_search_audit_records,records.data(),sampled).wait_and_throw();
            const auto group_key=[](const auto& r){
                return std::tuple{r.kind,r.launch,r.warp,r.step,r.ordinal};
            };
            std::sort(records.begin(),records.end(),[&](const auto& a,const auto& b){
                return std::tuple{group_key(a),a.key}<std::tuple{group_key(b),b.key};
            });
            std::uint64_t groups[5]{},lanes[5]{},unique[5]{},shared4[5]{};
            for(std::size_t begin=0;begin<records.size();) {
                std::size_t end=begin+1;
                while(end<records.size() && group_key(records[end])==group_key(records[begin]))++end;
                std::uint64_t u=1;
                for(std::size_t i=begin+1;i<end;++i)if(records[i].key!=records[i-1].key)++u;
                const auto kind=records[begin].kind;
                ++groups[kind];lanes[kind]+=end-begin;unique[kind]+=u;
                if(u<=4)++shared4[kind];
                begin=end;
            }
            constexpr const char* names[5]={"node","raw0","raw1","raw2","raw3"};
            std::cout<<"[em-search-key-audit] observed="<<observed
                     <<" sampled="<<sampled<<"\n";
            for(unsigned kind=0;kind<5;++kind) {
                std::cout<<"[em-search-key-audit] kind="<<names[kind]
                         <<" groups="<<groups[kind]<<" lanes="<<lanes[kind]
                         <<" unique="<<unique[kind]
                         <<" mean_unique="<<(groups[kind]?double(unique[kind])/groups[kind]:0.0)
                         <<" groups_unique_le4="<<shared4[kind]<<"\n";
            }
        }
        mem_tracker.free(delta_mean_device);mem_tracker.free(unified_energy_index);mem_tracker.free(unified_sections);mem_tracker.free(unified_materials);mem_tracker.free(unified_species);mem_tracker.free(unified_records);
        mem_tracker.free(unified_nodes);mem_tracker.free(unified_segments);mem_tracker.free(unified_audit);
        if constexpr(CARBON_EM_EMPTY_BUCKET_AUDIT) mem_tracker.free(unified_empty_bucket_audit);
        if constexpr(CARBON_EM_INTERVAL_WIDTH_AUDIT) mem_tracker.free(unified_interval_width_audit);
        if constexpr(CARBON_EM_SEARCH_KEY_AUDIT) {
            mem_tracker.free(unified_search_audit_records);
            mem_tracker.free(unified_search_audit_count);
        }
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) mem_tracker.free(sec_step_profile_device);
    }
    std::uint64_t schneider_inelastic_host = 0;
    if (schneider_inelastic_device != nullptr) {
        queue.copy(schneider_inelastic_device, &schneider_inelastic_host, 1);
    }
    std::array<std::uint64_t, 4> terminal_counts_host{};
    if (primary_terminal_counts_device != nullptr) {
        queue.copy(primary_terminal_counts_device, terminal_counts_host.data(), 4).wait_and_throw();
    }
    std::vector<PrimaryFirstInteractionRecord> first_interactions_host;
    if (first_interactions_device != nullptr && first_interactions_count_device != nullptr) {
        std::uint32_t first_int_count = 0;
        queue.copy(first_interactions_count_device, &first_int_count, 1).wait_and_throw();
        const auto actual_count = std::min(first_int_count, static_cast<std::uint32_t>(number_of_histories));
        first_interactions_host.resize(actual_count);
        if (actual_count > 0) {
            queue.copy(first_interactions_device, first_interactions_host.data(), actual_count).wait_and_throw();
        }
    }
    // Only shared species and grid ledgers are backed by device buffers.
    std::array<float, kCinel02SpeciesEnergySlots> cinel02_species_energy_host{};
    std::array<std::uint64_t, kCinel02SpeciesTerminalSlots> cinel02_species_terminal_host{};
    double grid_deposited_in_host = 0.0;
    double grid_deposited_out_host = 0.0;
    queue.copy(cinel02_species_energy_device, cinel02_species_energy_host.data(),
               kCinel02SpeciesEnergySlots);
    queue.copy(cinel02_species_terminal_device, cinel02_species_terminal_host.data(),
               kCinel02SpeciesTerminalSlots);
    if (grid_deposited_in_device != nullptr)
        queue.copy(grid_deposited_in_device, &grid_deposited_in_host, 1);
    if (grid_deposited_out_device != nullptr)
        queue.copy(grid_deposited_out_device, &grid_deposited_out_host, 1);
    std::array<std::uint64_t, kSchneiderDiagSlots> schneider_diag_host{};
    static_assert(kSchneiderDiagSlots ==
                      static_cast<std::size_t>(SchneiderDiagSlot::Count),
                  "Schneider host mirror must match the device schema");
    std::array<float, 12> schneider_float_host{};
    static_assert(12 == static_cast<std::size_t>(SchneiderFloatSlot::Count),
                  "Schneider float mirror must match the device schema");
    if (schneider_diag_device != nullptr) {
        queue.copy(schneider_diag_device, schneider_diag_host.data(), kSchneiderDiagSlots);
    }
    if (schneider_float_device != nullptr) {
        queue.copy(schneider_float_device, schneider_float_host.data(), kSchneiderFloatSlots);
    }
    std::array<std::uint32_t, 2> schneider_miss_counts_host{0, 0};
    std::array<std::uint32_t, 2> schneider_track_counts_host{0, 0};
    if (schneider_miss_count_device != nullptr) {
        queue.copy(schneider_miss_count_device, schneider_miss_counts_host.data(), 2);
    }
    if (schneider_track_count_device != nullptr) {
        queue.copy(schneider_track_count_device, schneider_track_counts_host.data(), 2);
    }
    uint32_t overflow_count_host = 0;
    float overflow_energy_host = 0.0F;
    if (secondary_overflow_count_device != nullptr) {
        queue.copy(secondary_overflow_count_device, &overflow_count_host, 1);
        queue.copy(secondary_overflow_energy_device, &overflow_energy_host, 1);
    }
    queue.wait_and_throw();

    // Electron worker ledgers are private FP32 sums. Widen only on the host
    // after transport, avoiding a shared FP64 atomic in the full FP32 worker.
    if(float_electron_grid_ledger) {
        std::vector<float> partial(2*material_queue_lanes);
        queue.copy(float_electron_grid_ledger,partial.data(),partial.size()).wait_and_throw();
        for(unsigned lane=0;lane<material_queue_lanes;++lane) {
            grid_deposited_in_host+=partial[2*lane];
            grid_deposited_out_host+=partial[2*lane+1];
        }
    }
    double electron_vr_gain_host=0;
    if(float_electron_vr_ledger) {
        std::vector<float> partial(material_queue_lanes);std::array<std::uint64_t,2> work{};
        queue.copy(float_electron_vr_ledger,partial.data(),partial.size());
        queue.copy(electron_vr_work,work.data(),2).wait_and_throw();
        for(const float x:partial)electron_vr_gain_host+=x;
        if(work[0]||work[1])
            std::cout<<"[electron-variance-reduction-work] roulette_killed="<<work[0]<<" roulette_survived="<<work[1]
                     <<" weight_gain_MeV="<<electron_vr_gain_host<<"\n";
    }

    // Per-record Schneider logs: counts are exact after the fence above.
    // Staged into host vectors here (device buffers are freed below);
    // moved into TransportResult after its declaration.
    std::vector<SchneiderMissRecord> schneider_miss_host;
    std::vector<SchneiderUnsupportedTrack> schneider_track_host;
    if (schneider_miss_device != nullptr && schneider_miss_counts_host[0] > 0) {
        const auto n_miss = std::min<std::uint32_t>(
            schneider_miss_counts_host[0], kSchneiderMissLogCap);
        schneider_miss_host.resize(n_miss);
        queue.copy(schneider_miss_device, schneider_miss_host.data(), n_miss)
            .wait_and_throw();
    }
    if (schneider_track_log_device != nullptr && schneider_track_counts_host[0] > 0) {
        const auto n_trk = std::min<std::uint32_t>(
            schneider_track_counts_host[0], kSchneiderTrackLogCap);
        schneider_track_host.resize(n_trk);
        queue.copy(schneider_track_log_device, schneider_track_host.data(), n_trk)
            .wait_and_throw();
    }

    if (primary_loss_query_audit) {
        std::array<std::uint64_t,28> audit{};
        queue.copy(primary_loss_query_audit, audit.data(), audit.size()).wait_and_throw();
        free_device(primary_loss_query_audit);
        for (int b=0;b<14;++b)
            std::cout << "PRIMARY_LOSS_QUERY," << b << ',' << audit[b] << ',' << audit[b+14] << '\n';
    }
    std::array<std::uint64_t,7> electron_joint_diag_host{};
    std::array<std::uint64_t,3> material_untracked_host{};
    if(material_untracked_device)queue.copy(material_untracked_device,material_untracked_host.data(),3).wait_and_throw();

    if(material_failure_device && material_untracked_host[2]) {
        MaterialPacketFailure f;queue.copy(material_failure_device,&f,1).wait_and_throw();
        const auto& c=f.before.cursor;
        std::cerr<<"[material-packet-failure] history="<<f.history<<" step="<<f.step<<" rng="<<f.rng
                 <<" status="<<int(f.after.status)<<" gap="<<int(f.after.gap)<<" section="<<c.section
                 <<" rho="<<c.density_g_cm3<<" physical_rho="<<c.physical_density_g_cm3
                 <<" pdg="<<c.pdg<<" KE="<<c.energy_MeV<<" W="<<f.before.weight_MeV
                 <<" position="<<c.position[0]<<','<<c.position[1]<<','<<c.position[2]<<'\n';
    }
    if(electron_joint_diag_device)queue.copy(electron_joint_diag_device,electron_joint_diag_host.data(),7).wait_and_throw();
    if(use_electron_fp32) {
        if(electron_joint_diag_host[6]!=electron_tasks_total)
            throw std::runtime_error("Independent FP32 electron queue completion mismatch; batch rejected");
        std::cout<<"[independent-fp32-completion] expected="<<electron_tasks_total
                 <<" completed="<<electron_joint_diag_host[6]<<"\n";
    }
    // Free buffers
    free_device(electron_joint_diag_device);
    free_immutable_device(cross_section_device);
    free_device(target_h_fraction_device);
    free_device(dose_device);
    free_device(primary_survival_device);
    free_device(inelastic_reaction_device);
    free_device(primary_terminal_counts_device);
    free_device(first_interactions_device);
    free_device(first_interactions_count_device);
    free_device(in_fov_dose_device);
    free_device(voxel_dose_device);
    free_device(primary_voxel_track_length_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(minibeam_component_voxel_dose_device);
    free_device(minibeam_energy_band_roi_dose_device);
    free_device(minibeam_primary_roi_dose_device);
    free_device(minibeam_spatial_audit_device);
    free_device(minibeam_fixed_region_by_x_bin_device);
    free_device(be_isotope_origin_voxel_dose_device);
    free_device(he4_hazard_audit_device);
    free_device(he_isotope_origin_voxel_dose_device);
    free_device(let_moments_device);
    free_device(voxel_let_moments_device);
    free_device(deposited_device);
    free_device(secondary_queue_device);
    free_device(generic_recoil_reference_sp_device);
    free_device(secondary_count_device);
    free_device(cinel02_interactions_device);
    free_device(cinel02_products_device);
    free_device(cinel02_energy_nodes_device);
    free_device(cinel02_event_offsets_device);
    free_device(cinel02_event_indices_device);
    free_device(cinel02_rate_groups_device);
    free_device(cinel02_rate_samples_device);
    free_device(cinel02_ct_rate_groups_device);
    free_device(cinel02_ct_rate_samples_device);
    free_device(cinel02_species_energy_device);
    free_device(cinel02_species_terminal_device);
    free_device(untracked_nuclear_device);
    free_device(other_terminal_energy_device);
    free_device(cutoff_stopped_energy_device);
#if defined(CARBON_ENABLE_MINIBEAM)
    free_device(copper_diagnostic_device);
    free_device(copper_capture_entry_device);
    free_device(copper_history_mask_device);
    free_device(minibeam_phase_space_device);
    free_device(minibeam_water_primary_plane_depths_device);
    free_device(minibeam_water_primary_plane_records_device);
    free_device(minibeam_fragment_phase_space_device);
    free_device(minibeam_fragment_phase_space_count_device);
    free_device(minibeam_copper_cascade_queue_device);
    free_device(minibeam_copper_cascade_count_device);
    free_device(beamline_removed_device);
    free_device(beamline_primary_survivor_device);
    free_device(beamline_air_loss_device);
    free_device(minibeam_event_counts_device);

    free_device(minibeam_fragment_miss_joint_counts_device);
    free_device(minibeam_fragment_miss_joint_energy_device);
    free_device(minibeam_fragment_miss_joint_depth_device);
    free_device(minibeam_fragment_miss_joint_remaining_device);
    free_device(minibeam_copper_sp_energies_device);
    free_device(minibeam_copper_sp_values_device);
    free_device(minibeam_copper_loss_e_device);
    free_device(minibeam_copper_loss_r_device);
    free_device(minibeam_copper_loss_d_device);

    free_device(air_ion_e_device);
    free_device(air_ion_rate_device);
    free_device(minibeam_air_sp_energies_device);
    free_device(minibeam_air_sp_values_device);
    free_device(minibeam_copper_xs_energies_device);
    free_device(minibeam_copper_xs_values_device);
    free_device(minibeam_copper_elastic_device);
    free_device(minibeam_copper_nodes_device);
    free_device(minibeam_copper_offsets_device);
    free_device(minibeam_copper_indices_device);
    free_device(minibeam_copper_interactions_device);
    free_device(minibeam_copper_products_device);
#endif
    free_device(secondary_overflow_count_device);
    free_device(secondary_overflow_energy_device);
    free_device(escaped_device);
    free_device(sampled_incident_device);
    free_device(steps_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(slab_radiation_lengths_device);
    if(!retained_ct_arrays) {
        free_device(ct_density_device);
        free_device(ct_material_device);
    }
    free_device(ct_xs_device);
    free_device(broadbeam_electron_mask);
    free_device(broadbeam_birth_regions);free_device(broadbeam_region_work);
    free_device(schneider_primary_xs_device);
    free_device(schneider_inelastic_device);
    free_device(schneider_diag_device);
    free_device(schneider_float_device);
    free_device(schneider_miss_device);
    free_device(schneider_track_log_device);
    free_device(schneider_miss_count_device);
    free_device(schneider_track_count_device);

    TransportResult result;
    result.fe_sampling_diagnostics_enabled=fe_diagnostics;
    if (fe_counts) {
        queue.copy(fe_counts,result.fe_sampling_counts.data(),8).wait_and_throw();
        free_device(fe_counts);
    }
    result.primary_elastic_interactions=elastic_counts_host[0];
    result.secondary_elastic_interactions=elastic_counts_host[1];
    result.elastic_post_em_null_collisions=elastic_counts_host[9]+elastic_counts_host[10];
    result.elastic_local_deposited_energy_MeV=elastic_energy_host[0];
    result.elastic_queued_charged_energy_MeV=elastic_energy_host[1];
    result.elastic_queue_overflow_energy_MeV=elastic_energy_host[2];
    result.elastic_queue_overflow=elastic_counts_host[11];
    result.schneider_miss_log = std::move(schneider_miss_host);
    result.electron_joint_diagnostics={electron_joint_diag_host[0],electron_joint_diag_host[1],electron_joint_diag_host[2],
        static_cast<double>(electron_joint_diag_host[3])*1e-6,static_cast<double>(electron_joint_diag_host[4])*1e-6,static_cast<double>(electron_joint_diag_host[5])*1e-6,electron_joint_diag_host[6]};
    result.material_electron_photon_untracked_MeV=double(material_untracked_host[0])*1e-6;
    result.material_electron_untracked_MeV=double(material_untracked_host[0]+material_untracked_host[1])*1e-6;
    result.electron_variance_reduction_weight_gain_MeV=electron_vr_gain_host;
    free_device(material_untracked_device);
    free_device(material_failure_device);

    result.schneider_unsupported_tracks = std::move(schneider_track_host);
    result.backend = "sycl-" + resolved_device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    const auto batch_has_energy_spread = std::any_of(
        config.primary_spot_batch.begin(), config.primary_spot_batch.end(),
        [](const auto& spot) { return spot.floats[1] > 0.0F; });
    if (config.beam_energy_spread > 0.0 || batch_has_energy_spread) {
        result.backend += "+espread";
    }
    if (!config.primary_spot_batch.empty()) {
        result.backend += "+spot-batch";
    }
    if (config.enable_tps_source) {
        result.backend += "+tps-source";
    }
#if defined(CARBON_ENABLE_MINIBEAM)
    if ((!BroadBeam && config.enable_minibeam)) {
        result.backend += config.minibeam_transport_mode == "copper_em"
            ? "+minibeam-copper" : "+minibeam-absorber";
    }
#endif
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_ct_grid && enable_ct_material_mcs) {
        result.backend += "+ct-material-mcs";
    }
    if (enable_layered_phantom) {
        result.backend += "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += "+hetero-insert";
    }
    if (enable_ct_grid) result.backend += "+ct-grid-unified-em+ct-dda";
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring";
        if (!voxel_scorer_clamps_transport) {
            result.backend += "+scorer-decoupled";
        }
    }
    if (enable_let_scoring) {
        result.backend += "+letd-scoring";
    }

    if (config.enable_fragment_species_scoring && !enable_secondary_transport) {
        result.secondary_carbon_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_boron_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_beryllium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_lithium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_helium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_proton_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_other_charged_deposited_energy_MeV.assign(number_of_bins, 0.0);
    }
    result.primary_survival_counts = std::move(primary_survival_host);
    result.inelastic_reaction_counts = std::move(inelastic_reaction_host);
    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);
    result.primary_voxel_track_length_mm =
        std::move(primary_voxel_track_length_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    result.minibeam_component_voxel_deposited_energy_MeV =
        std::move(minibeam_component_voxel_dose_host);
    result.minibeam_energy_band_roi_deposited_energy_MeV =
        std::move(minibeam_energy_band_roi_dose_host);
    result.minibeam_primary_roi_values =
        std::move(minibeam_primary_roi_dose_host);
    result.minibeam_spatial_audit_counts = minibeam_spatial_audit_host;
    result.be_isotope_origin_voxel_deposited_energy_MeV =
        std::move(be_isotope_origin_voxel_dose_host);
    result.he_isotope_origin_voxel_deposited_energy_MeV =
        std::move(he_isotope_origin_voxel_dose_host);
    result.helium4_hazard_audit = he4_hazard_audit_host;
    result.in_fov_deposited_energy_MeV = std::move(in_fov_dose_host);
    if (!config.fragment_birth_spectrum_output_file.empty()) {
        constexpr std::size_t categories = light_isotope_category_count;
        constexpr std::size_t generations = birth_generation_bin_count;
        result.birth_counts_by_generation.assign(categories * generations, 0);
        result.birth_ke_sum_MeV_by_generation.assign(categories * generations, 0.0);
        result.birth_mevu_hist.assign(birth_hist_plane_size(birth_mevu_bin_count), 0);
        result.birth_depth_hist.assign(birth_hist_plane_size(number_of_bins), 0);
        result.birth_cos_hist.assign(birth_hist_plane_size(birth_cos_bin_count), 0);
        result.birth_parent_mevu_hist.assign(
            birth_hist_plane_size(birth_parent_mevu_bin_count), 0);
        result.birth_parent_z_hist.assign(
            birth_hist_plane_size(birth_parent_z_bin_count), 0);
        result.birth_parent_product_mevu_hist.assign(birth_joint_plane_size(), 0);

        const auto parent_mevu_bin = birth_parent_mevu_bin(
            config.initial_total_energy_MeV(), config.primary_mass_number);
        const auto parent_z_bin = birth_parent_z_bin(config.primary_atomic_number);
        for (std::size_t birth_index = 0;
             birth_index < birth_secondaries_host.size(); ++birth_index) {
            const auto& fragment = birth_secondaries_host[birth_index];
            if (fragment.z == 2 && (fragment.a == 3 || fragment.a == 4 || fragment.a == 6)) {
                result.helium_birth_records.push_back({
                    fragment.parent_history, static_cast<unsigned>(fragment.generation),
                    static_cast<unsigned>(fragment.a), fragment.energy_MeV,
                    fragment.pos_x_mm, fragment.pos_y_mm, fragment.pos_z_mm,
                    fragment.dir_x, fragment.dir_y, fragment.dir_z, fragment.weight});
            }
            if (fragment.z == 6 && fragment.a == 12) {
                result.c12_birth_records.push_back({
                    birth_index, fragment.parent_history, fragment.rng_stream,
                    static_cast<unsigned>(fragment.generation),
                    static_cast<unsigned>(fragment.birth_region),
                    fragment.energy_MeV, fragment.pos_x_mm, fragment.pos_y_mm,
                    fragment.pos_z_mm, fragment.dir_x, fragment.dir_y,
                    fragment.dir_z, fragment.weight});
            }
            const auto generation = birth_generation_bin(
                static_cast<std::uint8_t>(fragment.generation));
            const auto category =
                light_isotope_category(fragment.z, fragment.a);
            if (category >= categories || fragment.a <= 0) {
                continue;
            }
            const auto summary_index = category * generations + generation;
            const auto mevu_bin =
                birth_mevu_bin(fragment.energy_MeV, fragment.a);
            const auto depth_bin = std::min(
                number_of_bins - 1,
                static_cast<std::size_t>(std::max(
                    0.0F, fragment.pos_z_mm / depth_bin_width_mm)));
            const auto cos_bin = birth_cos_bin(fragment.dir_z);
            ++result.birth_counts_by_generation[summary_index];
            result.birth_ke_sum_MeV_by_generation[summary_index] +=
                fragment.energy_MeV;
            ++result.birth_mevu_hist[birth_hist_index(
                category, generation, mevu_bin, birth_mevu_bin_count)];
            ++result.birth_depth_hist[birth_hist_index(
                category, generation, depth_bin, number_of_bins)];
            ++result.birth_cos_hist[birth_hist_index(
                category, generation, cos_bin, birth_cos_bin_count)];
            ++result.birth_parent_mevu_hist[birth_hist_index(
                category, generation, parent_mevu_bin, birth_parent_mevu_bin_count)];
            ++result.birth_parent_z_hist[birth_hist_index(
                category, generation, parent_z_bin, birth_parent_z_bin_count)];

            ++result.birth_parent_product_mevu_hist[birth_joint_index(
                category, generation, parent_mevu_bin, mevu_bin)];
        }
    }
    if (enable_let_scoring) {
        const auto extract_let_moment = [&](const std::size_t moment) {
            const auto begin = let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_bins);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.primary_letd_numerator = extract_let_moment(0);
        result.primary_letd_denominator = extract_let_moment(1);
        result.all_hadron_letd_numerator = extract_let_moment(2);
        result.all_hadron_letd_denominator = extract_let_moment(3);
    }
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        const auto extract_voxel_let_moment = [&](const std::size_t moment) {
            const auto begin = voxel_let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_voxels);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_voxels));
        };
        result.primary_voxel_letd_numerator = extract_voxel_let_moment(0);
        result.primary_voxel_letd_denominator = extract_voxel_let_moment(1);
        result.all_hadron_voxel_letd_numerator = extract_voxel_let_moment(2);
        result.all_hadron_voxel_letd_denominator = extract_voxel_let_moment(3);
    }

    result.initial_energy_MeV =
        std::accumulate(sampled_incident_host.begin(), sampled_incident_host.end(), 0.0);
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);
    if (config.enable_primary_loss_query_audit && !enable_inelastic) {
        // Host-only summary of existing readback: no device, RNG or scoring
        // changes. With nuclear transport disabled this is primary energy.
        const double mean = result.escaped_energy_MeV / escaped_host.size();
        double second = 0.0, third = 0.0;
        std::size_t positive = 0;
        for (const float energy : escaped_host) {
            const double delta = static_cast<double>(energy) - mean;
            second += delta * delta;
            third += delta * delta * delta;
            positive += energy > 0.0F;
        }
        second /= escaped_host.size();
        third /= escaped_host.size();
        auto sorted = escaped_host;
        std::sort(sorted.begin(), sorted.end());
        const auto quantile = [&](double p) {
            const double index = p * (sorted.size()-1);
            const auto low = static_cast<std::size_t>(index);
            const auto high = std::min(low+1,sorted.size()-1);
            return sorted[low] + (index-low)*(sorted[high]-sorted[low]);
        };
        const auto old_precision = std::cout.precision(17);
        std::cout << "PRIMARY_ESCAPE_ENERGY," << escaped_host.size() << ','
                  << positive << ',' << mean << ',' << second << ','
                  << (second > 0 ? third/std::pow(second,1.5) : 0.0) << ','
                  << quantile(.05) << ',' << quantile(.5) << ',' << quantile(.95) << '\n';
        std::cout.precision(old_precision);
    }
    result.untracked_nuclear_energy_MeV =
        std::accumulate(untracked_host.begin(), untracked_host.end(), 0.0);
    result.primary_inelastic_terminated_count = terminal_counts_host[0];
    result.primary_escaped_ct_count = terminal_counts_host[1];
    result.primary_stopped_count = terminal_counts_host[2];
    result.primary_other_terminal_count = terminal_counts_host[3];
    result.primary_inelastic_removed_kinetic_MeV = result.untracked_nuclear_energy_MeV;
    result.primary_other_terminal_kinetic_MeV =
        std::accumulate(other_terminal_host.begin(), other_terminal_host.end(), 0.0);
    result.primary_cutoff_stopped_energy_MeV =
        std::accumulate(cutoff_stopped_host.begin(), cutoff_stopped_host.end(), 0.0);
#if defined(CARBON_ENABLE_MINIBEAM)
    if ((!BroadBeam && config.enable_minibeam)) {



        result.beamline_removed_energy_MeV = std::accumulate(
            beamline_removed_host.begin(), beamline_removed_host.end(), 0.0);
        result.beamline_removed_energy_MeV += std::accumulate(
            beamline_air_loss_host.begin(), beamline_air_loss_host.end(), 0.0);
        const auto removed = static_cast<std::uint64_t>(std::count_if(
            beamline_removed_host.begin(), beamline_removed_host.end(),
            [](const float value) { return value > 0.0F; }));
        const auto primary_survivors = static_cast<std::uint64_t>(std::count_if(
            beamline_primary_survivor_host.begin(),
            beamline_primary_survivor_host.end(),
            [](const float value) { return value > 0.0F; }));
        std::uint64_t direct_survivors = 0;
        for (std::size_t history = 0; history < number_of_histories; ++history)
            if (beamline_primary_survivor_host[history] > 0.0F &&
                beamline_removed_host[history] == 0.0F) ++direct_survivors;
        result.minibeam.enabled = true;
        result.minibeam.incident_histories = number_of_histories;
        result.minibeam.copper_touched_histories = removed;
        result.minibeam.direct_air_slit_histories = direct_survivors;
        result.minibeam.water_entrance_primary = primary_survivors;
        result.minibeam.copper_nuclear_interactions = minibeam_event_counts_host[0];
        result.minibeam.copper_generated_direct_secondaries =
            minibeam_event_counts_host[1];
        result.minibeam.copper_charged_survivors = minibeam_event_counts_host[2];
        for (std::size_t category = 0; category < 9; ++category) {
            result.minibeam.copper_charged_survivors_by_species[category] =
                minibeam_event_counts_host[3 + category];
            result.minibeam.copper_charged_survivor_energy_by_species_MeV[
                category] = static_cast<double>(
                    minibeam_event_counts_host[12 + category]) / 1000.0;
            result.minibeam.copper_fragment_absorptions_by_species[category] =
                minibeam_event_counts_host[21 + category];
            result.minibeam
                .copper_fragment_absorbed_energy_by_species_MeV[category] =
                static_cast<double>(minibeam_event_counts_host[30 + category]) /
                1000.0;
            result.minibeam
                .copper_fragment_absorption_straight_copper_path_by_species_mm[
                    category] =
                static_cast<double>(minibeam_event_counts_host[39 + category]) /
                1000.0;
        }
        result.minibeam.copper_fragment_cascade_interactions =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_interactions_slot];
        result.minibeam.copper_fragment_cascade_lookup_hits =
            minibeam_event_counts_host[minibeam_fragment_cascade_hits_slot];
        for (std::size_t generation = 0;
             generation < MinibeamDiagnostics::copper_cascade_generation_count;
             ++generation) {
            result.minibeam
                .copper_fragment_cascade_interactions_by_generation[
                    generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_interaction_slot + generation];
            result.minibeam.copper_fragment_cascade_hits_by_generation[
                generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_hit_slot + generation];
            result.minibeam.copper_fragment_cascade_misses_by_generation[
                generation] = minibeam_event_counts_host[
                minibeam_fragment_generation_miss_slot + generation];
        }
        for (std::size_t miss = 0; miss < 6; ++miss) {
            result.minibeam.copper_fragment_cascade_lookup_misses[miss] =
                minibeam_event_counts_host[
                    minibeam_fragment_cascade_miss_slot + miss];
        }
        result.minibeam.copper_fragment_cascade_generated_charged =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_charged_slot];
        result.minibeam.copper_fragment_cascade_generated_neutral =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_neutral_slot];
        result.minibeam.copper_fragment_cascade_generated_unsupported =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_unsupported_slot];
        result.minibeam.copper_fragment_cascade_queue_overflows =
            minibeam_event_counts_host[
                minibeam_fragment_cascade_overflow_slot];
        result.minibeam.copper_fragment_cascade_local_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_cascade_local_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_untracked_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_cascade_untracked_keV_slot]) / 1000.0;
        for (std::size_t category = 0; category < 9; ++category) {
            result.minibeam
                .copper_fragment_ignored_nuclear_optical_depth_by_species[
                    category] =
                static_cast<double>(minibeam_event_counts_host[
                    minibeam_fragment_ignored_tau_micro_slot + category]) /
                1.0e6;
            result.minibeam.copper_fragment_terminal_tracks_by_species[
                category] = minibeam_event_counts_host[
                    minibeam_fragment_terminal_track_slot + category];
            result.minibeam
                .copper_fragment_ignored_reaction_probability_by_species[
                    category] =
                static_cast<double>(minibeam_event_counts_host[
                    minibeam_fragment_ignored_probability_micro_slot +
                    category]) /
                1.0e6;
            result.minibeam
                .copper_fragment_cascade_lookup_misses_by_species[category] =
                minibeam_event_counts_host[
                    minibeam_fragment_miss_species_slot + category];
        }
        for (std::size_t bin = 0;
             bin < MinibeamDiagnostics::fragment_lookup_energy_bin_count;
             ++bin) {
            result.minibeam
                .copper_fragment_cascade_lookup_misses_by_energy[bin] =
                minibeam_event_counts_host[
                minibeam_fragment_miss_energy_slot + bin];
        }
        result.minibeam.copper_fragment_miss_joint_counts =
            std::move(minibeam_fragment_miss_joint_counts_host);
        result.minibeam.copper_fragment_miss_joint_input_energy_keV =
            std::move(minibeam_fragment_miss_joint_energy_host);
        result.minibeam.copper_fragment_miss_joint_collision_depth_um =
            std::move(minibeam_fragment_miss_joint_depth_host);
        result.minibeam.copper_fragment_miss_joint_remaining_copper_um =
            std::move(minibeam_fragment_miss_joint_remaining_host);
        result.minibeam.copper_fragment_cascade_actual_input_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_actual_input_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_selected_input_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_selected_input_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_replay_output_energy_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_replay_output_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_selection_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_selection_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_closure_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_closure_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_mass_energy_mismatch_MeV =
            static_cast<double>(minibeam_event_counts_host[
                minibeam_fragment_mass_energy_mismatch_keV_slot]) / 1000.0;
        result.minibeam.copper_fragment_cascade_baryon_mismatch =
            minibeam_event_counts_host[
                minibeam_fragment_baryon_mismatch_slot];
        result.minibeam.beamline_removed_energy_MeV =
            result.beamline_removed_energy_MeV;
        for (std::size_t history = 0; history < number_of_histories; ++history) {
            const auto energy = static_cast<double>(
                beamline_primary_survivor_host[history]);
            if (!(energy > 0.0)) continue;
            result.minibeam.energy_sum_MeV += energy;
            result.minibeam.energy_squared_sum_MeV2 += energy * energy;
            if (beamline_removed_host[history] == 0.0F) {
                result.minibeam.direct_air_primary_energy_MeV += energy;
            } else {
                result.minibeam.copper_touched_primary_energy_MeV += energy;
                const auto bin = std::min(
                    static_cast<std::size_t>(energy / 200.0),
                    MinibeamDiagnostics::touched_energy_bin_count - 1);
                ++result.minibeam
                      .copper_touched_primary_energy_histogram[bin];
            }
        }
        if (!minibeam_phase_space_host.empty()) {
            result.minibeam_phase_space_records =
                std::move(minibeam_phase_space_host);
            for (const auto& record : result.minibeam_phase_space_records) {
                if (!record.valid) continue;
                result.minibeam.x_sum_mm += record.x_mm;
                result.minibeam.x_squared_sum_mm2 +=
                    static_cast<double>(record.x_mm) * record.x_mm;
                result.minibeam.y_sum_mm += record.y_mm;
                result.minibeam.y_squared_sum_mm2 +=
                    static_cast<double>(record.y_mm) * record.y_mm;
                result.minibeam.direction_x_sum += record.direction_x;
                result.minibeam.direction_x_squared_sum +=
                    static_cast<double>(record.direction_x) *
                    record.direction_x;
                result.minibeam.direction_y_sum += record.direction_y;
                result.minibeam.direction_y_squared_sum +=
                    static_cast<double>(record.direction_y) *
                    record.direction_y;
            }
        }
        result.minibeam_water_primary_plane_records =
            std::move(minibeam_water_primary_plane_records_host);
        result.minibeam_fragment_phase_space_records =
            std::move(minibeam_fragment_phase_space_host);
    }
#endif
    result.neutron_birth_records = std::move(neutron_birth_host);
    result.primary_first_interactions = std::move(first_interactions_host);
    result.secondary_queue_overflow = overflow_count_host;
    result.secondary_queue_overflow_energy_MeV = static_cast<double>(overflow_energy_host);
    // Retired CINEL02 diagnostics retain their zero-initialized result fields.
    // Current shared species/grid measurements must still be copied.
    for (std::size_t i = 0; i < kCinel02SpeciesEnergySlots; ++i) {
        result.cinel02_species_transport_ledger_MeV[i] =
            static_cast<double>(cinel02_species_energy_host[i]);
    }
    result.in_grid_deposited_energy_MeV = grid_deposited_in_host;
    result.outside_grid_deposited_energy_MeV = grid_deposited_out_host;
    result.cinel02_species_terminal_reason_counts = cinel02_species_terminal_host;

    // Aggregate the flat Schneider device schema into named diagnostics.
    {
        auto& sch = result.schneider_diagnostics;
        const auto slot = [&](SchneiderDiagSlot s) -> std::uint64_t {
            return schneider_diag_host[static_cast<std::uint32_t>(s)];
        };
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        sch.primary_rate_queries = slot(SchneiderDiagSlot::PrimaryRateQueries);
        sch.primary_hazards = slot(SchneiderDiagSlot::PrimaryHazards);
        sch.primary_exact_target_hits = slot(SchneiderDiagSlot::PrimaryExactTargetHits);
        sch.primary_missing_projectile = slot(SchneiderDiagSlot::PrimaryMissingProjectile);
        sch.primary_missing_target = slot(SchneiderDiagSlot::PrimaryMissingTarget);
        sch.primary_below_domain = slot(SchneiderDiagSlot::PrimaryBelowDomain);
        sch.primary_above_domain = slot(SchneiderDiagSlot::PrimaryAboveDomain);
        sch.primary_energy_gap_misses = slot(SchneiderDiagSlot::PrimaryEnergyGapMisses);
        sch.primary_empty_nodes = slot(SchneiderDiagSlot::PrimaryEmptyNodes);
        sch.primary_events_replayed = slot(SchneiderDiagSlot::PrimaryEventsReplayed);
        sch.primary_charged_products_born = slot(SchneiderDiagSlot::PrimaryChargedBorn);
        sch.primary_charged_products_queued = slot(SchneiderDiagSlot::PrimaryChargedQueued);
        sch.primary_charged_cutoff_kills = slot(SchneiderDiagSlot::PrimaryChargedCutoffKills);
        sch.primary_be6_kills = slot(SchneiderDiagSlot::PrimaryBe6Kills);
        sch.primary_queue_overflows = slot(SchneiderDiagSlot::PrimaryQueueOverflows);
        sch.secondary_tracks_started = slot(SchneiderDiagSlot::SecondaryTracksStarted);
        sch.secondary_steps = slot(SchneiderDiagSlot::SecondarySteps);
        sch.secondary_fe_c12_steps = slot(SchneiderDiagSlot::SecondaryFeC12Steps);
        sch.secondary_fe_he4_steps = slot(SchneiderDiagSlot::SecondaryFeHe4Steps);
        sch.secondary_fe_pdt_steps = slot(SchneiderDiagSlot::SecondaryFePdtSteps);
        sch.secondary_fe_other_charged_steps =
            slot(SchneiderDiagSlot::SecondaryFeOtherChargedSteps);
        sch.secondary_highland_steps =
            slot(SchneiderDiagSlot::SecondaryHighlandSteps);
        sch.secondary_rate_queries = slot(SchneiderDiagSlot::SecondaryRateQueries);
        sch.secondary_hazards = slot(SchneiderDiagSlot::SecondaryHazards);
        sch.secondary_exact_target_hits = slot(SchneiderDiagSlot::SecondaryExactTargetHits);
        sch.secondary_missing_projectile = slot(SchneiderDiagSlot::SecondaryMissingProjectile);
        sch.secondary_missing_target = slot(SchneiderDiagSlot::SecondaryMissingTarget);
        sch.secondary_below_domain = slot(SchneiderDiagSlot::SecondaryBelowDomain);
        sch.secondary_above_domain = slot(SchneiderDiagSlot::SecondaryAboveDomain);
        sch.secondary_energy_gap_misses = slot(SchneiderDiagSlot::SecondaryEnergyGapMisses);
        sch.secondary_empty_nodes = slot(SchneiderDiagSlot::SecondaryEmptyNodes);
        sch.secondary_events_replayed = slot(SchneiderDiagSlot::SecondaryEventsReplayed);
        sch.secondary_charged_products_born = slot(SchneiderDiagSlot::SecondaryChargedBorn);
        sch.secondary_charged_products_queued = slot(SchneiderDiagSlot::SecondaryChargedQueued);
        sch.secondary_charged_cutoff_kills = slot(SchneiderDiagSlot::SecondaryChargedCutoffKills);
        sch.secondary_be6_kills = slot(SchneiderDiagSlot::SecondaryBe6Kills);
        sch.secondary_queue_overflows = slot(SchneiderDiagSlot::SecondaryQueueOverflows);
        sch.secondary_stopped_before_replay = slot(SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
        sch.secondary_post_em_null_collisions = slot(SchneiderDiagSlot::SecondaryPostEmNullCollisions);
        sch.secondary_post_em_null_energy_MeV = fslot(SchneiderFloatSlot::PostEmNullEnergy);
        sch.primary_post_em_null_collisions = slot(SchneiderDiagSlot::PrimaryPostEmNullCollisions);
        // Exact fixed-point secondary sub-ledger (micro-MeV -> MeV). These
        // fields were historically always zero (never assigned); on the
        // Schneider path they now accumulate every secondary deposit/escape
        // at the same sites as the per-history arrays.
        result.secondary_deposited_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryDepositedMicroMeV)) * 1.0e-6;
        result.secondary_escaped_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryEscapedMicroMeV)) * 1.0e-6;
        sch.be6_topas_compat_kills = slot(SchneiderDiagSlot::Be6TopasCompatKills);
        sch.unsupported_projectile_steps = slot(SchneiderDiagSlot::UnsupportedProjectileSteps);
        sch.unsupported_projectile_tracks = slot(SchneiderDiagSlot::UnsupportedProjectileTracks);
        sch.unsupported_be6_tracks = slot(SchneiderDiagSlot::UnsupportedBe6Tracks);
        sch.unsupported_projectile_birth_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV)) * 1.0e-6;
        // Dropped counts come from the log-count buffers (device-side
        // overflow tallies), not the diag slots (unused for these two).
        sch.miss_log_dropped = schneider_miss_counts_host[1];
        sch.unsupported_log_dropped = schneider_track_counts_host[1];
        sch.unsupported_targets = slot(SchneiderDiagSlot::UnsupportedTargets);
        sch.queue_overflows = slot(SchneiderDiagSlot::QueueOverflows);
        sch.primary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::PrimaryMismatchSum);
        sch.primary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::PrimaryMismatchMax);
        sch.secondary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::SecondaryMismatchSum);
        sch.secondary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::SecondaryMismatchMax);
        sch.lookup_failure_energy_MeV = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        sch.be6_kill_energy_MeV = fslot(SchneiderFloatSlot::Be6KillEnergy);
        sch.neutral_product_kinetic_MeV = fslot(SchneiderFloatSlot::NeutralProductKinetic);
        sch.reaction_q_residual_MeV = fslot(SchneiderFloatSlot::ReactionQResidual);
    }

    // 8-part energy accounting ledger
    result.energy_ledger.E_continuous_ionizing = result.total_deposited_energy_MeV;
    result.energy_ledger.E_nuclear_local = 0.0;
    result.energy_ledger.E_transported_secondaries = result.queued_secondary_energy_MeV;
    result.energy_ledger.E_escaped_charged = result.escaped_energy_MeV;
    result.energy_ledger.E_neutral = result.untransported_neutral_energy_MeV;
    result.energy_ledger.E_cutoff_kill = result.primary_cutoff_stopped_energy_MeV;
    result.energy_ledger.E_unsupported = result.untransported_unsupported_charged_energy_MeV;
    result.energy_ledger.E_queue_overflow = result.secondary_queue_overflow_energy_MeV;
    // Split Schneider nuclear-vertex ledger, accumulated on device from
    // actually executed vertices (informational itemization; the legacy
    // untracked sink behavior above is unchanged this step).
    {
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        result.energy_ledger.E_be6_kill = fslot(SchneiderFloatSlot::Be6KillEnergy);
        result.energy_ledger.E_lookup_failure = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        result.energy_ledger.E_neutral_product_kinetic =
            fslot(SchneiderFloatSlot::NeutralProductKinetic);
        result.energy_ledger.E_reaction_q_residual = fslot(SchneiderFloatSlot::ReactionQResidual);
        result.energy_ledger.E_unsupported_charged =
            fslot(SchneiderFloatSlot::UnsupportedProductEnergy);
        result.energy_ledger.E_out_of_domain = fslot(SchneiderFloatSlot::OutOfDomainEnergy);
        if (schneider_diag_device != nullptr) {
            result.energy_ledger.E_transported_secondaries =
                fslot(SchneiderFloatSlot::SecondaryTransportBirthEnergy);
        }
    }

    if (neutron_birth_count_host > neutron_birth_buffer.capacity)
        throw std::runtime_error("Neutron birth audit overflow; reduce histories or increase neutron_birth_capacity");
    if (config.quality_reject_any_queue_overflow && overflow_count_host > 0) {
        throw std::runtime_error("Secondary particle queue overflow detected: discarded " +
                                 std::to_string(overflow_count_host) + " particles (" +
                                 std::to_string(overflow_energy_host) + " MeV). Shard must be split and rerun with fewer particles.");
    }

    result.nuclear_interactions =
        schneider_inelastic_host;
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

#if defined(CARBON_ENABLE_MINIBEAM)
    if (!copper_diagnostic_path.empty())
        write_copper_history_diagnostic(copper_diagnostic_path,copper_diagnostic_host,
            beamline_primary_survivor_host,deposited_host,steps_host,copper_history_mask_host);
    if(!copper_capture_path.empty()) {
        const std::filesystem::path directory(copper_capture_path);
        std::filesystem::create_directories(directory);
        write_copper_capture_records(directory/"input.bin",copper_diagnostic_host);
        write_copper_capture_records(directory/"water_primary.bin",copper_capture_entry_host);
        write_copper_capture_records(directory/"removed.bin",beamline_removed_host);
        write_copper_capture_records(directory/"air_loss.bin",beamline_air_loss_host);
        write_copper_capture_records(directory/"water_charged.bin",copper_capture_charged_host);
        std::vector<NeutronBirthRecord> copper_neutrons;
        for(const auto& birth:result.neutron_birth_records)
            if(birth.material_section==-2)copper_neutrons.push_back(birth);
        write_copper_capture_records(directory/"copper_neutrons.bin",copper_neutrons);
        std::cout << "[copper-response-capture] source_histories=" << number_of_histories
                  << " charged_at_water=" << copper_capture_charged_host.size()
                  << " copper_neutron_births=" << copper_neutrons.size()
                  << " entry_bytes=" << sizeof(PrimaryMediumEntryState)
                  << " charged_bytes=" << sizeof(SecondaryParticle)
                  << " neutron_bytes=" << sizeof(NeutronBirthRecord) << '\n';
    }
#endif
    if((!BroadBeam && config.enable_neutron_transport)) transport_neutrons_experimental(queue,config,result,&inputs);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.electron_kernel_seconds = electron_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;
}

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
    const StoppingPowerTable& stopping_power,const CrossSectionTable& cross_section,
    const std::string& device_name,SyclTransportContext* context) {
    if(config.transport_profile=="broadbeam_fast") {
        std::cout<<"[broadbeam-electron-tracking] requested="<<config.broadbeam_electron_tracking
                 <<" effective="<<(config.broadbeam_independent_electrons?"shared_fp32":
                    config.broadbeam_electron_tracking?"empty_uniform_water_selection":"disabled")
                 <<" density_threshold="<<config.broadbeam_electron_density_threshold
                 <<" halo_mm="<<config.broadbeam_electron_halo_mm<<'\n';
        if(config.broadbeam_independent_electrons)
            return transport_sycl_impl<true,true,3>(config,stopping_power,cross_section,device_name,context);

        return config.fermi_eyges_gaussian_only
            ? transport_sycl_impl<true,true>(config,stopping_power,cross_section,device_name,context)
            : transport_sycl_impl<true,false>(config,stopping_power,cross_section,device_name,context);
    }
    return transport_sycl_impl<false,false>(config,stopping_power,cross_section,device_name,context);
}

}  // namespace carbon

#endif
