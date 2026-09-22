#include "carbon/nuclear_collision.hpp"
#include "carbon/charged_species.hpp"
#include "carbon/secondary_schedule.hpp"
#include "carbon/delta_moments_data.hpp"
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
#include <sstream>
#include "carbon/cross_section.hpp"
#include "carbon/hadronic_cache_candidate.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/electron_joint_response.hpp"
#include "carbon/water_electron_response.hpp"
#include "carbon/material_electron_response.hpp"
#include "carbon/material_electron_device.hpp"
#include "carbon/electron_packet_transport.hpp"
#include "carbon/electron_short_range.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include "carbon/inelastic.hpp"
#include "carbon/inelastic_package_v2.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/schneider_ion_stopping_table.hpp"
#include "carbon/nuclear_elastic_scattering.hpp"
#include "carbon/all_ion_elastic.hpp"
#include "carbon/detail/device_memory_tracker.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"
#include "carbon/sha256.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/schneider_delta_tail.hpp"
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
#include <tuple>
#include <type_traits>
#include <vector>

namespace carbon {
template<int EmMode, bool ProductionPath = false>
class CarbonPrimaryTransportKernel;
template<int EmMode, bool ProductionPath = false, int ExactSpecies = -1>
class CarbonSecondaryTransportKernel;

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_profile.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"
#include "detail/sycl_cinel02_device.inc"

namespace {

#if defined(CARBON_ENABLE_MINIBEAM)
struct WaterEntrySecondaryReplay {
    std::vector<SecondaryParticle> particles;
    std::vector<float> incident_energy_by_history;
};

std::vector<std::string> split_replay_csv_row(const std::string& line) {
    const auto normalized =
        !line.empty() && line.back() == '\r'
            ? line.substr(0, line.size() - 1)
            : line;
    std::vector<std::string> fields;
    std::stringstream stream(normalized);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!normalized.empty() && normalized.back() == ',') fields.emplace_back();
    return fields;
}

template <typename T>
T parse_replay_integer(const std::string& text, const char* name,
                       const std::size_t line_number) {
    std::size_t consumed = 0;
    try {
        if constexpr (std::is_signed_v<T>) {
            const auto value = std::stoll(text, &consumed);
            if (consumed != text.size() ||
                value < static_cast<long long>(std::numeric_limits<T>::min()) ||
                value > static_cast<long long>(std::numeric_limits<T>::max()))
                throw std::out_of_range("integer range");
            return static_cast<T>(value);
        } else {
            const auto value = std::stoull(text, &consumed);
            if (consumed != text.size() ||
                value > static_cast<unsigned long long>(
                            std::numeric_limits<T>::max()))
                throw std::out_of_range("integer range");
            return static_cast<T>(value);
        }
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Invalid water-entry replay " + std::string(name) + " at line " +
            std::to_string(line_number) + ": " + text);
    }
}

float parse_replay_float(const std::string& text, const char* name,
                         const std::size_t line_number) {
    std::size_t consumed = 0;
    try {
        const auto value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value))
            throw std::out_of_range("non-finite float");
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Invalid water-entry replay " + std::string(name) + " at line " +
            std::to_string(line_number) + ": " + text);
    }
}

std::uint64_t replay_identity_hash(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

WaterEntrySecondaryReplay load_water_entry_secondary_replay(
    const std::filesystem::path& path, const std::size_t number_of_histories,
    const std::size_t queue_capacity,
    const bool allow_primary_c12, const bool allow_internal_births,
    const float phantom_length_mm) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(
            "Cannot open minibeam water-entry secondary replay: " +
            path.string());

    std::string line;
    if (!std::getline(input, line))
        throw std::runtime_error("Water-entry secondary replay is empty: " +
                                 path.string());
    const std::vector<std::string> required_header{
        "origin", "run_id", "event_id", "track_id", "parent_id", "pdg",
        "atomic_number", "mass_number", "kinetic_energy_MeV", "weight",
        "x_mm", "y_mm", "z_mm", "direction_x", "direction_y",
        "direction_z"};
    auto extended_header = required_header;
    extended_header.insert(extended_header.end(),
                           {"rng_stream", "generation", "birth_region"});
    const auto replay_header = split_replay_csv_row(line);
    const bool has_transport_state = replay_header == extended_header;
    if (replay_header != required_header && !has_transport_state)
        throw std::runtime_error(
            "Water-entry secondary replay header does not match the identity "
            "contract: " + path.string());

    WaterEntrySecondaryReplay replay;
    replay.incident_energy_by_history.assign(number_of_histories, 0.0F);
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = split_replay_csv_row(line);
        const auto expected_columns = has_transport_state
            ? extended_header.size() : required_header.size();
        if (fields.size() != expected_columns)
            throw std::runtime_error(
                "Water-entry secondary replay line " +
                std::to_string(line_number) + " has " +
                std::to_string(fields.size()) + " columns; expected " +
                std::to_string(expected_columns));
        const bool primary_c12_diagnostic =
            allow_primary_c12 && fields[0] == "primary";
        if (fields[0] != "fragment" && !primary_c12_diagnostic)
            throw std::runtime_error(
                "Water-entry secondary replay rejects origin='" + fields[0] +
                "' at line " + std::to_string(line_number));

        const auto run_id = parse_replay_integer<std::int64_t>(
            fields[1], "run_id", line_number);
        const auto event_id = parse_replay_integer<std::int64_t>(
            fields[2], "event_id", line_number);
        const auto track_id = parse_replay_integer<std::int64_t>(
            fields[3], "track_id", line_number);
        const auto parent_id = parse_replay_integer<std::int64_t>(
            fields[4], "parent_id", line_number);
        const auto pdg = parse_replay_integer<std::int64_t>(
            fields[5], "pdg", line_number);
        const auto z = parse_replay_integer<std::int16_t>(
            fields[6], "atomic_number", line_number);
        const auto a = parse_replay_integer<std::int16_t>(
            fields[7], "mass_number", line_number);
        const auto energy = parse_replay_float(
            fields[8], "kinetic_energy_MeV", line_number);
        const auto weight = parse_replay_float(fields[9], "weight", line_number);
        const auto x = parse_replay_float(fields[10], "x_mm", line_number);
        const auto y = parse_replay_float(fields[11], "y_mm", line_number);
        const auto depth = parse_replay_float(fields[12], "z_mm", line_number);
        auto dx = parse_replay_float(fields[13], "direction_x", line_number);
        auto dy = parse_replay_float(fields[14], "direction_y", line_number);
        auto dz = parse_replay_float(fields[15], "direction_z", line_number);
        const auto replay_rng_stream = has_transport_state
            ? parse_replay_integer<std::uint64_t>(
                  fields[16], "rng_stream", line_number)
            : std::uint64_t{0};
        const auto replay_generation = has_transport_state
            ? parse_replay_integer<std::uint8_t>(
                  fields[17], "generation", line_number)
            : std::uint8_t{0};
        const auto replay_birth_region = has_transport_state
            ? parse_replay_integer<std::uint8_t>(
                  fields[18], "birth_region", line_number)
            : std::uint8_t{0};

        if (z <= 0 || a < z || energy <= 0.0F ||
            std::abs(weight - 1.0F) > 1.0e-6F)
            throw std::runtime_error(
                "Water-entry secondary replay requires a positive ion, positive "
                "energy, and unit weight at line " +
                std::to_string(line_number));
        if (primary_c12_diagnostic &&
            !(z == 6 && a == 12 && parent_id == 0))
            throw std::runtime_error(
                "Primary secondary-path diagnostic accepts only parent-0 C12 "
                "at line " + std::to_string(line_number));
        if (!primary_c12_diagnostic && z == 6 && a == 12 && parent_id == 0)
            throw std::runtime_error(
                "Fragment replay contains parent-0 C12 at line " +
                std::to_string(line_number));
        const auto expected_pdg =
            z == 1 && a == 1
                ? std::int64_t{2212}
                : std::int64_t{1000000000} + std::int64_t{z} * 10000 +
                      std::int64_t{a} * 10;
        if (pdg != expected_pdg)
            throw std::runtime_error(
                "Water-entry replay PDG and (Z,A) disagree at line " +
                std::to_string(line_number));
        const auto norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        const bool valid_birth_position = allow_internal_births
            ? depth >= 0.0F && depth < phantom_length_mm
            : std::abs(depth) <= 1.0e-3F;
        const bool valid_birth_direction = allow_internal_births || dz > 0.0F;
        if (!(norm > 0.999F && norm < 1.001F) ||
            !valid_birth_direction || !valid_birth_position)
            throw std::runtime_error(
                allow_internal_births
                    ? "Internal secondary replay requires a unit direction and "
                      "a birth position inside the water phantom at line " +
                          std::to_string(line_number)
                    : "Water-entry replay requires a unit forward direction on "
                      "z=0 at line " + std::to_string(line_number));
        if (has_transport_state &&
            (replay_birth_region != minibeam_birth_region_copper &&
             replay_birth_region != minibeam_birth_region_water)) {
            throw std::runtime_error(
                "Internal secondary replay birth_region must be Copper or "
                "water at line " + std::to_string(line_number));
        }
        if (allow_internal_births && has_transport_state &&
            replay_birth_region != minibeam_birth_region_water) {
            throw std::runtime_error(
                "Internal water replay rejects a non-water birth at line " +
                std::to_string(line_number));
        }
        dx /= norm;
        dy /= norm;
        dz /= norm;

        if (replay.particles.size() >= number_of_histories)
            throw std::runtime_error(
                "number_of_histories must be at least the secondary replay record "
                "count so each injected particle has an independent energy ledger");
        if (replay.particles.size() >= queue_capacity)
            throw std::runtime_error(
                "secondary_queue_capacity is smaller than the secondary replay "
                "record count");
        const auto history = static_cast<std::uint32_t>(replay.particles.size());
        std::uint64_t identity = replay_identity_hash(
            static_cast<std::uint64_t>(run_id));
        identity ^= replay_identity_hash(static_cast<std::uint64_t>(event_id) + 1U);
        identity ^= replay_identity_hash(static_cast<std::uint64_t>(track_id) + 2U);
        identity ^= replay_identity_hash(static_cast<std::uint64_t>(parent_id) + 3U);

        SecondaryParticle particle{};
        particle.z = z;
        particle.a = a;
        particle.energy_MeV = energy;
        particle.pos_x_mm = x;
        particle.pos_y_mm = y;
        particle.pos_z_mm = depth;
        particle.dir_x = dx;
        particle.dir_y = dy;
        particle.dir_z = dz;
        particle.weight = 1.0F;
        particle.parent_history = history;
        particle.rng_stream = has_transport_state ? replay_rng_stream : identity;
        particle.generation = has_transport_state ? replay_generation : 0U;
        particle.birth_region = has_transport_state
            ? replay_birth_region
            : (allow_internal_births ? minibeam_birth_region_water
                                     : minibeam_birth_region_copper);
        replay.particles.push_back(particle);
        replay.incident_energy_by_history[history] = energy;
    }
    if (replay.particles.empty())
        throw std::runtime_error(
            "Water-entry secondary replay contains no particle records: " +
            path.string());
    return replay;
}
#endif

#if defined(CARBON_ENABLE_MINIBEAM)
constexpr std::size_t minibeam_event_counter_count = 138;
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
    float kinetic_energy_MeV, int atomic_number, int mass_number) noexcept {
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
    return carbon_stopping * charge * charge / 36.0F;
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
struct alignas(16) SecondaryResumeState {
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
    UnifiedEmState unified_secondary_state{};
#if CARBON_EM_LOCAL_AUDIT
    std::array<std::uint64_t,8> unified_secondary_audit{};
#endif
    uint32_t sec_steps{};
    std::uint64_t local_sec_rate_queries{};
    std::uint64_t local_sec_steps{};
    // Sparse inelastic tallies ([3],[4] of the device audit) are written
    // directly to the global device audit at event time and are not carried
    // across continuation launches.
    std::array<double,4> he4_audit{};
    std::array<std::uint64_t,CARBON_SECONDARY_STEP_PROFILE?60:0> sec_prof{};
};
static_assert(sizeof(SecondaryResumeState) % 16 == 0);
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

inline void score_minibeam_c12_roi_device(
    DoseAtomicT* scores, const std::size_t kind,
    const float kinetic_energy_MeV, const int voxel,
    const std::size_t number_of_depth_bins, const std::size_t voxel_bins_x,
    const std::size_t voxel_bins_y, const std::uint8_t* region_by_x_bin,
    const float value) {
    if (scores == nullptr || voxel < 0 || kind >= minibeam_c12_roi_kind_count ||
        !(value > 0.0F)) return;
    const auto plane = voxel_bins_x * voxel_bins_y;
    const auto depth = static_cast<std::size_t>(voxel) / plane;
    if (depth >= number_of_depth_bins) return;
    const auto x_bin = static_cast<std::size_t>(voxel) % voxel_bins_x;
    if (region_by_x_bin == nullptr) return;
    const auto region = static_cast<std::size_t>(region_by_x_bin[x_bin]);
    if (region >= minibeam_fixed_region_count) return;
    const auto energy_per_u = kinetic_energy_MeV / 12.0F;
    const std::size_t energy_band = energy_per_u < 50.0F ? 0U :
        (energy_per_u <= 300.0F ? 1U : 2U);
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
    const float* schneider_stopping_device,
    std::uint32_t schneider_sp_sections,
    std::uint32_t schneider_sp_energies,
    float schneider_sp_e_min,
    float schneider_sp_e_max,
    float schneider_sp_inv_dE,
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
        water_material = PureWaterMaterial::from_probe(config.unified_water_material_file,
                                                       config.unified_water_material_sha256);
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
    std::filesystem::path c12_cinel_file = config.ct_schneider_c12_cinel03_file;
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
            MaterialNuclearRates::water_primary(*rate_table,water_source_materials,water_material.hydrogen_mass_fraction));
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
        if (!file_sha256_matches(config.ct_elastic_section_rate_file,
                                 config.ct_elastic_section_rate_sha256))
            throw std::runtime_error("Elastic rate data SHA256 mismatch");
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
        if (host_cache && host_cache->c12_package && host_cache->c12_device_tables) {
            c12_dev = host_cache->c12_device_tables.get();
            ++host_cache->hits;
        } else {
            local_c12_pkg = std::make_shared<InelasticPackageV3Table>(
                InelasticPackageV3Table::from_binary(c12_cinel_file));
            local_c12_dev = std::make_shared<Cinel03DeviceTables>(
                local_c12_pkg->make_device_tables());
            c12_dev = local_c12_dev.get();
            if (host_cache) {
                host_cache->c12_package = local_c12_pkg;
                host_cache->c12_device_tables = local_c12_dev;
                ++host_cache->misses;
            }
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

        ctx.c12_energy_nodes = dev_c12_nodes;
        ctx.c12_event_offsets = dev_c12_offsets;
        ctx.c12_event_indices = dev_c12_indices;
        ctx.c12_interactions = dev_c12_ints;
        ctx.c12_products = dev_c12_prods;
        ctx.c12_node_count = static_cast<std::uint32_t>(c12_dev->energy_nodes.size());
        ctx.c12_total_events = static_cast<std::uint32_t>(c12_dev->interactions.size());
        ctx.c12_total_products = static_cast<std::uint32_t>(c12_dev->products.size());
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
            local_sec_pkg = std::make_shared<InelasticPackageV3Table>(
                InelasticPackageV3Table::from_binary(sec_cinel_file));
            local_sec_dev = std::make_shared<Cinel03DeviceTables>(
                local_sec_pkg->make_device_tables());
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

    // 5. Stopping power
    ctx.stopping_power_device = schneider_stopping_device;
    ctx.sp_sections = schneider_sp_sections;
    ctx.sp_energies = schneider_sp_energies;
    ctx.sp_e_min = schneider_sp_e_min;
    ctx.sp_e_max = schneider_sp_e_max;
    ctx.sp_inv_dE = schneider_sp_inv_dE;

    queue.wait_and_throw();

    return ctx;
}

template<int EmMode>
[[gnu::noinline]] TransportResult transport_sycl_impl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    RuntimeScope runtime_transport("transport_total");
    RuntimeScope runtime_setup("transport_setup_including_upload");
    config.validate();
    validate_schneider_ct_startup(config);

    // Verify provenance and metadata early before ANY GPU queue creation or device memory allocation
    auto is_valid_64hex = [](const std::string& s) -> bool {
        if (s.size() != 64) return false;
        for (char c : s) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                return false;
            }
        }
        return true;
    };

    std::string early_verified_sha256 = "";
    if (config.is_primary_attenuation_only_mode()) {
        const auto xs_file = config.ct_schneider_cross_section_file;
        if (xs_file.empty() || !std::filesystem::exists(xs_file)) {
            throw std::runtime_error("Schneider cross section file missing in validation mode: " + xs_file.string());
        }
        const auto meta_path = xs_file.parent_path() / (xs_file.stem().string() + ".metadata.json");
        if (!std::filesystem::exists(meta_path)) {
            throw std::runtime_error("Schneider cross section metadata file missing: " + meta_path.string());
        }
        std::ifstream meta_in(meta_path);
        if (!meta_in.is_open()) {
            throw std::runtime_error("Failed to open Schneider cross section metadata file: " + meta_path.string());
        }
        std::string source_sha256 = "";
        std::string line;
        while (std::getline(meta_in, line)) {
            if (line.find("\"data_sha256\"") != std::string::npos) {
                const auto colon = line.find(':');
                const auto quote1 = line.find('"', colon);
                const auto quote2 = line.find('"', quote1 + 1);
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                    break;
                }
            }
        }
        if (source_sha256.empty() || !is_valid_64hex(source_sha256)) {
            throw std::runtime_error(
                "Schneider cross section metadata missing or malformed data_sha256 in: " + meta_path.string());
        }
        if (!file_sha256_matches(xs_file, source_sha256)) {
            throw std::runtime_error("Schneider cross section runtime SHA256 mismatch");
        }
        early_verified_sha256 = source_sha256;
    } else if (!config.ct_schneider_cross_section_file.empty() &&
               std::filesystem::exists(config.ct_schneider_cross_section_file)) {
        const auto xs_file = config.ct_schneider_cross_section_file;
        const auto meta_path = xs_file.parent_path() / (xs_file.stem().string() + ".metadata.json");
        if (std::filesystem::exists(meta_path)) {
            std::ifstream meta_in(meta_path);
            if (meta_in.is_open()) {
                std::string source_sha256 = "";
                std::string line;
                while (std::getline(meta_in, line)) {
                    if (line.find("\"data_sha256\"") != std::string::npos) {
                        const auto colon = line.find(':');
                        const auto quote1 = line.find('"', colon);
                        const auto quote2 = line.find('"', quote1 + 1);
                        if (quote1 != std::string::npos && quote2 != std::string::npos) {
                            source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                            break;
                        }
                    }
                }
                if (!source_sha256.empty() && is_valid_64hex(source_sha256)) {
                    if (!file_sha256_matches(xs_file, source_sha256)) {
                        throw std::runtime_error("Schneider cross section runtime SHA256 mismatch");
                    }
                    early_verified_sha256 = source_sha256;
                }
            }
        }
    }

    std::string early_verified_stopping_sha256 = "";
    if (config.is_primary_attenuation_only_mode()) {
        const auto sp_file = config.ct_schneider_stopping_power_file;
        if (sp_file.empty() || !std::filesystem::exists(sp_file)) {
            throw std::runtime_error("Schneider stopping power file missing in validation mode: " + sp_file.string());
        }
        const auto meta_path = sp_file.parent_path() / (sp_file.stem().string() + ".metadata.json");
        if (!std::filesystem::exists(meta_path)) {
            throw std::runtime_error("Schneider stopping power metadata file missing: " + meta_path.string());
        }
        std::ifstream meta_in(meta_path);
        if (!meta_in.is_open()) {
            throw std::runtime_error("Failed to open Schneider stopping power metadata file: " + meta_path.string());
        }
        std::string source_sha256 = "";
        std::string line;
        while (std::getline(meta_in, line)) {
            if (line.find("\"data_sha256\"") != std::string::npos) {
                const auto colon = line.find(':');
                const auto quote1 = line.find('"', colon);
                const auto quote2 = line.find('"', quote1 + 1);
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                    break;
                }
            }
        }
        if (source_sha256.empty() || !is_valid_64hex(source_sha256)) {
            throw std::runtime_error(
                "Schneider stopping power metadata missing or malformed data_sha256 in: " + meta_path.string());
        }
        if (!file_sha256_matches(sp_file, source_sha256)) {
            throw std::runtime_error("Schneider stopping power runtime SHA256 mismatch");
        }
        early_verified_stopping_sha256 = source_sha256;
    } else if (!config.ct_schneider_stopping_power_file.empty() &&
               std::filesystem::exists(config.ct_schneider_stopping_power_file)) {
        const auto sp_file = config.ct_schneider_stopping_power_file;
        const auto meta_path = sp_file.parent_path() / (sp_file.stem().string() + ".metadata.json");
        if (std::filesystem::exists(meta_path)) {
            std::ifstream meta_in(meta_path);
            if (meta_in.is_open()) {
                std::string source_sha256 = "";
                std::string line;
                while (std::getline(meta_in, line)) {
                    if (line.find("\"data_sha256\"") != std::string::npos) {
                        const auto colon = line.find(':');
                        const auto quote1 = line.find('"', colon);
                        const auto quote2 = line.find('"', quote1 + 1);
                        if (quote1 != std::string::npos && quote2 != std::string::npos) {
                            source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                            break;
                        }
                    }
                }
                if (!source_sha256.empty() && is_valid_64hex(source_sha256)) {
                    if (!file_sha256_matches(sp_file, source_sha256)) {
                        throw std::runtime_error("Schneider stopping power runtime SHA256 mismatch");
                    }
                    early_verified_stopping_sha256 = source_sha256;
                }
            }
        }
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
    // Device-side inelastic tables share the stopping-power transport grid.
    const auto cross_section_table_size = table_size;

    const bool reuse_immutable_buffers = context != nullptr;
    if (context != nullptr) {
        // Identity is the content actually uploaded (stopping values, energy
        // grid, cumulative range and the cross-section resampled from these
        // arrays), not the file path. A zero cross-section hashes differently
        // from a real one.
        std::uint64_t physics_hash = 1469598103934665603ULL;
        physics_hash = fnv1a64_vec(stopping_power.values(), physics_hash);
        physics_hash = fnv1a64_vec(stopping_power.energies(), physics_hash);
        physics_hash = fnv1a64_vec(stopping_power.cumulative_ranges_mm(), physics_hash);
        physics_hash = fnv1a64_vec(cross_section.energies(), physics_hash);
        physics_hash = fnv1a64_vec(cross_section.values(), physics_hash);
        const std::string physics_signature = std::to_string(physics_hash);
        context->impl_->ensure_initialized(stopping_power, cross_section,
                                           physics_signature);
    }

    float* table_device = context != nullptr ? context->impl_->table_device
                                            : mem_tracker.allocate<float>(table_size);
    float* energy_grid_device =
        context != nullptr ? context->impl_->energy_grid_device
                           : (config.enable_csda_range_energy_loss
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr);
    float* cumulative_range_device =
        context != nullptr ? context->impl_->cumulative_range_device
                           : (config.enable_csda_range_energy_loss
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr);
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
    float* schneider_stopping_device = nullptr;
    std::uint32_t schneider_sp_sections = 0;
    std::uint32_t schneider_sp_energies = 0;
    float schneider_sp_e_min = 0.0F;
    float schneider_sp_e_max = 0.0F;
    float schneider_sp_inv_dE = 0.0F;
    bool use_schneider_stopping = false;
    // Scheme-2 secondary section stopping (SCHNIOSP v1). Null = current
    // legacy water path when the material bank is not explicitly configured.
    float* schneider_ion_sp_device = nullptr;
    bool use_schneider_ion_sp = false;
    std::uint32_t* ion_stopping_failures = nullptr;

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
    const auto use_material_tables =
        enable_layered_phantom && !config.slab_stopping_power_files.empty();
    const auto material_table_count = use_material_tables ? slab_layer_count : std::size_t{0};

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

    std::vector<float> material_sp_host(material_table_count * table_size);
    std::vector<float> material_xs_host(material_table_count * cross_section_table_size);
    if (use_material_tables) {
        for (std::size_t layer = 0; layer < slab_layer_count; ++layer) {
            const auto sp = StoppingPowerTable::from_csv(config.slab_stopping_power_files[layer]);
            const auto xs = CrossSectionTable::from_csv(config.slab_cross_section_files[layer]);
            for (std::size_t i = 0; i < table_size; ++i) {
                material_sp_host[layer * table_size + i] = static_cast<float>(sp.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                material_xs_host[layer * cross_section_table_size + i] =
                    static_cast<float>(xs.interpolate(table_energies[i]));
            }
        }
    }
    float* material_sp_device = material_table_count > 0
                                   ? mem_tracker.allocate<float>(material_sp_host.size())
                                   : nullptr;
    float* material_xs_device = material_table_count > 0
                                   ? mem_tracker.allocate<float>(material_xs_host.size())
                                   : nullptr;
    if (material_table_count > 0) {
        queue.copy(material_sp_host.data(), material_sp_device, material_sp_host.size());
        queue.copy(material_xs_host.data(), material_xs_device, material_xs_host.size())
            .wait_and_throw();
    }

    // Hetero insert
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
    const auto use_insert_material_tables =
        enable_hetero_insert && !config.insert_stopping_power_file.empty();

    std::vector<float> insert_sp_host(use_insert_material_tables ? table_size : 0);
    std::vector<float> insert_xs_host(use_insert_material_tables ? cross_section_table_size : 0);
    if (use_insert_material_tables) {
        const auto sp = StoppingPowerTable::from_csv(config.insert_stopping_power_file);
        const auto xs = CrossSectionTable::from_csv(config.insert_cross_section_file);
        for (std::size_t i = 0; i < table_size; ++i) {
            insert_sp_host[i] = static_cast<float>(sp.values()[i]);
        }
        for (std::size_t i = 0; i < cross_section_table_size; ++i) {
            insert_xs_host[i] = static_cast<float>(xs.interpolate(table_energies[i]));
        }
    }
    float* insert_sp_device = use_insert_material_tables
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr;
    float* insert_xs_device = use_insert_material_tables
                                  ? mem_tracker.allocate<float>(cross_section_table_size)
                                  : nullptr;
    if (use_insert_material_tables) {
        queue.copy(insert_sp_host.data(), insert_sp_device, table_size);
        queue.copy(insert_xs_host.data(), insert_xs_device, cross_section_table_size)
            .wait_and_throw();
    }

    // CT grid
    const auto enable_ct_grid = config.enable_ct_grid;
    float ct_origin_x = 0.0F, ct_origin_y = 0.0F, ct_origin_z = 0.0F;
    float ct_spacing_x = 0.0F, ct_spacing_y = 0.0F, ct_spacing_z = 0.0F;
    std::uint32_t ct_nx = 0, ct_ny = 0, ct_nz = 0;
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    float* ct_mass_sp_factor_lut_device = nullptr;
    float* ct_mass_sp_za_rel_device = nullptr;
    float* ct_sp_device = nullptr;
    float* ct_xs_device = nullptr;
    float* ct_ref_density_device = nullptr;
    std::uint32_t ct_n_mass_factors = 0;
    std::uint32_t ct_density_spr_n_rho = 0;
    float ct_mass_spr_log_rho_min = 0.0F;
    float ct_mass_spr_inv_dlog = 0.0F;
    bool use_ct_mass_sp = false;
    bool use_ct_density_mass_spr = false;
    bool use_ct_material_sp = false;
    bool use_ct_material_xs = false;
    bool ct_material_ids_are_schneider_sections = false;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;
    const auto ct_primary_midpoint_stopping = config.ct_primary_midpoint_stopping;

    if (std::getenv("CARBON_JOINT_EM_DATA") || std::getenv("CARBON_DIAGNOSTIC_PRIMARY_START_DEDX"))
        throw std::invalid_argument("Obsolete isolated EM environment switch; use explicit primary_em_model configuration");
    // Explicit research model; legacy transport remains the default.
    const bool unified_em=EmMode<0?config.em_model=="g4_material_joint_v1":EmMode==1;
    const float em_primary_step_scale=static_cast<float>(config.em_primary_step_scale);
    const float em_secondary_step_scale=static_cast<float>(config.em_secondary_step_scale);
    if(unified_em)std::cout<<"[em-performance] material_cache="<<CARBON_EM_MATERIAL_CACHE
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
    if(unified_em){
        const std::string unified_em_signature =
            config.em_package_file.string()+"|"+config.em_package_sha256+"|"+
            (config.em_delta_moments_file.empty()?std::string():config.em_delta_moments_file.string());
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
            unified_package=std::make_shared<UnifiedEmPackage>(
                runtime_call("unified_em_package_load_verify",[&]{return UnifiedEmPackage::load(config.em_package_file,config.em_package_sha256);}));
            const auto delta_path=config.em_delta_moments_file.empty()
                ? config.em_package_file.parent_path()/delta_moments_filename
                : config.em_delta_moments_file;
            unified_delta_means=std::make_shared<std::vector<float>>(
                runtime_call("delta_moments_load_verify",[&]{
                    return load_delta_moments(delta_path,config.em_package_sha256,unified_package->nodes.size());
                }));
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
        auto upload=[&]<class T>(const std::vector<T>& values){
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        std::cout<<"[physics-cache] unified_em="<<(unified_em_cache_hit?"hit":"miss")
                 <<" nodes="<<package.nodes.size()<<"\n";
        RuntimeScope t_em_device("setup_em_device_upload");
        delta_mean_device=upload(delta_means);
        std::cout<<"[delta-moments] condensed_partition_v1; aggregate Gamma; no delta clock; nodes="
                 <<package.nodes.size()<<" SHA256="<<delta_moments_sha256<<"\n";
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
        unified_sections=upload(sections);
        if constexpr(CARBON_EM_EXACT_INDEX) {
            const auto index=build_unified_em_index(package);
            unified_energy_index=upload(index);
        }
        unified_materials=upload(package.materials);unified_species=upload(package.species);
        unified_records=upload(package.records);unified_nodes=upload(package.nodes);unified_segments=upload(package.segments);
        if constexpr(CARBON_EM_SPLIT_SEARCH_KEYS) {
            std::vector<float> node_keys(package.nodes.size());
            for(std::size_t i=0;i<package.nodes.size();++i)node_keys[i]=package.nodes[i].energy;
            std::vector<float> segment_keys(package.segments.size());
            for(std::size_t i=0;i<package.segments.size();++i)segment_keys[i]=package.segments[i].lower;
            unified_node_energy_keys=upload(node_keys);
            unified_segment_lower_keys=upload(segment_keys);
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
        t_em_device.finish();
        std::cout<<"[unified-em] all 18 charged ions; water + Schneider density nodes; native particle step parameters plus 1% combined mean-loss guard; local aggregate delta deposition; density cut-onset/patient accuracy validation pending\n";
    }
    // One host-table cache per process, shared by the Schneider stopping/ion
    // stops below and by upload_schneider_ct_device_context(). Reset only when
    // the validated inputs or device change.
    SchneiderHostCache* schneider_host_cache = nullptr;
    if (context != nullptr) {
        auto& cache = context->impl_->schneider_host_cache;
        const std::string schneider_signature =
            config.ct_schneider_stopping_power_file.string() + "|" +
            config.ct_schneider_primary_rate_file.string() + "|" +
            config.ct_schneider_c12_cinel03_file.string() + "|" +
            config.ct_schneider_secondary_rate_file.string() + "|" +
            config.ct_schneider_secondary_cinel03_file.string() + "|" +
            config.ct_secondary_ion_section_stopping_file.string() + "|" +
            config.ct_secondary_ion_section_stopping_sha256 + "|" +
            config.ct_secondary_ion_section_stopping_metadata_sha256 + "|" +
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
    std::cout << "[stopping-config] primary_midpoint=" << ct_primary_midpoint_stopping
              << " secondary_exact_faces=" << ct_secondary_exact_faces
              << " secondary_ion_section_file=" << config.ct_secondary_ion_section_stopping_file
              << " sha256=" << config.ct_secondary_ion_section_stopping_sha256 << '\n';
    const auto ct_secondary_mcs_off = config.ct_secondary_mcs_off_diagnostic;

    const bool use_schneider_delta_tail =
        EmMode!=1 && !config.ct_schneider_delta_tail_file.empty();
    std::optional<SchneiderDeltaTailTable> schneider_delta_tail;
    float* schneider_delta_energies_device = nullptr;
    float* schneider_delta_fractions_device = nullptr;
    float* schneider_delta_radii_device = nullptr;
    std::uint8_t* schneider_delta_source_eligible_device = nullptr;
    std::uint64_t* schneider_delta_energy_device = nullptr;
    std::size_t schneider_delta_energy_count = 0;
    std::size_t schneider_delta_quantile_count = 0;
    // Optional longitudinal (forward) supplement. Shares the transverse
    // eligibility mask and section-0 scope; empty file disables it exactly.
    const bool use_schneider_delta_longitudinal =
        use_schneider_delta_tail &&
        !config.ct_schneider_delta_longitudinal_file.empty();
    const bool use_longitudinal_interface_mass = config.ct_longitudinal_interface_mass_diagnostic;
    const bool use_electron_joint = EmMode!=1 && !config.ct_electron_joint_response_diagnostic_file.empty();
    // Legacy C12-only diagnostic is separate from the all-projectile bank.
    const bool use_ct_elastic = config.ct_elastic_diagnostic;
    const bool ct_elastic_all_targets = config.ct_elastic_all_targets;
    // Production speed switch: skip reporting-only joint diagnostic atomics.
    // Slots 1 (energy domain) and 2 (miss/blocked) stay always-on: they feed
    // fail-closed quality gates. Skipped counters never feed transport.
    const bool enable_electron_joint_diagnostics = config.electron_joint_diagnostics;
    const bool minibeam_water_delta_v1 =
        config.minibeam_water_delta_response_model == "water_response_v1";
    const bool use_material_electron=EmMode!=1 && !config.material_electron_response_index_file.empty();
    const bool use_material_ct=use_material_electron && config.enable_ct_grid;
    const bool use_water_electron=EmMode!=1 && ((use_material_electron && !use_material_ct) || !config.water_electron_response_diagnostic_file.empty());
    std::optional<MaterialElectronDeviceBank> material_electron_bank;
    const MaterialElectronResponseView* material_electron_views=nullptr;
    std::size_t material_electron_view_count=0;
    std::uint64_t* material_untracked_device=nullptr; // photon / other, micro-MeV
    struct MaterialPacketFailure {
        std::uint64_t history{},step{},rng{};
        ElectronEnergyPacket before{},after{};
        ElectronContinuationAdvance advance{};
    };
    MaterialPacketFailure* material_failure_device=nullptr;
    std::uint64_t* short_range_hits_device=nullptr;
    const double material_response_density=config.water_density_g_per_cm3;
    WaterElectronChannel* water_electron_channels_device=nullptr;
    WaterElectronSample* water_electron_samples_device=nullptr;
    WaterElectronPathNode* water_electron_nodes_device=nullptr;
    std::uint32_t* water_electron_heads_device=nullptr;
    double* water_electron_radius_device=nullptr;
    std::size_t water_electron_node_count=0;
    std::size_t water_electron_channel_count=0;
    ElectronJointChannel* electron_joint_channels_device=nullptr;
    ElectronJointSample* electron_joint_samples_device=nullptr;
    ElectronPathRange* electron_path_ranges_device=nullptr;
    std::array<double,3>* electron_path_vectors_device=nullptr;
    MassPathBounds* electron_path_bounds_device=nullptr;
    std::string electron_path_sha256;
    std::uint64_t* electron_joint_diag_device=nullptr;
    std::array<double,kElectronJointSections> electron_joint_minimum{},electron_joint_maximum{};
    double electron_joint_ceiling=185;
    if(use_electron_joint) {
        const auto table=ElectronJointResponseTable::from_csv(config.ct_electron_joint_response_diagnostic_file,
            config.ct_electron_joint_response_sha256,config.ct_electron_joint_response_metadata_sha256);
        if(config.ct_electron_joint_patient_experiment && !table.full_schneider_scope)
            throw std::invalid_argument("Patient experiment requires full 25-section multi-energy schema 3, not a two-material pilot");
        // Explicitly authorized ISOLATED smoke diagnostic only. Config still
        // rejects research/production, spots, other energies and nuclear-on;
        // quality still always reports unvalidated_electron_joint_response.
        electron_joint_channels_device=mem_tracker.allocate<ElectronJointChannel>(kElectronJointChannels);
        electron_joint_samples_device=mem_tracker.allocate<ElectronJointSample>(table.samples.size());
        electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
        if(!electron_joint_channels_device || !electron_joint_samples_device || !electron_joint_diag_device)throw std::bad_alloc();
        queue.copy(table.channels.data(),electron_joint_channels_device,kElectronJointChannels).wait_and_throw();
        queue.copy(table.samples.data(),electron_joint_samples_device,table.samples.size()).wait_and_throw();
        if(!table.path_ranges.empty()) {
            electron_path_sha256=table.path_sha256;
            electron_path_ranges_device=mem_tracker.allocate<ElectronPathRange>(table.path_ranges.size());
            electron_path_vectors_device=mem_tracker.allocate<std::array<double,3>>(table.path_vectors.size());
            electron_path_bounds_device=mem_tracker.allocate<MassPathBounds>(table.path_ranges.size());
            if(!electron_path_ranges_device || !electron_path_vectors_device || !electron_path_bounds_device)throw std::bad_alloc();
            std::vector<MassPathBounds> bounds(table.path_ranges.size());
            for(std::size_t i=0;i<bounds.size();++i) {
                const auto r=table.path_ranges[i];
                bounds[i]=mass_path_bounds(r.count,[&](std::size_t j){return table.path_vectors[r.offset+j];});
            }
            queue.copy(bounds.data(),electron_path_bounds_device,bounds.size()).wait_and_throw();
            queue.copy(table.path_ranges.data(),electron_path_ranges_device,table.path_ranges.size()).wait_and_throw();
            queue.copy(table.path_vectors.data(),electron_path_vectors_device,table.path_vectors.size()).wait_and_throw();
        }
        queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
        electron_joint_minimum=table.minimum;electron_joint_maximum=table.maximum;
        electron_joint_ceiling=table.energy_ceiling_MeVu;
    }
    if(use_material_electron && !use_material_ct) {
            const auto index=MaterialElectronResponseIndex::load(config.material_electron_response_index_file,
                config.material_electron_response_index_sha256);
            const bool mapped=config.material_electron_response_memory_mode=="host_mapped";
            const auto budget=static_cast<std::size_t>(mapped?config.material_electron_response_host_budget_MiB:
                config.material_electron_response_device_budget_MiB)*1024*1024;
            material_electron_bank.emplace(queue,mapped?MaterialElectronMemory::host_mapped:MaterialElectronMemory::device);
            material_electron_bank->load(index,{{-1,material_response_density}},budget-7*sizeof(std::uint64_t));
            material_electron_views=material_electron_bank->views();
            material_electron_view_count=material_electron_bank->size();
            electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
            if(!electron_joint_diag_device)throw std::bad_alloc();
            queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
    } else if(use_water_electron || minibeam_water_delta_v1) {
        // water_response_v1 reuses the diagnostic table file keys as pins
        // for its own table path; the joint-EM forbiddance on the diagnostic
        // key itself is untouched.
        const auto response_table_path =
            minibeam_water_delta_v1 &&
                    !config.minibeam_water_delta_response_table_file.empty()
                ? config.minibeam_water_delta_response_table_file
                : config.water_electron_response_diagnostic_file;
        const auto table=WaterElectronResponseTable::load(response_table_path,
            config.water_electron_response_sha256,config.water_electron_response_metadata_sha256);
        water_electron_channel_count=table.channels.size();
        const double required_ceiling=(use_material_electron || config.water_electron_high_energy_diagnostic) ? 450.0 : 300.0;
        if(table.channels.empty() || table.channels.back().high<required_ceiling)
            throw std::invalid_argument("Water electron loaded table does not cover declared source domain");
        water_electron_channels_device=mem_tracker.allocate<WaterElectronChannel>(water_electron_channel_count);
        water_electron_samples_device=mem_tracker.allocate<WaterElectronSample>(table.samples.size());
        water_electron_nodes_device=mem_tracker.allocate<WaterElectronPathNode>(table.nodes.size());
        water_electron_heads_device=mem_tracker.allocate<std::uint32_t>(table.heads.size());
        water_electron_radius_device=mem_tracker.allocate<double>(table.prefix_radius.size());
        electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
        if(!water_electron_channels_device || !water_electron_samples_device || !water_electron_nodes_device ||
           !water_electron_heads_device || !water_electron_radius_device || !electron_joint_diag_device)throw std::bad_alloc();
        queue.copy(table.channels.data(),water_electron_channels_device,water_electron_channel_count).wait_and_throw();
        queue.copy(table.samples.data(),water_electron_samples_device,table.samples.size()).wait_and_throw();
        queue.copy(table.nodes.data(),water_electron_nodes_device,table.nodes.size()).wait_and_throw();
        queue.copy(table.heads.data(),water_electron_heads_device,table.heads.size()).wait_and_throw();
        queue.copy(table.prefix_radius.data(),water_electron_radius_device,table.prefix_radius.size()).wait_and_throw();
        queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
        water_electron_node_count=table.nodes.size();
    }
    if (use_schneider_delta_longitudinal && k_dose_atomic_fp32) {
        throw std::runtime_error(
            "ct_schneider_delta_longitudinal_file requires an FP64 dose build "
            "(CARBON_DOSE_FP32=OFF): the distributed forward shares are far below "
            "FP32 atomic granularity at clinical per-bin totals and would be "
            "silently dropped, failing energy closure");
    }
    const auto schneider_long_fraction_scale = static_cast<float>(
        config.ct_schneider_delta_longitudinal_scale);
    std::optional<SchneiderLongitudinalTable> schneider_longitudinal;
    double schneider_long_diagnostic_density = kLongitudinalReferenceDensityGPerCm3;
    LongitudinalDomainRecord* longitudinal_domain_device = nullptr;
    if (config.ct_longitudinal_homogeneous_density_diagnostic || use_longitudinal_interface_mass) {
        longitudinal_domain_device = mem_tracker.allocate<LongitudinalDomainRecord>(
            kLongitudinalDomainLogCap);
        if (!longitudinal_domain_device) throw std::bad_alloc();
    }
    float* schneider_long_energies_device = nullptr;
    float* schneider_long_fractions_device = nullptr;
    float* schneider_long_lambdas_device = nullptr;
    std::size_t schneider_long_energy_count = 0;
    if (use_schneider_delta_tail) {
        schneider_delta_tail = SchneiderDeltaTailTable::from_csv(
            config.ct_schneider_delta_tail_file);
        schneider_delta_energy_count = schneider_delta_tail->energy_count();
        schneider_delta_quantile_count = schneider_delta_tail->quantile_count();
        schneider_delta_energies_device =
            mem_tracker.allocate<float>(schneider_delta_energy_count);
        schneider_delta_fractions_device =
            mem_tracker.allocate<float>(schneider_delta_energy_count);
        schneider_delta_radii_device = mem_tracker.allocate<float>(
            schneider_delta_energy_count * schneider_delta_quantile_count);
        schneider_delta_energy_device = mem_tracker.allocate<std::uint64_t>(10);
        if (schneider_delta_energies_device == nullptr ||
            schneider_delta_fractions_device == nullptr ||
            schneider_delta_radii_device == nullptr ||
            schneider_delta_energy_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(schneider_delta_tail->energies_MeV_per_u().data(),
                   schneider_delta_energies_device, schneider_delta_energy_count);
        queue.copy(schneider_delta_tail->moved_fractions().data(),
                   schneider_delta_fractions_device, schneider_delta_energy_count);
        queue.copy(schneider_delta_tail->radii_mm().data(),
                   schneider_delta_radii_device,
                   schneider_delta_energy_count * schneider_delta_quantile_count);
        queue.fill(schneider_delta_energy_device, std::uint64_t{0}, 10)
            .wait_and_throw();
        if (use_schneider_delta_longitudinal) {
            schneider_longitudinal = SchneiderLongitudinalTable::from_csv(
                config.ct_schneider_delta_longitudinal_file);
            schneider_long_energy_count = schneider_longitudinal->energy_count();
            schneider_long_energies_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            schneider_long_fractions_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            schneider_long_lambdas_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            if (schneider_long_energies_device == nullptr ||
                schneider_long_fractions_device == nullptr ||
                schneider_long_lambdas_device == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(schneider_longitudinal->energies_MeV_per_u().data(),
                       schneider_long_energies_device, schneider_long_energy_count);
            queue.copy(schneider_longitudinal->forward_fractions().data(),
                       schneider_long_fractions_device, schneider_long_energy_count);
            queue.copy(schneider_longitudinal->lambdas_mm().data(),
                       schneider_long_lambdas_device, schneider_long_energy_count)
                .wait_and_throw();
        }
    }

    if (enable_ct_grid) {
        const auto grid = CtGrid::load(
            config.ct_grid_file, config.ct_schneider_file, config.ct_dicom_origin_mode);
        if(use_material_ct) {
            if(grid.file_version<CtGrid::version_v2)
                throw std::invalid_argument("Material electron transport requires Schneider section IDs");
            if(grid.nx!=config.voxel_bins_x || grid.ny!=config.voxel_bins_y || grid.nz!=config.voxel_bins_z ||
               std::abs(grid.spacing_x_mm-config.voxel_size_x_mm)>1e-6 ||
               std::abs(grid.spacing_y_mm-config.voxel_size_y_mm)>1e-6 ||
               std::abs(grid.spacing_z_mm-config.voxel_size_z_mm)>1e-6 || std::abs(grid.origin_z_mm)>1e-6)
                throw std::invalid_argument("Material CT packet scorer must align with CT grid, z origin zero");
            const auto response_index=MaterialElectronResponseIndex::load(config.material_electron_response_index_file,
                config.material_electron_response_index_sha256);
            std::vector<std::pair<int,double>> demand;
            for(std::size_t v=0;v<grid.material_id.size();++v)
                demand.emplace_back(grid.material_id[v],grid.density_g_per_cm3[v]);
            std::sort(demand.begin(),demand.end());demand.erase(std::unique(demand.begin(),demand.end()),demand.end());
            // Load both measured brackets in the same Schneider section.
            // required_tables rejects uncovered densities; no material fallback.
            response_index.required_tables(demand);
            const bool mapped=config.material_electron_response_memory_mode=="host_mapped";
            const auto budget=static_cast<std::size_t>(mapped?config.material_electron_response_host_budget_MiB:
                config.material_electron_response_device_budget_MiB)*1024*1024;
            material_electron_bank.emplace(queue,mapped?MaterialElectronMemory::host_mapped:MaterialElectronMemory::device);
            material_electron_bank->load(response_index,demand,
                budget-10*sizeof(std::uint64_t)-sizeof(MaterialPacketFailure),true,
                config.material_electron_short_range_mm>0);
            material_electron_views=material_electron_bank->views();material_electron_view_count=material_electron_bank->size();
            electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
            material_untracked_device=mem_tracker.allocate<std::uint64_t>(3);
            material_failure_device=mem_tracker.allocate<MaterialPacketFailure>(1);
            if(config.material_electron_short_range_mm>0) {
                short_range_hits_device=mem_tracker.allocate<std::uint64_t>(1);
                if(!short_range_hits_device)throw std::bad_alloc();
                queue.fill(short_range_hits_device,std::uint64_t{0},1).wait_and_throw();
            }
            if(!electron_joint_diag_device || !material_untracked_device)throw std::bad_alloc();
            queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
            if(!material_failure_device)throw std::bad_alloc();
            queue.fill(material_untracked_device,std::uint64_t{0},3).wait_and_throw();
        }
        if (use_longitudinal_interface_mass || (use_electron_joint && !config.ct_electron_joint_patient_experiment))
            validate_longitudinal_interface_grid(grid.density_g_per_cm3, grid.material_id);
        else if(use_schneider_delta_longitudinal)
            reject_unvalidated_longitudinal_heterogeneity(grid.density_g_per_cm3, grid.material_id);
        if (config.ct_longitudinal_homogeneous_density_diagnostic) {
            schneider_long_diagnostic_density = longitudinal_probe_density(
                grid.density_g_per_cm3, grid.material_id);
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
            grid.file_version >= CtGrid::version_v2;

        const auto voxel_count = static_cast<std::size_t>(ct_nx) * ct_ny * ct_nz;
        ct_density_device = mem_tracker.allocate<float>(voxel_count);
        ct_material_device = mem_tracker.allocate<std::uint8_t>(voxel_count);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();

        if (use_schneider_delta_tail) {
            if (grid.file_version < CtGrid::version_v2 || grid.material_id.size() != voxel_count) {
                throw std::runtime_error(
                    "Schneider delta-tail requires exact Schneider section IDs");
            }
            const auto aligned =
                config.voxel_bins_x == grid.nx && config.voxel_bins_y == grid.ny &&
                config.voxel_bins_z == grid.nz &&
                std::abs(config.voxel_size_x_mm - grid.spacing_x_mm) < 1.0e-6 &&
                std::abs(config.voxel_size_y_mm - grid.spacing_y_mm) < 1.0e-6 &&
                std::abs(config.voxel_size_z_mm - grid.spacing_z_mm) < 1.0e-6 &&
                std::abs(grid.origin_z_mm) < 1.0e-6;
            if (!aligned) {
                throw std::runtime_error(
                    "Schneider delta-tail requires a scorer exactly aligned to the CCTG grid");
            }
            // Grid edges are not material interfaces: inspect existing neighbours only.
            // Sampled endpoints still use the scorer-escape and material checks below.
            std::vector<std::uint8_t> eligible(voxel_count, 0U);
            const auto min_spacing = std::min({grid.spacing_x_mm, grid.spacing_y_mm,
                                               grid.spacing_z_mm});
            for (std::uint32_t iz = 0; iz < grid.nz; ++iz) {
                for (std::uint32_t iy = 0; iy < grid.ny; ++iy) {
                    for (std::uint32_t ix = 0; ix < grid.nx; ++ix) {
                        const auto index = ct_linear_index(ix, iy, iz, grid.nx, grid.ny);
                        if (grid.material_id[index] != 0U) continue;
                        bool clear = true;
                        if (grid.spacing_x_mm <= min_spacing * 1.001F) {
                            clear = clear && (ix == 0 || grid.material_id[index - 1] == 0U) &&
                                    (ix + 1 == grid.nx || grid.material_id[index + 1] == 0U);
                        }
                        if (grid.spacing_y_mm <= min_spacing * 1.001F) {
                            clear = clear &&
                                (iy == 0 || grid.material_id[index - grid.nx] == 0U) &&
                                (iy + 1 == grid.ny || grid.material_id[index + grid.nx] == 0U);
                        }
                        if (grid.spacing_z_mm <= min_spacing * 1.001F) {
                            const auto plane = static_cast<std::size_t>(grid.nx) * grid.ny;
                            clear = clear && (iz == 0 || grid.material_id[index - plane] == 0U) &&
                                    (iz + 1 == grid.nz || grid.material_id[index + plane] == 0U);
                        }
                        eligible[index] = clear ? 1U : 0U;
                    }
                }
            }
            schneider_delta_source_eligible_device =
                mem_tracker.allocate<std::uint8_t>(voxel_count);
            if (schneider_delta_source_eligible_device == nullptr) throw std::bad_alloc();
            queue.copy(eligible.data(), schneider_delta_source_eligible_device,
                       voxel_count).wait_and_throw();
        }

        use_ct_mass_sp = ct_material_ids_are_schneider_sections;
        use_ct_material_sp = !config.ct_water_stopping_power_file.empty() ||
                             !config.ct_bone_stopping_power_file.empty();
        use_ct_material_xs = !config.ct_bone_cross_section_file.empty() ||
                             !config.ct_schneider_cross_section_file.empty();

        use_schneider_primary_xs =
            ((ct_material_ids_are_schneider_sections && grid.mass_sp_za_rel.size() == 25) ||
             !config.ct_schneider_file.empty() ||
             !config.ct_schneider_cross_section_file.empty() ||
             config.is_schneider_ct_mode()) &&
            config.nuclear_model != "none";

        if (use_schneider_primary_xs) {
            if (config.primary_atomic_number != 6 || config.primary_mass_number != 12) {
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

                // Log table dimensions, byte count, source file, mode, and source SHA256 once
                std::cout << "[schneider-primary-xs] mode=primary-c12-section-resolved\n"
                          << "  sections=" << schneider_xs_sections << "\n"
                          << "  energies=" << schneider_xs_energies << "\n"
                          << "  bytes=" << total_elements * sizeof(float) << "\n"
                          << "  E_min=" << schneider_xs_e_min << " MeV/u\n"
                          << "  inv_dE=" << schneider_xs_inv_dE << "\n"
                          << "  source=" << config.ct_schneider_cross_section_file << "\n"
                          << "  source_sha256=" << (early_verified_sha256.empty() ? "unknown" : early_verified_sha256) << "\n";
            }
            if (config.is_primary_attenuation_only_mode()) {
                std::cout << "[ct-validation-mode] primary-attenuation-only\n"
                          << "  verified_source_sha256=" << early_verified_sha256 << "\n";
            }
        }

        if (ct_material_ids_are_schneider_sections) {
            // 1. Fail-closed Projectile Guard: Table v1 is validated exclusively for C12 (Z=6, A=12)
            if (config.primary_atomic_number != 6 || config.primary_mass_number != 12) {
                throw std::runtime_error(
                    "Schneider stopping power table v1 is currently validated exclusively for C12 (Z=6, A=12); "
                    "received primary ion Z=" + std::to_string(config.primary_atomic_number) +
                    ", A=" + std::to_string(config.primary_mass_number));
            }

            // 2. Fail-closed Energy Domain Guard: Primary birth energies must be within [0.01, 430.0] MeV/u
            const double initial_e = config.initial_energy_MeVu;
            if (!std::isfinite(initial_e) || initial_e < kSchneiderStoppingEnergyMin || initial_e > 430.0 + 1e-5) {
                throw std::invalid_argument(
                    "Schneider stopping power mode requires finite initial_energy_MeVu in [" +
                    std::to_string(kSchneiderStoppingEnergyMin) + ", 430.0] MeV/u, got " +
                    std::to_string(initial_e));
            }
            if (!std::isfinite(config.beam_energy_spread) || config.beam_energy_spread < 0.0) {
                throw std::invalid_argument("Schneider stopping power mode requires finite beam_energy_spread >= 0.0");
            }
            constexpr double kMaxGaussianSupport = 7.433851508; // sqrt(-2 ln 1e-12)
            if (config.beam_energy_spread > 0.0) {
                const double max_possible_e = initial_e * (1.0 + kMaxGaussianSupport * config.beam_energy_spread);
                const double min_possible_e = initial_e * (1.0 - kMaxGaussianSupport * config.beam_energy_spread);
                if (max_possible_e > kSchneiderStoppingEnergyMax) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: beam energy spread allows birth energies up to " +
                        std::to_string(max_possible_e) + " MeV/u, exceeding table maximum " +
                        std::to_string(kSchneiderStoppingEnergyMax) + " MeV/u");
                }
                if (min_possible_e < kSchneiderStoppingEnergyMin) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: beam energy spread allows birth energies down to " +
                        std::to_string(min_possible_e) + " MeV/u, below table minimum " +
                        std::to_string(kSchneiderStoppingEnergyMin) + " MeV/u");
                }
            }

            // Audit all primary spots in spot batch
            for (std::size_t si = 0; si < config.primary_spot_batch.size(); ++si) {
                const auto& spot = config.primary_spot_batch[si];
                const double spot_e = static_cast<double>(spot.floats[0]) / 12.0; // total MeV to MeV/u for C12
                if (!std::isfinite(spot_e) || spot_e < kSchneiderStoppingEnergyMin || spot_e > 430.0 + 1e-5) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: spot " + std::to_string(si) +
                        " energy " + std::to_string(spot_e) + " MeV/u is outside valid domain [" +
                        std::to_string(kSchneiderStoppingEnergyMin) + ", 430.0] MeV/u");
                }
                const double spot_spread = static_cast<double>(spot.floats[1]);
                if (!std::isfinite(spot_spread) || spot_spread < 0.0) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: spot " + std::to_string(si) +
                        " energy spread must be finite and >= 0.0");
                }
                if (spot_spread > 0.0) {
                    const double max_spot_e = spot_e * (1.0 + kMaxGaussianSupport * spot_spread);
                    const double min_spot_e = spot_e * (1.0 - kMaxGaussianSupport * spot_spread);
                    if (max_spot_e > kSchneiderStoppingEnergyMax) {
                        throw std::invalid_argument(
                            "Schneider stopping power mode: spot " + std::to_string(si) +
                            " energy spread allows birth energies up to " +
                            std::to_string(max_spot_e) + " MeV/u, exceeding table maximum " +
                            std::to_string(kSchneiderStoppingEnergyMax) + " MeV/u");
                    }
                    if (min_spot_e < kSchneiderStoppingEnergyMin) {
                        throw std::invalid_argument(
                            "Schneider stopping power mode: spot " + std::to_string(si) +
                            " energy spread allows birth energies down to " +
                            std::to_string(min_spot_e) + " MeV/u, below table minimum " +
                            std::to_string(kSchneiderStoppingEnergyMin) + " MeV/u");
                    }
                }
            }

            if (std::abs(config.ct_stopping_power_scale - 1.0) > 1e-6) {
                throw std::invalid_argument(
                    "Exact Schneider stopping power mode requires ct_stopping_power_scale == 1.0; calibration scaling is forbidden");
            }

            auto stopping_file = config.ct_schneider_stopping_power_file;
            if (stopping_file.empty()) {
                const auto default_stopping_bin = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
                if (std::filesystem::exists(default_stopping_bin)) {
                    stopping_file = default_stopping_bin;
                } else {
                    throw std::runtime_error(
                        "Schneider CT grid requires ct_schneider_stopping_power_file; silent fallback to legacy SP is strictly forbidden");
                }
            }
            if (!std::filesystem::exists(stopping_file)) {
                throw std::runtime_error(
                    "Schneider stopping power file missing: " + stopping_file.string());
            }

            // Verify all voxel material IDs are strictly in [0, 24]
            for (std::size_t vi = 0; vi < grid.material_id.size(); ++vi) {
                if (grid.material_id[vi] >= kSchneiderStoppingNumSections) {
                    throw std::runtime_error(
                        "Schneider CT grid contains invalid material ID " + std::to_string(grid.material_id[vi]) +
                        " at voxel " + std::to_string(vi) + " (must be in [0, 24])");
                }
            }

            RuntimeScope t_ro_sp_dev("setup_schneider_stopping_ion_device");
            const auto meta_path = stopping_file.parent_path() /
                                   (stopping_file.stem().string() + ".metadata.json");
            const SchneiderStoppingTable* stopping_table = nullptr;
            const std::vector<float>* flat_sp = nullptr;
            std::shared_ptr<SchneiderStoppingTable> local_stopping;
            std::shared_ptr<std::vector<float>> local_flat_sp;
            if (schneider_host_cache && schneider_host_cache->schneider_stopping &&
                schneider_host_cache->schneider_stopping_flat) {
                stopping_table = schneider_host_cache->schneider_stopping.get();
                flat_sp = schneider_host_cache->schneider_stopping_flat.get();
                ++schneider_host_cache->hits;
            } else {
                local_stopping = std::make_shared<SchneiderStoppingTable>(
                    SchneiderStoppingTable::from_binary(stopping_file, meta_path));
                local_flat_sp = std::make_shared<std::vector<float>>(
                    local_stopping->to_flat_mass_stopping_float());
                stopping_table = local_stopping.get();
                flat_sp = local_flat_sp.get();
                if (schneider_host_cache) {
                    schneider_host_cache->schneider_stopping = local_stopping;
                    schneider_host_cache->schneider_stopping_flat = local_flat_sp;
                    ++schneider_host_cache->misses;
                }
            }
            schneider_sp_sections = static_cast<std::uint32_t>(stopping_table->num_sections());
            schneider_sp_energies = static_cast<std::uint32_t>(stopping_table->num_energies());
            schneider_sp_e_min = static_cast<float>(stopping_table->energy_min_mevu());
            schneider_sp_e_max = static_cast<float>(stopping_table->energy_max_mevu());
            const float dE = static_cast<float>(stopping_table->energy_step_mevu());
            schneider_sp_inv_dE = 1.0F / dE;

            if (schneider_sp_sections != 25 || schneider_sp_energies != 4302) {
                throw std::runtime_error(
                    "Schneider stopping table dimension mismatch: sections=" + std::to_string(schneider_sp_sections) +
                    ", energies=" + std::to_string(schneider_sp_energies) + " (expected 25x4302)");
            }

            const std::size_t total_sp_elements = flat_sp->size();
            schneider_stopping_device = mem_tracker.allocate<float>(total_sp_elements);
            if (schneider_stopping_device == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(flat_sp->data(), schneider_stopping_device, total_sp_elements).wait_and_throw();
            use_schneider_stopping = true;

            if (!config.ct_secondary_ion_section_stopping_file.empty()) {
                const auto ion_meta_path =
                    config.ct_secondary_ion_section_stopping_file.parent_path() /
                    (config.ct_secondary_ion_section_stopping_file.stem().string() +
                     ".metadata.json");
                const std::vector<float>* flat_ion = nullptr;
                std::shared_ptr<SchneiderIonStoppingTable> local_ion_table;
                std::shared_ptr<std::vector<float>> local_flat_ion;
                if (schneider_host_cache &&
                    schneider_host_cache->schneider_ion_stopping &&
                    schneider_host_cache->schneider_ion_stopping_flat) {
                    flat_ion = schneider_host_cache->schneider_ion_stopping_flat.get();
                    ++schneider_host_cache->hits;
                } else {
                    // Verified once per session; reuse after that.
                    if (!file_sha256_matches(ion_meta_path,
                        config.ct_secondary_ion_section_stopping_metadata_sha256))
                        throw std::runtime_error(
                            "Section ion stopping metadata SHA256 mismatch");
                    if (!file_sha256_matches(
                            config.ct_secondary_ion_section_stopping_file,
                            config.ct_secondary_ion_section_stopping_sha256))
                        throw std::runtime_error(
                            "Section ion stopping data SHA256 mismatch");
                    local_ion_table = std::make_shared<SchneiderIonStoppingTable>(
                        SchneiderIonStoppingTable::from_binary(
                            config.ct_secondary_ion_section_stopping_file, ion_meta_path));
                    local_flat_ion = std::make_shared<std::vector<float>>(
                        local_ion_table->to_flat_float());
                    flat_ion = local_flat_ion.get();
                    if (schneider_host_cache) {
                        schneider_host_cache->schneider_ion_stopping = local_ion_table;
                        schneider_host_cache->schneider_ion_stopping_flat = local_flat_ion;
                        ++schneider_host_cache->misses;
                    }
                }
                schneider_ion_sp_device = mem_tracker.allocate<float>(flat_ion->size());
                if (schneider_ion_sp_device == nullptr) {
                    throw std::bad_alloc();
                }
                queue.copy(flat_ion->data(), schneider_ion_sp_device, flat_ion->size())
                    .wait_and_throw();
                use_schneider_ion_sp = true;
                ion_stopping_failures = mem_tracker.allocate<std::uint32_t>(8);
                queue.fill(ion_stopping_failures, std::uint32_t{0}, 8).wait_and_throw();
                std::cout << "[schneider-ion-stopping] mode=section-ion-v1 sections=25 "
                          << "species=18 energies=" << kSchneiderIonEnergies << " source="
                          << config.ct_secondary_ion_section_stopping_file << '\n';
            }
            t_ro_sp_dev.finish();

            std::cout << "[schneider-stopping-power] mode=exact-schneider-v1\n"
                      << "  sections=" << schneider_sp_sections << "\n"
                      << "  energies=" << schneider_sp_energies << "\n"
                      << "  bytes=" << total_sp_elements * sizeof(float) << "\n"
                      << "  E_min=" << schneider_sp_e_min << " MeV/u\n"
                      << "  E_max=" << schneider_sp_e_max << " MeV/u\n"
                      << "  inv_dE=" << schneider_sp_inv_dE << "\n"
                      << "  source=" << config.ct_schneider_stopping_power_file << "\n"
                      << "  source_sha256=" << (early_verified_stopping_sha256.empty() ? "unknown" : early_verified_stopping_sha256) << "\n";
        }

        // Exact C12 stopping only replaces the PRIMARY lookup. Secondary ions
        // still need their 25-section material factors; previously this guard
        // also suppressed their LUT, leaving water stopping times density.
        if (use_ct_mass_sp && (!use_schneider_stopping || config.ct_secondary_schneider_sp_diagnostic)) {
            if (config.ct_secondary_schneider_sp_diagnostic) {
                if (!ct_material_ids_are_schneider_sections || grid.mass_sp_za_rel.size()!=25 || grid.mass_sp_I_eV.size()!=25)
                    throw std::invalid_argument("Secondary Schneider stopping requires 25 exact section Z/A and I entries");
                for (std::size_t s=0;s<25;++s)
                    if (!std::isfinite(grid.mass_sp_za_rel[s]) || grid.mass_sp_za_rel[s]<=0 ||
                        !std::isfinite(grid.mass_sp_I_eV[s]) || grid.mass_sp_I_eV[s]<=0)
                        throw std::invalid_argument("Invalid secondary Schneider material parameters");
                std::cout << "[secondary-schneider-stopping] diagnostic: 25-section Z/A,I Bethe factors; exact primary table unchanged; no four-class LUT\n";
            }
            ct_n_mass_factors = static_cast<std::uint32_t>(grid.mass_sp_za_rel.size());
            std::vector<float> mass_factor_lut(
                static_cast<std::size_t>(ct_n_mass_factors) * table_size);
            const auto sp_scale = static_cast<float>(config.ct_stopping_power_scale);

            const auto try_density_spr = [&]() -> bool {
                // This diagnostic is exclusively Schneider-25. Do not enter
                // the legacy air/lung/bone density-LUT branch below.
                if (config.ct_secondary_schneider_sp_diagnostic) return false;
                if (!config.ct_use_density_mass_spr) {
                    return false;
                }
                const auto resolve = [](const std::filesystem::path& user_path,
                                        const std::filesystem::path& fallback) {
                    if (!user_path.empty()) {
                        return std::filesystem::exists(user_path) ? user_path : std::filesystem::path{};
                    }
                    return std::filesystem::exists(fallback) ? fallback : std::filesystem::path{};
                };
                const auto air_path = resolve(config.ct_air_stopping_power_file,
                                             "data/stopping_power_air_geant4_11_3_2.csv");
                const auto lung_path = resolve(config.ct_lung_stopping_power_file,
                                              "data/stopping_power_lung_geant4_11_3_2.csv");
                const auto bone_path = resolve(config.ct_bone_stopping_power_file,
                                              "data/stopping_power_bone_geant4_11_3_2.csv");
                if (air_path.empty() || lung_path.empty() || bone_path.empty()) {
                    return false;
                }
                const auto air_table = StoppingPowerTable::from_csv(air_path);
                const auto lung_table = StoppingPowerTable::from_csv(lung_path);
                const auto bone_table = StoppingPowerTable::from_csv(bone_path);
                const auto density_lut = build_density_mass_spr_lut(
                    stopping_power, air_table, lung_table, bone_table, sp_scale);
                if (density_lut.n_rho < 2 ||
                    density_lut.factors.size() !=
                        static_cast<std::size_t>(density_lut.n_rho) * table_size) {
                    return false;
                }
                mass_factor_lut = density_lut.factors;
                ct_density_spr_n_rho = density_lut.n_rho;
                ct_mass_spr_log_rho_min = density_lut.log_rho_min;
                ct_mass_spr_inv_dlog = density_lut.inv_dlog;
                use_ct_density_mass_spr = true;
                return true;
            };

            if (!try_density_spr()) {
                if (config.is_primary_attenuation_only_mode()) {
                    throw std::runtime_error(
                        "primary-attenuation-only mode requires density-mass-SPR LUT to be successfully constructed without fallback");
                }
                for (std::uint32_t sec = 0; sec < ct_n_mass_factors; ++sec) {
                    const auto za = grid.mass_sp_za_rel[sec];
                    const auto I_eV = grid.mass_sp_I_eV[sec];
                    const auto base = static_cast<std::size_t>(sec) * table_size;
                    for (std::size_t i = 0; i < table_size; ++i) {
                        mass_factor_lut[base + i] =
                            sp_scale * ct_mass_sp_energy_factor(
                                           za, I_eV,
                                           static_cast<float>(stopping_power.energies()[i]));
                    }
                }
            }
            const auto lut_bytes = mass_factor_lut.size();
            ct_mass_sp_factor_lut_device = mem_tracker.allocate<float>(lut_bytes);
            ct_mass_sp_za_rel_device = mem_tracker.allocate<float>(ct_n_mass_factors);
            queue.copy(mass_factor_lut.data(), ct_mass_sp_factor_lut_device, lut_bytes);
            queue.copy(grid.mass_sp_za_rel.data(), ct_mass_sp_za_rel_device, ct_n_mass_factors)
                .wait_and_throw();
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
    ElasticRecoilStoppingView recoil_stopping{};
    float* elastic_energy = nullptr; // local, queued charged, overflow
    std::uint32_t* elastic_audit = nullptr; // primary, secondary, bad query, unsupported recoil
    if(use_all_elastic) {
        const auto stop=ElasticRecoilStoppingTable::load(config.elastic_recoil_stopping_file,config.elastic_recoil_stopping_sha256);
        auto* keys=mem_tracker.allocate<std::int32_t>(stop.keys.size());
        auto* se=mem_tracker.allocate<float>(stop.energies.size());auto* sv=mem_tracker.allocate<float>(stop.values.size());
        if(!keys||!se||!sv)throw std::bad_alloc();
        queue.copy(stop.keys.data(),keys,stop.keys.size());queue.copy(stop.energies.data(),se,stop.energies.size());queue.copy(stop.values.data(),sv,stop.values.size()).wait_and_throw();
        recoil_stopping={keys,se,sv,static_cast<std::uint32_t>(stop.keys.size()/2),static_cast<std::uint32_t>(stop.energies.size())};
        const auto bank=AllIonElasticTable::load(config.all_ion_elastic_file,config.all_ion_elastic_sha256);
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
    const bool minibeam_copper_transport = config.enable_minibeam &&
        config.minibeam_transport_mode == "copper_em";
    float* minibeam_copper_sp_energies_device = nullptr;
    float* minibeam_copper_sp_values_device = nullptr;
    std::uint32_t minibeam_copper_sp_count = 0;
    float* minibeam_copper_loss_e_device = nullptr;
    float* minibeam_copper_loss_r_device = nullptr;
    float* minibeam_copper_loss_d_device = nullptr;
    std::uint32_t minibeam_copper_loss_count = 0;
    float* minibeam_water_urban_loss_e_device = nullptr;
    float* minibeam_water_urban_loss_r_device = nullptr;
    float* minibeam_water_urban_loss_d_device = nullptr;
    std::uint32_t minibeam_water_urban_loss_count = 0;
    double minibeam_water_urban_zeff =
        std::numeric_limits<double>::quiet_NaN();
    double minibeam_water_urban_radlen_mm =
        std::numeric_limits<double>::quiet_NaN();
    float* minibeam_air_sp_energies_device = nullptr;
    float* minibeam_air_sp_values_device = nullptr;
    std::uint32_t minibeam_air_sp_count = 0;
    float* minibeam_copper_xs_energies_device = nullptr;
    float* minibeam_copper_xs_values_device = nullptr;
    std::uint32_t minibeam_copper_xs_count = 0;
    CopperElasticRecord* minibeam_copper_elastic_device = nullptr;
    std::uint32_t minibeam_copper_elastic_count = 0;
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
    if (minibeam_copper_transport) {
        const auto copper_sp = StoppingPowerTable::from_csv(
            config.minibeam_copper_stopping_power_file);
        const auto air_sp = StoppingPowerTable::from_csv(
            config.minibeam_air_stopping_power_file);
        const auto copper_xs = CrossSectionTable::from_csv(
            config.minibeam_copper_cross_section_file);
        const auto elastic = load_copper_elastic_events(
            config.minibeam_copper_elastic_file);
        std::vector<float> copper_ion_sp_ratios;
        std::vector<std::uint8_t> copper_ion_sp_present;
        std::vector<float> copper_ion_xs;
        std::vector<std::uint8_t> copper_ion_xs_present;
        if (!config.minibeam_copper_ion_stopping_power_file.empty()) {
            const auto ion_stopping = IonStoppingPowerTables::from_csv(
                config.minibeam_copper_ion_stopping_power_file, copper_sp);
            copper_ion_sp_ratios = ion_stopping.ratios_to_carbon();
            copper_ion_sp_present = ion_stopping.species_present();
        }
        if (!config.minibeam_copper_ion_cross_section_file.empty()) {
            const auto ion_xs = IonCrossSectionTables::from_csv(
                config.minibeam_copper_ion_cross_section_file);
            copper_ion_xs = ion_xs.values();
            copper_ion_xs_present = ion_xs.species_present();
            minibeam_copper_ion_xs_grid_size =
                static_cast<std::uint32_t>(ion_xs.energy_grid_size());
            minibeam_copper_ion_xs_minimum_energy = ion_xs.minimum_energy_MeVu();
            minibeam_copper_ion_xs_inverse_step = 1.0F / ion_xs.energy_step_MeVu();
        }
        const auto package = InelasticPackageV3Table::from_binary(
            config.minibeam_copper_inclxx_file);
        const auto tables = package.make_device_tables();
        upload_float_pair(copper_sp.energies(), copper_sp.values(),
                          minibeam_copper_sp_energies_device,
                          minibeam_copper_sp_values_device);
        minibeam_copper_sp_count = static_cast<std::uint32_t>(copper_sp.energies().size());
        if (!config.minibeam_copper_loss_range_file.empty()) {
            const auto loss_table = UrbanLossRangeTable::from_csv(
                config.minibeam_copper_loss_range_file);
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
    // It feeds the water primary urban_v2 path, which also runs under
    // absorbing_geometry (water-entry replays). Gating it on copper_em left
    // the table null there, so every proposal returned a zero geom path and
    // the subdivision loop spun to its cap (effective hang). Upload whenever
    // minibeam is on and the file is configured.
    if (config.enable_minibeam &&
        !config.minibeam_water_urban_loss_range_file.empty()) {
        const auto water_table = UrbanLossRangeTable::from_csv(
            config.minibeam_water_urban_loss_range_file);
        upload_float_pair(water_table.energies_total_mev(),
                          water_table.ranges_mm(),
                          minibeam_water_urban_loss_e_device,
                          minibeam_water_urban_loss_r_device);
        std::vector<float> water_urban_wd(water_table.dedx_values().begin(),
                                          water_table.dedx_values().end());
        minibeam_water_urban_loss_d_device =
            mem_tracker.allocate<float>(water_urban_wd.size());
        if (!minibeam_water_urban_loss_d_device) throw std::bad_alloc();
        queue.copy(water_urban_wd.data(), minibeam_water_urban_loss_d_device,
                   water_urban_wd.size());
        minibeam_water_urban_loss_count =
            static_cast<std::uint32_t>(water_table.energies_total_mev().size());
        minibeam_water_urban_zeff = water_table.zeff();
        minibeam_water_urban_radlen_mm = water_table.radlen_mm();
        std::cout << "[minibeam-water] urban loss-range nodes="
                  << minibeam_water_urban_loss_count << " zeff="
                  << minibeam_water_urban_zeff << " radlen_mm="
                  << minibeam_water_urban_radlen_mm << '\n';
    }
    if (config.enable_minibeam &&
        config.minibeam_water_primary_mcs_model == "urban_v2" &&
        minibeam_water_urban_loss_count < 2) {
        throw std::runtime_error(
            "minibeam water urban_v2 selected but the loss-range table was "
            "not loaded; refusing to transport with null tables");
    }
#endif
    RuntimeScope t_sch_ctx("setup_schneider_ctx_host_and_device");
    const SchneiderCtDeviceContext schneider_ct_device_ctx = upload_schneider_ct_device_context(
        queue, mem_tracker, config,
        schneider_stopping_device,
        schneider_sp_sections, schneider_sp_energies,
        schneider_sp_e_min, schneider_sp_e_max, schneider_sp_inv_dE,
        schneider_host_cache);
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
    constexpr bool cinel02_topas_compatibility_mode = false;
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_let_scoring = EmMode!=1 && config.enable_let_scoring;
    const auto voxel_scorer_clamps_transport = config.voxel_scorer_clamps_transport;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    const auto voxel_bins_z = config.voxel_bins_z;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.voxel_size_z_mm);
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
        config.enable_minibeam && enable_charged_origin_voxel_scoring;
    const auto enable_minibeam_energy_band_roi_scoring =
        config.enable_minibeam_energy_band_roi_scoring;
    const auto enable_minibeam_primary_c12_roi_scoring =
        config.enable_minibeam_primary_c12_roi_scoring;
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
    const auto minibeam_c12_roi_value_count =
        minibeam_c12_roi_kind_count * minibeam_energy_band_count *
        minibeam_fixed_region_count * number_of_bins;
    auto* minibeam_c12_roi_dose_device =
        enable_minibeam_primary_c12_roi_scoring
            ? mem_tracker.allocate<DoseAtomicT>(minibeam_c12_roi_value_count)
            : nullptr;
    auto* minibeam_spatial_audit_device =
        enable_minibeam_primary_c12_roi_scoring
            ? mem_tracker.allocate<std::uint64_t>(minibeam_spatial_audit_slot_count)
            : nullptr;
    auto* minibeam_fixed_region_by_x_bin_device =
        (enable_minibeam_energy_band_roi_scoring ||
         enable_minibeam_primary_c12_roi_scoring)
            ? mem_tracker.allocate<std::uint8_t>(voxel_bins_x)
            : nullptr;
    if ((enable_minibeam_energy_band_roi_scoring ||
         enable_minibeam_primary_c12_roi_scoring) &&
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
#if defined(CARBON_ENABLE_MINIBEAM)
    const bool water_entry_secondary_replay =
        !config.minibeam_water_entry_secondary_replay_file.empty();
#else
    constexpr bool water_entry_secondary_replay = false;
#endif

    auto* deposited_device = mem_tracker.allocate<float>(number_of_histories);
    auto* sampled_incident_device = mem_tracker.allocate<float>(number_of_histories);
    auto* escaped_device = mem_tracker.allocate<float>(number_of_histories);
    auto* steps_device = mem_tracker.allocate<std::uint32_t>(number_of_histories);
    auto* untracked_nuclear_device = mem_tracker.allocate<float>(number_of_histories);
    auto* other_terminal_energy_device = mem_tracker.allocate<float>(number_of_histories);
    auto* cutoff_stopped_energy_device = mem_tracker.allocate<float>(number_of_histories);
#if defined(CARBON_ENABLE_MINIBEAM)
    const auto enable_minibeam_phase_space =
        config.enable_minibeam &&
        !config.minibeam_phase_space_output_file.empty();
    auto* minibeam_phase_space_device = enable_minibeam_phase_space
        ? mem_tracker.allocate<MinibeamPhaseSpaceRecord>(number_of_histories)
        : nullptr;
    const auto minibeam_water_primary_plane_count =
        config.minibeam_water_primary_plane_depths_mm.size();
    const auto enable_minibeam_water_primary_planes =
        config.enable_minibeam &&
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
    auto* beamline_removed_device = config.enable_minibeam
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_primary_survivor_device = config.enable_minibeam
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* beamline_air_loss_device = config.enable_minibeam
        ? mem_tracker.allocate<float>(number_of_histories) : nullptr;
    auto* minibeam_event_counts_device = config.enable_minibeam
        ? mem_tracker.allocate<std::uint64_t>(minibeam_event_counter_count)
        : nullptr;
    // Research-only diagnostic: total Water-Urban segments (one atomic per
    // history that used the urban water path).
    auto* water_urban_segment_count_device = config.enable_minibeam
        ? mem_tracker.allocate<std::uint64_t>(1)
        : nullptr;
    // Research-only Unified-water delta response ledger (fixed-point MeV*1e6
    // and counts; 17 slots, see printout below). No transport feedback.
    auto* minibeam_water_delta_diag_device = config.enable_minibeam
        ? mem_tracker.allocate<std::uint64_t>(17)
        : nullptr;
    const bool enable_minibeam_fragment_miss_joint =
        config.enable_minibeam &&
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
        (config.enable_minibeam && (beamline_removed_device == nullptr ||
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
    free_device(water_urban_segment_count_device);
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
        if (water_urban_segment_count_device != nullptr) {
            queue.fill(water_urban_segment_count_device, std::uint64_t{0}, 1);
        }
        if (minibeam_water_delta_diag_device != nullptr) {
            queue.fill(minibeam_water_delta_diag_device, std::uint64_t{0}, 17);
        }
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
        (enable_inelastic || water_entry_secondary_replay);
#if defined(CARBON_ENABLE_MINIBEAM)
    std::optional<WaterEntrySecondaryReplay> secondary_replay;
    if (water_entry_secondary_replay) {
        secondary_replay = load_water_entry_secondary_replay(
            config.minibeam_water_entry_secondary_replay_file,
            number_of_histories, max_secondaries,
            config.minibeam_water_entry_secondary_replay_allow_primary_c12,
            config.minibeam_water_entry_secondary_replay_allow_internal_births,
            static_cast<float>(config.phantom_length_mm));
        std::cout << "[water-entry-secondary-replay] loaded="
                  << secondary_replay->particles.size()
                  << " particles; histories-normalization="
                  << number_of_histories << '\n';
    }
#endif
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
#if defined(CARBON_ENABLE_MINIBEAM)
    auto* minibeam_copper_cascade_queue_device =
        need_secondary_buffers && config.enable_minibeam &&
                config.minibeam_copper_fragment_cascade_generations > 0
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* minibeam_copper_cascade_count_device =
        minibeam_copper_cascade_queue_device
            ? mem_tracker.allocate<std::uint32_t>(1)
            : nullptr;
    const auto enable_minibeam_fragment_phase_space =
        config.enable_minibeam && need_secondary_buffers &&
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
#endif
    if (need_secondary_buffers) {
        if (!secondary_queue_device || !secondary_count_device ||
            !secondary_overflow_count_device || !secondary_overflow_energy_device
#if defined(CARBON_ENABLE_MINIBEAM)
            || (enable_minibeam_fragment_phase_space &&
                (!minibeam_fragment_phase_space_device ||
                 !minibeam_fragment_phase_space_count_device))
            || (config.enable_minibeam &&
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
#if defined(CARBON_ENABLE_MINIBEAM)
    if (secondary_replay.has_value()) {
        queue.fill(deposited_device, 0.0F, number_of_histories);
        queue.fill(escaped_device, 0.0F, number_of_histories);
        queue.fill(steps_device, std::uint32_t{0}, number_of_histories);
        queue.copy(secondary_replay->incident_energy_by_history.data(),
                   sampled_incident_device, number_of_histories);
        queue.copy(secondary_replay->particles.data(), secondary_queue_device,
                   secondary_replay->particles.size());
        const auto replay_count =
            static_cast<std::uint32_t>(secondary_replay->particles.size());
        queue.copy(&replay_count, secondary_count_device, 1).wait_and_throw();
    }
#endif

    float* fluct_energy_device = nullptr;
    float* fluct_density_device = nullptr;
    float* fluct_probability_device = nullptr;
    float* fluct_quantile_device = nullptr;
    std::size_t fluct_energy_count = 0;
    std::size_t fluct_density_count = 0;
    std::size_t fluct_probability_count = 0;
    const bool fluct_fraction_hybrid = config.energy_straggling_model == "packaged_fluctuation_fraction_hybrid";
    const bool fluct_fraction_axis = config.energy_straggling_model == "packaged_fluctuation_fraction" || fluct_fraction_hybrid;
    std::uint32_t* fluct_domain_failures = nullptr;
    std::uint64_t* primary_loss_query_audit = nullptr;
    if (config.enable_primary_loss_query_audit) {
        primary_loss_query_audit = mem_tracker.allocate<std::uint64_t>(28);
        if (!primary_loss_query_audit) throw std::bad_alloc();
        queue.fill(primary_loss_query_audit, std::uint64_t{0}, 28).wait_and_throw();
    }
    if (config.uses_packaged_fluctuation()) {
        const auto host = carbon::EnergyLossFluctuationTable::from_csv(
            config.energy_straggling_package_file, fluct_fraction_axis);
        if (host.projectile_atomic_number() != 6 ||
            host.projectile_mass_number() != 12 ||
            host.material_name() != (fluct_fraction_axis ? "Water_75eV" : "G4_WATER"))
            throw std::runtime_error(
                "Packaged fluctuation must describe C-12 in G4_WATER");
        const auto to_float = [](const std::vector<double>& input) {
            std::vector<float> output(input.size());
            std::transform(input.begin(), input.end(), output.begin(),
                           [](double value) { return static_cast<float>(value); });
            return output;
        };
        const auto energies = to_float(host.energies_MeVu());
        const auto densities = to_float(host.second_axis_values());
        if (fluct_fraction_hybrid && densities.front() > 1.0e-4F)
            throw std::runtime_error("Hybrid fraction grid must cover f >= 1e-4");
        if (fluct_fraction_axis) {
            if (!use_unified_water || config.water_density_g_per_cm3 != 1.0)
                throw std::runtime_error("Fraction-axis candidate requires native homogeneous density-1 water");
            fluct_domain_failures = mem_tracker.allocate<std::uint32_t>(1);
            queue.fill(fluct_domain_failures, std::uint32_t{0}, 1).wait_and_throw();
        }
        const auto probabilities = to_float(host.probabilities());
        const auto quantiles = to_float(host.loss_ratio_quantiles());
        fluct_energy_count = energies.size();
        fluct_density_count = densities.size();
        fluct_probability_count = probabilities.size();
        fluct_energy_device = mem_tracker.allocate<float>(energies.size());
        fluct_density_device = mem_tracker.allocate<float>(densities.size());
        fluct_probability_device = mem_tracker.allocate<float>(probabilities.size());
        fluct_quantile_device = mem_tracker.allocate<float>(quantiles.size());
        if (!fluct_energy_device || !fluct_density_device ||
            !fluct_probability_device || !fluct_quantile_device)
            throw std::bad_alloc();
        queue.copy(energies.data(), fluct_energy_device, energies.size());
        queue.copy(densities.data(), fluct_density_device, densities.size());
        queue.copy(probabilities.data(), fluct_probability_device, probabilities.size());
        queue.copy(quantiles.data(), fluct_quantile_device, quantiles.size()).wait_and_throw();
        std::cout << "Loaded packaged C-12 fluctuation grid (" << fluct_energy_count
                  << " energies, " << fluct_density_count << " thicknesses, "
                  << fluct_probability_count << " quantiles)\n";
    }

    float* ion_species_sp_device = nullptr;
    float* ion_energy_grid_device = nullptr;
    float* ion_csda_a1_device = nullptr;
    if (enable_inelastic || water_entry_secondary_replay) {
        // Native unified water must honor its explicit material-specific table.
        // Preserve the frozen CT/legacy route until separately validated.
        std::filesystem::path ion_sp_path = use_unified_water
            ? config.particle_stopping_power_file
            : std::filesystem::path("data/ion_stopping_power_water_geant4_11_3_2.csv");
        if (use_unified_water) {
            if (ion_sp_path.empty() || !std::filesystem::exists(ion_sp_path)) {
                throw std::runtime_error("Unified water requires its configured ion stopping table");
            }
            // The device loader indexes rows by primary-table index. Reject
            // different energy grids before uploading rather than misindexing.
            (void)IonStoppingPowerTables::from_csv(ion_sp_path, stopping_power);
            std::cout << "Unified water active ion stopping table: " << ion_sp_path
                      << " SHA256=" << file_sha256_for_report(ion_sp_path) << '\n';
        }
        if (!std::filesystem::exists(ion_sp_path)) {
            const auto cur_p = std::filesystem::current_path();
            if (std::filesystem::exists(cur_p / "data" / "ion_stopping_power_water_geant4_11_3_2.csv")) {
                ion_sp_path = cur_p / "data" / "ion_stopping_power_water_geant4_11_3_2.csv";
            }
        }
        if (!std::filesystem::exists(ion_sp_path)) {
            throw std::runtime_error("Required ion stopping-power CSV not found: " + ion_sp_path.string());
        }
        const auto ion_sp_lut = load_ion_species_stopping_power_lut(ion_sp_path, table_size, 1.0F);
        ion_species_sp_device = mem_tracker.allocate<float>(18 * table_size);
        queue.copy(ion_sp_lut.data(), ion_species_sp_device, 18 * table_size).wait_and_throw();
        std::vector<float> energy_grid_host(table_size);
        std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                       energy_grid_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        ion_energy_grid_device = mem_tracker.allocate<float>(table_size);
        ion_csda_a1_device = mem_tracker.allocate<float>(18 * table_size);
        if (ion_energy_grid_device == nullptr || ion_csda_a1_device == nullptr) {
            throw std::bad_alloc();
        }
        std::vector<float> csda_host(18 * table_size, 0.0F);
        for (int species = 0; species < 18; ++species) {
            fill_a1_csda_range_mm(energy_grid_host.data(),
                                  ion_sp_lut.data() + static_cast<std::size_t>(species) * table_size,
                                  table_size,
                                  csda_host.data() + static_cast<std::size_t>(species) * table_size);
        }
        queue.copy(energy_grid_host.data(), ion_energy_grid_device, table_size);
        queue.copy(csda_host.data(), ion_csda_a1_device, 18 * table_size).wait_and_throw();
    }

    const auto primary_species_sp_idx = carbon::get_charged_species_idx(
        config.primary_atomic_number, config.primary_mass_number);
    if (enable_inelastic && primary_species_sp_idx < 0) {
        throw std::runtime_error(
            "Primary ion is absent from the explicit ion stopping-power table");
    }
    const auto primary_species_offset = primary_species_sp_idx >= 0
        ? static_cast<std::size_t>(primary_species_sp_idx) * table_size
        : 0U;
    const float* primary_water_sp_device = enable_inelastic
        ? ion_species_sp_device + primary_species_offset
        : table_device;
    const float* primary_csda_a1_device = enable_inelastic
        ? ion_csda_a1_device + primary_species_offset
        : cumulative_range_device;
    // EM-only transport never allocates the ion-species grid. Use the
    // primary grid uploaded with its cumulative-range table in that case.
    const float* primary_csda_energy_grid_device = enable_inelastic
        ? ion_energy_grid_device : energy_grid_device;

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
        (enable_minibeam_primary_c12_roi_scoring &&
         (minibeam_c12_roi_dose_device == nullptr ||
          minibeam_fixed_region_by_x_bin_device == nullptr ||
          minibeam_spatial_audit_device == nullptr)) ||
        (enable_let_scoring && let_moments_device == nullptr)) {
        throw std::bad_alloc();
    }

    if (!reuse_immutable_buffers) {
        std::vector<float> table_host(table_size);
        std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                       table_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(table_host.data(), table_device, table_size);
        if (config.enable_csda_range_energy_loss) {
            std::vector<float> energy_grid_host(table_size);
            std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                           energy_grid_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(energy_grid_host.data(), energy_grid_device, table_size);
            std::vector<float> cumulative_range_host(table_size);
            std::transform(stopping_power.cumulative_ranges_mm().begin(),
                           stopping_power.cumulative_ranges_mm().end(),
                           cumulative_range_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(cumulative_range_host.data(), cumulative_range_device, table_size);
        }
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
    if (enable_minibeam_primary_c12_roi_scoring) {
        queue.memset(minibeam_c12_roi_dose_device, 0,
                     minibeam_c12_roi_value_count *
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
            history_chunk = number_of_histories > 1000000 ? 34816 : 4096;
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

    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto inverse_depth_bin_width_mm = 1.0F / depth_bin_width_mm;
    const auto inverse_voxel_size_x_mm = 1.0F / voxel_size_x_mm;
    const auto inverse_voxel_size_y_mm = 1.0F / voxel_size_y_mm;
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto maximum_primary_steps = config.maximum_primary_steps;
    const auto energy_cutoff_MeV = static_cast<float>(
        use_schneider_stopping ? std::max(config.energy_cutoff_MeV, 12.0 * static_cast<double>(schneider_sp_e_min))
                               : config.energy_cutoff_MeV);

    if (is_cuda_backend && !config.enable_minibeam) {
        cuda_clock_warmup(queue, mem_tracker);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_step_stable_straggling =
        enable_energy_straggling && config.enable_step_stable_straggling;
    const auto straggling_sampling_length_mm =
        static_cast<float>(config.straggling_sampling_length_mm);
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto straggling_sampler = config.straggling_sampler_id();
    const auto use_packaged_fluctuation = config.uses_packaged_fluctuation();
    const auto enable_secondary_energy_straggling =
        config.enable_secondary_energy_straggling;

    std::array<float, max_straggling_scale_points> straggling_scale_energies{};
    std::array<float, max_straggling_scale_points> straggling_scale_values{};
    const auto straggling_scale_point_count =
        config.straggling_scale_energies_MeVu.size();
    for (std::size_t index = 0; index < straggling_scale_point_count; ++index) {
        straggling_scale_energies[index] =
            static_cast<float>(config.straggling_scale_energies_MeVu[index]);
        straggling_scale_values[index] =
            static_cast<float>(config.straggling_scale_values[index]);
    }

    const auto multiple_scattering_scale =
        static_cast<float>(config.multiple_scattering_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto active_water_radiation_length = use_unified_water
        ? schneider_ct_device_ctx.water_radiation_length_g_cm2 : water_radiation_length_g_per_cm2;
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    const auto c12_fermi_eyges =
        config.multiple_scattering_model == "fermi_eyges";
    const auto fermi_eyges_species_scope =
        config.fermi_eyges_species == "all_charged" ? 3 :
        (config.fermi_eyges_species == "c12_he4_pdt" ? 2 :
         (config.fermi_eyges_species == "c12_he4" ? 1 : 0));
    const auto fermi_eyges_use_species_water_parameters =
        config.fermi_eyges_parameter_set == "species_water";
    const auto c12_fermi_eyges_max_segment_mm =
        static_cast<float>(config.fermi_eyges_max_segment_mm);
    if (c12_fermi_eyges) {
        std::cout << "[ion-mcs] model=fermi_eyges species="
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
    const auto enable_minibeam = config.enable_minibeam;
    const auto minibeam_absorbing_geometry = config.enable_minibeam &&
        config.minibeam_transport_mode == "absorbing_geometry";
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
    const auto minibeam_copper_fermi_eyges_tail =
        config.minibeam_copper_mcs_model == "fermi_eyges_tail";
    const auto minibeam_copper_urban_msc =
        config.minibeam_copper_mcs_model == "urban";
    const auto minibeam_copper_urban_v2_msc =
        config.minibeam_copper_mcs_model == "urban_v2";
    const auto minibeam_copper_correlated_scattering =
        minibeam_copper_fermi_eyges_tail || minibeam_copper_urban_msc ||
        minibeam_copper_urban_v2_msc;
    const auto minibeam_copper_enable_mcs = config.minibeam_copper_enable_mcs;
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
    const auto minibeam_water_low_energy_mcs_transition = static_cast<float>(
        config.minibeam_water_low_energy_mcs_transition_MeVu);
    const auto minibeam_water_primary_low_energy_mcs_scale = static_cast<float>(
        config.minibeam_water_primary_low_energy_mcs_scale);
    const auto minibeam_water_primary_mcs_tail_strength = static_cast<float>(
        config.minibeam_water_primary_mcs_tail_strength);
    const auto minibeam_water_primary_mcs_tail_width = static_cast<float>(
        config.minibeam_water_primary_mcs_tail_width);
    const auto minibeam_water_primary_fermi_eyges_tail =
        config.minibeam_water_primary_mcs_model == "fermi_eyges_tail";
    const auto minibeam_water_primary_urban_v2 =
        config.minibeam_water_primary_mcs_model == "urban_v2";
    const auto minibeam_water_primary_mcs_max_segment_mm =
        static_cast<float>(
            config.minibeam_water_primary_mcs_max_segment_mm);
    const auto minibeam_water_primary_urban_max_step_mm =
        static_cast<float>(
            config.minibeam_water_primary_urban_max_step_mm);
    const auto minibeam_water_urban_zeff_f =
        static_cast<float>(minibeam_water_urban_zeff);
    const auto minibeam_water_urban_radlen_mm_f =
        static_cast<float>(minibeam_water_urban_radlen_mm);
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
    const auto minibeam_water_secondary_c12_fermi_eyges_tail =
        config.enable_minibeam &&
        config.minibeam_water_secondary_c12_mcs_model == "fermi_eyges_tail";
    const auto minibeam_water_secondary_c12_urban_v2 =
        config.enable_minibeam &&
        config.minibeam_water_secondary_c12_mcs_model == "urban_v2";
    const auto minibeam_water_secondary_c12_mcs_max_segment_mm =
        static_cast<float>(
            config.minibeam_water_secondary_c12_mcs_max_segment_mm);
    const auto minibeam_water_secondary_c12_enable_unified_em =
        config.enable_minibeam &&
        config.minibeam_water_secondary_c12_enable_unified_em;
    const auto minibeam_water_secondary_c12_post_sample_loss_scale =
        static_cast<float>(
            config.minibeam_water_secondary_c12_post_sample_loss_scale);
    if (minibeam_water_secondary_c12_fermi_eyges_tail) {
        std::cout << "[minibeam-water-secondary-c12-mcs] "
                     "model=fermi_eyges_tail scope=C12-only "
                     "other_species=legacy_highland max_segment_mm="
                  << minibeam_water_secondary_c12_mcs_max_segment_mm << '\n';
    }
    if (minibeam_water_secondary_c12_urban_v2) {
#if defined(CARBON_ENABLE_MINIBEAM)
        std::cout << "[minibeam-water-secondary-c12-mcs] "
                     "model=urban_v2 scope=C12-only "
                     "other_species=legacy_highland max_step_mm="
                  << minibeam_water_primary_urban_max_step_mm << '\n';
#else
        std::cout << "[minibeam-water-secondary-c12-mcs] model=urban_v2 "
                     "(minibeam backend off; no transport effect)"
                  << '\n';
#endif
    }
    if (minibeam_water_secondary_c12_enable_unified_em) {
        std::cout << "[minibeam-water-secondary-c12-em] model=unified-em "
                     "scope=C12-only other_species=formal-path\n";
    }
    if (minibeam_water_secondary_c12_post_sample_loss_scale != 1.0F) {
        std::cout << "[minibeam-water-secondary-c12-post-sample-loss] scale="
                  << minibeam_water_secondary_c12_post_sample_loss_scale
                  << " diagnostic-only\n";
    }
    const auto emittance_correlation_y = static_cast<float>(config.emittance_correlation_y);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const double electron_short_range_mm=config.material_electron_short_range_mm;
    if(electron_short_range_mm>0)
        std::cout<<"[research-short-range] threshold_mm="<<electron_short_range_mm
                 <<" childless complete tail, strict same-voxel containment; not accuracy validated\n";
    const auto enable_csda_range_energy_loss = config.enable_csda_range_energy_loss;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
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

    runtime_setup.finish();
    RuntimeScope runtime_steps("transport_loop_including_scoring_and_queue_transfers");
    double primary_kernel_seconds = 0.0;
#if CARBON_PRIMARY_PRODUCTION_SPECIALIZE
    const bool primary_production_path_eligible =
        EmMode == 1 && enable_ct_grid && enable_voxel_scoring && enable_inelastic &&
        enable_multiple_scattering && enable_ct_material_mcs &&
        use_schneider_primary_xs && use_schneider_stopping &&
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

    std::vector<sycl::event> primary_events;
    primary_events.reserve((number_of_histories + history_chunk - 1) / history_chunk);
    for (std::size_t hist_offset = 0;
         !water_entry_secondary_replay && hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        const auto launch_primary_chunk = [&](auto production_path_tag) {
            constexpr bool kProductionPrimaryPath =
                decltype(production_path_tag)::value;
            return queue.parallel_for<CarbonPrimaryTransportKernel<
                EmMode, kProductionPrimaryPath>>(
            sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                const auto lane = item.get_global_linear_id();
                if (lane >= chunk_count) {
                    return;
                }
                const auto global_history = hist_offset + lane;
#if CARBON_PRIMARY_CONTEXT_POINTER
                // Shadow the host-side closures with device-resident views so the
                // rest of the kernel body is byte-for-byte identical.
#if CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
                const auto& unified_device = *primary_unified_device_ptr;
#endif
                const auto& schneider_ct_device_ctx = *primary_schneider_ct_ptr;
#endif

                const PrimarySpotBatchEntry* spot = nullptr;
                std::uint64_t rng_history = global_history;
                if (primary_spot_count > 0) {
                    std::size_t lower = 0;
                    std::size_t upper = primary_spot_count;
                    while (lower + 1 < upper) {
                        const auto middle = lower + (upper - lower) / 2;
                        if (global_history < primary_spots_device[middle].history_begin) {
                            upper = middle;
                        } else {
                            lower = middle;
                        }
                    }
                    spot = primary_spots_device + lower;
                    rng_history = global_history - spot->history_begin;
                }
                const auto spot_seed = spot != nullptr ? spot->random_seed : random_seed;
                const auto spot_initial_energy_MeV =
                    spot != nullptr ? spot->floats[0] : initial_energy_MeV;
                const auto spot_energy_spread =
                    spot != nullptr ? spot->floats[1] : beam_energy_spread;

                auto energy_MeV = spot_initial_energy_MeV;
                if (spot_energy_spread > 0.0F) {
                    const auto u0 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 40), 1.0e-12F);
                    const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 41);
                    constexpr float two_pi = 6.2831853071795864769F;
                    const auto gauss =
                        sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                    energy_MeV =
                        spot_initial_energy_MeV * (1.0F + spot_energy_spread * gauss);
                    if (energy_MeV < energy_cutoff_MeV) {
                        energy_MeV = energy_cutoff_MeV;
                    }
                }

                const auto sampled_initial_energy_MeVu = energy_MeV * inverse_mass_number;
                // Store the energy actually transported after source sampling/cutoff.
                // This does not consume RNG or change the source distribution.
                sampled_incident_device[global_history] = energy_MeV;
                auto local_x_mm = 0.0F;
                auto local_y_mm = 0.0F;
                auto local_dx = 0.0F;
                auto local_dy = 0.0F;
                auto local_dz = 1.0F;
                if (enable_flat_source) {
                    local_x_mm = flat_source_half_width_x_mm *
                                 (2.0F * rng::uniform01(spot_seed, rng_history, 0, 34) - 1.0F);
                    local_y_mm = flat_source_half_width_y_mm *
                                 (2.0F * rng::uniform01(spot_seed, rng_history, 0, 35) - 1.0F);
                } else if (enable_emittance_source) {
                    const auto u0 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 30), 1.0e-12F);
                    const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 31);
                    const auto u2 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 32), 1.0e-12F);
                    const auto u3 = rng::uniform01(spot_seed, rng_history, 0, 33);
                    constexpr float two_pi = 6.2831853071795864769F;
                    const auto g0 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                    const auto g1 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::sin(two_pi * u1);
                    const auto g2 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                    const auto g3 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::sin(two_pi * u3);
                    const auto sigma_x = spot != nullptr ? spot->floats[2] : emittance_sigma_x_mm;
                    const auto sigma_y = spot != nullptr ? spot->floats[3] : emittance_sigma_y_mm;
                    const auto sigma_x_prime =
                        spot != nullptr ? spot->floats[4] : emittance_sigma_x_prime;
                    const auto sigma_y_prime =
                        spot != nullptr ? spot->floats[5] : emittance_sigma_y_prime;
                    local_x_mm = sigma_x * g0;
                    local_y_mm = sigma_y * g2;
                    const auto rho_x = sycl::clamp(
                        spot != nullptr ? spot->floats[6] : emittance_correlation_x,
                        -0.9999F, 0.9999F);
                    const auto rho_y = sycl::clamp(
                        spot != nullptr ? spot->floats[7] : emittance_correlation_y,
                        -0.9999F, 0.9999F);
                    const auto x_prime =
                        sigma_x_prime * (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                    const auto y_prime =
                        sigma_y_prime * (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                    const auto inv_norm =
                        sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                    local_dx = x_prime * inv_norm;
                    local_dy = y_prime * inv_norm;
                    local_dz = inv_norm;
                }

                const auto origin_x = spot != nullptr ? spot->floats[8] : source_origin_x_mm;
                const auto origin_y = spot != nullptr ? spot->floats[9] : source_origin_y_mm;
                const auto origin_z = spot != nullptr ? spot->floats[10] : source_origin_z_mm;
                const auto ux_x = spot != nullptr ? spot->floats[11] : beam_ux_x;
                const auto ux_y = spot != nullptr ? spot->floats[12] : beam_ux_y;
                const auto ux_z = spot != nullptr ? spot->floats[13] : beam_ux_z;
                const auto uy_x = spot != nullptr ? spot->floats[14] : beam_uy_x;
                const auto uy_y = spot != nullptr ? spot->floats[15] : beam_uy_y;
                const auto uy_z = spot != nullptr ? spot->floats[16] : beam_uy_z;
                const auto uz_x = spot != nullptr ? spot->floats[17] : beam_uz_x;
                const auto uz_y = spot != nullptr ? spot->floats[18] : beam_uz_y;
                const auto uz_z = spot != nullptr ? spot->floats[19] : beam_uz_z;

                auto position_x_mm = origin_x + ux_x * local_x_mm + uy_x * local_y_mm;
                auto position_y_mm = origin_y + ux_y * local_x_mm + uy_y * local_y_mm;
                auto position_z_mm = origin_z + ux_z * local_x_mm + uy_z * local_y_mm;
                const auto composed_direction = compose_tps_direction(
                    local_dx, local_dy, local_dz, ux_x, ux_y, ux_z,
                    uy_x, uy_y, uy_z, uz_x, uz_y, uz_z);
                auto direction_x = composed_direction.x;
                auto direction_y = composed_direction.y;
                auto direction_z = composed_direction.z;
                {
                    const auto inv_n = sycl::rsqrt(sycl::fmax(
                        1.0e-20F, direction_x * direction_x + direction_y * direction_y +
                                      direction_z * direction_z));
                    direction_x *= inv_n;
                    direction_y *= inv_n;
                    direction_z *= inv_n;
                }

#if defined(CARBON_ENABLE_MINIBEAM)
                float beamline_air_loss_MeV = 0.0F;
                const auto minibeam_block_entrance_z =
                    minibeam_block_center_z - 0.5F * minibeam_block_thickness;
                const auto minibeam_block_exit_z =
                    minibeam_block_center_z + 0.5F * minibeam_block_thickness;
                if (minibeam_copper_transport && direction_z > 1.0e-8F) {
                    const auto entrance_air_distance = sycl::fmax(
                        0.0F, (minibeam_block_entrance_z - position_z_mm) /
                                  direction_z);
                    const auto entrance_air_loss = sycl::fmin(
                        energy_MeV,
                        minibeam_linear_table(
                            minibeam_air_sp_energies_device,
                            minibeam_air_sp_values_device,
                            minibeam_air_sp_count,
                            energy_MeV * inverse_mass_number) *
                            entrance_air_distance);
                    energy_MeV -= entrance_air_loss;
                    beamline_air_loss_MeV += entrance_air_loss;
                }
                const auto minibeam_hits_copper = enable_minibeam &&
                    minibeam_ray_hits_cylindrical_copper(
                        position_x_mm, position_y_mm, position_z_mm,
                        direction_x, direction_y, direction_z,
                        minibeam_cos, minibeam_sin, minibeam_block_radius,
                        minibeam_block_center_z,
                        minibeam_block_thickness, minibeam_slit_count,
                        minibeam_slit_width, minibeam_slit_pitch,
                        0.5F * minibeam_slit_length,
                        minibeam_slit_thickness, minibeam_slit_offset);
                auto minibeam_primary_elastic = false;
                auto minibeam_source_slit = 0;
                // Versioned touch state (history scope): flushed to the
                // phase-space record after transport.  Legacy copper_touched
                // keeps the initial-ray meaning; ever_in_copper and the Cu
                // true-path accumulator below are the actual-touch observables.
                auto beamline_ever_in_copper_hist = false;
                auto beamline_cu_true_path_hist_mm = 0.0F;
                // urban_v2 step diagnostics (research-only aggregates).
                std::uint32_t beamline_cu_steps_hist = 0;
                std::uint32_t beamline_disp_below_hist = 0;
                std::uint32_t beamline_disp_accept_hist = 0;
                std::uint32_t beamline_disp_reduce_hist = 0;
                std::uint32_t beamline_disp_cancel_hist = 0;
                std::uint32_t beamline_cth_one_hist = 0;
                std::uint32_t beamline_limit_user_hist = 0;
                std::uint32_t beamline_limit_msc_hist = 0;
                std::uint32_t beamline_limit_geom_hist = 0;
                std::uint32_t beamline_limit_range_hist = 0;
                float beamline_g_sum_hist_mm = 0.0F;
                float beamline_t_sum_hist_mm = 0.0F;
                float beamline_delta_sum_hist_mm = 0.0F;
                float beamline_raw2_sum_hist_mm2 = 0.0F;
                float beamline_acc2_sum_hist_mm2 = 0.0F;
                if (enable_minibeam && direction_z > 1.0e-8F) {
                    const auto slit_entrance_distance =
                        (minibeam_block_entrance_z - position_z_mm) /
                        direction_z;
                    const auto slit_entrance_x =
                        position_x_mm + slit_entrance_distance * direction_x;
                    const auto slit_entrance_y =
                        position_y_mm + slit_entrance_distance * direction_y;
                    const auto slit_u = minibeam_cos * slit_entrance_x +
                        minibeam_sin * slit_entrance_y - minibeam_slit_offset;
                    minibeam_source_slit = nearest_minibeam_slit(
                        slit_u, minibeam_slit_pitch);
                }
                if (minibeam_absorbing_geometry && minibeam_hits_copper) {
                    beamline_removed_device[global_history] = energy_MeV;
                    energy_MeV = 0.0F;
                }
                if (minibeam_copper_transport && minibeam_hits_copper &&
                    direction_z > 1.0e-8F) {
                    const auto incident_beamline_energy = energy_MeV;
                    const auto block_entrance_z = minibeam_block_entrance_z;
                    const auto block_exit_z = minibeam_block_exit_z;
                    const auto to_entrance =
                        (block_entrance_z - position_z_mm) / direction_z;
                    if (to_entrance >= 0.0F) {
                        position_x_mm += to_entrance * direction_x;
                        position_y_mm += to_entrance * direction_y;
                        position_z_mm = block_entrance_z;
                    }
                    float removed_energy = 0.0F;
                    std::uint64_t beamline_step = 0;
                    float copper_nuclear_tau_remaining = 0.0F;
                    bool copper_nuclear_tau_active = false;
                    // Urban v2 fMinimal state (Phase 2/5): persists across the
                    // steps of one Cu traversal; at_boundary marks steps that
                    // start at the block entrance or a material interface.
                    UrbanV2TrackState urban_v2_state{};
                    bool urban_v2_at_boundary = true;
                    bool urban_v2_prev_in_cu = false;
                    UrbanV2GeomCtx urban_v2_geom{};
                    urban_v2_geom.cos_a = minibeam_cos;
                    urban_v2_geom.sin_a = minibeam_sin;
                    urban_v2_geom.radius_mm = minibeam_block_radius;
                    urban_v2_geom.slit_count = minibeam_slit_count;
                    urban_v2_geom.slit_width_mm = minibeam_slit_width;
                    urban_v2_geom.slit_pitch_mm = minibeam_slit_pitch;
                    urban_v2_geom.slit_half_len_mm = 0.5F * minibeam_slit_length;
                    urban_v2_geom.slit_offset_mm = minibeam_slit_offset;
                    urban_v2_geom.block_entrance_z_mm = minibeam_block_entrance_z;
                    urban_v2_geom.block_exit_z_mm = minibeam_block_exit_z;
                    UrbanV2LossTable urban_v2_loss_table{
                        minibeam_copper_loss_e_device,
                        minibeam_copper_loss_r_device,
                        minibeam_copper_loss_d_device,
                        static_cast<int>(minibeam_copper_loss_count)};
                    // (touch accumulators live at history scope:
                    // beamline_ever_in_copper_hist / beamline_cu_true_path_hist_mm)
                    constexpr std::uint64_t maximum_beamline_steps = 100000;
                    while (energy_MeV > energy_cutoff_MeV &&
                           direction_z > 1.0e-8F &&
                           position_z_mm < block_exit_z - 1.0e-6F &&
                           beamline_step < maximum_beamline_steps) {
                        // Limit the transported path, not only its axial
                        // projection. This keeps the Poisson tail truncation
                        // and every material interaction bounded for oblique
                        // slit-edge tracks.
                        const auto nominal_path_step = sycl::fmin(
                            minibeam_copper_max_step,
                            (block_exit_z - position_z_mm) / direction_z);
                        const auto path_step = minibeam_copper_exact_material_boundaries
                            ? minibeam_path_to_material_boundary(
                                  position_x_mm, position_y_mm, direction_x,
                                  direction_y, minibeam_cos, minibeam_sin,
                                  minibeam_block_radius, minibeam_slit_count,
                                  minibeam_slit_width, minibeam_slit_pitch,
                                  0.5F * minibeam_slit_length,
                                  minibeam_slit_offset, nominal_path_step)
                            : nominal_path_step;
                        const auto axial_step = path_step * direction_z;
                        const auto midpoint_x =
                            position_x_mm + 0.5F * path_step * direction_x;
                        const auto midpoint_y =
                            position_y_mm + 0.5F * path_step * direction_y;
                        const auto in_copper = minibeam_point_in_copper(
                            midpoint_x, midpoint_y, minibeam_cos, minibeam_sin,
                            minibeam_block_radius,
                            minibeam_slit_count, minibeam_slit_width,
                            minibeam_slit_pitch, 0.5F * minibeam_slit_length,
                            minibeam_slit_offset);
                        if (!in_copper) {
                            position_x_mm += path_step * direction_x;
                            position_y_mm += path_step * direction_y;
                            position_z_mm += axial_step;
                            const auto air_loss = sycl::fmin(
                                energy_MeV,
                                minibeam_linear_table(
                                    minibeam_air_sp_energies_device,
                                    minibeam_air_sp_values_device,
                                    minibeam_air_sp_count,
                                    energy_MeV * inverse_mass_number) *
                                    path_step);
                            energy_MeV -= air_loss;
                            beamline_air_loss_MeV += air_loss;
                            ++beamline_step;
                            urban_v2_prev_in_cu = false;
                            continue;
                        }
                        const auto energy_u = energy_MeV * inverse_mass_number;
                        const auto stopping_scale = minibeam_survivor_stopping_scale(
                            energy_u, minibeam_copper_survivor_energy_loss_scale,
                            minibeam_survivor_calibration_energies.data(),
                            minibeam_survivor_calibration_scales.data(),
                            minibeam_survivor_calibration_count);
                        const auto stopping = stopping_scale * minibeam_linear_table(
                            minibeam_copper_sp_energies_device,
                            minibeam_copper_sp_values_device,
                            minibeam_copper_sp_count, energy_u);
                        const auto rate_elastic_index = minibeam_elastic_nearest(
                            minibeam_copper_elastic_device,
                            minibeam_copper_elastic_count, energy_u);
                        const auto rate_elastic = minibeam_copper_enable_elastic
                            ? minibeam_copper_elastic_device[
                                  rate_elastic_index].macroscopic_rate_per_mm
                            : 0.0F;
                        const auto rate_inelastic = minibeam_copper_enable_inelastic
                            ? minibeam_linear_table(
                                  minibeam_copper_xs_energies_device,
                                  minibeam_copper_xs_values_device,
                                  minibeam_copper_xs_count, energy_u)
                            : 0.0F;
                        const auto rate_total = rate_elastic + rate_inelastic;
                        if (rate_total > 0.0F && !copper_nuclear_tau_active) {
                            const auto optical_uniform = sycl::fmax(
                                1.0e-7F, rng::uniform01(
                                    spot_seed, rng_history, beamline_step, 60));
                            copper_nuclear_tau_remaining = -sycl::log(optical_uniform);
                            copper_nuclear_tau_active = true;
                        }
                        const auto collision_in_step = rate_total > 0.0F &&
                            copper_nuclear_tau_active &&
                            copper_nuclear_tau_remaining <= rate_total * path_step;
                        const auto transport_path = collision_in_step
                            ? copper_nuclear_tau_remaining / rate_total
                            : path_step;
                        const Direction3F pre_scatter_direction{
                            direction_x, direction_y, direction_z};
                        auto correlated_scattering = CorrelatedScatteringStep{
                            pre_scatter_direction, Direction3F{0.0F, 0.0F, 0.0F}};
                        if (minibeam_copper_enable_mcs &&
                            minibeam_copper_correlated_scattering &&
                            energy_MeV > energy_cutoff_MeV) {
                            if (minibeam_copper_urban_v2_msc) {
                                // Re-entry after air also starts at a
                                // geometry boundary (reference stepStatus).
                                urban_v2_at_boundary = urban_v2_at_boundary ||
                                    !urban_v2_prev_in_cu;
                                // Phase 5: external TRUE candidate (user
                                // ceiling) -> fMinimal limit -> true->geom ->
                                // geometry truncation -> final true -> scatter.
                                // boundary_path_mm is the geometry-reachable
                                // length (exact material boundaries honored).
                                // The legacy E/stopping range is never used;
                                // currentRange comes from the loss table.
                                correlated_scattering =
                                    copper_urban_v2_propose_and_sample(
                                        pre_scatter_direction, energy_MeV, 6, 12,
                                        minibeam_copper_max_step, transport_path,
                                        position_x_mm, position_y_mm,
                                        position_z_mm, direction_x, direction_y,
                                        direction_z, urban_v2_geom,
                                        urban_v2_at_boundary, urban_v2_state,
                                        urban_v2_loss_table,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length,
                                        minibeam_copper_mcs_scale, spot_seed,
                                        rng_history, beamline_step, 50);
                                urban_v2_at_boundary =
                                    correlated_scattering.boundary_crossed;
                                urban_v2_prev_in_cu = true;
                            } else if (minibeam_copper_urban_msc) {
                                const auto copper_range_mm = stopping > 0.0F
                                    ? energy_MeV / stopping
                                    : 0.0F;
                                correlated_scattering = copper_urban_msc_step(
                                    pre_scatter_direction, energy_MeV, 6, 12,
                                    transport_path, minibeam_copper_density,
                                    minibeam_copper_radiation_length,
                                    copper_range_mm, minibeam_copper_mcs_scale,
                                    spot_seed, rng_history, beamline_step, 50);
                            } else {
                                correlated_scattering = copper_fermi_eyges_tail_step(
                                    pre_scatter_direction, energy_MeV, 6, 12,
                                    transport_path, minibeam_copper_density,
                                    minibeam_copper_radiation_length,
                                    minibeam_copper_mcs_scale, spot_seed,
                                    rng_history, beamline_step, 50);
                            }
                        }
                        // R4: explicit step proposal/finalization contract for
                        // urban_v2 only (legacy/highland/FE/urban untouched).
                        // Geometry consumes final g; loss/fluctuation and the
                        // reaction path below consume final t.  No post-hoc
                        // t/g rescaling of the ledger.
                        const auto urban_v2_proposal =
                            minibeam_copper_urban_v2_msc &&
                            correlated_scattering.proposal_valid;
                        // Fix B5: an invalid Urban proposal must fail the
                        // run, not silently advance the full path
                        // unscattered (flag only; the host throws).
                        if (minibeam_copper_urban_v2_msc &&
                            !correlated_scattering.proposal_valid &&
                            minibeam_event_counts_device != nullptr) {
                            sycl::atomic_ref<
                                std::uint64_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                minibeam_event_counts_device
                                    [minibeam_water_urban_fatal_slot])
                                .fetch_add(1U);
                        }
                        const auto geom_advance_mm = urban_v2_proposal
                            ? correlated_scattering.final_geom_path_mm
                            : transport_path;
                        const auto true_loss_path_mm = urban_v2_proposal
                            ? correlated_scattering.final_true_path_mm
                            : transport_path;
                        position_x_mm += geom_advance_mm * direction_x +
                            correlated_scattering.displacement_mm.x;
                        position_y_mm += geom_advance_mm * direction_y +
                            correlated_scattering.displacement_mm.y;
                        position_z_mm += geom_advance_mm * direction_z +
                            correlated_scattering.displacement_mm.z;
                        const auto predictor_loss = stopping * true_loss_path_mm;
                        const auto midpoint_energy_u = sycl::fmax(
                            0.0F, (energy_MeV - 0.5F * predictor_loss) *
                                      inverse_mass_number);
                        const auto midpoint_stopping_scale =
                            minibeam_survivor_stopping_scale(
                                midpoint_energy_u,
                                minibeam_copper_survivor_energy_loss_scale,
                                minibeam_survivor_calibration_energies.data(),
                                minibeam_survivor_calibration_scales.data(),
                                minibeam_survivor_calibration_count);
                        const auto midpoint_stopping = midpoint_stopping_scale *
                            minibeam_linear_table(
                                minibeam_copper_sp_energies_device,
                                minibeam_copper_sp_values_device,
                                minibeam_copper_sp_count, midpoint_energy_u);
                        const auto mean_loss = midpoint_stopping * true_loss_path_mm;
                        auto proposed_loss = mean_loss;
                        if (minibeam_copper_enable_energy_straggling) {
                            constexpr float copper_z_over_a_rel_water =
                                (29.0F / 63.546F) / 0.55509F;
                            const auto effective_charge =
                                ion_effective_charge_device(6, midpoint_energy_u);
                            const auto variance =
                                condensed_total_loss_variance_MeV2_device(
                                    midpoint_energy_u, 12, effective_charge,
                                    true_loss_path_mm, minibeam_copper_density,
                                    copper_z_over_a_rel_water);
                            const auto gaussian_u0 = sycl::fmax(
                                rng::uniform01(spot_seed, rng_history,
                                               beamline_step, 48),
                                1.0e-12F);
                            const auto gaussian_u1 = rng::uniform01(
                                spot_seed, rng_history, beamline_step, 49);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gaussian =
                                sycl::sqrt(-2.0F * sycl::log(gaussian_u0)) *
                                sycl::cos(two_pi * gaussian_u1);
                            proposed_loss = sycl::fmax(
                                0.0F, mean_loss + minibeam_copper_straggling_scale *
                                    sycl::sqrt(sycl::fmax(0.0F, variance)) * gaussian);
                        }
                        if (proposed_loss >=
                            energy_MeV - energy_cutoff_MeV) {
                            removed_energy += energy_MeV;
                            energy_MeV = 0.0F;
                            break;
                        }
                        const auto loss = proposed_loss;
                        energy_MeV -= loss;
                        removed_energy += loss;
                        if (rate_total > 0.0F && copper_nuclear_tau_active) {
                            copper_nuclear_tau_remaining = sycl::fmax(
                                0.0F, copper_nuclear_tau_remaining -
                                           rate_total * true_loss_path_mm);
                        }
                        if (in_copper) {
                            beamline_ever_in_copper_hist = true;
                            beamline_cu_true_path_hist_mm += true_loss_path_mm;
                            if (minibeam_copper_urban_v2_msc) {
                                ++beamline_cu_steps_hist;
                                const auto& sc = correlated_scattering;
                                beamline_g_sum_hist_mm +=
                                    sc.proposal_valid ? sc.final_geom_path_mm
                                                      : transport_path;
                                beamline_t_sum_hist_mm +=
                                    sc.proposal_valid ? sc.final_true_path_mm
                                                      : transport_path;
                                beamline_delta_sum_hist_mm +=
                                    sc.proposal_valid ? sc.stable_delta_mm
                                                      : 0.0F;
                                beamline_raw2_sum_hist_mm2 +=
                                    sc.raw_displacement_r_mm *
                                    sc.raw_displacement_r_mm;
                                beamline_acc2_sum_hist_mm2 +=
                                    sc.displacement_mm.x *
                                        sc.displacement_mm.x +
                                    sc.displacement_mm.y *
                                        sc.displacement_mm.y +
                                    sc.displacement_mm.z *
                                        sc.displacement_mm.z;
                                switch (sc.displacement_branch) {
                                    case 1: ++beamline_disp_below_hist; break;
                                    case 2: ++beamline_disp_accept_hist; break;
                                    case 3: ++beamline_disp_reduce_hist; break;
                                    case 4: ++beamline_disp_cancel_hist; break;
                                    default: break;
                                }
                                if (sc.cth_rounded_to_one) {
                                    ++beamline_cth_one_hist;
                                }
                                switch (sc.limit_reason) {
                                    case 1: ++beamline_limit_user_hist; break;
                                    case 2: ++beamline_limit_msc_hist; break;
                                    case 3: ++beamline_limit_geom_hist; break;
                                    case 4: ++beamline_limit_range_hist; break;
                                    default: break;
                                }
                            }
                        }
                        const auto collision_energy_u = energy_MeV * inverse_mass_number;
                        const auto elastic_index = minibeam_elastic_nearest(
                            minibeam_copper_elastic_device,
                            minibeam_copper_elastic_count, collision_energy_u);
                        const auto elastic_rate = minibeam_copper_enable_elastic
                            ? minibeam_copper_elastic_device[
                                  elastic_index].macroscopic_rate_per_mm
                            : 0.0F;
                        const auto inelastic_rate = minibeam_copper_enable_inelastic
                            ? minibeam_linear_table(
                                  minibeam_copper_xs_energies_device,
                                  minibeam_copper_xs_values_device,
                                  minibeam_copper_xs_count, collision_energy_u)
                            : 0.0F;
                        const auto total_nuclear_rate = elastic_rate + inelastic_rate;
                        if (minibeam_copper_enable_mcs &&
                            !minibeam_copper_correlated_scattering &&
                            energy_MeV > energy_cutoff_MeV) {
                            const auto theta = minibeam_copper_mcs_scale *
                                highland_projected_rms_angle_device(
                                    energy_MeV, 6, 12, transport_path,
                                    minibeam_copper_density,
                                    minibeam_copper_radiation_length);
                            const auto scattered = scatter_direction(
                                Direction3F{direction_x, direction_y, direction_z},
                                theta, spot_seed, rng_history, beamline_step, 50);
                            direction_x = scattered.x;
                            direction_y = scattered.y;
                            direction_z = scattered.z;
                        } else if (minibeam_copper_enable_mcs &&
                                   minibeam_copper_correlated_scattering) {
                            direction_x = correlated_scattering.direction.x;
                            direction_y = correlated_scattering.direction.y;
                            direction_z = correlated_scattering.direction.z;
                        }
                        if (collision_in_step && total_nuclear_rate > 0.0F) {
                            copper_nuclear_tau_active = false;
                            const auto choose = rng::uniform01(
                                spot_seed, rng_history, beamline_step, 61) *
                                total_nuclear_rate;
                            if (choose < elastic_rate) {
                                minibeam_primary_elastic = true;
                                const auto sample_index = minibeam_elastic_local_sample(
                                    minibeam_copper_elastic_device,
                                    minibeam_copper_elastic_count, collision_energy_u,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 63));
                                const auto& sample = minibeam_copper_elastic_device[
                                    sample_index];
                                const auto outcome = elastic_two_body(
                                    energy_MeV, direction_x, direction_y, direction_z,
                                    1.0 - 2.0 * sample.transfer_fraction,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 62),
                                    11174.86323534, sample.target_mass_MeV);
                                removed_energy += outcome.recoil_ke_MeV;
                                energy_MeV = outcome.projectile_ke_MeV;
                                direction_x = outcome.proj_dir_x;
                                direction_y = outcome.proj_dir_y;
                                direction_z = outcome.proj_dir_z;
                            } else {
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    nuclear_count(minibeam_event_counts_device[0]);
                                nuclear_count.fetch_add(1U);
                                const auto lookup = cinel03_lookup_event_device(
                                    minibeam_copper_nodes_device,
                                    minibeam_copper_node_count,
                                    minibeam_copper_offsets_device,
                                    minibeam_copper_indices_device,
                                    minibeam_copper_event_count,
                                    6, 12, 29, collision_energy_u,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 63),
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 64));
                                float queued_energy = 0.0F;
                                if (lookup.status == Cinel03LookupStatus::Hit) {
                                    const auto& event = minibeam_copper_interactions_device[
                                        lookup.event_index];
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generated_count(minibeam_event_counts_device[1]);
                                    generated_count.fetch_add(event.direct_product_count);
                                    const auto phi = 6.2831853071795864769F *
                                        rng::uniform01(spot_seed, rng_history,
                                                       beamline_step, 65);
                                    const auto event_cos = sycl::cos(phi);
                                    const auto event_sin = sycl::sin(phi);
                                    for (std::uint32_t product_index = 0;
                                         product_index < event.direct_product_count;
                                         ++product_index) {
                                        const auto flat_index =
                                            event.product_offset + product_index;
                                        if (flat_index >= minibeam_copper_product_count) break;
                                        const auto& product =
                                            minibeam_copper_products_device[flat_index];
                                        if ((product.role != 0 && product.role != 1) ||
                                            product.z <= 0 || product.a <= 0 ||
                                            product.kinetic_energy_MeV <= energy_cutoff_MeV) {
                                            continue;
                                        }
                                        const auto local_direction =
                                            rotate_cinel03_event_azimuth(
                                                product.local_direction_x,
                                                product.local_direction_y,
                                                product.local_direction_z,
                                                event_cos, event_sin);
                                        auto child_direction = rotate_local_direction(
                                            local_direction.x, local_direction.y,
                                            local_direction.z,
                                            Direction3F{direction_x, direction_y,
                                                        direction_z});
                                        auto child_energy = product.kinetic_energy_MeV;
                                        auto child_x = position_x_mm;
                                        auto child_y = position_y_mm;
                                        auto child_z = position_z_mm;
                                        std::uint32_t child_step = 0;
                                        float child_copper_segment_path = 0.0F;
                                        float child_nuclear_tau_remaining = 0.0F;
                                        bool child_nuclear_tau_active = false;
                                        const auto child_species_category =
                                            product.z == 6 ? 0U :
                                            product.z == 5 ? 1U :
                                            product.z == 4 ? 2U :
                                            product.z == 3 ? 3U :
                                            product.z == 2 ? 4U :
                                            (product.z == 1 && product.a == 1) ? 5U :
                                            (product.z == 1 && product.a == 2) ? 6U :
                                            (product.z == 1 && product.a == 3) ? 7U : 8U;
                                        const auto child_rng_stream =
                                            rng::event_product_stream(
                                                rng_history, beamline_step,
                                                rng::branch_role_primary_charged,
                                                product_index);
                                        while (child_energy > energy_cutoff_MeV &&
                                               child_direction.z > 1.0e-8F &&
                                               child_z < block_exit_z - 1.0e-6F &&
                                               child_step < 100000U) {
                                            const auto nominal_child_path = sycl::fmin(
                                                minibeam_copper_max_step,
                                                (block_exit_z - child_z) /
                                                    child_direction.z);
                                            const auto child_path =
                                                minibeam_copper_exact_material_boundaries
                                                ? minibeam_path_to_material_boundary(
                                                      child_x, child_y,
                                                      child_direction.x,
                                                      child_direction.y,
                                                      minibeam_cos, minibeam_sin,
                                                      minibeam_block_radius,
                                                      minibeam_slit_count,
                                                      minibeam_slit_width,
                                                      minibeam_slit_pitch,
                                                      0.5F * minibeam_slit_length,
                                                      minibeam_slit_offset,
                                                      nominal_child_path)
                                                : nominal_child_path;
                                            const auto child_mid_x = child_x +
                                                0.5F * child_path * child_direction.x;
                                            const auto child_mid_y = child_y +
                                                0.5F * child_path * child_direction.y;
                                            const auto child_in_copper =
                                                minibeam_point_in_copper(
                                                    child_mid_x, child_mid_y,
                                                    minibeam_cos, minibeam_sin,
                                                    minibeam_block_radius,
                                                    minibeam_slit_count,
                                                    minibeam_slit_width,
                                                    minibeam_slit_pitch,
                                                    0.5F * minibeam_slit_length,
                                                    minibeam_slit_offset);
                                            auto child_inelastic_rate = 0.0F;
                                            auto child_collision_in_step = false;
                                            auto child_transport_path = child_path;
                                            if (child_in_copper) {
                                                child_inelastic_rate =
                                                    minibeam_copper_ion_inelastic_rate(
                                                        minibeam_copper_ion_xs_device,
                                                        minibeam_copper_ion_xs_present_device,
                                                        minibeam_copper_ion_xs_grid_size,
                                                        minibeam_copper_ion_xs_minimum_energy,
                                                        minibeam_copper_ion_xs_inverse_step,
                                                        child_energy,
                                                        product.z, product.a);
                                                if (child_inelastic_rate > 0.0F &&
                                                    !child_nuclear_tau_active) {
                                                    const auto optical_uniform =
                                                        sycl::fmax(
                                                            1.0e-7F,
                                                            rng::uniform01(
                                                                spot_seed,
                                                                child_rng_stream,
                                                                child_step, 2));
                                                    child_nuclear_tau_remaining =
                                                        -sycl::log(optical_uniform);
                                                    child_nuclear_tau_active = true;
                                                }
                                                child_collision_in_step =
                                                    child_inelastic_rate > 0.0F &&
                                                    child_nuclear_tau_active &&
                                                    child_nuclear_tau_remaining <=
                                                        child_inelastic_rate * child_path;
                                                if (child_collision_in_step) {
                                                    child_transport_path =
                                                        child_nuclear_tau_remaining /
                                                        child_inelastic_rate;
                                                }
                                            }
                                            child_x += child_transport_path *
                                                child_direction.x;
                                            child_y += child_transport_path *
                                                child_direction.y;
                                            child_z += child_transport_path *
                                                child_direction.z;
                                            if (child_in_copper) {
                                                const auto child_stopping =
                                                    minibeam_copper_ion_stopping(
                                                        minibeam_copper_sp_energies_device,
                                                        minibeam_copper_sp_values_device,
                                                        minibeam_copper_sp_count,
                                                        minibeam_copper_ion_sp_ratios_device,
                                                        minibeam_copper_ion_sp_present_device,
                                                        child_energy, product.z, product.a);
                                                const auto predictor_child_loss =
                                                    child_stopping * child_transport_path;
                                                const auto child_midpoint_energy =
                                                    sycl::fmax(
                                                        0.0F,
                                                        child_energy - 0.5F *
                                                            predictor_child_loss);
                                                const auto child_midpoint_stopping =
                                                    minibeam_copper_ion_stopping(
                                                        minibeam_copper_sp_energies_device,
                                                        minibeam_copper_sp_values_device,
                                                        minibeam_copper_sp_count,
                                                        minibeam_copper_ion_sp_ratios_device,
                                                        minibeam_copper_ion_sp_present_device,
                                                        child_midpoint_energy,
                                                        product.z, product.a);
                                                const auto mean_child_loss =
                                                    child_midpoint_stopping *
                                                    child_transport_path;
                                                auto proposed_child_loss = mean_child_loss;
                                                if (minibeam_copper_fragment_enable_energy_straggling) {
                                                    constexpr float copper_z_over_a_rel_water =
                                                        (29.0F / 63.546F) / 0.55509F;
                                                    const auto child_energy_u =
                                                        child_midpoint_energy /
                                                        static_cast<float>(product.a);
                                                    const auto effective_charge =
                                                        ion_effective_charge_device(
                                                            product.z, child_energy_u);
                                                    const auto variance =
                                                        condensed_total_loss_variance_MeV2_device(
                                                            child_energy_u, product.a,
                                                            effective_charge,
                                                            child_transport_path,
                                                            minibeam_copper_density,
                                                            copper_z_over_a_rel_water);
                                                    const auto gaussian_u0 = sycl::fmax(
                                                        rng::uniform01(
                                                            spot_seed, child_rng_stream,
                                                            child_step, 10),
                                                        1.0e-12F);
                                                    const auto gaussian_u1 = rng::uniform01(
                                                        spot_seed, child_rng_stream,
                                                        child_step, 11);
                                                    constexpr float two_pi =
                                                        6.2831853071795864769F;
                                                    const auto gaussian = sycl::sqrt(
                                                        -2.0F * sycl::log(gaussian_u0)) *
                                                        sycl::cos(two_pi * gaussian_u1);
                                                    proposed_child_loss = sycl::fmax(
                                                        0.0F,
                                                        mean_child_loss +
                                                            minibeam_copper_fragment_straggling_scale *
                                                                sycl::sqrt(sycl::fmax(
                                                                    0.0F, variance)) *
                                                                gaussian);
                                                }
                                                if (proposed_child_loss >=
                                                    child_energy -
                                                        energy_cutoff_MeV) {
                                                    child_energy = 0.0F;
                                                    break;
                                                }
                                                const auto child_loss =
                                                    proposed_child_loss;
                                                const auto scatter_energy =
                                                    child_energy - 0.5F * child_loss;
                                                child_energy -= child_loss;
                                                if (!child_collision_in_step &&
                                                    child_inelastic_rate > 0.0F &&
                                                    child_nuclear_tau_active) {
                                                    child_nuclear_tau_remaining -=
                                                        child_inelastic_rate *
                                                        child_transport_path;
                                                }
                                                if (minibeam_copper_enable_mcs) {
                                                    const auto previous_path =
                                                        child_copper_segment_path;
                                                    child_copper_segment_path +=
                                                        child_transport_path;
                                                    const auto total_rms =
                                                        highland_projected_rms_angle_device(
                                                            scatter_energy, product.z,
                                                            product.a,
                                                            child_copper_segment_path,
                                                            minibeam_copper_density,
                                                            minibeam_copper_radiation_length);
                                                    const auto previous_rms =
                                                        highland_projected_rms_angle_device(
                                                            scatter_energy, product.z,
                                                            product.a, previous_path,
                                                            minibeam_copper_density,
                                                            minibeam_copper_radiation_length);
                                                    child_direction = scatter_direction(
                                                        child_direction,
                                                        minibeam_copper_fragment_mcs_scale *
                                                            sycl::sqrt(sycl::fmax(
                                                                0.0F,
                                                                total_rms * total_rms -
                                                                    previous_rms *
                                                                        previous_rms)),
                                                        spot_seed, child_rng_stream,
                                                        child_step, 0);
                                                }
                                                if (child_collision_in_step) {
                                                    child_nuclear_tau_active = false;
                                                    if (minibeam_copper_cascade_queue_device != nullptr &&
                                                        product.a > 0) {
                                                        sycl::atomic_ref<
                                                            std::uint64_t,
                                                            sycl::memory_order::relaxed,
                                                            sycl::memory_scope::device,
                                                            sycl::access::address_space::global_space>
                                                            reaction_count(
                                                                minibeam_event_counts_device[
                                                                    minibeam_fragment_cascade_interactions_slot]);
                                                        reaction_count.fetch_add(1U);
                                                        sycl::atomic_ref<
                                                            std::uint64_t,
                                                            sycl::memory_order::relaxed,
                                                            sycl::memory_scope::device,
                                                            sycl::access::address_space::global_space>
                                                            generation_reaction_count(
                                                                minibeam_event_counts_device[
                                                                    minibeam_fragment_generation_interaction_slot]);
                                                        generation_reaction_count.fetch_add(1U);
                                                        const auto fragment_lookup =
                                                            cinel03_lookup_event_device(
                                                                minibeam_copper_nodes_device,
                                                                minibeam_copper_node_count,
                                                                minibeam_copper_offsets_device,
                                                                minibeam_copper_indices_device,
                                                                minibeam_copper_event_count,
                                                                product.z, product.a, 29,
                                                                child_energy /
                                                                    static_cast<float>(product.a),
                                                                rng::uniform01(
                                                                    spot_seed, child_rng_stream,
                                                                    child_step, 20),
                                                                rng::uniform01(
                                                                    spot_seed, child_rng_stream,
                                                                    child_step, 21));
                                                        if (fragment_lookup.status ==
                                                            Cinel03LookupStatus::Hit) {
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                hit_count(
                                                                    minibeam_event_counts_device[
                                                                        minibeam_fragment_cascade_hits_slot]);
                                                            hit_count.fetch_add(1U);
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                generation_hit_count(
                                                                    minibeam_event_counts_device[
                                                                        minibeam_fragment_generation_hit_slot]);
                                                            generation_hit_count.fetch_add(1U);
                                                            const auto& fragment_event =
                                                                minibeam_copper_interactions_device[
                                                                    fragment_lookup.event_index];
                                                            const auto local_keV =
                                                                static_cast<std::uint64_t>(
                                                                    sycl::fmax(
                                                                        0.0F,
                                                                        fragment_event
                                                                            .process_local_deposit_MeV) *
                                                                        1000.0F + 0.5F);
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                local_energy(
                                                                    minibeam_event_counts_device[
                                                                        minibeam_fragment_cascade_local_keV_slot]);
                                                            local_energy.fetch_add(local_keV);
                                                            const auto fragment_phi =
                                                                6.2831853071795864769F *
                                                                rng::uniform01(
                                                                    spot_seed, child_rng_stream,
                                                                    child_step, 22);
                                                            const auto fragment_cos =
                                                                sycl::cos(fragment_phi);
                                                            const auto fragment_sin =
                                                                sycl::sin(fragment_phi);
                                                            auto accounted_charged = 0.0F;
                                                            auto replay_product_energy = 0.0F;
                                                            auto replay_product_rest_mass = 0.0F;
                                                            std::int32_t replay_product_a = 0;
                                                            for (std::uint32_t cascade_index = 0;
                                                                 cascade_index <
                                                                     fragment_event.direct_product_count;
                                                                 ++cascade_index) {
                                                                const auto cascade_flat =
                                                                    fragment_event.product_offset +
                                                                    cascade_index;
                                                                if (cascade_flat >=
                                                                    minibeam_copper_product_count) break;
                                                                const auto& cascade_product =
                                                                    minibeam_copper_products_device[
                                                                        cascade_flat];
                                                                replay_product_energy += sycl::fmax(
                                                                    0.0F,
                                                                    cascade_product.kinetic_energy_MeV);
                                                                replay_product_rest_mass += sycl::fmax(
                                                                    0.0F,
                                                                    cascade_product.rest_mass);
                                                                replay_product_a += sycl::max(
                                                                    0, static_cast<int>(cascade_product.a));
                                                                const bool supported_charged =
                                                                    (cascade_product.role == 0 ||
                                                                     cascade_product.role == 1) &&
                                                                    cascade_product.z > 0 &&
                                                                    cascade_product.a > 0;
                                                                const auto count_slot =
                                                                    supported_charged
                                                                        ? minibeam_fragment_cascade_charged_slot
                                                                        : (cascade_product.z <= 0
                                                                               ? minibeam_fragment_cascade_neutral_slot
                                                                               : minibeam_fragment_cascade_unsupported_slot);
                                                                sycl::atomic_ref<
                                                                    std::uint64_t,
                                                                    sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::device,
                                                                    sycl::access::address_space::global_space>
                                                                    product_count_ref(
                                                                        minibeam_event_counts_device[
                                                                            count_slot]);
                                                                product_count_ref.fetch_add(1U);
                                                                if (!supported_charged ||
                                                                    cascade_product.kinetic_energy_MeV <=
                                                                        energy_cutoff_MeV) {
                                                                    continue;
                                                                }
                                                                accounted_charged +=
                                                                    cascade_product.kinetic_energy_MeV;
                                                                const auto cascade_local =
                                                                    rotate_cinel03_event_azimuth(
                                                                        cascade_product.local_direction_x,
                                                                        cascade_product.local_direction_y,
                                                                        cascade_product.local_direction_z,
                                                                        fragment_cos, fragment_sin);
                                                                const auto cascade_direction =
                                                                    rotate_local_direction(
                                                                        cascade_local.x,
                                                                        cascade_local.y,
                                                                        cascade_local.z,
                                                                        child_direction);
                                                                sycl::atomic_ref<
                                                                    std::uint32_t,
                                                                    sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::device,
                                                                    sycl::access::address_space::global_space>
                                                                    cascade_queue_count(
                                                                        *minibeam_copper_cascade_count_device);
                                                                const auto cascade_output =
                                                                    cascade_queue_count.fetch_add(1U);
                                                                if (cascade_output < max_secondaries) {
                                                                    SecondaryParticle cascade_child{};
                                                                    cascade_child.z = cascade_product.z;
                                                                    cascade_child.a = cascade_product.a;
                                                                    cascade_child.energy_MeV =
                                                                        cascade_product.kinetic_energy_MeV;
                                                                    cascade_child.pos_x_mm = child_x;
                                                                    cascade_child.pos_y_mm = child_y;
                                                                    cascade_child.pos_z_mm = child_z;
                                                                    cascade_child.dir_x =
                                                                        cascade_direction.x;
                                                                    cascade_child.dir_y =
                                                                        cascade_direction.y;
                                                                    cascade_child.dir_z =
                                                                        cascade_direction.z;
                                                                    cascade_child.weight = 1.0F;
                                                                    cascade_child.parent_history =
                                                                        global_history;
                                                                    cascade_child.rng_stream =
                                                                        rng::event_product_stream(
                                                                            child_rng_stream,
                                                                            child_step,
                                                                            rng::branch_role_cascade_charged,
                                                                            cascade_index);
                                                                    // Copper generation is carried by
                                                                    // this dedicated queue, not by the
                                                                    // downstream water-generation field.
                                                                    cascade_child.generation = 1U;
                                                                    cascade_child.birth_region =
                                                                        minibeam_birth_region_copper;
                                                                    minibeam_copper_cascade_queue_device[
                                                                        cascade_output] = cascade_child;
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
                                                            const auto untracked_keV =
                                                                static_cast<std::uint64_t>(
                                                                    sycl::fmax(
                                                                        0.0F,
                                                                        child_energy -
                                                                            fragment_event
                                                                                .process_local_deposit_MeV -
                                                                            accounted_charged) *
                                                                        1000.0F + 0.5F);
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                untracked_energy(
                                                                    minibeam_event_counts_device[
                                                                        minibeam_fragment_cascade_untracked_keV_slot]);
                                                            untracked_energy.fetch_add(
                                                                untracked_keV);
                                                            const auto selected_input =
                                                                fragment_lookup.selected_energy_MeV_per_u *
                                                                static_cast<float>(product.a);
                                                            const auto replay_output =
                                                                sycl::fmax(0.0F, fragment_event.parent_energy_MeV) +
                                                                sycl::fmax(
                                                                    0.0F,
                                                                    fragment_event.process_local_deposit_MeV) +
                                                                replay_product_energy;
                                                            const auto add_energy_counter =
                                                                [&](const std::size_t slot,
                                                                    const float value) {
                                                                    sycl::atomic_ref<
                                                                        std::uint64_t,
                                                                        sycl::memory_order::relaxed,
                                                                        sycl::memory_scope::device,
                                                                        sycl::access::address_space::global_space>
                                                                        counter(
                                                                            minibeam_event_counts_device[
                                                                                slot]);
                                                                    counter.fetch_add(
                                                                        static_cast<std::uint64_t>(
                                                                            sycl::fmax(0.0F, value) *
                                                                                1000.0F + 0.5F));
                                                                };
                                                            add_energy_counter(
                                                                minibeam_fragment_actual_input_keV_slot,
                                                                child_energy);
                                                            add_energy_counter(
                                                                minibeam_fragment_selected_input_keV_slot,
                                                                selected_input);
                                                            add_energy_counter(
                                                                minibeam_fragment_replay_output_keV_slot,
                                                                replay_output);
                                                            add_energy_counter(
                                                                minibeam_fragment_selection_mismatch_keV_slot,
                                                                sycl::fabs(child_energy - selected_input));
                                                            add_energy_counter(
                                                                minibeam_fragment_closure_mismatch_keV_slot,
                                                                sycl::fabs(selected_input - replay_output));
                                                            const bool parent_survives =
                                                                fragment_event.parent_energy_MeV > 0.0F;
                                                            const auto copper_target_mass =
                                                                fragment_event.target_a == 63
                                                                    ? 58603.7301743F
                                                                    : (fragment_event.target_a == 65
                                                                           ? 60465.0342192F
                                                                           : sycl::fmax(
                                                                                 0.0F,
                                                                                 static_cast<float>(
                                                                                     fragment_event.target_a) *
                                                                                         931.49410242F -
                                                                                     29.0F * 0.51099895F));
                                                            const auto mass_energy_in =
                                                                selected_input +
                                                                fragment_event.parent_rest_mass +
                                                                copper_target_mass;
                                                            const auto mass_energy_out =
                                                                replay_output + replay_product_rest_mass +
                                                                (parent_survives
                                                                     ? fragment_event.parent_rest_mass
                                                                     : 0.0F);
                                                            add_energy_counter(
                                                                minibeam_fragment_mass_energy_mismatch_keV_slot,
                                                                sycl::fabs(mass_energy_in -
                                                                           mass_energy_out));
                                                            const auto input_a =
                                                                static_cast<int>(product.a) +
                                                                static_cast<int>(fragment_event.target_a);
                                                            const auto output_a =
                                                                replay_product_a +
                                                                (parent_survives
                                                                     ? static_cast<int>(
                                                                           fragment_event.parent_a)
                                                                     : 0);
                                                            if (input_a != output_a) {
                                                                sycl::atomic_ref<
                                                                    std::uint64_t,
                                                                    sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::device,
                                                                    sycl::access::address_space::global_space>
                                                                    baryon_miss(
                                                                        minibeam_event_counts_device[
                                                                            minibeam_fragment_baryon_mismatch_slot]);
                                                                baryon_miss.fetch_add(1U);
                                                            }
                                                        } else {
                                                            const auto miss_index =
                                                                static_cast<std::size_t>(
                                                                    fragment_lookup.status) - 1U;
                                                            if (miss_index < 6U) {
                                                                sycl::atomic_ref<
                                                                    std::uint64_t,
                                                                    sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::device,
                                                                    sycl::access::address_space::global_space>
                                                                    miss_count(
                                                                        minibeam_event_counts_device[
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
                                                                        minibeam_fragment_generation_miss_slot]);
                                                            generation_miss_count.fetch_add(1U);
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                species_miss(
                                                                    minibeam_event_counts_device[
                                                                        minibeam_fragment_miss_species_slot +
                                                                        child_species_category]);
                                                            species_miss.fetch_add(1U);
                                                            const auto miss_energy_u =
                                                                child_energy /
                                                                static_cast<float>(product.a);
                                                            const auto miss_energy_bin =
                                                                sycl::min(
                                                                    15U,
                                                                    static_cast<std::uint32_t>(
                                                                        sycl::fmax(0.0F,
                                                                                   miss_energy_u) /
                                                                        25.0F));
                                                            sycl::atomic_ref<
                                                                std::uint64_t,
                                                                sycl::memory_order::relaxed,
                                                                sycl::memory_scope::device,
                                                                sycl::access::address_space::global_space>
                                                                energy_miss(
                                                                    minibeam_event_counts_device[
                                                                    minibeam_fragment_miss_energy_slot +
                                                                    miss_energy_bin]);
                                                            energy_miss.fetch_add(1U);
                                                            if (minibeam_fragment_miss_joint_counts_device != nullptr &&
                                                                miss_index < MinibeamDiagnostics::fragment_miss_reason_count) {
                                                                const auto za_index =
                                                                    product.z >= 0 && product.z <= 6 &&
                                                                            product.a >= 0 && product.a <= 12
                                                                        ? static_cast<std::size_t>(product.z) * 13U +
                                                                              static_cast<std::size_t>(product.a)
                                                                        : MinibeamDiagnostics::fragment_miss_za_count - 1U;
                                                                const auto joint_energy_bin = sycl::min(
                                                                    127U,
                                                                    static_cast<std::uint32_t>(
                                                                        sycl::fmax(0.0F, miss_energy_u) / 5.0F));
                                                                const auto joint_index =
                                                                    ((za_index *
                                                                          MinibeamDiagnostics::fragment_miss_reason_count +
                                                                      miss_index) *
                                                                         MinibeamDiagnostics::fragment_miss_joint_energy_bin_count) +
                                                                    joint_energy_bin;
                                                                const auto block_entry_z =
                                                                    minibeam_block_center_z -
                                                                    0.5F * minibeam_block_thickness;
                                                                const auto collision_depth_um =
                                                                    static_cast<std::uint64_t>(sycl::fmax(
                                                                        0.0F, child_z - block_entry_z) * 1000.0F + 0.5F);
                                                                const auto path_to_plane = child_direction.z > 1.0e-8F
                                                                    ? sycl::fmax(0.0F,
                                                                          (block_exit_z - child_z) /
                                                                              child_direction.z)
                                                                    : 0.0F;
                                                                const auto remaining_um =
                                                                    static_cast<std::uint64_t>(
                                                                        minibeam_straight_copper_path_to_plane(
                                                                            child_x, child_y,
                                                                            child_direction.x,
                                                                            child_direction.y,
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
                                                                        sycl::fmax(0.0F, child_energy) *
                                                                            1000.0F +
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
                                                    }
                                                    if (minibeam_fragment_phase_space_device != nullptr) {
                                                    sycl::atomic_ref<
                                                        std::uint64_t,
                                                        sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        absorbed_count(
                                                            minibeam_event_counts_device[
                                                                21U + child_species_category]);
                                                    absorbed_count.fetch_add(1U);
                                                    const auto absorbed_energy_keV =
                                                        static_cast<std::uint64_t>(
                                                            sycl::fmax(0.0F, child_energy) *
                                                                1000.0F + 0.5F);
                                                    sycl::atomic_ref<
                                                        std::uint64_t,
                                                        sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        absorbed_energy(
                                                            minibeam_event_counts_device[
                                                                30U + child_species_category]);
                                                    absorbed_energy.fetch_add(
                                                        absorbed_energy_keV);
                                                    const auto path_to_downstream_plane =
                                                        sycl::fmax(
                                                            0.0F,
                                                            (block_exit_z - child_z) /
                                                                child_direction.z);
                                                    const auto straight_copper_path_um =
                                                        static_cast<std::uint64_t>(
                                                            minibeam_straight_copper_path_to_plane(
                                                                child_x, child_y,
                                                                child_direction.x,
                                                                child_direction.y,
                                                                minibeam_cos, minibeam_sin,
                                                                minibeam_block_radius,
                                                                minibeam_slit_count,
                                                                minibeam_slit_width,
                                                                minibeam_slit_pitch,
                                                                0.5F * minibeam_slit_length,
                                                                minibeam_slit_offset,
                                                                path_to_downstream_plane) *
                                                                1000.0F + 0.5F);
                                                    sycl::atomic_ref<
                                                        std::uint64_t,
                                                        sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        straight_copper_path(
                                                            minibeam_event_counts_device[
                                                                39U + child_species_category]);
                                                    straight_copper_path.fetch_add(
                                                        straight_copper_path_um);
                                                    }
                                                    child_energy = 0.0F;
                                                    break;
                                                }
                                            } else {
                                                child_copper_segment_path = 0.0F;
                                            }
                                            ++child_step;
                                        }
                                        if (child_energy <= energy_cutoff_MeV ||
                                            child_direction.z <= 1.0e-8F ||
                                            child_step == 100000U) continue;
                                        const auto to_water = -child_z / child_direction.z;
                                        if (to_water < 0.0F) continue;
                                        child_x += to_water * child_direction.x;
                                        child_y += to_water * child_direction.y;
                                        if (secondary_queue_device != nullptr) {
                                            sycl::atomic_ref<
                                                std::uint32_t,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                count_ref(*secondary_count_device);
                                            const auto output = count_ref.fetch_add(1U);
                                            if (output < max_secondaries) {
                                                SecondaryParticle child{};
                                                child.z = product.z;
                                                child.a = product.a;
                                                child.energy_MeV = child_energy;
                                                child.pos_x_mm = child_x;
                                                child.pos_y_mm = child_y;
                                                child.pos_z_mm = 0.0F;
                                                child.dir_x = child_direction.x;
                                                child.dir_y = child_direction.y;
                                                child.dir_z = child_direction.z;
                                                child.weight = 1.0F;
                                                child.parent_history = global_history;
                                                child.rng_stream = rng::event_product_stream(
                                                    rng_history, beamline_step,
                                                    rng::branch_role_primary_charged,
                                                    product_index);
                                                child.birth_region =
                                                    minibeam_birth_region_copper;
                                                secondary_queue_device[output] = child;
#if defined(CARBON_ENABLE_MINIBEAM)
                                                if (minibeam_fragment_phase_space_device != nullptr) {
                                                    sycl::atomic_ref<
                                                        std::uint32_t,
                                                        sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        phase_count(
                                                            *minibeam_fragment_phase_space_count_device);
                                                    const auto phase_index =
                                                        phase_count.fetch_add(1U);
                                                    if (phase_index < max_secondaries) {
                                                        MinibeamFragmentPhaseSpaceRecord record{};
                                                        record.history = global_history;
                                                        record.atomic_number = product.z;
                                                        record.mass_number = product.a;
                                                        record.kinetic_energy_MeV = child_energy;
                                                        record.x_mm = child_x;
                                                        record.y_mm = child_y;
                                                        record.direction_x = child_direction.x;
                                                        record.direction_y = child_direction.y;
                                                        record.direction_z = child_direction.z;
                                                        minibeam_fragment_phase_space_device[
                                                            phase_index] = record;
                                                    }
                                                }
#endif
                                                queued_energy += child_energy;
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    survivor_count(
                                                        minibeam_event_counts_device[2]);
                                                survivor_count.fetch_add(1U);
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    species_count(
                                                        minibeam_event_counts_device[
                                                            3U + child_species_category]);
                                                species_count.fetch_add(1U);
                                                const auto energy_keV =
                                                    static_cast<std::uint64_t>(
                                                        sycl::fmax(0.0F, child_energy) *
                                                            1000.0F + 0.5F);
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    species_energy(
                                                        minibeam_event_counts_device[
                                                            12U + child_species_category]);
                                                species_energy.fetch_add(energy_keV);
                                            } else {
                                                sycl::atomic_ref<
                                                    std::uint32_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    overflow(*secondary_overflow_count_device);
                                                overflow.fetch_add(1U);
                                            }
                                        }
                                    }
                                }
                                removed_energy = sycl::fmax(
                                    removed_energy,
                                    incident_beamline_energy - queued_energy);
                                energy_MeV = 0.0F;
                                break;
                            }
                        }
                        ++beamline_step;
                    }
                    if (energy_MeV > 0.0F) {
                        removed_energy = sycl::fmax(
                            0.0F, incident_beamline_energy - energy_MeV);
                    }
                    beamline_removed_device[global_history] =
                        sycl::fmax(0.0F, removed_energy);
                }
                if (minibeam_copper_transport &&
                    energy_MeV > energy_cutoff_MeV &&
                    direction_z > 1.0e-8F) {
                    auto remaining_air_distance =
                        -minibeam_block_exit_z / direction_z;
                    if (!minibeam_hits_copper) {
                        remaining_air_distance +=
                            minibeam_block_thickness / direction_z;
                    }
                    std::uint32_t air_step = 0;
                    while (remaining_air_distance > 0.0F &&
                           energy_MeV > energy_cutoff_MeV &&
                           air_step < 1024U) {
                        const auto step_mm =
                            sycl::fmin(1.0F, remaining_air_distance);
                        const auto air_loss = sycl::fmin(
                            energy_MeV,
                            minibeam_linear_table(
                                minibeam_air_sp_energies_device,
                                minibeam_air_sp_values_device,
                                minibeam_air_sp_count,
                                energy_MeV * inverse_mass_number) *
                                step_mm);
                        energy_MeV -= air_loss;
                        beamline_air_loss_MeV += air_loss;
                        remaining_air_distance -= step_mm;
                        ++air_step;
                    }
                    if (energy_MeV <= energy_cutoff_MeV) {
                        energy_MeV = 0.0F;
                    }
                }
                if (enable_minibeam) {
                    beamline_air_loss_device[global_history] =
                        beamline_air_loss_MeV;
                }
                if (enable_minibeam && energy_MeV > energy_cutoff_MeV) {
                    beamline_primary_survivor_device[global_history] = energy_MeV;
                }
#endif

                if (enable_tps_source) {
                    auto t_enter = 0.0F;
                    auto t_exit = 1.0e30F;
                    auto hit = true;
                    auto intersect_slab = [&](const float position, const float direction,
                                              const float lower, const float upper) {
                        if (sycl::fabs(direction) < 1.0e-8F) {
                            if (position < lower || position >= upper) {
                                hit = false;
                            }
                            return;
                        }
                        auto first = (lower - position) / direction;
                        auto second = (upper - position) / direction;
                        if (first > second) {
                            const auto temporary = first;
                            first = second;
                            second = temporary;
                        }
                        t_enter = sycl::fmax(t_enter, first);
                        t_exit = sycl::fmin(t_exit, second);
                        if (t_exit < t_enter) {
                            hit = false;
                        }
                    };
                    if ((kProductionPrimaryPath || enable_voxel_scoring)) {
                        intersect_slab(position_x_mm, direction_x, voxel_min_x_mm, voxel_max_x_mm);
                        intersect_slab(position_y_mm, direction_y, voxel_min_y_mm, voxel_max_y_mm);
                    }
                    intersect_slab(position_z_mm, direction_z, 0.0F, phantom_length_mm);
                    if (hit && t_exit >= t_enter) {
                        const auto entry = t_enter + 1.0e-4F;
                        position_x_mm += entry * direction_x;
                        position_y_mm += entry * direction_y;
                        position_z_mm += entry * direction_z;
                    }
                } else {
                    if (position_z_mm < 0.0F && sycl::fabs(direction_z) > 1.0e-8F) {
                        const auto t_plane = -position_z_mm / direction_z;
                        position_x_mm += t_plane * direction_x;
                        position_y_mm += t_plane * direction_y;
                        position_z_mm = 0.0F;
                    }
                }

#if defined(CARBON_ENABLE_MINIBEAM)
                if (minibeam_phase_space_device != nullptr &&
                    energy_MeV > energy_cutoff_MeV &&
                    position_z_mm >= 0.0F &&
                    position_z_mm < phantom_length_mm) {
                    MinibeamPhaseSpaceRecord record{};
                    record.history = global_history;
                    record.slit = minibeam_source_slit;
                    record.copper_touched = minibeam_hits_copper ? 1U : 0U;
                    record.copper_elastic = minibeam_primary_elastic ? 1U : 0U;
                    record.valid = 1U;
                    // Versioned touch: legacy field frozen as initial-ray;
                    // actual-touch observables below.
                    record.initial_ray_hits_copper =
                        minibeam_hits_copper ? 1U : 0U;
                    record.ever_in_copper =
                        beamline_ever_in_copper_hist ? 1U : 0U;
                    record.cumulative_cu_true_path_mm =
                        beamline_cu_true_path_hist_mm;
                    record.cu_steps = beamline_cu_steps_hist;
                    record.disp_below_min = beamline_disp_below_hist;
                    record.disp_accept = beamline_disp_accept_hist;
                    record.disp_reduce = beamline_disp_reduce_hist;
                    record.disp_cancel = beamline_disp_cancel_hist;
                    record.cth_eq_one = beamline_cth_one_hist;
                    record.cu_g_sum_mm = beamline_g_sum_hist_mm;
                    record.cu_t_sum_mm = beamline_t_sum_hist_mm;
                    record.cu_delta_sum_mm = beamline_delta_sum_hist_mm;
                    record.cu_raw_disp_sum2_mm2 = beamline_raw2_sum_hist_mm2;
                    record.cu_acc_disp_sum2_mm2 = beamline_acc2_sum_hist_mm2;
                    record.limit_user = beamline_limit_user_hist;
                    record.limit_msc = beamline_limit_msc_hist;
                    record.limit_geom = beamline_limit_geom_hist;
                    record.limit_range = beamline_limit_range_hist;
                    record.kinetic_energy_MeV = energy_MeV;
                    record.x_mm = position_x_mm;
                    record.y_mm = position_y_mm;
                    record.direction_x = direction_x;
                    record.direction_y = direction_y;
                    record.direction_z = direction_z;
                    minibeam_phase_space_device[global_history] = record;
                }
#endif

                auto history_deposited_MeV = 0.0F;
                auto history_water_electron_escaped_MeV=0.0F;
                std::uint64_t local_schneider_rate_queries = 0;
                std::uint32_t steps = 0;
                // Research-only water Urban state (one fMinimal track per
                // primary history; first water segment starts at a boundary).
                UrbanV2TrackState water_urban_state{};
                bool water_urban_seen_segment = false;
                std::uint32_t water_urban_seg_hist = 0;
                StepStableStragglingState<float> stable_straggling;
                stable_straggling.initialize(straggling_sampling_length_mm);

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
                const int unified_primary_species=unified_em?unified_device.species_index(primary_atomic_number,primary_mass_number):-1;

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
                        enable_hetero_insert &&
                        inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                             insert_x_min, insert_x_max, insert_y_min,
                                             insert_y_max, insert_z_min, insert_z_max);
                    auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                        position_z_mm, slab_z_ends_device, slab_densities_device,
                        slab_layer_count, water_density_g_per_cm3);
                    if (in_insert) {
                        local_density_g_per_cm3 = insert_density_g_per_cm3;
                    }
                    std::uint8_t ct_material = 2;
                    auto in_ct = false;
                    if ((kProductionPrimaryPath || enable_ct_grid)) {
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
                    if(unified_em && (kProductionPrimaryPath || enable_ct_grid) && !in_ct) {
                        unified_primary_escaped_ct=true;
                        break;
                    }

                    const auto layer_for_material =
                        slab_layer_count > 0
                            ? slab_layer_index(position_z_mm, slab_z_ends_device, slab_layer_count)
                            : 0U;
                    float stopping_power_MeV_per_mm = 0.0F;
                    if ((kProductionPrimaryPath || enable_ct_grid)) {
                        if ((kProductionPrimaryPath || use_schneider_stopping) && in_ct) {
                            const auto floating_sp_index = (energy_MeVu - schneider_sp_e_min) * schneider_sp_inv_dE;
                            auto sp_index = static_cast<int>(sycl::floor(floating_sp_index));
                            sp_index = sycl::max(0, sycl::min(sp_index, static_cast<int>(schneider_sp_energies) - 2));
                            const auto sp_fraction = sycl::clamp(floating_sp_index - static_cast<float>(sp_index), 0.0F, 1.0F);

                            const auto sec_id = static_cast<std::size_t>(
                                sycl::min(static_cast<std::uint32_t>(ct_material), schneider_sp_sections - 1));
                            const auto base = sec_id * schneider_sp_energies;
                            const auto mass_sp = schneider_stopping_device[base + sp_index] +
                                                 sp_fraction * (schneider_stopping_device[base + sp_index + 1] -
                                                                schneider_stopping_device[base + sp_index]);
                            stopping_power_MeV_per_mm = mass_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        } else {
                            const auto water_sp =
                                table_device[index] +
                                fraction * (table_device[index + 1] - table_device[index]);
                            if (use_ct_mass_sp && in_ct && ct_mass_sp_factor_lut_device != nullptr &&
                                ct_n_mass_factors > 0) {
                                const auto mass_factor = ct_lookup_mass_sp_factor(
                                    ct_mass_sp_factor_lut_device,
                                    use_ct_density_mass_spr ? ct_density_spr_n_rho : ct_n_mass_factors,
                                    table_size, use_ct_density_mass_spr,
                                    ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                    static_cast<std::uint32_t>(ct_material), local_density_g_per_cm3,
                                    static_cast<std::size_t>(index), fraction,
                                    [](float x) { return sycl::log(x); });
                                stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                                    water_sp, local_density_g_per_cm3, mass_factor);
                            } else {
                                stopping_power_MeV_per_mm =
                                    water_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                            }
                        }
                    } else if (in_insert && use_insert_material_tables) {
                        stopping_power_MeV_per_mm =
                            insert_sp_device[static_cast<std::size_t>(index)] +
                            fraction *
                                (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                                 insert_sp_device[static_cast<std::size_t>(index)]);
                    } else if (material_table_count > 0) {
                        const auto base = static_cast<std::size_t>(layer_for_material) * table_size;
                        stopping_power_MeV_per_mm =
                            material_sp_device[base + static_cast<std::size_t>(index)] +
                            fraction *
                                (material_sp_device[base + static_cast<std::size_t>(index) + 1] -
                                 material_sp_device[base + static_cast<std::size_t>(index)]);
                    } else {
                        const auto table_stopping_power_MeV_per_mm =
                            primary_water_sp_device[index] +
                            fraction * (primary_water_sp_device[index + 1] -
                                        primary_water_sp_device[index]);
                        const auto scale_density =
                            (slab_layer_count > 0 || in_insert) ? local_density_g_per_cm3 : 1.0F;
                        stopping_power_MeV_per_mm =
                            (slab_layer_count > 0 || in_insert)
                                ? table_stopping_power_MeV_per_mm * scale_density
                                : table_stopping_power_MeV_per_mm;
                    }

                    auto step_mm = sycl::fmin(
                        maximum_step_mm,
                        maximum_relative_energy_loss * energy_MeV /
                            sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                    step_mm = sycl::fmax(step_mm, 1.0e-5F);
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
                    if(unified_em){
                        const int section=(kProductionPrimaryPath || enable_ct_grid)?(in_ct?int(ct_material):-2):-1;
                        if(!CARBON_EM_MATERIAL_CACHE || !unified_primary_state.valid || unified_primary_state.section!=section ||
                           unified_primary_state.density!=local_density_g_per_cm3)
                            unified_primary_state=unified_device.select(section,local_density_g_per_cm3,unified_primary_species);
                        if(!unified_primary_state.covers(energy_MeV)){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,1,section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,0);
                            unified_count(0);break;}
                        if(unified_primary_clock.last_section!=unified_primary_state.section || unified_primary_clock.last_density!=unified_primary_state.density)primary_hadronic_cache.valid=false;
                        // Preserve material cache invalidation without updating a delta clock.
                        unified_primary_clock.last_section=unified_primary_state.section;
                        unified_primary_clock.last_density=unified_primary_state.density;
                        unified_primary_pre=unified_primary_state.prepare(energy_MeV);
                        step_mm=sycl::fmin((em_primary_step_scale!=1.f?unified_primary_state.research_step(energy_MeV,CARBON_EM_STEP_CACHE?unified_primary_pre:unified_primary_state.prepare(energy_MeV),em_primary_step_scale):(CARBON_EM_STEP_CACHE?unified_primary_state.step(unified_primary_pre):unified_primary_state.step(energy_MeV))),unified_primary_distance);
                        // Bound combined mean loss, not the old restricted range alone.
                        const float delta_sp=unified_primary_state.mix(unified_primary_pre.lo.delta_stopping,unified_primary_pre.hi.delta_stopping);
                        if(delta_sp>0) step_mm=sycl::fmin(step_mm,.01f*energy_MeV/sycl::fmax(1e-12f,delta_sp+unified_primary_state.mix(unified_primary_pre.lo.stopping,unified_primary_pre.hi.stopping)));
                    }

                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                        if (dz_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dz_step);
                        }
                    }
                    if (slab_layer_count > 0) {
                        const auto slab_step = distance_to_slab_interface_mm(
                            position_z_mm, direction_z, slab_z_ends_device, slab_layer_count,
                            phantom_length_mm);
                        if (slab_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, slab_step);
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                            insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                        if (insert_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, insert_step);
                        }
                    }
                    CtFaceClampResult ct_clamp_res{step_mm, false, 0};
                    if ((kProductionPrimaryPath || enable_ct_grid) && in_ct) {
                        ct_clamp_res = clamp_step_to_ct_faces_exact(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz);
                        step_mm = ct_clamp_res.step_mm;
                    }
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
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
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
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

                    if (enable_energy_straggling && enable_step_stable_straggling) {
                        stable_straggling.prepare_step(
                            step_mm, phantom_length_mm - position_z_mm);
                    }

                    bool inelastic_this_step = false;
                    // Elastic candidates are resolved only by the configured CT/water bank.
                    bool ct_elastic_this_step = false;
                    std::int16_t cinel02_target_z_step = 0;
                    std::int16_t cinel02_target_a_step = 0;
                    if ((kProductionPrimaryPath || enable_inelastic) &&
                        energy_MeV > energy_cutoff_MeV) {
                        const auto cur_e_u = energy_MeV * inverse_mass_number;
                        if ((kProductionPrimaryPath || use_schneider_primary_xs)) {
                            if (in_ct || (!kProductionPrimaryPath && use_unified_water)) {
                                const std::uint32_t section_id = (!kProductionPrimaryPath && use_unified_water) ? 255U : static_cast<std::uint32_t>(
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
                                    ((!kProductionPrimaryPath && use_all_elastic) && (in_ct || (!kProductionPrimaryPath && use_unified_water))) ||
                                    (use_ct_elastic && in_ct &&
                                    schneider_ct_device_ctx.elastic_sampler.rate_version == 3);
                                if ((!kProductionPrimaryPath && use_all_elastic) && elastic_armed) {
                                    const float rate=all_elastic.rate(16,(!kProductionPrimaryPath && use_unified_water)?25:section_id,cur_e_u);
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
                                if (unified_em) {
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
                                    nuclear_tau_remaining, nuclear_tau_active, step_mm, macro_tot, u_nuc);
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

                    // Diagnostic only: CT face clamping above guarantees the
                    // segment stays in its starting CT voxel. Do not score a
                    // laterally escaped segment into a clamped edge voxel.
                    if ((!kProductionPrimaryPath && enable_primary_voxel_fluence) &&
                        (!(kProductionPrimaryPath || enable_ct_grid) || in_ct)) {
                        sycl::atomic_ref<
                            DoseAtomicT, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_track_length(
                                primary_voxel_track_length_device[voxel_index]);
                        atomic_track_length.fetch_add(
                            static_cast<DoseAtomicT>(step_mm));
                    }

                    float mean_loss_MeV = 0.0F;
                    if(unified_em){
                        // The unified loss sampler computes the physical mean below.
                        // This separate query is needed only by the optional audit.
                        if ((kProductionPrimaryPath ? nullptr : primary_loss_query_audit))
                            mean_loss_MeV=unified_primary_state.mean(energy_MeV,step_mm);
                    } else if (ct_primary_midpoint_stopping && (kProductionPrimaryPath || use_schneider_stopping) && in_ct) {
                        mean_loss_MeV=midpoint_continuous_energy_loss(energy_MeV,step_mm,
                            stopping_power_MeV_per_mm,[&](float mid_energy) {
                                const float f=(mid_energy*inverse_mass_number-schneider_sp_e_min)*schneider_sp_inv_dE;
                                const int i=sycl::clamp(static_cast<int>(sycl::floor(f)),0,static_cast<int>(schneider_sp_energies)-2);
                                const float w=sycl::clamp(f-static_cast<float>(i),0.0F,1.0F);
                                const auto base=static_cast<std::size_t>(ct_material)*schneider_sp_energies;
                                const float mass_sp=schneider_stopping_device[base+i]+w*(schneider_stopping_device[base+i+1]-schneider_stopping_device[base+i]);
                                return mass_sp*sycl::fmax(local_density_g_per_cm3,1.0e-6F);
                            });
                    } else if (enable_csda_range_energy_loss && energy_grid_device != nullptr &&
                        cumulative_range_device != nullptr) {
                        const auto end_energy_MeVu = csda_energy_after_distance_device(
                            primary_csda_energy_grid_device, primary_water_sp_device,
                            primary_csda_a1_device,
                            table_size, energy_MeVu, step_mm, primary_mass_number);
                        mean_loss_MeV = sycl::clamp(
                            energy_MeV -
                                static_cast<float>(primary_mass_number) * end_energy_MeVu,
                            0.0F, energy_MeV);
                    } else if (material_table_count == 0 && !in_insert &&
                               (!use_ct_mass_sp || !in_ct)) {
                        const auto mid_energy_MeV = sycl::fmax(
                            minimum_table_energy * static_cast<float>(primary_mass_number),
                            energy_MeV - 0.5F * stopping_power_MeV_per_mm * step_mm);
                        const auto mid_energy_MeVu = mid_energy_MeV * inverse_mass_number;
                        const auto mid_floating_index =
                            (mid_energy_MeVu - minimum_table_energy) * inverse_table_step;
                        auto mid_index =
                            static_cast<int>(sycl::floor(mid_floating_index));
                        mid_index = sycl::max(
                            0, sycl::min(mid_index, static_cast<int>(table_size) - 2));
                        const auto mid_fraction = sycl::clamp(
                            mid_floating_index - static_cast<float>(mid_index),
                            0.0F, 1.0F);
                        const auto mid_sp =
                            (primary_water_sp_device[mid_index] +
                             mid_fraction * (primary_water_sp_device[mid_index + 1] -
                                             primary_water_sp_device[mid_index])) *
                            sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        mean_loss_MeV = mid_sp * step_mm;
                    } else {
                        mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                    }
                    auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);

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

                    if (enable_energy_straggling && !unified_em) {
                        // Explicit smoke hybrid: retain the existing Gaussian
                        // only for f<1e-4; never substitute it for missing energy
                        // coverage or other out-of-domain package queries.
                        if (use_packaged_fluctuation &&
                            (!fluct_fraction_hybrid || mean_loss_MeV / energy_MeV >= 1.0e-4F)) {
                            const auto fluct_coordinate = fluct_fraction_axis
                                ? mean_loss_MeV / energy_MeV
                                : local_density_g_per_cm3 * step_mm / 10.0F;
                            if (fluct_fraction_axis && mean_loss_MeV > 0 &&
                                (energy_MeVu < fluct_energy_device[0] ||
                                 energy_MeVu > fluct_energy_device[fluct_energy_count-1] ||
                                 fluct_coordinate < fluct_density_device[0] ||
                                 fluct_coordinate > fluct_density_device[fluct_density_count-1])) {
                                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, sycl::access::address_space::global_space>
                                    fail(*fluct_domain_failures);
                                fail.fetch_add(1);
                            }
                            const auto u_loss = rng::uniform01(
                                spot_seed, rng_history, steps, 2);
                            const auto ratio = sample_energy_loss_ratio_from_grid(
                                fluct_energy_device, fluct_energy_count,
                                fluct_density_device, fluct_density_count,
                                fluct_probability_device, fluct_probability_count,
                                fluct_quantile_device, energy_MeVu,
                                fluct_coordinate, u_loss);
                            const auto local_scale = interpolate_straggling_scale(
                                energy_MeVu, straggling_scale_energies,
                                straggling_scale_values,
                                straggling_scale_point_count, straggling_scale);
                            const auto scaled_ratio =
                                scale_energy_loss_ratio_preserving_mean(ratio, local_scale);
                            deposited_MeV = sycl::clamp(
                                mean_loss_MeV * scaled_ratio, 0.0F, energy_MeV);
                        } else {
                        const auto local_scale = interpolate_straggling_scale(
                            energy_MeVu, straggling_scale_energies, straggling_scale_values,
                            straggling_scale_point_count, straggling_scale);
                        const auto effective_charge =
                            ion_effective_charge_device(primary_atomic_number, energy_MeVu);
                        const auto variance_MeV2 =
                            condensed_total_loss_variance_with_mass_MeV2_device(
                                energy_MeV, primary_rest_mass_MeV, effective_charge, step_mm,
                                local_density_g_per_cm3);
                        if (enable_step_stable_straggling) {
                            if (!stable_straggling.block_active) {
                                const auto u0 = sycl::fmax(
                                    rng::uniform01(spot_seed, rng_history,
                                                   stable_straggling.block_index, 0),
                                    1.0e-12F);
                                const auto u1 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 1);
                                const auto u2 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 2);
                                constexpr float two_pi = 6.2831853071795864769F;
                                const auto gauss =
                                    sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                                stable_straggling.begin_block(
                                    mean_loss_MeV, variance_MeV2, step_mm, local_scale, gauss,
                                    energy_MeV, u2, straggling_sampler);
                            }
                            deposited_MeV = stable_straggling.consume_loss(step_mm, energy_MeV);
                        } else {
                            const auto u0 = sycl::fmax(
                                rng::uniform01(spot_seed, rng_history, steps, 0), 1.0e-12F);
                            const auto u1 = rng::uniform01(spot_seed, rng_history, steps, 1);
                            const auto u2 = rng::uniform01(spot_seed, rng_history, steps, 2);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gauss =
                                sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                            const auto sigma_MeV =
                                local_scale * sycl::sqrt(sycl::fmax(0.0F, variance_MeV2));
                            deposited_MeV = sample_condensed_energy_loss(
                                mean_loss_MeV, sigma_MeV, gauss, u2, energy_MeV, straggling_sampler);
                        }
                        }
                    }

                    float em_continuous_sampled_MeV = 0.0F;
                    float em_delta_sampled_MeV = 0.0F;
                    if(unified_em){
                        auto draw=unified_em_loss(unified_primary_state,unified_primary_clock,energy_MeV,step_mm,unified_primary_rate,unified_primary_distance,enable_energy_straggling,unified_primary_pre,unified_uniform);
                        if(!draw.valid){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,20+static_cast<int>(draw.failure_stage),unified_primary_state.section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,step_mm,draw.mean,draw.delta_mean,draw.delta_variance);
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
                    float material_untracked_MeV=0;
                    // Research-only Unified-water delta spatial response.
                    // Transport (loss/MCS/steps/RNG) is untouched: only the
                    // scoring location of the step loss changes. The shared
                    // two-phase helper is the same code as the legacy
                    // electron diagnostic site. RNG dims 80-83 are dedicated
                    // (verified free); C12/MCS/straggling/nuclear streams
                    // (0-7,40s,50s,60s) are never consumed here.
                    if (minibeam_water_delta_v1 && unified_em &&
                        deposited_MeV > 0.0F && !in_ct && !in_insert &&
                        slab_layer_count == 0 &&
                        primary_atomic_number == 6 &&
                        primary_mass_number == 12 &&
                        water_electron_channels_device != nullptr &&
                        minibeam_water_delta_diag_device != nullptr) {
                        auto delta_diag = [&](int slot, std::uint64_t value) {
                            sycl::atomic_ref<std::uint64_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                a(minibeam_water_delta_diag_device[slot]);
                            a.fetch_add(value);
                        };
                        const auto delta_packet = sample_water_delta_packet(
                            deposited_MeV, energy_MeVu,
                            rng::uniform01(spot_seed, rng_history, steps, 80),
                            rng::uniform01(spot_seed, rng_history, steps, 81),
                            water_electron_channels_device,
                            water_electron_samples_device,
                            water_electron_heads_device,
                            water_electron_channel_count);
                        delta_diag(0, 1);
                        delta_diag(7, static_cast<std::uint64_t>(
                                          deposited_MeV * 1e6));
                        delta_diag(6, static_cast<std::uint64_t>(
                                          em_delta_after_scale_MeV * 1e6));
                        if (delta_packet.valid &&
                            delta_packet.packet_mev > 0.0) {
                            const double phi =
                                6.2831853071795864769 *
                                rng::uniform01(spot_seed, rng_history, steps,
                                               82);
                            const auto axis = Direction3F{
                                direction_x, direction_y, direction_z};
                            const auto ex = rotate_local_direction(
                                static_cast<float>(sycl::cos(phi)),
                                static_cast<float>(sycl::sin(phi)), 0, axis);
                            const auto ey = rotate_local_direction(
                                static_cast<float>(-sycl::sin(phi)),
                                static_cast<float>(sycl::cos(phi)), 0, axis);
                            const auto placed = place_water_delta_packet(
                                delta_packet, position_x_mm, position_y_mm,
                                position_z_mm, direction_x, direction_y,
                                direction_z, step_mm, ex.x, ex.y, ex.z, ey.x,
                                ey.y, ey.z,
                                rng::uniform01(spot_seed, rng_history, steps,
                                               83),
                                water_electron_nodes_device,
                                water_electron_node_count,
                                water_electron_radius_device,
                                phantom_length_mm, minibeam_slit_pitch);
                            if (!placed.valid) {
                                delta_diag(5, 1);
                            } else {
                                // Partition (table semantics, verbatim):
                                // relocated = loss*fraction, escape =
                                // loss*unresolved, residual stays local.
                                // No double counting: Unified delta is only
                                // reported (slot 6), never moved twice.
                                // forward_shifted removes the moved part from
                                // the source-bin depth pending (the immediate
                                // atomics below credit the destination).
                                const float relocated = static_cast<float>(
                                    delta_packet.packet_mev);
                                const float escaped = static_cast<float>(
                                    delta_packet.unresolved_mev);
                                local_voxel_deposit_MeV -=
                                    (relocated + escaped);
                                forward_shifted_MeV += (relocated + escaped);
                                water_physical_escape_MeV += escaped;
                                delta_diag(1, static_cast<std::uint64_t>(
                                                  deposited_MeV * 1e6));
                                delta_diag(2, static_cast<std::uint64_t>(
                                                  (deposited_MeV - relocated -
                                                   escaped) *
                                                  1e6));
                                delta_diag(3, static_cast<std::uint64_t>(
                                                  relocated * 1e6));
                                delta_diag(4, static_cast<std::uint64_t>(
                                                  escaped * 1e6));
                                delta_diag(8 + placed.birth_roi * 3 +
                                               placed.deposit_roi,
                                           static_cast<std::uint64_t>(
                                               relocated * 1e6));
                                if (placed.path_status == 0) {
                                    const int ix = static_cast<int>(sycl::floor(
                                        (placed.point_x - voxel_min_x_mm) /
                                        voxel_size_x_mm));
                                    const int iy = static_cast<int>(sycl::floor(
                                        (placed.point_y - voxel_min_y_mm) /
                                        voxel_size_y_mm));
                                    const int iz = static_cast<int>(sycl::floor(
                                        placed.point_z / voxel_size_z_mm));
                                    if (ix >= 0 && iy >= 0 && iz >= 0 &&
                                        ix < static_cast<int>(voxel_bins_x) &&
                                        iy < static_cast<int>(voxel_bins_y) &&
                                        iz < static_cast<int>(voxel_bins_z)) {
                                        const auto target =
                                            (static_cast<std::size_t>(iz) *
                                                 voxel_bins_y +
                                             static_cast<std::size_t>(iy)) *
                                                voxel_bins_x +
                                            static_cast<std::size_t>(ix);
                                        const int dz = static_cast<int>(
                                            sycl::floor(placed.point_z /
                                                        depth_bin_width_mm));
                                        auto add_delta = [&](auto* address) {
                                            using T = std::remove_pointer_t<
                                                decltype(address)>;
                                            sycl::atomic_ref<
                                                T, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::
                                                    global_space>
                                                atom(*address);
                                            atom.fetch_add(
                                                static_cast<T>(relocated));
                                        };
                                        add_delta(voxel_dose_device + target);
                                        if (dz >= 0 &&
                                            dz < static_cast<int>(
                                                     number_of_bins)) {
                                            add_delta(dose_device + dz);
                                            if (in_fov_dose_device != nullptr) {
                                                add_delta(in_fov_dose_device +
                                                          dz);
                                            }
                                        }
                                    } else {
                                        delta_tail_escaped_scorer_MeV +=
                                            relocated;
                                        water_physical_escape_MeV += relocated;
                                    }
                                } else {
                                    delta_tail_escaped_scorer_MeV += relocated;
                                    water_physical_escape_MeV += relocated;
                                }
                            }
                        }
                    }
                    if(use_material_ct && in_ct && deposited_MeV>0) {
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
                        auto uniform=[&]() {packet_rng=packet_rng*6364136223846793005ULL+1442695040888963407ULL;
                            return double(packet_rng>>11)*0x1.0p-53;};
                        count(0,1);
                        const double density_u=uniform(),birth_u=uniform();
                        const double along=uniform();
                        const std::array<double,3> birth_position{
                            position_x_mm+along*step_mm*direction_x,position_y_mm+along*step_mm*direction_y,
                            position_z_mm+along*step_mm*direction_z};
                        const ElectronCtGeometry geometry{{ct_nx,ct_ny,ct_nz},{ct_origin_x,ct_origin_y,ct_origin_z},
                            {ct_spacing_x,ct_spacing_y,ct_spacing_z},ct_density_device,ct_material_device};
                        const auto birth_material=electron_ct_point_material(geometry,birth_position,
                            {direction_x,direction_y,direction_z});
                        MaterialElectronBirthDraw material_birth;
                        if(birth_material.valid)material_birth=sample_material_electron_birth(birth_material.section,
                            birth_material.density,energy_MeVu,density_u,birth_u,
                            material_electron_views,material_electron_view_count);
                        const auto birth=material_birth.birth;
                        const auto ti=material_birth.table;
                        if(!birth.valid) {
                            count(1,1);untracked(deposited_MeV,false);
                            local_voxel_deposit_MeV=0;forward_shifted_MeV+=deposited_MeV;
                        } else if(birth.fraction>0) {
                            const float weight=deposited_MeV*static_cast<float>(birth.fraction);
                            local_voxel_deposit_MeV-=weight;forward_shifted_MeV+=weight;
                            ElectronEnergyPacket packet;packet.weight_MeV=weight;
                            packet.material_table_index=ti;
                            packet.cursor=bind_electron_birth(birth,material_electron_views[ti],
                                birth_position,{direction_x,direction_y,direction_z},uniform());
                            packet.cursor.physical_density_g_cm3=birth_material.density;
                            packet.status=packet.cursor.valid?ElectronPacketStatus::active:ElectronPacketStatus::invalid;
                            if(electron_short_range_mm>0 && electron_short_range_contained(
                                packet,material_electron_views[ti],geometry,electron_short_range_mm)) {
                                packet.status=ElectronPacketStatus::deposited;
                                packet.deposit_position=packet.cursor.position;
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> hits(*short_range_hits_device);
                                hits.fetch_add(1);
                            }
                            auto previous=packet;auto previous_rng=packet_rng;
                            for(unsigned j=0;j<4096 && packet.status==ElectronPacketStatus::active;++j) {
                                previous=packet;previous_rng=packet_rng;
                                packet=transport_electron_packet_step(packet,material_electron_views,
                                    material_electron_view_count,geometry,uniform);
                            }
                            count(6,1);
                            if(packet.status==ElectronPacketStatus::deposited) {
                                const auto p=packet.deposit_position;
                                const int ix=static_cast<int>(sycl::floor((p[0]-voxel_min_x_mm)/voxel_size_x_mm));
                                const int iy=static_cast<int>(sycl::floor((p[1]-voxel_min_y_mm)/voxel_size_y_mm));
                                const int iz=static_cast<int>(sycl::floor(p[2]/voxel_size_z_mm));
                                auto add=[&](auto* address) {
                                    using T=std::remove_pointer_t<decltype(address)>;
                                    sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(weight));
                                };
                                if(ix>=0 && iy>=0 && iz>=0 && ix<static_cast<int>(voxel_bins_x) &&
                                   iy<static_cast<int>(voxel_bins_y) && iz<static_cast<int>(voxel_bins_z)) {
                                    const auto target=(static_cast<std::size_t>(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;
                                    add(voxel_dose_device+target);
                                    const int dz=static_cast<int>(sycl::floor(p[2]/depth_bin_width_mm));
                                    if(dz>=0 && dz<static_cast<int>(number_of_bins)) {
                                        add(dose_device+dz);if(in_fov_dose_device)add(in_fov_dose_device+dz);
                                    }
                                    if((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))add(charged_origin_voxel_dose_device+target);
                                    if((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))add(minibeam_component_voxel_dose_device+target);
                                    count(3,static_cast<std::uint64_t>(double(weight)*1e6));
                                } else {delta_tail_escaped_scorer_MeV+=weight;count(4,static_cast<std::uint64_t>(double(weight)*1e6));}
                            } else if(packet.status==ElectronPacketStatus::escaped) {
                                water_physical_escape_MeV+=weight;history_water_electron_escaped_MeV+=weight;
                                count(4,static_cast<std::uint64_t>(double(weight)*1e6));
                            } else {
                                const bool photon=packet.status==ElectronPacketStatus::coverage_missing &&
                                    packet.gap==ElectronPacketGap::photon_continuation;
                                untracked(weight,photon);
                                if(!photon) {
                                    count(2,1); // unresolved electrons/invalid/cap remain hard failures
                                    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space> first(material_untracked_device[2]);
                                    if(first.fetch_add(1)==0) {
                                        MaterialPacketFailure f;f.history=global_history;f.step=steps;f.rng=previous_rng;
                                        f.before=previous;f.after=packet;
                                        for(std::size_t t=0;t<material_electron_view_count;++t)
                                            if(material_electron_views[t].section==previous.cursor.section &&
                                               material_electron_views[t].density_g_cm3==previous.cursor.density_g_cm3)
                                                f.advance=advance_electron_continuation(previous.cursor,material_electron_views[t],geometry);
                                        *material_failure_device=f;
                                    }
                                }
                            }
                        }
                    }
                    if(use_water_electron && deposited_MeV>0) {
                        auto counter=[&](int slot,double value) {
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);
                            a.fetch_add(static_cast<std::uint64_t>(value));
                        };
                        counter(0,1);
                        // Shared water-draw redistribution (Phase E helper).
                        // Identical draws, order and values as the inline code
                        // it replaces; the material-electron override below
                        // still substitutes its own draw wholesale.
                        auto delta_packet = sample_water_delta_packet(
                            deposited_MeV, energy_MeVu,
                            rng::uniform01(spot_seed,rng_history,steps,21),
                            rng::uniform01(spot_seed,rng_history,steps,23),
                            water_electron_channels_device,
                            water_electron_samples_device,
                            water_electron_heads_device,
                            water_electron_channel_count);
                        auto draw = delta_packet.draw;
                        const WaterElectronPathNode* response_nodes=water_electron_nodes_device;
                        const double* response_radius=water_electron_radius_device;
                        auto response_node_count=water_electron_node_count;
                        if(use_material_electron) {
                            const auto selected=sample_material_electron_response(-1,material_response_density,energy_MeVu,
                                rng::uniform01(spot_seed,rng_history,steps,24),
                                rng::uniform01(spot_seed,rng_history,steps,21),rng::uniform01(spot_seed,rng_history,steps,23),
                                material_electron_views,material_electron_view_count);
                            draw=selected.response;
                            if(selected.status!=MaterialElectronStatus::hit)draw.valid=false;
                            else {
                                const auto table=material_electron_views[selected.table_index];
                                response_nodes=table.nodes;response_radius=table.prefix_radius;
                                response_node_count=table.node_count;
                            }
                        }
                        if(!draw.valid) {counter(1,1);counter(5,static_cast<double>(deposited_MeV)*1e6);}
                        else {
                            // Finite-source photon remainder is EXPLICITLY unresolved.
                            // In this unvalidated EM-only pilot it is carried as escaping
                            // energy, never renormalized into the charged response.
                            const float unresolved=deposited_MeV*static_cast<float>(draw.unresolved);
                            local_voxel_deposit_MeV-=unresolved;forward_shifted_MeV+=unresolved;
                            water_physical_escape_MeV+=unresolved;
                            const float packet=deposited_MeV*static_cast<float>(draw.fraction);
                            if(packet>0) {
                                const double phi=6.2831853071795864769*rng::uniform01(spot_seed,rng_history,steps,22);
                                const auto axis=Direction3F{direction_x,direction_y,direction_z};
                                const auto ex=rotate_local_direction(static_cast<float>(sycl::cos(phi)),static_cast<float>(sycl::sin(phi)),0,axis);
                                const auto ey=rotate_local_direction(static_cast<float>(-sycl::sin(phi)),static_cast<float>(sycl::cos(phi)),0,axis);
                                const double birth=rng::uniform01(spot_seed,rng_history,steps,20);
                                WaterDeltaPacket placed_packet;
                                placed_packet.valid = true;
                                placed_packet.packet_mev = packet;
                                placed_packet.unresolved_mev = unresolved;
                                placed_packet.draw = draw;
                                const auto placed = place_water_delta_packet(
                                    placed_packet, position_x_mm, position_y_mm,
                                    position_z_mm, direction_x, direction_y,
                                    direction_z, step_mm, ex.x, ex.y, ex.z,
                                    ey.x, ey.y, ey.z, birth, response_nodes,
                                    response_node_count, response_radius,
                                    phantom_length_mm, 0.0);
                                const auto status = !placed.valid
                                    ? WaterElectronPathStatus::invalid
                                    : (placed.path_status == 0
                                           ? WaterElectronPathStatus::contained
                                           : WaterElectronPathStatus::escaped);
                                if(status==WaterElectronPathStatus::invalid) {counter(2,1);counter(5,static_cast<double>(packet)*1e6);}
                                else {
                                    local_voxel_deposit_MeV-=packet;forward_shifted_MeV+=packet;
                                    counter(6,1);
                                    if(status==WaterElectronPathStatus::escaped) {
                                        water_physical_escape_MeV+=packet;counter(4,static_cast<double>(packet)*1e6);
                                    } else {
                                        const double x=placed.point_x;
                                        const double y=placed.point_y;
                                        const double z=placed.point_z;
                                        const int ix=static_cast<int>(sycl::floor((x-voxel_min_x_mm)/voxel_size_x_mm));
                                        const int iy=static_cast<int>(sycl::floor((y-voxel_min_y_mm)/voxel_size_y_mm));
                                        const int iz=static_cast<int>(sycl::floor(z/voxel_size_z_mm));
                                        auto add=[&](auto* address) {
                                            using T=std::remove_pointer_t<decltype(address)>;
                                            sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(packet));
                                        };
                                        // Native water has no transverse material edge at
                                        // the dose ROI. Only the final point determines its tally.
                                        if(ix>=0 && iy>=0 && iz>=0 && ix<static_cast<int>(voxel_bins_x) &&
                                           iy<static_cast<int>(voxel_bins_y) && iz<static_cast<int>(voxel_bins_z)) {
                                            const auto target=(static_cast<std::size_t>(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;
                                            const int dz=static_cast<int>(sycl::floor(z/depth_bin_width_mm));
                                            add(voxel_dose_device+target);
                                            if(dz>=0 && dz<static_cast<int>(number_of_bins)) {
                                                add(dose_device+dz);if(in_fov_dose_device)add(in_fov_dose_device+dz);
                                            }
                                            if((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))add(charged_origin_voxel_dose_device+target);
                                            if((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))add(minibeam_component_voxel_dose_device+target);
                                            counter(3,static_cast<double>(packet)*1e6);
                                        } else {delta_tail_escaped_scorer_MeV+=packet;counter(4,static_cast<double>(packet)*1e6);}
                                    }
                                }
                            }
                        }
                        history_water_electron_escaped_MeV+=water_physical_escape_MeV;
                    }
                    if(use_electron_joint && in_ct && (kProductionPrimaryPath || enable_voxel_scoring) &&
                       voxel_index<number_of_voxels && deposited_MeV>0) {
                        auto counter=[&](int slot,std::uint64_t amount) {
                            if(!enable_electron_joint_diagnostics && slot!=1 && slot!=2) return;
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                                sycl::memory_scope::device,sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);a.fetch_add(amount);
                        };
                        counter(0,1);
                        const auto draw=sample_electron_joint_device(ct_material,energy_MeVu,
                            rng::uniform01(spot_seed,rng_history,steps,21),electron_joint_channels_device,
                            electron_joint_samples_device,electron_joint_minimum,electron_joint_maximum,electron_joint_ceiling);
                        if(draw.status!=ElectronJointStatus::hit) {
                            counter(draw.status==ElectronJointStatus::energy_domain ? 1 : 2,1);
                            counter(5,static_cast<std::uint64_t>(static_cast<double>(deposited_MeV)*1e6));
                        } else if(draw.fraction>0) {
                            // This sampled endpoint is already an energy-weighted FULL
                            // response deposit. Do NOT distribute it uniformly along the
                            // ray again, and do NOT also apply the old transverse tail.
                            const float packet=deposited_MeV*static_cast<float>(draw.fraction);
                            const double radius=sycl::sqrt(draw.longitudinal_mass_g_cm2*draw.longitudinal_mass_g_cm2+
                                                           draw.radial_mass_g_cm2*draw.radial_mass_g_cm2);
                            if((radius>0 || electron_path_vectors_device) && packet>0) {
                                const double phi=6.2831853071795864769*rng::uniform01(spot_seed,rng_history,steps,22);
                                const auto ray=radius>0 ? rotate_local_direction(
                                    static_cast<float>(draw.radial_mass_g_cm2*sycl::cos(phi)/radius),
                                    static_cast<float>(draw.radial_mass_g_cm2*sycl::sin(phi)/radius),
                                    static_cast<float>(draw.longitudinal_mass_g_cm2/radius),
                                    Direction3F{direction_x,direction_y,direction_z}) : Direction3F{0,0,1};
                                const double birth=rng::uniform01(spot_seed,rng_history,steps,20);
                                std::size_t target=voxel_index;int target_z=bin;
                                LongitudinalMarchResult march;
                                if(electron_path_vectors_device) {
                                    counter(6,1);
                                    const auto range=electron_path_ranges_device[draw.sample_index];
                                    const auto ex=rotate_local_direction(static_cast<float>(sycl::cos(phi)),static_cast<float>(sycl::sin(phi)),0,Direction3F{direction_x,direction_y,direction_z});
                                    const auto ey=rotate_local_direction(static_cast<float>(-sycl::sin(phi)),static_cast<float>(sycl::cos(phi)),0,Direction3F{direction_x,direction_y,direction_z});
                                    // FP32 electron march: identical algorithm to the
                                    // validated double march; endpoints agree to
                                    // ~1e-4 mm (voxel scale 0.5 mm). Validated by
                                    // gamma equivalence, not bitwise identity.
                                    const std::array<float,3> birth_f{
                                        position_x_mm+static_cast<float>(birth)*step_mm*direction_x,
                                        position_y_mm+static_cast<float>(birth)*step_mm*direction_y,
                                        position_z_mm+static_cast<float>(birth)*step_mm*direction_z};
                                    const std::array<float,3> origin_f{ct_origin_x,ct_origin_y,ct_origin_z};
                                    const std::array<float,3> spacing_f{ct_spacing_x,ct_spacing_y,ct_spacing_z};
                                    const std::array<int,3> dims_i{static_cast<int>(ct_nx),static_cast<int>(ct_ny),static_cast<int>(ct_nz)};
                                    auto density_f=[&](const std::array<int,3>& c) {
                                        return ct_density_device[(static_cast<std::size_t>(c[2])*ct_ny+c[1])*ct_nx+c[0]];};
                                    MassPathResult path;
                                    path.endpoint={birth_f[0],birth_f[1],birth_f[2]};
                                    const auto stored_bounds=electron_path_bounds_device[draw.sample_index];
                                    MassPathBoundsT<float> bounds_f;
                                    for(int bi=0;bi<3;++bi) {
                                        bounds_f.low[bi]=static_cast<float>(stored_bounds.low[bi]);
                                        bounds_f.high[bi]=static_cast<float>(stored_bounds.high[bi]);
                                        bounds_f.net[bi]=static_cast<float>(stored_bounds.net[bi]);
                                    }
                                    const std::array<std::array<float,3>,3> basis_f{
                                        {{ex.x,ex.y,ex.z},{ey.x,ey.y,ey.z},{direction_x,direction_y,direction_z}}};
                                    std::array<float,3> probe_f{birth_f[0],birth_f[1],birth_f[2]};
                                    const bool same_cell=try_same_voxel_mass_path<float>(probe_f,
                                        bounds_f,basis_f,origin_f,spacing_f,dims_i,density_f);
                                    if(same_cell) {
                                        path.endpoint={probe_f[0],probe_f[1],probe_f[2]};
                                    }
                                    if(same_cell)path.completed_segments=range.count;
                                    else path=replay_mass_polyline_indexed<float>(
                                        birth_f,range.count,
                                        [&](std::size_t j) {
                                            const auto v=electron_path_vectors_device[range.offset+j];
                                            return std::array<float,3>{
                                                static_cast<float>(v[0])*ex.x+static_cast<float>(v[1])*ey.x+static_cast<float>(v[2])*direction_x,
                                                static_cast<float>(v[0])*ex.y+static_cast<float>(v[1])*ey.y+static_cast<float>(v[2])*direction_y,
                                                static_cast<float>(v[0])*ex.z+static_cast<float>(v[1])*ey.z+static_cast<float>(v[2])*direction_z};
                                        },origin_f,spacing_f,dims_i,density_f);
                                    march.invalid=path.invalid;march.escaped=path.escaped;
                                    if(!march.invalid && !march.escaped) {
                                        const int x=static_cast<int>(sycl::floor((path.endpoint[0]-ct_origin_x)/ct_spacing_x));
                                        const int y=static_cast<int>(sycl::floor((path.endpoint[1]-ct_origin_y)/ct_spacing_y));
                                        const int z=static_cast<int>(sycl::floor((path.endpoint[2]-ct_origin_z)/ct_spacing_z));
                                        if(x<0 || y<0 || z<0 || x>=static_cast<int>(ct_nx) || y>=static_cast<int>(ct_ny) || z>=static_cast<int>(ct_nz))march.escaped=true;
                                        else {target=(static_cast<std::size_t>(z)*ct_ny+y)*ct_nx+x;target_z=z;}
                                    }
                                } else march=march_longitudinal_mass_segments(
                                    {position_x_mm+birth*step_mm*direction_x,position_y_mm+birth*step_mm*direction_y,position_z_mm+birth*step_mm*direction_z},
                                    {ray.x,ray.y,ray.z},{ct_origin_x,ct_origin_y,ct_origin_z},
                                    {ct_spacing_x,ct_spacing_y,ct_spacing_z},
                                    {static_cast<int>(ct_nx),static_cast<int>(ct_ny),static_cast<int>(ct_nz)},radius,
                                    [&](const std::array<int,3>& cell) {
                                        return static_cast<double>(ct_density_device[(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0]]);
                                    },[&](const std::array<int,3>& cell,double) {
                                        target=(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0];target_z=cell[2];
                                    });
                                if(march.invalid || march.blocked)counter(2,1);
                                else {
                                    if(march.escaped) {
                                        delta_tail_escaped_scorer_MeV+=packet;
                                        counter(4,static_cast<std::uint64_t>(static_cast<double>(packet)*1e6));
                                    } else {
                                        auto add=[&](auto* address) {
                                            using T=std::remove_pointer_t<decltype(address)>;
                                            sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(packet));
                                        };
                                        if(target==voxel_index && target_z==bin) {
                                            // The replay still happened. Cache its original
                                            // packet separately, preserving rounded dE-p
                                            // and p rather than replacing them by dE.
                                            same_voxel_electron_packet_MeV=packet;
                                        } else {
                                            add(voxel_dose_device+target);add(dose_device+target_z);
                                            if(in_fov_dose_device)add(in_fov_dose_device+target_z);
                                            if((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))add(charged_origin_voxel_dose_device+target);
                                            if((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))add(minibeam_component_voxel_dose_device+target);
                                        }
                                        counter(3,static_cast<std::uint64_t>(static_cast<double>(packet)*1e6));
                                    }
                                    local_voxel_deposit_MeV-=packet;forward_shifted_MeV+=packet;
                                }
                            }
                        }
                    }
                    if (!use_electron_joint && use_schneider_delta_tail && in_ct && ct_material == 0U &&
                        (kProductionPrimaryPath || enable_voxel_scoring) && voxel_index < number_of_voxels &&
                        schneider_delta_source_eligible_device[voxel_index] != 0U) {
                        float moved_fraction = 0.0F;
                        float radius_mm = 0.0F;
                        schneider_delta_tail_lookup_device(
                            energy_MeVu,
                            rng::uniform01(spot_seed, rng_history, steps, 17),
                            schneider_delta_energies_device,
                            schneider_delta_fractions_device,
                            schneider_delta_radii_device,
                            schneider_delta_energy_count,
                            schneider_delta_quantile_count,
                            moved_fraction, radius_mm);
                        const auto moved_MeV =
                            deposited_MeV * sycl::clamp(moved_fraction, 0.0F, 0.5F);
                        if (moved_MeV > 0.0F && radius_mm > 0.0F) {
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto phi = two_pi * rng::uniform01(
                                spot_seed, rng_history, steps, 18);
                            const auto transverse = rotate_local_direction(
                                sycl::cos(phi), sycl::sin(phi), 0.0F,
                                Direction3F{direction_x, direction_y, direction_z});
                            const auto source_x = position_x_mm + 0.5F * step_mm * direction_x;
                            const auto source_y = position_y_mm + 0.5F * step_mm * direction_y;
                            const auto source_z = position_z_mm + 0.5F * step_mm * direction_z;
                            const auto destination_x = source_x + radius_mm * transverse.x;
                            const auto destination_y = source_y + radius_mm * transverse.y;
                            const auto destination_z = source_z + radius_mm * transverse.z;
                            float destination_density = 0.0F;
                            std::uint8_t destination_material = 255U;
                            const auto destination_in_section0 = ct_sample(
                                destination_x, destination_y, destination_z,
                                ct_origin_x, ct_origin_y, ct_origin_z,
                                ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                ct_nx, ct_ny, ct_nz, ct_density_device,
                                ct_material_device, destination_density,
                                destination_material) && destination_material == 0U;
                            const auto destination_voxel_x = static_cast<int>(sycl::floor(
                                (destination_x - voxel_min_x_mm) / voxel_size_x_mm));
                            const auto destination_voxel_y = static_cast<int>(sycl::floor(
                                (destination_y - voxel_min_y_mm) / voxel_size_y_mm));
                            const auto destination_bin = static_cast<int>(sycl::floor(
                                destination_z / depth_bin_width_mm));
                            const auto destination_in_scorer =
                                destination_voxel_x >= 0 &&
                                destination_voxel_x < static_cast<int>(voxel_bins_x) &&
                                destination_voxel_y >= 0 &&
                                destination_voxel_y < static_cast<int>(voxel_bins_y) &&
                                destination_bin >= 0 &&
                                destination_bin < static_cast<int>(number_of_bins);
                            const auto fixed = static_cast<std::uint64_t>(
                                static_cast<double>(moved_MeV) * 1.0e6);
                            if (destination_in_section0 && destination_in_scorer) {
                                const auto destination_voxel =
                                    static_cast<std::size_t>(destination_bin) * voxel_plane_size +
                                    static_cast<std::size_t>(destination_voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(destination_voxel_x);
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_delta(voxel_dose_device[destination_voxel]);
                                atomic_delta.fetch_add(static_cast<DoseAtomicT>(moved_MeV));
                                // Mirror the actual 3-D destination. Even a pencil beam
                                // can cross a depth face between step start and midpoint.
                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    target_depth(dose_device[destination_bin]);
                                target_depth.fetch_add(static_cast<DepthAtomicT>(moved_MeV));
                                if (in_fov_dose_device) {
                                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        target_fov(in_fov_dose_device[destination_bin]);
                                    target_fov.fetch_add(static_cast<DepthAtomicT>(moved_MeV));
                                }
                                transverse_relocated_MeV = moved_MeV;
                                if ((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring)) {
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_origin(charged_origin_voxel_dose_device[
                                            destination_voxel]);
                                    atomic_origin.fetch_add(
                                        static_cast<DoseAtomicT>(moved_MeV));
                                }
                                if ((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring)) {
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_component(minibeam_component_voxel_dose_device[
                                            destination_voxel]);
                                    atomic_component.fetch_add(
                                        static_cast<DoseAtomicT>(moved_MeV));
                                }
                                local_voxel_deposit_MeV -= moved_MeV;
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_moved(schneider_delta_energy_device[0]);
                                atomic_moved.fetch_add(fixed);
                            } else if (!destination_in_scorer) {
                                // The aligned CCTG and dose scorer share the same bounds.
                                // A sampled delta-electron endpoint outside those bounds must
                                // leave the voxel score instead of being folded back into the
                                // edge voxel. Transfer it from the in-grid sink to outside-grid.
                                local_voxel_deposit_MeV -= moved_MeV;
                                delta_tail_escaped_scorer_MeV += moved_MeV;
                                transverse_escaped_MeV = moved_MeV;
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_escape(schneider_delta_energy_device[2]);
                                atomic_escape.fetch_add(fixed);
                            } else {
                                // Cross-material electron transport is outside this section-0
                                // LUT. Preserve the energy locally rather than aliasing a target.
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fallback(schneider_delta_energy_device[1]);
                                atomic_fallback.fetch_add(fixed);
                            }
                        }
                    }
                        if (use_schneider_delta_longitudinal && in_ct && (kProductionPrimaryPath || enable_voxel_scoring) &&
                            voxel_index < number_of_voxels && deposited_MeV > 0.0F &&
                            (use_longitudinal_interface_mass || (ct_material==0U &&
                             schneider_delta_source_eligible_device[voxel_index]!=0U))) {
                            // Forward supplement: carry a fitted fraction of the local
                            // deposit downstream along the particle direction with an
                            // exponential range, distributed uniformly along the ray
                            // (diagnostic approximation, not validated electron physics).
                            // Only reference-density section-0 voxels receive exact
                            // path shares. At unsupported density/material, retain the
                            // remainder at source; at scorer exit, book escape. This
                            // does not establish correct air-tissue interface transport.
                            float forward_fraction = 0.0F;
                            float forward_lambda_mm = 0.0F;
                            const bool longitudinal_covered = schneider_longitudinal_lookup_device(
                                energy_MeVu,
                                schneider_long_energies_device,
                                schneider_long_fractions_device,
                                schneider_long_lambdas_device,
                                schneider_long_energy_count,
                                forward_fraction, forward_lambda_mm);
                            if (!longitudinal_covered) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    count(schneider_delta_energy_device[6]),
                                    energy(schneider_delta_energy_device[7]);
                                const auto domain_slot = count.fetch_add(1);
                                if (longitudinal_domain_device &&
                                    domain_slot < kLongitudinalDomainLogCap) {
                                    longitudinal_domain_device[domain_slot] = {
                                        global_history, static_cast<std::uint32_t>(steps),
                                        sampled_initial_energy_MeVu, energy_MeVu,
                                        position_z_mm, step_mm, deposited_MeV};
                                }
                                energy.fetch_add(static_cast<std::uint64_t>(
                                    static_cast<double>(deposited_MeV) * 1e6));
                            }
                            const auto forward_MeV =
                                deposited_MeV * sycl::clamp(
                                    forward_fraction * schneider_long_fraction_scale,
                                    0.0F, 0.5F);
                            // Default remains reference-only. Optional probe has already
                            // verified the ENTIRE grid is homogeneous at one known density.
                            const double rho_ref = schneider_long_diagnostic_density;
                            if (forward_MeV > 0.0F && forward_lambda_mm > 0.0F) {
                                const auto u = rng::uniform01(spot_seed, rng_history, steps, 19);
                                const double range = -static_cast<double>(forward_lambda_mm) *
                                    (kLongitudinalReferenceDensityGPerCm3 / rho_ref) *
                                    sycl::log(1.0 - sycl::clamp(static_cast<double>(u), 0.0, 0.99999988));
                                double fwd_escaped_MeV = 0, fwd_kept_MeV = 0, fwd_scored_MeV = 0;
                                if (use_longitudinal_interface_mass && range > 0) {
                                    // Diagnostic hypothesis: the fixed air kernel expressed
                                    // in mass thickness, with sources on BOTH sides. Same
                                    // fraction/range table; no interface-fitted multiplier.
                                    // The ion loses energy along the ENTIRE step, not at
                                    // its midpoint. In tissue lambda is sub-voxel; midpoint
                                    // emission biases cross-face fluence. Independent tag20
                                    // integrates uniform per-step births without a grid phase.
                                    const double birth_fraction=rng::uniform01(spot_seed,rng_history,steps,20);
                                    const auto march = march_longitudinal_mass_segments(
                                        {position_x_mm + birth_fraction * step_mm * direction_x,
                                         position_y_mm + birth_fraction * step_mm * direction_y,
                                         position_z_mm + birth_fraction * step_mm * direction_z},
                                        {direction_x, direction_y, direction_z},
                                        {ct_origin_x, ct_origin_y, ct_origin_z},
                                        {ct_spacing_x, ct_spacing_y, ct_spacing_z},
                                        {static_cast<int>(ct_nx), static_cast<int>(ct_ny),
                                         static_cast<int>(ct_nz)}, range * rho_ref / 10.0,
                                        [&](const std::array<int,3>& cell) {
                                            return static_cast<double>(ct_density_device[
                                                (static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0]]);
                                        },
                                        [&](const std::array<int,3>& cell,double fraction) {
                                            const auto index=(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0];
                                            const double share=static_cast<double>(forward_MeV)*fraction;
                                            auto add=[&](auto* address) {
                                                using T=std::remove_pointer_t<decltype(address)>;
                                                sycl::atomic_ref<T,sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space> atom(*address);
                                                atom.fetch_add(static_cast<T>(share));
                                            };
                                            add(voxel_dose_device+index);
                                            add(dose_device+cell[2]);
                                            if(in_fov_dose_device) add(in_fov_dose_device+cell[2]);
                                            if((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))
                                                add(charged_origin_voxel_dose_device+index);
                                            if((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))
                                                add(minibeam_component_voxel_dose_device+index);
                                            fwd_scored_MeV+=share;
                                        });
                                    const double remainder=sycl::fmax(0.0,static_cast<double>(forward_MeV)-fwd_scored_MeV);
                                    if(march.escaped) fwd_escaped_MeV=remainder;
                                    else fwd_kept_MeV=remainder;
                                    if(march.invalid || march.blocked) {
                                        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space> invalid(schneider_delta_energy_device[9]);
                                        invalid.fetch_add(1);
                                    }
                                } else if (range > 0 &&
                                    sycl::fabs(static_cast<double>(local_density_g_per_cm3)-rho_ref)
                                        <= rho_ref * 1e-4) {
                                    const auto march = march_longitudinal_segments(
                                        {position_x_mm + 0.5 * step_mm * direction_x,
                                         position_y_mm + 0.5 * step_mm * direction_y,
                                         position_z_mm + 0.5 * step_mm * direction_z},
                                        {direction_x, direction_y, direction_z},
                                        {ct_origin_x, ct_origin_y, ct_origin_z},
                                        {ct_spacing_x, ct_spacing_y, ct_spacing_z},
                                        {static_cast<int>(ct_nx), static_cast<int>(ct_ny),
                                         static_cast<int>(ct_nz)}, range,
                                        [&](const std::array<int,3>& cell, double length) {
                                            const auto index = (static_cast<std::size_t>(cell[2]) *
                                                ct_ny + cell[1]) * ct_nx + cell[0];
                                            if (ct_material_device[index] != 0U ||
                                                sycl::fabs(static_cast<double>(ct_density_device[index]) -
                                                           rho_ref) > rho_ref * 1e-4) return false;
                                            const double share = static_cast<double>(forward_MeV) * length / range;
                                            auto add = [&](auto* address) {
                                                using T = std::remove_pointer_t<decltype(address)>;
                                                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space> atom(*address);
                                                atom.fetch_add(static_cast<T>(share));
                                            };
                                            add(voxel_dose_device + index);
                                            add(dose_device + cell[2]);
                                            if (in_fov_dose_device) add(in_fov_dose_device + cell[2]);
                                            if ((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))
                                                add(charged_origin_voxel_dose_device + index);
                                            if ((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))
                                                add(minibeam_component_voxel_dose_device + index);
                                            fwd_scored_MeV += share;
                                            return true;
                                        });
                                    const double remainder = sycl::fmax(
                                        0.0, static_cast<double>(forward_MeV)-fwd_scored_MeV);
                                    if (march.escaped) fwd_escaped_MeV = remainder;
                                    else fwd_kept_MeV = remainder;
                                    if (march.invalid) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            invalid(schneider_delta_energy_device[9]);
                                        invalid.fetch_add(1);
                                    }
                                } else fwd_kept_MeV = forward_MeV;
                                local_voxel_deposit_MeV -= forward_MeV - fwd_kept_MeV;
                                delta_tail_escaped_scorer_MeV += fwd_escaped_MeV;
                                forward_shifted_MeV = forward_MeV - fwd_kept_MeV;
                                const auto fwd_fixed = static_cast<std::uint64_t>(
                                    fwd_scored_MeV * 1.0e6);
                                const auto fwd_esc_fixed = static_cast<std::uint64_t>(
                                    static_cast<double>(fwd_escaped_MeV) * 1.0e6);
                                const auto fwd_kept_fixed = static_cast<std::uint64_t>(
                                    static_cast<double>(fwd_kept_MeV) * 1.0e6);
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fwd_moved(schneider_delta_energy_device[3]);
                                atomic_fwd_moved.fetch_add(fwd_fixed);
                                if (fwd_kept_fixed > 0U) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_fwd_kept(schneider_delta_energy_device[4]);
                                    atomic_fwd_kept.fetch_add(fwd_kept_fixed);
                                }
                                if (fwd_esc_fixed > 0U) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_fwd_escape(schneider_delta_energy_device[5]);
                                    atomic_fwd_escape.fetch_add(fwd_esc_fixed);
                                }
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
                        if (enable_minibeam_primary_c12_roi_scoring) {
                            const auto score_c12 = [&](std::size_t kind, float value) {
                                score_minibeam_c12_roi_device(
                                    minibeam_c12_roi_dose_device, kind, energy_MeV,
                                    voxel_index, number_of_bins, voxel_bins_x,
                                    voxel_bins_y, minibeam_fixed_region_by_x_bin_device,
                                    value);
                            };
                            score_c12(minibeam_c12_roi_local_total_deposit,
                                      local_voxel_deposit_MeV +
                                          same_voxel_electron_packet_MeV);
                            score_c12(minibeam_c12_roi_continuous_sampled,
                                      em_continuous_sampled_MeV);
                            score_c12(minibeam_c12_roi_delta_sampled,
                                      em_delta_sampled_MeV);
                            score_c12(minibeam_c12_roi_continuous_after_scale,
                                      em_continuous_after_scale_MeV);
                            score_c12(minibeam_c12_roi_delta_after_scale,
                                      em_delta_after_scale_MeV);
                            score_c12(minibeam_c12_roi_fluence, step_mm);
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
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device,
                        (kProductionPrimaryPath || enable_voxel_scoring) && voxel_index >= 0,
                        deposited_MeV - delta_tail_escaped_scorer_MeV-water_physical_escape_MeV-material_untracked_MeV);
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device, false,
                        delta_tail_escaped_scorer_MeV);
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
                        if ((kProductionPrimaryPath || enable_ct_grid) && in_ct && (kProductionPrimaryPath || enable_ct_material_mcs)) {
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
                        } else if (slab_layer_count > 0) {
                            radiation_length_g_per_cm2 =
                                slab_radiation_lengths_device[layer_for_material];
                        }
                        // Generic C12 FE path. Yields to the minibeam water
                        // branch below when a non-legacy minibeam primary
                        // model is selected (FE-tail or Urban v2); otherwise
                        // the minibeam branch would be dead code whenever the
                        // generic fermi_eyges model is on.
                        if (c12_fermi_eyges &&
                            primary_atomic_number == 6 &&
                            primary_mass_number == 12 &&
                            !(enable_minibeam &&
                              (minibeam_water_primary_fermi_eyges_tail ||
                               minibeam_water_primary_urban_v2))) {
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
                                c12_fermi_eyges_transport_step(
                                    Direction3F{direction_x, direction_y,
                                                direction_z},
                                    energy_MeV, deposited_MeV, step_mm,
                                    c12_fermi_eyges_max_segment_mm,
                                    local_density_g_per_cm3,
                                    radiation_length_g_per_cm2, spot_seed,
                                    rng_history,
                                    static_cast<std::uint64_t>(steps), 40U,
                                    observation_path_mm);
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
#if defined(CARBON_ENABLE_MINIBEAM)
                        else if (enable_minibeam &&
                            minibeam_water_primary_fermi_eyges_tail &&
                            !in_ct && !in_insert && slab_layer_count == 0) {
                            auto segment_direction = Direction3F{
                                direction_x, direction_y, direction_z};
                            auto segment_offset = Direction3F{
                                0.0F, 0.0F, 0.0F};
                            auto traversed_mm = 0.0F;
                            std::uint32_t segment_index = 0U;
                            while (traversed_mm < step_mm) {
                                const auto segment_mm = sycl::fmin(
                                    minibeam_water_primary_mcs_max_segment_mm,
                                    step_mm - traversed_mm);
                                std::size_t observation_plane =
                                    minibeam_water_primary_plane_count;
                                auto observation_fraction = -1.0F;
                                if (minibeam_water_primary_plane_records_device !=
                                        nullptr &&
                                    segment_direction.z > 0.0F) {
                                    const auto segment_start_z =
                                        position_z_mm + segment_offset.z;
                                    const auto segment_end_z = segment_start_z +
                                        segment_direction.z * segment_mm;
                                    for (std::size_t plane = 0;
                                         plane < minibeam_water_primary_plane_count;
                                         ++plane) {
                                        auto& candidate_record =
                                            minibeam_water_primary_plane_records_device[
                                                global_history *
                                                    minibeam_water_primary_plane_count +
                                                plane];
                                        const auto plane_depth_mm =
                                            minibeam_water_primary_plane_depths_device[
                                                plane];
                                        if (!candidate_record.valid &&
                                            segment_start_z < plane_depth_mm &&
                                            segment_end_z >= plane_depth_mm) {
                                            observation_plane = plane;
                                            observation_fraction = sycl::clamp(
                                                (plane_depth_mm - segment_start_z) /
                                                    (segment_end_z -
                                                     segment_start_z),
                                                1.0e-6F, 1.0F - 1.0e-6F);
                                            break;
                                        }
                                    }
                                }
                                const auto energy_fraction =
                                    (traversed_mm + 0.5F * segment_mm) /
                                    step_mm;
                                const auto correlated =
                                    water_c12_fermi_eyges_tail_step(
                                        segment_direction,
                                        sycl::fmax(
                                            energy_cutoff_MeV,
                                            energy_MeV - energy_fraction *
                                                deposited_MeV),
                                        segment_mm,
                                        local_density_g_per_cm3,
                                        radiation_length_g_per_cm2,
                                        spot_seed, rng_history,
                                        static_cast<std::uint64_t>(steps) *
                                                1024U +
                                            segment_index,
                                        40U, observation_fraction);
                                if (correlated.observation_valid &&
                                    observation_plane <
                                        minibeam_water_primary_plane_count) {
                                    const auto observation_path_mm =
                                        observation_fraction * segment_mm;
                                    auto crossing_x = position_x_mm +
                                        segment_offset.x +
                                        segment_direction.x *
                                            observation_path_mm +
                                        correlated
                                            .observation_displacement_mm.x;
                                    auto crossing_y = position_y_mm +
                                        segment_offset.y +
                                        segment_direction.y *
                                            observation_path_mm +
                                        correlated
                                            .observation_displacement_mm.y;
                                    const auto crossing_z = position_z_mm +
                                        segment_offset.z +
                                        segment_direction.z *
                                            observation_path_mm +
                                        correlated
                                            .observation_displacement_mm.z;
                                    const auto plane_depth_mm =
                                        minibeam_water_primary_plane_depths_device[
                                            observation_plane];
                                    // The bridge is parameterized by path length.
                                    // Project its state by the tiny residual axial
                                    // distance to the exact scoring surface.
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
                                        (traversed_mm + observation_path_mm) /
                                            step_mm,
                                        0.0F, 1.0F);
                                    record.history = global_history;
                                    record.particle_id = global_history;
                                    record.rng_stream = rng_history;
                                    record.plane_index = static_cast<std::uint32_t>(
                                        observation_plane);
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
                                segment_offset.x +=
                                    segment_direction.x * segment_mm +
                                    correlated.displacement_mm.x;
                                segment_offset.y +=
                                    segment_direction.y * segment_mm +
                                    correlated.displacement_mm.y;
                                segment_offset.z +=
                                    segment_direction.z * segment_mm +
                                    correlated.displacement_mm.z;
                                segment_direction = correlated.direction;
                                traversed_mm += segment_mm;
                                ++segment_index;
                            }
                            direction_x = segment_direction.x;
                            direction_y = segment_direction.y;
                            direction_z = segment_direction.z;
                            water_scattering_displacement = Direction3F{
                                segment_offset.x - seg_dir_x * step_mm,
                                segment_offset.y - seg_dir_y * step_mm,
                                segment_offset.z - seg_dir_z * step_mm};
                        }
                        else if (enable_minibeam &&
                            minibeam_water_primary_urban_v2 &&
                            !in_ct && !in_insert && slab_layer_count == 0) {
                            // Research-only table-driven Geant4-11.3.2 Urban
                            // for primary C12 in Water_75eV. Transport steps
                            // are subdivided to the Urban max step (0.05 mm,
                            // matching the TOPAS water MaxStepSize); loss and
                            // straggling still use the full step_mm, so only
                            // direction/displacement change vs FE-tail.
                            // Plane records use endpoint projection and never
                            // alter transport (observation-only); the bias vs
                            // a true crossing is O(segment^2), ~1e-4 mm here.
                            UrbanV2LossTable water_urban_table{
                                minibeam_water_urban_loss_e_device,
                                minibeam_water_urban_loss_r_device,
                                minibeam_water_urban_loss_d_device,
                                static_cast<int>(
                                    minibeam_water_urban_loss_count)};
                            auto segment_direction = Direction3F{
                                direction_x, direction_y, direction_z};
                            auto segment_offset = Direction3F{
                                0.0F, 0.0F, 0.0F};
                            auto traversed_mm = 0.0F;
                            std::uint32_t segment_index = 0U;
                            const bool first_water_segment =
                                !water_urban_seen_segment;
                            while (traversed_mm < step_mm) {
                                if (segment_index >= 1000000U) {
                                    // Non-progress guard: a proposal with zero
                                    // (or FP32-stalling) final_geom_path_mm
                                    // would spin this loop forever and hang
                                    // the kernel. Break and count the event;
                                    // legitimate use needs <= ~200 segments
                                    // (see test_water_urban_subdivision_
                                    // robustness), so the cap is 5000x clear
                                    // of physics.
                                    // Fix B5: the cap trip ALSO raises the
                                    // fatal flag (slot 137): breaking with a
                                    // partial space path while the full macro
                                    // step's energy was already scored must
                                    // fail the run, never pass silently.
                                    if (minibeam_event_counts_device !=
                                        nullptr) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::
                                                global_space>(
                                            minibeam_event_counts_device
                                                [minibeam_water_urban_subdiv_cap_slot])
                                            .fetch_add(1U);
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::
                                                global_space>(
                                            minibeam_event_counts_device
                                                [minibeam_water_urban_fatal_slot])
                                            .fetch_add(1U);
                                    }
                                    break;
                                }
                                ++water_urban_seg_hist;
                                const auto segment_mm = sycl::fmin(
                                    minibeam_water_primary_urban_max_step_mm,
                                    step_mm - traversed_mm);
                                const auto seg_start_x =
                                    position_x_mm + segment_offset.x;
                                const auto seg_start_y =
                                    position_y_mm + segment_offset.y;
                                const auto seg_start_z =
                                    position_z_mm + segment_offset.z;
                                const auto seg_e = sycl::fmax(
                                    energy_cutoff_MeV,
                                    energy_MeV - traversed_mm / step_mm *
                                                     deposited_MeV);
                                const auto urban_scatter =
                                    water_urban_v2_propose_and_sample(
                                        segment_direction, seg_e, 6, 12,
                                        minibeam_water_primary_urban_max_step_mm,
                                        segment_mm, seg_start_x, seg_start_y,
                                        seg_start_z, segment_direction.x,
                                        segment_direction.y,
                                        segment_direction.z, voxel_min_x_mm,
                                        voxel_max_x_mm, voxel_min_y_mm,
                                        voxel_max_y_mm, 0.0F,
                                        phantom_length_mm,
                                        first_water_segment &&
                                            segment_index == 0U,
                                        water_urban_state, water_urban_table,
                                        minibeam_water_urban_zeff_f,
                                        minibeam_water_urban_radlen_mm_f, 1.0F,
                                        spot_seed, rng_history,
                                        static_cast<std::uint64_t>(steps) *
                                                1024U +
                                            segment_index,
                                        70U);
                                // Fix B5: invalid proposal or zero progress
                                // must not spin to the cap (burning 1M
                                // iterations) nor continue silently. Raise
                                // the fatal flag and stop this history's
                                // water transport; the host fails the run.
                                if (!urban_scatter.proposal_valid ||
                                    !(urban_scatter.final_geom_path_mm >
                                      0.0F)) {
                                    if (minibeam_event_counts_device !=
                                        nullptr) {
                                        sycl::atomic_ref<
                                            std::uint64_t,
                                            sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::
                                                global_space>(
                                            minibeam_event_counts_device
                                                [minibeam_water_urban_fatal_slot])
                                            .fetch_add(1U);
                                    }
                                    break;
                                }
                                const auto seg_end_x =
                                    seg_start_x +
                                    urban_scatter.final_geom_path_mm *
                                        segment_direction.x +
                                    urban_scatter.displacement_mm.x;
                                const auto seg_end_y =
                                    seg_start_y +
                                    urban_scatter.final_geom_path_mm *
                                        segment_direction.y +
                                    urban_scatter.displacement_mm.y;
                                const auto seg_end_z =
                                    seg_start_z +
                                    urban_scatter.final_geom_path_mm *
                                        segment_direction.z +
                                    urban_scatter.displacement_mm.z;
                                if (minibeam_water_primary_plane_records_device !=
                                        nullptr &&
                                    urban_scatter.direction.z > 0.0F) {
                                    for (std::size_t plane = 0;
                                         plane <
                                         minibeam_water_primary_plane_count;
                                         ++plane) {
                                        auto& candidate_record =
                                            minibeam_water_primary_plane_records_device[
                                                global_history *
                                                    minibeam_water_primary_plane_count +
                                                plane];
                                        const auto plane_depth_mm =
                                            minibeam_water_primary_plane_depths_device[
                                                plane];
                                        if (!candidate_record.valid &&
                                            seg_start_z < plane_depth_mm &&
                                            seg_end_z >= plane_depth_mm) {
                                            const auto residual_path =
                                                (plane_depth_mm - seg_end_z) /
                                                urban_scatter.direction.z;
                                            const auto step_fraction =
                                                sycl::clamp(
                                                    (traversed_mm +
                                                     urban_scatter
                                                         .final_geom_path_mm) /
                                                        step_mm,
                                                    0.0F, 1.0F);
                                            candidate_record.history =
                                                global_history;
                                            candidate_record.particle_id =
                                                global_history;
                                            candidate_record.rng_stream =
                                                rng_history;
                                            candidate_record.plane_index =
                                                static_cast<std::uint32_t>(
                                                    plane);
                                            candidate_record.atomic_number =
                                                static_cast<std::int16_t>(
                                                    primary_atomic_number);
                                            candidate_record.mass_number =
                                                static_cast<std::int16_t>(
                                                    primary_mass_number);
                                            candidate_record.depth_mm =
                                                plane_depth_mm;
                                            candidate_record
                                                .kinetic_energy_MeV = sycl::fmax(
                                                0.0F, energy_MeV -
                                                          step_fraction *
                                                              deposited_MeV);
                                            candidate_record.weight = 1.0F;
                                            candidate_record.x_mm =
                                                seg_end_x +
                                                residual_path *
                                                    urban_scatter.direction.x;
                                            candidate_record.y_mm =
                                                seg_end_y +
                                                residual_path *
                                                    urban_scatter.direction.y;
                                            candidate_record.direction_x =
                                                urban_scatter.direction.x;
                                            candidate_record.direction_y =
                                                urban_scatter.direction.y;
                                            candidate_record.direction_z =
                                                urban_scatter.direction.z;
                                            candidate_record.valid = 1U;
                                            break;
                                        }
                                    }
                                }
                                segment_offset.x +=
                                    segment_direction.x *
                                        urban_scatter.final_geom_path_mm +
                                    urban_scatter.displacement_mm.x;
                                segment_offset.y +=
                                    segment_direction.y *
                                        urban_scatter.final_geom_path_mm +
                                    urban_scatter.displacement_mm.y;
                                segment_offset.z +=
                                    segment_direction.z *
                                        urban_scatter.final_geom_path_mm +
                                    urban_scatter.displacement_mm.z;
                                segment_direction = urban_scatter.direction;
                                traversed_mm +=
                                    urban_scatter.final_geom_path_mm;
                                ++segment_index;
                            }
                            water_urban_seen_segment = true;
                            direction_x = segment_direction.x;
                            direction_y = segment_direction.y;
                            direction_z = segment_direction.z;
                            water_scattering_displacement = Direction3F{
                                segment_offset.x - seg_dir_x * step_mm,
                                segment_offset.y - seg_dir_y * step_mm,
                                segment_offset.z - seg_dir_z * step_mm};
                        }
#endif
                        else {
                            auto theta0 = highland_projected_rms_angle_device(
                                energy_MeV, primary_atomic_number, primary_mass_number,
                                step_mm, local_density_g_per_cm3,
                                radiation_length_g_per_cm2) * multiple_scattering_scale;
#if defined(CARBON_ENABLE_MINIBEAM)
                            if (enable_minibeam &&
                                minibeam_water_low_energy_mcs_transition > 0.0F) {
                                const auto energy_fraction = sycl::clamp(
                                    energy_MeV * inverse_mass_number /
                                        minibeam_water_low_energy_mcs_transition,
                                    0.0F, 1.0F);
                                const auto low_energy_scale =
                                    minibeam_water_primary_low_energy_mcs_scale +
                                    (1.0F - minibeam_water_primary_low_energy_mcs_scale) *
                                        energy_fraction;
                                theta0 *= low_energy_scale;
                            }
#endif
                            float angular_scale = 1.0F;
#if defined(CARBON_ENABLE_MINIBEAM)
                            if (enable_minibeam &&
                                minibeam_water_primary_mcs_tail_strength > 0.0F &&
                                minibeam_water_primary_mcs_tail_width > 0.0F &&
                                !in_ct && !in_insert && slab_layer_count == 0) {
                                const auto step_radiation_lengths =
                                    local_density_g_per_cm3 * (step_mm / 10.0F) /
                                    radiation_length_g_per_cm2;
                                const auto tail_probability = sycl::clamp(
                                    minibeam_water_primary_mcs_tail_strength *
                                        step_radiation_lengths,
                                    0.0F, 0.25F);
                                const auto tail_variance_fraction =
                                    minibeam_water_primary_mcs_tail_strength *
                                    minibeam_water_primary_mcs_tail_width *
                                    minibeam_water_primary_mcs_tail_width;
                                const auto core_scale = sycl::sqrt(sycl::fmax(
                                    0.0F, (1.0F - tail_variance_fraction) /
                                              (1.0F - tail_probability)));
                                const auto tail_scale =
                                    minibeam_water_primary_mcs_tail_width /
                                    sycl::sqrt(sycl::fmax(
                                        step_radiation_lengths, 1.0e-12F));
                                angular_scale = rng::uniform01(
                                    spot_seed, rng_history, steps, 7) <
                                        tail_probability
                                    ? tail_scale
                                    : core_scale;
                            }
#endif
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
                    if (enable_minibeam_primary_c12_roi_scoring &&
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
                         (!c12_fermi_eyges &&
                          !minibeam_water_primary_fermi_eyges_tail)) &&
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

                    if ((kProductionPrimaryPath || enable_ct_grid) && in_ct && ct_clamp_res.hit_face && !inelastic_this_step && !ct_elastic_this_step) {
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

                    if (absolute_direction_z >= 1.0e-6F) {
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
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
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
                    if ((kProductionPrimaryPath || enable_voxel_scoring) && (!kProductionPrimaryPath && voxel_scorer_clamps_transport) &&
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

                    // Elastic endpoint: full bank uses target-specific TOPAS samples;
                    // the separate legacy diagnostic retains its explicit isotropic law.
                    // Supported recoils are queued; below-cutoff energy is scored locally.
                    if((!kProductionPrimaryPath && use_all_elastic) && ct_elastic_this_step && energy_MeV>energy_cutoff_MeV &&
                       all_elastic.rate(16,(!kProductionPrimaryPath && use_unified_water)?25:interaction_section,energy_MeV*inverse_mass_number)==0) {
                        ct_elastic_this_step=false;
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[10]).fetch_add(1);
                    }
                    if ((use_ct_elastic || (!kProductionPrimaryPath && use_all_elastic)) && ct_elastic_this_step &&
                        energy_MeV > energy_cutoff_MeV) {
                        const float u_tgt = rng::uniform01(
                            spot_seed, rng_history, steps, 12);
                        const auto draw = (!kProductionPrimaryPath && use_all_elastic) ? all_elastic.draw(16,
                            (!kProductionPrimaryPath && use_unified_water)?25:interaction_section,energy_MeV,direction_x,direction_y,direction_z,
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


                    if ((kProductionPrimaryPath || use_schneider_primary_xs) && (in_ct || (!kProductionPrimaryPath && use_unified_water)) && inelastic_this_step) {
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
                            if (unified_em) {
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
                            schneider_ct_device_ctx.c12_energy_nodes,
                            schneider_ct_device_ctx.c12_node_count,
                            schneider_ct_device_ctx.c12_event_offsets,
                            schneider_ct_device_ctx.c12_event_indices,
                            schneider_ct_device_ctx.c12_total_events,
                            6, 12, target_z,
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
                                6, 12, target_z,
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
                        const auto& event = schneider_ct_device_ctx.c12_interactions[primary_lookup.event_index];
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
                            if (prod_offset + ip >= schneider_ct_device_ctx.c12_total_products) break;
                            const auto& product = schneider_ct_device_ctx.c12_products[prod_offset + ip];

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
                                            minibeam_primary_c12_component_category *
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

                if (water_urban_seg_hist > 0 &&
                    water_urban_segment_count_device != nullptr) {
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        seg_count(water_urban_segment_count_device[0]);
                    seg_count.fetch_add(water_urban_seg_hist);
                }

                if(unified_em && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit(unified_audit,unified_primary_audit);

                const bool inside_phantom = !unified_primary_escaped_ct && (position_z_mm >= 0.0F && position_z_mm < phantom_length_mm);

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
                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV+history_water_electron_escaped_MeV;
                steps_device[global_history] = steps;
            });
        };
#if CARBON_PRIMARY_PRODUCTION_SPECIALIZE
        auto kernel_event = use_production_primary_path
            ? launch_primary_chunk(std::true_type{})
            : launch_primary_chunk(std::false_type{});
#else
        auto kernel_event = launch_primary_chunk(std::false_type{});
#endif
        primary_events.push_back(kernel_event);
    }
    // The queue is in order, so all primary chunks retain exactly the previous
    // execution order.  Waiting once on the final event avoids a host/device
    // round trip and a flushed progress line between every chunk.
    if (!primary_events.empty()) {
        primary_events.back().wait_and_throw();
        for (const auto& event : primary_events)
            primary_kernel_seconds += event_duration_seconds(event);
        std::cout << "[progress] primary batches completed: "
                  << number_of_histories << "/" << number_of_histories
                  << " histories (" << primary_events.size() << " launches)\n";
    }

#if defined(CARBON_ENABLE_MINIBEAM)
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
                                energy, cascade.z, cascade.a);
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
                                    midpoint_energy, cascade.z, cascade.a);
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
    const bool enable_secondary_unified_em = config.enable_secondary_unified_em;
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
                    float * untracked_nuclear_device;
                    float * ion_species_sp_device;
                    std::size_t table_size;
                    bool unified_em;
                    bool enable_secondary_unified_em;
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
                    bool use_ct_mass_sp;
                    float * ct_mass_sp_factor_lut_device;
                    std::uint32_t ct_n_mass_factors;
                    std::uint32_t * elastic_audit;
                    bool use_schneider_ion_sp;
                    float * schneider_ion_sp_device;
                    std::uint32_t * ion_stopping_failures;
                    bool use_ct_density_mass_spr;
                    std::uint32_t ct_density_spr_n_rho;
                    float ct_mass_spr_log_rho_min;
                    float ct_mass_spr_inv_dlog;
                    float maximum_step_mm;
                    unsigned int * unified_failure_count;
                    UnifiedEmFailureRecord * unified_failure_records;
                    float em_secondary_step_scale;
                    float depth_bin_width_mm;
                    bool ct_secondary_exact_faces;
                    bool ct_skip_homogeneous_face_clamp;
                    bool use_unified_water;
                    AllIonElasticView all_elastic;
                    bool enable_secondary_energy_straggling;
                    bool use_packaged_fluctuation;
                    float * fluct_energy_device;
                    std::size_t fluct_energy_count;
                    float * fluct_density_device;
                    std::size_t fluct_density_count;
                    float * fluct_probability_device;
                    std::size_t fluct_probability_count;
                    float * fluct_quantile_device;
                    std::array<float, max_straggling_scale_points> straggling_scale_energies;
                    std::array<float, max_straggling_scale_points> straggling_scale_values;
                    std::size_t straggling_scale_point_count;
                    float straggling_scale;
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
                    bool minibeam_water_secondary_c12_fermi_eyges_tail;
                    float minibeam_water_secondary_c12_mcs_max_segment_mm;
                    bool minibeam_water_secondary_c12_urban_v2;
                    float minibeam_water_primary_urban_max_step_mm;
                    float* minibeam_water_urban_loss_e_device;
                    float* minibeam_water_urban_loss_r_device;
                    float* minibeam_water_urban_loss_d_device;
                    std::uint32_t minibeam_water_urban_loss_count;
                    float minibeam_water_urban_zeff_f;
                    float minibeam_water_urban_radlen_mm_f;
                    bool minibeam_water_secondary_c12_enable_unified_em;
                    float minibeam_water_secondary_c12_post_sample_loss_scale;
                    std::uint64_t * minibeam_event_counts_device;
                    bool water_entry_secondary_replay;
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
                    untracked_nuclear_device,
                    ion_species_sp_device,
                    table_size,
                    unified_em,
                    enable_secondary_unified_em,
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
                    use_ct_mass_sp,
                    ct_mass_sp_factor_lut_device,
                    ct_n_mass_factors,
                    elastic_audit,
                    use_schneider_ion_sp,
                    schneider_ion_sp_device,
                    ion_stopping_failures,
                    use_ct_density_mass_spr,
                    ct_density_spr_n_rho,
                    ct_mass_spr_log_rho_min,
                    ct_mass_spr_inv_dlog,
                    maximum_step_mm,
                    unified_failure_count,
                    unified_failure_records,
                    em_secondary_step_scale,
                    depth_bin_width_mm,
                    ct_secondary_exact_faces,
                    ct_skip_homogeneous_face_clamp,
                    use_unified_water,
                    all_elastic,
                    enable_secondary_energy_straggling,
                    use_packaged_fluctuation,
                    fluct_energy_device,
                    fluct_energy_count,
                    fluct_density_device,
                    fluct_density_count,
                    fluct_probability_device,
                    fluct_probability_count,
                    fluct_quantile_device,
                    straggling_scale_energies,
                    straggling_scale_values,
                    straggling_scale_point_count,
                    straggling_scale,
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
                    minibeam_water_secondary_c12_fermi_eyges_tail,
                    minibeam_water_secondary_c12_mcs_max_segment_mm,
                    minibeam_water_secondary_c12_urban_v2,
                    minibeam_water_primary_urban_max_step_mm,
                    minibeam_water_urban_loss_e_device,
                    minibeam_water_urban_loss_r_device,
                    minibeam_water_urban_loss_d_device,
                    minibeam_water_urban_loss_count,
                    minibeam_water_urban_zeff_f,
                    minibeam_water_urban_radlen_mm_f,
                    minibeam_water_secondary_c12_enable_unified_em,
                    minibeam_water_secondary_c12_post_sample_loss_scale,
                    minibeam_event_counts_device,
                    water_entry_secondary_replay,
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
                    EmMode == 1 && enable_secondary_unified_em && enable_ct_grid &&
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
                        const int recoil_sp_idx=(kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_all_elastic))?CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).projectile(transport_z,transport_a):-1;
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
                        const float* ion_sp_table =
                            &CARBON_SECONDARY_CONTEXT_FIELD(ion_species_sp_device)[
                                static_cast<std::size_t>(charged_sp_idx<0?0:charged_sp_idx) * CARBON_SECONDARY_CONTEXT_FIELD(table_size)];

                        bool sec_terminal_recorded = false;
                        bool unified_secondary_escaped_ct = false;
                        ContinuousSpeciesTrackTally continuous_species_tally;
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

                        const bool c12_water_unified_secondary =
#if defined(CARBON_ENABLE_MINIBEAM)
                            !kProductionSecondaryPath &&
                            CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water) &&
                            CARBON_SECONDARY_CONTEXT_FIELD(
                                minibeam_water_secondary_c12_enable_unified_em) &&
                            transport_z == 6 && transport_a == 12;
#else
                            false;
#endif
                        const bool unified_secondary=
                            (EmMode < 0 ? CARBON_SECONDARY_CONTEXT_FIELD(unified_em)
                                        : EmMode == 1) &&
                            !generic_recoil &&
                            (kProductionSecondaryPath ? true :
                                (CARBON_SECONDARY_CONTEXT_FIELD(
                                     enable_secondary_unified_em) ||
                                 c12_water_unified_secondary));
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
                        constexpr uint32_t kSecondaryMaxSteps = 30000U;
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
                            unified_secondary_state=saved.unified_secondary_state;
                            unified_secondary_state.tables=&CARBON_SECONDARY_CONTEXT_FIELD(unified_device);
#if CARBON_EM_SEARCH_KEY_AUDIT
                            unified_secondary_state.search_audit=&unified_search_audit;
#endif
#if CARBON_EM_LOCAL_AUDIT
                            unified_secondary_audit=saved.unified_secondary_audit;
#endif
                            sec_steps=saved.sec_steps;
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

                            if (bin_z < 0 || bin_z >= static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) break;

#if defined(CARBON_ENABLE_MINIBEAM)
                            // A requested plane may coincide with the current
                            // step start (notably the water-exit endpoint after
                            // an exactly truncated preceding step). Record that
                            // state before the next EM/MCS draw instead of
                            // requiring a strict interior crossing or moving
                            // the diagnostic plane upstream by a few microns.
                            if ((kProductionSecondaryPath ? false :
                                 CARBON_SECONDARY_CONTEXT_FIELD(
                                     water_entry_secondary_replay)) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(
                                    minibeam_water_primary_plane_records_device) !=
                                    nullptr &&
                                sec_dz > 1.0e-6F) {
                                for (std::size_t plane = 0;
                                     plane < CARBON_SECONDARY_CONTEXT_FIELD(
                                                 minibeam_water_primary_plane_count);
                                     ++plane) {
                                    auto& record = CARBON_SECONDARY_CONTEXT_FIELD(
                                        minibeam_water_primary_plane_records_device)[
                                        frag.parent_history *
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_count) +
                                        plane];
                                    if (record.valid) continue;
                                    const auto plane_depth =
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_primary_plane_depths_device)[
                                            plane];
                                    const auto endpoint_tolerance =
                                        8.0F * std::numeric_limits<float>::epsilon() *
                                        sycl::fmax(1.0F, sycl::fabs(plane_depth));
                                    if (sycl::fabs(sec_z - plane_depth) >
                                        endpoint_tolerance) {
                                        continue;
                                    }
                                    const auto residual_path =
                                        (plane_depth - sec_z) / sec_dz;
                                    record.history = frag.parent_history;
                                    record.transport_path = 1U;
                                    record.particle_id = frag.rng_stream;
                                    record.rng_stream = frag.rng_stream;
                                    record.plane_index =
                                        static_cast<std::uint32_t>(plane);
                                    record.atomic_number =
                                        static_cast<std::int16_t>(transport_z);
                                    record.mass_number =
                                        static_cast<std::int16_t>(transport_a);
                                    record.depth_mm = plane_depth;
                                    record.kinetic_energy_MeV = sec_e;
                                    record.weight = frag.weight;
                                    record.x_mm = sec_x + residual_path * sec_dx;
                                    record.y_mm = sec_y + residual_path * sec_dy;
                                    record.direction_x = sec_dx;
                                    record.direction_y = sec_dy;
                                    record.direction_z = sec_dz;
                                    record.valid = 1U;
                                }
                            }
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
                            if (unified_secondary && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && !sec_in_ct) {
                                unified_secondary_escaped_ct = true;
                                break;
                            }
                            constexpr auto exposure_cell = std::numeric_limits<std::uint32_t>::max();
                            constexpr bool secondary_generation_eligible = false;
                            Cinel02DeviceRateLookup exposure_h_lookup{};
                            Cinel02DeviceRateLookup exposure_o_lookup{};
                            const bool exposure_h_covered = exposure_h_lookup.covered;
                            const bool exposure_o_covered = exposure_o_lookup.covered;
                            const bool exposure_rate_covered =
                                exposure_h_covered && exposure_o_covered;
                            const auto flt_idx = (sec_e_u - CARBON_SECONDARY_CONTEXT_FIELD(minimum_table_energy)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_table_step);
                            auto sp_idx = static_cast<int>(sycl::floor(flt_idx));
                            sp_idx = sycl::max(0, sycl::min(sp_idx, static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(table_size)) - 2));
                            const auto sp_frac = sycl::clamp(flt_idx - static_cast<float>(sp_idx), 0.0F, 1.0F);
                            auto sec_sp = (ion_sp_table[sp_idx] +
                                           sp_frac * (ion_sp_table[sp_idx + 1] -
                                                      ion_sp_table[sp_idx]));
                            const bool sec_use_mass_sp_factor =
                                (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct && CARBON_SECONDARY_CONTEXT_FIELD(use_ct_mass_sp) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_sp_factor_lut_device) != nullptr &&
                                CARBON_SECONDARY_CONTEXT_FIELD(ct_n_mass_factors) > 0U;
                            float sec_material_factor = 1.0F;
                            // Enabled bank is mandatory inside CT. A failed query aborts
                            // the run at readback; never substitute a water/material factor.
                            auto scheme2_linear_sp = [&](float energy_mevu) {
                                if(generic_recoil) {
                                    const float v=CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).stopping(recoil_sp_idx,sec_in_ct?sec_ct_material:25,energy_mevu);
                                    if(!(v>0)){sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(CARBON_SECONDARY_CONTEXT_FIELD(elastic_audit)[6]).fetch_add(1);}
                                    return v*sec_local_density_g_per_cm3;
                                }
                                if (!(CARBON_SECONDARY_CONTEXT_FIELD(use_schneider_ion_sp) && sec_in_ct)) return -1.0F;
                                const float unit_sp = schneider_ion_stopping_lookup(
                                    CARBON_SECONDARY_CONTEXT_FIELD(schneider_ion_sp_device), static_cast<std::size_t>(charged_sp_idx),
                                    static_cast<std::size_t>(sec_ct_material), energy_mevu);
                                if (!(unit_sp > 0) || !sycl::isfinite(sec_local_density_g_per_cm3) ||
                                    !(sec_local_density_g_per_cm3 > 0)) {
                                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>
                                        failures(*CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures));
                                    const auto failure_index = failures.fetch_add(1);
                                    if (failure_index == 0) {
                                        CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures)[1] = transport_z;
                                        CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures)[2] = transport_a;
                                        CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures)[3] = sec_ct_material;
                                        CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures)[4] = sycl::bit_cast<std::uint32_t>(energy_mevu);
                                        CARBON_SECONDARY_CONTEXT_FIELD(ion_stopping_failures)[5] = sycl::bit_cast<std::uint32_t>(sec_local_density_g_per_cm3);
                                    }
                                    return -1.0F;
                                }
                                return unit_sp * sec_local_density_g_per_cm3;
                            };
                            bool sec_scheme2_hit = false;
                            {
                                const float scheme2_sp = scheme2_linear_sp(sec_e_u);
                                if ((generic_recoil||(CARBON_SECONDARY_CONTEXT_FIELD(use_schneider_ion_sp) && sec_in_ct)) && !(scheme2_sp > 0)) break;
                                if (scheme2_sp > 0.0F) {
                                    sec_sp = scheme2_sp;
                                    sec_scheme2_hit = true;
                                }
                            }
                            if (sec_use_mass_sp_factor && !sec_scheme2_hit) {
                                sec_material_factor = ct_lookup_mass_sp_factor(
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_sp_factor_lut_device),
                                    CARBON_SECONDARY_CONTEXT_FIELD(use_ct_density_mass_spr) ? CARBON_SECONDARY_CONTEXT_FIELD(ct_density_spr_n_rho)
                                                            : CARBON_SECONDARY_CONTEXT_FIELD(ct_n_mass_factors),
                                    CARBON_SECONDARY_CONTEXT_FIELD(table_size), CARBON_SECONDARY_CONTEXT_FIELD(use_ct_density_mass_spr),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_spr_log_rho_min), CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_spr_inv_dlog),
                                    static_cast<std::uint32_t>(sec_ct_material),
                                    sec_local_density_g_per_cm3,
                                    static_cast<std::size_t>(sp_idx), sp_frac,
                                    [](float x) { return sycl::log(x); });
                            }
                            // Scheme-2 value is already linear stopping at
                            // local density; bypass water x density scaling.
                            if (!sec_scheme2_hit) {
                                sec_sp = secondary_material_stopping_power(
                                    sec_sp, (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct,
                                    sec_local_density_g_per_cm3,
                                    sec_use_mass_sp_factor, sec_material_factor);
                            }

                            if (sec_sp <= 1.0e-6F) break;

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
                                sec_step_mm=sycl::fmin((CARBON_SECONDARY_CONTEXT_FIELD(em_secondary_step_scale)!=1.f?unified_secondary_state.research_step(sec_e,CARBON_EM_STEP_CACHE?unified_secondary_pre:unified_secondary_state.prepare(sec_e),CARBON_SECONDARY_CONTEXT_FIELD(em_secondary_step_scale)):(CARBON_EM_STEP_CACHE?unified_secondary_state.step(unified_secondary_pre):unified_secondary_state.step(sec_e))),unified_secondary_distance);
                                // Bound combined mean loss, not the old restricted range alone.
                                const float delta_sp=unified_secondary_state.mix(unified_secondary_pre.lo.delta_stopping,unified_secondary_pre.hi.delta_stopping);
                                if(delta_sp>0) sec_step_mm=sycl::fmin(sec_step_mm,.01f*sec_e/sycl::fmax(1e-12f,delta_sp+unified_secondary_state.mix(unified_secondary_pre.lo.stopping,unified_secondary_pre.hi.stopping)));
                            }

                            bool depth_boundary_limited = false;
                            if (sec_dz > 1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z + 1) * CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm);
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            } else if (sec_dz < -1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z) * CARBON_SECONDARY_CONTEXT_FIELD(depth_bin_width_mm);
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 0.0F && dz_step <= sec_step_mm) {
                                    sec_step_mm = dz_step;
                                    depth_boundary_limited = true;
                                }
                            }
                            CtFaceClampResult sec_face_clamp{sec_step_mm,false,0};
                            if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces))) {
                                sec_face_clamp = clamp_step_to_ct_faces_exact(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_origin_z),
                                    CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_x), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_y), CARBON_SECONDARY_CONTEXT_FIELD(ct_spacing_z), CARBON_SECONDARY_CONTEXT_FIELD(ct_nx), CARBON_SECONDARY_CONTEXT_FIELD(ct_ny), CARBON_SECONDARY_CONTEXT_FIELD(ct_nz));
                                sec_step_mm = sec_face_clamp.step_mm;
                            } else if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct) {
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
                            if(generic_recoil)sec_step_mm=sycl::fmin(sec_step_mm,0.05F*sec_e/sec_sp);
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
                               (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_all_elastic)) && !generic_recoil && (sec_in_ct || (kProductionSecondaryPath ? false : CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)))) {
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
                            if(secondary_inelastic)schneider_diag_increment_device(CARBON_SECONDARY_CONTEXT_FIELD(schneider_diag_device),SchneiderDiagSlot::SecondaryHazards);

                            // Record the actual charged-track exposure after any
                            // collision truncation.  Coverage is evaluated at the
                            // same step-start energy used by the runtime hazard;
                            // generation-blocked path is kept separate and never
                            // folded into an apparent uncovered rate segment.
                            if (exposure_cell != std::numeric_limits<std::uint32_t>::max()) {
                                if (secondary_generation_eligible) {
                                    if (exposure_rate_covered) {
                                    } else {
                                    }
                                    if (!exposure_h_covered) {
                                    }
                                    if (!exposure_o_covered) {
                                    }
                                    if (exposure_h_covered) {
                                    }
                                    if (exposure_o_covered) {
                                    }
                                    if (exposure_rate_covered) {
                                    }
                                } else {
                                    // Counterfactual hazard for this blocked segment.
                                    // Keep H/O separate when only one target is covered;
                                    // total is defined only for a complete H/O pair, exactly
                                    // as in the runtime target selector.
                                    if (exposure_h_covered) {
                                    }
                                    if (exposure_o_covered) {
                                    }
                                    if (exposure_rate_covered) {
                                    }
                                }
                            }

                            // Midpoint loss
                            const auto mid_e_u = sycl::fmax(generic_recoil?CARBON_SECONDARY_CONTEXT_FIELD(recoil_stopping).energies[0]:0.01F, (sec_e - 0.5F * sec_sp * sec_step_mm) * frag_inv_a);
                            const auto mid_flt = (mid_e_u - CARBON_SECONDARY_CONTEXT_FIELD(minimum_table_energy)) * CARBON_SECONDARY_CONTEXT_FIELD(inverse_table_step);
                            auto mid_idx = static_cast<int>(sycl::floor(mid_flt));
                            mid_idx = sycl::max(0, sycl::min(mid_idx, static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(table_size)) - 2));
                            const auto mid_fr = sycl::clamp(mid_flt - static_cast<float>(mid_idx), 0.0F, 1.0F);
                            // Unified EM overwrites dE below: keep only the
                            // scheme2 break check, skip the dead midpoint
                            // stopping evaluation.
                            auto mid_sp = 0.0F;
                            // Query the same material bank at the predicted midpoint.
                            // Already linear at local density: bypass the
                            // water x density scaling entirely.
                            bool mid_scheme2_hit = false;
                            {
                                const float scheme2_mid = scheme2_linear_sp(mid_e_u);
                                if ((generic_recoil||(CARBON_SECONDARY_CONTEXT_FIELD(use_schneider_ion_sp) && sec_in_ct)) && !(scheme2_mid > 0)) break;
                                if (scheme2_mid > 0.0F) {
                                    mid_sp = scheme2_mid;
                                    mid_scheme2_hit = true;
                                }
                            }
                            if (!unified_secondary) {
                                mid_sp = (ion_sp_table[mid_idx] +
                                          mid_fr * (ion_sp_table[mid_idx + 1] -
                                                    ion_sp_table[mid_idx]));
                                // The midpoint value drives dE and must use the same
                                // Schneider density/material scaling as sec_sp at the
                                // step start. Previously this remained a density-1
                                // water value, over-stopping secondaries by ~1/rho
                                // (about 25x in the RT06423 air section).
                                float mid_material_factor = 1.0F;
                                if (sec_use_mass_sp_factor) {
                                    mid_material_factor = ct_lookup_mass_sp_factor(
                                        CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_sp_factor_lut_device),
                                        CARBON_SECONDARY_CONTEXT_FIELD(use_ct_density_mass_spr) ? CARBON_SECONDARY_CONTEXT_FIELD(ct_density_spr_n_rho)
                                                                : CARBON_SECONDARY_CONTEXT_FIELD(ct_n_mass_factors),
                                        CARBON_SECONDARY_CONTEXT_FIELD(table_size), CARBON_SECONDARY_CONTEXT_FIELD(use_ct_density_mass_spr),
                                        CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_spr_log_rho_min), CARBON_SECONDARY_CONTEXT_FIELD(ct_mass_spr_inv_dlog),
                                        static_cast<std::uint32_t>(sec_ct_material),
                                        sec_local_density_g_per_cm3,
                                        static_cast<std::size_t>(mid_idx), mid_fr,
                                        [](float x) { return sycl::log(x); });
                                }
                            if (!mid_scheme2_hit) {
                                mid_sp = secondary_material_stopping_power(
                                    mid_sp, (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct,
                                    sec_local_density_g_per_cm3,
                                    sec_use_mass_sp_factor, mid_material_factor);
                            }
                            }

                            float dE = sycl::fmin(mid_sp * sec_step_mm, sec_e);
                            if (!unified_secondary && CARBON_SECONDARY_CONTEXT_FIELD(enable_secondary_energy_straggling) &&
                                CARBON_SECONDARY_CONTEXT_FIELD(use_packaged_fluctuation) && transport_z == 6 && transport_a == 12) {
                                const auto u_loss = rng::uniform01(
                                    2026, frag.rng_stream, sec_steps, 2);
                                const auto ratio = sample_energy_loss_ratio_from_grid(
                                    CARBON_SECONDARY_CONTEXT_FIELD(fluct_energy_device), CARBON_SECONDARY_CONTEXT_FIELD(fluct_energy_count),
                                    CARBON_SECONDARY_CONTEXT_FIELD(fluct_density_device), CARBON_SECONDARY_CONTEXT_FIELD(fluct_density_count),
                                    CARBON_SECONDARY_CONTEXT_FIELD(fluct_probability_device), CARBON_SECONDARY_CONTEXT_FIELD(fluct_probability_count),
                                    CARBON_SECONDARY_CONTEXT_FIELD(fluct_quantile_device), sec_e_u,
                                    CARBON_SECONDARY_CONTEXT_FIELD(water_density_g_per_cm3) * sec_step_mm / 10.0F, u_loss);
                                const auto local_scale = interpolate_straggling_scale(
                                    sec_e_u, CARBON_SECONDARY_CONTEXT_FIELD(straggling_scale_energies),
                                    CARBON_SECONDARY_CONTEXT_FIELD(straggling_scale_values),
                                    CARBON_SECONDARY_CONTEXT_FIELD(straggling_scale_point_count), CARBON_SECONDARY_CONTEXT_FIELD(straggling_scale));
                                const auto scaled_ratio =
                                    scale_energy_loss_ratio_preserving_mean(ratio, local_scale);
                                dE = sycl::clamp(dE * scaled_ratio, 0.0F, sec_e);
                            }

                            // CINEL02 packages are captured at hadronic PostStepDoIt:
                            // advance the continuous EM state to the collision point
                            // before selecting and replaying the final state.
                            const auto collision_input_dx = sec_dx;
                            const auto collision_input_dy = sec_dy;
                            const auto collision_input_dz = sec_dz;
                            if(unified_secondary){
                                auto draw=unified_em_loss(unified_secondary_state,unified_secondary_clock,sec_e,sec_step_mm,unified_secondary_rate,unified_secondary_distance,CARBON_SECONDARY_CONTEXT_FIELD(enable_secondary_energy_straggling),unified_secondary_pre,unified_secondary_uniform);
                                if(!draw.valid){
                                    record_unified_em_failure(CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_count),CARBON_SECONDARY_CONTEXT_FIELD(unified_failure_records),40+static_cast<int>(draw.failure_stage),unified_secondary_state.section,transport_z,transport_a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);unified_secondary_count(7);break;}
                                dE=draw.loss;unified_secondary_count(2);unified_secondary_count(3,draw.proposed);unified_secondary_count(4,draw.accepted);
                                unified_secondary_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_secondary_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                            }
#if defined(CARBON_ENABLE_MINIBEAM)
                            if ((kProductionSecondaryPath ? false :
                                 CARBON_SECONDARY_CONTEXT_FIELD(
                                     minibeam_water_secondary_c12_post_sample_loss_scale)) != 1.0F &&
                                transport_z == 6 && transport_a == 12 &&
                                (kProductionSecondaryPath ? false :
                                 CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) &&
                                !sec_in_ct) {
                                const auto raw_loss = dE;
                                dE = sycl::fmin(
                                    sec_e,
                                    raw_loss * CARBON_SECONDARY_CONTEXT_FIELD(
                                                   minibeam_water_secondary_c12_post_sample_loss_scale));
                                auto add_loss_counter = [&](std::size_t slot,
                                                            std::uint64_t value) {
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_event_counts_device)[slot])
                                        .fetch_add(value);
                                };
                                add_loss_counter(
                                    minibeam_secondary_c12_raw_loss_micro_slot,
                                    static_cast<std::uint64_t>(raw_loss * 1.0e6F + 0.5F));
                                add_loss_counter(
                                    minibeam_secondary_c12_scaled_loss_micro_slot,
                                    static_cast<std::uint64_t>(dE * 1.0e6F + 0.5F));
                                add_loss_counter(
                                    minibeam_secondary_c12_scaled_loss_step_slot, 1U);
                            }
#endif
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
                            if ((kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces)) && sec_face_clamp.hit_face && !secondary_inelastic && !secondary_elastic) {
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
                            const auto continuous_mid_e_u = sycl::fmax(
                                0.0F, (sec_e - 0.5F * dE) * frag_inv_a);
                            const auto continuous_end_e_u = sycl::fmax(
                                0.0F, post_em_e * frag_inv_a);
                            Cinel02DeviceRateLookup continuous_h_mid{};
                            Cinel02DeviceRateLookup continuous_o_mid{};
                            Cinel02DeviceRateLookup continuous_h_end{};
                            Cinel02DeviceRateLookup continuous_o_end{};
                            const bool continuous_rate_covered =
                                exposure_h_lookup.covered && exposure_o_lookup.covered &&
                                continuous_h_mid.covered && continuous_o_mid.covered &&
                                continuous_h_end.covered && continuous_o_end.covered;
                            if (exposure_cell != std::numeric_limits<std::uint32_t>::max()) {
                                if (continuous_rate_covered) {
                                    const auto lambda_start =
                                        exposure_h_lookup.value_per_mm +
                                        exposure_o_lookup.value_per_mm;
                                    const auto lambda_mid =
                                        continuous_h_mid.value_per_mm +
                                        continuous_o_mid.value_per_mm;
                                    const auto lambda_end =
                                        continuous_h_end.value_per_mm +
                                        continuous_o_end.value_per_mm;
                                    const auto tau_continuous =
                                        cinel02_simpson_hazard(
                                            lambda_start, lambda_mid, lambda_end,
                                            sec_step_mm);
                                } else {
                                }
                            }
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
                                (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(ct_secondary_exact_faces)) && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_ct_grid)) && sec_in_ct && (kProductionSecondaryPath ? true : CARBON_SECONDARY_CONTEXT_FIELD(enable_voxel_scoring));
                            int continuous_step_voxel = -1;
                            if (source_voxel_continuous_loss) {
                                const int sx=static_cast<int>(sycl::floor((sec_x-CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_x_mm))*CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_x_mm)));
                                const int sy=static_cast<int>(sycl::floor((sec_y-CARBON_SECONDARY_CONTEXT_FIELD(voxel_min_y_mm))*CARBON_SECONDARY_CONTEXT_FIELD(inverse_voxel_size_y_mm)));
                                if (sx>=0 && sy>=0 && bin_z>=0 && sx<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x)) &&
                                    sy<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y)) && bin_z<static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(number_of_bins))) {
                                    continuous_step_voxel=(bin_z*static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_y))+sy)*static_cast<int>(CARBON_SECONDARY_CONTEXT_FIELD(voxel_bins_x))+sx;
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
                                            if (product.role == 0 && product.z > 0 &&
                                                product.a > 0 &&
                                                cinel02_should_topas_compat_kill(
                                                    cinel02_topas_compatibility_mode,
                                                    product.z, product.a)) {
                                                // Match TOPAS/Geant4's unsupported prompt-ion
                                                // fallback: generated, then killed before queue,
                                                // with no daughter and no local deposit.
                                                continue;
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
                            if (!secondary_inelastic) {
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                            }

                            if (cinel02_should_apply_secondary_mcs(
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
                                const bool secondary_uses_fe =
                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                        c12_fermi_eyges) &&
                                    ion_uses_fermi_eyges(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            fermi_eyges_species_scope),
                                        transport_z, transport_a);
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
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(
                                             water_entry_secondary_replay)) &&
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_primary_plane_records_device) != nullptr &&
                                        collision_input_dz > 0.0F) {
                                        for (std::size_t plane = 0;
                                             plane < CARBON_SECONDARY_CONTEXT_FIELD(
                                                 minibeam_water_primary_plane_count);
                                             ++plane) {
                                            auto& candidate =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_records_device)[
                                                    frag.parent_history *
                                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                                            minibeam_water_primary_plane_count) +
                                                    plane];
                                            const auto plane_depth =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_depths_device)[plane];
                                            if (!candidate.valid &&
                                                fe_start_z < plane_depth &&
                                                post_em_z >= plane_depth) {
                                                observation_plane = plane;
                                                observation_path_mm = sycl::clamp(
                                                    (plane_depth - fe_start_z) /
                                                        collision_input_dz,
                                                    1.0e-6F,
                                                    sec_step_mm - 1.0e-6F);
                                                break;
                                            }
                                        }
                                    }
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
                                                fermi_eyges_use_species_water_parameters));
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
#if defined(CARBON_ENABLE_MINIBEAM)
                                else if (
                                    (kProductionSecondaryPath ? false :
                                     CARBON_SECONDARY_CONTEXT_FIELD(
                                         minibeam_water_secondary_c12_fermi_eyges_tail)) &&
                                    transport_z == 6 && transport_a == 12 &&
                                    (kProductionSecondaryPath ? false :
                                     CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) &&
                                    !sec_in_ct) {
                                    sycl::atomic_ref<std::uint64_t,
                                                     sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_event_counts_device)[
                                            minibeam_secondary_c12_fe_step_slot])
                                        .fetch_add(1U);
                                    auto segment_x = post_em_x -
                                        collision_input_dx * sec_step_mm;
                                    auto segment_y = post_em_y -
                                        collision_input_dy * sec_step_mm;
                                    auto segment_z = post_em_z -
                                        collision_input_dz * sec_step_mm;
                                    auto segment_direction = Direction3F{
                                        collision_input_dx, collision_input_dy,
                                        collision_input_dz};
                                    auto traversed_mm = 0.0F;
                                    std::uint32_t segment_index = 0U;
                                    while (traversed_mm < sec_step_mm) {
                                        sycl::atomic_ref<std::uint64_t,
                                                         sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>(
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_event_counts_device)[
                                                minibeam_secondary_c12_fe_segment_slot])
                                            .fetch_add(1U);
                                        const auto segment_mm = sycl::fmin(
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_secondary_c12_mcs_max_segment_mm),
                                            sec_step_mm - traversed_mm);
                                        std::size_t observation_plane =
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_count);
                                        auto observation_fraction = -1.0F;
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(
                                                 water_entry_secondary_replay)) &&
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_records_device) != nullptr &&
                                            segment_direction.z > 0.0F) {
                                            const auto segment_end_z = segment_z +
                                                segment_direction.z * segment_mm;
                                            for (std::size_t plane = 0;
                                                 plane < CARBON_SECONDARY_CONTEXT_FIELD(
                                                     minibeam_water_primary_plane_count);
                                                 ++plane) {
                                                auto& candidate =
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_water_primary_plane_records_device)[
                                                        frag.parent_history *
                                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                                minibeam_water_primary_plane_count) +
                                                        plane];
                                                const auto plane_depth =
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_water_primary_plane_depths_device)[plane];
                                                if (!candidate.valid &&
                                                    segment_z < plane_depth &&
                                                    segment_end_z >= plane_depth) {
                                                    observation_plane = plane;
                                                    observation_fraction = sycl::clamp(
                                                        (plane_depth - segment_z) /
                                                            (segment_end_z - segment_z),
                                                        1.0e-6F, 1.0F - 1.0e-6F);
                                                    break;
                                                }
                                            }
                                        }
                                        const auto energy_fraction =
                                            (traversed_mm + 0.5F * segment_mm) /
                                            sec_step_mm;
                                        const auto correlated =
                                            water_c12_fermi_eyges_tail_step(
                                                segment_direction,
                                                sycl::fmax(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        energy_cutoff_MeV),
                                                    sec_e +
                                                        (1.0F - energy_fraction) * dE),
                                                segment_mm,
                                                sec_local_density_g_per_cm3,
                                                sec_radiation_length_g_per_cm2,
                                                2026, frag.rng_stream,
                                                static_cast<std::uint64_t>(sec_steps) *
                                                        1024U +
                                                    segment_index,
                                                100U, observation_fraction);
                                        if (correlated.observation_valid &&
                                            observation_plane <
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_count)) {
                                            const auto observation_path =
                                                observation_fraction * segment_mm;
                                            auto crossing_x = segment_x +
                                                segment_direction.x * observation_path +
                                                correlated.observation_displacement_mm.x;
                                            auto crossing_y = segment_y +
                                                segment_direction.y * observation_path +
                                                correlated.observation_displacement_mm.y;
                                            auto crossing_z = segment_z +
                                                segment_direction.z * observation_path +
                                                correlated.observation_displacement_mm.z;
                                            const auto plane_depth =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_depths_device)[
                                                    observation_plane];
                                            if (correlated.observation_direction.z > 1.0e-6F) {
                                                const auto residual =
                                                    (plane_depth - crossing_z) /
                                                    correlated.observation_direction.z;
                                                crossing_x += residual *
                                                    correlated.observation_direction.x;
                                                crossing_y += residual *
                                                    correlated.observation_direction.y;
                                                crossing_z = plane_depth;
                                            }
                                            auto& record =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_records_device)[
                                                    frag.parent_history *
                                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                                            minibeam_water_primary_plane_count) +
                                                    observation_plane];
                                            record.history = frag.parent_history;
                                            record.transport_path = 1U;
                                            record.particle_id = frag.rng_stream;
                                            record.rng_stream = frag.rng_stream;
                                            record.plane_index = static_cast<std::uint32_t>(
                                                observation_plane);
                                            record.atomic_number = static_cast<std::int16_t>(transport_z);
                                            record.mass_number = static_cast<std::int16_t>(transport_a);
                                            record.depth_mm = plane_depth;
                                            const auto total_fraction = sycl::clamp(
                                                (traversed_mm + observation_path) /
                                                    sec_step_mm,
                                                0.0F, 1.0F);
                                            record.kinetic_energy_MeV = sycl::fmax(
                                                0.0F,
                                                sec_e + (1.0F - total_fraction) * dE);
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
                                        segment_x += segment_direction.x * segment_mm +
                                            correlated.displacement_mm.x;
                                        segment_y += segment_direction.y * segment_mm +
                                            correlated.displacement_mm.y;
                                        segment_z += segment_direction.z * segment_mm +
                                            correlated.displacement_mm.z;
                                        segment_direction = correlated.direction;
                                        traversed_mm += segment_mm;
                                        ++segment_index;
                                    }
                                    sec_x = segment_x;
                                    sec_y = segment_y;
                                    sec_z = segment_z;
                                    sec_dx = segment_direction.x;
                                    sec_dy = segment_direction.y;
                                    sec_dz = segment_direction.z;
                                }
#endif
#if defined(CARBON_ENABLE_MINIBEAM)
                                else if (
                                    (kProductionSecondaryPath ? false :
                                     CARBON_SECONDARY_CONTEXT_FIELD(
                                         minibeam_water_secondary_c12_urban_v2)) &&
                                    transport_z == 6 && transport_a == 12 &&
                                    (kProductionSecondaryPath ? false :
                                     CARBON_SECONDARY_CONTEXT_FIELD(use_unified_water)) &&
                                    !sec_in_ct) {
                                     // Research-only secondary-C12 Urban
                                     // (Geant4-11.3.2, Water_75eV table shared
                                     // with the primary path). Scope, RNG and
                                     // stepping mirror the primary urban
                                     // branch; differences are documented:
                                     // - at_boundary is FALSE on every step
                                     //   (fix B6): a newborn reference track
                                     //   starts non-boundary, keeping
                                     //   tlimit at its non-binding init
                                     //   instead of recomputing 0.2*range
                                     //   per macro step. Persistent
                                     //   cross-resume MSC state in the queue
                                     //   record remains a follow-up (tlimit
                                     //   is non-binding except at range end;
                                     //   revisit if end-of-range sensitivity
                                     //   appears).
                                    // - RNG dims 110+ (FE uses 100+; primary
                                    //   urban uses 70+; 58/59 shared for the
                                    //   tlimit draw on independent streams).
                                    // - Plane records use the primary-urban
                                    //   endpoint projection (observation-only);
                                    //   no production default change.
                                    sycl::atomic_ref<std::uint64_t,
                                                     sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>(
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_event_counts_device)[
                                            minibeam_water_secondary_c12_urban_step_slot])
                                        .fetch_add(1U);
                                    UrbanV2LossTable sec_urban_table{
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_urban_loss_e_device),
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_urban_loss_r_device),
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_urban_loss_d_device),
                                        static_cast<int>(
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_urban_loss_count))};
                                    auto segment_x = post_em_x -
                                        collision_input_dx * sec_step_mm;
                                    auto segment_y = post_em_y -
                                        collision_input_dy * sec_step_mm;
                                    auto segment_z = post_em_z -
                                        collision_input_dz * sec_step_mm;
                                    auto segment_direction = Direction3F{
                                        collision_input_dx, collision_input_dy,
                                        collision_input_dz};
                                    auto traversed_mm = 0.0F;
                                    std::uint32_t segment_index = 0U;
                                    UrbanV2TrackState sec_urban_state{};
                                    while (traversed_mm < sec_step_mm) {
                                        if (segment_index >= 1000000U) {
                                            // Fix B5: cap trip raises fatal
                                            // (see primary branch comment).
                                            if (CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_event_counts_device) !=
                                                nullptr) {
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::
                                                        global_space>(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_event_counts_device)
                                                        [minibeam_water_urban_subdiv_cap_slot])
                                                    .fetch_add(1U);
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::
                                                        global_space>(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_event_counts_device)
                                                        [minibeam_water_urban_fatal_slot])
                                                    .fetch_add(1U);
                                            }
                                            break;
                                        }
                                        const auto segment_mm = sycl::fmin(
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_urban_max_step_mm),
                                            sec_step_mm - traversed_mm);
                                        // Fix B6: segment-START energy (Epre
                                        // semantics, mirroring the primary
                                        // branch and the reference
                                        // SampleScattering input). The old
                                        // midpoint form double-counts the
                                        // sampler's internal energy
                                        // prediction.
                                        const auto energy_fraction =
                                            traversed_mm / sec_step_mm;
                                        const auto seg_e = sycl::fmax(
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                energy_cutoff_MeV),
                                            sec_e +
                                                (1.0F - energy_fraction) * dE);
                                        const auto urban_scatter =
                                            water_urban_v2_propose_and_sample(
                                                segment_direction, seg_e, 6, 12,
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_urban_max_step_mm),
                                                segment_mm,
                                                segment_x, segment_y, segment_z,
                                                segment_direction.x,
                                                segment_direction.y,
                                                segment_direction.z,
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    voxel_min_x_mm),
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    voxel_max_x_mm),
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    voxel_min_y_mm),
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    voxel_max_y_mm),
                                                0.0F,
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    phantom_length_mm),
                                                false, sec_urban_state,
                                                sec_urban_table,
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_urban_zeff_f),
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_urban_radlen_mm_f),
                                                1.0F, 2026, frag.rng_stream,
                                                static_cast<std::uint64_t>(
                                                        sec_steps) *
                                                        1024U +
                                                    segment_index,
                                                110U);
                                        // Fix B5: invalid/zero-progress ->
                                        // fatal + stop (see primary branch).
                                        if (!urban_scatter.proposal_valid ||
                                            !(urban_scatter
                                                  .final_geom_path_mm >
                                              0.0F)) {
                                            if (CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_event_counts_device) !=
                                                nullptr) {
                                                sycl::atomic_ref<
                                                    std::uint64_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::
                                                        global_space>(
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_event_counts_device)
                                                        [minibeam_water_urban_fatal_slot])
                                                    .fetch_add(1U);
                                            }
                                            break;
                                        }
                                        const auto seg_end_x = segment_x +
                                            segment_direction.x *
                                                urban_scatter
                                                    .final_geom_path_mm +
                                            urban_scatter.displacement_mm.x;
                                        const auto seg_end_y = segment_y +
                                            segment_direction.y *
                                                urban_scatter
                                                    .final_geom_path_mm +
                                            urban_scatter.displacement_mm.y;
                                        const auto seg_end_z = segment_z +
                                            segment_direction.z *
                                                urban_scatter
                                                    .final_geom_path_mm +
                                            urban_scatter.displacement_mm.z;
                                        // Observation-only plane records
                                        // (endpoint projection, never alters
                                        // transport), mirroring the primary
                                        // urban branch.
                                        if ((kProductionSecondaryPath ? false :
                                             CARBON_SECONDARY_CONTEXT_FIELD(
                                                 water_entry_secondary_replay)) &&
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_records_device) !=
                                                nullptr &&
                                            urban_scatter.direction.z >
                                                0.0F) {
                                            for (std::size_t plane = 0;
                                                 plane <
                                                 CARBON_SECONDARY_CONTEXT_FIELD(
                                                     minibeam_water_primary_plane_count);
                                                 ++plane) {
                                                auto& candidate =
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_water_primary_plane_records_device)[
                                                        frag.parent_history *
                                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                                minibeam_water_primary_plane_count) +
                                                        plane];
                                                const auto plane_depth =
                                                    CARBON_SECONDARY_CONTEXT_FIELD(
                                                        minibeam_water_primary_plane_depths_device)[plane];
                                                if (!candidate.valid &&
                                                    segment_z <
                                                        plane_depth &&
                                                    seg_end_z >= plane_depth) {
                                                    const auto residual_path =
                                                        (plane_depth -
                                                         seg_end_z) /
                                                        urban_scatter
                                                            .direction.z;
                                                    const auto step_fraction =
                                                        sycl::clamp(
                                                            (traversed_mm +
                                                             urban_scatter
                                                                 .final_geom_path_mm) /
                                                                sec_step_mm,
                                                            0.0F, 1.0F);
                                                    candidate.history =
                                                        frag.parent_history;
                                                    candidate.transport_path =
                                                        1U;
                                                    candidate.particle_id =
                                                        frag.rng_stream;
                                                    candidate.rng_stream =
                                                        frag.rng_stream;
                                                    candidate.plane_index =
                                                        static_cast<std::uint32_t>(
                                                            plane);
                                                    candidate.atomic_number =
                                                        static_cast<std::int16_t>(
                                                            transport_z);
                                                    candidate.mass_number =
                                                        static_cast<std::int16_t>(
                                                            transport_a);
                                                    candidate.depth_mm =
                                                        plane_depth;
                                                    candidate
                                                        .kinetic_energy_MeV =
                                                        sycl::fmax(
                                                            0.0F,
                                                            sec_e +
                                                                (1.0F -
                                                                 step_fraction) *
                                                                    dE);
                                                    candidate.weight =
                                                        frag.weight;
                                                    candidate.x_mm =
                                                        seg_end_x +
                                                        residual_path *
                                                            urban_scatter
                                                                .direction.x;
                                                    candidate.y_mm =
                                                        seg_end_y +
                                                        residual_path *
                                                            urban_scatter
                                                                .direction.y;
                                                    candidate.direction_x =
                                                        urban_scatter
                                                            .direction.x;
                                                    candidate.direction_y =
                                                        urban_scatter
                                                            .direction.y;
                                                    candidate.direction_z =
                                                        urban_scatter
                                                            .direction.z;
                                                    candidate.valid = 1U;
                                                    break;
                                                }
                                            }
                                        }
                                        segment_x = seg_end_x;
                                        segment_y = seg_end_y;
                                        segment_z = seg_end_z;
                                        segment_direction =
                                            urban_scatter.direction;
                                        traversed_mm += urban_scatter
                                            .final_geom_path_mm;
                                        ++segment_index;
                                    }
                                    sec_x = segment_x;
                                    sec_y = segment_y;
                                    sec_z = segment_z;
                                    sec_dx = segment_direction.x;
                                    sec_dy = segment_direction.y;
                                    sec_dz = segment_direction.z;
                                }
#endif
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
#if defined(CARBON_ENABLE_MINIBEAM)
                                    if ((kProductionSecondaryPath ? false :
                                         CARBON_SECONDARY_CONTEXT_FIELD(
                                             water_entry_secondary_replay)) &&
                                        transport_z == 6 && transport_a == 12 &&
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_primary_plane_records_device) != nullptr &&
                                        collision_input_dz > 0.0F) {
                                        const auto start_x = post_em_x -
                                            collision_input_dx * sec_step_mm;
                                        const auto start_y = post_em_y -
                                            collision_input_dy * sec_step_mm;
                                        const auto start_z = post_em_z -
                                            collision_input_dz * sec_step_mm;
                                        for (std::size_t plane = 0;
                                             plane < CARBON_SECONDARY_CONTEXT_FIELD(
                                                 minibeam_water_primary_plane_count);
                                             ++plane) {
                                            const auto plane_depth =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_depths_device)[plane];
                                            if (!(start_z < plane_depth &&
                                                  post_em_z >= plane_depth)) continue;
                                            auto& record =
                                                CARBON_SECONDARY_CONTEXT_FIELD(
                                                    minibeam_water_primary_plane_records_device)[
                                                    frag.parent_history *
                                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                                            minibeam_water_primary_plane_count) +
                                                    plane];
                                            if (record.valid) continue;
                                            const auto fraction = sycl::clamp(
                                                (plane_depth - start_z) /
                                                    (post_em_z - start_z),
                                                0.0F, 1.0F);
                                            record.history = frag.parent_history;
                                            record.transport_path = 1U;
                                            record.particle_id = frag.rng_stream;
                                            record.rng_stream = frag.rng_stream;
                                            record.plane_index =
                                                static_cast<std::uint32_t>(plane);
                                            record.atomic_number = static_cast<std::int16_t>(transport_z);
                                            record.mass_number = static_cast<std::int16_t>(transport_a);
                                            record.depth_mm = plane_depth;
                                            record.kinetic_energy_MeV = sycl::fmax(
                                                0.0F,
                                                sec_e + (1.0F - fraction) * dE);
                                            record.weight = frag.weight;
                                            record.x_mm = start_x +
                                                fraction * collision_input_dx *
                                                    sec_step_mm;
                                            record.y_mm = start_y +
                                                fraction * collision_input_dy *
                                                    sec_step_mm;
                                            record.direction_x = collision_input_dx;
                                            record.direction_y = collision_input_dy;
                                            record.direction_z = collision_input_dz;
                                            record.valid = 1U;
                                        }
                                    }
#endif
                                }
                            }
#if defined(CARBON_ENABLE_MINIBEAM)
                            else if ((kProductionSecondaryPath ? false :
                                      CARBON_SECONDARY_CONTEXT_FIELD(
                                          water_entry_secondary_replay)) &&
                                     transport_z == 6 && transport_a == 12 &&
                                     CARBON_SECONDARY_CONTEXT_FIELD(
                                         minibeam_water_primary_plane_records_device) != nullptr &&
                                     collision_input_dz > 0.0F) {
                                const auto start_x = post_em_x -
                                    collision_input_dx * sec_step_mm;
                                const auto start_y = post_em_y -
                                    collision_input_dy * sec_step_mm;
                                const auto start_z = post_em_z -
                                    collision_input_dz * sec_step_mm;
                                for (std::size_t plane = 0;
                                     plane < CARBON_SECONDARY_CONTEXT_FIELD(
                                         minibeam_water_primary_plane_count);
                                     ++plane) {
                                    const auto plane_depth =
                                        CARBON_SECONDARY_CONTEXT_FIELD(
                                            minibeam_water_primary_plane_depths_device)[plane];
                                    if (!(start_z < plane_depth &&
                                          post_em_z >= plane_depth)) continue;
                                    auto& record = CARBON_SECONDARY_CONTEXT_FIELD(
                                        minibeam_water_primary_plane_records_device)[
                                        frag.parent_history *
                                            CARBON_SECONDARY_CONTEXT_FIELD(
                                                minibeam_water_primary_plane_count) +
                                        plane];
                                    if (record.valid) continue;
                                    const auto fraction = sycl::clamp(
                                        (plane_depth - start_z) /
                                            (post_em_z - start_z),
                                        0.0F, 1.0F);
                                    record.history = frag.parent_history;
                                    record.transport_path = 1U;
                                    record.particle_id = frag.rng_stream;
                                    record.rng_stream = frag.rng_stream;
                                    record.plane_index =
                                        static_cast<std::uint32_t>(plane);
                                    record.atomic_number = static_cast<std::int16_t>(transport_z);
                                    record.mass_number = static_cast<std::int16_t>(transport_a);
                                    record.depth_mm = plane_depth;
                                    record.kinetic_energy_MeV = sycl::fmax(
                                        0.0F, sec_e + (1.0F - fraction) * dE);
                                    record.weight = frag.weight;
                                    record.x_mm = start_x + fraction *
                                        collision_input_dx * sec_step_mm;
                                    record.y_mm = start_y + fraction *
                                        collision_input_dy * sec_step_mm;
                                    record.direction_x = collision_input_dx;
                                    record.direction_y = collision_input_dy;
                                    record.direction_z = collision_input_dz;
                                    record.valid = 1U;
                                }
                            }
#endif

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
                            saved.unified_secondary_state=unified_secondary_state;
#if CARBON_EM_LOCAL_AUDIT
                            saved.unified_secondary_audit=unified_secondary_audit;
#endif
                            saved.sec_steps=sec_steps;
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
                cgh.parallel_for<CarbonSecondaryTransportKernel<EmMode,
                                                                 kProductionSecondaryPath,
                                                                 kExactSpecies>>(
                    sycl::nd_range<1>{secondary_global_size, secondary_local_size},
                    secondary_kernel);
#else
                cgh.parallel_for<CarbonSecondaryTransportKernel<EmMode,
                                                                 kProductionSecondaryPath,
                                                                 kExactSpecies>>(
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
    std::vector<double> minibeam_c12_roi_dose_host;
    if (enable_minibeam_primary_c12_roi_scoring) {
        std::vector<DoseAtomicT> device_host(minibeam_c12_roi_value_count);
        queue.copy(minibeam_c12_roi_dose_device, device_host.data(),
                   minibeam_c12_roi_value_count).wait_and_throw();
        minibeam_c12_roi_dose_host.resize(minibeam_c12_roi_value_count);
        std::transform(device_host.begin(), device_host.end(),
                       minibeam_c12_roi_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }
    std::array<std::uint64_t, minibeam_spatial_audit_slot_count>
        minibeam_spatial_audit_host{};
    if (enable_minibeam_primary_c12_roi_scoring) {
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
        if (water_urban_segment_count_device != nullptr) {
            std::uint64_t water_urban_segments = 0;
            queue.copy(water_urban_segment_count_device, &water_urban_segments,
                       1);
            std::cout << "[minibeam-water] urban segments total="
                      << water_urban_segments << '\n';
            std::cout << "[minibeam-water] urban subdiv cap trips="
                      << minibeam_event_counts_host
                             [minibeam_water_urban_subdiv_cap_slot]
                      << '\n';
            // Fix B5 hard gate (§9.1): any invalid proposal, zero-progress
            // segment, or cap trip fails the run. Structured record: the
            // fatal slot count plus the cap-trip count above.
            const std::uint64_t urban_fatal =
                minibeam_event_counts_host[minibeam_water_urban_fatal_slot];
            std::cout << "[minibeam-water] urban fatal proposals="
                      << urban_fatal << '\n';
            if (urban_fatal > 0) {
                throw std::runtime_error(
                    "Urban MSC fatal: " + std::to_string(urban_fatal) +
                    " invalid/zero-progress/capped proposals (slot " +
                    std::to_string(minibeam_water_urban_fatal_slot) +
                    "); cap trips=" +
                    std::to_string(minibeam_event_counts_host
                                       [minibeam_water_urban_subdiv_cap_slot]) +
                    ". Partial space paths with full-macro-step energy loss"
                    " must never pass as successful dose.");
            }
        }
        if (minibeam_water_delta_diag_device != nullptr) {
            std::array<std::uint64_t, 17> water_delta_diag{};
            queue.copy(minibeam_water_delta_diag_device,
                       water_delta_diag.data(), water_delta_diag.size());
            // Slots: 0 steps, 1 input, 2 residual, 3 relocated, 4 escaped,
            // 5 invalid, 6 delta-sum, 7 loss-sum, 8..16 birth->deposit ROI.
            std::cout << "[minibeam-water] delta response steps="
                      << water_delta_diag[0]
                      << " input_MeV=" << water_delta_diag[1] * 1e-6
                      << " residual_MeV=" << water_delta_diag[2] * 1e-6
                      << " relocated_MeV=" << water_delta_diag[3] * 1e-6
                      << " escaped_MeV=" << water_delta_diag[4] * 1e-6
                      << " invalid=" << water_delta_diag[5]
                      << " delta_sum_MeV=" << water_delta_diag[6] * 1e-6
                      << " loss_sum_MeV=" << water_delta_diag[7] * 1e-6 << '\n';
            std::cout << "[minibeam-water] delta ROI rows birth->dep:";
            for (int b = 0; b < 3; ++b) {
                for (int d = 0; d < 3; ++d) {
                    std::cout << (d == 0 ? " " : "/")
                              << water_delta_diag[8 + b * 3 + d] * 1e-6;
                }
                if (b < 2) std::cout << " |";
            }
            std::cout << '\n';
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
        if(unified_em){
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
    if (ion_stopping_failures) {
        std::array<std::uint32_t, 8> failures{};
        queue.copy(ion_stopping_failures, failures.data(), failures.size()).wait_and_throw();
        free_device(ion_stopping_failures);
        free_device(schneider_ion_sp_device);
        if (failures[0]) {
            float energy, density;
            std::memcpy(&energy, &failures[4], sizeof(float));
            std::memcpy(&density, &failures[5], sizeof(float));
            throw std::runtime_error("Schneider ion stopping domain failure: " + std::to_string(failures[0]) +
                " first Z=" + std::to_string(failures[1]) + " A=" + std::to_string(failures[2]) +
                " section=" + std::to_string(failures[3]) + " E_MeVu=" + std::to_string(energy) +
                " rho=" + std::to_string(density));
        }
    }
    if (fluct_domain_failures) {
        std::uint32_t failures=0;
        queue.copy(fluct_domain_failures, &failures, 1).wait_and_throw();
        free_device(fluct_domain_failures);
        if (failures) throw std::runtime_error("Fraction-axis fluctuation outside validated grid: " + std::to_string(failures));
    }
    std::array<std::uint64_t, 10> schneider_delta_energy_host{};
    std::array<std::uint64_t,7> electron_joint_diag_host{};
    std::array<std::uint64_t,3> material_untracked_host{};
    if(material_untracked_device)queue.copy(material_untracked_device,material_untracked_host.data(),3).wait_and_throw();
    if(material_failure_device && material_untracked_host[2]) {
        MaterialPacketFailure f;queue.copy(material_failure_device,&f,1).wait_and_throw();
        const auto& c=f.before.cursor;
        std::cerr.precision(17);
        std::cerr<<"[material-packet-failure] history="<<f.history<<" step="<<f.step<<" rng="<<f.rng
                 <<" status="<<int(f.after.status)<<" gap="<<int(f.after.gap)<<" advance="<<int(f.advance.status)
                 <<" boundary="<<int(f.advance.boundary.status)<<" section="<<c.section
                 <<" rho="<<c.density_g_cm3<<" physical_rho="<<c.physical_density_g_cm3
                 <<" source="<<c.source<<" row="<<c.row<<" pdg="<<c.pdg<<" KE="<<c.energy_MeV
                 <<" W="<<f.before.weight_MeV<<" fraction="<<c.fraction
                 <<" position="<<c.position[0]<<','<<c.position[1]<<','<<c.position[2]
                 <<" advances="<<f.after.advances<<" crossings="<<f.after.boundary_restarts
                 <<" restarts="<<f.after.source_restarts<<" residual="<<f.advance.energy_residual_MeV<<'\n';
    }
    if(electron_joint_diag_device)queue.copy(electron_joint_diag_device,electron_joint_diag_host.data(),7).wait_and_throw();
    if (schneider_delta_energy_device != nullptr) {
        queue.copy(schneider_delta_energy_device,
                   schneider_delta_energy_host.data(),
                   schneider_delta_energy_host.size()).wait_and_throw();
    }

    std::vector<LongitudinalDomainRecord> longitudinal_domain_host;
    if (longitudinal_domain_device) {
        longitudinal_domain_host.resize(std::min<std::uint64_t>(
            schneider_delta_energy_host[6], kLongitudinalDomainLogCap));
        if (!longitudinal_domain_host.empty())
            queue.copy(longitudinal_domain_device, longitudinal_domain_host.data(),
                       longitudinal_domain_host.size()).wait_and_throw();
        std::sort(longitudinal_domain_host.begin(), longitudinal_domain_host.end(),
            [](const auto& a, const auto& b) {
                return a.history < b.history || (a.history == b.history && a.step < b.step);
            });
    }
    // Free buffers
    free_device(electron_path_bounds_device);
    free_device(water_electron_channels_device);free_device(water_electron_samples_device);
    free_device(water_electron_nodes_device);free_device(water_electron_heads_device);free_device(water_electron_radius_device);
    free_device(electron_joint_channels_device);free_device(electron_joint_samples_device);free_device(electron_joint_diag_device);
    free_device(electron_path_ranges_device);free_device(electron_path_vectors_device);
    free_immutable_device(table_device);
    free_immutable_device(energy_grid_device);
    free_immutable_device(cumulative_range_device);
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
    free_device(minibeam_c12_roi_dose_device);
    free_device(minibeam_spatial_audit_device);
    free_device(minibeam_fixed_region_by_x_bin_device);
    free_device(be_isotope_origin_voxel_dose_device);
    free_device(he4_hazard_audit_device);
    free_device(he_isotope_origin_voxel_dose_device);
    free_device(let_moments_device);
    free_device(voxel_let_moments_device);
    free_device(deposited_device);
    free_device(secondary_queue_device);
    free_device(secondary_count_device);
    free_device(ion_species_sp_device);
    free_device(ion_energy_grid_device);
    free_device(ion_csda_a1_device);
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
    free_device(water_urban_segment_count_device);
    free_device(minibeam_water_delta_diag_device);
    free_device(minibeam_fragment_miss_joint_counts_device);
    free_device(minibeam_fragment_miss_joint_energy_device);
    free_device(minibeam_fragment_miss_joint_depth_device);
    free_device(minibeam_fragment_miss_joint_remaining_device);
    free_device(minibeam_copper_sp_energies_device);
    free_device(minibeam_copper_sp_values_device);
    free_device(minibeam_copper_loss_e_device);
    free_device(minibeam_copper_loss_r_device);
    free_device(minibeam_copper_loss_d_device);
    free_device(minibeam_water_urban_loss_e_device);
    free_device(minibeam_water_urban_loss_r_device);
    free_device(minibeam_water_urban_loss_d_device);
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
    free_device(fluct_energy_device);
    free_device(fluct_density_device);
    free_device(fluct_probability_device);
    free_device(fluct_quantile_device);
    free_device(secondary_overflow_count_device);
    free_device(secondary_overflow_energy_device);
    free_device(escaped_device);
    free_device(sampled_incident_device);
    free_device(steps_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(slab_radiation_lengths_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_factor_lut_device);
    free_device(ct_mass_sp_za_rel_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);
    free_device(schneider_delta_energies_device);
    free_device(schneider_delta_fractions_device);
    free_device(schneider_delta_radii_device);
    free_device(schneider_delta_source_eligible_device);
    free_device(schneider_delta_energy_device);
    free_device(longitudinal_domain_device);
    free_device(schneider_long_energies_device);
    free_device(schneider_long_fractions_device);
    free_device(schneider_long_lambdas_device);
    free_device(schneider_primary_xs_device);
    free_device(schneider_inelastic_device);
    free_device(schneider_diag_device);
    free_device(schneider_float_device);
    free_device(schneider_miss_device);
    free_device(schneider_track_log_device);
    free_device(schneider_miss_count_device);
    free_device(schneider_track_count_device);

    TransportResult result;
    result.primary_elastic_interactions=elastic_counts_host[0];
    result.secondary_elastic_interactions=elastic_counts_host[1];
    result.elastic_post_em_null_collisions=elastic_counts_host[9]+elastic_counts_host[10];
    result.elastic_local_deposited_energy_MeV=elastic_energy_host[0];
    result.elastic_queued_charged_energy_MeV=elastic_energy_host[1];
    result.elastic_queue_overflow_energy_MeV=elastic_energy_host[2];
    result.elastic_queue_overflow=elastic_counts_host[11];
    result.schneider_miss_log = std::move(schneider_miss_host);
    result.schneider_primary_delta_tail_moved_MeV =
        static_cast<double>(schneider_delta_energy_host[0]) * 1.0e-6;
    result.schneider_primary_delta_tail_fallback_MeV =
        static_cast<double>(schneider_delta_energy_host[1]) * 1.0e-6;
    result.schneider_primary_delta_tail_escaped_scorer_MeV =
        static_cast<double>(schneider_delta_energy_host[2]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_moved_MeV =
        static_cast<double>(schneider_delta_energy_host[3]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_fallback_MeV =
        static_cast<double>(schneider_delta_energy_host[4]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_escaped_scorer_MeV =
        static_cast<double>(schneider_delta_energy_host[5]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_domain_queries = schneider_delta_energy_host[6];
    result.schneider_primary_delta_longitudinal_domain_energy_MeV =
        static_cast<double>(schneider_delta_energy_host[7]) * 1e-6;
    result.schneider_primary_delta_longitudinal_invalid_marches = schneider_delta_energy_host[9];
    result.longitudinal_diagnostic_density_g_cm3 = schneider_long_diagnostic_density;
    result.longitudinal_domain_log = std::move(longitudinal_domain_host);
    result.electron_joint_diagnostics={electron_joint_diag_host[0],electron_joint_diag_host[1],electron_joint_diag_host[2],
        static_cast<double>(electron_joint_diag_host[3])*1e-6,static_cast<double>(electron_joint_diag_host[4])*1e-6,static_cast<double>(electron_joint_diag_host[5])*1e-6,electron_joint_diag_host[6]};
    result.electron_ordered_path_sha256=electron_path_sha256;
    result.material_electron_photon_untracked_MeV=double(material_untracked_host[0])*1e-6;
    result.material_electron_untracked_MeV=double(material_untracked_host[0]+material_untracked_host[1])*1e-6;
    free_device(material_untracked_device);
    free_device(material_failure_device);
    if(short_range_hits_device) {
        std::uint64_t hits=0;
        queue.copy(short_range_hits_device,&hits,1).wait_and_throw();
        std::cout<<"[research-short-range] shortcut_packets="<<hits<<'\n';
        free_device(short_range_hits_device);
    }
    result.schneider_unsupported_tracks = std::move(schneider_track_host);
    result.backend = "sycl-" + resolved_device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    if (config.uses_moment_matched_straggling()) {
        result.backend += "+moment-matched-straggling";
    }
    if (config.enable_step_stable_straggling) {
        result.backend += "+step-stable-primary-straggling";
    }
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
    if (config.enable_minibeam) {
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
        result.backend += use_material_tables ? "+layered-material" : "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += use_insert_material_tables ? "+hetero-insert-material" : "+hetero-insert";
    }
    if (enable_ct_grid) {
        if (use_ct_mass_sp) {
            result.backend += use_ct_density_mass_spr ? "+ct-grid-density-spr+ct-dda"
                                                     : "+ct-grid-mass-sp-lut+ct-dda";
        } else if (use_ct_material_sp) {
            result.backend += "+ct-grid-material+ct-dda";
        } else {
            result.backend += "+ct-grid+ct-dda";
        }
    }
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
    result.minibeam_c12_roi_values =
        std::move(minibeam_c12_roi_dose_host);
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
    if (config.enable_minibeam) {
        if (minibeam_water_secondary_c12_fermi_eyges_tail) {
            std::cout << "[minibeam-water-secondary-c12-fe-segments] steps="
                      << minibeam_event_counts_host[
                             minibeam_secondary_c12_fe_step_slot]
                      << " segments="
                      << minibeam_event_counts_host[
                             minibeam_secondary_c12_fe_segment_slot]
                      << '\n';
        }
        if (minibeam_water_secondary_c12_urban_v2) {
            std::cout << "[minibeam-water-secondary-c12-urban-steps] steps="
                      << minibeam_event_counts_host[
                             minibeam_water_secondary_c12_urban_step_slot]
                      << '\n';
        }
        if (minibeam_water_secondary_c12_post_sample_loss_scale != 1.0F) {
            std::cout << "[minibeam-water-secondary-c12-post-sample-loss] "
                         "raw_MeV="
                      << static_cast<double>(minibeam_event_counts_host[
                             minibeam_secondary_c12_raw_loss_micro_slot]) * 1.0e-6
                      << " scaled_MeV="
                      << static_cast<double>(minibeam_event_counts_host[
                             minibeam_secondary_c12_scaled_loss_micro_slot]) * 1.0e-6
                      << " steps="
                      << minibeam_event_counts_host[
                             minibeam_secondary_c12_scaled_loss_step_slot]
                      << '\n';
        }
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

    if (config.quality_reject_any_queue_overflow && overflow_count_host > 0) {
        throw std::runtime_error("Secondary particle queue overflow detected: discarded " +
                                 std::to_string(overflow_count_host) + " particles (" +
                                 std::to_string(overflow_energy_host) + " MeV). Shard must be split and rerun with fewer particles.");
    }

    result.nuclear_interactions =
        schneider_inelastic_host;
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;
}

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
    const StoppingPowerTable& stopping_power,const CrossSectionTable& cross_section,
    const std::string& device_name,SyclTransportContext* context) {
#if CARBON_EM_SPECIALIZE
    if(config.em_model=="g4_material_joint_v1")
        return transport_sycl_impl<1>(config,stopping_power,cross_section,device_name,context);
    return transport_sycl_impl<0>(config,stopping_power,cross_section,device_name,context);
#else
    return transport_sycl_impl<-1>(config,stopping_power,cross_section,device_name,context);
#endif
}

}  // namespace carbon

#endif
