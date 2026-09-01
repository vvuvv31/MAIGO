#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

// These records deliberately mirror startup/package_tools/cinel02.py. They
// are packed because they are an on-disk contract, not a device ABI.
#pragma pack(push, 1)
struct Cinel02InteractionRecord {
    std::uint64_t run_id{0};
    std::uint32_t thread_id{0};
    std::uint64_t event_id{0};
    std::uint32_t track_id{0};
    std::uint32_t parent_track_id{0};
    std::uint32_t interaction_sequence{0};
    std::int32_t projectile_pdg{0};
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    float projectile_charge{0.0F};
    float projectile_rest_mass{0.0F};
    float projectile_excitation{0.0F};
    float collision_energy_MeV{0.0F};
    float collision_energy_MeV_per_u{0.0F};
    float collision_x_mm{0.0F};
    float collision_y_mm{0.0F};
    float collision_z_mm{0.0F};
    float collision_direction_x{0.0F};
    float collision_direction_y{0.0F};
    float collision_direction_z{1.0F};
    float collision_time_ns{0.0F};
    float proper_time_ns{0.0F};
    float track_weight{1.0F};
    float step_length_mm{0.0F};
    std::int32_t material_id{0};
    std::int32_t cuts_couple_id{0};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    std::uint32_t target_isotope_id{0};
    std::int32_t process_type{0};
    std::int32_t process_subtype{0};
    std::int32_t model_id{-1};
    std::int32_t parent_status{0};
    std::int32_t parent_pdg{0};
    std::int16_t parent_z{0};
    std::int16_t parent_a{0};
    float parent_charge{0.0F};
    float parent_rest_mass{0.0F};
    float parent_excitation{0.0F};
    float parent_energy_MeV{0.0F};
    float parent_direction_x{0.0F};
    float parent_direction_y{0.0F};
    float parent_direction_z{1.0F};
    float parent_x_mm{0.0F};
    float parent_y_mm{0.0F};
    float parent_z_mm{0.0F};
    float parent_time_ns{0.0F};
    float parent_proper_time_ns{0.0F};
    float parent_weight{1.0F};
    float process_local_deposit_MeV{0.0F};
    float nonionizing_deposit_MeV{0.0F};
    std::uint32_t direct_product_count{0};
    std::uint32_t unsupported_product_count{0};
    float unsupported_product_energy_MeV{0.0F};
    float audit_pre_energy_MeV{0.0F};
    float audit_pre_direction_x{0.0F};
    float audit_pre_direction_y{0.0F};
    float audit_pre_direction_z{1.0F};
    float audit_pre_x_mm{0.0F};
    float audit_pre_y_mm{0.0F};
    float audit_pre_z_mm{0.0F};
    float audit_whole_step_deposit_MeV{0.0F};
    float source_initial_energy_MeV{0.0F};
    float absolute_depth_mm{0.0F};
    char material_name[64]{};
    char target_isotope_name[32]{};
    char process_name[64]{};
    char model_name[64]{};
};

struct Cinel02ProductRecord {
    std::int32_t pdg{0};
    std::int16_t z{0};
    std::int16_t a{0};
    float charge{0.0F};
    float rest_mass{0.0F};
    float excitation{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{1.0F};
    float local_direction_x{0.0F};
    float local_direction_y{0.0F};
    float local_direction_z{1.0F};
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float creation_time_ns{0.0F};
    float weight{1.0F};
    std::int32_t role{0};
};

struct Cinel02CellIndex {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    std::uint32_t energy_bin{0};
    std::uint64_t interaction_offset{0};
    std::uint64_t interaction_count{0};
    float energy_lower_MeV_per_u{0.0F};
    float energy_upper_MeV_per_u{0.0F};
};

struct Cinel02PackageHeader {
    char magic[8]{};
    std::uint32_t version{0};
    std::uint32_t header_size{0};
    std::uint32_t endian_marker{0};
    std::uint32_t index_record_size{0};
    std::uint32_t interaction_record_size{0};
    std::uint32_t product_record_size{0};
    std::uint32_t flags{0};
    std::uint64_t cell_count{0};
    std::uint64_t interaction_count{0};
    std::uint64_t product_count{0};
    std::uint64_t file_size{0};
    float minimum_energy_MeV_per_u{0.0F};
    float energy_bin_width_MeV_per_u{0.0F};
    std::uint64_t minimum_events_per_bin{0};
    // With cinel02_global_energy_index_flag, this is the persisted node count;
    // legacy packages keep it zero and the loader reconstructs the index.
    std::uint64_t reserved{0};
    // A package is only replayable with the rate campaign that produced it.
    // The bundle manifest repeats this UUID and the host preflight compares
    // both values before uploading either table.
    char campaign_uuid[40]{};
};

// A global energy node merges all interactions with the same projectile,
// target, and captured collision energy.  The persisted node array is sorted
// lexicographically by these fields; the persisted uint64 event_offsets and
// event_indices form a prefix-indexed permutation of the complete interaction
// table.  The SYCL upload narrows both arrays to uint32 after the runtime
// checks the device-layout limit.
struct Cinel02EnergyNode {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    float collision_energy_MeV_per_u{0.0F};
};
#pragma pack(pop)

static_assert(sizeof(Cinel02InteractionRecord) == 476);
static_assert(sizeof(Cinel02ProductRecord) == 72);
static_assert(sizeof(Cinel02CellIndex) == 36);
static_assert(sizeof(Cinel02PackageHeader) == 132);
static_assert(sizeof(Cinel02EnergyNode) == 12);

inline constexpr std::uint32_t cinel02_global_energy_index_flag = 1U << 4;

// Compact device-side views. The raw/package records above retain the full
// audit state; these views omit strings, times, and provenance that are not
// needed by the transport kernel.
struct Cinel02DeviceInteraction {
    float incident_energy_MeV_per_u{0.0F};
    float process_local_deposit_MeV{0.0F};
    float nonionizing_deposit_MeV{0.0F};
    std::uint32_t product_offset{0};
    std::uint32_t product_count{0};
    float parent_energy_MeV{0.0F};
    std::int32_t parent_pdg{0};
    std::int16_t parent_z{0};
    std::int16_t parent_a{0};
    float parent_charge{0.0F};
    float parent_rest_mass_MeV{0.0F};
    float parent_excitation_MeV{0.0F};
    float parent_weight{1.0F};
    // Projectile-local final parent direction. The on-disk interaction keeps
    // the captured global direction; the host upload converts it using the
    // captured collision axis so replay can rotate it onto the runtime track.
    float parent_local_direction_x{0.0F};
    float parent_local_direction_y{0.0F};
    float parent_local_direction_z{1.0F};
    std::int32_t parent_status{0};
};
static_assert(sizeof(Cinel02DeviceInteraction) == 64);

struct Cinel02DeviceProduct {
    std::int32_t pdg{0};
    std::int16_t z{0};
    std::int16_t a{0};
    float charge{0.0F};
    float rest_mass_MeV{0.0F};
    float excitation_MeV{0.0F};
    float weight{1.0F};
    float kinetic_energy_MeV{0.0F};
    float local_direction_x{0.0F};
    float local_direction_y{0.0F};
    float local_direction_z{1.0F};
    float global_direction_x{0.0F};
    float global_direction_y{0.0F};
    float global_direction_z{1.0F};
    std::int32_t role{0};
};
static_assert(sizeof(Cinel02DeviceProduct) == 56);

struct Cinel02DeviceTables {
    std::vector<Cinel02DeviceInteraction> interactions{};
    std::vector<Cinel02DeviceProduct> products{};
    std::vector<Cinel02EnergyNode> energy_nodes{};
    std::vector<std::uint32_t> event_offsets{};
    std::vector<std::uint32_t> event_indices{};

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return interactions.size() * sizeof(Cinel02DeviceInteraction) +
               products.size() * sizeof(Cinel02DeviceProduct) +
               energy_nodes.size() * sizeof(Cinel02EnergyNode) +
               event_offsets.size() * sizeof(std::uint32_t) +
               event_indices.size() * sizeof(std::uint32_t);
    }
};

struct Cinel02FixedReplayView {
    const Cinel02InteractionRecord* serialized_interaction{nullptr};
    const Cinel02DeviceInteraction* compact_interaction{nullptr};
    const Cinel02ProductRecord* serialized_products{nullptr};
    const Cinel02DeviceProduct* compact_products{nullptr};
    std::uint32_t product_count{0};

    [[nodiscard]] explicit operator bool() const noexcept {
        return serialized_interaction != nullptr && compact_interaction != nullptr;
    }
};

// Final-state packages do not carry rate information. Rates are loaded from
// an independent exposure campaign. Each group owns a contiguous, strictly
// increasing energy sample range for one projectile/target pair.
struct Cinel02RateGroup {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    std::uint32_t sample_offset{0};
    std::uint32_t sample_count{0};
    // The exposure CSV stores a macroscopic rate at this reference target
    // number density. Runtime material binding converts it to microscopic
    // sigma and multiplies by the active material number density.
    float reference_number_density_per_mm3{0.0F};
};
static_assert(sizeof(Cinel02RateGroup) == 20);

struct Cinel02RateSample {
    float energy_MeV_per_u{0.0F};
    float macroscopic_cross_section_per_mm{0.0F};
};
static_assert(sizeof(Cinel02RateSample) == 8);

// One active target isotope for the CINEL02 material resolver. A negative
// material_section denotes the homogeneous manifest material. Non-negative
// sections are Schneider material-section ids; their number density is
// scaled by the sampled local voxel density relative to
// reference_material_density_g_per_cm3.
struct Cinel02MaterialTarget {
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    std::int16_t material_section{-1};
    std::int16_t reserved{0};
    float reference_number_density_per_mm3{0.0F};
    float number_density_per_mm3{0.0F};
    float reference_material_density_g_per_cm3{1.0F};
};
static_assert(sizeof(Cinel02MaterialTarget) == 20);

struct InelasticRateLookup {
    bool covered{false};
    double value_per_mm{0.0};
};

class InelasticRateV2Table {
public:
    static InelasticRateV2Table from_csv(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<Cinel02RateGroup>& groups() const noexcept;
    [[nodiscard]] const std::vector<Cinel02RateSample>& samples() const noexcept;
    // Bind the reference number density declared by the bundle manifest. The
    // operation is deliberately explicit: an unbound production rate table
    // cannot be used for n_i * sigma_i material resolution.
    void bind_reference_number_densities(
        const std::vector<Cinel02MaterialTarget>& targets);
    [[nodiscard]] bool has_reference_number_densities() const noexcept;
    [[nodiscard]] InelasticRateLookup lookup(int projectile_z, int projectile_a,
                                             int target_z, int target_a,
                                             double energy_MeV_per_u) const noexcept;
    [[nodiscard]] double interpolate(int projectile_z, int projectile_a,
                                     int target_z, int target_a,
                                     double energy_MeV_per_u) const noexcept;

private:
    std::vector<Cinel02RateGroup> groups_;
    std::vector<Cinel02RateSample> samples_;
    bool reference_number_densities_bound_{false};
};

class InelasticPackageV2Table {
public:
    static InelasticPackageV2Table from_binary(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<Cinel02CellIndex>& cells() const noexcept;
    [[nodiscard]] const std::vector<Cinel02InteractionRecord>& interactions() const noexcept;
    [[nodiscard]] const std::vector<Cinel02ProductRecord>& products() const noexcept;
    [[nodiscard]] const std::vector<Cinel02EnergyNode>& energy_nodes() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& event_offsets() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& event_indices() const noexcept;
    [[nodiscard]] std::uint64_t find_event(
        int projectile_z, int projectile_a, int target_z, int target_a,
        float collision_energy_MeV_per_u, float maximum_energy_mismatch_MeV_per_u,
        float u01) const noexcept;
    [[nodiscard]] const Cinel02CellIndex* find_cell(
        int projectile_z, int projectile_a, int target_z, int target_a,
        float collision_energy_MeV_per_u, float maximum_energy_mismatch_MeV_per_u) const noexcept;
    [[nodiscard]] const Cinel02InteractionRecord* interaction(
        const Cinel02CellIndex& cell, std::uint64_t offset) const noexcept;
    [[nodiscard]] Cinel02DeviceTables make_device_tables() const;
    [[nodiscard]] Cinel02FixedReplayView fixed_replay(
        std::uint64_t event_index, const Cinel02DeviceTables& compact) const noexcept;
    [[nodiscard]] const Cinel02ProductRecord* products_for(
        const Cinel02InteractionRecord& interaction) const noexcept;
    [[nodiscard]] std::uint32_t product_offset(
        const Cinel02InteractionRecord& interaction) const noexcept;
    [[nodiscard]] float minimum_energy_MeV_per_u() const noexcept;
    [[nodiscard]] float energy_bin_width_MeV_per_u() const noexcept;
    [[nodiscard]] std::uint64_t minimum_events_per_bin() const noexcept;
    [[nodiscard]] const std::string& campaign_uuid() const noexcept;

    // Target-first selection is intentionally a small, explicit primitive.
    // The caller supplies target-specific macroscopic cross sections obtained
    // from the material/rate campaign; no product-derived mixture is allowed.
    [[nodiscard]] static int select_target_z(
        float u01, float hydrogen_macroscopic_xs, float oxygen_macroscopic_xs) noexcept;

private:
    float minimum_energy_MeV_per_u_{0.0F};
    float energy_bin_width_MeV_per_u_{1.0F};
    std::uint64_t minimum_events_per_bin_{0};
    std::string campaign_uuid_{};
    std::vector<Cinel02CellIndex> cells_;
    std::vector<Cinel02InteractionRecord> interactions_;
    std::vector<Cinel02ProductRecord> products_;
    std::vector<std::uint32_t> product_offsets_;
    std::vector<Cinel02EnergyNode> energy_nodes_;
    std::vector<std::uint64_t> event_offsets_;
    std::vector<std::uint64_t> event_indices_;
};

}  // namespace carbon
