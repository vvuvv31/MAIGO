#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <limits>

namespace carbon {

#pragma pack(push, 1)

struct Cinel03InteractionRecord {
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
    std::int16_t target_element_z{0}; // Key target element atomic number (1-100)
    std::int16_t target_a{0};         // Actual target isotope mass number when available
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

struct Cinel03ProductRecord {
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

// Event key: projectile_Z, projectile_A, target_element_Z, energy_bin
// Material section is NOT part of the event key.
struct Cinel03CellIndex {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_element_z{0};
    std::int16_t reserved{0};
    std::uint32_t energy_bin{0};
    std::uint64_t interaction_offset{0};
    std::uint64_t interaction_count{0};
    float energy_lower_MeV_per_u{0.0F};
    float energy_upper_MeV_per_u{0.0F};
};

struct Cinel03PackageHeader {
    char magic[8]{};          // "CINPKG04"
    std::uint32_t version{0}; // 4
    std::uint32_t header_size{0};
    std::uint32_t endian_marker{0}; // 0x01020304
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
    std::uint64_t energy_node_count{0};
    char campaign_uuid[40]{};
    std::uint32_t checksum_crc32{0};
};

struct Cinel03EnergyNode {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_element_z{0};
    std::int16_t reserved{0};
    float collision_energy_MeV_per_u{0.0F};
};

#pragma pack(pop)

static_assert(sizeof(Cinel03InteractionRecord) == 476);
static_assert(sizeof(Cinel03ProductRecord) == 72);
static_assert(sizeof(Cinel03CellIndex) == 36);
static_assert(sizeof(Cinel03PackageHeader) == 136);
static_assert(sizeof(Cinel03EnergyNode) == 12);

inline constexpr std::uint32_t cinel03_global_energy_index_flag = 1U << 4;
inline constexpr std::uint32_t cinel03_crc32_checksum_flag = 1U << 5;

// Compact GPU Device structs
struct Cinel03DeviceInteraction {
    float collision_energy_MeV_per_u{0.0F};
    float process_local_deposit_MeV{0.0F};
    float nonionizing_deposit_MeV{0.0F};
    std::uint32_t product_offset{0};
    std::uint32_t direct_product_count{0};
    float parent_energy_MeV{0.0F};
    std::int32_t parent_pdg{0};
    std::int16_t parent_z{0};
    std::int16_t parent_a{0};
    float parent_charge{0.0F};
    float parent_rest_mass{0.0F};
    float parent_excitation{0.0F};
    float parent_weight{1.0F};
    float parent_local_dir_x{0.0F};
    float parent_local_dir_y{0.0F};
    float parent_local_dir_z{1.0F};
    std::int16_t target_element_z{0};
    std::int16_t target_a{0};
};

struct Cinel03DeviceProduct {
    float kinetic_energy_MeV{0.0F};
    float local_direction_x{0.0F};
    float local_direction_y{0.0F};
    float local_direction_z{1.0F};
    float charge{0.0F};
    float rest_mass{0.0F};
    float excitation{0.0F};
    float weight{1.0F};
    std::int32_t pdg{0};
    std::int16_t z{0};
    std::int16_t a{0};
    std::int32_t role{0};
};

struct Cinel03DeviceTables {
    std::vector<Cinel03DeviceInteraction> interactions{};
    std::vector<Cinel03DeviceProduct> products{};
    std::vector<Cinel03EnergyNode> energy_nodes{};
    std::vector<std::uint32_t> event_offsets{};
    std::vector<std::uint32_t> event_indices{};

    [[nodiscard]] std::size_t device_bytes() const noexcept {
        return interactions.size() * sizeof(Cinel03DeviceInteraction) +
               products.size() * sizeof(Cinel03DeviceProduct) +
               energy_nodes.size() * sizeof(Cinel03EnergyNode) +
               event_offsets.size() * sizeof(std::uint32_t) +
               event_indices.size() * sizeof(std::uint32_t);
    }
};

struct Cinel03FixedReplayView {
    const Cinel03InteractionRecord* serialized_interaction{nullptr};
    const Cinel03DeviceInteraction* compact_interaction{nullptr};
    const Cinel03ProductRecord* serialized_products{nullptr};
    const Cinel03DeviceProduct* compact_products{nullptr};
    std::uint32_t product_count{0};
};

inline bool cinel03_node_less(
    const Cinel03EnergyNode& node,
    int pz, int pa, int tz, float energy) noexcept
{
    if (node.projectile_z != pz) return node.projectile_z < pz;
    if (node.projectile_a != pa) return node.projectile_a < pa;
    if (node.target_element_z != tz) return node.target_element_z < tz;
    return node.collision_energy_MeV_per_u < energy;
}

inline bool cinel03_key_less(
    int pz, int pa, int tz, float energy,
    const Cinel03EnergyNode& node) noexcept
{
    if (pz != node.projectile_z) return pz < node.projectile_z;
    if (pa != node.projectile_a) return pa < node.projectile_a;
    if (tz != node.target_element_z) return tz < node.target_element_z;
    return energy < node.collision_energy_MeV_per_u;
}

// Target Element Registry for Schneider Materials
// Elements in Schneider CT materials: H (1), C (6), N (7), O (8), Na (11),
// Mg (12), P (15), S (16), Cl (17), Ar (18), K (19), Ca (20), Ti (22).
inline constexpr std::array<int, 13> kSchneiderTargetElements{
    1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22
};

[[nodiscard]] inline bool is_registered_schneider_target_element(int z) noexcept {
    for (int elem_z : kSchneiderTargetElements) {
        if (elem_z == z) return true;
    }
    return false;
}

[[nodiscard]] inline bool is_valid_elemental_target(int z) noexcept {
    return z >= 1 && z <= 100;
}

// ---------------------------------------------------------------------------
// Strict CINEL03 lookup contract (production):
//  - Exact key only: (projectile_Z, projectile_A, target_element_Z).
//  - NO target alias, NO closest-Z fallback, NO water/O substitution.
//  - NO unbounded endpoint clamp / nearest-neighbor: the query must bracket
//    inside the exact channel domain [energy_min, energy_max], and the
//    bracketing node gap must not exceed a fixed pre-declared threshold.
//  - In-domain queries between adjacent nodes use stochastic bracketing:
//      E0 <= Eq <= E1,  P(select E1) = (Eq - E0) / (E1 - E0).
//    No per-event kinetic-energy rescaling is applied.
// ---------------------------------------------------------------------------

enum class Cinel03LookupStatus : std::uint8_t {
    Hit = 0,
    MissingProjectile = 1,
    MissingTarget = 2,
    BelowEnergyDomain = 3,
    AboveEnergyDomain = 4,
    EnergyGapTooLarge = 5,
    EmptyNode = 6
};

struct Cinel03LookupResult {
    std::uint32_t event_index{0xFFFFFFFFU};
    std::uint32_t energy_node_index{0xFFFFFFFFU};
    Cinel03LookupStatus status{Cinel03LookupStatus::MissingProjectile};
    float query_energy_MeV_per_u{0.0F};
    float selected_energy_MeV_per_u{0.0F};
    float absolute_energy_mismatch_MeV_per_u{0.0F};
};

// Fixed pre-declared maximum bracketing-node gap. Chosen once from the
// audited C12 campaign (observed per-channel max gap ~2.8 MeV/u, see
// data/schneider/cinel03_c12_targets.channels.json); it is NOT relaxed
// after seeing production results. Secondary channels are sparser, so
// out-of-gap secondary queries are honestly reported as EnergyGapTooLarge
// instead of being silently clamped.
inline constexpr float kCinel03MaxAllowedNodeGapMeVperU = 5.0F;

// Exact-channel bounded stochastic-bracketing lookup. Device-safe (no
// exceptions, no dynamic allocation). u_bracket drives the E0/E1 choice,
// u_event drives the intra-node event pick; both must be in [0,1].
inline Cinel03LookupResult cinel03_lookup_event_device(
    const Cinel03EnergyNode* energy_nodes,
    std::uint32_t node_count,
    const std::uint32_t* event_offsets,
    const std::uint32_t* event_indices,
    std::uint32_t total_events,
    int proj_z, int proj_a, int target_z,
    float energy_MeV_per_u,
    float u_bracket, float u_event,
    float max_allowed_gap_MeV_per_u = kCinel03MaxAllowedNodeGapMeVperU) noexcept
{
    Cinel03LookupResult out{};
    out.query_energy_MeV_per_u = energy_MeV_per_u;
    if (energy_nodes == nullptr || event_offsets == nullptr ||
        event_indices == nullptr || node_count == 0) {
        out.status = Cinel03LookupStatus::MissingProjectile;
        return out;
    }

    // Projectile block: all nodes are sorted by (pz, pa, tz, energy).
    std::uint32_t p_low = 0;
    std::uint32_t p_high = node_count;
    while (p_low < p_high) {
        const std::uint32_t mid = p_low + (p_high - p_low) / 2;
        const auto& n = energy_nodes[mid];
        if (n.projectile_z < proj_z ||
            (n.projectile_z == proj_z && n.projectile_a < proj_a)) {
            p_low = mid + 1;
        } else {
            p_high = mid;
        }
    }
    const std::uint32_t proj_first = p_low;
    p_high = node_count;
    while (p_low < p_high) {
        const std::uint32_t mid = p_low + (p_high - p_low) / 2;
        const auto& n = energy_nodes[mid];
        if (n.projectile_z == proj_z && n.projectile_a == proj_a) {
            p_low = mid + 1;
        } else {
            p_high = mid;
        }
    }
    const std::uint32_t proj_last = p_low;
    if (proj_first >= proj_last) {
        out.status = Cinel03LookupStatus::MissingProjectile;
        return out;
    }

    // Exact target sub-block. NO closest-Z fallback: absence is MissingTarget.
    std::uint32_t t_low = proj_first;
    std::uint32_t t_high = proj_last;
    while (t_low < t_high) {
        const std::uint32_t mid = t_low + (t_high - t_low) / 2;
        if (energy_nodes[mid].target_element_z < target_z) {
            t_low = mid + 1;
        } else {
            t_high = mid;
        }
    }
    const std::uint32_t channel_first = t_low;
    t_high = proj_last;
    while (t_low < t_high) {
        const std::uint32_t mid = t_low + (t_high - t_low) / 2;
        if (energy_nodes[mid].target_element_z == target_z) {
            t_low = mid + 1;
        } else {
            t_high = mid;
        }
    }
    const std::uint32_t channel_last = t_low;
    if (channel_first >= channel_last) {
        out.status = Cinel03LookupStatus::MissingTarget;
        return out;
    }

    const float channel_min = energy_nodes[channel_first].collision_energy_MeV_per_u;
    const float channel_max = energy_nodes[channel_last - 1].collision_energy_MeV_per_u;
    if (!(energy_MeV_per_u >= channel_min)) {
        out.status = Cinel03LookupStatus::BelowEnergyDomain;
        out.selected_energy_MeV_per_u = channel_min;
        out.absolute_energy_mismatch_MeV_per_u = channel_min - energy_MeV_per_u;
        return out;
    }
    if (energy_MeV_per_u > channel_max) {
        out.status = Cinel03LookupStatus::AboveEnergyDomain;
        out.selected_energy_MeV_per_u = channel_max;
        out.absolute_energy_mismatch_MeV_per_u = energy_MeV_per_u - channel_max;
        return out;
    }

    // Bracket: E0 = greatest node <= Eq, E1 = smallest node >= Eq.
    std::uint32_t e_low = channel_first;
    std::uint32_t e_high = channel_last;
    while (e_low < e_high) {
        const std::uint32_t mid = e_low + (e_high - e_low) / 2;
        if (energy_nodes[mid].collision_energy_MeV_per_u < energy_MeV_per_u) {
            e_low = mid + 1;
        } else {
            e_high = mid;
        }
    }
    std::uint32_t node_e1 = e_low;
    if (node_e1 >= channel_last) {
        node_e1 = channel_last - 1;
    }
    std::uint32_t node_e0 = node_e1;
    if (node_e1 > channel_first &&
        energy_nodes[node_e1].collision_energy_MeV_per_u > energy_MeV_per_u) {
        node_e0 = node_e1 - 1;
    }
    const float e0 = energy_nodes[node_e0].collision_energy_MeV_per_u;
    const float e1 = energy_nodes[node_e1].collision_energy_MeV_per_u;

    std::uint32_t chosen_node = node_e0;
    if (e1 > e0) {
        const float gap = e1 - e0;
        if (gap > max_allowed_gap_MeV_per_u) {
            out.status = Cinel03LookupStatus::EnergyGapTooLarge;
            out.selected_energy_MeV_per_u = (energy_MeV_per_u - e0 <= e1 - energy_MeV_per_u) ? e0 : e1;
            out.absolute_energy_mismatch_MeV_per_u =
                (energy_MeV_per_u - e0 <= e1 - energy_MeV_per_u) ? (energy_MeV_per_u - e0) : (e1 - energy_MeV_per_u);
            return out;
        }
        const float p_up = (energy_MeV_per_u - e0) / gap;
        const float ub = u_bracket < 0.0F ? 0.0F : (u_bracket >= 1.0F ? 0.9999999F : u_bracket);
        chosen_node = (ub < p_up) ? node_e1 : node_e0;
    }

    const std::uint32_t first_event = event_offsets[chosen_node];
    const std::uint32_t last_event = event_offsets[chosen_node + 1];
    if (first_event >= last_event || last_event > total_events) {
        out.status = Cinel03LookupStatus::EmptyNode;
        out.energy_node_index = chosen_node;
        out.selected_energy_MeV_per_u = energy_nodes[chosen_node].collision_energy_MeV_per_u;
        float mismatch = energy_MeV_per_u - out.selected_energy_MeV_per_u;
        out.absolute_energy_mismatch_MeV_per_u = mismatch < 0.0F ? -mismatch : mismatch;
        return out;
    }
    const std::uint32_t count = last_event - first_event;
    const float ue = u_event < 0.0F ? 0.0F : (u_event >= 1.0F ? 0.9999999F : u_event);
    std::uint32_t pick = static_cast<std::uint32_t>(ue * static_cast<float>(count));
    if (pick >= count) {
        pick = count - 1;
    }
    out.event_index = event_indices[first_event + pick];
    out.energy_node_index = chosen_node;
    out.status = Cinel03LookupStatus::Hit;
    out.selected_energy_MeV_per_u = energy_nodes[chosen_node].collision_energy_MeV_per_u;
    float mismatch = energy_MeV_per_u - out.selected_energy_MeV_per_u;
    out.absolute_energy_mismatch_MeV_per_u = mismatch < 0.0F ? -mismatch : mismatch;
    return out;
}

// Legacy index-only wrapper. Exact-target + bounded-domain semantics, no
// fallback. The single uniform drives both the bracket choice and the
// intra-node pick. Prefer cinel03_lookup_event_device for production, which
// reports the failure reason instead of collapsing it to 0xFFFFFFFF.
inline std::uint32_t cinel03_find_event_device(
    const Cinel03EnergyNode* energy_nodes,
    std::uint32_t node_count,
    const std::uint32_t* event_offsets,
    const std::uint32_t* event_indices,
    std::uint32_t total_events,
    int proj_z, int proj_a, int target_z,
    float energy, float /*tolerance_ignored_use_fixed_gap*/, float u01) noexcept
{
    const auto result = cinel03_lookup_event_device(
        energy_nodes, node_count, event_offsets, event_indices, total_events,
        proj_z, proj_a, target_z, energy, u01, u01);
    return result.status == Cinel03LookupStatus::Hit ? result.event_index : 0xFFFFFFFFU;
}

class InelasticPackageV3Table {
public:
    static constexpr std::uint64_t invalid = std::numeric_limits<std::uint64_t>::max();

    static InelasticPackageV3Table from_binary(const std::filesystem::path& path);
    void to_binary(const std::filesystem::path& path) const;

    [[nodiscard]] const std::vector<Cinel03CellIndex>& cells() const noexcept;
    [[nodiscard]] const std::vector<Cinel03InteractionRecord>& interactions() const noexcept;
    [[nodiscard]] const std::vector<Cinel03ProductRecord>& products() const noexcept;
    [[nodiscard]] const std::vector<Cinel03EnergyNode>& energy_nodes() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& event_offsets() const noexcept;
    [[nodiscard]] const std::vector<std::uint64_t>& event_indices() const noexcept;

    struct ChannelDomain {
        bool found_projectile{false};
        bool found_target{false};
        float energy_min_MeV_per_u{0.0F};
        float energy_max_MeV_per_u{0.0F};
        std::size_t node_count{0};
        float maximum_node_gap_MeV_per_u{0.0F};
    };

    [[nodiscard]] ChannelDomain channel_domain(
        int projectile_z, int projectile_a, int target_element_z) const noexcept;

    // Strict host lookup with the exact device semantics (exact target, bounded
    // domain, stochastic bracketing). u_bracket/u_event must be in [0,1].
    [[nodiscard]] Cinel03LookupResult lookup_event(
        int projectile_z, int projectile_a, int target_element_z,
        float collision_energy_MeV_per_u,
        float u_bracket, float u_event,
        float max_allowed_gap_MeV_per_u = kCinel03MaxAllowedNodeGapMeVperU) const noexcept;

    // Primary event lookup by elemental target Z (legacy index API).
    // Exact-target + bounded-domain semantics; tolerance is accepted for
    // backward compatibility but the fixed channel gap rule governs.
    [[nodiscard]] std::uint64_t find_event(
        int projectile_z, int projectile_a, int target_element_z,
        float collision_energy_MeV_per_u, float maximum_energy_mismatch_MeV_per_u,
        float u01, bool audit_mode = false, std::uint64_t* missing_target_counter = nullptr) const;

    [[nodiscard]] const Cinel03CellIndex* find_cell(
        int projectile_z, int projectile_a, int target_element_z,
        float collision_energy_MeV_per_u, float maximum_energy_mismatch_MeV_per_u) const noexcept;

    [[nodiscard]] const Cinel03InteractionRecord* interaction(
        const Cinel03CellIndex& cell, std::uint64_t offset) const noexcept;

    [[nodiscard]] Cinel03DeviceTables make_device_tables() const;

    [[nodiscard]] Cinel03FixedReplayView fixed_replay(
        std::uint64_t event_index, const Cinel03DeviceTables& compact) const noexcept;

    [[nodiscard]] const Cinel03ProductRecord* products_for(
        const Cinel03InteractionRecord& interaction) const noexcept;

    [[nodiscard]] std::uint32_t product_offset(
        const Cinel03InteractionRecord& interaction) const noexcept;

    [[nodiscard]] float minimum_energy_MeV_per_u() const noexcept;
    [[nodiscard]] float energy_bin_width_MeV_per_u() const noexcept;
    [[nodiscard]] std::uint64_t minimum_events_per_bin() const noexcept;
    [[nodiscard]] const std::string& campaign_uuid() const noexcept;

    // Builder / Mutator for synthetic test packages
    void set_metadata(float min_energy_MeV_per_u, float bin_width_MeV_per_u,
                      std::uint64_t min_events_per_bin, const std::string& uuid);
    void add_event(const Cinel03InteractionRecord& interaction,
                   const std::vector<Cinel03ProductRecord>& products);
    void finalize();

private:
    float minimum_energy_MeV_per_u_{0.0F};
    float energy_bin_width_MeV_per_u_{1.0F};
    std::uint64_t minimum_events_per_bin_{0};
    std::string campaign_uuid_{};
    std::vector<Cinel03CellIndex> cells_{};
    std::vector<Cinel03InteractionRecord> interactions_{};
    std::vector<Cinel03ProductRecord> products_{};
    std::vector<std::uint32_t> product_offsets_{};
    std::vector<Cinel03EnergyNode> energy_nodes_{};
    std::vector<std::uint64_t> event_offsets_{};
    std::vector<std::uint64_t> event_indices_{};
};

} // namespace carbon
