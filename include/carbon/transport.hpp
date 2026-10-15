#pragma once

#include "carbon/cross_section.hpp"
#include "carbon/particle.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/schneider_ct_device_context.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace carbon {

class SyclTransportContext;

struct MinibeamDiagnostics {
    static constexpr std::size_t touched_energy_bin_count = 12;
    bool enabled{false};
    std::uint64_t incident_histories{0};
    std::uint64_t direct_air_slit_histories{0};
    std::uint64_t copper_touched_histories{0};
    std::uint64_t copper_nuclear_interactions{0};
    std::uint64_t copper_generated_direct_secondaries{0};
    std::uint64_t copper_charged_survivors{0};
    std::uint64_t water_entrance_primary{0};
    double beamline_removed_energy_MeV{0.0};
    double direct_air_primary_energy_MeV{0.0};
    double copper_touched_primary_energy_MeV{0.0};
    // C, B, Be, Li, He, p, d, t, other.
    std::array<std::uint64_t, 9> copper_charged_survivors_by_species{};
    std::array<double, 9> copper_charged_survivor_energy_by_species_MeV{};
    // Every direct-fragment Copper reinteraction, including reactions whose
    // final-state lookup is disabled or misses its strict domain.
    std::array<std::uint64_t, 9> copper_fragment_absorptions_by_species{};
    std::array<double, 9> copper_fragment_absorbed_energy_by_species_MeV{};
    // Straight-ahead Copper material length after the terminal interaction.
    // Finite slit air and cylindrical side escape are excluded; subsequent
    // stochastic scattering is intentionally not predicted by this audit.
    std::array<double, 9>
        copper_fragment_absorption_straight_copper_path_by_species_mm{};
    std::uint64_t copper_fragment_cascade_interactions{0};
    std::uint64_t copper_fragment_cascade_lookup_hits{0};
    std::array<std::uint64_t, 6> copper_fragment_cascade_lookup_misses{};
    static constexpr std::size_t copper_cascade_generation_count = 3;
    std::array<std::uint64_t, copper_cascade_generation_count>
        copper_fragment_cascade_interactions_by_generation{};
    std::array<std::uint64_t, copper_cascade_generation_count>
        copper_fragment_cascade_hits_by_generation{};
    std::array<std::uint64_t, copper_cascade_generation_count>
        copper_fragment_cascade_misses_by_generation{};
    std::uint64_t copper_fragment_cascade_generated_charged{0};
    std::uint64_t copper_fragment_cascade_generated_neutral{0};
    std::uint64_t copper_fragment_cascade_generated_unsupported{0};
    std::uint64_t copper_fragment_cascade_queue_overflows{0};
    double copper_fragment_cascade_local_energy_MeV{0.0};
    double copper_fragment_cascade_untracked_energy_MeV{0.0};
    // C, B, Be, Li, He, p, d, t, other.  Optical depth is accumulated for
    // products at the configured terminal Copper generation whose further
    // reactions are deliberately suppressed by the convergence cap.
    std::array<double, 9>
        copper_fragment_ignored_nuclear_optical_depth_by_species{};
    std::array<std::uint64_t, 9>
        copper_fragment_terminal_tracks_by_species{};
    // Sum over terminal tracks of 1-exp(-integrated optical depth).
    std::array<double, 9>
        copper_fragment_ignored_reaction_probability_by_species{};
    std::array<std::uint64_t, 9>
        copper_fragment_cascade_lookup_misses_by_species{};
    static constexpr std::size_t fragment_lookup_energy_bin_count = 16;
    std::array<std::uint64_t, fragment_lookup_energy_bin_count>
        copper_fragment_cascade_lookup_misses_by_energy{};
    static constexpr std::size_t fragment_miss_za_count = 7 * 13 + 1;
    static constexpr std::size_t fragment_miss_reason_count = 6;
    static constexpr std::size_t fragment_miss_joint_energy_bin_count = 128;
    static constexpr std::size_t fragment_miss_joint_cell_count =
        copper_cascade_generation_count * fragment_miss_za_count *
        fragment_miss_reason_count * fragment_miss_joint_energy_bin_count;
    std::vector<std::uint64_t> copper_fragment_miss_joint_counts{};
    std::vector<std::uint64_t> copper_fragment_miss_joint_input_energy_keV{};
    std::vector<std::uint64_t> copper_fragment_miss_joint_collision_depth_um{};
    std::vector<std::uint64_t> copper_fragment_miss_joint_remaining_copper_um{};
    double copper_fragment_cascade_actual_input_energy_MeV{0.0};
    double copper_fragment_cascade_selected_input_energy_MeV{0.0};
    double copper_fragment_cascade_replay_output_energy_MeV{0.0};
    double copper_fragment_cascade_selection_mismatch_MeV{0.0};
    double copper_fragment_cascade_closure_mismatch_MeV{0.0};
    double copper_fragment_cascade_mass_energy_mismatch_MeV{0.0};
    std::uint64_t copper_fragment_cascade_baryon_mismatch{0};
    std::array<std::uint64_t, touched_energy_bin_count>
        copper_touched_primary_energy_histogram{};
    double energy_sum_MeV{0.0};
    double energy_squared_sum_MeV2{0.0};
};

// Diagnostic-only track-local reduction. No values feed particle transport or
// dose scorers. Commit once after the stepping loop, including replay breaks.
struct ContinuousSpeciesTrackTally {
    float all_MeV{0.0F};
    float fov_MeV{0.0F};
    void add_all(float value) noexcept {
        if (std::isfinite(value) && value > 0.0F) all_MeV += value;
    }
    void add_fov(float value) noexcept {
        if (std::isfinite(value) && value > 0.0F) fov_MeV += value;
    }
};

inline bool replay_vertex_in_scoring_box(
    const std::array<float,3>& point, const std::array<float,3>& minimum,
    const std::array<float,3>& maximum) noexcept {
    for (std::size_t i=0; i<3; ++i)
        if (!(point[i]>=minimum[i] && point[i]<maximum[i])) return false;
    return true;
}

struct IonSpeciesLedgerSchema {
    static constexpr std::size_t species_count = 18;
    static constexpr std::size_t metric_count = 13;
    static constexpr std::size_t terminal_reason_count = 6;
    enum Metric : std::size_t {
        queued_birth_kinetic = 0, continuous_deposit_all = 1,
        continuous_deposit_fov = 2, nuclear_local_deposit_all = 3,
        nuclear_local_deposit_fov = 4, terminal_deposit_all = 5,
        terminal_deposit_fov = 6, boundary_escape_kinetic = 7,
        reaction_export_kinetic = 8, step_limit_escape_kinetic = 9,
        reaction_import_kinetic = 10,
        // Informational CINEL03 successful-replay subsets, not extra sinks.
        cinel03_replay_input_fov = 11, cinel03_replay_step_dE_fov = 12
    };
    enum TerminalReason : std::size_t {
        initial_below_cutoff = 0, reaction_killed = 1, terminal_deposit = 2,
        boundary_escape = 3, step_limit = 4, continuous_stop = 5
    };
};

// Secondary-transport exposure ledger used by the reaction-survival audit.
// The path sums are keyed by the isotope and transport generation of each
// secondary track, then by its step-start kinetic energy. Coverage is kept
// separate from generation eligibility so an absent hazard is not silently
// interpreted as a physical zero-rate segment.

// A secondary inelastic hazard suppresses the normal per-step MCS only when a
// valid CINEL02 final state was actually replayed. Lookup misses and invalid
// package records are null collisions: the post-EM state is retained and the
// same MCS that a non-nuclear step would receive must still be applied.
constexpr bool ion_should_apply_secondary_mcs(
    const bool secondary_inelastic, const bool replay_succeeded,
    const bool enable_multiple_scattering, const float kinetic_energy_MeV,
    const float energy_cutoff_MeV) noexcept {
    return enable_multiple_scattering && kinetic_energy_MeV > energy_cutoff_MeV &&
           (!secondary_inelastic || !replay_succeeded);
}

// Collision-step voxel commit primitive. Mirrors the paired voxel-scorer
// guard exactly: in-grid ⟺ the deposit would be scored. The kernel calls it
// at every secondary collision-step voxel commit (replay hit, lookup miss,
// common path) so no early break can bypass the step dE. Unit-tested.
inline void secondary_step_voxel_commit(float& pending_voxel_MeV,
                                        const bool enable_voxel_scoring,
                                        const int cur_voxel,
                                        const float amount_MeV) noexcept {
    if (enable_voxel_scoring && cur_voxel >= 0) pending_voxel_MeV += amount_MeV;
}

struct EnergyAccountingLedger {
    double E_transported_secondaries{0.0};
    // Split nuclear-vertex ledger (populated on the Schneider CT path; the
    // broad `untracked` sink must NOT be used to absorb these terms and then
    // be called physical closure).
    double E_charged_birth{0.0};
    double E_neutral_product_kinetic{0.0};
    double E_reaction_q_residual{0.0};
    double E_unsupported_charged{0.0};
    double E_be6_kill{0.0};
    double E_lookup_failure{0.0};
    double E_out_of_domain{0.0};

};

// Independent Schneider-CT nuclear diagnostics. This is intentionally NOT
// every counter below has a
// unique schema slot (see SchneiderDiagSlot) and the host aggregates the
// flat device buffer into these named fields. "100% hit rate" is defined
// strictly as events_replayed / hazards, never over rate queries or steps.
struct SchneiderNuclearDiagnostics {
    std::uint64_t primary_rate_queries{0};
    std::uint64_t primary_hazards{0};
    std::uint64_t primary_exact_target_hits{0};
    std::uint64_t primary_missing_projectile{0};
    std::uint64_t primary_missing_target{0};
    std::uint64_t primary_below_domain{0};
    std::uint64_t primary_above_domain{0};
    std::uint64_t primary_energy_gap_misses{0};
    std::uint64_t primary_empty_nodes{0};

    std::uint64_t primary_events_replayed{0};
    std::uint64_t primary_charged_products_born{0};
    std::uint64_t primary_charged_products_queued{0};
    std::uint64_t primary_charged_cutoff_kills{0};
    std::uint64_t primary_be6_kills{0};
    std::uint64_t primary_queue_overflows{0};

    std::uint64_t secondary_tracks_started{0};
    std::uint64_t secondary_steps{0};
    // Diagnostic partition of secondary steps on which MCS was sampled.
    std::uint64_t secondary_fe_c12_steps{0};
    std::uint64_t secondary_fe_he4_steps{0};
    std::uint64_t secondary_fe_pdt_steps{0};
    std::uint64_t secondary_fe_other_charged_steps{0};
    std::uint64_t secondary_highland_steps{0};
    std::uint64_t secondary_rate_queries{0};
    std::uint64_t secondary_hazards{0};
    std::uint64_t secondary_exact_target_hits{0};
    std::uint64_t secondary_missing_projectile{0};
    std::uint64_t secondary_missing_target{0};
    std::uint64_t secondary_below_domain{0};
    std::uint64_t secondary_above_domain{0};
    std::uint64_t secondary_energy_gap_misses{0};
    std::uint64_t secondary_empty_nodes{0};
    std::uint64_t secondary_events_replayed{0};
    std::uint64_t secondary_charged_products_born{0};
    std::uint64_t secondary_charged_products_queued{0};
    std::uint64_t secondary_charged_cutoff_kills{0};
    std::uint64_t secondary_be6_kills{0};
    std::uint64_t secondary_queue_overflows{0};
    std::uint64_t secondary_stopped_before_replay{0};
    // Sampled candidates resolved as post-EM null collisions (all channel
    // partials zero at the collision-point energy): track continues with
    // post-EM energy, no lookup, no local deposit, no energy loss.
    // Declared research approximation; reported, never failed.
    std::uint64_t secondary_post_em_null_collisions{0};
    // Involved post-EM track kinetic summed over null draws (MeV).
    double secondary_post_em_null_energy_MeV{0.0};
    std::uint64_t primary_post_em_null_collisions{0};

    std::uint64_t be6_topas_compat_kills{0};
    // Per-step nuclear-evaluation count for registry-unknown projectiles
    // (one increment per transport step evaluated, NOT per particle).
    // Coverage decisions must use unsupported_projectile_tracks below.
    std::uint64_t unsupported_projectile_steps{0};
    // Generation-eligible tracks whose projectile (Z/A) has no entry in the
    // secondary rate-table registry. Counted once per track at track start.
    std::uint64_t unsupported_projectile_tracks{0};
    // Of those, Be6 tracks (frozen TopasCompatKill policy identity).
    std::uint64_t unsupported_be6_tracks{0};
    double unsupported_projectile_birth_energy_MeV{0.0};
    // Bounded per-record logs dropped (capacity overflow) counts.
    std::uint64_t miss_log_dropped{0};
    std::uint64_t unsupported_log_dropped{0};
    std::uint64_t unsupported_targets{0};
    std::uint64_t queue_overflows{0};

    double primary_selected_energy_mismatch_sum{0.0};
    double primary_selected_energy_mismatch_max{0.0};
    double secondary_selected_energy_mismatch_sum{0.0};
    double secondary_selected_energy_mismatch_max{0.0};

    double lookup_failure_energy_MeV{0.0};
    double be6_kill_energy_MeV{0.0};
    double neutral_product_kinetic_MeV{0.0};
    double reaction_q_residual_MeV{0.0};

    [[nodiscard]] double primary_hit_rate() const noexcept {
        return primary_hazards == 0
                   ? 1.0
                   : static_cast<double>(primary_events_replayed) /
                         static_cast<double>(primary_hazards);
    }
    [[nodiscard]] double secondary_hit_rate() const noexcept {
        return secondary_hazards == 0
                   ? 1.0
                   : static_cast<double>(secondary_events_replayed) /
                         static_cast<double>(secondary_hazards);
    }
};

// Flat device-counter schema for SchneiderNuclearDiagnostics. The device
// buffer layout is slot-major: slot * 1 uint64 entry. Host aggregation maps
// each slot to the named field; never interpret raw indices elsewhere.
enum class SchneiderDiagSlot : std::uint32_t {
    PrimaryRateQueries = 0,
    PrimaryHazards = 1,
    PrimaryExactTargetHits = 2,
    PrimaryMissingProjectile = 3,
    PrimaryMissingTarget = 4,
    PrimaryBelowDomain = 5,
    PrimaryAboveDomain = 6,
    PrimaryEnergyGapMisses = 7,
    PrimaryEmptyNodes = 8,
    PrimaryEventsReplayed = 9,
    PrimaryChargedBorn = 10,
    PrimaryChargedQueued = 11,
    PrimaryChargedCutoffKills = 12,
    PrimaryBe6Kills = 13,
    PrimaryQueueOverflows = 14,
    SecondaryTracksStarted = 15,
    SecondarySteps = 16,
    SecondaryRateQueries = 17,
    SecondaryHazards = 18,
    SecondaryExactTargetHits = 19,
    SecondaryMissingProjectile = 20,
    SecondaryMissingTarget = 21,
    SecondaryBelowDomain = 22,
    SecondaryAboveDomain = 23,
    SecondaryEnergyGapMisses = 24,
    SecondaryEmptyNodes = 25,
    SecondaryEventsReplayed = 26,
    SecondaryChargedBorn = 27,
    SecondaryChargedQueued = 28,
    SecondaryChargedCutoffKills = 29,
    SecondaryBe6Kills = 30,
    SecondaryQueueOverflows = 31,
    // Sampled collision whose post-EM collision-point energy is already at
    // or below cutoff: the interaction never occurs (continuous stopping
    // owns the energy); counted explicitly so hazards still close exactly.
    SecondaryStoppedBeforeReplay = 32,
    Be6TopasCompatKills = 33,
    // Registry-unknown projectile evaluations: one increment PER TRANSPORT
    // STEP (not per particle). See unsupported_projectile_tracks for the
    // per-track count used by coverage gates.
    UnsupportedProjectileSteps = 34,
    UnsupportedTargets = 35,
    QueueOverflows = 36,
    // Fixed-point secondary energy sub-ledger (units: micro-MeV, i.e. value
    // 1e6 == 1 MeV). A single-address float accumulator would silently drop
    // sub-MeV deposits once the running sum exceeds ~1e7 MeV; uint64 atomic
    // add is exact. Host converts with /1e6.
    SecondaryDepositedMicroMeV = 37,
    SecondaryEscapedMicroMeV = 38,
    // Generation-eligible tracks with registry-unknown projectile, counted
    // once per track at track start (not per step).
    UnsupportedProjectileTracks = 39,
    // Birth kinetic energy of those tracks, fixed-point micro-MeV.
    UnsupportedProjectileBirthEnergyMicroMeV = 40,
    // Of those, tracks with the frozen TopasCompatKill identity (Be6):
    // policy-covered (continuous slowing only, never a hazard), reported
    // separately so the coverage gate can exempt them explicitly.
    UnsupportedBe6Tracks = 41,
    // Sampled nuclear candidate whose post-EM collision-point partials are
    // all zero (slowing left every channel domain between hazard sampling
    // and the collision point): no package query is issued, the track keeps
    // its post-EM energy/position and continues transport. Research
    // approximation (E_h/E_c distribution skew is NOT integrated); reported,
    // never a lookup failure, never a stop.
    SecondaryPostEmNullCollisions = 42,
    // Primary-side counterpart (hazard fired, slowing emptied all C12
    // channels before the collision point): same no-query continuation.
    PrimaryPostEmNullCollisions = 43,
    SecondaryFeC12Steps = 44,
    SecondaryFeHe4Steps = 45,
    SecondaryFePdtSteps = 46,
    SecondaryFeOtherChargedSteps = 47,
    SecondaryHighlandSteps = 48,
    Count = 49
};

// Device float accumulator slots paired with SchneiderDiagSlot. Sum slots
// use relaxed atomic add; max slots use atomic compare-exchange.
enum class SchneiderFloatSlot : std::uint32_t {
    PrimaryMismatchSum = 0,
    PrimaryMismatchMax = 1,
    SecondaryMismatchSum = 2,
    SecondaryMismatchMax = 3,
    LookupFailureEnergy = 4,
    Be6KillEnergy = 5,
    NeutralProductKinetic = 6,
    ReactionQResidual = 7,
    UnsupportedProductEnergy = 8,
    SecondaryTransportBirthEnergy = 9,
    OutOfDomainEnergy = 10,
    // Post-EM null-collision involved energy (post-EM track kinetic at the
    // null draw), plain sum in MeV (float slot, not fixed-point: diagnostic).
    PostEmNullEnergy = 11,
    Count = 12
};

// One failed CINEL03 lookup (primary or secondary) on the Schneider path.
// Device POD (36 B); channel Emin/Emax, partial rates and demand are joined
// offline from the package + rate tables (deterministic given the fields).
struct SchneiderMissRecord {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::int16_t target_z{0};
    // Lookup outcome: 0=below, 1=above, 2=missing projectile,
    // 3=missing target, 4=gap too large, 5=empty node.
    std::uint8_t status{0};
    std::uint8_t section_id{0};
    std::uint8_t is_primary{0};
    std::uint8_t generation{0};
    std::uint16_t pad{0};
    float query_energy_MeV_per_u{0.0F};   // post-EM lookup energy
    float step_dE_MeV{0.0F};              // this step's EM loss (pre-EM = query + dE/A)
    float total_rate_per_mm{0.0F};        // macro total at sampling (density x mass)
    float hazard_step_mm{0.0F};           // sampled collision distance (0 if n/a)
    float incident_energy_MeV{0.0F};      // parent kinetic at vertex (post-EM)
    float birth_energy_MeV{0.0F};         // track-start kinetic
};
static_assert(sizeof(SchneiderMissRecord) == 36, "miss record layout");

// One generation-eligible track whose projectile has no rate-table entry.
// Device POD (24 B); birth position allows offline CT-section lookup.
struct SchneiderUnsupportedTrack {
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    std::uint16_t generation{0};
    std::uint16_t pad{0};
    float birth_energy_MeV{0.0F};
    float birth_x_mm{0.0F};
    float birth_y_mm{0.0F};
    float birth_z_mm{0.0F};
};
static_assert(sizeof(SchneiderUnsupportedTrack) == 24, "unsupported-track layout");

struct TransportResult {
    bool conditional_copper_response_enabled{false};
    // True only when the dedicated local-electron CT proton primary ran.
    bool primary_first_order_delta_enabled{false};
    bool carbon_primary_first_order_delta_enabled{false};
    bool carbon_secondary_first_order_delta_enabled{false};
    bool ct_primary_first_order_restricted_enabled{false};
    bool ct_secondary_first_order_restricted_enabled{false};
    bool ct_broadbeam_step_cap_enabled{false};

    EnergyAccountingLedger energy_ledger{};
    SchneiderNuclearDiagnostics schneider_diagnostics{};
    // Bounded per-record logs (empty on water path / when logging disabled).
    std::vector<SchneiderMissRecord> schneider_miss_log{};
    std::vector<SchneiderUnsupportedTrack> schneider_unsupported_tracks{};
    // Fixed Schneider mismatch/energy float slots on device. Always derived
    // from the enum so future slots cannot be silently truncated by a stale
    // constant here.
    static constexpr std::size_t schneider_float_slot_count =
        static_cast<std::size_t>(SchneiderFloatSlot::Count);
    static_assert(schneider_float_slot_count ==
                      static_cast<std::size_t>(SchneiderFloatSlot::Count),
                  "schneider_float_slot_count must track SchneiderFloatSlot::Count");
    static constexpr std::size_t species_ledger_species_count =
        IonSpeciesLedgerSchema::species_count;
    static constexpr std::size_t species_ledger_metric_count =
        IonSpeciesLedgerSchema::metric_count;
    std::vector<double> deposited_energy_MeV;
    std::vector<double> voxel_deposited_energy_MeV;
    std::vector<double> in_fov_deposited_energy_MeV;
    std::vector<double> primary_deposited_energy_MeV;
    std::vector<double> secondary_carbon_deposited_energy_MeV;
    std::vector<double> secondary_boron_deposited_energy_MeV;
    std::vector<double> secondary_beryllium_deposited_energy_MeV;
    std::vector<double> secondary_lithium_deposited_energy_MeV;
    std::vector<double> secondary_helium_deposited_energy_MeV;
    std::vector<double> secondary_proton_deposited_energy_MeV;
    std::vector<double> secondary_other_charged_deposited_energy_MeV;
    double initial_energy_MeV{0.0};
    double total_deposited_energy_MeV{0.0};
    double escaped_energy_MeV{0.0};
    // Energy lost/deposited/terminated before the scored phantom by the
    // optional minibeam beamline.
    double beamline_removed_energy_MeV{0.0};
    double untracked_nuclear_energy_MeV{0.0};
    std::uint64_t nuclear_interactions{0};
    // Step 12 primary-only & validation counters
    std::uint64_t primary_inelastic_terminated_count{0};
    std::uint64_t primary_escaped_ct_count{0};
    std::uint64_t primary_stopped_count{0};
    std::uint64_t primary_other_terminal_count{0};
    double primary_inelastic_removed_kinetic_MeV{0.0};
    double primary_other_terminal_kinetic_MeV{0.0};
    double primary_cutoff_stopped_energy_MeV{0.0};
    std::array<double, species_ledger_species_count * species_ledger_metric_count>
        ion_species_transport_ledger_MeV{};
    // Explicit in-grid/outside-grid deposited-energy split (global MeV).
    // Every deposited-ledger credit is mirrored here with that site's paired
    // voxel-scorer guard, so total ≈ in + outside and voxel_sum ≈ in_grid.
    double in_grid_deposited_energy_MeV{0.0};
    double outside_grid_deposited_energy_MeV{0.0};
    std::array<std::uint64_t,
               species_ledger_species_count *
                   IonSpeciesLedgerSchema::terminal_reason_count>
        ion_species_terminal_reason_counts{};
    std::uint64_t secondary_queue_overflow{0};
    double generated_direct_secondary_energy_MeV{0.0};
    double queued_secondary_energy_MeV{0.0};
    double secondary_queue_overflow_energy_MeV{0.0};
    double nuclear_energy_not_in_direct_secondaries_MeV{0.0};
    std::uint64_t primary_elastic_interactions{0};
    std::uint64_t secondary_elastic_interactions{0};
    std::uint64_t elastic_post_em_null_collisions{0};
    double elastic_local_deposited_energy_MeV{0.0};
    double elastic_queued_charged_energy_MeV{0.0};
    std::uint64_t elastic_queue_overflow{0};
    double elastic_queue_overflow_energy_MeV{0.0};
    double secondary_deposited_energy_MeV{0.0};
    double secondary_escaped_energy_MeV{0.0};
    std::uint64_t cascade_interactions{0};
    double cascade_selection_energy_distance_sum_MeVu{0.0};
    double cascade_selection_energy_distance_max_MeVu{0.0};
    std::uint64_t cascade_queue_overflow{0};
    double queued_cascade_energy_MeV{0.0};
    double cascade_nuclear_energy_MeV{0.0};
    std::uint64_t total_steps{0};
    double elapsed_seconds{0.0};
    double primary_kernel_seconds{0.0};
    double secondary_kernel_seconds{0.0};
    std::string backend;
    MinibeamDiagnostics minibeam;

    // Closure includes deposited, escaped and explicitly untracked energy.
    [[nodiscard]] double physical_relative_energy_balance_error() const noexcept;
};

class SyclTransportContext {
public:
    explicit SyclTransportContext(const std::string& device_name);
    ~SyclTransportContext();

    SyclTransportContext(SyclTransportContext&&) noexcept;
    SyclTransportContext& operator=(SyclTransportContext&&) noexcept;
    SyclTransportContext(const SyclTransportContext&) = delete;
    SyclTransportContext& operator=(const SyclTransportContext&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    template<class Pipeline>
    friend TransportResult transport_pipeline(
        const TransportConfig&, const StoppingPowerTable&,
        const std::string&, SyclTransportContext*);
    friend TransportResult transport_sycl(
        const TransportConfig&, const StoppingPowerTable&,
        const std::string&, SyclTransportContext*);
};

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const std::string& device_name,
                               SyclTransportContext* context = nullptr);
std::string describe_sycl_device(const std::string& device_name);

}  // namespace carbon
