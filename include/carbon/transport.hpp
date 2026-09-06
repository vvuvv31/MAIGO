#pragma once

#include "carbon/cross_section.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/transport_profile.hpp"
#include "carbon/schneider_ct_device_context.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace carbon {

class SyclTransportContext;

struct MinibeamDiagnostics {
    static constexpr std::size_t slit_count = 15;
    static constexpr std::size_t touched_energy_bin_count = 12;
    static constexpr std::size_t fragment_energy_bin_count = 20;
    bool enabled{false};
    std::uint64_t incident_histories{0};
    std::uint64_t direct_air_slit_histories{0};
    std::uint64_t copper_touched_histories{0};
    std::uint64_t copper_nuclear_interactions{0};
    std::uint64_t copper_generated_direct_secondaries{0};
    std::uint64_t copper_charged_survivors{0};
    std::uint64_t copper_neutral_survivors{0};
    std::uint64_t water_entrance_primary{0};
    double beamline_removed_energy_MeV{0.0};
    double copper_charged_survivor_energy_MeV{0.0};
    double copper_neutral_survivor_energy_MeV{0.0};
    double direct_air_primary_energy_MeV{0.0};
    double copper_touched_primary_energy_MeV{0.0};
    // C, B, Be, Li, He, p, d, t, other.
    std::array<std::uint64_t, 9> copper_charged_survivors_by_species{};
    std::array<double, 9> copper_charged_survivor_energy_by_species_MeV{};
    std::array<std::uint64_t, slit_count>
        water_entrance_primary_by_slit{};
    std::array<std::uint64_t, slit_count>
        collimator_entrance_primary_by_slit{};
    std::array<std::uint64_t, slit_count>
        direct_air_primary_by_slit{};
    std::array<std::uint64_t, touched_energy_bin_count>
        copper_touched_primary_energy_histogram{};
    // Water-entrance kinetic-energy spectra in 50 MeV bins. The final bin
    // includes overflow. These diagnostics expose the short-range tail that
    // controls the first few millimetres of minibeam dose.
    std::array<std::uint64_t, fragment_energy_bin_count>
        copper_deuteron_energy_histogram{};
    std::array<std::uint64_t, fragment_energy_bin_count>
        copper_triton_energy_histogram{};
    std::array<std::uint64_t, fragment_energy_bin_count>
        copper_helium_energy_histogram{};
    double energy_sum_MeV{0.0};
    double energy_squared_sum_MeV2{0.0};
    double x_sum_mm{0.0};
    double x_squared_sum_mm2{0.0};
    double y_sum_mm{0.0};
    double y_squared_sum_mm2{0.0};
    double direction_x_sum{0.0};
    double direction_x_squared_sum{0.0};
    double direction_y_sum{0.0};
    double direction_y_squared_sum{0.0};
};

enum class Cinel02UnstableIonPolicy : std::uint8_t {
    StableForTransport = 0,
    RejectUnsupported = 1,
    TopasCompatKill = 2,
    PromptDecayKernel = 3,
};

// Table-driven policy for product identities whose lifetime/data semantics are
// not represented by the normal condensed-history transport tables.  Keep the
// default transport behavior stable for every identity except the explicitly
// classified Be-6 reference case.
constexpr Cinel02UnstableIonPolicy cinel02_unstable_ion_policy(
    const int atomic_number, const int mass_number) noexcept {
    return atomic_number == 4 && mass_number == 6
        ? Cinel02UnstableIonPolicy::TopasCompatKill
        : Cinel02UnstableIonPolicy::StableForTransport;
}

constexpr bool cinel02_should_topas_compat_kill(
    const bool compatibility_mode, const int atomic_number,
    const int mass_number) noexcept {
    return compatibility_mode &&
           cinel02_unstable_ion_policy(atomic_number, mass_number) ==
               Cinel02UnstableIonPolicy::TopasCompatKill;
}

struct Cinel02SpeciesLedgerSchema {
    static constexpr std::size_t species_count = 18;
    static constexpr std::size_t metric_count = 11;
    static constexpr std::size_t terminal_reason_count = 6;
    enum Metric : std::size_t {
        queued_birth_kinetic = 0, continuous_deposit_all = 1,
        continuous_deposit_fov = 2, nuclear_local_deposit_all = 3,
        nuclear_local_deposit_fov = 4, terminal_deposit_all = 5,
        terminal_deposit_fov = 6, boundary_escape_kinetic = 7,
        reaction_export_kinetic = 8, step_limit_escape_kinetic = 9,
        reaction_import_kinetic = 10
    };
    enum TerminalReason : std::size_t {
        initial_below_cutoff = 0, reaction_killed = 1, terminal_deposit = 2,
        boundary_escape = 3, step_limit = 4, continuous_stop = 5
    };
};

struct Cinel02ReplayLedgerSchema {
    static constexpr std::size_t species_count = Cinel02SpeciesLedgerSchema::species_count;
    static constexpr std::size_t target_count = 2;
    static constexpr std::size_t generation_count = 3;
    // Fine replay occupancy bins (10 MeV/u) keep package support and misses
    // separable inside the broad 50 MeV/u physics ranges.
    static constexpr std::size_t energy_bin_count = 40;
    static constexpr float energy_bin_width_MeV_per_u = 10.0F;
    static constexpr std::size_t status_count = 5;
    static constexpr std::size_t outcome_count = 2;
    static constexpr std::size_t status_cell_count =
        species_count * target_count * generation_count * energy_bin_count;
    static constexpr std::size_t status_slot_count =
        status_cell_count * status_count;
    static constexpr std::size_t parent_outcome_cell_count =
        species_count * target_count * generation_count * outcome_count;
    static constexpr std::size_t transition_cell_count =
        species_count * species_count;
    enum Status : std::size_t {
        collision_candidate = 0,
        replay_valid = 1,
        replay_lookup_miss = 2,
        replay_invalid_event = 3,
        post_em_below_cutoff = 4
    };
};

// Secondary-transport exposure ledger used by the reaction-survival audit.
// The path sums are keyed by the isotope and transport generation of each
// secondary track, then by its step-start kinetic energy. Coverage is kept
// separate from generation eligibility so an absent hazard is not silently
// interpreted as a physical zero-rate segment.
struct Cinel02ExposureLedgerSchema {
    static constexpr std::size_t species_count =
        Cinel02SpeciesLedgerSchema::species_count;
    static constexpr std::size_t generation_count =
        Cinel02ReplayLedgerSchema::generation_count;
    static constexpr std::size_t energy_bin_count =
        Cinel02ReplayLedgerSchema::energy_bin_count;
    static constexpr float energy_bin_width_MeV_per_u =
        Cinel02ReplayLedgerSchema::energy_bin_width_MeV_per_u;
    static constexpr std::size_t cell_count =
        species_count * generation_count * energy_bin_count;
    static constexpr std::size_t sum_metric_count = 18;
    static constexpr std::size_t count_metric_count = 4;
    static constexpr std::size_t sum_slot_count = cell_count * sum_metric_count;
    static constexpr std::size_t count_slot_count = cell_count * count_metric_count;

    enum SumMetric : std::size_t {
        path_mm_total = 0,
        path_mm_generation_eligible = 1,
        path_mm_generation_blocked = 2,
        path_mm_rate_covered = 3,
        path_mm_rate_uncovered = 4,
        path_mm_h_uncovered = 5,
        path_mm_o_uncovered = 6,
        hazard_h = 7,
        hazard_o = 8,
        hazard_total = 9,
        // Counterfactual optical depth for generation-blocked steps; diagnostic only.
        hazard_blocked_h = 10,
        hazard_blocked_o = 11,
        hazard_blocked_total = 12,
        // Continuous-rate quadrature over the actual start/end energy of a
        // transport step. These are diagnostic-only and never feed the
        // runtime hazard sampler.
        path_mm_continuous_rate_covered = 13,
        path_mm_continuous_rate_uncovered = 14,
        hazard_continuous = 15,
        hazard_blocked_continuous = 16,
        stopping_loss_MeV = 17
    };
    enum CountMetric : std::size_t {
        collision_candidates = 0,
        replay_valid = 1,
        parent_killed = 2,
        parent_continued = 3
    };
};

// A secondary inelastic hazard suppresses the normal per-step MCS only when a
// valid CINEL02 final state was actually replayed. Lookup misses and invalid
// package records are null collisions: the post-EM state is retained and the
// same MCS that a non-nuclear step would receive must still be applied.
constexpr bool cinel02_should_apply_secondary_mcs(
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

// Simpson quadrature for the continuous optical-depth audit. The caller
// supplies rates evaluated at the beginning, midpoint and end energies of the
// same linearized stopping segment. This helper is deliberately independent
// of the runtime step-start hazard sampler.
inline constexpr float cinel02_simpson_hazard(
    const float lambda_start_per_mm, const float lambda_mid_per_mm,
    const float lambda_end_per_mm, const float step_mm) noexcept {
    return step_mm * (lambda_start_per_mm + 4.0F * lambda_mid_per_mm +
                      lambda_end_per_mm) / 6.0F;
}

struct PrimaryFirstInteractionRecord {
    float x_mm{0.0F};
    float y_mm{0.0F};
    float depth_mm{0.0F};
    float energy_MeVu{0.0F};
    std::uint32_t section_id{0};
    float density_g_per_cm3{0.0F};
};

struct BraggPeakMetrics {
    double peak_depth_mm{0.0};
    double peak_dose_MeV{0.0};
    double r80_distal_mm{std::numeric_limits<double>::quiet_NaN()};
    double r50_distal_mm{std::numeric_limits<double>::quiet_NaN()};
    bool found_r80{false};
    bool found_r50{false};
};

struct EnergyAccountingLedger {
    double E_continuous_ionizing{0.0};
    double E_nuclear_local{0.0};
    double E_transported_secondaries{0.0};
    double E_escaped_charged{0.0};
    double E_neutral{0.0};
    double E_cutoff_kill{0.0};
    double E_unsupported{0.0};
    double E_queue_overflow{0.0};
    // Split nuclear-vertex ledger (populated on the Schneider CT path; the
    // broad `untracked` sink must NOT be used to absorb these terms and then
    // be called physical closure).
    double E_charged_birth{0.0};
    double E_charged_terminal_deposit{0.0};
    double E_neutral_product_kinetic{0.0};
    double E_reaction_q_residual{0.0};
    double E_unsupported_charged{0.0};
    double E_be6_kill{0.0};
    double E_lookup_failure{0.0};
    double E_out_of_domain{0.0};

    [[nodiscard]] double total_accounted_MeV() const noexcept {
        return E_continuous_ionizing + E_nuclear_local + E_transported_secondaries +
               E_escaped_charged + E_neutral + E_cutoff_kill + E_unsupported + E_queue_overflow;
    }
};

// Independent Schneider-CT nuclear diagnostics. This is intentionally NOT
// the CINEL02 water `cinel02_diagnostics` array: every counter below has a
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
    Count = 44
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

struct LongitudinalDomainRecord {
    std::uint64_t history;
    std::uint32_t step;
    float initial_energy_MeVu;
    float query_energy_MeVu;
    float step_start_z_mm;
    float step_length_mm;
    float retained_MeV;
};
struct ElectronJointDiagnostics {
    std::uint64_t queries{},domain_misses{},invalid_marches{};
    double redistributed_MeV{},escaped_MeV{},domain_retained_MeV{};
};
inline constexpr std::size_t kLongitudinalDomainLogCap = 4096;

struct TransportResult {
    std::vector<LongitudinalDomainRecord> longitudinal_domain_log{};
    ElectronJointDiagnostics electron_joint_diagnostics{};
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
        Cinel02SpeciesLedgerSchema::species_count;
    static constexpr std::size_t species_ledger_metric_count =
        Cinel02SpeciesLedgerSchema::metric_count;
    std::vector<double> deposited_energy_MeV;
    // Informational subset of deposited energy redistributed by the optional
    // Schneider section-0 primary delta-tail proxy. The fallback term stayed
    // at the production voxel because the sampled endpoint left section 0.
    double schneider_primary_delta_tail_moved_MeV{0.0};
    double schneider_primary_delta_tail_fallback_MeV{0.0};
    // Sampled delta-tail energy whose endpoint left the aligned 3-D scorer.
    double schneider_primary_delta_tail_escaped_scorer_MeV{0.0};
    // Informational subset redistributed by the optional longitudinal
    // (forward) supplement: moved along the beam and distributed over the
    // march voxels (air or tissue) inside the scorer, kept locally when a
    // march share falls inside the scorer but outside the CT grid, escaped
    // when it leaves the scorer.
    double schneider_primary_delta_longitudinal_moved_MeV{0.0};
    double schneider_primary_delta_longitudinal_fallback_MeV{0.0};
    double schneider_primary_delta_longitudinal_escaped_scorer_MeV{0.0};
    std::uint64_t schneider_primary_delta_longitudinal_domain_queries{0};
    double schneider_primary_delta_longitudinal_domain_energy_MeV{0.0};
    std::uint64_t schneider_primary_delta_longitudinal_invalid_marches{0};
    double longitudinal_diagnostic_density_g_cm3{0.0};
    std::vector<double> voxel_deposited_energy_MeV;
    // Primary track length accumulated per scorer voxel (mm).
    std::vector<double> primary_voxel_track_length_mm;
    std::vector<double> in_fov_deposited_energy_MeV;
    // Category-major layout: category * number_of_voxels + voxel index.
    std::vector<double> charged_origin_voxel_deposited_energy_MeV;
    std::vector<double> be_isotope_origin_voxel_deposited_energy_MeV;
    std::vector<double> neutral_origin_voxel_deposited_energy_MeV;
    std::vector<double> primary_deposited_energy_MeV;
    std::vector<double> secondary_carbon_deposited_energy_MeV;
    std::vector<double> secondary_boron_deposited_energy_MeV;
    std::vector<double> secondary_beryllium_deposited_energy_MeV;
    std::vector<double> secondary_lithium_deposited_energy_MeV;
    std::vector<double> secondary_helium_deposited_energy_MeV;
    std::vector<double> secondary_proton_deposited_energy_MeV;
    std::vector<double> secondary_other_charged_deposited_energy_MeV;
    // Validation-only depth tallies (empty in production).
    std::vector<double> primary_fluence_mm;
    std::vector<double> secondary_carbon_fluence_mm;
    std::vector<double> secondary_boron_fluence_mm;
    std::vector<double> secondary_beryllium_fluence_mm;
    std::vector<double> secondary_lithium_fluence_mm;
    std::vector<double> secondary_helium_fluence_mm;
    std::vector<double> secondary_proton_fluence_mm;
    std::vector<double> secondary_other_charged_fluence_mm;
    std::vector<std::uint64_t> primary_survival_counts;
    std::vector<std::uint64_t> inelastic_reaction_counts;
    // HadronLET raw dose-weighted moments. Numerator unit:
    // MeV * MeV/mm/(g/cm3); denominator unit: MeV.
    std::vector<double> primary_letd_numerator;
    std::vector<double> primary_letd_denominator;
    std::vector<double> all_hadron_letd_numerator;
    std::vector<double> all_hadron_letd_denominator;
    // Category-major primary, secondary C, B, Be, Li, He, p, other.
    std::vector<double> charged_origin_letd_numerator;
    std::vector<double> charged_origin_letd_denominator;
    // Optional category-major p, d, t, He-3, He-4, N, O, F LET moments.
    std::vector<double> light_isotope_letd_numerator;
    std::vector<double> light_isotope_letd_denominator;
    // Optional selected-isotope/element birth spectra. Layout documented
    // in particle.hpp birth_* constants. Counts are event tallies (not /primary).
    // Histograms are species × generation × bin (see birth_hist_index).
    std::vector<std::uint64_t> birth_counts_by_generation;  // cat * gen_bins
    std::vector<double> birth_ke_sum_MeV_by_generation;     // cat * gen_bins
    std::vector<std::uint64_t> birth_mevu_hist;             // cat*gen*mevu_bins
    std::vector<std::uint64_t> birth_depth_hist;            // cat*gen*depth_bins
    std::vector<std::uint64_t> birth_cos_hist;              // cat*gen*cos_bins
    std::vector<std::uint64_t> birth_parent_mevu_hist;      // cat*gen*parent_mevu
    std::vector<std::uint64_t> birth_parent_z_hist;         // cat*gen*parent_z
    // cat*gen*(parent_mevu_bins * product_mevu_bins)
    std::vector<std::uint64_t> birth_parent_product_mevu_hist;
    // Same four moments on the optional voxel grid (z-major, x fastest).
    std::vector<double> primary_voxel_letd_numerator;
    std::vector<double> primary_voxel_letd_denominator;
    std::vector<double> all_hadron_voxel_letd_numerator;
    std::vector<double> all_hadron_voxel_letd_denominator;
    std::vector<double> neutron_origin_deposited_energy_MeV;
    std::vector<double> gamma_origin_deposited_energy_MeV;
    double initial_energy_MeV{0.0};
    double total_deposited_energy_MeV{0.0};
    double escaped_energy_MeV{0.0};
    // Energy lost/deposited/terminated before the scored phantom by the
    // optional minibeam beamline.
    double beamline_removed_energy_MeV{0.0};
    double untracked_nuclear_energy_MeV{0.0};
    double fred_model_unassigned_MeV{0.0};
    std::uint64_t nuclear_interactions{0};
    // Step 12 primary-only & validation counters
    std::uint64_t primary_inelastic_terminated_count{0};
    std::uint64_t primary_escaped_ct_count{0};
    std::uint64_t primary_stopped_count{0};
    std::uint64_t primary_other_terminal_count{0};
    double primary_inelastic_removed_kinetic_MeV{0.0};
    double primary_other_terminal_kinetic_MeV{0.0};
    double primary_cutoff_stopped_energy_MeV{0.0};
    std::vector<PrimaryFirstInteractionRecord> primary_first_interactions{};
    // Fixed-layout CINEL02 runtime ledger; zero for other nuclear models.
    static constexpr std::size_t cinel02_diagnostic_slot_count = 1668;
    std::array<std::uint64_t, cinel02_diagnostic_slot_count> cinel02_diagnostics{};
    // Aggregate kinetic-energy classification for valid CINEL02 replays.
    // Layout is emitted with the energy ledger JSON.
    std::array<double, 8> cinel02_energy_ledger_MeV{};
    std::array<double, species_ledger_species_count * species_ledger_metric_count>
        cinel02_species_transport_ledger_MeV{};
    // Explicit in-grid/outside-grid deposited-energy split (global MeV).
    // Every deposited-ledger credit is mirrored here with that site's paired
    // voxel-scorer guard, so total ≈ in + outside and voxel_sum ≈ in_grid.
    double in_grid_deposited_energy_MeV{0.0};
    double outside_grid_deposited_energy_MeV{0.0};
    std::array<std::uint64_t,
               species_ledger_species_count *
                   Cinel02SpeciesLedgerSchema::terminal_reason_count>
        cinel02_species_terminal_reason_counts{};
    // Explicit sink used only by TOPAS/Geant4 compatibility mode for
    // unsupported prompt-unstable products (currently Be-6).  This is not
    // dose, local nuclear deposit, reaction export, or physical escape.
    std::array<std::uint64_t, species_ledger_species_count>
        cinel02_topas_compat_discarded_counts{};
    std::array<double, species_ledger_species_count>
        cinel02_topas_compat_discarded_kinetic_MeV{};
    // Signed package/runtime incident-energy handoff diagnostics. The sums
    // are in MeV/u and are keyed by the projectile isotope of each valid
    // CINEL02 replay. Positive/negative counts classify the signed delta.
    std::array<double, species_ledger_species_count>
        cinel02_replay_delta_MeV_per_u{};
    std::array<double, species_ledger_species_count>
        cinel02_replay_abs_delta_MeV_per_u{};
    std::array<std::uint64_t, species_ledger_species_count>
        cinel02_replay_delta_positive_counts{};
    std::array<std::uint64_t, species_ledger_species_count>
        cinel02_replay_delta_negative_counts{};
    std::array<std::uint64_t, species_ledger_species_count>
        cinel02_replay_valid_counts{};
    // Compact isotope × target × generation × 10-MeV/u-bin replay ledger.
    std::array<std::uint64_t, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_counts{};
    // Per replay-status cell energies.  The replay query energy is the
    // post-EM collision-point energy; rate query is the pre-EM energy used
    // for hazard/target selection.  Their difference is the continuous loss
    // accrued to the collision point (within atomic accumulation precision).
    std::array<double, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_rate_query_energy_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_replay_query_energy_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_continuous_loss_to_collision_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_delta_MeV_per_u{};
    std::array<double, Cinel02ReplayLedgerSchema::status_slot_count>
        cinel02_replay_status_abs_delta_MeV_per_u{};
    // Secondary transport exposure: sums are in mm (path) or dimensionless
    // integrated hazard; count metrics are event tallies. The generation
    // index is the SecondaryParticle transport generation (0 = direct
    // product, 1 = first cascade child, 2 = generation 2+).
    std::array<double, Cinel02ExposureLedgerSchema::sum_slot_count>
        cinel02_secondary_exposure_sums{};
    std::array<std::uint64_t, Cinel02ExposureLedgerSchema::count_slot_count>
        cinel02_secondary_exposure_counts{};
    // Isotope-resolved parent outcome energy ledger.
    std::array<std::uint64_t, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_counts{};
    std::array<double, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_incident_energy_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_after_energy_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_local_deposit_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_export_MeV{};
    std::array<double, Cinel02ReplayLedgerSchema::parent_outcome_cell_count>
        cinel02_parent_outcome_import_MeV{};
    // Generated and successfully queued isotope-to-isotope transitions.
    std::array<std::uint64_t, Cinel02ReplayLedgerSchema::transition_cell_count>
        cinel02_generated_transition_counts{};
    std::array<double, Cinel02ReplayLedgerSchema::transition_cell_count>
        cinel02_generated_transition_kinetic_MeV{};
    std::array<std::uint64_t, Cinel02ReplayLedgerSchema::transition_cell_count>
        cinel02_queued_transition_counts{};
    std::array<double, Cinel02ReplayLedgerSchema::transition_cell_count>
        cinel02_queued_transition_kinetic_MeV{};
    std::array<std::uint64_t, 18> fred_isotope_counts{};
    std::uint64_t fred_inelastic_events{0};
    std::uint64_t fred_retry_sum{0};
    std::uint64_t fred_energy_scaled_events{0};
    std::uint64_t fred_projectile_az_open_events{0};
    std::uint64_t fred_leftover_target_a_sum{0};
    std::uint64_t fred_leftover_target_z_sum{0};
    std::uint64_t fred_leftover_projectile_a_sum{0};
    std::uint64_t fred_leftover_projectile_z_sum{0};
    double fred_model_residual_MeV{0.0};
    double fred_q_MeV{0.0};
    double fred_neutron_ke_MeV{0.0};
    double fred_remnant_local_MeV{0.0};
    std::uint64_t fred_resample_failed_events{0};
    double fred_resample_failed_energy_MeV{0.0};
    std::uint64_t fred_product_capacity_overflow_events{0};
    double fred_product_capacity_overflow_energy_MeV{0.0};
    float fred_invert_error_proj_h{0.0F};
    float fred_invert_error_proj_o{0.0F};
    float fred_invert_error_tgt_h{0.0F};
    float fred_invert_error_tgt_o{0.0F};
    std::uint64_t sampled_reaction_packages{0};
    std::uint64_t generated_direct_secondaries{0};
    std::uint64_t queued_secondaries{0};
    std::uint64_t secondary_queue_overflow{0};
    double generated_direct_secondary_energy_MeV{0.0};
    double queued_secondary_energy_MeV{0.0};
    double secondary_queue_overflow_energy_MeV{0.0};
    double untransported_neutral_energy_MeV{0.0};
    double untransported_unsupported_charged_energy_MeV{0.0};
    double nuclear_energy_not_in_direct_secondaries_MeV{0.0};
    std::uint64_t primary_elastic_interactions{0};
    double elastic_local_deposited_energy_MeV{0.0};
    double elastic_queued_charged_energy_MeV{0.0};
    double elastic_queued_neutral_energy_MeV{0.0};
    std::uint64_t elastic_queue_overflow{0};
    double elastic_queue_overflow_energy_MeV{0.0};
    std::uint64_t transported_secondaries{0};
    std::uint64_t secondary_transport_steps{0};
    double secondary_deposited_energy_MeV{0.0};
    double secondary_escaped_energy_MeV{0.0};
    std::uint64_t cascade_interactions{0};
    std::uint64_t cascade_selection_exact{0};
    std::uint64_t cascade_selection_expanded{0};
    std::uint64_t cascade_selection_nearest{0};
    std::uint64_t cascade_selection_no_coverage{0};
    double cascade_selection_energy_distance_sum_MeVu{0.0};
    double cascade_selection_energy_distance_max_MeVu{0.0};
    std::uint64_t generated_cascade_products{0};
    std::uint64_t queued_cascade_secondaries{0};
    std::uint64_t cascade_queue_overflow{0};
    double queued_cascade_energy_MeV{0.0};
    double cascade_nuclear_energy_MeV{0.0};
    std::uint64_t queued_neutrals{0};
    std::uint64_t neutral_queue_overflow{0};
    std::uint64_t transported_neutrals{0};
    std::uint64_t neutral_interactions{0};
    std::uint64_t neutral_transport_steps{0};
    double queued_neutral_energy_MeV{0.0};
    double neutral_queue_overflow_energy_MeV{0.0};
    double neutral_deposited_energy_MeV{0.0};
    double neutral_escaped_energy_MeV{0.0};
    double residual_neutral_energy_MeV{0.0};
    double charged_from_neutral_energy_MeV{0.0};
    double neutral_unsupported_product_energy_MeV{0.0};
    double neutral_package_closure_residual_MeV{0.0};
    std::uint64_t queued_electrons{0};
    std::uint64_t electron_queue_overflow{0};
    std::uint64_t transported_electrons{0};
    std::uint64_t transported_positrons{0};
    std::uint64_t electron_transport_steps{0};
    double queued_electron_energy_MeV{0.0};
    double electron_queue_overflow_energy_MeV{0.0};
    double electron_deposited_energy_MeV{0.0};
    double electron_escaped_energy_MeV{0.0};
    double electron_radiative_energy_MeV{0.0};
    double positron_annihilation_reserve_MeV{0.0};
    std::uint64_t electron_generated_gammas{0};
    std::uint64_t electron_gamma_queue_overflow{0};
    double electron_brems_gamma_energy_MeV{0.0};
    double positron_annihilation_gamma_energy_MeV{0.0};
    double electron_gamma_queue_overflow_energy_MeV{0.0};
    double electromagnetic_generation_residual_MeV{0.0};
    std::uint64_t total_steps{0};
    double elapsed_seconds{0.0};
    double primary_kernel_seconds{0.0};
    double secondary_kernel_seconds{0.0};
    double neutral_kernel_seconds{0.0};
    double electron_kernel_seconds{0.0};
    double charged_after_neutral_kernel_seconds{0.0};
    std::string backend;
    MinibeamDiagnostics minibeam;
    // Populated only when built with CARBON_TRANSPORT_PROFILE=1.
    TransportProfile profile;

    // Accounting closure includes explicit TOPAS compatibility sinks.  The
    // physical closure intentionally excludes those sinks so a reference
    // model defect cannot be mistaken for deposited or escaped energy.
    [[nodiscard]] double physical_relative_energy_balance_error() const noexcept;
    [[nodiscard]] double topas_compat_discarded_kinetic_total_MeV() const noexcept;
    [[nodiscard]] double relative_energy_balance_error() const noexcept;
};

[[nodiscard]] double choose_step_mm(double energy_MeV,
                                    double stopping_power_MeV_per_mm,
                                    double maximum_step_mm,
                                    double maximum_relative_energy_loss);

[[nodiscard]] std::vector<double> compute_idd_from_3d_voxel_dose(
    const std::vector<double>& voxel_dose_MeV,
    std::size_t nx, std::size_t ny, std::size_t nz);

[[nodiscard]] BraggPeakMetrics compute_bragg_peak_metrics(
    const std::vector<double>& idd_energy_MeV,
    double bin_width_z_mm,
    double z_min_mm = 0.0);

TransportResult transport_serial(const TransportConfig& config,
                                 const StoppingPowerTable& stopping_power,
                                 const CrossSectionTable& cross_section);

#ifdef CARBON_HAS_SYCL
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

    friend TransportResult transport_sycl(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, SyclTransportContext*);
#if defined(CARBON_ENABLE_MINIBEAM)
    friend TransportResult transport_sycl_legacy(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, SyclTransportContext*);
    friend TransportResult transport_sycl_minibeam(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, SyclTransportContext*);
#endif
};

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context = nullptr);
#if defined(CARBON_ENABLE_MINIBEAM)
// Dual kernels: legacy is the master-compatible water/CT path; minibeam is the
// Copper beamline path. transport_sycl() dispatches at runtime.
TransportResult transport_sycl_legacy(
    const TransportConfig& config, const StoppingPowerTable& stopping_power,
    const CrossSectionTable& cross_section, const std::string& device_name,
    SyclTransportContext* context = nullptr);
TransportResult transport_sycl_minibeam(
    const TransportConfig& config, const StoppingPowerTable& stopping_power,
    const CrossSectionTable& cross_section, const std::string& device_name,
    SyclTransportContext* context = nullptr);
#endif
std::string describe_sycl_device(const std::string& device_name);
#endif

}  // namespace carbon
