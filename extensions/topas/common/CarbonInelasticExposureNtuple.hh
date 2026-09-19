#ifndef CarbonInelasticExposureNtuple_hh
#define CarbonInelasticExposureNtuple_hh

#include "TsVNtupleScorer.hh"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

class G4Isotope;
class G4Material;
class G4ParticleDefinition;
class G4Step;
class G4Track;

// Records the complete source-primary counting-process exposure used to
// estimate target-specific CINEL02 hazards.  One exposure row is emitted for
// every target isotope and every energy-bin piece of every primary step.  A
// history_outcome row is emitted at primary-track end, including when the
// primary never had an inelastic collision.  Collision rows are separate so
// their target isotope and authoritative incident energy are never inferred
// from a path segment.
class CarbonInelasticExposureNtuple : public TsVNtupleScorer {
public:
    CarbonInelasticExposureNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonInelasticExposureNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfTrack(const G4Track*) override;
    void UserHookForEndOfEvent() override;
    void UserHookForEndOfRun() override;
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override;

protected:
    void Clear() override;

private:
    struct TargetSpec {
        G4int z{0};
        G4int a{0};
        G4int isotope_id{-1};
        G4String name;
        G4double number_density_per_mm3{0.0};
    };

    struct EnergySegment {
        G4int bin{-1};
        G4double length_mm{0.0};
    };

    struct CollisionInfo {
        G4bool is_inelastic{false};
        G4double energy_mev_per_u{0.0};
        G4int target_z{0};
        G4int target_a{0};
        G4int target_isotope_id{-1};
        G4String target_name;
        G4double target_number_density_per_mm3{0.0};
        G4String material_name;
        G4String process_name;
        G4String energy_source;
    };

    using HistoryKey = std::tuple<G4int, G4int, G4int, G4int>;
    using CellKey = std::tuple<G4int, G4int, G4int>;
    using CompactCellKey = std::tuple<G4int, G4int, G4int, G4int, G4int>;

    // The normal TsVNtuple path emits one row per step.  That is useful for
    // interactive inspection but is unnecessarily large for production rate
    // extraction and can overwhelm TOPAS' MT ntuple merger.  Keep the exact
    // sufficient statistics worker-local and publish a compact CSV in the
    // scorer destructor.  The offline aggregator consumes these rows in the
    // same way as the unaggregated contract.
    struct CompactCell {
        G4int projectile_z{0};
        G4int projectile_a{0};
        G4int target_z{0};
        G4int target_a{0};
        G4int target_isotope_id{-1};
        G4String target_name;
        G4String material_name;
        G4int energy_bin{-1};
        G4double energy_low{0.0};
        G4double energy_high{0.0};
        G4double energy{0.0};
        G4double weighted_track_length_mm{0.0};
        G4double target_areal_density_weighted_per_mm2{0.0};
        G4double density_sum_per_mm3{0.0};
        G4int density_samples{0};
        G4int history_count{0};
        G4double history_weight_sum{0.0};
        G4double history_weight_squared_sum{0.0};
        G4int censored_history_count{0};
        G4double censored_history_weight_sum{0.0};
        G4double censored_history_weight_squared_sum{0.0};
        G4int collision_count{0};
        G4double collision_weight_sum{0.0};
        G4double collision_weight_squared_sum{0.0};
        G4double collision_energy_weighted_sum{0.0};
    };

    struct HistoryState {
        G4int run_id{0};
        G4int thread_id{0};
        G4int event_id{0};
        G4int track_id{0};
        G4int projectile_z{0};
        G4int projectile_a{0};
        G4double history_weight{1.0};
        G4double source_energy_mev_per_u{0.0};
        G4bool have_weight{false};
        G4bool had_inelastic_collision{false};
        std::map<CellKey, TargetSpec> touched_cells;
        std::set<std::pair<G4int, G4int>> collision_targets;
        G4int next_collision_sequence{0};
    };

    static G4int ReadRequiredInteger(
        TsParameterManager*, const G4String&, const char* label);
    static G4double ReadRequiredDouble(
        TsParameterManager*, const G4String&, const char* label);
    static void Fatal(const char* code, const G4String& message);

    G4bool IsConfiguredProjectile(const G4Track*) const;
    G4int CurrentThreadID() const;
    HistoryKey KeyFor(const G4Track*);
    HistoryState& GetOrCreateHistory(const G4Track*);

    std::vector<TargetSpec> TargetsForMaterial(const G4Material*) const;
    static const TargetSpec* FindTarget(
        const std::vector<TargetSpec>&, G4int z, G4int a);
    G4int EnergyBin(G4double energy_mev_per_u) const;
    std::vector<EnergySegment> SplitStep(
        G4double pre_energy_mev_per_u,
        G4double post_energy_mev_per_u,
        G4double step_length_mm) const;
    CollisionInfo CollisionForStep(const G4Step*);

    void SetCommonRow(
        const HistoryState&, G4int target_z, G4int target_a,
        G4int target_isotope_id, const G4String& target_name,
        G4int energy_bin, G4double energy_mev_per_u);
    void FillExposureRow(
        const HistoryState&, const TargetSpec&, G4int energy_bin,
        G4double length_mm, G4double track_weight,
        G4int step_index, const G4String& material_name);
    void FillCollisionRow(
        const HistoryState&, const CollisionInfo&, G4int energy_bin,
        G4double collision_weight, G4int step_index,
        G4int collision_sequence);
    void FillHistoryOutcomeRow(
        const HistoryState&, const TargetSpec&, G4int energy_bin,
        G4bool censored);
    CompactCell& GetCompactCell(
        G4int projectile_z, G4int projectile_a,
        G4int target_z, G4int target_a, G4int target_isotope_id,
        const G4String& target_name, const G4String& material_name,
        G4int energy_bin, G4double energy_mev_per_u);
    void FinalizeHistory(HistoryState&);
    void FinalizeHistoriesForEvent(G4int run_id, G4int thread_id, G4int event_id);
    void FlushCompactOutput();

    G4int projectile_z_{0};
    G4int projectile_a_{0};
    // A 0/0 projectile selector is a deliberate wildcard for charged-ion
    // secondaries.  It is only legal together with IncludeSecondaries and
    // keeps one compact scorer sufficient for a multi-projectile cascade
    // extraction.
    G4bool include_secondaries_{false};
    G4double energy_bin_min_mev_per_u_{0.0};
    G4double energy_bin_width_mev_per_u_{1.0};
    G4int energy_bin_count_{0};
    G4bool require_authoritative_collision_state_{true};

    // Row state.  The scorer is worker-local and fills one ntuple row at a
    // time, so these addresses remain stable for TsVNtuple::Fill().
    G4String record_kind_;
    G4int run_id_{0};
    G4int thread_id_{0};
    G4int event_id_{0};
    G4int track_id_{0};
    G4int projectile_z_row_{0};
    G4int projectile_a_row_{0};
    G4int target_z_{0};
    G4int target_a_{0};
    G4int target_isotope_id_{-1};
    G4String target_isotope_name_;
    G4String material_name_;
    G4int energy_bin_id_{-1};
    G4float energy_low_mev_per_u_{0.0F};
    G4float energy_high_mev_per_u_{0.0F};
    G4float energy_mev_per_u_{0.0F};
    G4float collision_energy_mev_per_u_{0.0F};
    G4float track_length_mm_{0.0F};
    G4float track_weight_{0.0F};
    G4double weighted_track_length_mm_{0.0};
    G4double target_number_density_per_mm3_{0.0};
    G4double target_areal_density_weighted_per_mm2_{0.0};
    G4int history_count_{0};
    G4double history_weight_sum_{0.0};
    G4double history_weight_squared_sum_{0.0};
    G4int censored_history_count_{0};
    G4double censored_history_weight_sum_{0.0};
    G4double censored_history_weight_squared_sum_{0.0};
    G4int collision_count_{0};
    G4double collision_weight_sum_{0.0};
    G4double collision_weight_squared_sum_{0.0};
    G4float source_energy_mev_per_u_{0.0F};
    G4int step_index_{0};
    G4int collision_sequence_{-1};
    G4String collision_process_;
    G4String collision_energy_source_;

    std::map<HistoryKey, HistoryState> histories_;
    std::map<CompactCellKey, CompactCell> compact_cells_;
    std::string compact_output_file_;
    G4int compact_worker_id_{-1};
    std::uint64_t compact_identity_{0};
    std::uint64_t compact_row_id_{0};
    std::uint64_t target_isotope_pointer_missing_count_{0};
    bool compact_output_initialized_{false};
};

#endif
