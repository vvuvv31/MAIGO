#ifndef CarbonSchneiderThinSlabValidationScorer_hh
#define CarbonSchneiderThinSlabValidationScorer_hh

#include "TsVScorer.hh"
#include "G4ThreeVector.hh"

#include <vector>
#include <string>
#include <set>
#include <map>

class CarbonSchneiderThinSlabValidationScorer : public TsVScorer {
public:
    CarbonSchneiderThinSlabValidationScorer(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);
    ~CarbonSchneiderThinSlabValidationScorer() override;

    G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* touchable) override;
    void UserHookForEndOfEvent() override;
    void UserHookForEndOfRun() override;
    void AbsorbResultsFromWorkerScorer(TsVScorer* worker_scorer) override;
    void Output() override;
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}

    struct FirstInteractionRecord {
        G4int event_id;
        G4double depth_mm;
        G4double step_pre_energy_mevu;
        G4String process_name;
        G4int track_id;
        G4int parent_id;
        G4bool is_hadronic_inelastic;
    };

private:
    void WriteResultsToJson();

    // Configuration parameters
    G4int section_id_{8};
    G4String material_name_{"SchneiderTissue_HU_100"};
    G4double nominal_energy_mevu_{100.0};
    G4double slab_thickness_mm_{10.0};
    G4double slab_trans_z_mm_{0.0};
    G4int num_depth_bins_{5};
    G4String output_json_path_{};

    // Tracking state for current event
    G4int current_event_id_{-1};
    G4int current_run_id_{-1};
    G4bool primary_entered_{false};
    G4bool primary_reacted_{false};
    std::set<G4int> crossed_checkpoints_{};

    // Aggregated statistics across histories
    G4long total_entering_primaries_{0};
    G4long total_first_inelastic_count_{0};
    G4long total_contamination_count_{0};

    // Per-depth checkpoint statistics
    std::vector<G4double> checkpoint_depths_mm_{};
    std::vector<G4long> bin_surviving_counts_{};
    std::vector<G4double> bin_energy_sums_mevu_{};
    std::vector<G4double> bin_energy_sq_sums_mevu_{};
    std::vector<G4long> bin_energy_sample_counts_{};

    // Process breakdown
    std::map<G4String, G4long> process_counts_{};
    std::map<G4int, G4long> target_element_counts_{};
    std::map<G4int, G4long> secondary_species_counts_{};

    // Sample of detailed first interactions (capped to prevent huge JSON)
    std::vector<FirstInteractionRecord> first_interactions_{};
    size_t max_detailed_interactions_{20000};
    G4long first_interaction_sample_overflow_count_{0};
};

#endif
