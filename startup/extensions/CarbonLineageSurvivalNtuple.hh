#ifndef CarbonLineageSurvivalNtuple_hh
#define CarbonLineageSurvivalNtuple_hh

#include "TsVNtupleScorer.hh"

#include <unordered_map>

// Diagnostic scorer for isotope-resolved secondary survival.  It emits one
// row per transport episode for Li-6/Li-7 and Be-7/Be-9/Be-10 products of a
// hadronic cascade.  An episode ends at the first subsequent inelastic
// interaction, a decay/stop/boundary, or the end of the Geant4 event.  The
// scorer is observational only and is not consumed by the package compiler.
class CarbonLineageSurvivalNtuple : public TsVNtupleScorer {
public:
    CarbonLineageSurvivalNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~CarbonLineageSurvivalNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;

private:
    struct TrackLineage {
        G4int generation = 0;
        G4int episode_index = 0;
        G4int atomic_number = 0;
        G4int atomic_mass = 0;
        G4int parent_id = 0;
        G4double birth_kinetic_energy_mev = 0.0;
        G4double path_length_mm = 0.0;
        G4int step_count = 0;
        G4double weight = 1.0;
        G4ThreeVector birth_position;
        G4ThreeVector birth_direction;
        G4ThreeVector last_position;
        G4double last_kinetic_energy_mev = 0.0;
        G4String creator_process;
    };

    void ResetEvent(G4int run_id, G4int event_id);
    void EmitEpisode(const TrackLineage&, G4int track_id,
                     G4double terminal_kinetic_energy_mev,
                     G4double interaction_kinetic_energy_mev,
                     G4int reacted, G4int censored,
                     const G4String& terminal_reason,
                     const G4String& terminal_process,
                     const G4ThreeVector& terminal_position);
    static G4bool IsTrackedIsotope(const G4ParticleDefinition*);
    static G4bool IsHadronicInelastic(const G4VProcess*);

    G4int cached_run_id_{-1};
    G4int cached_event_id_{-1};
    std::unordered_map<G4int, TrackLineage> episodes_;
    std::unordered_map<G4int, G4int> generations_;

    G4String record_kind_;
    G4int run_id_{0};
    G4int thread_id_{0};
    G4int event_id_{0};
    G4int track_id_{0};
    G4int parent_id_{0};
    G4int atomic_number_{0};
    G4int atomic_mass_{0};
    G4int generation_{0};
    G4int episode_index_{0};
    G4int step_count_{0};
    G4int reacted_{0};
    G4int censored_{0};
    G4float weight_{1.0F};
    G4double birth_kinetic_energy_mev_{0.0};
    G4double terminal_kinetic_energy_mev_{0.0};
    G4double interaction_kinetic_energy_mev_{0.0};
    G4double path_length_mm_{0.0};
    G4double birth_x_mm_{0.0};
    G4double birth_y_mm_{0.0};
    G4double birth_z_mm_{0.0};
    G4double terminal_x_mm_{0.0};
    G4double terminal_y_mm_{0.0};
    G4double terminal_z_mm_{0.0};
    G4String creator_process_;
    G4String terminal_reason_;
    G4String terminal_process_;
};

#endif
