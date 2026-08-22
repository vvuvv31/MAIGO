#ifndef EnergyLossFluctuationNtuple_hh
#define EnergyLossFluctuationNtuple_hh

#include "TsVNtupleScorer.hh"

// Writes one row per primary history crossing an EM-only thin slab.  The
// entry-to-exit kinetic-energy difference is the fluctuation sample used by
// the runtime package compiler; local deposition is retained as a diagnostic.
class EnergyLossFluctuationNtuple : public TsVNtupleScorer {
public:
    EnergyLossFluctuationNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);

    ~EnergyLossFluctuationNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;

private:
    void ResetEvent();

    G4int configured_z_{0};
    G4int configured_a_{0};

    G4int run_id_{-1};
    G4int event_id_{-1};
    G4int thread_id_{-1};
    G4int primary_track_id_{-1};
    G4int atomic_number_{0};
    G4int mass_number_{0};
    G4String material_name_;
    G4double entry_energy_mev_{0.0};
    G4double exit_energy_mev_{0.0};
    G4double kinetic_energy_loss_mev_{0.0};
    G4double primary_local_deposit_mev_{0.0};
    G4double path_length_mm_{0.0};
    G4int step_count_{0};
    G4bool material_consistent_{true};
    G4bool completed_{false};
    G4String completion_status_;

    G4bool saw_primary_{false};
};

#endif
