#ifndef CarbonDecayNtuple_hh
#define CarbonDecayNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonDecayNtuple : public TsVNtupleScorer {
public:
    CarbonDecayNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~CarbonDecayNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4String record_kind_;
    G4int run_id_{0};
    G4int thread_id_{0};
    G4int event_id_{0};
    G4int track_id_{0};
    G4int parent_id_{0};
    G4int pdg_id_{0};
    G4String particle_name_;
    G4int atomic_number_{0};
    G4int atomic_mass_{0};
    G4float charge_e_{0.0F};
    G4float rest_mass_MeV_{0.0F};
    G4float excitation_MeV_{0.0F};
    G4float kinetic_energy_MeV_{0.0F};
    G4float pre_kinetic_energy_MeV_{0.0F};
    G4float post_kinetic_energy_MeV_{0.0F};
    G4float step_deposit_MeV_{0.0F};
    G4float step_length_mm_{0.0F};
    G4float direction_x_{0.0F};
    G4float direction_y_{0.0F};
    G4float direction_z_{1.0F};
    G4float vertex_x_mm_{0.0F};
    G4float vertex_y_mm_{0.0F};
    G4float vertex_z_mm_{0.0F};
    G4float global_time_ns_{0.0F};
    G4float proper_time_ns_{0.0F};
    G4String creator_process_;
    G4String post_process_;
    G4int creator_process_type_{0};
    G4int creator_process_subtype_{0};
    G4int creator_model_id_{0};
    G4int track_status_{0};
};

#endif
