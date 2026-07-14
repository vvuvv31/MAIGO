#ifndef CarbonCascadeNtuple_hh
#define CarbonCascadeNtuple_hh

#include "TsVNtupleScorer.hh"

#include <unordered_map>

class CarbonCascadeNtuple : public TsVNtupleScorer {
public:
    CarbonCascadeNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~CarbonCascadeNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    struct InteractionContext {
        G4float incident_energy_mev = 0.0F;
        G4int projectile_z = 0;
        G4int projectile_a = 0;
    };

    G4String record_kind_;
    G4int run_id_ = 0;
    G4int event_id_ = 0;
    G4int interaction_track_id_ = 0;
    G4int track_id_ = 0;
    G4int parent_id_ = 0;
    G4int pdg_id_ = 0;
    G4String particle_name_;
    G4int atomic_number_ = 0;
    G4int atomic_mass_ = 0;
    G4float charge_e_ = 0.0F;
    G4float kinetic_energy_mev_ = 0.0F;
    G4float incident_energy_mev_ = 0.0F;
    G4double macroscopic_inelastic_per_mm_ = 0.0;
    G4int projectile_z_ = 0;
    G4int projectile_a_ = 0;
    G4float vertex_x_mm_ = 0.0F;
    G4float vertex_y_mm_ = 0.0F;
    G4float vertex_z_mm_ = 0.0F;
    G4float direction_x_ = 0.0F;
    G4float direction_y_ = 0.0F;
    G4float direction_z_ = 0.0F;
    G4float weight_ = 0.0F;
    G4String process_name_;
    G4int process_type_ = 0;
    G4int process_subtype_ = 0;
    G4int creator_model_id_ = 0;

    G4int cached_event_id_ = -1;
    std::unordered_map<G4int, InteractionContext> interactions_;
};

#endif
