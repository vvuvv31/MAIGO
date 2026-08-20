#ifndef CarbonElasticNtuple_hh
#define CarbonElasticNtuple_hh

#include "TsVNtupleScorer.hh"

#include "G4ThreeVector.hh"

#include <unordered_map>
#include <vector>

// The Carbon prefix is retained for TOPAS extension compatibility.  The
// scorer itself is projectile-agnostic and is configured with ProjectileZ/A.
class CarbonElasticNtuple : public TsVNtupleScorer {
public:
    CarbonElasticNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~CarbonElasticNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    struct InteractionContext {
        G4int interaction_id = 0;
        G4int interaction_track_id = 0;
        G4int projectile_pdg = 0;
        G4int projectile_z = 0;
        G4int projectile_a = 0;
        G4float incident_energy_mev_per_u = 0.0F;
        G4float outgoing_energy_mev_per_u = 0.0F;
        G4float local_deposit_mev = 0.0F;
        G4double macroscopic_elastic_per_mm = 0.0;
        G4ThreeVector incident_direction;
        G4ThreeVector outgoing_direction;
        G4ThreeVector vertex;
        G4String process_name;
        G4int process_type = 0;
        G4int process_subtype = 0;
        G4int generation = 0;
    };

    G4String record_kind_;
    G4int run_id_ = 0;
    G4int thread_id_ = 0;
    G4int event_id_ = 0;
    G4int interaction_id_ = 0;
    G4int interaction_track_id_ = 0;
    G4int track_id_ = 0;
    G4int parent_id_ = 0;
    G4int projectile_pdg_id_ = 0;
    G4int pdg_id_ = 0;
    G4String particle_name_;
    G4int atomic_number_ = 0;
    G4int atomic_mass_ = 0;
    G4float charge_e_ = 0.0F;
    G4float incident_energy_mev_per_u_ = 0.0F;
    G4float outgoing_projectile_energy_mev_per_u_ = 0.0F;
    G4float kinetic_energy_mev_ = 0.0F;
    G4float local_deposit_mev_ = 0.0F;
    G4double macroscopic_elastic_per_mm_ = 0.0;
    G4float vertex_x_mm_ = 0.0F;
    G4float vertex_y_mm_ = 0.0F;
    G4float vertex_z_mm_ = 0.0F;
    G4float incident_direction_x_ = 0.0F;
    G4float incident_direction_y_ = 0.0F;
    G4float incident_direction_z_ = 0.0F;
    G4float direction_x_ = 0.0F;
    G4float direction_y_ = 0.0F;
    G4float direction_z_ = 0.0F;
    G4int generation_ = 0;
    G4String transport_disposition_;
    G4String process_name_;
    G4int process_type_ = 0;
    G4int process_subtype_ = 0;
    G4int creator_model_id_ = 0;

    G4int projectile_z_ = 0;
    G4int projectile_a_ = 0;
    G4int cached_event_id_ = -1;
    G4int next_interaction_id_ = 0;
    std::unordered_map<G4int, std::vector<InteractionContext>> interactions_;
};

#endif
