#ifndef CarbonReactionNtuple_hh
#define CarbonReactionNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonReactionNtuple : public TsVNtupleScorer {
public:
    CarbonReactionNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonReactionNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4String record_kind_;
    G4int run_id_ = 0;
    G4int event_id_ = 0;
    G4int track_id_ = 0;
    G4int parent_id_ = 0;
    G4int pdg_id_ = 0;
    G4int atomic_number_ = 0;
    G4int atomic_mass_ = 0;
    G4int process_type_ = 0;
    G4int process_subtype_ = 0;
    G4int creator_model_id_ = 0;
    G4float charge_e_ = 0.0F;
    G4float kinetic_energy_mev_ = 0.0F;
    G4float incident_energy_mev_ = 0.0F;
    G4float vertex_x_mm_ = 0.0F;
    G4float vertex_y_mm_ = 0.0F;
    G4float vertex_z_mm_ = 0.0F;
    G4float direction_x_ = 0.0F;
    G4float direction_y_ = 0.0F;
    G4float direction_z_ = 0.0F;
    G4float weight_ = 0.0F;
    G4int cached_reaction_event_id_ = -1;
    G4float cached_incident_energy_mev_ = 0.0F;
    G4String particle_name_;
    G4String creator_process_;
};

#endif
