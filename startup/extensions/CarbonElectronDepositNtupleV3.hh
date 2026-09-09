#pragma once
#include "TsVNtupleScorer.hh"
#include <map>
#include <set>
#include <utility>
class G4Track;
class G4Material;

// Schema 3: 29 legacy columns plus generating parent step ID and post state.
// parent_ke/dir are the PRE state of the exact generating step. The post
// state is also exported. Neither is called an exact process-internal KE.
// Only direct C12 electron births have parent_valid=1; descendants inherit
// the root association offline. No track user information or RNG is changed.
class CarbonElectronDepositNtupleV3 : public TsVNtupleScorer {
public:
    CarbonElectronDepositNtupleV3(TsParameterManager*, TsMaterialManager*,
        TsGeometryManager*, TsScoringManager*, TsExtensionManager*,
        G4String, G4String, G4String, G4bool);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
private:
    G4int run_, event_, track_, parent_, pdg_, step_;
    G4double birth_x_, birth_y_, birth_z_, birth_ke_;
    G4double pre_x_, pre_y_, pre_z_, post_x_, post_y_, post_z_;
    G4double edep_, pre_ke_, post_ke_, weight_, density_;
    G4int schema_version_, parent_valid_, creator_process_id_, parent_step_;
    G4double parent_ke_, parent_dir_x_, parent_dir_y_, parent_dir_z_, birth_density_;
    G4double parent_post_ke_, parent_post_dx_, parent_post_dy_, parent_post_dz_;
    // Opt-in schema 4: actual electron momentum is not the step chord.
    G4bool record_transport_state_{false};
    G4double pre_dx_, pre_dy_, pre_dz_, post_dx_, post_dy_, post_dz_;
    G4double step_length_, post_density_;
    G4int pre_material_, post_material_, post_step_status_, track_status_;
    std::set<G4int> reported_materials_;
    void ReportMaterial(const G4Material*);
    struct ParentState {
        G4double ke, dx, dy, dz, post_ke, post_dx, post_dy, post_dz;
        G4int step;
    };
    std::pair<G4int, G4int> active_event_{-1,-1};
    std::map<const G4Track*, ParentState> pending_;
    std::map<G4int, ParentState> track_state_;
    std::map<G4int, G4double> birth_density_cache_;
    static G4int CreatorProcessId(const G4String&, bool);
};
