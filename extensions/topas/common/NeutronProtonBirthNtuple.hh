#pragma once
#include "TsVNtupleScorer.hh"
#include <map>
#include <set>
class G4Track;
// Read-only secondary records at the generating step; includes all parents
// so neutron production can be separated from charged/EM cascade feedback.
class NeutronProtonBirthNtuple : public TsVNtupleScorer {
public:
 NeutronProtonBirthNtuple(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
  TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
 G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
private:
 G4int active_event_{-1};
 std::map<G4int,std::set<const G4Track*>> seen_;
 G4int event_,parent_pdg_,parent_track_,parent_step_,process_,material_,target_z_,target_a_;
 G4double parent_energy_,energy_,x_,y_,z_,dx_,dy_,dz_,weight_,parent_dx_,parent_dy_,parent_dz_;
};
