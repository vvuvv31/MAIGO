#pragma once
#include "TsVNtupleScorer.hh"
#include <set>
#include <utility>
// Independent actual-material tag: one row per primary with a positive step
// in copper. Join to phase data by (run,event,track); no track state mutation.
class PrimaryCopperTouchNtuple : public TsVNtupleScorer {
public:
    PrimaryCopperTouchNtuple(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
        TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
    G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
private:
    G4int run_{},event_{},track_{};
    G4double length_{},energy_{};
    std::pair<G4int,G4int> event_key_{-1,-1};
    std::set<G4int> seen_;
};
