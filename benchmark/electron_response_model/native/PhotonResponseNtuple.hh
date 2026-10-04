#pragma once
#include "TsVNtupleScorer.hh"
// Read-only primary photon interaction recording. Descendants run normally.
class PhotonResponseNtuple : public TsVNtupleScorer {
public:
    PhotonResponseNtuple(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
        TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
    G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;
private:
    G4int run_{-1},event_{-1},step_{},kind_{},pdg_{},process_{};
    G4double energy_{},remaining_{},deposit_{},children_{},pre_[3]{},post_[3]{},length_{},density_{};
    double initial_{},last_{},total_{},maxResidual_{};bool root_{},dumped_{};
    void Require(bool,const char*);
    void DumpRates(const G4Step*);
};
