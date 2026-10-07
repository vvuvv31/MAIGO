#pragma once
#include "TsVScorer.hh"
// Isolated conditional extraction. Invokes the initialized native process
// (including production cuts/null collisions); never used as a dose scorer.
class PhotonConditionalDump : public TsVScorer {
public:
    PhotonConditionalDump(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
        TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
    G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
    void Output() override {}
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}
private:
    bool done_{};
    void Require(bool,const char*);
};
