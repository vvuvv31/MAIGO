#pragma once
#include "TsVScorer.hh"
// Isolated native boundary-physics probe; never a dose/reference scorer.
class ElectronBoundaryProbe : public TsVScorer {
public:
    ElectronBoundaryProbe(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
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
