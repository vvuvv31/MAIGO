#pragma once
#include "TsVScorer.hh"

class MaterialDefinitionDump : public TsVScorer {
public:
    MaterialDefinitionDump(TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String, G4String, G4bool);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void Output() override {}
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}
private:
    G4String output_;
    bool done_{false};
};
