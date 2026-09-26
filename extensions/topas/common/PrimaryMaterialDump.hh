#pragma once
#include "TsVScorer.hh"
// Source ZA and target material are taken from a real primary step.
class PrimaryMaterialDump : public TsVScorer {
public:
 PrimaryMaterialDump(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
  TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
 G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
 void Output() override {}
 void Clear() override {}
 void RestoreResultsFromFile() override {}
 void AccumulateEvent() override {}
 void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}
private:
 G4String stem_;
 bool done_=false;
};
