#pragma once
#include "PrimaryElectronTransportNtuple.hh"

// Record one complete rare history using the unchanged authoritative recorder.
// All selected-history steps and branches reach the base scorer; no RNG calls.
class SelectedElectronTransportNtuple : public PrimaryElectronTransportNtuple {
public:
    SelectedElectronTransportNtuple(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
        TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
    G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
private:
    G4int selected_;
};
