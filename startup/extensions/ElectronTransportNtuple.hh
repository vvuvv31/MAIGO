#ifndef ElectronTransportNtuple_hh
#define ElectronTransportNtuple_hh

#include "TsVNtupleScorer.hh"

class ElectronTransportNtuple : public TsVNtupleScorer {
public:
    ElectronTransportNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String, G4String,
        G4bool);

    ~ElectronTransportNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4double kinetic_energy_MeV_ = 0.0;
    G4double electron_collisional_MeV_per_mm_ = 0.0;
    G4double electron_radiative_MeV_per_mm_ = 0.0;
    G4double electron_total_MeV_per_mm_ = 0.0;
    G4double positron_collisional_MeV_per_mm_ = 0.0;
    G4double positron_radiative_MeV_per_mm_ = 0.0;
    G4double positron_total_MeV_per_mm_ = 0.0;
};

#endif
