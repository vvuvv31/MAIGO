#ifndef IonCrossSectionNtuple_hh
#define IonCrossSectionNtuple_hh

#include "TsVNtupleScorer.hh"

class IonCrossSectionNtuple : public TsVNtupleScorer {
public:
    IonCrossSectionNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~IonCrossSectionNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_{false};
    G4int atomic_number_{0};
    G4int mass_number_{0};
    G4float energy_mev_per_u_{0.0F};
    G4double macroscopic_inelastic_per_mm_{0.0};
    G4double mean_free_path_mm_{0.0};
};

#endif
