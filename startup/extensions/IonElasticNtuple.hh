#ifndef IonElasticNtuple_hh
#define IonElasticNtuple_hh

#include "TsVNtupleScorer.hh"

class IonElasticNtuple : public TsVNtupleScorer {
public:
    IonElasticNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~IonElasticNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4int atomic_number_ = 0;
    G4int mass_number_ = 0;
    G4float pre_energy_mev_per_u_ = 0.0F;
    G4float post_energy_mev_per_u_ = 0.0F;
    G4float recoil_energy_mev_ = 0.0F;
    G4float scattering_cosine_ = 1.0F;
    G4double macroscopic_elastic_per_mm_ = 0.0;
};

#endif
