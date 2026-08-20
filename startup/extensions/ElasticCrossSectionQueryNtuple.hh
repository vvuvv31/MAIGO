#ifndef ElasticCrossSectionQueryNtuple_hh
#define ElasticCrossSectionQueryNtuple_hh

#include "TsVNtupleScorer.hh"

class ElasticCrossSectionQueryNtuple : public TsVNtupleScorer {
public:
    ElasticCrossSectionQueryNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~ElasticCrossSectionQueryNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4float energy_mev_per_u_ = 0.0F;
    G4double macroscopic_elastic_per_mm_ = 0.0;
};

#endif
