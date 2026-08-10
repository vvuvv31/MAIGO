#ifndef CarbonMaterialPropertiesNtuple_hh
#define CarbonMaterialPropertiesNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonMaterialPropertiesNtuple : public TsVNtupleScorer {
public:
    CarbonMaterialPropertiesNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String scorer_name,
        G4String quantity, G4String output_file, G4bool is_sub_scorer);
    ~CarbonMaterialPropertiesNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4String material_name_;
    G4double density_g_per_cm3_ = 0.0;
    G4double radiation_length_mm_ = 0.0;
    G4double mass_radiation_length_g_per_cm2_ = 0.0;
};

#endif
