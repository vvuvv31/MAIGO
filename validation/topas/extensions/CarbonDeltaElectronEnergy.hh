#ifndef CarbonDeltaElectronEnergy_hh
#define CarbonDeltaElectronEnergy_hh

#include "TsVBinnedScorer.hh"

class CarbonDeltaElectronEnergy : public TsVBinnedScorer {
public:
    CarbonDeltaElectronEnergy(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer = false);

    ~CarbonDeltaElectronEnergy() override;
    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override;
};

#endif
