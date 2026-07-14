#ifndef CarbonDoseOrigin_hh
#define CarbonDoseOrigin_hh

#include "TsVBinnedScorer.hh"

class CarbonDoseOrigin : public TsVBinnedScorer {
public:
    CarbonDoseOrigin(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonDoseOrigin() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4int selected_category_ = -1;
};

#endif
