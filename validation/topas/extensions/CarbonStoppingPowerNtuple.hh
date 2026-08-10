#ifndef CarbonStoppingPowerNtuple_hh
#define CarbonStoppingPowerNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonStoppingPowerNtuple : public TsVNtupleScorer {
public:
    CarbonStoppingPowerNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonStoppingPowerNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4float energy_mev_per_u_ = 0.0F;
    G4float total_energy_mev_ = 0.0F;
    G4double electronic_dedx_mev_per_mm_ = 0.0;
    G4double total_dedx_mev_per_mm_ = 0.0;
    G4double csda_range_mm_ = 0.0;
};

#endif
