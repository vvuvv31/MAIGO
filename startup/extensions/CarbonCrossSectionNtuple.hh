#ifndef CarbonCrossSectionNtuple_hh
#define CarbonCrossSectionNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonCrossSectionNtuple : public TsVNtupleScorer {
public:
    CarbonCrossSectionNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonCrossSectionNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4float energy_mev_per_u_ = 0.0F;
    G4float total_energy_mev_ = 0.0F;
    G4double sigma_h_barn_ = 0.0;
    G4double sigma_o_barn_ = 0.0;
    G4double macro_h_per_mm_ = 0.0;
    G4double macro_o_per_mm_ = 0.0;
    G4double macro_water_per_mm_ = 0.0;
    G4double mean_free_path_mm_ = 0.0;
};

#endif
