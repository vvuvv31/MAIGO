#ifndef CarbonLossRangeNtuple_hh
#define CarbonLossRangeNtuple_hh

#include "TsVNtupleScorer.hh"

class CarbonLossRangeNtuple : public TsVNtupleScorer {
public:
    CarbonLossRangeNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~CarbonLossRangeNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4float energy_mev_per_u_ = 0.0F;
    G4float total_energy_mev_ = 0.0F;
    // Restricted-loss range for the ACTUAL material-cuts couple of the
    // scoring volume (theRangeTableForLoss semantics: G4VEnergyLossProcess
    // ionIoni tables with the production cuts). This is the Urban currentRange.
    G4double loss_range_mm_ = 0.0;
    // CSDA range, documentation/comparison only. NOT used by Urban.
    G4double csda_range_mm_ = 0.0;
    // Restricted dE/dx for the same couple (Urban GetDEDX). Compared against
    // the compiled unrestricted stopping CSV at validation time.
    G4double restricted_dedx_mev_per_mm_ = 0.0;
    // Inverse-table round trip: GetKineticEnergy(loss_range) - E.
    G4double inverse_residual_mev_ = 0.0;
};

#endif
