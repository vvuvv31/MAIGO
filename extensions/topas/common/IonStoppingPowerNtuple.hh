#ifndef IonStoppingPowerNtuple_hh
#define IonStoppingPowerNtuple_hh

#include "TsVNtupleScorer.hh"

class IonStoppingPowerNtuple : public TsVNtupleScorer {
public:
    IonStoppingPowerNtuple(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    ~IonStoppingPowerNtuple() override;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_ = false;
    G4int atomic_number_ = 0;
    G4int mass_number_ = 0;
    G4float energy_mev_per_u_ = 0.0F;
    G4double unrestricted_dedx_mev_per_mm_ = 0.0;
    G4double restricted_dedx_mev_per_mm_ = 0.0;
    G4double transport_table_dedx_mev_per_mm_ = 0.0;
    G4double nuclear_dedx_mev_per_mm_ = 0.0;
    G4double delta_electron_fraction_ = 0.0;
};

#endif
