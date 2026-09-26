// Header for CarbonSchneiderStoppingPowerDump
// Extracts Geant4/TOPAS C12 mass and linear stopping powers across all 25 Schneider sections
// on the 4001-node transport energy grid (0.01 to 400.01 MeV/u).

#ifndef CarbonSchneiderStoppingPowerDump_hh
#define CarbonSchneiderStoppingPowerDump_hh

#include "TsVScorer.hh"

class CarbonSchneiderStoppingPowerDump : public TsVScorer {
public:
    CarbonSchneiderStoppingPowerDump(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    virtual ~CarbonSchneiderStoppingPowerDump();

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void Output() override;
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}

private:
    G4int projectile_z_{6};
    G4int projectile_a_{12};
    G4double min_energy_mevu_{0.01};
    G4double max_energy_mevu_{430.11};
    G4double energy_step_mevu_{0.1};
    G4String output_json_file_{""};
    G4String output_csv_file_{""};
    G4bool filled_{false};
};

#endif
