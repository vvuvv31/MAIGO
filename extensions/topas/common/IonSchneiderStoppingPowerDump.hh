// Header for IonSchneiderStoppingPowerDump
// Extracts Geant4/TOPAS UNRESTRICTED electronic stopping powers for the 18
// GPU charged species across all 25 Schneider sections on the 4302-node
// transport energy grid (0.01 to 430.11 MeV/u). Feeds MAIGO SCHNIOSP v1
// (tools/compile_schneider_ion_stopping.py). Unrestricted totals only:
// the electron packet takes its share from deposited energy.

#ifndef IonSchneiderStoppingPowerDump_hh
#define IonSchneiderStoppingPowerDump_hh

#include "TsVScorer.hh"

class IonSchneiderStoppingPowerDump : public TsVScorer {
public:
    IonSchneiderStoppingPowerDump(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);

    virtual ~IonSchneiderStoppingPowerDump();

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void Output() override;
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}

private:
    G4double min_energy_mevu_{0.01};
    G4double max_energy_mevu_{430.11};
    G4double energy_step_mevu_{0.1};
    G4String output_json_file_{""};
    G4String output_csv_file_{""};
    G4bool filled_{false};
};

#endif
