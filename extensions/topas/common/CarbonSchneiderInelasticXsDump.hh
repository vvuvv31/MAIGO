#ifndef CarbonSchneiderInelasticXsDump_hh
#define CarbonSchneiderInelasticXsDump_hh

#include "TsVScorer.hh"

class CarbonSchneiderInelasticXsDump : public TsVScorer {
public:
    CarbonSchneiderInelasticXsDump(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);
    ~CarbonSchneiderInelasticXsDump() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void Output() override;
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}

private:
    void DumpCrossSections();

    G4bool dumped_{false};
    G4String output_json_path_{};
    G4String output_csv_path_{};
    G4double min_energy_mevu_{0.5};
    G4double max_energy_mevu_{430.0};
    G4double energy_step_mevu_{0.5};
};

#endif
