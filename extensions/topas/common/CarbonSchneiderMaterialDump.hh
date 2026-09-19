#ifndef CarbonSchneiderMaterialDump_hh
#define CarbonSchneiderMaterialDump_hh

#include "TsVScorer.hh"

class CarbonSchneiderMaterialDump : public TsVScorer {
public:
    CarbonSchneiderMaterialDump(
        TsParameterManager* parameter_manager,
        TsMaterialManager* material_manager,
        TsGeometryManager* geometry_manager,
        TsScoringManager* scoring_manager,
        TsExtensionManager* extension_manager,
        G4String scorer_name,
        G4String quantity,
        G4String output_file,
        G4bool is_sub_scorer);
    ~CarbonSchneiderMaterialDump() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void Output() override;
    void Clear() override {}
    void RestoreResultsFromFile() override {}
    void AccumulateEvent() override {}
    void AbsorbResultsFromWorkerScorer(TsVScorer*) override {}

private:
    void DumpSchneiderMaterials();

    G4bool dumped_{false};
    G4String output_path_{};
};

#endif
