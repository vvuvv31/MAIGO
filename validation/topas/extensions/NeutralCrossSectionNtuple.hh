#ifndef NeutralCrossSectionNtuple_hh
#define NeutralCrossSectionNtuple_hh

#include "TsVNtupleScorer.hh"

class NeutralCrossSectionNtuple : public TsVNtupleScorer {
public:
    NeutralCrossSectionNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~NeutralCrossSectionNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_{false};
    G4int pdg_id_{0};
    G4float energy_mev_{0.0F};
    G4double macroscopic_total_per_mm_{0.0};
    G4double mean_free_path_mm_{0.0};
};

#endif
