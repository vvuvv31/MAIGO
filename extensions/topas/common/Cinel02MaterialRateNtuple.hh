#ifndef Cinel02MaterialRateNtuple_hh
#define Cinel02MaterialRateNtuple_hh

#include "TsVNtupleScorer.hh"

#include <vector>

class Cinel02MaterialRateNtuple : public TsVNtupleScorer {
public:
    Cinel02MaterialRateNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~Cinel02MaterialRateNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4bool filled_{false};
    G4int material_section_{-1};
    G4double energy_min_mev_per_u_{0.0};
    G4double energy_width_mev_per_u_{1.0};
    G4int energy_count_{401};

    G4String material_name_;
    G4int projectile_z_{0};
    G4int projectile_a_{0};
    G4int target_z_{0};
    G4int target_a_{0};
    G4double material_density_g_per_cm3_{0.0};
    G4double target_mass_fraction_{0.0};
    G4double target_number_density_per_mm3_{0.0};
    G4float energy_mev_per_u_{0.0F};
    G4double microscopic_cross_section_barn_{0.0};
    G4double macroscopic_cross_section_per_mm_{0.0};
};

#endif
