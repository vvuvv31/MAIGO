#ifndef IonNuclearLETNtuple_hh
#define IonNuclearLETNtuple_hh

#include "TsVNtupleScorer.hh"

class IonNuclearLETNtuple : public TsVNtupleScorer {
public:
    IonNuclearLETNtuple(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~IonNuclearLETNtuple() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    G4int event_id_ = 0;
    G4int track_id_ = 0;
    G4int parent_id_ = 0;
    G4int atomic_number_ = 0;
    G4int mass_number_ = 0;
    G4float depth_mm_ = 0.0F;
    G4float step_length_mm_ = 0.0F;
    G4float pre_energy_mev_ = 0.0F;
    G4float post_energy_mev_ = 0.0F;
    G4float mean_energy_mev_ = 0.0F;
    G4float local_deposit_mev_ = 0.0F;
    G4float delta_electron_mev_ = 0.0F;
    G4double electronic_dedx_mev_per_mm_ = 0.0;
    G4double density_g_per_cm3_ = 0.0;
    G4double letd_numerator_ = 0.0;
    G4double letd_denominator_mev_ = 0.0;
    G4String process_name_;
};

#endif
