// Scorer for CarbonDeltaElectronEnergy

#include "CarbonDeltaElectronEnergy.hh"

#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4Track.hh"

CarbonDeltaElectronEnergy::CarbonDeltaElectronEnergy(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVBinnedScorer(
          parameter_manager,
          material_manager,
          geometry_manager,
          scoring_manager,
          extension_manager,
          scorer_name,
          quantity,
          output_file,
          is_sub_scorer) {
    SetUnit("MeV");
}

CarbonDeltaElectronEnergy::~CarbonDeltaElectronEnergy() = default;

G4bool CarbonDeltaElectronEnergy::ProcessHits(
    G4Step* step,
    G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    const auto* particle = step->GetTrack()->GetDefinition();
    if (particle->GetAtomicNumber() < 1 || step->GetStepLength() <= 0.) {
        return false;
    }

    // Match myHadronLET_Denominator exactly. Copying the track-owned step is
    // important here: querying the parallel-world scorer's step directly can
    // expose the cumulative secondary vector instead of the current-step slice.
    G4Step track_step = *step->GetTrack()->GetStep();
    G4double electron_energy = 0.;
    const auto* secondaries = track_step.GetSecondaryInCurrentStep();
    for (const auto* secondary : *secondaries) {
        if (secondary->GetDefinition()->GetPDGEncoding() == 11) {
            electron_energy += secondary->GetKineticEnergy();
        }
    }
    if (electron_energy <= 0.) {
        return false;
    }
    AccumulateHit(step, electron_energy);
    return true;
}
