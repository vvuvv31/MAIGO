// Scorer for IonElasticNtuple

#include "IonElasticNtuple.hh"

#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

IonElasticNtuple::IonElasticNtuple(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVNtupleScorer(
          parameter_manager, material_manager, geometry_manager,
          scoring_manager, extension_manager, scorer_name, quantity,
          output_file, is_sub_scorer) {
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&mass_number_, "Mass Number A");
    fNtuple->RegisterColumnF(
        &pre_energy_mev_per_u_, "Pre-step Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(
        &post_energy_mev_per_u_, "Post-step Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(
        &recoil_energy_mev_, "Projectile Energy Loss (MeV)", "");
    fNtuple->RegisterColumnF(
        &scattering_cosine_, "Projectile Scattering Cosine", "");
    fNtuple->RegisterColumnD(
        &macroscopic_elastic_per_mm_,
        "Macroscopic Elastic Cross Section (1/mm)", "");
}

IonElasticNtuple::~IonElasticNtuple() = default;

G4bool IonElasticNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    const auto* process =
        step->GetPostStepPoint()->GetProcessDefinedStep();
    if (process == nullptr ||
        process->GetProcessName().find("Elastic") == G4String::npos) {
        return false;
    }
    const auto* definition = step->GetTrack()->GetDefinition();
    atomic_number_ = definition->GetAtomicNumber();
    mass_number_ = definition->GetAtomicMass();
    if (atomic_number_ <= 0 || mass_number_ <= 0) {
        return false;
    }

    const auto pre_energy = step->GetPreStepPoint()->GetKineticEnergy();
    const auto post_energy = step->GetPostStepPoint()->GetKineticEnergy();
    pre_energy_mev_per_u_ = static_cast<G4float>(
        pre_energy / (static_cast<G4double>(mass_number_) * MeV));
    post_energy_mev_per_u_ = static_cast<G4float>(
        post_energy / (static_cast<G4double>(mass_number_) * MeV));
    recoil_energy_mev_ =
        static_cast<G4float>((pre_energy - post_energy) / MeV);
    scattering_cosine_ = static_cast<G4float>(
        step->GetPreStepPoint()->GetMomentumDirection().dot(
            step->GetPostStepPoint()->GetMomentumDirection()));
    const auto* material = step->GetPreStepPoint()->GetMaterial();
    const auto macroscopic =
        G4HadronicProcessStore::Instance()
            ->GetElasticCrossSectionPerVolume(
                definition, pre_energy, material);
    macroscopic_elastic_per_mm_ = macroscopic / (1.0 / mm);
    fNtuple->Fill();
    return true;
}
