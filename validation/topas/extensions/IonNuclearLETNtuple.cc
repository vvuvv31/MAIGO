// Scorer for IonNuclearLETNtuple

#include "IonNuclearLETNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

IonNuclearLETNtuple::IonNuclearLETNtuple(
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
    fNtuple->RegisterColumnI(&event_id_, "Event ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent ID");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&mass_number_, "Mass Number A");
    fNtuple->RegisterColumnF(&depth_mm_, "Depth (mm)", "");
    fNtuple->RegisterColumnF(&step_length_mm_, "Step Length (mm)", "");
    fNtuple->RegisterColumnF(&pre_energy_mev_, "Pre-step Energy (MeV)", "");
    fNtuple->RegisterColumnF(&post_energy_mev_, "Post-step Energy (MeV)", "");
    fNtuple->RegisterColumnF(&mean_energy_mev_, "Mean Energy (MeV)", "");
    fNtuple->RegisterColumnF(
        &local_deposit_mev_, "Local Energy Deposit (MeV)", "");
    fNtuple->RegisterColumnF(
        &delta_electron_mev_, "Step-born Electron Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &electronic_dedx_mev_per_mm_, "Electronic dEdx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &density_g_per_cm3_, "Density (g/cm3)", "");
    fNtuple->RegisterColumnD(
        &letd_numerator_,
        "HadronLET Numerator (MeV*MeV/mm/(g/cm3))", "");
    fNtuple->RegisterColumnD(
        &letd_denominator_mev_, "HadronLET Denominator (MeV)", "");
    fNtuple->RegisterColumnS(&process_name_, "Process Name");
}

IonNuclearLETNtuple::~IonNuclearLETNtuple() = default;

G4bool IonNuclearLETNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    const auto* process =
        step->GetPostStepPoint()->GetProcessDefinedStep();
    if (process == nullptr || process->GetProcessType() != fHadronic ||
        process->GetProcessName().find("Inelastic") == G4String::npos) {
        return false;
    }

    const auto* track = step->GetTrack();
    const auto* definition = track->GetDefinition();
    atomic_number_ = definition->GetAtomicNumber();
    mass_number_ = definition->GetAtomicMass();
    if (atomic_number_ <= 0 || mass_number_ <= 0 ||
        step->GetStepLength() <= 0.0) {
        return false;
    }

    const auto* pre = step->GetPreStepPoint();
    const auto* post = step->GetPostStepPoint();
    const auto* material = pre->GetMaterial();
    const auto pre_energy = pre->GetKineticEnergy();
    const auto post_energy = post->GetKineticEnergy();
    const auto mean_energy = 0.5 * (pre_energy + post_energy);
    const auto local_deposit = step->GetTotalEnergyDeposit();

    auto delta_electron_energy = 0.0;
    const auto* secondaries = step->GetSecondaryInCurrentStep();
    if (secondaries != nullptr) {
        for (const auto* secondary : *secondaries) {
            if (secondary->GetDefinition()->GetPDGEncoding() == 11) {
                delta_electron_energy += secondary->GetKineticEnergy();
            }
        }
    }

    G4EmCalculator calculator;
    const auto electronic_dedx =
        calculator.ComputeElectronicDEDX(mean_energy, definition, material);
    const auto density = material->GetDensity();

    event_id_ = GetEventID();
    track_id_ = track->GetTrackID();
    parent_id_ = track->GetParentID();
    depth_mm_ = static_cast<G4float>(post->GetPosition().z() / mm);
    step_length_mm_ = static_cast<G4float>(step->GetStepLength() / mm);
    pre_energy_mev_ = static_cast<G4float>(pre_energy / MeV);
    post_energy_mev_ = static_cast<G4float>(post_energy / MeV);
    mean_energy_mev_ = static_cast<G4float>(mean_energy / MeV);
    local_deposit_mev_ = static_cast<G4float>(local_deposit / MeV);
    delta_electron_mev_ =
        static_cast<G4float>(delta_electron_energy / MeV);
    electronic_dedx_mev_per_mm_ = electronic_dedx / (MeV / mm);
    density_g_per_cm3_ = density / (g / cm3);
    letd_numerator_ =
        (local_deposit / MeV) * electronic_dedx_mev_per_mm_ /
        density_g_per_cm3_;
    letd_denominator_mev_ =
        (local_deposit + delta_electron_energy) / MeV;
    process_name_ = process->GetProcessName();
    fNtuple->Fill();
    return true;
}
